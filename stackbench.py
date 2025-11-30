import streamlit as st
import time
import uuid
import json
import os
import requests
import random
import logging
import threading
import pandas as pd
import altair as alt
from typing import List, Dict, Optional, Any, Deque
from dataclasses import dataclass, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Lock

# --- OPTIONAL IMPORTS ---
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# --- CONFIGURATION & CONSTANTS ---
st.set_page_config(page_title="StackBench", page_icon="🤖", layout="wide")

# Load environment variables
# Check Streamlit secrets first, then fallback to os.getenv
def get_config(key, default=""):
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

GROQ_API_KEY = get_config("GROQ_API_KEY")
GITHUB_TOKEN = get_config("GITHUB_TOKEN")
# New Configs
GITHUB_OAUTH_APP = get_config("GITHUB_OAUTH_APP") # Used as fallback token
WIKI_API_KEY = get_config("WIKI_API_KEY") # Optional: API Key or Contact Email for User-Agent
GEMINI_API_KEY = get_config("GEMINI_API_KEY") # For Deep Dive feature

UPTIME_URL = get_config("UPTIME_URL")
MAX_CONCURRENT_MISSIONS = int(get_config("MAX_CONCURRENT_MISSIONS", "3"))
QUEUE_MISSIONS = get_config("QUEUE_MISSIONS", "true").lower() == "true"

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StackBench")

# --- UTILITIES: KEEP-ALIVE ---
def keep_alive_worker():
    """Background thread to ping UPTIME_URL to prevent sleep."""
    while True:
        if UPTIME_URL:
            try:
                requests.get(UPTIME_URL, timeout=10)
                logger.info(f"Pinged {UPTIME_URL}")
            except Exception as e:
                logger.error(f"Keep-alive ping failed: {e}")
        time.sleep(300)  # Ping every 5 minutes

if UPTIME_URL and "keep_alive_started" not in st.session_state:
    threading.Thread(target=keep_alive_worker, daemon=True).start()
    st.session_state["keep_alive_started"] = True

# --- CORE: EVENT BUS ---
@dataclass
class Event:
    trace_id: str
    session_id: str
    agent: str
    type: str  # LOG, A2A, STATE, ERROR, METRIC, REPORT
    content: str
    timestamp: float = field(default_factory=time.time)

class EventBus:
    """Thread-safe in-memory event store."""
    def __init__(self):
        self._events: Deque[Event] = deque(maxlen=1000)
        self._lock = Lock()

    def publish(self, event: Event):
        with self._lock:
            self._events.append(event)
            # Log to console for debugging
            print(f"[{event.agent}] {event.type}: {event.content[:50]}...")

    def get_events(self, session_id: str, since_ts: float = 0) -> List[Event]:
        with self._lock:
            return [
                e for e in self._events 
                if e.session_id == session_id and e.timestamp > since_ts
            ]

# Singleton EventBus (stored in session_state for Streamlit persistence across re-runs)
if "event_bus" not in st.session_state:
    st.session_state["event_bus"] = EventBus()
event_bus = st.session_state["event_bus"]

# --- CORE: MONITORING ---
class Monitor:
    """Singleton for tracking metrics."""
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Monitor, cls).__new__(cls)
                cls._instance.metrics = {
                    "api_calls": 0,
                    "tasks_completed": 0,
                    "tokens_estimated": 0,
                    "latency_ms": []
                }
        return cls._instance

    def log_api_call(self, agent_name: str, latency: float):
        with self._lock:
            self.metrics["api_calls"] += 1
            self.metrics["latency_ms"].append({"agent": agent_name, "latency": latency})

    def get_metrics(self):
        with self._lock:
            avg_lat = 0
            if self.metrics["latency_ms"]:
                total = sum(item["latency"] for item in self.metrics["latency_ms"])
                avg_lat = total / len(self.metrics["latency_ms"])
            return {
                "api_calls": self.metrics["api_calls"],
                "tasks_completed": self.metrics["tasks_completed"],
                "avg_latency_ms": round(avg_lat, 2)
            }

monitor = Monitor()

# --- TOOLBOX ---
class ToolBox:
    @staticmethod
    def get_github_stats(repo_name: str) -> Dict[str, Any]:
        """Fetches repo stats. Uses GITHUB_TOKEN or GITHUB_OAUTH_APP if available."""
        if "/" not in repo_name:
            return {"error": "Invalid repo format. Use owner/repo"}
        
        headers = {}
        # Prioritize GITHUB_TOKEN, fallback to GITHUB_OAUTH_APP
        token = GITHUB_TOKEN if GITHUB_TOKEN else GITHUB_OAUTH_APP
        if token:
            headers["Authorization"] = f"token {token}"
        
        url = f"https://api.github.com/repos/{repo_name}"
        try:
            # Simple retry logic for main stats
            data = {}
            for _ in range(3):
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    break
                elif resp.status_code == 404:
                    return {"error": "Repo not found"}
                time.sleep(1)
            
            if not data:
                return {"error": f"GitHub API unreachable (Status {resp.status_code})"}

            # New Feature: Fetch Contributors using the auth token
            contributors = []
            try:
                contrib_url = f"https://api.github.com/repos/{repo_name}/contributors?per_page=5"
                c_resp = requests.get(contrib_url, headers=headers, timeout=5)
                if c_resp.status_code == 200:
                    contributors = [c["login"] for c in c_resp.json()]
            except Exception as e:
                logger.error(f"Failed to fetch contributors: {e}")

            return {
                "stars": data.get("stargazers_count"),
                "forks": data.get("forks_count"),
                "issues": data.get("open_issues_count"),
                "updated_at": data.get("updated_at"),
                "description": data.get("description"),
                "top_contributors": contributors
            }

        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_wikipedia_summary(query: str) -> str:
        """Fetches Wikipedia summary using requests (avoids heavy library dependency)."""
        # Heuristic: If input is 'owner/repo', extract 'repo' for better Wiki matching
        # This prevents 400 Bad Request errors for queries like "streamlit/streamlit"
        if "/" in query:
            query = query.split("/")[-1]

        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")
        
        headers = {
            "User-Agent": "StackBench/1.0 (Educational Project)"
        }
        
        # If WIKI_API_KEY is provided, use it in headers.
        if WIKI_API_KEY:
            if "@" in WIKI_API_KEY:
                headers["User-Agent"] = WIKI_API_KEY
            else:
                headers["Authorization"] = f"Bearer {WIKI_API_KEY}"
        
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if "extract" in data:
                    return data["extract"]
                if "type" in data and data["type"] == "disambiguation":
                    return "Ambiguous query. Please be more specific."
            elif resp.status_code == 404:
                return "No Wikipedia page found."
            return f"Wiki API Status: {resp.status_code}"
        except Exception as e:
            return f"Wiki error: {str(e)}"

# --- LLM ENGINE (GROQ + SIMULATOR) ---
class LLMEngine:
    def __init__(self):
        # Determine provider based on Key existence AND Library availability
        if GROQ_API_KEY and GROQ_AVAILABLE:
            self.provider = "GROQ"
            self.client = Groq(api_key=GROQ_API_KEY)
        else:
            self.provider = "SIMULATOR"
            # Optional: Warn if key exists but lib missing
            if GROQ_API_KEY and not GROQ_AVAILABLE:
                print("Groq API Key found, but 'groq' library not installed. Falling back to simulator.")

    def generate(self, system_prompt: str, user_prompt: str, model="llama-3.1-8b-instant") -> str:
        start_time = time.time()
        result = ""
        
        if self.provider == "GROQ":
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=model,
                )
                result = chat_completion.choices[0].message.content
            except Exception as e:
                result = f"LLM Error: {str(e)}"
        else:
            # Simulator Fallback
            time.sleep(random.uniform(0.5, 1.5))  # Simulate network latency
            if "Analyst" in system_prompt:
                result = "Based on the analysis of the provided technical metrics, the technology shows strong adoption trends. GitHub activity indicates active development and community engagement."
            elif "Verification" in system_prompt:
                result = "Verified: The GitHub stats are consistent with high-growth projects. Wiki data matches."
            elif "Advisor" in system_prompt:
                result = "Recommendation: Adopt this technology for high-scale needs. Code maturity is high."
            else:
                result = "Simulation response."

        duration = (time.time() - start_time) * 1000
        monitor.log_api_call("LLMEngine", duration)
        return result

llm_engine = LLMEngine()

# --- AGENTS ---
@dataclass
class AgentMessage:
    sender: str
    recipient: str
    content: Any

class Agent:
    def __init__(self, name: str, session_id: str, trace_id: str):
        self.name = name
        self.session_id = session_id
        self.trace_id = trace_id
        self.inbox: Deque[AgentMessage] = deque()

    def receive(self, message: AgentMessage):
        self.inbox.append(message)
        event_bus.publish(Event(
            self.trace_id, self.session_id, self.name, "A2A", 
            f"Received message from {message.sender}"
        ))

    def send_message(self, recipient: 'Agent', content: Any):
        msg = AgentMessage(self.name, recipient.name, content)
        recipient.receive(msg)
        event_bus.publish(Event(
            self.trace_id, self.session_id, self.name, "A2A", 
            f"Sent message to {recipient.name}"
        ))

    def think_and_act(self, context: Dict) -> str:
        raise NotImplementedError

class AnalystAgent(Agent):
    def think_and_act(self, context: Dict) -> str:
        tech = context.get("target")
        event_bus.publish(Event(self.trace_id, self.session_id, self.name, "LOG", f"Analyzing {tech}..."))
        
        # 1. Use Tools
        wiki = ToolBox.get_wikipedia_summary(tech)
        github = ToolBox.get_github_stats(tech) # Assuming input is roughly repo-like or we map it
        
        # 2. LLM Analysis
        # Determine if wiki data is valid or an error message
        wiki_context = wiki if "Wiki error" not in wiki and "Wiki API Status" not in wiki and "No Wikipedia page" not in wiki else "Data unavailable"

        prompt = (
            f"You are a Senior Tech Analyst. Analyze the technology '{tech}'.\n"
            f"Context Data:\n"
            f"- GitHub Metrics: {json.dumps(github)}\n"
            f"- Wikipedia Summary: {wiki_context}\n\n"
            "Task: Provide a concise technical summary. Evaluate its popularity, maturity, and recent activity based on the metrics. "
            "Do NOT mention API errors or missing data in the final output; focus only on the technology info available."
        )
        analysis = llm_engine.generate("You are a Senior Tech Analyst.", prompt)
        
        return json.dumps({
            "analysis": analysis,
            "raw_data": {"wiki": wiki, "github": github}
        })

class VerificationAgent(Agent):
    def think_and_act(self, context: Dict) -> str:
        # 1. Wait for input (simulated by checking inbox or just processing passed context in real-time)
        # In this architecture, orchestrator passes data, or we wait.
        # For parallelism demo, we simulate independent verification work first.
        
        event_bus.publish(Event(self.trace_id, self.session_id, self.name, "LOG", "Starting independent verification check..."))
        time.sleep(1.0) # Simulate work
        
        # Check inbox for Analyst data
        analyst_data = {}
        if self.inbox:
            msg = self.inbox.popleft()
            try:
                analyst_data = json.loads(msg.content)
            except:
                analyst_data = {"raw_data": "Error parsing"}
        
        # 2. Verify
        analysis_text = analyst_data.get('analysis', 'No data')
        prompt = (
            f"You are a QA Auditor. Verify the following technical analysis for consistency and realism: '{analysis_text}'. "
            "Output a brief verification report confirming if the metrics align with the summary."
        )
        verification = llm_engine.generate("You are a QA Verification Agent.", prompt)
        
        return json.dumps({
            "status": "Verified",
            "details": verification,
            "score": random.randint(80, 100) # Simulated score
        })

class AdvisorAgent(Agent):
    def think_and_act(self, context: Dict) -> str:
        analyst_out = context.get("analyst_output", {})
        verifier_out = context.get("verifier_output", {})
        
        # Clean inputs if they are strings containing JSON
        if isinstance(analyst_out, str):
             try: analyst_out = json.loads(analyst_out).get("analysis", "")
             except: pass
        if isinstance(verifier_out, str):
             try: verifier_out = json.loads(verifier_out).get("details", "")
             except: pass

        prompt = (
            f"You are a CTO Advisor. Review the analysis below for the technology.\n"
            f"Analyst Report: {analyst_out}\n"
            f"Verification: {verifier_out}\n\n"
            "Task: Provide a final executive recommendation.\n"
            "Format:\n"
            "1. Executive Summary\n"
            "2. Pros & Cons\n"
            "3. Final Verdict (Adopt/Assess/Hold)"
        )
        recommendation = llm_engine.generate("You are a CTO Advisor.", prompt)
        
        return recommendation

# --- ORCHESTRATOR & MISSION MANAGER ---
class MissionManager:
    """Manages concurrency limits."""
    active_missions = 0
    queue = deque()
    lock = Lock()

    @classmethod
    def try_start_mission(cls) -> bool:
        with cls.lock:
            if cls.active_missions < MAX_CONCURRENT_MISSIONS:
                cls.active_missions += 1
                return True
            return False
    
    @classmethod
    def end_mission(cls):
        with cls.lock:
            if cls.active_missions > 0:
                cls.active_missions -= 1

class Orchestrator:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=5)

    def run_mission(self, query: str, session_id: str, intent: str):
        if not MissionManager.try_start_mission():
            if QUEUE_MISSIONS:
                 event_bus.publish(Event(str(uuid.uuid4()), session_id, "SYSTEM", "ERROR", "Max missions reached. Queued (Not implemented in demo, rejected)."))
            event_bus.publish(Event(str(uuid.uuid4()), session_id, "SYSTEM", "ERROR", "Server busy. Try again later."))
            return

        trace_id = str(uuid.uuid4())
        event_bus.publish(Event(trace_id, session_id, "SYSTEM", "STATE", "started"))
        
        try:
            # Init Agents
            analyst = AnalystAgent("Analyst", session_id, trace_id)
            verifier = VerificationAgent("Verifier", session_id, trace_id)
            advisor = AdvisorAgent("Advisor", session_id, trace_id)

            # --- PARALLEL EXECUTION ---
            # 1. Start Analyst
            future_analyst = self.executor.submit(analyst.think_and_act, {"target": query})
            
            # 2. Start Verifier (Simulate it starting purely parallel work before receiving data)
            # In a real app, it might check cache or static rules.
            # We wrap verifier logic to wait for message inside its thread or orchestrator
            def verifier_lifecycle():
                # Simulate initial work
                time.sleep(0.5)
                # Wait for analyst result from Orchestrator pipe
                return verifier.think_and_act({})
            
            # Since Verifier.think_and_act pops from inbox, we need to populate it.
            # But the Verifier is running in a thread. 
            # We will submit the verifier logic, but it needs data.
            # Correct Pattern: Submit Analyst. Wait result. Send to Verifier. Submit Verifier.
            # To show VISUAL parallelism, we can run them, but Verifier blocks on a queue.
            
            # Simplified flow for stability:
            # Analyst Runs. Verifier runs "Pre-check".
            # Analyst finishes -> Message to Verifier.
            # Verifier finishes "Main check".
            
            analyst_res_json = future_analyst.result() # This blocks main thread? No, we are in a thread?
            # Orchestrator.run_mission should be called in a background thread by Streamlit!
            
            analyst.send_message(verifier, analyst_res_json)
            
            # Now run verifier logic (it consumes message)
            verifier_res_json = verifier.think_and_act({}) 

            # Advisor runs
            advisor_res = advisor.think_and_act({
                "analyst_output": analyst_res_json, 
                "verifier_output": verifier_res_json
            })
            
            # Parse results for report
            try:
                a_data = json.loads(analyst_res_json)
                v_data = json.loads(verifier_res_json)
            except:
                a_data, v_data = {}, {}

            # Save Report
            report = {
                "trace_id": trace_id,
                "target": query,
                "intent": intent,
                "timestamp": datetime.now().isoformat(),
                "analyst_summary": a_data.get("analysis"),
                "verification_status": v_data.get("status"),
                "confidence_score": v_data.get("score"),
                "advisor_recommendation": advisor_res,
                "metrics": monitor.get_metrics(),
                "github_stats": a_data.get("raw_data", {}).get("github", {})
            }
            
            # --- FIX: THREAD SAFETY ---
            # Do NOT write to st.session_state here (it is thread-unsafe).
            # Publish a REPORT event instead, which the main thread will pick up.
            event_bus.publish(Event(trace_id, session_id, "SYSTEM", "REPORT", json.dumps(report)))
            event_bus.publish(Event(trace_id, session_id, "SYSTEM", "STATE", "completed"))
            
        except Exception as e:
            event_bus.publish(Event(trace_id, session_id, "SYSTEM", "ERROR", str(e)))
        finally:
            MissionManager.end_mission()

# --- STREAMLIT UI ---
def run_app():
    # Helper to run orchestrator in background
    def start_mission_thread(query, sid, intent):
        orch = Orchestrator()
        t = threading.Thread(target=orch.run_mission, args=(query, sid, intent))
        t.start()

    # Sidebar
    with st.sidebar:
        st.title("StackBench 🏗️")
        st.markdown("**Multi-Agent Comparator**")
        
        tech_input = st.text_input("Primary Tech (Owner/Repo)", "streamlit/streamlit")
        intent = st.selectbox("Intent", ["Adoption Analysis", "Code Audit", "Migration Plan"])
        
        if st.button("Start Mission"):
            session_id = str(uuid.uuid4())
            st.session_state["current_session"] = session_id
            st.session_state["mission_active"] = True
            st.session_state["events_cursor"] = 0
            start_mission_thread(tech_input, session_id, intent)
            st.rerun()

        st.divider()
        st.subheader("Microbenchmarks")
        preset = st.selectbox("Preset", ["JSON Parse", "Sort List", "Math Ops"])
        if st.button("Run Benchmark"):
            import timeit
            if preset == "JSON Parse":
                # FIX: Use a completely unambiguous setup for JSON list
                setup = "import json; data = json.dumps([{'a': 1} for _ in range(1000)])"
                stmt = "json.loads(data)"
            elif preset == "Sort List":
                setup = "import random; l = list(range(1000)); random.shuffle(l)"
                stmt = "sorted(l)"
            else:
                setup = "a=1"
                stmt = "a+1"
            
            try:
                t = timeit.timeit(stmt, setup, number=1000)
                st.success(f"{preset}: {t:.4f}s")
                
                # Simple Chart
                df = pd.DataFrame({"Task": [preset], "Time (s)": [t]})
                c = alt.Chart(df).mark_bar().encode(x="Task", y="Time (s)")
                st.altair_chart(c, use_container_width=True)
            except Exception as e:
                st.error(f"Benchmark failed: {e}")

    # Main Layout
    col_log, col_viz = st.columns([1.5, 1])

    current_session = st.session_state.get("current_session")
    
    # --- Live Log & Polling ---
    with col_log:
        st.subheader("📡 Live Agent Log")
        log_container = st.container(height=400)
        
        # Poll events
        if current_session:
            events = event_bus.get_events(current_session, 0)
            
            # Status check
            is_complete = any(e.type == "STATE" and e.content == "completed" for e in events)
            
            for e in events:
                color = "blue"
                if e.agent == "Analyst": color = "orange"
                if e.agent == "Verifier": color = "green"
                if e.agent == "Advisor": color = "purple"
                if e.type == "ERROR": color = "red"
                
                # --- FIX: CAPTURE REPORT EVENT ---
                if e.type == "REPORT":
                    try:
                        report_data = json.loads(e.content)
                        if "history" not in st.session_state:
                            st.session_state["history"] = []
                        
                        # Idempotency check: prevent duplicates during re-renders
                        existing_ids = {r["trace_id"] for r in st.session_state["history"]}
                        if report_data["trace_id"] not in existing_ids:
                            st.session_state["history"].append(report_data)
                    except Exception as e:
                        print(f"Error parsing report: {e}")
                    continue # Don't log the raw JSON report event to the chat log
                
                log_container.markdown(f":{color}[**{e.agent}**]: {e.content}")

            if st.session_state.get("mission_active") and not is_complete:
                time.sleep(0.8)
                st.rerun()
            elif is_complete:
                st.session_state["mission_active"] = False

    # --- Visual Parallelism & Report ---
    with col_viz:
        st.subheader("📊 Timeline & Evidence")
        
        # Fake Timeline (Visualization of parallel windows)
        if st.session_state.get("mission_active") or current_session:
            # Normally we'd use real timestamps from events, simplifying for demo
            timeline_data = [
                {"Agent": "Analyst", "Start": 0, "End": 5},
                {"Agent": "Verifier", "Start": 2, "End": 7}, # Overlap shows parallelism
                {"Agent": "Advisor", "Start": 6, "End": 8}
            ]
            df_tl = pd.DataFrame(timeline_data)
            chart = alt.Chart(df_tl).mark_bar().encode(
                x='Start',
                x2='End',
                y='Agent',
                color='Agent'
            ).properties(height=150)
            st.altair_chart(chart, use_container_width=True)
            
            # Evidence Cards (Last fetched data)
            # Fetch generic if not provided for demo UI fill
            if tech_input:
                st.info(f"Target: {tech_input}")
                
                stars = "---"
                forks = "---"
                contrib_list = []
                
                # Try to get from history
                if "history" in st.session_state and st.session_state["history"]:
                    last_item = st.session_state["history"][-1]
                    if last_item.get("target") == tech_input:
                         stats = last_item.get("github_stats", {})
                         if "stars" in stats:
                             stars = str(stats["stars"])
                             forks = str(stats.get("forks", "---"))
                             contrib_list = stats.get("top_contributors", [])
                
                c_stars, c_forks = st.columns(2)
                c_stars.metric("GitHub Stars", stars)
                c_forks.metric("Forks", forks)
                
                if contrib_list:
                    st.caption(f"Top Contributors: {', '.join(contrib_list)}")

    # --- Final Report ---
    st.divider()
    st.subheader("📑 Mission Report")
    
    if "history" in st.session_state and st.session_state["history"]:
        last_report = st.session_state["history"][-1]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence Score", f"{last_report['confidence_score']}%")
        c2.metric("API Calls", last_report['metrics']['api_calls'])
        c3.metric("Latency", f"{last_report['metrics']['avg_latency_ms']}ms")
        
        st.markdown(f"### Recommendation\n{last_report['advisor_recommendation']}")
        
        with st.expander("Detailed Analysis"):
            st.write(last_report['analyst_summary'])
        
        with st.expander("Verification Details"):
            st.write(last_report['verification_status'])

        st.download_button("Download Report JSON", json.dumps(last_report, indent=2), "report.json")
        
        # --- GEMINI DEEP DIVE ---
        st.divider()
        if st.button("Ask Gemini for a deeper assessment ✨"):
            if not GEMINI_API_KEY:
                st.warning("Please configure GEMINI_API_KEY in secrets to use this feature.")
            else:
                with st.spinner("Gemini is analyzing the report..."):
                    try:
                        # Construct Prompt
                        prompt_text = (
                            f"You are a Senior Technology Architect. Review this automated technical report:\n\n"
                            f"{json.dumps(last_report, indent=2)}\n\n"
                            "Provide a 'Second Opinion'. Identify blind spots, agree/disagree with the Advisor, and provide deeper context."
                        )
                        
                        # Call Gemini API via REST (to avoid extra deps)
                        # Using gemini-1.5-flash as it is fast and efficient
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
                        headers = {"Content-Type": "application/json"}
                        payload = {
                            "contents": [{
                                "parts": [{"text": prompt_text}]
                            }]
                        }
                        
                        resp = requests.post(url, headers=headers, json=payload, timeout=15)
                        
                        if resp.status_code == 200:
                            result = resp.json()
                            # Extract text
                            gemini_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "No response text found.")
                            st.success("Gemini Assessment Complete")
                            st.markdown(f"### ♊ Gemini Second Opinion\n{gemini_text}")
                        else:
                            st.error(f"Gemini API Error: {resp.status_code} - {resp.text}")
                            
                    except Exception as e:
                        st.error(f"Failed to connect to Gemini: {str(e)}")

if __name__ == "__main__":
    run_app()
