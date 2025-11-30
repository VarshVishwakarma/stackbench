STACKBENCH — Multi-Agent Framework Comparator
AI-Powered · Evidence-Driven · Parallel · Developer-Focused

StackBench is a next-generation multi-agent research copilot that analyzes developer frameworks using:

⚙️ Real GitHub Metrics

📘 Clean Wikipedia Summaries with Smart Fallbacks

🧠 LLM-Reasoned Analysis via Groq + Gemini

🤝 Agent-to-Agent Collaboration

📡 Real-Time Logs, Metrics & Parallel Timeline Visualization

It delivers fast, verifiable, and architect-grade recommendations — all inside a modern Streamlit UI.

🚀 Why StackBench? — The Problem

Picking the right technology is a minefield:

❌ Documentation is scattered
❌ GitHub activity is hard to measure manually
❌ Wikipedia info is unreliable or outdated
❌ LLMs hallucinate without verified data
❌ Engineers rarely agree on the same evaluation criteria

💡 StackBench: The Solution

StackBench uses multiple autonomous AI agents—all running in parallel—to perform:

🔍 1. Evidence Collection

GitHub Stats → ⭐ Stars | 🍴 Forks | 🐞 Issues | 👤 Contributors

Wikipedia Summary → Clean, structured, fallback-resistant

🧪 2. Verification

Cross-checks claims

Validates GitHub & Wiki evidence

Produces a confidence score

🧠 3. Architectural Recommendation

Executive summary

Pros & Cons

CTO-level verdict

♊ 4. Gemini Deep-Dive

A “Second Opinion” analysis that identifies blind spots and adds context.

🔥 Core Features (with visual flair)
🧠 Multi-Agent Architecture

📊 Analyst Agent — Collects evidence and generates the technical summary

🔍 Verification Agent — Validates claims, reduces hallucinations

🏛️ Advisor Agent — Crafts the final architectural verdict

🔗 A2A EventBus — Traceable, timestamped agent communication

🧩 Real Integrations

✔ GitHub REST API (stars, forks, issues, contributors)
✔ Wikipedia Summary API (fallback logic + sanitization)
✔ Groq Llama Models for blazing-fast LLM responses
✔ Gemini Models for deeper insights
✔ Simulator Mode when LLM keys are missing

📡 Live Observability Dashboard

Realtime Event Log with color-coded agents

Parallel Execution Timeline — visually shows concurrency

Per-Agent Metrics — API calls, latency, tasks

Confidence Scoring System

Microbenchmark Performance Charts (Altair)

⚙️ Scalable Orchestration

True parallelism with ThreadPoolExecutor

Mission concurrency control

Queue/reject logic for demo hall stability

📦 Downloadable Report

Analyst Output

Verification Notes

Confidence Score

Advisor Recommendation

Evidence + Metrics

JSON Export

🏛️ Architecture Diagram
Streamlit UI
     │
     │  Start Mission
     ▼
┌────────────────────┐
│   Orchestrator     │
└────────────────────┘
     │
     ├────► Analyst Agent ───┐
     │                       │  A2A Messages
     ├────► Verifier Agent ◄─┘
     │
     └────► Advisor Agent

Agents → EventBus (logs)
Metrics → Monitor
Report → JSON Output

🧱 Tech Stack
Component	Technology
Frontend/UI	Streamlit
Agents	Python OOP Agents
LLM Backend	Groq, Gemini, Simulator
Evidence APIs	GitHub REST, Wikipedia
Parallelism	ThreadPoolExecutor
Observability	EventBus + Monitor
Charts	Altair
⚙️ Setup Instructions
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Add Secrets

File: .streamlit/secrets.toml

GROQ_API_KEY=""
GITHUB_TOKEN=""
GEMINI_API_KEY=""
WIKI_API_KEY=""
UPTIME_URL=""
MAX_CONCURRENT_MISSIONS="3"
QUEUE_MISSIONS="true"

3️⃣ Run the app
streamlit run app.py

🧪 How to Use StackBench
1. Enter a GitHub repo

Example:

pytorch/pytorch

2. Select a mission

Adoption Analysis

Code Audit

Migration Plan

3. Watch agents run in parallel

Analyst ↔ Verifier show overlapping timeline bars.

4. Explore evidence

GitHub stats

Wikipedia summary

Agent logs

5. Ask Gemini for deeper assessment
6. Download the final JSON report
📄 Example Output (Snapshot)
{
  "target": "streamlit/streamlit",
  "analyst_summary": "...",
  "verification_status": "Verified",
  "confidence_score": 92,
  "advisor_recommendation": "...",
  "github_stats": {},
  "metrics": {}
}

⚠️ Limitations

Wikipedia entries can be outdated

GitHub API rate limits apply

Fallback simulator used when LLM keys missing

🧭 Roadmap

Vector-based retrieval

Multi-mission analytics dashboard

Snippet-level code quality scoring

OpenAPI fact verification

🏁 Conclusion

StackBench demonstrates excellence in:

✨ Multi-Agent Collaboration
✨ Evidence-Driven Analysis
✨ Parallel Orchestration
✨ Live Observability & Metrics
✨ Gemini + Groq Hybrid Reasoning
✨ Clean, Modern UI
