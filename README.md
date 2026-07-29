# Agentic OS: Autonomous AI Orchestration Engine

Agentic OS is a state-of-the-art framework that treats AI agents like programs running on an Operating System. Instead of a linear, rigid prompt chain, this system utilizes a **Directed Acyclic Graph (DAG) Orchestrator** to schedule, monitor, and execute autonomous AI agents. 

By wrapping unpredictable Large Language Models inside traditional operating system constraints—including dynamic process scheduling, persistent procedural memory, and robust fault-tolerance—Agentic OS delivers enterprise-grade reliability and self-improving AI workflows.

---

## 🚀 Key Features

* **Hybrid Agentic Architecture:** Combines macro-level DAG orchestration (for reliable planning and fault tolerance) with micro-level autonomous agents (which use the ReAct paradigm to dynamically decide which tools to use).
* **Self-Improving Procedural Memory:** When an AI agent hallucinates or fails a task, the system catches the error, intercepts the "crash", and initiates an autonomous retry loop. It extracts the root cause of the failure and permanently stores the "Lesson Learned" in an SQLite database, ensuring the AI never makes the same mistake twice.
* **Deterministic Verification Layer:** Hard-coded guardrails prevent the AI from outputting unsupported claims, fabricated URLs, or improperly formatted data.
* **Tool-Use Integration:** Isolated AI agents are granted secure, sandboxed access to external tools (like Web Search, URL Scrapers, and Python Code Executors) via a central `ToolRegistry`.
* **Rich Real-Time UI:** A beautifully designed Streamlit frontend that visualizes internal AI agent thoughts, execution graphs, latency, and the Reflection Database in real-time.

---

## 🧠 How The Architecture Works

Agentic OS mimics a traditional computer operating system:

1. **The Kernel (DAG Orchestrator):** `app/main.py` acts as the process scheduler. It spawns "workers" (Researchers, Writers, Verifiers), passes state between them, and rewires the execution graph dynamically if a node fails.
2. **RAM & Hard Drive (Memory Management):** Short-term working memory (`WorkerContext`) holds immediate draft text during execution. Long-term memory (`app/store.py` and SQLite) permanently stores Semantic facts and Procedural lessons.
3. **System Calls (Tool Registry):** Agents cannot directly access the internet. They make "Tool Calls" to the `ToolRegistry`, which safely executes Python scripts to search the web or read files, and hands the data back.
4. **Exception Handling:** The system treats AI hallucinations like a software crash (e.g., a Segmentation Fault). The Verifier node catches these, prevents the system from failing, and forces the drafting agent to fix the output.

---

## 🛠️ Tech Stack

* **Backend Orchestration:** Python 3.12+, FastAPI, Uvicorn, Pydantic
* **AI / LLM Layer:** Ollama (100% Local Inference), LangGraph concepts, ReAct paradigm
* **Database & Memory:** SQLite, Vector Embeddings (Semantic & Procedural Memory)
* **Frontend Visualization:** Streamlit, Mermaid.js (for DAG graphs)

---

## 💻 Installation & Setup

### Prerequisites
1. **Python 3.11+** installed on your machine.
2. **Ollama** installed and running locally. You will need to pull your preferred model (e.g., `ollama run qwen2.5:7b`).

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/agentic-os.git
cd agentic-os
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ How to Run the OS

Agentic OS requires two separate processes to run simultaneously: the FastAPI backend (The OS Kernel) and the Streamlit frontend (The Dashboard).

### Step 1: Start the Backend (Kernel)
Open a terminal in the root directory and run:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
*This will start the orchestrator and initialize the SQLite database (`data/agentos.db`).*

### Step 2: Start the Frontend (UI)
Open a **second** terminal, activate your virtual environment, and run:
```bash
streamlit run ui/app.py
```
*This will open the beautiful dashboard in your browser where you can submit tasks, watch the DAG execute in real-time, and view the Reflection Database.*

---

## 🧹 Managing Memory

To prove the system's ability to self-improve, you can wipe its long-term procedural memory to start fresh. 

Run the included script:
```bash
python reset_memory.py
```
This will delete all historical Failure Logs and Procedural Lessons from the database, allowing you to demonstrate the AI failing, learning, and succeeding on subsequent runs.
