# AgentOS: A Self-Improving Multi-Agent Operating System

*An operating system for AI agents that learns from failures and becomes better over time.*

**AgentOS** shifts the paradigm from simple User-to-LLM chatbots to a comprehensive **Windows/Linux for AI Agents**. It manages state, memory, tool usage, planning, and task execution for a fleet of specialized AI workers. 

Built completely locally using Ollama (e.g. Qwen, Llama3) and Nomic-Embed-Text for vector embeddings, AgentOS protects privacy and guarantees offline accessibility.

## Architecture & Features

AgentOS provides the foundational primitives required for true agentic autonomy:

1. **Local Compute Engine**: Driven completely by local Ollama instances (`OllamaClient`). No API keys required.
2. **Deterministic Verification Layer**: Validates agent outputs against strict formatting, citation, and relevance rules *before* exposing them to the user.
3. **Failure Database & Reflection**: When an agent makes a mistake, it catches the error, reflects to generate a "procedural lesson", and retries the task.
4. **Persistent Long-Term Memory (`MemoryStore`)**:
   - **Procedural Memory**: Lessons learned from past failures, dynamically injected into the system prompt for similar future tasks.
   - **Episodic Memory**: Stored experiences and factual summaries retrieved using semantic vector similarity (powered by local embeddings) and BM25-style keyword fallback.
5. **Dynamic Planning (`PlanScheduler`)**: Automatically decomposes complex objectives into an executable DAG (Directed Acyclic Graph) of sub-nodes.
6. **Multi-Agent Orchestration (`WorkerDispatcher`)**: Routes specialized tasks to specialized agents (e.g., `ResearchWorker`, `AnalystWorker`, `WriterWorker`, `VerifierWorker`).
7. **Tool-Use Integration**: Agents can invoke native JSON-schema tools in a ReAct loop to interact with `MemoryQueryTool` (for past experiences) and `WebSearchTool` (for current information).

## Project Structure

```text
app/
├── api/             # API routes
├── models/          # Pydantic models & SQL schemas
├── tools/           # JSON Schema tools (Memory, Wikipedia)
├── workers/         # Multi-Agent worker implementations
├── database.py      # SQLite task and trace tracking
├── memory.py        # Semantic memory DB with ranking algorithms
├── planner.py       # DAG node scheduling and execution logic
├── reflection.py    # Automatic error deduction and procedural learning
├── verification.py  # Deterministic constraints logic
└── main.py          # FastAPI application
```

## Running the Application

### 1. Requirements
- Python 3.12+
- `pip install -r requirements.txt`
- [Ollama](https://ollama.com/) running locally with models available (e.g., `qwen2.5` and `nomic-embed-text`).

### 2. Start the Server
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
*(Optionally use `--reload` for development)*

### 3. Usage

AgentOS exposes several key endpoints for interacting with the system:

**Baseline Task (Phase 1):**
```bash
POST /tasks/{task_id}/run
```

**Verified Self-Improving Task (Phase 3):**
```bash
POST /tasks/{task_id}/run-verified
```
*(Runs the task, verifies it, and auto-reflects/retries if it fails).*

**Fully Orchestrated Multi-Agent DAG (Phase 5 & 6):**
```bash
POST /tasks/{task_id}/run-orchestrated
```
*(Executes a dynamic node-by-node plan using Researchers, Analysts, and Writers equipped with Tool access).*

## Benchmarking (PM-Bench Inspired)

Run the included `benchmark.py` script to compare the Baseline performance against the Orchestrated AgentOS pipeline.

```bash
python benchmark.py
```

The benchmark evaluates:
- **Success Rate**: Verified accuracy against strict logical constraints.
- **Latency**: Total execution time per task.
- **Tokens**: Total token overhead of the orchestration layer.

*Expect the orchestrated approach to achieve significantly higher success rates at the cost of execution time and tokens, especially as the procedural failure database accumulates lessons.*
