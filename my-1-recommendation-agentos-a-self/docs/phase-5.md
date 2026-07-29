# Phase 5 — Multi-Agent Workers & Orchestration

## What is implemented

### Specialized Workers
We implemented a multi-agent system where different node types in the execution plan are handled by specialized workers. Each worker inherits from `BaseWorker` and uses a generic `chat()` method added to the `OllamaClient`.

| Worker | Role | Node Type | Description |
|--------|------|-----------|-------------|
| **ResearchWorker** | Researcher | `research` | Extracts relevant facts from source evidence based on the objective. Will accurately point out if evidence is insufficient. |
| **AnalystWorker** | Analyst | `analyze` | Synthesizes prior research outputs and structures arguments for drafting, without writing the final text. |
| **WriterWorker** | Writer | `draft` | Drafts the final response using synthesized context and evidence. Incorporates prior reflection lessons. |
| **VerifierWorker** | Verifier | `verify` | Executes the deterministic checks (and LLM verify if passed). Fails the node if verification issues are found. |

### Dispatcher and Context
The `WorkerDispatcher` routes a `PlanNode` to the correct worker based on its `type`.
Workers receive a `WorkerContext` containing:
- The overall task objective
- Evidence items
- Previously executed node outputs (dependencies)
- Procedural reflection lessons to avoid prior mistakes
- The execution budget

### Orchestrated Run Endpoint
We implemented `POST /tasks/{task_id}/run-orchestrated`:
1. Ensures a plan exists (generates one if missing).
2. Uses `PlanScheduler` to find `ready` nodes.
3. Dispatches each ready node to the appropriate specialized worker.
4. If a worker succeeds, its output is saved in context and the node is marked `completed`.
5. If a worker (e.g. `VerifierWorker`) fails, the orchestration halts, marks the task `failed`, and returns the issues.
6. Auto-stores episodic memory (Phase 4 integration) using the final draft output if the task succeeds.

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/tasks/{task_id}/run-orchestrated` | Runs the full workflow graph node-by-node via specialized agents. |

## Test coverage

Added `tests/test_workers.py` to cover:
- Worker initialization and specific role prompt handling
- Verifier success paths
- The full mocked `run-orchestrated` DAG execution
Total tests: 42 (all passing).
