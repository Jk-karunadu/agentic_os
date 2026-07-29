# Phase 2 — Planning and controlled execution

## What is implemented

- Qwen3 creates JSON task graphs using only the approved node types:
  `research`, `analyze`, `draft`, and `verify`.
- Python validates every graph before it is persisted: node limits, unique IDs,
  dependency existence, no self-dependencies, and acyclicity.
- Each task plan is versioned in SQLite. Replanning creates a new version rather
  than overwriting the earlier plan.
- The deterministic scheduler manages node states:
  `pending -> ready -> running -> completed|failed`.
- A node can start only when all dependencies completed successfully. A failed
  node does not unlock its dependents.

## API workflow

1. `POST /tasks/{task_id}/plan` creates a validated plan.
2. `GET /tasks/{task_id}/plan/ready` lists nodes eligible for dispatch.
3. `POST /tasks/{task_id}/plan/nodes/{node_id}/start` begins one ready node.
4. `POST /tasks/{task_id}/plan/nodes/{node_id}/complete` records success or
   failure.
5. `POST /tasks/{task_id}/replan` creates a new plan version after a failure.

Worker-specific execution is added in later phases; this phase intentionally
keeps scheduling separate from tool execution so permissions and failure
handling remain testable.
