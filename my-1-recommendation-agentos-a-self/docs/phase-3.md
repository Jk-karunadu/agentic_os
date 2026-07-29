# Phase 3 — Verification, Reflection & Failure Database

## What is implemented

### Expanded deterministic verifier

Six check types run without any LLM call:

| Check | Severity | Trigger |
|-------|----------|---------|
| `citation_mismatch` | high | Answer cites a URL not in supplied evidence |
| `missing_citation` | medium | Evidence was supplied but answer cites no URLs |
| `source_not_cited` | low | A supplied evidence source was never referenced |
| `missing_section` | medium | A required section heading is absent |
| `content_too_short` | medium | Answer < 100 chars when evidence was provided |
| `fabricated_url` | high | (Logical extension of citation_mismatch) |

The verifier accepts an optional `required_sections` list from the eval rubric.

### Expanded reflection module

- Processes **all** verification issues (not just the first).
- Classifies each issue into the failure taxonomy: `retrieval_failure`,
  `unsupported_claim`, `citation_mismatch`, `planning_error`,
  `source_quality_failure`, `missing_section`, `content_too_short`.
- Generates a composite lesson aggregating unique triggers and repair actions.
- Confidence is set by the highest-severity issue (0.92 for high, 0.80 for
  medium, 0.65 for low).
- `llm_reflect()` uses Qwen3 for root-cause analysis with automatic fallback to
  the deterministic path when Ollama is unavailable.

### Failure database

SQLite `failures` table stores structured records:

- `task_id`, `plan_version`, `failing_node`
- `error_type` (from the taxonomy), `error_detail`
- `verification_json`, `lesson_json`
- `repair_attempted`, `repair_succeeded`
- `model`, `created_at`

API endpoints:

- `GET /failures` — list all failures (optionally filter by `task_id`).
- `GET /failures/by-type/{error_type}` — query by error category.

### Retry-with-lesson endpoint

`POST /tasks/{task_id}/run-verified` chains the full self-improvement loop:

1. **Run** the task with the local LLM.
2. **Verify** the answer deterministically.
3. If passed → return immediately.
4. If failed → **reflect** (generate lesson) → **store** lesson + failure →
   **retry once** with lessons injected as constraints → **verify** retry →
   **update** failure record with repair outcome → return composite result.

The retry prompt injects lesson `recommended_action` texts as explicit
constraints in the system prompt.

### Key design decisions

- Retries are capped at **one** to prevent runaway loops.
- The deterministic verifier always runs first; the LLM verifier is only invoked
  when the deterministic pass succeeds and evidence is available.
- Failure records are immutable after creation (only `repair_attempted` and
  `repair_succeeded` are updated).
- The `POST /tasks/{task_id}/run` baseline is preserved for Phase 7 ablation.

## Test coverage

15 new tests across `test_verification.py` (8) and `test_reflection.py` (7).
Total: 26 tests, all passing.
