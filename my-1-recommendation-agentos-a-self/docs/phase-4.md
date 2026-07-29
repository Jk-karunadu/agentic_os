# Phase 4 — Persistent Memory System

## What is implemented

### Memory types

| Type | Purpose | Auto-stored |
|------|---------|-------------|
| **Episodic** | Task runs: objective, outcome, answer excerpt | After every `run-verified` completion |
| **Procedural** | Lessons from reflection: what went wrong, recommended fix | After every reflection |
| **Semantic** | Durable facts extracted from verified answers | Manual / future LLM extraction |

### Storage

SQLite tables:

- `memory_items`: id, type, text, embedding (BLOB), importance, source_task_id,
  created_at, last_accessed, success_score, valid_from, valid_to, metadata_json
- `entity_relations`: subject, predicate, object, source_id, confidence,
  valid_from, valid_to

### Embeddings

- Primary: Ollama `nomic-embed-text` model via `/api/embed`
- Fallback: keyword-overlap scoring when the embedding model is unavailable

### Weighted retrieval formula

```
score = 0.45 × similarity
      + 0.20 × importance
      + 0.15 × recency
      + 0.15 × success_score
      + 0.05 × type_match
```

- **Similarity**: cosine (if embeddings available) or keyword overlap (fallback)
- **Recency**: exponential decay with a 1-week half-life
- **Type match**: 1.0 if queried type matches the item type

Weights are hand-tuned; Phase 7 may adjust them using held-out evaluation data.

### Explainable retrieval

`POST /memory/explain` returns a `MemoryScoreBreakdown` for each result showing
the individual component scores.  This supports the architecture principle that
memory retrieval must be inspectable.

### Graph memory (entity relations)

Stored in `entity_relations`.  Queryable by subject, predicate, and/or object.
Intended for knowledge-graph patterns like:

```
Paper ──proposes──> Method
Method ──evaluated_on──> Benchmark
```

### Automatic memory lifecycle

When `POST /tasks/{id}/run-verified` completes:

1. **Episodic memory** is auto-stored with the task objective, outcome, and
   answer excerpt.  Importance is 0.7 for passed tasks, 0.5 for failed.
2. **Procedural memory** is auto-stored from each reflection lesson with the
   lesson's confidence as importance.

### Memory safety

- Every memory item tracks `source_task_id`, `created_at`, and `valid_from/to`.
- `prune_expired()` removes items past their `valid_to` date.
- Reflections are stored as procedural advice, not as evidence for research claims
  (per architecture decision C6).

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/memory` | Store a memory item |
| POST | `/memory/query` | Retrieve ranked memories |
| GET | `/memory/{id}` | Inspect a single memory |
| POST | `/memory/explain` | Retrieval score breakdowns |
| POST | `/memory/relations` | Store an entity relation |
| GET | `/memory/relations/query` | Query relations |

## Test coverage

11 new tests in `test_memory.py`.  Total project tests: 37, all passing.
