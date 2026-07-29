# Architecture decisions

## C1: One local model instance

The system starts with one Qwen3 8B instance in Ollama. Roles are separate
prompts and permissions, executed sequentially. This preserves RAM on a 16 GB
laptop and makes traces comparable.

## C2: Structured state before agent conversation

Agents exchange typed `Task`, `Plan`, `Evidence`, `NodeResult`,
`VerificationResult`, and `Lesson` records. They do not rely on unrestricted
free-form messages as the source of truth.

## C3: Deterministic checks before LLM verification

The system first checks schemas, URLs, citation/source linkage, required
sections, limits, and graph validity in Python. The LLM verifier is reserved
for evidence-to-claim comparison.

## C4: SQLite before managed services

SQLite stores task state, traces, failures, and lessons. A separate vector
store is introduced only in Phase 4, after a retrieval baseline exists.

## C5: Recipes instead of generated code

"Tool synthesis" first means versioned, allowlisted JSON workflow recipes.
Candidate recipes require evaluation and human approval before use.

## C6: No unverified memory as fact

Memories include type, source, timestamp, confidence, and validity period.
Reflections are procedural advice, not evidence for research claims.
