# Phase 0 — Research and implementation contract

## Objective

Build a local-first AgentOS prototype that answers research-synthesis tasks
through an inspectable control loop:

`plan -> execute -> verify -> reflect -> store lesson -> replan/retry`

The system is not claimed to retrain an LLM. "Self-improving" means that it
uses verified experience to improve memory retrieval, prompt/routing choices,
planning, and approved reusable workflows.

## Research question

> For repeated multi-step research-synthesis tasks, do reflection-driven
> procedural memory and evidence-grounded verification improve task quality and
> citation quality over a stateless local-agent baseline at an acceptable cost
> and latency?

## Initial domain and non-goals

### In scope

- Research questions that need primary-source discovery, evidence extraction,
  comparison, drafting, and citation checking.
- A single local model served by Ollama.
- Read-only research tools in the first implementation.
- Structured traces, budgets, deterministic checks, and reproducible tests.

### Explicitly deferred

- Autonomous code or shell execution.
- Email, payments, authentication, browser login, and any irreversible action.
- Vision, SQL, Neo4j, Kubernetes, and arbitrary generated tools.
- Multiple simultaneous model instances.

## Success metrics

| Metric | Measurement |
|---|---|
| Task score | Fixed human/deterministic rubric, 0–10 |
| Citation precision | Correctly supported cited claims / cited claims |
| Citation coverage | Important factual claims with evidence / all important factual claims |
| Plan completion | Successfully completed DAG nodes / planned nodes |
| Verification catch rate | Seeded or later-confirmed defects caught / all defects |
| Retry rate | Retries / runs |
| Memory utility | Held-out repeated-task score with memory minus without memory |
| Cost and latency | Local runtime seconds, prompt/output tokens, and RAM telemetry |

## System invariants

1. LLM outputs are suggestions; Python validates all schemas and permissions.
2. Tools are explicit capabilities, never arbitrary model-generated commands.
3. Every factual claim must retain source provenance.
4. The verifier receives the evidence, not merely the draft.
5. Reflection records a lesson only after a concrete failure signal.
6. Automatic retries are capped at one in early phases.
7. Runs have limits for nodes, tools, time, context, and tokens.
8. Every experiment records configuration and model version.

## Acceptance gate for Phase 0

- [x] Single domain selected: research synthesis.
- [x] Research question written.
- [x] Metrics defined before implementation.
- [x] Safety and scope boundaries set.
- [x] Development and held-out tasks drafted.
- [ ] User confirms the chosen scope before Phase 1 installs dependencies.
