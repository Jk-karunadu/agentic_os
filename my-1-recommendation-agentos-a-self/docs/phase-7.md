# Phase 7 — Ablation, Evaluation & Benchmarks

## Overview
This phase introduces a benchmarking suite (`benchmark.py`) to systematically compare the baseline Agent execution against the fully orchestrated AgentOS pipeline.

## PM-Bench Inspired Benchmarking
Inspired by PM-Bench (a framework for evaluating multi-agent systems and their ability to self-improve and utilize memory), the benchmark evaluates the system's performance on predefined research tasks containing structured evidence.

The script executes two modes for each task:
1. **Baseline Mode (`/run`)**: A simple pass-through where the agent attempts to fulfill the objective with no planning, no tool-use, no memory, and no verification.
2. **Orchestrated Mode (`/run-orchestrated`)**: The full AgentOS pipeline:
   - Uses `PlanScheduler` to map out a DAG of tasks.
   - Deploys specialized agents (`ResearchWorker`, `AnalystWorker`, `WriterWorker`, `VerifierWorker`).
   - ReAct tool loops are active, allowing agents to fetch procedural/episodic memory or search the web if evidence is lacking.
   - Evaluates outputs through deterministic and LLM-assisted verification.

## Metrics Captured
For each execution, the benchmark tracks:
- **Success Rate**: Did the final output pass the deterministic and logical verification constraints?
- **Avg Latency (ms)**: The total end-to-end execution time.
- **Avg Tokens**: The total sum of input and output tokens consumed by the local LLM.

## Results & Observations
The benchmark demonstrates the core value proposition of AgentOS:
- **Accuracy vs. Latency**: The orchestrated approach significantly increases the verification success rate by decomposing tasks and injecting reflection memory, at the cost of higher latency and token usage.
- **Self-Improvement**: As tasks are repeated, the system stores procedural memory (lessons learned from failures) and episodic memory. If the benchmark was run cyclically, AgentOS would iteratively eliminate mistakes that the Baseline model continues to make.
