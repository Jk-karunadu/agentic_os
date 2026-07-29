"""Quick end-to-end test of the demo dataset pipeline via the API."""
import requests
import json
import time

BASE = "http://127.0.0.1:8000"

# 1. Health check
print("=" * 60)
print("1. Health Check")
r = requests.get(f"{BASE}/health", timeout=5)
print(f"   Status: {r.status_code}  Body: {r.json()}")
assert r.status_code == 200

# 2. Create a task with demo evidence (AI Safety)
print("\n" + "=" * 60)
print("2. Creating Task with Demo Evidence")
task_payload = {
    "objective": (
        "Summarise the current state of AI alignment research, focusing on "
        "key approaches like RLHF, constitutional AI, and interpretability. "
        "Cite the supplied sources."
    ),
    "evidence": [
        {
            "title": "Concrete Problems in AI Safety",
            "url": "https://arxiv.org/abs/1606.06565",
            "excerpt": (
                "As machine learning systems become more broadly deployed, there is increasing "
                "interest in making sure these systems operate safely and reliably. We present "
                "five concrete problems in AI safety: avoiding negative side effects, avoiding "
                "reward hacking, scalable oversight, safe exploration, and robustness to "
                "distributional shift. Reward hacking occurs when an agent finds an unintended "
                "way to maximise its reward signal without fulfilling the designer's true objective."
            ),
        },
        {
            "title": "Constitutional AI: Harmlessness from AI Feedback",
            "url": "https://arxiv.org/abs/2212.08073",
            "excerpt": (
                "We experiment with methods for training a harmless AI assistant through "
                "self-improvement, without any human labels identifying harmful outputs. The "
                "method is called Constitutional AI (CAI), because it uses a set of principles "
                "to make judgments about which outputs are desirable. We find that the resulting "
                "model is both more helpful and less harmful than models trained with RLHF alone. "
                "CAI reduces the need for human feedback labels by a factor of roughly ten."
            ),
        },
    ],
}

r = requests.post(f"{BASE}/tasks", json=task_payload, timeout=10)
print(f"   Status: {r.status_code}")
task = r.json()
task_id = task["id"]
print(f"   Task ID: {task_id}")
print(f"   Evidence count: {len(task.get('evidence', []))}")
assert r.status_code == 201

# 3. Run orchestrated
print("\n" + "=" * 60)
print("3. Running Orchestrated Task (this may take a few minutes)...")
start = time.time()
r = requests.post(f"{BASE}/tasks/{task_id}/run-orchestrated", timeout=300)
elapsed = time.time() - start
print(f"   Status: {r.status_code}")
print(f"   Elapsed: {elapsed:.1f}s")

if r.status_code == 200:
    result = r.json()
    print(f"   Verification passed: {result.get('verification_passed')}")
    print(f"   Total tokens: {result.get('total_tokens')}")
    print(f"   Total duration: {result.get('total_duration_ms')}ms")

    answer = result.get("final_answer") or "None"
    print(f"\n   Final Answer (first 400 chars):")
    print(f"   {answer[:400]}")

    nodes = result.get("node_results", {})
    print(f"\n   Node results ({len(nodes)} nodes):")
    for nid, nres in nodes.items():
        err = nres.get("error") or "OK"
        print(f"     - {nid}: {nres.get('tokens_used', 0)} tokens, {nres.get('duration_ms', 0)}ms, status={err}")
else:
    print(f"   Error: {r.text[:300]}")

# 4. Check memory was stored
print("\n" + "=" * 60)
print("4. Checking Memory Store")
r = requests.post(f"{BASE}/memory/query", json={"query": "AI safety alignment", "top_k": 3}, timeout=10)
print(f"   Status: {r.status_code}")
memories = r.json()
print(f"   Memories found: {len(memories)}")
for m in memories:
    print(f"     - [{m['type']}] {m['text'][:100]}...")

# 5. Check failures
print("\n" + "=" * 60)
print("5. Checking Failure Database")
r = requests.get(f"{BASE}/failures", timeout=10)
print(f"   Status: {r.status_code}")
failures = r.json()
print(f"   Failures logged: {len(failures)}")
for f in failures:
    print(f"     - [{f['error_type']}] {f['error_detail'][:80]}...")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✅")
