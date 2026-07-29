"""Tests for the persistent memory system (Phase 4)."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi.testclient import TestClient

from app.embeddings import keyword_similarity
from app.memory import MemoryStore
from app.models import (
    EntityRelation, MemoryItem, MemoryQuery, MemoryType,
    RunTrace, TaskResult, TaskStatus,
)


# ── Embedding fallback ──────────────────────────────────────────────

def test_keyword_similarity_basic() -> None:
    assert keyword_similarity("agent memory retrieval", "memory retrieval for agents") > 0.5
    assert keyword_similarity("agent memory", "unrelated topic about cars") < 0.2
    assert keyword_similarity("", "anything") == 0.0


# ── MemoryStore unit tests ──────────────────────────────────────────

def _tmp_store(tmp_path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "test_mem.db"))


def test_store_and_retrieve_memory(tmp_path) -> None:
    ms = _tmp_store(tmp_path)
    item = MemoryItem(
        type=MemoryType.PROCEDURAL, text="Always cite sources in research briefs.",
        importance=0.9, source_task_id=str(uuid4()),
    )
    ms.store_memory(item)

    results = ms.retrieve(MemoryQuery(query="cite sources in research", top_k=3))
    assert len(results) >= 1
    assert results[0].id == item.id


def test_retrieve_filters_by_type(tmp_path) -> None:
    ms = _tmp_store(tmp_path)
    ms.store_memory(MemoryItem(type=MemoryType.EPISODIC, text="Task about memory benchmarks completed."))
    ms.store_memory(MemoryItem(type=MemoryType.PROCEDURAL, text="Lesson: always verify citations."))

    results = ms.retrieve(
        MemoryQuery(query="citations", memory_type=MemoryType.PROCEDURAL, top_k=5),
    )
    assert all(r.type == MemoryType.PROCEDURAL for r in results)


def test_retrieve_filters_by_min_importance(tmp_path) -> None:
    ms = _tmp_store(tmp_path)
    ms.store_memory(MemoryItem(type=MemoryType.SEMANTIC, text="Low importance fact.", importance=0.1))
    ms.store_memory(MemoryItem(type=MemoryType.SEMANTIC, text="High importance fact.", importance=0.9))

    results = ms.retrieve(MemoryQuery(query="fact", min_importance=0.5, top_k=10))
    assert all(r.importance >= 0.5 for r in results)


def test_explain_retrieval_returns_breakdowns(tmp_path) -> None:
    ms = _tmp_store(tmp_path)
    ms.store_memory(MemoryItem(type=MemoryType.PROCEDURAL, text="Lesson about agent reflection."))
    ms.store_memory(MemoryItem(type=MemoryType.EPISODIC, text="Task on agent planning completed."))

    breakdowns = ms.explain_retrieval(MemoryQuery(query="agent reflection", top_k=5))
    assert len(breakdowns) >= 1
    bd = breakdowns[0]
    assert bd.total_score > 0
    assert bd.similarity >= 0
    assert bd.recency >= 0


def test_recency_decays_over_time(tmp_path) -> None:
    ms = _tmp_store(tmp_path)
    old_time = datetime.now(UTC) - timedelta(days=30)
    recent = MemoryItem(type=MemoryType.EPISODIC, text="Recent task about memory systems.")
    old = MemoryItem(
        type=MemoryType.EPISODIC, text="Old task about memory systems.",
        last_accessed=old_time, created_at=old_time,
    )
    ms.store_memory(recent)
    ms.store_memory(old)

    results = ms.retrieve(MemoryQuery(query="memory systems", top_k=2))
    # Recent should rank higher due to recency boost.
    assert results[0].id == recent.id


def test_prune_expired_memories(tmp_path) -> None:
    ms = _tmp_store(tmp_path)
    expired = MemoryItem(
        type=MemoryType.SEMANTIC, text="Temporary fact that is now expired.",
        valid_to=datetime.now(UTC) - timedelta(hours=1),
    )
    current = MemoryItem(type=MemoryType.SEMANTIC, text="Current valid fact.")
    ms.store_memory(expired)
    ms.store_memory(current)

    removed = ms.prune_expired()
    assert removed == 1
    assert ms.get_memory(expired.id) is None
    assert ms.get_memory(current.id) is not None


# ── Entity relations ────────────────────────────────────────────────

def test_store_and_query_relations(tmp_path) -> None:
    ms = _tmp_store(tmp_path)
    rel = EntityRelation(subject="Reflexion", predicate="proposes", object="episodic memory for agents")
    stored = ms.store_relation(rel)
    assert stored.id is not None

    results = ms.query_relations(subject="Reflexion")
    assert len(results) == 1
    assert results[0].predicate == "proposes"

    all_rels = ms.query_relations()
    assert len(all_rels) >= 1


# ── Memory API integration ──────────────────────────────────────────

def test_memory_api_store_and_query() -> None:
    from app.main import app
    client = TestClient(app)

    # Store a memory.
    item = client.post("/memory", json={
        "type": "procedural",
        "text": "When drafting, always cite the supplied evidence URLs.",
        "importance": 0.85,
    })
    assert item.status_code == 201
    memory_id = item.json()["id"]

    # Get by ID.
    fetched = client.get(f"/memory/{memory_id}")
    assert fetched.status_code == 200
    assert "cite" in fetched.json()["text"]

    # Query.
    results = client.post("/memory/query", json={
        "query": "cite evidence URLs in drafts",
        "top_k": 3,
    })
    assert results.status_code == 200
    assert len(results.json()) >= 1

    # Explain.
    explained = client.post("/memory/explain", json={
        "query": "cite evidence",
        "top_k": 3,
    })
    assert explained.status_code == 200
    assert len(explained.json()) >= 1
    assert "total_score" in explained.json()[0]


def test_memory_not_found_returns_404() -> None:
    from app.main import app
    client = TestClient(app)
    resp = client.get("/memory/nonexistent-id-12345")
    assert resp.status_code == 404


def test_run_verified_auto_stores_memories() -> None:
    """The run-verified endpoint should auto-store episodic + procedural memories."""
    from app.main import app, llm

    def fake_brief(task_id, objective, evidence):
        return TaskResult(
            task_id=task_id, answer="Short.",
            trace=RunTrace(task_id=task_id, model="test", status=TaskStatus.COMPLETED),
        )

    def fake_retry(task_id, objective, evidence, lessons):
        return TaskResult(
            task_id=task_id,
            answer="## Comparison\nPer https://example.com/s, agents learn. " * 3,
            trace=RunTrace(task_id=task_id, model="test", status=TaskStatus.COMPLETED),
        )

    original_brief = llm.research_brief
    original_retry = getattr(llm, "research_brief_with_lessons", None)
    llm.research_brief = fake_brief
    llm.research_brief_with_lessons = fake_retry
    try:
        client = TestClient(app)
        task_id = client.post("/tasks", json={
            "objective": "Compare approaches using supplied evidence only.",
            "evidence": [{"title": "S", "url": "https://example.com/s", "excerpt": "Evidence."}],
        }).json()["id"]

        client.post(f"/tasks/{task_id}/run-verified")

        # Should have auto-stored memories.
        memories = client.post("/memory/query", json={
            "query": "Compare approaches",
            "top_k": 10,
        })
        assert memories.status_code == 200
        texts = [m["text"] for m in memories.json()]
        # At least an episodic + procedural memory should exist.
        has_episodic = any("Task:" in t for t in texts)
        has_procedural = any("Lesson" in t for t in texts)
        assert has_episodic or has_procedural
    finally:
        llm.research_brief = original_brief
        if original_retry:
            llm.research_brief_with_lessons = original_retry
