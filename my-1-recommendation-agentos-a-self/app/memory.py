"""Persistent memory: store, retrieve with weighted ranking, and explain.

Storage: SQLite tables for memory_items and entity_relations.
Retrieval: weighted formula combining similarity, importance, recency,
success score, and type match.  When embeddings are available they drive
similarity; otherwise keyword overlap is used as a fallback.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from app.embeddings import cosine_similarity, embed_single, keyword_similarity
from app.models import (
    EntityRelation,
    MemoryItem,
    MemoryQuery,
    MemoryScoreBreakdown,
    MemoryType,
)

# ── Retrieval weights (hand-tuned; Phase 7 may adjust) ──────────────
W_SIMILARITY = 0.45
W_IMPORTANCE = 0.20
W_RECENCY = 0.15
W_SUCCESS = 0.15
W_TYPE_MATCH = 0.05

# Recency half-life in hours — memories older than this decay toward 0.
_RECENCY_HALF_LIFE_HOURS = 168  # 1 week


class MemoryStore:
    """SQLite-backed memory with weighted retrieval and provenance."""

    def __init__(self, database_path: str) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialise(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB,
                    importance REAL DEFAULT 0.5,
                    source_task_id TEXT,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    success_score REAL DEFAULT 0.0,
                    valid_from TEXT,
                    valid_to TEXT,
                    metadata_json TEXT DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    source_id TEXT,
                    confidence REAL DEFAULT 0.5,
                    valid_from TEXT,
                    valid_to TEXT
                )
                """
            )

    # ── Store ────────────────────────────────────────────────────────

    def store_memory(self, item: MemoryItem) -> MemoryItem:
        """Persist a memory item, computing its embedding if possible."""
        embedding_blob = None
        vec = embed_single(item.text)
        if vec is not None:
            embedding_blob = json.dumps(vec).encode()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_items
                (id, type, text, embedding, importance, source_task_id,
                 created_at, last_accessed, success_score, valid_from, valid_to, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id, item.type.value, item.text, embedding_blob,
                    item.importance, item.source_task_id,
                    item.created_at.isoformat(), item.last_accessed.isoformat(),
                    item.success_score,
                    item.valid_from.isoformat() if item.valid_from else None,
                    item.valid_to.isoformat() if item.valid_to else None,
                    item.metadata_json,
                ),
            )
        return item

    def get_memory(self, memory_id: str) -> MemoryItem | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_items WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._row_to_item(row) if row else None

    # ── Retrieve with weighted ranking ───────────────────────────────

    def retrieve(
        self, query: MemoryQuery, query_type: MemoryType | None = None,
    ) -> list[MemoryItem]:
        """Return the top-k memory items ranked by the weighted formula."""
        candidates = self._load_candidates(query)
        if not candidates:
            return []

        query_vec = embed_single(query.query)
        now = datetime.now(UTC)

        scored: list[tuple[float, MemoryItem]] = []
        for item, embedding_blob in candidates:
            sim = self._compute_similarity(query.query, query_vec, item.text, embedding_blob)
            rec = self._recency_score(item.last_accessed, now)
            tm = 1.0 if (query_type and item.type == query_type) else 0.0

            total = (
                W_SIMILARITY * sim
                + W_IMPORTANCE * item.importance
                + W_RECENCY * rec
                + W_SUCCESS * item.success_score
                + W_TYPE_MATCH * tm
            )
            scored.append((total, item))

        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[: query.top_k]

        # Update last_accessed for returned items.
        with self._connect() as conn:
            for _, item in top:
                conn.execute(
                    "UPDATE memory_items SET last_accessed = ? WHERE id = ?",
                    (now.isoformat(), item.id),
                )

        return [item for _, item in top]

    def explain_retrieval(
        self, query: MemoryQuery, query_type: MemoryType | None = None,
    ) -> list[MemoryScoreBreakdown]:
        """Return ranked items with full score breakdowns."""
        candidates = self._load_candidates(query)
        if not candidates:
            return []

        query_vec = embed_single(query.query)
        now = datetime.now(UTC)

        breakdowns: list[MemoryScoreBreakdown] = []
        for item, embedding_blob in candidates:
            sim = self._compute_similarity(query.query, query_vec, item.text, embedding_blob)
            rec = self._recency_score(item.last_accessed, now)
            tm = 1.0 if (query_type and item.type == query_type) else 0.0

            total = (
                W_SIMILARITY * sim
                + W_IMPORTANCE * item.importance
                + W_RECENCY * rec
                + W_SUCCESS * item.success_score
                + W_TYPE_MATCH * tm
            )
            breakdowns.append(MemoryScoreBreakdown(
                memory_id=item.id, text=item.text[:200],
                total_score=round(total, 4),
                similarity=round(sim, 4),
                importance=round(item.importance, 4),
                recency=round(rec, 4),
                success=round(item.success_score, 4),
                type_match=round(tm, 4),
            ))

        breakdowns.sort(key=lambda b: b.total_score, reverse=True)
        return breakdowns[: query.top_k]

    # ── Entity relations (graph memory) ──────────────────────────────

    def store_relation(self, rel: EntityRelation) -> EntityRelation:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO entity_relations
                (subject, predicate, object, source_id, confidence, valid_from, valid_to)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rel.subject, rel.predicate, rel.object, rel.source_id,
                    rel.confidence,
                    rel.valid_from.isoformat() if rel.valid_from else None,
                    rel.valid_to.isoformat() if rel.valid_to else None,
                ),
            )
            rel.id = cursor.lastrowid
        return rel

    def query_relations(
        self, subject: str | None = None, predicate: str | None = None,
        object_: str | None = None,
    ) -> list[EntityRelation]:
        clauses: list[str] = []
        params: list[str] = []
        if subject:
            clauses.append("subject = ?")
            params.append(subject)
        if predicate:
            clauses.append("predicate = ?")
            params.append(predicate)
        if object_:
            clauses.append("object = ?")
            params.append(object_)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM entity_relations{where} ORDER BY id", params
            ).fetchall()
        return [
            EntityRelation(
                id=r["id"], subject=r["subject"], predicate=r["predicate"],
                object=r["object"], source_id=r["source_id"],
                confidence=r["confidence"],
                valid_from=r["valid_from"], valid_to=r["valid_to"],
            )
            for r in rows
        ]

    # ── Self-improvement & procedural memory ────────────────────────

    def retrieve_relevant_lessons(
        self, query: str, top_k: int = 3, min_score: float = 0.20,
    ) -> list[MemoryItem]:
        """Retrieve top-k procedural lessons semantically relevant to the task query."""
        mq = MemoryQuery(query=query, memory_type=MemoryType.PROCEDURAL, top_k=top_k)
        candidates = self._load_candidates(mq)
        if not candidates:
            return []

        query_vec = embed_single(query)
        now = datetime.now(UTC)

        scored: list[tuple[float, MemoryItem]] = []
        for item, embedding_blob in candidates:
            sim = self._compute_similarity(query, query_vec, item.text, embedding_blob)
            rec = self._recency_score(item.last_accessed, now)

            # Heuristic score balancing relevance, importance, and historical success
            score = (
                0.50 * sim
                + 0.25 * item.importance
                + 0.15 * item.success_score
                + 0.10 * rec
            )
            if score >= min_score:
                scored.append((score, item))

        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:top_k]

        with self._connect() as conn:
            for _, item in top:
                conn.execute(
                    "UPDATE memory_items SET last_accessed = ? WHERE id = ?",
                    (now.isoformat(), item.id),
                )

        return [item for _, item in top]

    def store_procedural_lesson(
        self,
        task_id: str,
        trigger: str,
        lesson_text: str,
        recommended_action: str,
        confidence: float,
        lesson_id: int | None = None,
    ) -> MemoryItem:
        """Store a procedural lesson, deduplicating if a highly similar lesson already exists."""
        text = f"Trigger: {trigger} | Lesson: {lesson_text} | Action: {recommended_action}"
        query_vec = embed_single(text)

        # Check existing procedural items for high similarity (> 0.85) to avoid clutter
        mq = MemoryQuery(query=text, memory_type=MemoryType.PROCEDURAL, top_k=5)
        candidates = self._load_candidates(mq)
        for item, blob in candidates:
            sim = self._compute_similarity(text, query_vec, item.text, blob)
            if sim >= 0.85:
                # Merge / reinforce existing memory
                item.importance = min(1.0, round(item.importance + 0.1, 2))
                item.success_score = min(1.0, round(item.success_score + 0.1, 2))
                item.text = text
                item.last_accessed = datetime.now(UTC)
                meta = json.loads(item.metadata_json or "{}")
                if lesson_id:
                    meta["lesson_id"] = lesson_id
                item.metadata_json = json.dumps(meta)
                return self.store_memory(item)

        meta_json = json.dumps({"lesson_id": lesson_id, "trigger": trigger, "action": recommended_action})
        item = MemoryItem(
            type=MemoryType.PROCEDURAL,
            text=text,
            importance=round(confidence, 2),
            source_task_id=str(task_id),
            success_score=0.5,
            metadata_json=meta_json,
        )
        return self.store_memory(item)

    def reward_memory(self, memory_id: str, delta: float = 0.15) -> None:
        """Increase memory success score after a task passes verification."""
        with self._connect() as conn:
            row = conn.execute("SELECT success_score FROM memory_items WHERE id = ?", (memory_id,)).fetchone()
            if row:
                new_score = min(1.0, round(row["success_score"] + delta, 3))
                conn.execute(
                    "UPDATE memory_items SET success_score = ?, last_accessed = ? WHERE id = ?",
                    (new_score, datetime.now(UTC).isoformat(), memory_id),
                )

    def penalize_memory(self, memory_id: str, delta: float = 0.25) -> None:
        """Decrease memory success score if task failed despite applying it."""
        with self._connect() as conn:
            row = conn.execute("SELECT success_score, importance FROM memory_items WHERE id = ?", (memory_id,)).fetchone()
            if row:
                new_score = max(0.0, round(row["success_score"] - delta, 3))
                new_imp = max(0.1, round(row["importance"] - 0.1, 3))
                conn.execute(
                    "UPDATE memory_items SET success_score = ?, importance = ?, last_accessed = ? WHERE id = ?",
                    (new_score, new_imp, datetime.now(UTC).isoformat(), memory_id),
                )

    def store_exemplar(
        self, task_id: str, objective: str, plan_summary: str, final_answer: str, tokens_used: int = 0,
    ) -> MemoryItem:
        """Crystallize a successful task execution as an episodic exemplar for few-shot learning."""
        text = f"OBJECTIVE:\n{objective}\n\nSUCCESSFUL PLAN:\n{plan_summary}\n\nVERIFIED SOLUTION:\n{final_answer[:1500]}"
        item = MemoryItem(
            type=MemoryType.EPISODIC,
            text=text,
            importance=0.85,
            source_task_id=str(task_id),
            success_score=1.0,
            metadata_json=json.dumps({"tokens_used": tokens_used, "objective": objective}),
        )
        return self.store_memory(item)

    def retrieve_exemplars(self, query: str, top_k: int = 1) -> list[MemoryItem]:
        """Retrieve highest-scoring previous successful exemplars for a similar objective."""
        mq = MemoryQuery(query=query, memory_type=MemoryType.EPISODIC, top_k=top_k)
        candidates = self._load_candidates(mq)
        if not candidates:
            return []

        query_vec = embed_single(query)
        now = datetime.now(UTC)
        scored: list[tuple[float, MemoryItem]] = []
        for item, blob in candidates:
            sim = self._compute_similarity(query, query_vec, item.text, blob)
            rec = self._recency_score(item.last_accessed, now)
            total = 0.60 * sim + 0.20 * item.success_score + 0.10 * item.importance + 0.10 * rec
            if sim >= 0.40:
                scored.append((total, item))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    # ── Pruning ──────────────────────────────────────────────────────

    def prune_expired(self) -> int:
        """Remove memory items whose valid_to has passed. Returns count."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM memory_items WHERE valid_to IS NOT NULL AND valid_to < ?",
                (now,),
            )
            return cursor.rowcount

    def prune_ineffective(self, threshold: float = 0.20) -> int:
        """Prune procedural memory items that have proven ineffective."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM memory_items WHERE type = 'procedural' AND success_score < ? AND importance < 0.3",
                (threshold,),
            )
            return cursor.rowcount

    # ── Private helpers ──────────────────────────────────────────────

    def _load_candidates(
        self, query: MemoryQuery,
    ) -> list[tuple[MemoryItem, bytes | None]]:
        """Load candidate rows from SQLite, applying filters."""
        clauses: list[str] = ["importance >= ?"]
        params: list[object] = [query.min_importance]
        if query.memory_type:
            clauses.append("type = ?")
            params.append(query.memory_type.value)

        where = " WHERE " + " AND ".join(clauses)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memory_items{where}", params
            ).fetchall()

        results: list[tuple[MemoryItem, bytes | None]] = []
        for row in rows:
            item = self._row_to_item(row)
            results.append((item, row["embedding"]))
        return results

    @staticmethod
    def _compute_similarity(
        query_text: str,
        query_vec: list[float] | None,
        item_text: str,
        embedding_blob: bytes | None,
    ) -> float:
        """Cosine similarity if embeddings available, else keyword overlap."""
        if query_vec and embedding_blob:
            try:
                item_vec = json.loads(embedding_blob)
                return max(cosine_similarity(query_vec, item_vec), 0.0)
            except (json.JSONDecodeError, TypeError):
                pass
        return keyword_similarity(query_text, item_text)

    @staticmethod
    def _recency_score(last_accessed: datetime, now: datetime) -> float:
        """Exponential decay based on hours since last access."""
        if last_accessed.tzinfo is None:
            from datetime import timezone
            last_accessed = last_accessed.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            from datetime import timezone
            now = now.replace(tzinfo=timezone.utc)
        hours = max((now - last_accessed).total_seconds() / 3600, 0)
        return math.exp(-0.693 * hours / _RECENCY_HALF_LIFE_HOURS)

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"], type=row["type"], text=row["text"],
            importance=row["importance"], source_task_id=row["source_task_id"],
            created_at=row["created_at"], last_accessed=row["last_accessed"],
            success_score=row["success_score"],
            valid_from=row["valid_from"], valid_to=row["valid_to"],
            metadata_json=row["metadata_json"] or "{}",
        )
