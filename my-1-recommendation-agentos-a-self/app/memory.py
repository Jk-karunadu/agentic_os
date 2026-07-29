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
