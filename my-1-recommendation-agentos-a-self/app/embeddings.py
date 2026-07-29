"""Local embedding support via Ollama's /api/embed endpoint.

Primary model: nomic-embed-text (pulled separately in Ollama).
Fallback: keyword-overlap scoring when the embedding model is unavailable.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import httpx

from app.config import settings

_EMBED_MODEL = "nomic-embed-text"


def embed(texts: list[str]) -> list[list[float]] | None:
    """Return embeddings for a list of texts, or ``None`` on failure."""
    if not texts:
        return []
    try:
        response = httpx.post(
            f"{settings.ollama_base_url}/api/embed",
            json={"model": _EMBED_MODEL, "input": texts},
            timeout=30.0,
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings")
        if embeddings and len(embeddings) == len(texts):
            return embeddings
        return None
    except (httpx.HTTPError, KeyError, TypeError):
        return None


def embed_single(text: str) -> list[float] | None:
    """Convenience wrapper for a single text."""
    result = embed([text])
    return result[0] if result else None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Keyword fallback ────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenise(text: str) -> Counter[str]:
    return Counter(_WORD_RE.findall(text.lower()))


def keyword_similarity(query: str, text: str) -> float:
    """Simple keyword-overlap similarity in [0, 1].

    Used as a fallback when the embedding model is unavailable.
    """
    q_tokens = _tokenise(query)
    t_tokens = _tokenise(text)
    if not q_tokens or not t_tokens:
        return 0.0
    overlap = sum((q_tokens & t_tokens).values())
    total = sum(q_tokens.values())
    return min(overlap / total, 1.0) if total else 0.0
