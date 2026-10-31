"""Search for the records relevant to a question (spec §6.2)."""

from qdrant_client import QdrantClient

from obot.core import store
from obot.core.embedder import Embedder

RRF_K = 60


def retrieve(
    question: str, embedder: Embedder, qdrant: QdrantClient, bot: str, top_k: int
) -> list[store.Hit]:
    """Dense search only; sparse search and RRF come with hybrid search."""
    (embedding,) = embedder.encode([question])
    return store.search_dense(qdrant, bot, embedding.dense, top_k)


def rrf(rankings: list[list[store.Hit]], top_k: int) -> list[store.Hit]:
    """Reciprocal rank fusion: score = Σ 1 / (RRF_K + rank), rank from 1."""
    scores: dict = {}
    hits: dict = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.record_id] = scores.get(hit.record_id, 0.0) + 1 / (RRF_K + rank)
            hits.setdefault(hit.record_id, hit)
    ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [store.Hit(i, scores[i], hits[i].record) for i in ordered]
