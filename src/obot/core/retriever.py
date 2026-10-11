"""Search for the records relevant to a question (spec §6.2)."""

from qdrant_client import QdrantClient

from obot.core import store
from obot.core.embedder import Embedder


def retrieve(
    question: str, embedder: Embedder, qdrant: QdrantClient, bot: str, top_k: int
) -> list[store.Hit]:
    """Dense search only; sparse search and RRF come with hybrid search."""
    (embedding,) = embedder.encode([question])
    return store.search_dense(qdrant, bot, embedding.dense, top_k)
