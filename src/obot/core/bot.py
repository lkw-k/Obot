"""Bot object: ask(question, history)."""

from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient

from obot.core.embedder import Embedder
from obot.core.generator import generate_answer
from obot.core.retriever import retrieve
from obot.core.store import Hit
from obot.llm.base import LLMClient


@dataclass
class Answer:
    answer: str
    route: str
    sources: list[Hit] = field(default_factory=list)
    sql: dict[str, Any] | None = None


class Bot:
    def __init__(
        self,
        name: str,
        qdrant: QdrantClient,
        embedder: Embedder,
        llm: LLMClient,
        top_k: int,
    ):
        self.name = name
        self.qdrant = qdrant
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k

    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int | None = None,
    ) -> Answer:
        """Search route only; routing and Text-to-SQL come later."""
        hits = retrieve(
            question, self.embedder, self.qdrant, self.name, top_k or self.top_k
        )
        answer = generate_answer(self.llm, question, hits, history or [])
        return Answer(answer=answer, route="search", sources=hits)
