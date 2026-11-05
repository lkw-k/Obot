"""Bot object: ask(question, history)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from obot.core import sql_agent
from obot.core.embedder import Embedder
from obot.core.generator import generate_answer, generate_sql_answer
from obot.core.retriever import retrieve
from obot.core.router import route
from obot.core.schema import load_schema
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
        bot_dir: Path,
        routing: bool,
    ):
        self.name = name
        self.qdrant = qdrant
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k
        self.bot_dir = bot_dir
        self.routing = routing

    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int | None = None,
    ) -> Answer:
        """Route to Text-to-SQL or hybrid search; failed SQL falls back to search."""
        history = history or []
        if self.routing:
            schema = load_schema(self.bot_dir / "schema.json")
            if route(self.llm, question, schema) == "aggregate":
                result = sql_agent.query(
                    self.llm, question, schema, self.bot_dir / "records.db"
                )
                if result is not None:
                    sql, rows = result
                    answer = generate_sql_answer(self.llm, question, sql, rows, history)
                    return Answer(
                        answer=answer,
                        route="aggregate",
                        sql={"query": sql, "rows": rows},
                    )
        if top_k is None:
            top_k = self.top_k
        hits = retrieve(question, self.embedder, self.qdrant, self.name, top_k)
        answer = generate_answer(self.llm, question, hits, history)
        return Answer(answer=answer, route="search", sources=hits)
