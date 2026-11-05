"""Search / aggregate classification (spec §6.1)."""

import json
from typing import Literal

from obot.core.schema import Schema
from obot.llm.base import LLMClient

Route = Literal["search", "aggregate"]

SYSTEM_PROMPT = (
    "You classify questions about a dataset.\n"
    "- aggregate: counts, sums, averages, max/min, sorting, ranking, "
    "or full lists of records matching conditions.\n"
    "- search: anything else, such as finding or describing specific records.\n"
    'Reply with JSON only: {"route": "search"} or {"route": "aggregate"}.'
)


def route(llm: LLMClient, question: str, schema: Schema) -> Route:
    """Ask the LLM for the route; anything unreadable falls back to search."""
    fields = "\n".join(f"- {f.name} ({f.role})" for f in schema.fields)
    prompt = f"Fields:\n{fields}\n\nQuestion: {question}"
    reply = llm.chat(
        SYSTEM_PROMPT, [{"role": "user", "content": prompt}], json_mode=True
    )
    try:
        value = json.loads(reply)["route"]
    except (ValueError, TypeError, KeyError):
        return "search"
    return "aggregate" if value == "aggregate" else "search"
