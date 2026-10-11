"""Prompt building and answer generation (spec §6.4)."""

from obot.core.chunker import chunk_text
from obot.core.store import Hit
from obot.llm.base import LLMClient

HISTORY_LIMIT = 10

SYSTEM_PROMPT = (
    "You answer questions about a dataset using the records in the context.\n"
    "- Answer only from the provided context.\n"
    "- If the context does not contain the answer, say that you don't know.\n"
    "- Answer in the same language as the question.\n"
    "- When you refer to a record, include its id."
)


def generate_answer(
    llm: LLMClient, question: str, hits: list[Hit], history: list[dict]
) -> str:
    context = "\n\n".join(f"[id: {h.record_id}]\n{chunk_text(h.record)}" for h in hits)
    prompt = f"Context:\n{context or '(no records found)'}\n\nQuestion: {question}"
    messages = [*history[-HISTORY_LIMIT:], {"role": "user", "content": prompt}]
    return llm.chat(SYSTEM_PROMPT, messages)
