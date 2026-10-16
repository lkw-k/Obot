"""Request/response models."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[Message] = []
    top_k: int | None = Field(default=None, ge=1)


class Source(BaseModel):
    id: Any
    score: float
    record: dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    route: str
    sources: list[Source]
    sql: dict[str, Any] | None


class BotSummary(BaseModel):
    name: str
    status: str


class BotDetail(BotSummary):
    record_count: int | None = None
    # Not named "schema": it would shadow BaseModel.schema.
    schema_: dict[str, Any] | None = Field(default=None, serialization_alias="schema")
    error: str | None = None
