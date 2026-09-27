"""LLMClient interface, LLMError, provider factory."""

from typing import Protocol


class LLMClient(Protocol):
    def chat(
        self, system: str, messages: list[dict], json_mode: bool = False
    ) -> str: ...
