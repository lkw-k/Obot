"""LLMClient interface, LLMError, provider factory."""

import re
from typing import Protocol

from obot.config import LLMConfig, Secrets

JSON_INSTRUCTION = "Reply with a single JSON object only, without any other text."


class LLMError(Exception):
    """The LLM provider could not be reached or returned an error."""


class LLMClient(Protocol):
    def chat(
        self, system: str, messages: list[dict], json_mode: bool = False
    ) -> str: ...


def strip_code_fence(text: str) -> str:
    """Remove a Markdown code fence wrapping the whole reply, if any."""
    match = re.fullmatch(r"\s*```[\w-]*\s*\n(.*?)\n?\s*```\s*", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def create_llm(config: LLMConfig, secrets: Secrets) -> LLMClient:
    # Imported here so only the selected provider's SDK is loaded.
    if config.provider == "ollama":
        from obot.llm.ollama import OllamaClient

        return OllamaClient(config)
    raise LLMError(f"llm.provider '{config.provider}' is not supported yet")
