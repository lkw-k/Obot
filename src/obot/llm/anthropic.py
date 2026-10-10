"""Anthropic provider."""

from typing import Any

import anthropic

from obot.config import LLMConfig
from obot.llm.base import JSON_INSTRUCTION, LLMError, strip_code_fence

MAX_TOKENS = 2048


class AnthropicClient:
    # No temperature: the current Messages API no longer accepts it.
    def __init__(self, config: LLMConfig, api_key: str, client: Any = None):
        self.model = config.model
        self.client = client or anthropic.Anthropic(api_key=api_key)

    def chat(self, system: str, messages: list[dict], json_mode: bool = False) -> str:
        if json_mode:
            system = f"{system}\n\n{JSON_INSTRUCTION}"
        try:
            response = self.client.messages.create(
                model=self.model,
                system=system,
                messages=messages,
                max_tokens=MAX_TOKENS,
            )
        except anthropic.APIError as e:
            raise LLMError(f"anthropic: {e}") from e
        text = "".join(b.text for b in response.content if b.type == "text")
        return strip_code_fence(text) if json_mode else text
