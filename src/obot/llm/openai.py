"""OpenAI provider."""

from typing import Any

import openai

from obot.config import LLMConfig
from obot.llm.base import JSON_INSTRUCTION, LLMError, strip_code_fence


class OpenAIClient:
    def __init__(self, config: LLMConfig, api_key: str, client: Any = None):
        self.model = config.model
        self.temperature = config.temperature
        self.client = client or openai.OpenAI(api_key=api_key)

    def chat(self, system: str, messages: list[dict], json_mode: bool = False) -> str:
        extra: dict[str, Any] = {}
        if json_mode:
            system = f"{system}\n\n{JSON_INSTRUCTION}"
            extra["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, *messages],
                temperature=self.temperature,
                **extra,
            )
        except openai.APIError as e:
            raise LLMError(f"openai: {e}") from e
        text = response.choices[0].message.content or ""
        return strip_code_fence(text) if json_mode else text
