"""Ollama provider."""

from typing import Any

import httpx
import ollama

from obot.config import LLMConfig
from obot.llm.base import JSON_INSTRUCTION, LLMError, strip_code_fence


class OllamaClient:
    def __init__(self, config: LLMConfig, client: Any = None):
        self.model = config.model
        self.temperature = config.temperature
        self.client = client or ollama.Client(host=config.base_url)

    def chat(self, system: str, messages: list[dict], json_mode: bool = False) -> str:
        if json_mode:
            system = f"{system}\n\n{JSON_INSTRUCTION}"
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, *messages],
                format="json" if json_mode else None,
                options={"temperature": self.temperature},
            )
        except (ollama.ResponseError, httpx.HTTPError, ConnectionError) as e:
            raise LLMError(f"ollama: {e}") from e
        text = response.message.content or ""
        return strip_code_fence(text) if json_mode else text
