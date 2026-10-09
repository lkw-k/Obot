from types import SimpleNamespace

import httpx
import ollama
import pytest

from obot.config import LLMConfig, Secrets
from obot.llm.base import JSON_INSTRUCTION, LLMError, create_llm, strip_code_fence
from obot.llm.ollama import OllamaClient


@pytest.mark.parametrize(
    "text, expected",
    [
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}```', '{"a": 1}'),
        ('  {"a": 1}\n', '{"a": 1}'),
        ("see ```x``` here", "see ```x``` here"),
    ],
)
def test_strip_code_fence(text, expected):
    assert strip_code_fence(text) == expected


class FakeOllamaSDK:
    def __init__(self, reply="hi", error=None):
        self.reply = reply
        self.error = error
        self.kwargs = None

    def chat(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(message=SimpleNamespace(content=self.reply))


def test_ollama_sends_system_and_messages():
    sdk = FakeOllamaSDK()
    client = OllamaClient(LLMConfig(model="m", temperature=0.5), client=sdk)

    reply = client.chat("sys", [{"role": "user", "content": "q"}])

    assert reply == "hi"
    assert sdk.kwargs["model"] == "m"
    assert sdk.kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]
    assert sdk.kwargs["format"] is None
    assert sdk.kwargs["options"] == {"temperature": 0.5}


def test_ollama_json_mode_instructs_and_strips_fence():
    sdk = FakeOllamaSDK(reply='```json\n{"route": "search"}\n```')
    client = OllamaClient(LLMConfig(), client=sdk)

    reply = client.chat("sys", [], json_mode=True)

    assert reply == '{"route": "search"}'
    assert sdk.kwargs["format"] == "json"
    assert JSON_INSTRUCTION in sdk.kwargs["messages"][0]["content"]


@pytest.mark.parametrize(
    "error",
    [
        ollama.ResponseError("model not found"),
        httpx.ConnectError("refused"),
        ConnectionError("Failed to connect to Ollama"),
    ],
)
def test_ollama_errors_become_llm_error(error):
    client = OllamaClient(LLMConfig(), client=FakeOllamaSDK(error=error))
    with pytest.raises(LLMError):
        client.chat("sys", [])


def test_create_llm_ollama():
    assert isinstance(create_llm(LLMConfig(), Secrets()), OllamaClient)
