from types import SimpleNamespace

import anthropic
import httpx
import ollama
import openai
import pytest

from obot.config import LLMConfig, Secrets
from obot.llm.anthropic import AnthropicClient
from obot.llm.base import JSON_INSTRUCTION, LLMError, create_llm, strip_code_fence
from obot.llm.ollama import OllamaClient
from obot.llm.openai import OpenAIClient

REQUEST = httpx.Request("POST", "http://test")


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


class FakeCreate:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def anthropic_sdk(text="hi", error=None):
    block = SimpleNamespace(type="text", text=text)
    messages = FakeCreate(SimpleNamespace(content=[block]), error)
    return SimpleNamespace(messages=messages)


def openai_sdk(text="hi", error=None):
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    completions = FakeCreate(SimpleNamespace(choices=[choice]), error)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_anthropic_passes_system_separately():
    sdk = anthropic_sdk()
    client = AnthropicClient(LLMConfig(model="m"), "key", client=sdk)

    reply = client.chat("sys", [{"role": "user", "content": "q"}])

    assert reply == "hi"
    kwargs = sdk.messages.kwargs
    assert kwargs["system"] == "sys"
    assert kwargs["messages"] == [{"role": "user", "content": "q"}]
    assert kwargs["model"] == "m"


def test_anthropic_json_mode_instructs_and_strips_fence():
    sdk = anthropic_sdk('```json\n{"a": 1}\n```')
    client = AnthropicClient(LLMConfig(), "key", client=sdk)

    assert client.chat("sys", [], json_mode=True) == '{"a": 1}'
    assert JSON_INSTRUCTION in sdk.messages.kwargs["system"]


def test_anthropic_errors_become_llm_error():
    error = anthropic.APIConnectionError(request=REQUEST)
    client = AnthropicClient(LLMConfig(), "key", client=anthropic_sdk(error=error))
    with pytest.raises(LLMError):
        client.chat("sys", [])


def test_openai_sends_system_message():
    sdk = openai_sdk()
    client = OpenAIClient(LLMConfig(model="m", temperature=0.3), "key", client=sdk)

    reply = client.chat("sys", [{"role": "user", "content": "q"}])

    assert reply == "hi"
    kwargs = sdk.chat.completions.kwargs
    assert kwargs["messages"][0] == {"role": "system", "content": "sys"}
    assert kwargs["temperature"] == 0.3
    assert "response_format" not in kwargs


def test_openai_json_mode_uses_json_object_and_strips_fence():
    sdk = openai_sdk('```\n{"a": 1}\n```')
    client = OpenAIClient(LLMConfig(), "key", client=sdk)

    assert client.chat("sys", [], json_mode=True) == '{"a": 1}'
    assert sdk.chat.completions.kwargs["response_format"] == {"type": "json_object"}


def test_openai_errors_become_llm_error():
    error = openai.APIConnectionError(request=REQUEST)
    client = OpenAIClient(LLMConfig(), "key", client=openai_sdk(error=error))
    with pytest.raises(LLMError):
        client.chat("sys", [])


@pytest.mark.parametrize(
    "provider, cls", [("anthropic", AnthropicClient), ("openai", OpenAIClient)]
)
def test_create_llm_hosted_providers(provider, cls):
    secrets = Secrets(anthropic_api_key="a", openai_api_key="o")
    assert isinstance(create_llm(LLMConfig(provider=provider), secrets), cls)
