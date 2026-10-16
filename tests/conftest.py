import hashlib
import threading

import pytest

from obot.core.embedder import DENSE_DIM, Embedding


class FakeEmbedder:
    """Deterministic vectors from text hashes; no model download."""

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        result = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            dense = [digest[i % len(digest)] / 255 + 0.01 for i in range(DENSE_DIM)]
            sparse = {
                int.from_bytes(hashlib.sha256(w.encode()).digest()[:2]): 1.0
                for w in text.split()
            }
            result.append(Embedding(dense=dense, sparse=sparse))
        return result


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


class BlockingEmbedder:
    """Holds builds until released, so tests can observe the building status."""

    def __init__(self, inner):
        self.inner = inner
        self.release = threading.Event()

    def encode(self, texts):
        self.release.wait(timeout=10)
        return self.inner.encode(texts)


@pytest.fixture
def blocking_embedder(fake_embedder):
    embedder = BlockingEmbedder(fake_embedder)
    yield embedder
    embedder.release.set()


class FakeLLM:
    """Returns a fixed reply and records every call."""

    def __init__(self, reply="fake answer"):
        self.reply = reply
        self.calls = []

    def chat(self, system, messages, json_mode=False):
        self.calls.append(
            {"system": system, "messages": messages, "json_mode": json_mode}
        )
        return self.reply


@pytest.fixture
def fake_llm():
    return FakeLLM()
