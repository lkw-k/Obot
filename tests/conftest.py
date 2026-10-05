import hashlib

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
