import logging

from obot.core import embedder as embedder_module
from obot.core.embedder import MAX_LENGTH, BGEM3Embedder


class FakeTokenizer:
    def __call__(self, texts, add_special_tokens=True):
        return {"input_ids": [[0] * len(t) for t in texts]}


class FakeM3Model:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.batches = []

    def encode(self, texts, **kwargs):
        self.batches.append((list(texts), kwargs))
        return {
            "dense_vecs": [[float(len(t))] * 3 for t in texts],
            "lexical_weights": [{"17": 0.5, "4": 0.25} for _ in texts],
        }


def test_converts_sparse_keys_to_int_and_keeps_order():
    model = FakeM3Model()
    out = BGEM3Embedder(model).encode(["a", "bbb"])
    assert [e.dense for e in out] == [[1.0] * 3, [3.0] * 3]
    assert out[0].sparse == {17: 0.5, 4: 0.25}
    assert all(isinstance(k, int) for k in out[0].sparse)
    _, kwargs = model.batches[0]
    assert kwargs["return_dense"] and kwargs["return_sparse"]
    assert kwargs["max_length"] == MAX_LENGTH


def test_encodes_in_batches(monkeypatch):
    monkeypatch.setattr(embedder_module, "BATCH_SIZE", 2)
    model = FakeM3Model()
    out = BGEM3Embedder(model).encode(["a", "b", "c", "d", "e"])
    assert len(out) == 5
    assert [len(texts) for texts, _ in model.batches] == [2, 2, 1]


def test_warns_on_truncated_records(caplog):
    model = FakeM3Model()
    with caplog.at_level(logging.WARNING):
        BGEM3Embedder(model).encode(["short", "x" * (MAX_LENGTH + 1)])
    assert "record 1" in caplog.text
    assert "record 0" not in caplog.text
