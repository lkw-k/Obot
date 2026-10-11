import pytest

from obot.core.embedder import DENSE_DIM, Embedding
from obot.core.retriever import retrieve
from obot.core.store import open_qdrant, write_points


def _vec(*hot):
    v = [0.0] * DENSE_DIM
    for i in hot:
        v[i] = 1.0
    return v


class QueryEmbedder:
    def __init__(self, vector):
        self.vector = vector

    def encode(self, texts):
        return [Embedding(dense=self.vector, sparse={}) for _ in texts]


@pytest.fixture
def qdrant(tmp_path):
    client = open_qdrant(tmp_path / "qdrant")
    records = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    embeddings = [
        Embedding(dense=_vec(0), sparse={1: 1.0}),
        Embedding(dense=_vec(0, 1), sparse={2: 1.0}),
        Embedding(dense=_vec(2), sparse={3: 1.0}),
    ]
    write_points(client, "bot", ["a", "b", "c"], records, embeddings)
    yield client
    client.close()


def test_retrieve_orders_by_dense_similarity(qdrant):
    hits = retrieve("q", QueryEmbedder(_vec(0)), qdrant, "bot", top_k=3)

    assert [h.record_id for h in hits] == ["a", "b", "c"]
    assert hits[0].record == {"id": "a"}
    assert hits[0].score > hits[1].score > hits[2].score


def test_retrieve_returns_top_k(qdrant):
    hits = retrieve("q", QueryEmbedder(_vec(2)), qdrant, "bot", top_k=1)

    assert [h.record_id for h in hits] == ["c"]
