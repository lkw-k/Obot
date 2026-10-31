import pytest

from obot.core.embedder import DENSE_DIM, Embedding
from obot.core.retriever import retrieve, rrf
from obot.core.store import Hit, open_qdrant, write_points


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


def _hits(*ids):
    return [Hit(i, 0.0, {"id": i}) for i in ids]


def test_rrf_scores_sum_reciprocal_ranks():
    fused = rrf([_hits("a", "b"), _hits("b", "a")], top_k=2)

    assert {h.record_id for h in fused} == {"a", "b"}
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1].score == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_single_ranking_keeps_order():
    fused = rrf([_hits("a", "b", "c")], top_k=3)

    assert [h.record_id for h in fused] == ["a", "b", "c"]
    assert [h.score for h in fused] == pytest.approx([1 / 61, 1 / 62, 1 / 63])
