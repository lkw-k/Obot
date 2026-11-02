import pytest

from obot.core.embedder import DENSE_DIM, Embedding
from obot.core.retriever import retrieve, rrf
from obot.core.store import Hit, open_qdrant, search_dense, write_points


def _vec(*hot):
    v = [0.0] * DENSE_DIM
    for i in hot:
        v[i] = 1.0
    return v


class QueryEmbedder:
    def __init__(self, vector, sparse=None):
        self.vector = vector
        self.sparse = sparse or {}

    def encode(self, texts):
        return [Embedding(dense=self.vector, sparse=self.sparse) for _ in texts]


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
    assert [h.score for h in hits] == pytest.approx([1 / 61, 1 / 62, 1 / 63])


def test_retrieve_fuses_dense_and_sparse_ranks(qdrant):
    # dense ranks a > b > c, sparse ranks b first: b wins on the summed score
    embedder = QueryEmbedder(_vec(0), sparse={2: 1.0})

    hits = retrieve("q", embedder, qdrant, "bot", top_k=3)

    assert hits[0].record_id == "b"
    assert hits[0].score == pytest.approx(1 / 62 + 1 / 61)


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


def test_rrf_overlap_beats_single_list_top():
    fused = rrf([_hits("a", "b", "c"), _hits("x", "b", "y")], top_k=5)

    assert fused[0].record_id == "b"
    assert len(fused) == 5


def test_rrf_disjoint_lists_interleave_and_truncate():
    fused = rrf([_hits("a", "b"), _hits("x", "y")], top_k=3)

    assert [h.score for h in fused] == pytest.approx([1 / 61, 1 / 61, 1 / 62])
    assert {h.record_id for h in fused[:2]} == {"a", "x"}


def test_rrf_keeps_record_from_first_occurrence():
    first = [Hit("a", 0.9, {"src": "dense"})]
    second = [Hit("a", 0.1, {"src": "sparse"})]

    (hit,) = rrf([first, second], top_k=1)

    assert hit.record == {"src": "dense"}


def test_exact_id_query_is_rescued_by_sparse_search(tmp_path):
    # "P-1005" is semantically far from the query but shares its exact token
    client = open_qdrant(tmp_path / "ids")
    ids = [f"P-100{i}" for i in range(6)]
    embeddings = [Embedding(dense=_vec(0, i + 1), sparse={i: 1.0}) for i in range(5)]
    embeddings.append(Embedding(dense=_vec(9), sparse={99: 1.0}))
    write_points(client, "bot", ids, [{"id": i} for i in ids], embeddings)
    embedder = QueryEmbedder(_vec(0), sparse={99: 1.0})

    dense_only = search_dense(client, "bot", _vec(0), limit=2)
    hits = retrieve("P-1005", embedder, client, "bot", top_k=2)
    client.close()

    assert "P-1005" not in [h.record_id for h in dense_only]
    assert hits[0].record_id == "P-1005"
