import pytest

from obot.core.bot import Bot
from obot.core.store import open_qdrant, write_points


@pytest.fixture
def bot(tmp_path, fake_embedder, fake_llm):
    qdrant = open_qdrant(tmp_path / "qdrant")
    records = [{"id": f"P-{i}", "name": f"item {i}"} for i in range(4)]
    texts = [f"id: {r['id']}\nname: {r['name']}" for r in records]
    ids = [r["id"] for r in records]
    write_points(qdrant, "shop", ids, records, fake_embedder.encode(texts))
    yield Bot("shop", qdrant, fake_embedder, fake_llm, top_k=3)
    qdrant.close()


def test_ask_returns_search_answer_with_sources(bot, fake_llm):
    result = bot.ask("which item?")

    assert result.answer == "fake answer"
    assert result.route == "search"
    assert result.sql is None
    assert len(result.sources) == 3
    prompt = fake_llm.calls[0]["messages"][-1]["content"]
    for hit in result.sources:
        assert f"[id: {hit.record_id}]" in prompt


def test_ask_passes_history_and_top_k(bot, fake_llm):
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    result = bot.ask("which item?", history=history, top_k=1)

    assert len(result.sources) == 1
    assert fake_llm.calls[0]["messages"][:-1] == history
