import json

import pytest

from obot.core.bot import Bot
from obot.core.schema import analyze_schema, save_schema
from obot.core.store import column_names, open_qdrant, write_points, write_records

AGGREGATE = json.dumps({"route": "aggregate"})
COUNT_SQL = "SELECT COUNT(*) AS n FROM records"


@pytest.fixture
def make_bot(tmp_path, fake_embedder, fake_llm):
    qdrant = open_qdrant(tmp_path / "qdrant")
    records = [{"id": f"P-{i}", "name": f"item {i}"} for i in range(4)]
    texts = [f"id: {r['id']}\nname: {r['name']}" for r in records]
    ids = [r["id"] for r in records]
    write_points(qdrant, "shop", ids, records, fake_embedder.encode(texts))
    schema = analyze_schema(records)
    schema.columns = column_names([f.name for f in schema.fields])
    write_records(tmp_path / "shop" / "records.db", schema, ids, records)
    save_schema(schema, tmp_path / "shop" / "schema.json")

    def make(routing=True):
        return Bot(
            "shop", qdrant, fake_embedder, fake_llm, 3, tmp_path / "shop", routing
        )

    yield make
    qdrant.close()


@pytest.fixture
def bot(make_bot):
    return make_bot()


def test_ask_returns_search_answer_with_sources(bot, fake_llm):
    result = bot.ask("which item?")

    assert result.answer == "fake answer"
    assert result.route == "search"
    assert result.sql is None
    assert len(result.sources) == 3
    prompt = fake_llm.calls[-1]["messages"][-1]["content"]
    for hit in result.sources:
        assert f"[id: {hit.record_id}]" in prompt


def test_ask_passes_history_and_top_k(bot, fake_llm):
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    result = bot.ask("which item?", history=history, top_k=1)

    assert len(result.sources) == 1
    assert fake_llm.calls[-1]["messages"][:-1] == history


def test_ask_ranks_exact_id_match_first(bot):
    result = bot.ask("P-2")

    assert result.sources[0].record_id == "P-2"
    assert result.sources[0].score > 1 / 61  # dense + sparse rank contributions


def test_aggregate_question_is_answered_with_sql(bot, fake_llm):
    fake_llm.reply = [AGGREGATE, json.dumps({"sql": COUNT_SQL}), "four items"]

    result = bot.ask("how many items?")

    assert result.answer == "four items"
    assert result.route == "aggregate"
    assert result.sources == []
    assert result.sql == {"query": COUNT_SQL, "rows": [{"n": 4}]}
    prompt = fake_llm.calls[-1]["messages"][-1]["content"]
    assert f"SQL: {COUNT_SQL}" in prompt
    assert 'Rows: [{"n": 4}]' in prompt


def test_failed_sql_falls_back_to_search(bot, fake_llm):
    bad = json.dumps({"sql": "DELETE FROM records"})
    fake_llm.reply = [AGGREGATE, bad, bad, "searched"]

    result = bot.ask("how many items?")

    assert result.answer == "searched"
    assert result.route == "search"
    assert result.sql is None
    assert len(result.sources) == 3


def test_routing_disabled_skips_the_router(make_bot, fake_llm):
    result = make_bot(routing=False).ask("how many items?")

    assert result.route == "search"
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["json_mode"] is False
