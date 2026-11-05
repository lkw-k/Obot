import json
import sqlite3

import pytest

from obot.core.schema import FieldInfo, Schema
from obot.core.sql_agent import MAX_ROWS, SQLError, _connect_ro, query, run_sql
from obot.core.store import write_records

SCHEMA = Schema(
    id_field="id",
    record_count=150,
    fields=[
        FieldInfo(name="id", role="id"),
        FieldInfo(name="price", role="number"),
        FieldInfo(name="created_at", role="date"),
    ],
    columns={"id": "id", "price": "price", "created_at": "created_at"},
)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "records.db"
    records = [
        {"id": f"P-{i}", "price": i * 10, "created_at": f"2024-01-{i % 28 + 1:02d}"}
        for i in range(150)
    ]
    write_records(path, SCHEMA, [r["id"] for r in records], records)
    return path


def _sql(sql):
    return json.dumps({"sql": sql})


def test_select_returns_rows_as_dicts(db):
    rows = run_sql(db, "SELECT id, price FROM records WHERE price = 30")

    assert rows == [{"id": "P-3", "price": 30}]


def test_created_at_column_is_queried_normally(db):
    rows = run_sql(db, "SELECT MAX(created_at) AS latest FROM records")

    assert rows == [{"latest": "2024-01-28"}]


def test_trailing_semicolon_is_stripped(db):
    assert run_sql(db, "SELECT COUNT(*) AS n FROM records;  ") == [{"n": 150}]


def test_with_query_is_allowed(db):
    sql = "WITH c AS (SELECT price FROM records) SELECT SUM(price) AS s FROM c"

    assert run_sql(db, sql) == [{"s": sum(i * 10 for i in range(150))}]


def test_results_are_limited(db):
    assert len(run_sql(db, "SELECT * FROM records")) == MAX_ROWS


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO records (id) VALUES ('x')",
        "UPDATE records SET price = 0",
        "DELETE FROM records",
        "DROP TABLE records",
        "CREATE TABLE t (a)",
        "PRAGMA table_info(records)",
        "ATTACH DATABASE ':memory:' AS x",
    ],
)
def test_non_select_statements_are_rejected(db, sql):
    with pytest.raises(SQLError):
        run_sql(db, sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; DROP TABLE records",
        "SELECT 1; DELETE FROM records;",
        "WITH x AS (DELETE FROM records RETURNING id) SELECT * FROM x",
    ],
)
def test_writes_and_extra_statements_inside_select_are_rejected(db, sql):
    with pytest.raises(SQLError):
        run_sql(db, sql)
    assert run_sql(db, "SELECT COUNT(*) AS n FROM records") == [{"n": 150}]


def test_connection_is_read_only(db):
    # Second line of defense, independent of the authorizer.
    conn = _connect_ro(db)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DELETE FROM records")
    conn.close()


def test_query_returns_sql_and_rows(db, fake_llm):
    fake_llm.reply = _sql("SELECT COUNT(*) AS n FROM records")

    assert query(fake_llm, "how many?", SCHEMA, db) == (
        "SELECT COUNT(*) AS n FROM records",
        [{"n": 150}],
    )
    (call,) = fake_llm.calls
    assert call["json_mode"] is True
    prompt = call["messages"][0]["content"]
    assert "Table: records" in prompt
    assert '- "price" INTEGER (number)' in prompt
    assert '- "created_at" TEXT (date)' in prompt
    assert '"id": "P-2"' in prompt and '"id": "P-3"' not in prompt  # 3 samples
    assert prompt.endswith("Question: how many?")


def test_query_retries_once_with_the_error(db, fake_llm):
    fake_llm.reply = [_sql("SELECT nope FROM records"), _sql("SELECT 1 AS one")]

    assert query(fake_llm, "q", SCHEMA, db) == ("SELECT 1 AS one", [{"one": 1}])
    retry = fake_llm.calls[1]["messages"]
    assert retry[1] == {
        "role": "assistant",
        "content": _sql("SELECT nope FROM records"),
    }
    assert "no such column: nope" in retry[2]["content"]


@pytest.mark.parametrize("first", ["not json", '{"sql": 1}', '{"query": "x"}'])
def test_unreadable_reply_is_retried(db, fake_llm, first):
    fake_llm.reply = [first, _sql("SELECT 1 AS one")]

    assert query(fake_llm, "q", SCHEMA, db) == ("SELECT 1 AS one", [{"one": 1}])


def test_query_gives_up_after_the_retry(db, fake_llm):
    fake_llm.reply = [_sql("DELETE FROM records"), _sql("DROP TABLE records")]

    assert query(fake_llm, "q", SCHEMA, db) is None
    assert len(fake_llm.calls) == 2
