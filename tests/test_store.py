import sqlite3

import pytest

from obot.core.embedder import Embedding
from obot.core.schema import FieldInfo, Schema
from obot.core.store import (
    StoreLockedError,
    column_names,
    count_points,
    count_rows,
    open_qdrant,
    point_id,
    write_points,
    write_records,
)


@pytest.fixture
def qdrant(tmp_path):
    client = open_qdrant(tmp_path / "qdrant")
    yield client
    client.close()


def _emb(n):
    return [Embedding(dense=[1.0] * 1024, sparse={i + 1: 0.5}) for i in range(n)]


def test_column_names_normalize_and_dedupe():
    cols = column_names(
        ["price.amount", "price_amount", "주문일", "2024 sales", "ID", "id"]
    )
    assert cols == {
        "price.amount": "price_amount",
        "price_amount": "price_amount_2",
        "주문일": "주문일",
        "2024 sales": "_2024_sales",
        "ID": "ID",
        "id": "id_2",
    }


def test_point_id_is_stable_uuid5():
    assert point_id("products", 1) == point_id("products", 1)
    assert point_id("products", 1) != point_id("reviews", 1)


def test_write_points_stores_vectors_and_payload(qdrant):
    records = [{"id": "a", "name": "x"}, {"id": "b", "name": "y"}]
    write_points(qdrant, "bot", ["a", "b"], records, _emb(2))
    assert count_points(qdrant, "bot") == 2
    (point,) = qdrant.retrieve("bot", [point_id("bot", "a")], with_vectors=True)
    assert point.payload == {"record_id": "a", "record": {"id": "a", "name": "x"}}
    assert set(point.vector) == {"dense", "sparse"}
    assert point.vector["sparse"].indices == [1]


def test_write_points_recreates_collection(qdrant):
    write_points(qdrant, "bot", [1, 2, 3], [{}, {}, {}], _emb(3))
    write_points(qdrant, "bot", [1], [{}], _emb(1))
    assert count_points(qdrant, "bot") == 1


def test_second_client_on_same_folder_is_locked(tmp_path, qdrant):
    with pytest.raises(StoreLockedError, match="obot serve"):
        open_qdrant(tmp_path / "qdrant")


def _schema(fields, id_field, columns):
    return Schema(
        id_field=id_field,
        record_count=0,
        fields=[FieldInfo(name=n, role=r) for n, r in fields],
        columns=columns,
    )


def test_write_records_types_follow_roles(tmp_path):
    fields = [
        ("sku", "id"),
        ("qty", "number"),
        ("price.amount", "number"),
        ("note", "text"),
    ]
    schema = _schema(fields, "sku", column_names([n for n, _ in fields]))
    records = [
        {"sku": "A", "qty": 2, "price.amount": 1.5, "note": "hi"},
        {"sku": "B", "qty": None, "price.amount": 3, "note": None},
    ]
    db = tmp_path / "bot" / "records.db"
    write_records(db, schema, ["A", "B"], records)
    assert count_rows(db) == 2
    conn = sqlite3.connect(db)
    types = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(records)")}
    rows = conn.execute("SELECT * FROM records ORDER BY sku").fetchall()
    conn.close()
    assert types == {
        "sku": "TEXT",
        "qty": "INTEGER",
        "price_amount": "REAL",
        "note": "TEXT",
    }
    assert rows == [("A", 2, 1.5, "hi"), ("B", None, 3.0, None)]


def test_write_records_unconvertible_number_becomes_null(tmp_path):
    schema = _schema([("n", "number")], "_obot_id", {"_obot_id": "_obot_id", "n": "n"})
    db = tmp_path / "records.db"
    write_records(db, schema, [0, 1], [{"n": 1.5}, {"n": "oops"}])
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT _obot_id, n FROM records").fetchall()
    conn.close()
    assert rows == [(0, 1.5), (1, None)]


def test_write_records_recreates_table(tmp_path):
    schema = _schema([("a", "text")], "_obot_id", {"_obot_id": "_obot_id", "a": "a"})
    db = tmp_path / "records.db"
    write_records(db, schema, [0, 1], [{"a": "x"}, {"a": "y"}])
    write_records(db, schema, [0], [{"a": "z"}])
    assert count_rows(db) == 1
