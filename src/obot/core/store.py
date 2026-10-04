"""Qdrant + SQLite read/write (spec §5.6)."""

import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from obot.core.embedder import DENSE_DIM, Embedding
from obot.core.schema import FALLBACK_ID_FIELD, Schema

TABLE = "records"
UPSERT_BATCH = 256


class StoreLockedError(Exception):
    """The Qdrant folder is already open in another process (e.g. the server)."""


def open_qdrant(path: str | Path) -> QdrantClient:
    try:
        return QdrantClient(path=str(path))
    except RuntimeError as e:
        raise StoreLockedError(
            f"{path} is in use by another process. Stop `obot serve`, or use "
            "POST /bots/{name}/rebuild and POST /bots/{name}/chat instead."
        ) from e


def point_id(bot: str, record_id: Any) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{bot}:{record_id}"))


def write_points(
    client: QdrantClient,
    bot: str,
    ids: list[Any],
    records: list[dict[str, Any]],
    embeddings: list[Embedding],
) -> None:
    """Empty (or create) the bot's collection, then insert one point per record."""
    if client.collection_exists(bot):
        # Not delete_collection: in local mode on Windows the deleted collection's
        # files stay open, the folder is not removed, and a recreated collection
        # reloads the old points.
        client.delete(
            bot, points_selector=models.FilterSelector(filter=models.Filter())
        )
    else:
        client.create_collection(
            bot,
            vectors_config={
                "dense": models.VectorParams(
                    size=DENSE_DIM, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
    points = [
        models.PointStruct(
            id=point_id(bot, record_id),
            vector={
                "dense": emb.dense,
                "sparse": models.SparseVector(
                    indices=list(emb.sparse), values=list(emb.sparse.values())
                ),
            },
            payload={"record_id": record_id, "record": record},
        )
        for record_id, record, emb in zip(ids, records, embeddings, strict=True)
    ]
    for start in range(0, len(points), UPSERT_BATCH):
        client.upsert(bot, points[start : start + UPSERT_BATCH])


def count_points(client: QdrantClient, bot: str) -> int:
    return client.count(bot, exact=True).count


def column_names(fields: list[str]) -> dict[str, str]:
    """Map field names to unique SQLite column names (\\w kept, others → _)."""
    columns: dict[str, str] = {}
    used: set[str] = set()
    for field in fields:
        base = re.sub(r"\W", "_", field) or "_"
        if base[0].isdigit():
            base = "_" + base
        name, n = base, 1
        while name.lower() in used:  # SQLite column names are case-insensitive
            n += 1
            name = f"{base}_{n}"
        used.add(name.lower())
        columns[field] = name
    return columns


def write_records(
    db_path: str | Path,
    schema: Schema,
    ids: list[Any],
    records: list[dict[str, Any]],
) -> None:
    """Drop and recreate the records table from the schema's column mapping."""
    roles = {f.name: f.role for f in schema.fields}
    fields = list(schema.columns)
    types = {f: _column_type(roles.get(f), f, records) for f in fields}

    def row(record_id: Any, record: dict[str, Any]) -> list[Any]:
        return [
            record_id if f == FALLBACK_ID_FIELD else _convert(record.get(f), types[f])
            for f in fields
        ]

    columns = ", ".join(f'"{schema.columns[f]}" {types[f]}' for f in fields)
    placeholders = ", ".join("?" for _ in fields)
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
        conn.execute(f"CREATE TABLE {TABLE} ({columns})")
        conn.executemany(
            f"INSERT INTO {TABLE} VALUES ({placeholders})",
            [row(i, r) for i, r in zip(ids, records, strict=True)],
        )
    conn.close()


def count_rows(db_path: str | Path) -> int:
    with sqlite3.connect(db_path) as conn:
        (n,) = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()
    conn.close()
    return n


def _column_type(role: str | None, field: str, records: list[dict[str, Any]]) -> str:
    if field == FALLBACK_ID_FIELD:
        return "INTEGER"
    if role != "number":
        return "TEXT"
    values = [r.get(field) for r in records if r.get(field) is not None]
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return "INTEGER"
    return "REAL"


def _convert(value: Any, column_type: str) -> Any:
    if value is None:
        return None
    if column_type == "TEXT":
        return value if isinstance(value, str) else str(value)
    if isinstance(value, bool):
        return None
    try:
        return int(value) if column_type == "INTEGER" else float(value)
    except (TypeError, ValueError):
        return None
