"""Text-to-SQL with safe execution (spec §6.3)."""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from obot.core.schema import Schema
from obot.core.store import TABLE
from obot.llm.base import LLMClient

MAX_ROWS = 100
SAMPLE_ROWS = 3
_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}

SYSTEM_PROMPT = (
    "You write one SQLite SELECT query that answers a question about a table.\n"
    "- Use only the given table and columns; quote column names with double quotes.\n"
    "- Read only: SELECT or WITH, a single statement, no recursive CTEs.\n"
    'Reply with JSON only: {"sql": "SELECT ..."}.'
)


class SQLError(Exception):
    """The query was rejected or failed to run."""


def run_sql(db_path: str | Path, sql: str) -> list[dict[str, Any]]:
    """Run a read-only query, at most MAX_ROWS rows. Raises SQLError."""
    sql = sql.strip().removesuffix(";").strip()
    if not re.match(r"(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise SQLError("the query must start with SELECT or WITH")
    conn = _connect_ro(db_path)
    try:
        conn.set_authorizer(_authorize)
        conn.row_factory = sqlite3.Row
        # Own lines so a trailing comment cannot swallow ") LIMIT"; fetchmany
        # because an unclosed /* still can.
        cursor = conn.execute(f"SELECT * FROM (\n{sql}\n) LIMIT {MAX_ROWS}")
        rows = cursor.fetchmany(MAX_ROWS)
    except sqlite3.Error as e:
        raise SQLError(str(e)) from e
    finally:
        conn.close()
    return [dict(row) for row in rows]


def query(
    llm: LLMClient, question: str, schema: Schema, db_path: str | Path
) -> tuple[str, list[dict[str, Any]]] | None:
    """Generate and run SQL, retrying once with the error; None if both fail."""
    try:
        prompt = _prompt(question, schema, db_path)
    except SQLError:
        return None
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(2):
        reply = llm.chat(SYSTEM_PROMPT, messages, json_mode=True)
        try:
            sql = json.loads(reply)["sql"]
            if not isinstance(sql, str):
                raise SQLError('"sql" must be a string')
            return sql, run_sql(db_path, sql)
        except (ValueError, TypeError, KeyError, SQLError) as e:
            error = f"{type(e).__name__}: {e}"
        if attempt == 0:
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"That failed with {error}. Fix it."},
            ]
    return None


def _prompt(question: str, schema: Schema, db_path: str | Path) -> str:
    roles = {f.name: f.role for f in schema.fields}
    conn = _connect_ro(db_path)
    try:
        types = {row[1]: row[2] for row in conn.execute(f"PRAGMA table_info({TABLE})")}
    finally:
        conn.close()
    # _obot_id has a column but no entry in schema.fields.
    columns = "\n".join(
        f'- "{column}" {types.get(column, "")} ({roles.get(field, "id")})'
        for field, column in schema.columns.items()
    )
    samples = json.dumps(
        run_sql(db_path, f"SELECT * FROM {TABLE} LIMIT {SAMPLE_ROWS}"),
        ensure_ascii=False,
    )
    return (
        f"Table: {TABLE}\nColumns:\n{columns}\n"
        f"Sample rows: {samples}\n\nQuestion: {question}"
    )


def _connect_ro(db_path: str | Path) -> sqlite3.Connection:
    try:
        return sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as e:
        raise SQLError(str(e)) from e


def _authorize(action: int, *_: Any) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY
