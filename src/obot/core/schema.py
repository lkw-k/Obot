"""Field role analysis (spec §5.3)."""

import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel

from obot.llm.base import LLMClient

logger = logging.getLogger(__name__)

Role = Literal["id", "number", "date", "category", "text"]
ROLES: tuple[str, ...] = get_args(Role)

SCHEMA_VERSION = 1
FALLBACK_ID_FIELD = "_obot_id"
DATE_SAMPLE_SIZE = 100
DATE_MIN_RATIO = 0.9
CATEGORY_MIN_UNIQUE = 20
CATEGORY_UNIQUE_RATIO = 0.05
CATEGORY_MAX_AVG_LEN = 30
LLM_SAMPLE_RECORDS = 3
ID_TOKENS = frozenset({"id", "uuid", "guid"})


class FieldInfo(BaseModel):
    name: str
    role: Role


class Schema(BaseModel):
    obot_schema_version: int = SCHEMA_VERSION
    id_field: str
    record_count: int
    fields: list[FieldInfo]


def analyze_schema(records: list[dict[str, Any]]) -> Schema:
    """Assign one role per field; the first matching rule wins."""
    columns = _columns(records)
    id_field = _pick_id_field(columns)
    roles: dict[str, Role] = {
        name: "id" if name == id_field else _classify(values)
        for name, values in columns.items()
    }
    return _build(roles, len(records))


def save_schema(schema: Schema, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(schema.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_schema(path: str | Path) -> Schema:
    return Schema.model_validate_json(Path(path).read_text(encoding="utf-8"))


_REFINE_SYSTEM = (
    "You classify the fields of a dataset. Roles: "
    "id (unique, non-null identifier), number (numeric), date (ISO date), "
    "category (short string with few distinct values), text (free text). "
    'Reply with JSON only: {"roles": {"<field name>": "<role>"}}. '
    "Include only fields whose role should change. At most one id field."
)


def refine_with_llm(
    schema: Schema, records: list[dict[str, Any]], llm: LLMClient
) -> Schema:
    """Ask the LLM to correct roles; apply only entries that fit the data."""
    payload = {
        "fields": [f.model_dump() for f in schema.fields],
        "samples": records[:LLM_SAMPLE_RECORDS],
    }
    reply = llm.chat(
        _REFINE_SYSTEM,
        [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        json_mode=True,
    )
    try:
        changes = json.loads(reply)["roles"]
    except (ValueError, TypeError, KeyError):
        logger.warning("schema refinement: unreadable LLM reply, ignored")
        return schema
    if not isinstance(changes, dict):
        logger.warning("schema refinement: unreadable LLM reply, ignored")
        return schema

    columns = _columns(records)
    roles: dict[str, Role] = {f.name: f.role for f in schema.fields}
    many_ids = sum(role == "id" for role in changes.values()) > 1
    applied: set[str] = set()
    for name, role in changes.items():
        if name not in roles or role not in ROLES:
            reason = "unknown field or role"
        elif role == "id" and many_ids:
            reason = "more than one id proposed"
        elif not _fits(role, columns[name]):
            reason = "does not fit the data"
        else:
            roles[name] = role
            applied.add(name)
            continue
        logger.warning("schema refinement: %s=%s ignored (%s)", name, role, reason)

    # A newly applied id replaces the old one; the old one gets rule roles back.
    ids = [n for n, r in roles.items() if r == "id"]
    if len(ids) > 1:
        for name in ids:
            if name not in applied:
                roles[name] = _classify(columns[name])
    return _build(roles, schema.record_count)


def _build(roles: dict[str, Role], record_count: int) -> Schema:
    id_field = next((n for n, r in roles.items() if r == "id"), FALLBACK_ID_FIELD)
    return Schema(
        id_field=id_field,
        record_count=record_count,
        fields=[FieldInfo(name=n, role=r) for n, r in roles.items()],
    )


def _columns(records: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Values per field, fields in first-seen order; missing values are None."""
    names: dict[str, None] = {}
    for record in records:
        for key in record:
            names.setdefault(key, None)
    return {name: [r.get(name) for r in records] for name in names}


def _pick_id_field(columns: dict[str, list[Any]]) -> str | None:
    """First id-named field that qualifies, else the first field if it qualifies."""
    for name, values in columns.items():
        if _has_id_token(name) and _is_id_column(values):
            return name
    first = next(iter(columns), None)
    if first is not None and _is_id_column(columns[first]):
        return first
    return None


def _classify(values: list[Any]) -> Role:
    """Role for a field that is not the id field."""
    present = [v for v in values if v is not None]
    if not present:
        return "text"
    if _all_numbers(present):
        return "number"
    if not all(isinstance(v, str) for v in present):
        return "text"
    if _is_date_column(present):
        return "date"
    if _is_category_column(present, len(values)):
        return "category"
    return "text"


def _fits(role: Role, values: list[Any]) -> bool:
    """Whether the data allows this role (used to check LLM suggestions)."""
    present = [v for v in values if v is not None]
    strings = all(isinstance(v, str) for v in present)
    if role == "id":
        return _is_id_column(values)
    if role == "number":
        return bool(present) and _all_numbers(present)
    if role == "date":
        return bool(present) and strings and _is_date_column(present)
    if role == "category":
        return strings
    return True


def _has_id_token(name: str) -> bool:
    """id / uuid / guid as a word: product_id, order.id, productId — not 'paid'."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return not ID_TOKENS.isdisjoint(re.split(r"[^a-z0-9]+", spaced.lower()))


def _is_id_column(values: list[Any]) -> bool:
    """Unique, non-null, and every value an int or a non-empty string."""
    if not values:
        return False
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int | str):
            return False
        if isinstance(v, str) and not v.strip():
            return False
    return len(set(values)) == len(values)


def _all_numbers(values: list[Any]) -> bool:
    return all(isinstance(v, int | float) and not isinstance(v, bool) for v in values)


def _is_date_column(values: list[str]) -> bool:
    sample = values[:DATE_SAMPLE_SIZE]
    parsed = sum(_is_iso_date(v) for v in sample)
    return parsed >= DATE_MIN_RATIO * len(sample)


def _is_iso_date(value: str) -> bool:
    text = value.strip()
    # fromisoformat also accepts compact forms like "20260927"; require dashes.
    if not re.match(r"\d{4}-\d{2}-\d{2}", text):
        return False
    for parse in (date.fromisoformat, datetime.fromisoformat):
        try:
            parse(text)
            return True
        except ValueError:
            continue
    return False


def _is_category_column(values: list[str], record_count: int) -> bool:
    max_unique = max(CATEGORY_MIN_UNIQUE, CATEGORY_UNIQUE_RATIO * record_count)
    avg_len = sum(len(v) for v in values) / len(values)
    return len(set(values)) <= max_unique and avg_len <= CATEGORY_MAX_AVG_LEN
