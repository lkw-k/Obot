"""Field role analysis (spec §5.3)."""

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel

from obot.llm.base import LLMClient

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
    names = _field_names(records)
    roles: dict[str, Role] = {}
    for position, name in enumerate(names):
        values = [r.get(name) for r in records]
        allow_id = "id" not in roles.values()
        roles[name] = _classify(name, position, values, allow_id)
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
    """Ask the LLM to correct roles; keep the rule-based schema on any bad reply."""
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
    except (json.JSONDecodeError, TypeError, KeyError):
        return schema
    if not isinstance(changes, dict):
        return schema

    roles: dict[str, Role] = {f.name: f.role for f in schema.fields}
    for name, role in changes.items():
        if name not in roles or role not in ROLES:
            return schema
        roles[name] = role

    id_fields = [n for n, r in roles.items() if r == "id"]
    if len(id_fields) > 1:
        return schema
    if id_fields and not _unique_non_null([r.get(id_fields[0]) for r in records]):
        return schema
    return _build(roles, schema.record_count)


def _build(roles: dict[str, Role], record_count: int) -> Schema:
    id_field = next((n for n, r in roles.items() if r == "id"), FALLBACK_ID_FIELD)
    return Schema(
        id_field=id_field,
        record_count=record_count,
        fields=[FieldInfo(name=n, role=r) for n, r in roles.items()],
    )


def _field_names(records: list[dict[str, Any]]) -> list[str]:
    """Field names in first-seen order across all records."""
    seen: dict[str, None] = {}
    for record in records:
        for key in record:
            seen.setdefault(key, None)
    return list(seen)


def _classify(name: str, position: int, values: list[Any], allow_id: bool) -> Role:
    if allow_id and (position == 0 or _has_id_token(name)) and _unique_non_null(values):
        return "id"

    present = [v for v in values if v is not None]
    if not present:
        return "text"
    if all(isinstance(v, int | float) and not isinstance(v, bool) for v in present):
        return "number"
    if not all(isinstance(v, str) for v in present):
        return "text"
    if _is_date_column(present):
        return "date"
    if _is_category_column(present, len(values)):
        return "category"
    return "text"


def _has_id_token(name: str) -> bool:
    """'id' as a word: id, product_id, order.id, productId — not 'paid' or 'width'."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return "id" in re.split(r"[^a-z0-9]+", spaced.lower())


def _unique_non_null(values: list[Any]) -> bool:
    if not values or any(v is None for v in values):
        return False
    try:
        return len(set(values)) == len(values)
    except TypeError:
        return False


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
