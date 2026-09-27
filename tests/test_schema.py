import json
from pathlib import Path

import pytest

from obot.core.loader import load_file
from obot.core.schema import (
    Schema,
    analyze_schema,
    load_schema,
    refine_with_llm,
    save_schema,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _roles(records):
    return {f.name: f.role for f in analyze_schema(records).fields}


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = []

    def chat(self, system, messages, json_mode=False):
        self.calls.append((system, messages, json_mode))
        return self.reply


# --- id -------------------------------------------------------------------


def test_id_by_field_name():
    records = [{"name": "a", "product_id": 1}, {"name": "a", "product_id": 2}]
    schema = analyze_schema(records)
    assert schema.id_field == "product_id"
    assert _roles(records)["product_id"] == "id"


def test_id_by_first_field():
    records = [{"code": "x1", "n": 1}, {"code": "x2", "n": 1}]
    assert analyze_schema(records).id_field == "code"


@pytest.mark.parametrize("name", ["paid", "width", "valid"])
def test_id_substring_inside_word_is_not_id(name):
    records = [{"a": 1, name: "u1"}, {"a": 1, name: "u2"}]
    assert analyze_schema(records).id_field == "_obot_id"


@pytest.mark.parametrize("name", ["id", "userId", "order.id", "item-ID"])
def test_id_token_variants(name):
    records = [{"a": 1, name: "u1"}, {"a": 1, name: "u2"}]
    assert analyze_schema(records).id_field == name


def test_id_requires_unique_and_non_null():
    dup = [{"id": 1}, {"id": 1}]
    null = [{"id": 1}, {"id": None}]
    missing = [{"id": 1}, {"other": 2}]
    for records in (dup, null, missing):
        assert analyze_schema(records).id_field == "_obot_id"


def test_only_one_id_field():
    records = [{"id": 1, "sku_id": "a"}, {"id": 2, "sku_id": "b"}]
    roles = _roles(records)
    assert roles["id"] == "id"
    assert roles["sku_id"] != "id"


def test_fallback_id_when_none():
    records = [{"v": 1}, {"v": 1}]
    schema = analyze_schema(records)
    assert schema.id_field == "_obot_id"
    assert "_obot_id" not in [f.name for f in schema.fields]


# --- number / date / category / text ---------------------------------------


def _col(values):
    """Records whose first field is a non-unique filler so it never becomes id."""
    return [{"k": 0, "v": v} for v in values]


def test_number_int_and_float_with_nulls():
    assert _roles(_col([1, 2.5, None, 1]))["v"] == "number"


def test_bool_is_not_number():
    assert _roles(_col([True, False, True]))["v"] == "text"


def test_numeric_strings_are_not_number():
    assert _roles(_col(["1", "2", "1"]))["v"] != "number"


def test_date_at_90_percent():
    values = ["2026-01-01"] * 9 + ["unknown"]
    assert _roles(_col(values))["v"] == "date"


def test_not_date_below_90_percent():
    values = ["2026-01-01"] * 8 + ["unknown"] * 2
    assert _roles(_col(values))["v"] != "date"


def test_datetime_counts_as_date_but_compact_form_does_not():
    assert _roles(_col(["2026-09-27T12:00:00", "2026-09-27"]))["v"] == "date"
    assert _roles(_col(["20260927", "20260928"]))["v"] != "date"


def test_category_within_unique_limit():
    values = [f"c{i % 20}" for i in range(100)]  # 20 unique ≤ max(20, 5)
    assert _roles(_col(values))["v"] == "category"


def test_too_many_unique_values_is_text():
    values = [f"c{i % 21}" for i in range(100)]  # 21 unique > max(20, 5)
    assert _roles(_col(values))["v"] == "text"


def test_category_limit_scales_with_record_count():
    values = [f"c{i % 50}" for i in range(1000)]  # 50 unique ≤ 5% of 1000
    assert _roles(_col(values))["v"] == "category"


def test_long_strings_are_text():
    values = ["x" * 31, "y" * 31, "x" * 31]
    assert _roles(_col(values))["v"] == "text"


def test_mixed_types_and_all_null_are_text():
    records = [{"k": 0, "mixed": 1, "empty": None}, {"k": 0, "mixed": "a"}]
    roles = _roles(records)
    assert roles["mixed"] == "text"
    assert roles["empty"] == "text"


def test_field_order_is_first_seen():
    records = [{"k": 0, "b": 1}, {"k": 0, "a": 2, "b": 3}]
    assert [f.name for f in analyze_schema(records).fields] == ["k", "b", "a"]


# --- save / load / examples -------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    schema = analyze_schema([{"id": 1, "v": "a"}, {"id": 2, "v": "b"}])
    path = tmp_path / "bot" / "schema.json"
    save_schema(schema, path)
    assert load_schema(path) == schema
    assert json.loads(path.read_text(encoding="utf-8"))["obot_schema_version"] == 1


EXPECTED = {
    "products.json": (
        "product_id",
        {
            "price.amount": "number",
            "category": "category",
            "description": "text",
            "released_at": "date",
        },
    ),
    "reviews.jsonl": (
        "review_id",
        {"score": "number", "body": "text", "created_at": "date"},
    ),
    "orders.csv": (
        "_obot_id",
        {"지역": "category", "주문일": "date", "수량": "number"},
    ),
}


@pytest.mark.parametrize("filename", sorted(EXPECTED))
def test_examples_generate_schema_json(tmp_path, filename):
    records = load_file(EXAMPLES / filename)
    path = tmp_path / Path(filename).stem / "schema.json"
    save_schema(analyze_schema(records), path)

    schema = load_schema(path)
    id_field, roles = EXPECTED[filename]
    assert schema.id_field == id_field
    assert schema.record_count == len(records)
    actual = {f.name: f.role for f in schema.fields}
    for name, role in roles.items():
        assert actual[name] == role


# --- LLM refinement ---------------------------------------------------------


def _base() -> tuple[Schema, list[dict]]:
    records = [
        {"k": 0, "code": "A", "note": "x"},
        {"k": 0, "code": "B", "note": "y"},
        {"k": 0, "code": "C", "note": "z"},
        {"k": 0, "code": "D", "note": "w"},
    ]
    return analyze_schema(records), records


def test_refine_applies_valid_changes():
    schema, records = _base()
    llm = FakeLLM('{"roles": {"note": "text", "code": "id"}}')
    refined = refine_with_llm(schema, records, llm)
    roles = {f.name: f.role for f in refined.fields}
    assert roles["note"] == "text"
    assert refined.id_field == "code"

    system, messages, json_mode = llm.calls[0]
    assert json_mode is True
    assert len(json.loads(messages[0]["content"])["samples"]) == 3


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        "[]",
        '{"other": {}}',
        '{"roles": ["note"]}',
        '{"roles": {"missing": "text"}}',  # unknown field
        '{"roles": {"note": "blob"}}',  # unknown role
        '{"roles": {"code": "id", "note": "id"}}',  # two id fields
        '{"roles": {"k": "id"}}',  # values are not unique
    ],
)
def test_refine_keeps_rule_based_result_on_bad_reply(reply):
    schema, records = _base()
    assert refine_with_llm(schema, records, FakeLLM(reply)) == schema


# --- id preference and value types -------------------------------------------


def test_named_id_beats_unique_first_field():
    records = [{"name": "a", "product_id": 1}, {"name": "b", "product_id": 2}]
    roles = _roles(records)
    assert analyze_schema(records).id_field == "product_id"
    assert roles["name"] != "id"


@pytest.mark.parametrize("name", ["uuid", "order_guid", "userUuid"])
def test_uuid_and_guid_names_are_id(name):
    records = [{"a": 1, name: "u1"}, {"a": 1, name: "u2"}]
    assert analyze_schema(records).id_field == name


def test_unique_float_first_field_is_not_id():
    records = [{"price": 1.5, "v": "a"}, {"price": 2.5, "v": "a"}]
    schema = analyze_schema(records)
    assert schema.id_field == "_obot_id"
    assert _roles(records)["price"] == "number"


def test_blank_string_values_are_not_id():
    assert analyze_schema([{"id": "a"}, {"id": " "}]).id_field == "_obot_id"


# --- LLM refinement: per-entry validation -------------------------------------


def test_refine_applies_valid_entries_and_drops_invalid_ones():
    schema, records = _base()
    reply = '{"roles": {"note": "text", "missing": "text", "code": "blob"}}'
    refined = refine_with_llm(schema, records, FakeLLM(reply))
    roles = {f.name: f.role for f in refined.fields}
    assert roles["note"] == "text"
    assert roles["code"] == {f.name: f.role for f in schema.fields}["code"]


def test_refine_rejects_role_that_does_not_fit_data():
    schema, records = _base()
    reply = '{"roles": {"note": "number", "code": "date", "k": "text"}}'
    refined = refine_with_llm(schema, records, FakeLLM(reply))
    roles = {f.name: f.role for f in refined.fields}
    before = {f.name: f.role for f in schema.fields}
    assert roles["note"] == before["note"]
    assert roles["code"] == before["code"]
    assert roles["k"] == "text"


def test_refine_new_id_reclassifies_previous_id():
    records = [{"id": i, "sku": f"S{i}"} for i in range(4)]
    schema = analyze_schema(records)
    assert schema.id_field == "id"
    refined = refine_with_llm(schema, records, FakeLLM('{"roles": {"sku": "id"}}'))
    roles = {f.name: f.role for f in refined.fields}
    assert refined.id_field == "sku"
    assert roles["id"] == "number"


def test_refine_two_id_proposals_ignores_only_ids():
    schema, records = _base()
    reply = '{"roles": {"code": "id", "note": "id", "k": "text"}}'
    refined = refine_with_llm(schema, records, FakeLLM(reply))
    roles = {f.name: f.role for f in refined.fields}
    assert refined.id_field == "_obot_id"
    assert roles["k"] == "text"


def test_refine_previous_id_takes_role_from_reply():
    records = [{"id": i, "sku": f"S{i}"} for i in range(4)]
    schema = analyze_schema(records)
    reply = '{"roles": {"sku": "id", "id": "text"}}'
    refined = refine_with_llm(schema, records, FakeLLM(reply))
    roles = {f.name: f.role for f in refined.fields}
    assert refined.id_field == "sku"
    assert roles["id"] == "text"


def test_refine_demoting_id_without_new_id_falls_back():
    records = [{"id": i, "v": "a"} for i in range(4)]
    schema = analyze_schema(records)
    refined = refine_with_llm(schema, records, FakeLLM('{"roles": {"id": "number"}}'))
    assert refined.id_field == "_obot_id"
    assert {f.name: f.role for f in refined.fields}["id"] == "number"
