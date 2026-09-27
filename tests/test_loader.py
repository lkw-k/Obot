import json
import logging
from pathlib import Path

import pytest

from obot.core.loader import LoadError, flatten, load_file


def _write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_text(text, encoding=encoding, newline="")
    return path


def test_json_top_level_array(tmp_path):
    f = _write(tmp_path / "a.json", '[{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]')
    assert load_file(f) == [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]


def test_json_object_with_one_array_field(tmp_path):
    f = _write(tmp_path / "a.json", '{"total": 1, "items": [{"id": 1}]}')
    assert load_file(f) == [{"id": 1}]


@pytest.mark.parametrize(
    "text",
    [
        '{"a": [{"id": 1}], "b": [{"id": 2}]}',  # two array fields
        '{"a": {"id": 1}}',  # no array field
        '"just a string"',
        "{not json",
    ],
)
def test_json_invalid_shapes_raise(tmp_path, text):
    f = _write(tmp_path / "a.json", text)
    with pytest.raises(LoadError):
        load_file(f)


def test_json_skips_non_object_items(tmp_path, caplog):
    f = _write(tmp_path / "a.json", '[{"id": 1}, 5, "x"]')
    with caplog.at_level(logging.WARNING):
        assert load_file(f) == [{"id": 1}]
    assert "not an object" in caplog.text


def test_jsonl_skips_blank_and_invalid_lines(tmp_path, caplog):
    f = _write(tmp_path / "a.jsonl", '{"id": 1}\n\n   \n{broken\n{"id": 2}\n')
    with caplog.at_level(logging.WARNING):
        assert load_file(f) == [{"id": 1}, {"id": 2}]
    assert "a.jsonl:4" in caplog.text


@pytest.mark.parametrize("suffix", [".json", ".jsonl", ".csv"])
def test_no_valid_records_raises(tmp_path, suffix):
    body = {".json": "[1, 2]", ".jsonl": "{broken\n\n", ".csv": "a,b\n"}[suffix]
    f = _write(tmp_path / f"a{suffix}", body)
    with pytest.raises(LoadError, match="no valid records"):
        load_file(f)


def test_unsupported_extension_raises(tmp_path):
    with pytest.raises(LoadError, match="unsupported"):
        load_file(_write(tmp_path / "a.txt", "x"))


def test_flatten_nested_objects_and_lists():
    record = {"a": {"b": 1, "c": {"d": "x"}}, "tags": ["가", "b"], "e": None}
    assert flatten(record) == {
        "a.b": 1,
        "a.c.d": "x",
        "tags": json.dumps(["가", "b"], ensure_ascii=False),
        "e": None,
    }


def test_json_records_are_flattened(tmp_path):
    f = _write(tmp_path / "a.json", '[{"id": 1, "price": {"amount": 10}, "t": [1]}]')
    assert load_file(f) == [{"id": 1, "price.amount": 10, "t": "[1]"}]


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-8", "cp949"])
def test_csv_encodings(tmp_path, encoding):
    f = _write(tmp_path / "a.csv", "이름,지역\n홍길동,서울\n", encoding=encoding)
    assert load_file(f) == [{"이름": "홍길동", "지역": "서울"}]


def test_csv_undecodable_raises(tmp_path):
    f = tmp_path / "a.csv"
    f.write_bytes(b"a,b\n\xff\xff\xff,1\n")
    with pytest.raises(LoadError, match="cannot decode"):
        load_file(f)


def test_csv_numeric_conversion_and_empty_values(tmp_path):
    f = _write(
        tmp_path / "a.csv",
        "i,neg,f,e,s,empty,under,nan\n42,-7,3.5,1e3,12abc,,1_000,nan\n",
    )
    assert load_file(f) == [
        {
            "i": 42,
            "neg": -7,
            "f": 3.5,
            "e": 1000.0,
            "s": "12abc",
            "empty": None,
            "under": "1_000",
            "nan": "nan",
        }
    ]


def test_csv_short_row_fills_none_and_long_row_is_skipped(tmp_path, caplog):
    f = _write(tmp_path / "a.csv", "a,b\n1\n1,2,3\n4,5\n")
    with caplog.at_level(logging.WARNING):
        assert load_file(f) == [{"a": 1, "b": None}, {"a": 4, "b": 5}]
    assert "too many columns" in caplog.text


def test_csv_header_required(tmp_path):
    with pytest.raises(LoadError, match="header"):
        load_file(_write(tmp_path / "a.csv", ""))


def test_csv_leading_zero_column_stays_strings(tmp_path):
    f = _write(tmp_path / "a.csv", "code,zip\n007,01234\n12,63000\n")
    assert load_file(f) == [
        {"code": "007", "zip": "01234"},
        {"code": "12", "zip": "63000"},
    ]


def test_csv_zero_and_decimals_are_numbers(tmp_path):
    f = _write(tmp_path / "a.csv", "v\n0\n0.5\n10\n")
    assert load_file(f) == [{"v": 0}, {"v": 0.5}, {"v": 10}]


def test_csv_column_with_any_text_stays_strings(tmp_path):
    f = _write(tmp_path / "a.csv", "v,w\n1,\nN/A,2\n")
    assert load_file(f) == [{"v": "1", "w": None}, {"v": "N/A", "w": 2}]


def test_jsonl_non_object_line_logs_line_number(tmp_path, caplog):
    f = _write(tmp_path / "a.jsonl", '{"id": 1}\n\n5\n[1]\n{"id": 2}\n')
    with caplog.at_level(logging.WARNING):
        assert load_file(f) == [{"id": 1}, {"id": 2}]
    assert "a.jsonl:3: not an object" in caplog.text
    assert "a.jsonl:4: not an object" in caplog.text


def test_csv_overflowing_number_keeps_column_as_strings(tmp_path):
    f = _write(tmp_path / "a.csv", "v\n1\n1e400\n")
    assert load_file(f) == [{"v": "1"}, {"v": "1e400"}]


def test_csv_numeric_cells_with_spaces(tmp_path):
    f = _write(tmp_path / "a.csv", "v\n 42 \n7\n")
    assert load_file(f) == [{"v": 42}, {"v": 7}]


def test_flatten_key_collision_keeps_last_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        assert flatten({"a.b": 1, "a": {"b": 2}}) == {"a.b": 2}
    assert "'a.b' appears twice" in caplog.text


def test_csv_duplicate_header_warns(tmp_path, caplog):
    f = _write(tmp_path / "a.csv", "a,a\n1,2\n")
    with caplog.at_level(logging.WARNING):
        assert load_file(f) == [{"a": 2}]
    assert "duplicate header a" in caplog.text
