from obot.core.chunker import chunk_text


def test_serializes_fields_as_lines_in_order():
    record = {"name": "키보드", "price.amount": 1200, "tags": '["a", "b"]'}
    assert chunk_text(record) == 'name: 키보드\nprice.amount: 1200\ntags: ["a", "b"]'


def test_excludes_null_and_empty_strings():
    record = {"a": None, "b": "", "c": 0, "d": False, "e": "x"}
    assert chunk_text(record) == "c: 0\nd: False\ne: x"


def test_empty_record_gives_empty_text():
    assert chunk_text({"a": None}) == ""
