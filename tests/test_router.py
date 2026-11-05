import pytest

from obot.core.router import SYSTEM_PROMPT, route
from obot.core.schema import FieldInfo, Schema

SCHEMA = Schema(
    id_field="id",
    record_count=2,
    fields=[FieldInfo(name="id", role="id"), FieldInfo(name="price", role="number")],
)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"route": "aggregate"}', "aggregate"),
        ('{"route": "search"}', "search"),
        ('{"route": "other"}', "search"),
        ('{"other": "aggregate"}', "search"),
        ('["aggregate"]', "search"),
        ("aggregate", "search"),
    ],
)
def test_route_parses_reply_or_falls_back_to_search(fake_llm, reply, expected):
    fake_llm.reply = reply

    assert route(fake_llm, "how many?", SCHEMA) == expected


def test_prompt_has_fields_and_question_in_json_mode(fake_llm):
    route(fake_llm, "most expensive?", SCHEMA)

    (call,) = fake_llm.calls
    assert call["system"] == SYSTEM_PROMPT
    assert call["json_mode"] is True
    content = call["messages"][-1]["content"]
    assert "- id (id)\n- price (number)" in content
    assert content.endswith("Question: most expensive?")
