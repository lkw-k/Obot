from obot.core.generator import (
    HISTORY_LIMIT,
    SYSTEM_PROMPT,
    generate_answer,
    generate_sql_answer,
)
from obot.core.store import Hit


def test_prompt_contains_context_with_ids(fake_llm):
    hits = [Hit("P-1", 0.9, {"name": "earphone", "price": 99000, "note": None})]

    answer = generate_answer(fake_llm, "cheap earphones?", hits, [])

    assert answer == "fake answer"
    (call,) = fake_llm.calls
    assert call["system"] == SYSTEM_PROMPT
    prompt = call["messages"][-1]
    assert prompt["role"] == "user"
    assert "[id: P-1]\nname: earphone\nprice: 99000" in prompt["content"]
    assert "note" not in prompt["content"]
    assert prompt["content"].endswith("Question: cheap earphones?")


def test_empty_context_is_marked(fake_llm):
    generate_answer(fake_llm, "q", [], [])

    assert "(no records found)" in fake_llm.calls[0]["messages"][-1]["content"]


def test_history_keeps_last_messages(fake_llm):
    history = [{"role": "user", "content": str(i)} for i in range(15)]

    generate_answer(fake_llm, "q", [], history)

    messages = fake_llm.calls[0]["messages"]
    assert messages[:-1] == history[-HISTORY_LIMIT:]


def test_sql_prompt_contains_query_and_rows(fake_llm):
    history = [{"role": "user", "content": "hi"}]

    generate_sql_answer(fake_llm, "가장 비싼?", "SELECT 1", [{"이름": "a"}], history)

    (call,) = fake_llm.calls
    assert call["system"] == SYSTEM_PROMPT
    assert call["messages"][:-1] == history
    assert call["messages"][-1]["content"] == (
        'Context:\nSQL: SELECT 1\nRows: [{"이름": "a"}]\n\nQuestion: 가장 비싼?'
    )


def test_empty_sql_result_is_marked(fake_llm):
    generate_sql_answer(fake_llm, "q", "SELECT 1", [], [])

    assert "Rows: (no rows)" in fake_llm.calls[0]["messages"][-1]["content"]
