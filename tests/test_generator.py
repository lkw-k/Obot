from obot.core.generator import HISTORY_LIMIT, SYSTEM_PROMPT, generate_answer
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
