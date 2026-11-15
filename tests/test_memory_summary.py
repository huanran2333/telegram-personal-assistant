"""LangMem summary boundary and privacy regressions, without external models."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.memory_summary import _source_messages


def _turn(index: int):
    return [HumanMessage(id=f"u{index}", content=f"The project number is {index}."),
            AIMessage(id=f"a{index}", content="Noted.")]


def test_summary_retains_latest_complete_turns_and_excludes_active_request():
    history = _turn(1) + _turn(2) + _turn(3) + [HumanMessage(id="u4", content="What number?")]
    sources, mapping = _source_messages(history, retained_groups=2)
    assert [m.id for m in sources] == ["turn:a1"]
    assert mapping == {"turn:a1": ("u1", "a1")}


def test_summary_never_forwards_tool_payload_or_partial_tool_round():
    history = [HumanMessage(id="u1", content="Create a task"),
               AIMessage(id="call1", content="", tool_calls=[{"id": "t1", "name": "create_todo", "args": {}}]),
               ToolMessage(id="result1", tool_call_id="t1", content="PRIVATE TOOL PAYLOAD"),
               AIMessage(id="a1", content="Created."),
               *_turn(2), *_turn(3), *_turn(4),
               HumanMessage(id="u5", content="Now change it"),
               AIMessage(id="call2", content="", tool_calls=[{"id": "t2", "name": "update_todo", "args": {}}])]
    sources, mapping = _source_messages(history, retained_groups=2)
    assert [message.id for message in sources] == ["turn:a1", "turn:a2"]
    assert mapping["turn:a1"] == ("u1", "a1")
    assert all("PRIVATE TOOL PAYLOAD" not in message.content for message in sources)


def test_summary_rejects_missing_stable_source_ids():
    history = [HumanMessage(content="no id"), AIMessage(id="answer", content="ok"),
               *_turn(2), *_turn(3)]
    with pytest.raises(ValueError, match="stable IDs"):
        _source_messages(history, retained_groups=2)


def test_summary_retained_groups_must_be_positive():
    with pytest.raises(ValueError):
        _source_messages([], retained_groups=0)
