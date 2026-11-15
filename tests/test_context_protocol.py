"""Table-driven provider message protocol regressions."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.context_protocol import message_groups, select_chat_context, select_todo_context


def _call(*ids):
    return AIMessage(content="", tool_calls=[{"name": "get_todo", "args": {}, "id": i} for i in ids])


@pytest.mark.parametrize("history,expected", [
    ([HumanMessage(content="a"), _call("a"), ToolMessage(content="x", tool_call_id="a"),
      AIMessage(content="done")], 4),
    ([HumanMessage(content="a"), _call("a", "b"),
      ToolMessage(content="x", tool_call_id="a"), ToolMessage(content="y", tool_call_id="b"),
      AIMessage(content="done")], 5),
    ([ToolMessage(content="orphan", tool_call_id="a"), HumanMessage(content="safe")], 1),
    ([HumanMessage(content="incomplete"), _call("a", "b"),
      ToolMessage(content="x", tool_call_id="a")], 0),
])
def test_group_integrity(history, expected):
    selected = select_todo_context(history, max_messages=2)
    assert len(selected) == expected
    assert not selected or isinstance(selected[0], HumanMessage)


def test_two_turns_do_not_cut_tool_round_at_boundary():
    first = [HumanMessage(content="first"), _call("a"),
             ToolMessage(content="x" * 10000, tool_call_id="a"), AIMessage(content="done")]
    second = [HumanMessage(content="second"), AIMessage(content="yes")]
    assert select_todo_context(first + second, max_messages=5) == second
    assert select_todo_context(first + second, max_messages=6) == first + second


def test_orphan_inside_turn_is_not_forwarded():
    invalid = [HumanMessage(content="bad"), ToolMessage(content="orphan", tool_call_id="x")]
    good = [HumanMessage(content="good"), AIMessage(content="okay")]
    assert not message_groups(invalid)[0].complete
    assert select_todo_context(invalid + good) == good
    assert select_chat_context(invalid + good) == good


def test_serialized_restart_restores_valid_multi_tool_group():
    original = [HumanMessage(content="do it"), _call("a", "b"),
                ToolMessage(content="one", tool_call_id="a"),
                ToolMessage(content="two", tool_call_id="b"), AIMessage(content="done")]
    restored = [type(message).model_validate(message.model_dump()) for message in original]
    assert len(select_todo_context(restored, max_messages=3)) == len(original)


def test_summary_boundary_adjacent_to_tool_round_is_safe():
    history = [HumanMessage(content="old"), AIMessage(content="old answer"),
               HumanMessage(content="new"), _call("a"),
               ToolMessage(content="result", tool_call_id="a"), AIMessage(content="final")]
    assert select_todo_context(history, max_messages=4) == history[2:]


def test_active_valid_tool_result_is_allowed_but_older_unfinished_turn_is_not():
    active = [HumanMessage(content="current"), _call("a"),
              ToolMessage(content="result", tool_call_id="a")]
    assert select_todo_context(active) == active
    older = [HumanMessage(content="old"), _call("b"),
             ToolMessage(content="old result", tool_call_id="b")]
    finished = [HumanMessage(content="new"), AIMessage(content="done")]
    assert select_todo_context(older + finished) == finished


def test_tool_retry_round_keeps_both_call_result_pairs():
    history = [HumanMessage(content="update the task"), _call("first"),
               ToolMessage(content="temporary error", tool_call_id="first"),
               _call("retry"), ToolMessage(content="updated", tool_call_id="retry"),
               AIMessage(content="done")]
    assert select_todo_context(history, max_messages=3) == history


def test_recent_pronoun_turn_retains_prior_completed_todo_turn():
    history = [HumanMessage(content="Create thesis outline task"), AIMessage(content="Created"),
               HumanMessage(content="Set its priority high")]
    assert select_todo_context(history, max_messages=10) == history
