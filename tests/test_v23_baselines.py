"""Local invariants for V2.3 measurement programs (not quality scores)."""
import pytest

pytest.skip("V2.3 intent-routed evaluation is not the current unified Agent path",
            allow_module_level=True)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.nodes import todo_context_messages
from evals.v23_long_context import synthetic_turn
from evals.v23_tool_ab import TOOL_CASES, schema_chars
from evals.v23_full_tool_ab import CASES


def test_long_thread_fixture_preserves_complete_tool_call_result_pair():
    turn = synthetic_turn(20)
    assert isinstance(turn[0], HumanMessage)
    assert isinstance(turn[1], AIMessage)
    assert isinstance(turn[2], ToolMessage)
    assert turn[1].tool_calls[0]["id"] == turn[2].tool_call_id
    active_history = [message for index in range(1, 21) for message in synthetic_turn(index)]
    active_window = todo_context_messages(active_history)
    assert any(isinstance(message, AIMessage) and message.tool_calls for message in active_window)
    assert any(isinstance(message, ToolMessage) and message.tool_call_id == "eval-tool-20"
               for message in active_window)
    history = [message for index in range(1, 151) for message in synthetic_turn(index)]
    assert len(history) == 302
    window = todo_context_messages(history)
    assert len(window) <= 10
    assert any(isinstance(message, HumanMessage) for message in window)
    assert not any("review is on Wednesday" in str(message.content) for message in window)


def test_tool_ab_only_defines_first_tool_contract_and_never_calls_tools():
    assert len(TOOL_CASES) == 5
    assert {"search_todos", "update_todo"} == TOOL_CASES[3][2]
    assert schema_chars([]) == 0


def test_full_loop_ab_corpus_covers_db_outcomes_without_delete_or_external_write():
    assert {case[0] for case in CASES} == {
        "create", "update", "search", "deadline", "priority_tag", "multi_step"}
    assert all("Delete" not in prompt and "calendar" not in prompt.lower()
               for _, prompt, _ in CASES)
