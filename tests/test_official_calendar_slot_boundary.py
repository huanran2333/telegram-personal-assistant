"""A derived Todo time needs a completed same-turn Calendar candidate result."""
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.official_permissions import _verified_calendar_slot


REQUEST = "先查明天下午空闲，再把复习 FastAPI 安排到合适的时间。"
ARGS = {"title": "复习 FastAPI", "planned_start_str": "2026-11-04T13:00:00+08:00",
        "planned_end_str": "2026-11-04T14:00:00+08:00",
        "time_source_quote": "把复习 FastAPI 安排到合适的时间"}
CALL = {"name": "calendar_candidate_slots", "id": "verified-read",
        "args": {"title": "复习 FastAPI", "duration_minutes": 60,
                 "range_start": "2026-11-04T13:00:00+08:00",
                 "range_end": "2026-11-04T18:00:00+08:00"}}
SLOT = {"start": "2026-11-04T13:00:00+08:00",
        "end": "2026-11-04T14:00:00+08:00"}


def state(*messages, turn="turn-1"):
    return {"active_turn_id": turn, "messages": list(messages)}


def test_completed_same_turn_exact_slot_is_accepted():
    messages = (HumanMessage(id="turn-1", content=REQUEST),
                AIMessage(content="", tool_calls=[CALL]),
                ToolMessage(content=json.dumps([SLOT]), tool_call_id="verified-read"))
    assert _verified_calendar_slot(state(*messages), ARGS, REQUEST)


def test_same_batch_without_tool_result_is_not_evidence():
    messages = (HumanMessage(id="turn-1", content=REQUEST),
                AIMessage(content="", tool_calls=[CALL, {
                    "name": "create_todo", "id": "premature", "args": ARGS}]))
    assert not _verified_calendar_slot(state(*messages), ARGS, REQUEST)


def test_error_empty_mismatch_and_old_turn_are_not_evidence():
    human = HumanMessage(id="turn-1", content=REQUEST)
    call = AIMessage(content="", tool_calls=[CALL])
    for result in (
        ToolMessage(content="unavailable", tool_call_id="verified-read", status="error"),
        ToolMessage(content="[]", tool_call_id="verified-read"),
        ToolMessage(content=json.dumps([{"start": "2026-11-04T14:00:00+08:00",
                                         "end": "2026-11-04T15:00:00+08:00"}]),
                    tool_call_id="verified-read"),
        ToolMessage(content=json.dumps([SLOT]), tool_call_id="other-call"),
    ):
        assert not _verified_calendar_slot(state(human, call, result), ARGS, REQUEST)
    old = state(human, call,
                ToolMessage(content=json.dumps([SLOT]), tool_call_id="verified-read"),
                HumanMessage(id="turn-2", content=REQUEST), turn="turn-2")
    assert not _verified_calendar_slot(old, ARGS, REQUEST)


def test_candidate_for_different_title_or_unquoted_request_is_rejected():
    messages = (HumanMessage(id="turn-1", content=REQUEST),
                AIMessage(content="", tool_calls=[{**CALL, "args": {**CALL["args"],
                                                                "title": "另一个任务"}}]),
                ToolMessage(content=json.dumps([SLOT]), tool_call_id="verified-read"))
    assert not _verified_calendar_slot(state(*messages), ARGS, REQUEST)
    assert not _verified_calendar_slot(
        state(HumanMessage(id="turn-1", content=REQUEST),
              AIMessage(content="", tool_calls=[CALL]),
              ToolMessage(content=json.dumps([SLOT]), tool_call_id="verified-read")),
        {**ARGS, "time_source_quote": "另一个任务的时间"}, REQUEST)
