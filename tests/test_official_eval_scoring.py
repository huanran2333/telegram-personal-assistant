"""Boundary checks for the independent Evaluation scorer, no model or DB."""
from experiments.official_agent_eval_20261020.test_official_eval import _grade, EvalCalendarWorld


def _state(owner="owner", todos=None):
    return {"owner": owner, "todos": todos or [], "actions": []}


def test_rejection_is_not_wrong_model_proposal():
    case = {"goals": [{"kind": "reject_delete", "title": "T"}],
            "allowed_writes": [], "decisions": {"T": "reject"}}
    todo = {"id": "one", "owner": "owner", "title": "T"}
    messages = [
        {"type": "AIMessage", "tool_calls": [{"name": "propose_todo_delete", "id": "c", "args": {}}], "content": ""},
        {"type": "ToolMessage", "tool_call_id": "c", "status": "error",
         "content": "User rejected the tool call"},
        {"type": "AIMessage", "tool_calls": [], "content": "T 未删除"},
    ]
    approval = [{"decision": "reject", "target": "Delete Todo: T", "executed": False}]
    grade = _grade(case, _state(todos=[todo]), _state(todos=[todo]), messages, approval, [])
    assert grade["goals"][0]["env_action"] == "PASS"
    assert grade["wrong_proposals_blocked"] == []


def test_read_failure_without_title_is_uncertain_not_scorer_error():
    case = {"goals": [{"kind": "calendar_read", "contains": "Blueboard"}],
            "allowed_writes": [], "decisions": {}}
    messages = [{"type": "AIMessage", "tool_calls": [], "content": "当天无会议"}]
    grade = _grade(case, _state(), _state(), messages, [], [])
    assert grade["goals"][0]["env_action"] == "FAIL"
    assert grade["communication"] == "UNCERTAIN"


async def test_fake_calendar_query_must_match_event_text():
    fake = EvalCalendarWorld()
    rows = await fake.search_events("meeting", "2026-11-10T00:00:00+08:00",
                                    "2026-11-11T00:00:00+08:00")
    assert rows == []
    rows = await fake.search_events("Blueboard", "2026-11-10T00:00:00+08:00",
                                    "2026-11-11T00:00:00+08:00")
    assert len(rows) == 1
