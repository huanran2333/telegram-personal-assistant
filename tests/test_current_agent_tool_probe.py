"""The current-schema probe must not turn missing usage into a made-up saving."""

from evals.current_agent_tool_probe import CASES, first_tool_ok, route_token_total, token_total


def test_case_ids_unique_and_personal_vs_standalone_calendar_distinct():
    assert len({case[0] for case in CASES}) == len(CASES)
    by_id = {case[0]: case for case in CASES}
    assert by_id["personal_meeting"][2:] == ("todo", None, {"create_todo"})
    assert by_id["standalone_calendar_create"][2:] == (
        "calendar_create", None, {"prepare_action_request"})


def test_first_tool_scoring_and_missing_usage_fail_closed():
    assert first_tool_ok(["read_calendar_events", "list_todos"], {"read_calendar_events"})
    assert not first_tool_ok([], {"read_calendar_events"})
    assert not first_tool_ok(["delete_todo"], {"prepare_action_request"})
    rows = [{"scoped": {"usage": {"input": 10}},
             "route_usage": {"input_tokens": 3}},
            {"scoped": {"usage": {"input": None}},
             "route_usage": {"input_tokens": 4}}]
    assert token_total(rows, "scoped", "input") is None
    assert route_token_total(rows, "input_tokens") == 7
