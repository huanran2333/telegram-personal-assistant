"""Evaluator calibration: incorrect and unsafe outputs must fail closed."""
import json
from copy import deepcopy

import pytest

from app.agent.planner import RouteModelError, RoutedRequest, analyze_route
from app.agent.prompt import SYSTEM_PROMPT
from evals.quality_graders import (grade_memory_retrieval, grade_todo_environment,
                                   grade_tool_trajectory, grade_reminder_definition)
from evals.quality_suite import DATASET, compare, grade_route, load_dataset, summarize
from evals.quality_todo_runner import compare as compare_todo
from evals.quality_todo_runner import load_dataset as load_task_dataset
from evals.quality_regrade import regrade_route


def test_dataset_is_versioned_unique_split_and_synthetic():
    corpus, digest = load_dataset()
    assert len(corpus["cases"]) >= 20
    assert len(digest) == 64
    assert {case["split"] for case in corpus["cases"]} == {"dev", "holdout"}
    assert all("G4B-" not in case["prompt"] for case in corpus["cases"])


def test_dataset_duplicate_and_missing_fields_rejected(tmp_path):
    corpus, _ = load_dataset()
    corpus["cases"].append(deepcopy(corpus["cases"][0]))
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(corpus), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicated"):
        load_dataset(path)
    corpus["cases"].pop()
    del corpus["cases"][0]["prompt"]
    path.write_text(json.dumps(corpus), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_dataset(path)


def test_provider_error_is_not_a_successful_clarification():
    case = {"expected_intent": "clarify", "expected_typed_plan": False}
    assert grade_route(case, route_error=ValueError("ambiguous"))["route_ok"] is True
    assert grade_route(case, route_error=RouteModelError("provider failed"))["route_ok"] is False
    assert grade_route(case, route_error=TimeoutError())["route_ok"] is False


def test_route_operation_and_plan_are_independent_scores():
    case = {"expected_intent": "todo", "expected_todo_operation": "update",
            "expected_typed_plan": False}
    route = RoutedRequest(intent="todo", todo_operation="list")
    scored = grade_route(case, route=route, plan=type("Plan", (), {"requires_plan": False})())
    assert scored["route_ok"] is True
    assert scored["operation_ok"] is False
    assert scored["plan_ok"] is True


def test_intent_correct_but_intentionally_blocked_plan_is_not_route_failure():
    case = {"expected_intent": "cross_domain", "expected_typed_plan": False,
            "expected_plan_error": True}
    scored = grade_route(case, route=RoutedRequest(intent="cross_domain"),
                         plan_error=ValueError("unsafe read-then-write composition"))
    assert scored["route_ok"] is True
    assert scored["plan_ok"] is True
    assert scored["failure_stage"] == "plan"


def test_real_planner_intentionally_blocks_calendar_read_then_todo_write():
    route = RoutedRequest(intent="cross_domain", todo_operation="create",
                          date_reference="tomorrow", title="Synthetic follow-up",
                          todo_title="Synthetic follow-up")
    with pytest.raises(ValueError, match="separate plan"):
        analyze_route(route, "Asia/Shanghai",
                      request="查看明天下午的日历安排，然后创建跟进合成项目的待办。")


def test_trajectory_enforces_order_args_forbidden_and_exposes_extra():
    actual = [{"name": "list_todos", "args": {"user_id": "u", "status": "pending"}},
              {"name": "update_todo", "args": {"user_id": "u", "status": "done"}}]
    assert grade_tool_trajectory(actual, [{"name": "list_todos", "args": {"user_id": "u"}},
                                          {"name": "update_todo", "args": {"status": "done"}}])["passed"]
    assert not grade_tool_trajectory(actual, [{"name": "update_todo", "args": {}},
                                              {"name": "list_todos", "args": {}}])["passed"]
    assert not grade_tool_trajectory(actual, [], forbidden={"update_todo"})["passed"]


def test_memory_retrieval_separates_recall_precision_and_forbidden():
    result = grade_memory_retrieval(["valid", "noise", "stale"], {"valid", "missing"},
                                    forbidden_ids={"stale"}, k=3)
    assert result["recall_at_k"] == 0.5
    assert result["precision_at_k"] == pytest.approx(1 / 3)
    assert result["safe"] is False


def test_environment_checks_db_state_not_reply_and_catches_cross_owner():
    before = [{"id": "other", "user_id": "other-user", "title": "Other synthetic", "status": "pending"}]
    after = before + [{"id": "new", "user_id": "owner", "title": "Synthetic task", "status": "pending"}]
    assert grade_todo_environment(before, after, owner_id="owner",
                                  expected={"Synthetic task": "pending"})["passed"]
    corrupted = deepcopy(after)
    corrupted[0]["status"] = "done"
    assert not grade_todo_environment(before, corrupted, owner_id="owner",
                                      expected={"Synthetic task": "pending"})["passed"]
    assert not grade_todo_environment(before, before, owner_id="owner",
                                      expected={"Synthetic task": "pending"})["passed"]


def test_reminder_definition_checks_exact_local_time_and_no_dispatch():
    before = [{"id": "owned", "user_id": "owner", "title": "Synthetic task",
               "status": "pending", "reminder_at": None, "reminder_version": 0}]
    rows = [{"id": "owned", "user_id": "owner", "title": "Synthetic task",
             "status": "pending", "reminder_at": "2030-01-15T01:00:00+00:00",
             "reminder_version": 1}]
    args = {"owner_id": "owner", "title": "Synthetic task",
            "expected_local": "2030-01-15 09:00", "timezone_name": "Asia/Shanghai",
            "new_occurrences": 0, "new_deliveries": 0, "expected_version_delta": 1}
    assert grade_reminder_definition(before, rows, **args)["passed"]
    assert not grade_reminder_definition(before, rows, **(args | {"new_deliveries": 1}))["passed"]
    assert not grade_reminder_definition(before, rows, **(args | {"new_occurrences": 1}))["passed"]
    assert not grade_reminder_definition(before, rows, **(args | {"expected_local": "2030-01-15 10:00"}))["passed"]
    assert not grade_reminder_definition(before, rows, **(args | {"owner_id": "other"}))["passed"]
    duplicate = deepcopy(rows)
    duplicate[0]["reminder_version"] = 2
    assert not grade_reminder_definition(before, duplicate, **args)["passed"]


def test_reminder_dataset_is_versioned_and_only_uses_synthetic_titles():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "evals/datasets/agent_quality_reminder_v1.json"
    data, digest = load_task_dataset(path)
    assert len(data["cases"]) == 4 and len(digest) == 64
    assert {case["split"] for case in data["cases"]} == {"dev", "holdout"}
    assert all(case["expected_reminder_local"] == "2030-01-15 09:00" for case in data["cases"])


def test_reminder_dataset_rejects_non_synthetic_seed(tmp_path):
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / "evals/datasets/agent_quality_reminder_v1.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    data["cases"][0]["initial_todos"] = ["private existing task"]
    target = tmp_path / "unsafe.json"
    target.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="case-owned synthetic"):
        load_task_dataset(target)


def test_synthetic_calendar_read_grader_checks_range_owner_and_no_write():
    from evals.quality_calendar_read_runner import SyntheticCalendar, grade_case, load_dataset
    cases, digest = load_dataset()
    assert len(cases["cases"]) == 4 and len(digest) == 64
    case = cases["cases"][0]
    calendar = SyntheticCalendar()
    calendar.searches.append({"query": "", "start": "2030-01-15T00:00:00+08:00",
                              "end": "2030-01-16T00:00:00+08:00"})
    result = {"analysis": {"mode": "calendar_read", "actions": [
        {"operation": "search_events", "args": calendar.searches[0]}]},
        "read_results": {"events": [{"id": "synthetic-bluebird-event"}]},
        "response": "2030-01-15 09:00 Synthetic bluebird review"}
    assert grade_case(case, result, calendar)["passed"]
    calendar.write_attempts = 1
    assert not grade_case(case, result, calendar)["passed"]
    calendar.write_attempts = 0
    result["analysis"]["actions"][0]["args"]["start"] = "2030-01-14T00:00:00+08:00"
    assert not grade_case(case, result, calendar)["passed"]
    other = cases["cases"][2]
    denied = {"analysis": {"mode": "calendar_read", "actions": [
        {"operation": "search_events", "args": {"start": "2030-01-15T00:00:00+08:00",
                                                "end": "2030-01-16T00:00:00+08:00"}}]},
              "read_results": {"error": "Calendar is not connected for this user"},
              "response": "Calendar is not connected for this user"}
    assert grade_case(other, denied, SyntheticCalendar())["passed"]


def test_summary_keeps_null_usage_and_unmeasured_task_success():
    rows = [{"category": "todo", "route_ok": True, "operation_ok": None, "plan_ok": None,
             "failure_category": None, "model_calls": 1, "input_tokens": None,
             "output_tokens": None, "latency_ms": 10}]
    result = summarize(rows)
    assert result["route"] == {"passed": 1, "measured": 1, "rate": 1.0}
    assert result["token_usage"]["total"] is None
    assert result["task_success"] == "NOT_MEASURED"


def test_provider_outage_is_not_scored_as_model_zero_percent():
    rows = [{"category": "todo", "actual_intent": "provider_error", "route_ok": False,
             "operation_ok": False, "plan_ok": None, "failure_category": "RouteModelError",
             "model_calls": 0, "input_tokens": None, "output_tokens": None,
             "latency_ms": 100}]
    result = summarize(rows)
    assert result["status"] == "INFRA_FAILURE"
    assert result["route"]["rate"] is None
    assert result["provider_error_attempts"] == 1


def test_expected_clarification_is_not_an_experiment_failure():
    rows = [{"category": "clarification", "expected_intent": "clarify",
             "actual_intent": "clarify", "route_ok": True, "operation_ok": None,
             "plan_ok": None, "failure_category": "ValueError", "model_calls": 1,
             "input_tokens": 10, "output_tokens": 2, "latency_ms": 10}]
    assert summarize(rows)["failures"] == {}


def test_todo_missing_target_prompt_keeps_authoritative_stop_rule():
    assert "If search reports no match, stop" in SYSTEM_PROMPT
    assert "Never claim the change succeeded without a successful tool result" in SYSTEM_PROMPT


def test_compare_rejects_changed_corpus_or_attempt_set():
    row = {"case_id": "x", "repeat": 1, "route_ok": False}
    manifest = {"dataset_version": "v1", "dataset_sha256": "0" * 64,
                "split": "dev", "repeats": 1, "model_provider": "openai",
                "model_id": "synthetic", "fixed_now": "fixed", "timezone": "UTC", "evidence": "test"}
    base = {"manifest": manifest, "rows": [row], "summary": {"route": {"passed": 0}}}
    candidate = deepcopy(base)
    candidate["rows"][0]["route_ok"] = True
    assert compare(base, candidate)["route_transitions"] == {"0->1": 1}
    candidate["manifest"]["dataset_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="conditions differ"):
        compare(base, candidate)


def test_todo_comparison_rejects_changed_target_and_counts_trajectory_transition():
    row = {"case_id": "synthetic", "repeat": 1, "task_passed": True,
           "trajectory": {"passed": False}}
    manifest = {"dataset_version": "v1", "dataset_sha256": "0" * 64,
                "model_provider": "openai", "model_id": "synthetic",
                "split": "dev", "repeats": 1, "evidence": "test",
                "target": {"database": "assistant", "port": 55988}}
    base = {"manifest": manifest, "rows": [row], "summary": {"attempts": 1}}
    candidate = deepcopy(base)
    candidate["rows"][0]["trajectory"]["passed"] = True
    assert compare_todo(base, candidate)["transitions"]["trajectory"]["0->1"] == 1
    candidate["manifest"]["target"]["port"] = 5432
    with pytest.raises(ValueError, match="conditions differ"):
        compare_todo(base, candidate)


def test_cached_regrade_requires_exact_dataset_and_preserves_original():
    corpus, digest = load_dataset()
    case = corpus["cases"][0]
    row = {"case_id": case["id"], "category": case["category"],
           "actual_intent": case["expected_intent"], "route_ok": True,
           "operation_ok": None, "plan_ok": None, "failure_category": None,
           "model_calls": 1, "input_tokens": 1, "output_tokens": 1, "latency_ms": 1}
    report = {"manifest": {"dataset_sha256": digest}, "rows": [row]}
    rescored = regrade_route(report)
    assert rescored["rows"][0]["expected_intent"] == case["expected_intent"]
    assert "expected_intent" not in report["rows"][0]
    report["manifest"]["dataset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        regrade_route(report)
