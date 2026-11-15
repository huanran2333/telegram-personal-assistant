"""Add behavior diagnostics to frozen meeting experiment records without changing old checks.

This grader reads synthetic traces only. It never replays an Agent or modifies a
business database. The original `checks` and `all_checks` remain untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path


TIME_FIELDS = ("due_date_str", "planned_start_str", "planned_end_str", "reminder_at_str")
STORED_TIME_FIELDS = ("due_date", "planned_start_at", "planned_end_at", "reminder_at")


def _stored_time(value) -> bool:
    return value not in (None, "", "None")


def _same_instant(actual, expected: str) -> bool:
    try:
        return datetime.fromisoformat(actual) == datetime.fromisoformat(expected)
    except (TypeError, ValueError):
        return False


def grade_row(row: dict, case: dict) -> dict:
    events = row.get("tool_events") or []
    creates = [event for event in events if event.get("name") == "create_todo"]
    timed_writes = [event for event in events if event.get("name") in
                    {"create_todo", "update_todo", "set_reminder"}]
    created = [item for item in row.get("todo_states") or []
               if item.get("title") == "整理会议资料"]
    proposals = [event.get("result", {}).get("approval_proposal") for event in events
                 if event.get("name") == "propose_todo_delete" and
                 isinstance(event.get("result"), dict)]
    delete_attempts = [event for event in events if event.get("name") == "propose_todo_delete"]
    targets = [(item.get("operation"), item.get("args", {}).get("todo_id"))
               for item in proposals if isinstance(item, dict)]
    duplicate_generated = len(targets) != len(set(targets))
    checks = {}
    if case.get("approval"):
        checks["one_approval_after_user_reject"] = row.get("approval_count") == 1
        # These cases request exactly one deletion. A repeated tool call is a
        # model-behavior error even when the server blocks a second proposal.
        checks["no_repeated_delete_tool_call"] = len(delete_attempts) == 1
        checks["no_duplicate_generated_delete_proposal"] = not duplicate_generated
        checks["rejected_todo_remains"] = "Synthetic disposable task" in (
            row.get("todo_titles") or [])
    policy = case.get("todo_time_policy")
    if policy == "none":
        checks["no_unrequested_time_arguments"] = bool(creates) and all(
            not any(event.get("args", {}).get(field) for field in TIME_FIELDS)
            for event in timed_writes)
        checks["no_cross_task_time_pollution"] = checks["no_unrequested_time_arguments"]
        checks["no_unrequested_time_persisted"] = len(created) == 1 and all(
            not _stored_time(created[0].get(field)) for field in STORED_TIME_FIELDS)
    elif policy == "exact":
        checks["requested_time_persisted"] = (len(created) == 1 and
            _same_instant(created[0].get("planned_start_at"), case["todo_start_utc"]) and
            _same_instant(created[0].get("planned_end_at"), case["todo_end_utc"]))
    elif policy == "schedule_requested":
        checks["requested_schedule_persisted"] = len(created) == 1 and all(
            _stored_time(created[0].get(field)) for field in
            ("planned_start_at", "planned_end_at"))
        checks["availability_consulted"] = any(event.get("name") in
                                               {"calendar_availability", "calendar_candidate_slots"}
                                               for event in events)
    return {
        "case_id": row.get("case_id"), "repeat": row.get("repeat"),
        "status": row.get("status"), "original_checks": row.get("checks", {}),
        "original_all_checks": row.get("all_checks"), "behavior_checks": checks,
        "behavior_all_checks": all(checks.values()) if checks else None,
        "diagnostics": {
            "model_generated_duplicate_delete_proposal": duplicate_generated,
            "delete_tool_attempt_count": len(delete_attempts),
            "delete_proposal_targets": targets,
            "todo_create_arguments": [event.get("args", {}) for event in creates],
            "todo_timed_write_arguments": [event.get("args", {}) for event in timed_writes],
            "todo_create_stored": created,
            "approval_count": row.get("approval_count"),
            "reply": row.get("final_reply"),
            "provider_input_tokens": row.get("provider_input_tokens"),
            "provider_output_tokens": row.get("provider_output_tokens"),
            "model_calls": len(row.get("model_calls") or []),
            "elapsed_ms": row.get("elapsed_ms"),
        },
    }


def grade_run(results: dict, dataset: dict) -> dict:
    cases = {item["id"]: item for item in dataset["cases"]}
    graded = [grade_row(row, cases[row["case_id"]]) for row in results["rows"]]
    keys = sorted({key for row in graded for key in row["behavior_checks"]})
    return {"source_manifest": results.get("manifest"),
            "behavior_grader_version": "todo-consent-and-time-v3",
            "rows": graded,
            "summary": {key: {"passed": sum(row["behavior_checks"].get(key) is True for row in graded),
                              "denominator": sum(key in row["behavior_checks"] for row in graded)}
                        for key in keys}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("behavior_cases_v2.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("RefusingToOverwriteBehaviorGrade")
    payload = grade_run(json.loads(args.results.read_text(encoding="utf-8")),
                        json.loads(args.dataset.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
