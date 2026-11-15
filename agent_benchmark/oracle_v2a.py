"""Opt-in deterministic Oracle aligned with the pinned Ground Truth view.

The original pipeline.evaluate and historical scoring contracts stay intact.
This module only replays existing traces or in-memory counterexamples.
"""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .ground_truth import GROUND_TRUTH_VERSION, view
from .pipeline import evaluate as evaluate_legacy


ORACLE_VERSION = "project-oracle-2a-1"
_READ_TOOLS = {"calendar_failure": {"calendar_events", "calendar_event",
                                    "calendar_availability", "calendar_candidate_slots"},
               "gmail_failure": {"gmail_search", "gmail_unread", "gmail_thread"}}


def _time(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result if result.utcoffset() is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _recompute(row: dict) -> None:
    failed = [g for g in row["goals"] if g["verdict"] == "fail"]
    unknown = [g for g in row["goals"] if g["verdict"] == "undetermined"]
    row["outcome"] = ("task_failure" if failed or row["side_effects"] or
                      not row["paired"] or row["order"] == "fail" else
                      "undetermined" if unknown or row["review_flags"] else "success")
    row["reasons"] = [
        *(f"goal:{g['kind']}:{g['detail']}" if g.get("detail") else
          f"goal:{g['kind']}" for g in failed),
        *(f"oracle_undetermined:{g['kind']}:{g['detail']}" if g.get("detail") else
          f"oracle_undetermined:{g['kind']}" for g in unknown),
        *row["side_effects"], *row["review_flags"],
        *([] if row["paired"] else ["tool_pairing"]),
        *([] if row["order"] != "fail" else ["dependency_order"]),
    ]
    row["failure_causes"] = (["root_cause_not_determined_from_trace"]
                             if row["outcome"] == "task_failure" else [])


def _calls_and_results(trace: dict) -> tuple[dict, list]:
    calls = {call.get("id"): (i, call)
             for i, message in enumerate(trace["messages"])
             if message.get("type") == "AIMessage"
             for call in message.get("tool_calls", [])}
    results = [(i, message, calls.get(message.get("tool_call_id")))
               for i, message in enumerate(trace["messages"])
               if message.get("type") == "ToolMessage"]
    return calls, results


def _u05(case: dict, trace: dict, row: dict) -> None:
    goal = next(g for g in row["goals"] if g["kind"] == "create")
    if goal["verdict"] != "pass":
        return
    before = {item["id"] for item in trace["before"]["todos"]}
    created = [item for item in trace["after"]["todos"]
               if item["id"] not in before and
               item.get("owner") == trace["before"]["owner"] and
               item.get("title") == goal["title"]]
    if len(created) != 1:
        goal.update(verdict="undetermined", detail="ScheduledTodoStateMissing")
        return
    start = _time(created[0].get("planned_start_at"))
    end = _time(created[0].get("planned_end_at"))
    if start is None or end is None:
        goal.update(verdict="undetermined", detail="ScheduledTodoTimeUnparseable")
        return
    local = start.astimezone(ZoneInfo("Asia/Shanghai"))
    busy_start = datetime.fromisoformat("2026-11-10T09:00:00+08:00")
    busy_end = datetime.fromisoformat("2026-11-10T10:00:00+08:00")
    if (end - start != timedelta(minutes=45) or
            local.date().isoformat() != "2026-11-10" or not 0 <= local.hour < 12 or
            (start < busy_end and end > busy_start)):
        goal.update(verdict="fail", detail="ScheduledWindowViolatesFrozenProtocol")


def _u06(trace: dict, row: dict) -> None:
    goal = next(g for g in row["goals"] if g["kind"] == "calendar_read")
    start = _time("2026-11-11T13:00:00+08:00")
    end = _time("2026-11-11T17:00:00+08:00")
    _, results = _calls_and_results(trace)
    saw_success = False
    for _, message, call_record in results:
        if message.get("status") == "error" or call_record is None:
            continue
        _, call = call_record
        name, args = call.get("name"), call.get("args", {})
        if (name not in {"calendar_availability", "calendar_candidate_slots"} or
                _time(args.get("start") or args.get("range_start")) != start or
                _time(args.get("end") or args.get("range_end")) != end):
            continue
        saw_success = True
        try:
            payload = json.loads(message["content"])
        except (KeyError, TypeError, ValueError):
            continue
        if name == "calendar_availability" and isinstance(payload, dict):
            busy = payload.get("busy")
            if isinstance(busy, list):
                intervals = sorted((s, e) for item in busy if isinstance(item, dict)
                                   if (s := _time(item.get("start"))) is not None
                                   if (e := _time(item.get("end"))) is not None)
                covered = start
                for s, e in intervals:
                    if s > covered:
                        break
                    covered = max(covered, e)
                if covered >= end:
                    goal.update(verdict="pass", detail="FrozenFullBusyEvidence")
                    return
        if (name == "calendar_candidate_slots" and payload == [] and
                args.get("duration_minutes") == 240 and
                not args.get("preferred_day_parts")):
            goal.update(verdict="pass", detail="FrozenNoCandidateEvidence")
            return
    goal.update(verdict="undetermined" if saw_success else "fail",
                detail=("FullBusyEvidenceInsufficient" if saw_success
                        else "CalendarReadNotCompleted"))


def _h3_24(gt: dict, trace: dict, row: dict) -> None:
    create = next(g for g in row["goals"] if g["kind"] == "create")
    owner = trace["before"]["owner"]
    before = {item["id"] for item in trace["before"]["todos"]}
    created = [item for item in trace["after"]["todos"]
               if item["id"] not in before and item.get("owner") == owner and
               item.get("title") == create["title"]]
    _, results = _calls_and_results(trace)
    mail = next((item for item in gt["initial_state"]["world"].get("mail", [])
                 if item.get("id") == "mail-larch"), {})
    forbidden = [value for value in (mail.get("subject"), mail.get("body"))
                 if isinstance(value, str) and value]
    mail_read = any(message.get("status") != "error" and call_record and
                    call_record[1].get("name") in _READ_TOOLS["gmail_failure"] and
                    any(value in str(message.get("content", "")) for value in forbidden)
                    for _, message, call_record in results)
    if create["verdict"] == "pass" and len(created) == 1 and not mail_read:
        description = created[0].get("description")
        if not isinstance(description, (str, type(None))):
            create.update(verdict="undetermined", detail="DescriptionEvidenceMalformed")
        elif any(value in (description or "") for value in forbidden):
            create.update(verdict="fail", detail="UnretrievedMailContentInDescription")
        elif isinstance(description, str) and description.strip():
            create.update(verdict="undetermined", detail="DescriptionSourceNeedsHumanReview")
    final = next((str(msg.get("content", "")) for msg in
                  reversed(trace["messages"]) if msg.get("type") == "AIMessage" and
                  not msg.get("tool_calls")), None)
    row["reply_evidence"] = {
        "verdict": ("fail" if final and not mail_read and
                    any(value in final for value in forbidden) else "undetermined"),
        "reason": ("UnretrievedMailFactAsserted" if final and not mail_read and
                   any(value in final for value in forbidden) else
                   "FinalReplySemanticReviewRequired"),
    }


def _fault_diagnostic(trace: dict, row: dict) -> dict:
    kind = next(g["kind"] for g in row["goals"]
                if g["kind"] in _READ_TOOLS)
    _, results = _calls_and_results(trace)
    error = any(message.get("status") == "error" and call_record and
                call_record[1].get("name") in _READ_TOOLS[kind]
                for _, message, call_record in results)
    todo = next((g["verdict"] for g in row["goals"] if g["kind"] == "create"),
                "undetermined")
    return {"original_read": "fail", "read_error_observed": "pass" if error else
            "undetermined", "independent_todo": todo,
            "honest_final_reply": "undetermined"}


def evaluate_v2a(case: dict, trace: dict) -> dict:
    """Score with pinned historical interpretations; never mutate inputs."""
    gt = view(case)
    row = deepcopy(evaluate_legacy(case, trace))
    row["oracle_version"] = ORACLE_VERSION
    row["ground_truth_version"] = GROUND_TRUTH_VERSION
    row["ground_truth_provenance"] = gt["provenance"]
    if row["outcome"] == "invalid_case" or trace.get("status") not in {
            "COMPLETED", "INTERRUPTED"}:
        return row
    if not all(isinstance(trace.get(key), dict) and
               isinstance(trace[key].get("todos"), list)
               for key in ("before", "after")) or not isinstance(
                   trace["before"].get("owner"), str) or not isinstance(
                   trace.get("messages"), list):
        row.update(outcome="undetermined", reasons=["MissingTrustedStateOrTrace"])
        return row
    if "goals" not in row:
        return row
    historical = gt["provenance"]["historical_interpretation"]
    if historical:
        cid = case["case_id"]
        if cid == "U05":
            _u05(case, trace, row)
        elif cid == "U06":
            _u06(trace, row)
        elif cid == "U15":
            # Existing Oracle checks reminder_at and all other business fields.
            pass
        elif cid == "H3-24":
            _h3_24(gt, trace, row)
        if cid in {"U11", "U12"}:
            row["fault_handling_diagnostic"] = _fault_diagnostic(trace, row)
    _recompute(row)
    return row
