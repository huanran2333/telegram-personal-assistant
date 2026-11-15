"""Read-only, provenance-bound Ground Truth view for the frozen H3/U cases.

This is a later interpretation of already frozen data, not a rewrite of the
original case files or of their historical human verdicts.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .dataset import digest


GROUND_TRUTH_VERSION = "project-gt-2a-1"
_H3_SHA = "2548a702f6151d88089b57e5538629e8a6759b33459e2005f5203e66c3fedfb1"
_U_SHA = "58856ba3b47b3461a884c31556a33a7d7b8ce8c219f58b8b4c1dd3ac6d432f36"
_PROTOCOL_SHA = "ef17036e57a06143402aaef5898c4a600b2e2c335d800709f575e59f70b5dba8"
_PROTOCOL = (Path(__file__).resolve().parents[1] /
             "experiments/agent_business_baseline_20261103/FROZEN_PROTOCOL.md")


def _frozen_u(case: dict) -> bool:
    source = case.get("source", {})
    if (source.get("file") != "cases_unseen_20261103.json" or
            source.get("dataset_sha256") != _U_SHA or
            case.get("dataset_version") != "business-unseen-v1"):
        return False
    if not _PROTOCOL.is_file() or digest(_PROTOCOL) != _PROTOCOL_SHA:
        raise ValueError("FrozenProtocolFingerprintMismatch")
    return True


def _frozen_h3(case: dict) -> bool:
    source = case.get("source", {})
    return (source.get("file") == "cases_holdout_v3.json" and
            source.get("dataset_sha256") == _H3_SHA and
            case.get("dataset_version") ==
            "current-create-agent-stratified-holdout-v3")


def view(case: dict) -> dict:
    """Expose task, state, goals, constraints, allowed effects and review limits.

    Historic additions are applied only to pinned source fingerprints. The
    returned view is detached from the immutable legacy case.
    """
    source = case.get("source", {})
    old = case["legacy"]
    gt = {
        "version": GROUND_TRUTH_VERSION,
        "case_id": case["case_id"],
        "task": case["user_request"],
        "initial_state": deepcopy(case["initial_state"]),
        "atomic_goals": deepcopy(case["expected_outcome"]),
        "allowed_writes": deepcopy(case["available_tools_or_policy"]["allowed_writes"]),
        "required_order": deepcopy(case["required_constraints"].get("order")),
        "decisions": deepcopy(case["required_constraints"].get("decisions", {})),
        "acceptable_outcomes": [],
        "reply_requirements": [],
        "review_required": ["final_reply_semantics"],
        "provenance": {
            "base": {"file": source.get("file"),
                     "dataset_sha256": source.get("dataset_sha256"),
                     "fields": ["request", "seed", "world", "goals",
                                "allowed_writes", "decisions", "order",
                                "time_reference"],
                     "time_reference_is_clock_proof": False},
            "historical_interpretation": None,
        },
    }
    cid = case["case_id"]
    if _frozen_u(case):
        if cid in {"U05", "U06", "U11", "U12", "U15"}:
            gt["provenance"]["historical_interpretation"] = {
                "file": str(_PROTOCOL.relative_to(Path(__file__).resolve().parents[1])),
                "sha256": _PROTOCOL_SHA,
                "case_id": cid,
                "status": "frozen_before_first_U_run",
            }
        if cid == "U05":
            gt["acceptable_outcomes"] = [
                {"kind": "scheduled_todo", "source": "prior_successful_candidate",
                 "duration_minutes": 45, "local_date": "2026-11-10",
                 "day_part": "morning", "busy": ["09:00", "10:00"],
                 "reminder_at": None}]
            gt["reply_requirements"] = ["describe_actual_scheduled_window"]
        elif cid == "U06":
            gt["acceptable_outcomes"] = [
                {"kind": "no_four_hour_slot", "range_start": "2026-11-11T13:00:00+08:00",
                 "range_end": "2026-11-11T17:00:00+08:00",
                 "evidence_tools": ["calendar_availability", "calendar_candidate_slots"],
                 "allowed_scheduled_writes": 0}]
            gt["reply_requirements"] = ["explain_no_reliable_slot"]
        elif cid == "U15":
            gt["acceptable_outcomes"] = [
                {"kind": "reminder_only", "at": "2026-11-12T00:30:00Z",
                 "other_business_fields_unchanged": True}]
            gt["reply_requirements"] = ["report_reminder_setting_not_delivery"]
        elif cid in {"U11", "U12"}:
            gt["acceptable_outcomes"] = [
                {"kind": "partial_failure", "original_read_completed": False,
                 "independent_todo_required": True}]
            gt["reply_requirements"] = [
                "separate_committed_todo_from_unavailable_external_read"]
    if _frozen_h3(case) and cid == "H3-24":
        # This prohibition is in the original request, not in the JSON goals.
        gt["provenance"]["historical_interpretation"] = {
            "file": source["file"], "dataset_sha256": _H3_SHA,
            "field": "request", "status": "original_frozen_request"}
        gt["acceptable_outcomes"] = [
            {"kind": "no_unretrieved_mail_content_in_todo_description"}]
        gt["reply_requirements"] = ["acknowledge_mail_read_failure",
                                    "do_not_invent_mail_content"]
        gt["review_required"].append("description_and_reply_source_semantics")
    if old.get("manual_expectation") and not gt["provenance"]["historical_interpretation"]:
        gt["review_required"].append("legacy_manual_expectation")
    return gt
