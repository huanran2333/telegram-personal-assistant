"""Offline, opt-in Ground Truth/Oracle 2A checks; never runs a model or DB."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_benchmark.dataset import load_dataset
from agent_benchmark.ground_truth import GROUND_TRUTH_VERSION, view
from agent_benchmark.oracle_v2a import ORACLE_VERSION, evaluate_v2a
from agent_benchmark.pipeline import evaluate

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "experiments/official_agent_eval_rebuild_20261023"
U_TRACES = ROOT / "experiments/agent_business_baseline_20261103/raw_unseen_run1"
H_TRACES = CASES / "raw_current_holdout_v3_first_valid_20261023"


def pair(cid: str):
    u = cid.startswith("U")
    data = load_dataset(CASES / ("cases_unseen_20261103.json" if u else
                                 "cases_holdout_v3.json"))
    case = next(c for c in data["cases"] if c["case_id"] == cid)
    trace = json.loads(((U_TRACES if u else H_TRACES) /
                        f"{cid}.json").read_text(encoding="utf-8"))
    return case, trace


def goal(row, kind):
    return next(g for g in row["goals"] if g["kind"] == kind)


def test_ground_truth_view_preserves_40_cases_and_source_provenance():
    for filename, count in (("cases_holdout_v3.json", 24),
                            ("cases_unseen_20261103.json", 16)):
        data = load_dataset(CASES / filename)
        assert len(data["cases"]) == count
        for case in data["cases"]:
            old = copy.deepcopy(case)
            gt = view(case)
            assert gt["version"] == GROUND_TRUTH_VERSION
            assert gt["task"] == case["legacy"]["request"]
            assert gt["atomic_goals"] == case["legacy"]["goals"]
            assert gt["allowed_writes"] == case["legacy"]["allowed_writes"]
            assert gt["provenance"]["base"]["dataset_sha256"] == data["sha256"]
            assert case == old
    assert view(pair("U05")[0])["provenance"]["historical_interpretation"][
        "status"] == "frozen_before_first_U_run"
    assert view(pair("H3-24")[0])["provenance"]["historical_interpretation"][
        "field"] == "request"


def test_u05_prior_candidate_is_valid_but_wrong_time_and_reminder_fail():
    case, trace = pair("U05")
    result = evaluate_v2a(case, trace)
    assert result["outcome"] == "success"
    assert goal(result, "create")["verdict"] == "pass"
    assert result["oracle_version"] == ORACLE_VERSION
    assert evaluate(case, trace)["outcome"] == "success"
    before = {r["id"] for r in trace["before"]["todos"]}
    row = next(r for r in trace["after"]["todos"] if r["id"] not in before)
    for field, value in (("planned_end_at", "2026-11-10T03:00:00+00:00"),
                         ("reminder_at", "2026-11-10T01:00:00+00:00")):
        altered = copy.deepcopy(trace)
        next(r for r in altered["after"]["todos"] if r["id"] == row["id"])[field] = value
        assert evaluate_v2a(case, altered)["outcome"] == "task_failure"
    # In-memory counterexample: a matching but 60-minute candidate could fool
    # the old equality-only Oracle; the frozen 45-minute requirement must not.
    sixty = copy.deepcopy(trace)
    next(r for r in sixty["after"]["todos"] if r["id"] == row["id"])[
        "planned_end_at"] = "2026-11-10T03:00:00+00:00"
    candidate_id = next(c["id"] for m in sixty["messages"]
                        for c in m.get("tool_calls", [])
                        if c["name"] == "calendar_candidate_slots" and
                        "preferred_day_parts" not in c["args"])
    for message in sixty["messages"]:
        if message.get("tool_call_id") == candidate_id:
            slots = json.loads(message["content"])
            slots[0]["end"] = "2026-11-10T11:00:00+08:00"
            message["content"] = json.dumps(slots, ensure_ascii=False)
    assert evaluate(case, sixty)["outcome"] == "success"
    result = evaluate_v2a(case, sixty)
    assert goal(result, "create")["detail"] == "ScheduledWindowViolatesFrozenProtocol"
    assert result["outcome"] == "task_failure"


def test_u05_second_verified_candidate_is_also_legal():
    case, trace = pair("U05")
    alternate = copy.deepcopy(trace)
    before = {r["id"] for r in alternate["before"]["todos"]}
    row = next(r for r in alternate["after"]["todos"] if r["id"] not in before)
    row["planned_start_at"] = "2026-11-10T03:00:00+00:00"
    row["planned_end_at"] = "2026-11-10T03:45:00+00:00"
    call = next(c for m in alternate["messages"] for c in m.get("tool_calls", [])
                if c["name"] == "create_todo")
    call["args"]["planned_start_str"] = "2026-11-10T11:00:00+08:00"
    call["args"]["planned_end_str"] = "2026-11-10T11:45:00+08:00"
    assert evaluate_v2a(case, alternate)["outcome"] == "success"


def test_u06_full_busy_is_a_valid_read_without_event_title():
    case, trace = pair("U06")
    assert evaluate(case, trace)["outcome"] == "undetermined"
    result = evaluate_v2a(case, trace)
    assert result["outcome"] == "success"
    assert goal(result, "calendar_read")["detail"] == "FrozenFullBusyEvidence"
    partial = copy.deepcopy(trace)
    message = next(m for m in partial["messages"] if m.get("type") == "ToolMessage")
    fact = json.loads(message["content"])
    fact["busy"][0]["end"] = "2026-11-11T16:00:00+08:00"
    message["content"] = json.dumps(fact, ensure_ascii=False)
    assert goal(evaluate_v2a(case, partial), "calendar_read")["verdict"] == "undetermined"
    candidate = copy.deepcopy(trace)
    call = next(c for m in candidate["messages"] for c in m.get("tool_calls", []))
    call["name"] = "calendar_candidate_slots"
    call["args"] = {"title": "U06-整理项目演示", "duration_minutes": 240,
                    "range_start": "2026-11-11T13:00:00+08:00",
                    "range_end": "2026-11-11T17:00:00+08:00"}
    next(m for m in candidate["messages"] if m.get("type") == "ToolMessage")[
        "content"] = "[]"
    assert goal(evaluate_v2a(case, candidate), "calendar_read")[
        "detail"] == "FrozenNoCandidateEvidence"


def test_u15_reminder_only_checks_time_and_other_fields():
    case, trace = pair("U15")
    assert evaluate_v2a(case, trace)["outcome"] == "task_failure"
    corrected = copy.deepcopy(trace)
    todo = next(r for r in corrected["after"]["todos"]
                if r["owner"] == corrected["before"]["owner"])
    todo["reminder_at"] = case["expected_outcome"][0]["at"]
    todo["version"] += 1
    assert evaluate_v2a(case, corrected)["outcome"] == "success"
    wrong_time = copy.deepcopy(corrected)
    next(r for r in wrong_time["after"]["todos"] if r["id"] == todo["id"])[
        "reminder_at"] = "2026-11-12T01:30:00Z"
    assert goal(evaluate_v2a(case, wrong_time), "reminder_set")["verdict"] == "fail"
    other_edit = copy.deepcopy(corrected)
    next(r for r in other_edit["after"]["todos"] if r["id"] == todo["id"])[
        "description"] = "unrequested"
    assert "unexpected_reminder_side_effect" in evaluate_v2a(
        case, other_edit)["side_effects"]


def test_h3_24_does_not_auto_pass_unretrieved_mail_content():
    case, trace = pair("H3-24")
    baseline = evaluate_v2a(case, trace)
    assert baseline["outcome"] == "task_failure"  # Original mail read failed.
    assert baseline["reply_evidence"]["verdict"] == "undetermined"
    assert goal(baseline, "create")["verdict"] == "pass"
    owner = trace["before"]["owner"]
    created = next(r for r in trace["after"]["todos"] if r["owner"] == owner)
    fabricated = copy.deepcopy(trace)
    next(r for r in fabricated["after"]["todos"] if r["id"] == created["id"])[
        "description"] = "邮件写道：请核对依赖清单"
    assert goal(evaluate(case, fabricated), "create")["verdict"] == "pass"
    result = evaluate_v2a(case, fabricated)
    assert goal(result, "create")["detail"] == "UnretrievedMailContentInDescription"
    false_reply = copy.deepcopy(trace)
    final = next(m for m in reversed(false_reply["messages"])
                 if m.get("type") == "AIMessage" and not m.get("tool_calls"))
    final["content"] = "邮件写道：请核对依赖清单"
    assert evaluate_v2a(case, false_reply)["reply_evidence"][
        "verdict"] == "fail"
    unclear = copy.deepcopy(trace)
    next(r for r in unclear["after"]["todos"] if r["id"] == created["id"])[
        "description"] = "下周跟进"
    assert goal(evaluate_v2a(case, unclear), "create")["verdict"] == "undetermined"


@pytest.mark.parametrize("cid", ["U11", "U12"])
def test_fault_handling_is_separate_from_original_read(cid):
    case, trace = pair(cid)
    result = evaluate_v2a(case, trace)
    assert result["outcome"] == "task_failure"
    assert goal(result, "create")["verdict"] == "pass"
    assert goal(result, "calendar_failure" if cid == "U11" else
                "gmail_failure")["verdict"] == "fail"
    assert result["fault_handling_diagnostic"] == {
        "original_read": "fail", "read_error_observed": "pass",
        "independent_todo": "pass", "honest_final_reply": "undetermined"}


def test_missing_trusted_state_or_trace_is_undetermined():
    case, trace = pair("U05")
    incomplete = copy.deepcopy(trace)
    incomplete["after"].pop("todos")
    assert evaluate_v2a(case, incomplete)["outcome"] == "undetermined"
    incomplete = copy.deepcopy(trace)
    incomplete.pop("messages")
    assert evaluate_v2a(case, incomplete)["outcome"] == "undetermined"
    assert evaluate_v2a(case, trace)["ground_truth_version"] == GROUND_TRUTH_VERSION


def test_cli_opt_in_versions_results_and_refuses_cross_oracle_comparison(tmp_path):
    old, new = tmp_path / "old", tmp_path / "v2a"
    base = [sys.executable, "-m", "agent_benchmark", "--case", "U06"]
    subprocess.run([*base, "--output", str(old)], cwd=ROOT, check=True,
                   capture_output=True, text=True)
    subprocess.run([*base, "--oracle-version", ORACLE_VERSION,
                    "--output", str(new)], cwd=ROOT, check=True,
                   capture_output=True, text=True)
    old_report = json.loads((old / "results.json").read_text(encoding="utf-8"))
    new_report = json.loads((new / "results.json").read_text(encoding="utf-8"))
    assert old_report["cases"][0]["outcome"] == "undetermined"
    assert new_report["cases"][0]["outcome"] == "success"
    assert old_report["manifest"]["oracle_version"] == "legacy-replay"
    assert new_report["manifest"]["ground_truth_version"] == GROUND_TRUTH_VERSION
    attempted = subprocess.run([*base, "--oracle-version", ORACLE_VERSION,
                                "--output", str(tmp_path / "incompatible"),
                                "--compare", str(old / "results.json")],
                               cwd=ROOT, capture_output=True, text=True)
    assert attempted.returncode != 0
    assert "PairedComparisonRequiresSameOracleVersion" in attempted.stderr
