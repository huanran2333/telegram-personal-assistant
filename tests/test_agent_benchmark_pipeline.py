"""Offline pipeline tests: no model, database, Telegram or Google calls."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_benchmark.dataset import InvalidDataset, load_dataset
from agent_benchmark.pipeline import evaluate, summarize
from agent_benchmark.mining import candidate_from_failure
from agent_benchmark.methodology import describe, profile

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "experiments/official_agent_eval_rebuild_20261023/cases_unseen_20261103.json"
TRACES = ROOT / "experiments/agent_business_baseline_20261103/raw_unseen_run1"


def _case_and_trace(case_id):
    dataset = load_dataset(DATASET)
    case = next(c for c in dataset["cases"] if c["case_id"] == case_id)
    trace = json.loads((TRACES / f"{case_id}.json").read_text(encoding="utf-8"))
    return case, trace


def test_dataset_adapter_preserves_frozen_fingerprint_and_metadata():
    dataset = load_dataset(DATASET)
    assert len(dataset["cases"]) == 16
    assert dataset["sha256"] == "58856ba3b47b3461a884c31556a33a7d7b8ce8c219f58b8b4c1dd3ac6d432f36"
    case = dataset["cases"][0]
    assert case["source"]["generation_model"] == "not_recorded"
    assert case["dataset_version"] == dataset["version"]
    assert case["oracle_spec"]["kind"] == "legacy_state_and_tool_results"
    older = load_dataset(ROOT / "experiments/official_agent_eval_rebuild_20261023/cases_holdout_v3.json")
    assert len(older["cases"]) == 24
    assert older["version"] != dataset["version"]


def test_duplicate_and_unreviewed_write_contract_rejected(tmp_path):
    raw = json.loads(DATASET.read_text(encoding="utf-8"))
    raw["cases"].append(dict(raw["cases"][0]))
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(InvalidDataset, match="DuplicateCaseId"):
        load_dataset(path)
    raw["cases"].pop()
    raw["cases"][0]["allowed_writes"] = []
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(InvalidDataset, match="CreateNotAllowed"):
        load_dataset(path)


def test_three_real_runner_replays_keep_partial_failure_in_denominator():
    rows = [evaluate(*_case_and_trace(case_id)) for case_id in ("U01", "U09", "U11")]
    assert [row["outcome"] for row in rows] == ["success", "success", "task_failure"]
    assert [g["verdict"] for g in rows[-1]["goals"]] == ["pass", "fail"]
    assert all(row["paired"] and row["reply_verdict"] == "manual_review_required" for row in rows)
    assert summarize(rows)["total"] == 3
    metrics = summarize(rows)["metric_denominators"]
    assert metrics["atomic_goals"] == {"passed": 4, "failed": 1,
                                       "undetermined": 0, "recorded": 5}
    assert metrics["deterministic_task"]["reply_manual_review"] == 3
    assert metrics["robustness"]["injected_fault_cases"] == 1


def test_design_profile_is_independent_of_model_execution():
    data = profile()
    assert [item["cases"] for item in data["datasets"]] == [24, 16]
    h3 = data["rows"][data["datasets"][0]["version"]]
    unseen = data["rows"][data["datasets"][1]["version"]]
    assert len(h3) == 24 and len(unseen) == 16
    assert sum(row["length"] == "9+" for row in h3 + unseen) == 0
    assert sum(row["negative"] for row in h3 + unseen) > 0
    case = next(c for c in load_dataset(DATASET)["cases"] if c["case_id"] == "U14")
    assert describe(case)["planned_steps"] == 4
    assert describe(case)["difficulty"] == "困难"


def test_wrong_case_extra_write_and_missing_tool_result_are_not_success():
    case, trace = _case_and_trace("U01")
    trace["case_id"] = "wrong"
    assert evaluate(case, trace)["outcome"] == "invalid_case"
    trace["case_id"] = "U01"
    trace["after"]["todos"].append({"id": "extra", "owner": trace["before"]["owner"],
                                      "title": "unrequested"})
    result = evaluate(case, trace)
    assert result["outcome"] == "task_failure"
    assert "unexpected_todo_create" in result["side_effects"]
    trace["messages"] = [m for m in trace["messages"] if m.get("type") != "ToolMessage"]
    assert evaluate(case, trace)["paired"] is False


def test_timeout_remains_in_denominator_and_is_not_business_failure():
    case, trace = _case_and_trace("U01")
    trace["status"] = "TIMEOUT"
    result = evaluate(case, trace)
    assert result["outcome"] == "timeout"
    assert summarize([result])["total"] == 1
    trace["status"] = "ERROR"
    trace["error_category"] = "ValueError"
    result = evaluate(case, trace)
    assert result["outcome"] == "undetermined"
    assert result["reasons"] == ["ExecutionStatus:ERROR"]


def test_remote_write_without_reconciliation_is_undetermined_not_failed():
    case, trace = _case_and_trace("U01")
    trace["calendar_writes"] = [{"operation": "create", "status": "UNKNOWN"}]
    result = evaluate(case, trace)
    assert result["outcome"] == "undetermined"
    assert result["side_effects"] == []
    assert result["review_flags"] == ["calendar_write_requires_remote_review"]


def test_calendar_scheduled_create_requires_complete_matching_window_and_no_extra_time():
    case, trace = _case_and_trace("U05")
    assert evaluate(case, trace)["outcome"] == "success"
    before_ids = {row["id"] for row in trace["before"]["todos"]}
    probes = (
        ({"planned_start_at": None, "planned_end_at": None}, "CreateMissingPlannedWindow"),
        ({"planned_end_at": "2026-11-10T03:00:00+00:00"}, "CreatePlannedWindowOutsideCandidate"),
        ({"due_date": "2026-11-10T04:00:00+00:00"}, "CreateUnexpectedDueDate"),
        ({"reminder_at": "2026-11-10T01:00:00+00:00"}, "CreateUnexpectedReminder"),
    )
    for change, detail in probes:
        modified = copy.deepcopy(trace)
        created = next(row for row in modified["after"]["todos"] if row["id"] not in before_ids)
        created.update(change)
        result = evaluate(case, modified)
        assert result["outcome"] == "task_failure"
        assert detail in result["reasons"] or f"goal:create:{detail}" in result["reasons"]
    other_target = copy.deepcopy(trace)
    for message in other_target["messages"]:
        for call in message.get("tool_calls", []):
            if call["name"] == "calendar_candidate_slots":
                call["args"]["title"] = "unrelated-todo"
    result = evaluate(case, other_target)
    assert result["outcome"] == "task_failure"
    assert "goal:create:CreateWithoutVerifiedCandidate" in result["reasons"]
    malformed = copy.deepcopy(trace)
    slot_ids = {call["id"] for message in malformed["messages"]
                for call in message.get("tool_calls", [])
                if call["name"] == "calendar_candidate_slots"}
    for message in malformed["messages"]:
        if message.get("tool_call_id") in slot_ids and message.get("status") != "error":
            message["content"] = '{"start":"2026-11-10T10:00:00+08:00"}'
    result = evaluate(case, malformed)
    assert result["outcome"] == "undetermined"
    assert "oracle_undetermined:create:CreateCandidateResultUnparseable" in result["reasons"]


def test_missing_time_ground_truth_stays_undetermined():
    case, trace = _case_and_trace("U01")
    case = copy.deepcopy(case)
    trace = copy.deepcopy(trace)
    case["expected_outcome"][0].pop("untimed")
    trace["case"] = case["legacy"]
    result = evaluate(case, trace)
    assert result["outcome"] == "undetermined"
    assert "oracle_undetermined:create:CreateTimeGroundTruthMissing" in result["reasons"]


def test_explicit_due_date_ground_truth_checks_the_field():
    case, trace = _case_and_trace("U01")
    case, trace = copy.deepcopy(case), copy.deepcopy(trace)
    goal = case["expected_outcome"][0]
    goal.pop("untimed")
    goal["due_date"] = "2026-11-10"
    trace["case"] = case["legacy"]
    created = next(row for row in trace["after"]["todos"]
                   if row["id"] not in {old["id"] for old in trace["before"]["todos"]})
    created["due_date"] = "2026-11-10"
    assert evaluate(case, trace)["outcome"] == "success"
    created["due_date"] = "2026-11-11"
    result = evaluate(case, trace)
    assert result["outcome"] == "task_failure"
    assert "goal:create:CreateExplicitTimeMismatch" in result["reasons"]
    created["due_date"] = "2026-11-10"
    created["planned_start_at"] = "2026-11-10T09:00:00+08:00"
    result = evaluate(case, trace)
    assert result["outcome"] == "undetermined"
    assert "oracle_undetermined:create:CreateUnspecifiedExtraTimeNeedsReview" in result["reasons"]


def test_h3_description_requirement_is_not_ignored():
    root = ROOT / "experiments/official_agent_eval_rebuild_20261023"
    case = next(c for c in load_dataset(root / "cases_holdout_v3.json")["cases"]
                if c["case_id"] == "H3-03")
    trace = json.loads((root / "raw_current_holdout_v3_first_valid_20261023/H3-03.json").read_text(encoding="utf-8"))
    assert evaluate(case, trace)["outcome"] == "success"
    created = next(row for row in trace["after"]["todos"]
                   if row["id"] not in {old["id"] for old in trace["before"]["todos"]})
    created["description"] = ""
    result = evaluate(case, trace)
    assert result["outcome"] == "task_failure"
    assert "goal:create:CreateDescriptionMismatch" in result["reasons"]


def test_reminder_only_goal_rejects_unrequested_description_change():
    case, trace = _case_and_trace("U15")
    modified = copy.deepcopy(trace)
    todo = next(row for row in modified["after"]["todos"]
                if row["owner"] == modified["before"]["owner"])
    todo["reminder_at"] = case["expected_outcome"][0]["at"]
    todo["version"] += 1
    assert evaluate(case, modified)["outcome"] == "success"
    todo["description"] = "unrequested edit"
    result = evaluate(case, modified)
    assert result["outcome"] == "task_failure"
    assert "unexpected_reminder_side_effect" in result["side_effects"]


def test_failure_observations_do_not_claim_unproven_root_cause():
    for cid, outcome in (("U05", "success"), ("U06", "undetermined"),
                         ("U11", "task_failure"), ("U13", "task_failure")):
        case, trace = _case_and_trace(cid)
        result = evaluate(case, trace)
        assert result["outcome"] == outcome
        assert "business_state_or_tool_selection" not in result.get("failure_causes", [])
        if cid == "U05":
            assert "tool_error_observed" in result["observations"]
            assert result["failure_causes"] == []
        if cid == "U11":
            assert "tool_error_observed" in result["observations"]
            assert result["failure_causes"] == ["root_cause_not_determined_from_trace"]
        if cid == "U13":
            assert result["failure_causes"] == ["root_cause_not_determined_from_trace"]


def test_cli_writes_reports_without_overwrite_and_mines_pending_candidates(tmp_path):
    output = tmp_path / "replay"
    cmd = [sys.executable, "-m", "agent_benchmark", "--cases", "U01,U09,U11",
           "--output", str(output), "--mine"]
    first = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    assert '"cases": 3' in first.stdout
    report = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert report["summary"]["outcomes"] == {"success": 2, "task_failure": 1}
    assert len(report["manifest"]["replay_scorer_sha256"]) == 64
    assert (output / "results.jsonl").exists() and (output / "results.csv").exists()
    assert "| todo_single | 1/1 |" in (output / "REPORT.md").read_text(encoding="utf-8")
    candidates = json.loads((output / "failure_candidates.json").read_text(encoding="utf-8"))
    assert len(candidates) == 1 and candidates[0]["status"] == "pending_human_review"
    assert candidates[0]["variant"]["split"] == "candidate"
    assert candidates[0]["variant"]["initial_state"] == {"requires_human_curation": True}
    assert subprocess.run(cmd, cwd=ROOT, capture_output=True).returncode != 0


def test_failure_variant_does_not_promote_unreviewed_ground_truth():
    case, trace = _case_and_trace("U11")
    proposal = candidate_from_failure(case, evaluate(case, trace), "synthetic-hash")
    assert proposal["variant"]["oracle_spec"]["kind"] == "pending_human_ground_truth"
    assert "U11-检查配置-变体待审核" in proposal["variant"]["user_request"]
    assert proposal["variant"]["dataset_version"] == "candidate-unreleased"


def test_same_dataset_pairing_and_category_filter(tmp_path):
    base = tmp_path / "base"
    paired = tmp_path / "paired"
    cmd = [sys.executable, "-m", "agent_benchmark", "--category", "todo_single",
           "--output", str(base)]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, check=True)
    cmd[-1] = str(paired)
    subprocess.run([*cmd, "--compare", str(base / "results.json")], cwd=ROOT,
                   capture_output=True, check=True)
    result = json.loads((paired / "results.json").read_text(encoding="utf-8"))
    assert result["manifest"]["paired_comparison"] == [
        {"case_id": "U01", "before": "success", "after": "success"}]
