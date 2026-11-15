"""Offline concise replay and shared evidence packet tests."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from agent_benchmark.concise_report import summarize_concise
from agent_benchmark.dataset import digest, load_dataset
from agent_benchmark.judge_evidence import build_fault_packet, has_tool_fault
from agent_benchmark.oracle_v2a import evaluate_v2a
from agent_benchmark.pipeline import summarize

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "experiments/official_agent_eval_rebuild_20261023"
DATA = CASES / "cases_unseen_20261103.json"
TRACES = ROOT / "experiments/agent_business_baseline_20261103/raw_unseen_run1"


def _pair(cid):
    case = next(c for c in load_dataset(DATA)["cases"] if c["case_id"] == cid)
    path = TRACES / f"{cid}.json"
    return case, json.loads(path.read_text(encoding="utf-8")), digest(path)


def test_concise_counts_match_existing_oracle_and_keep_unscored_in_denominator():
    cases = load_dataset(DATA)["cases"]
    rows = [evaluate_v2a(case, json.loads((TRACES / f"{case['case_id']}.json").read_text(encoding="utf-8")))
            for case in cases]
    summary = summarize_concise(rows, [], cases)
    old = summarize(rows)
    assert summary["pass_rate"] == {"passed": 12, "denominator": 16,
                                    "percent": 75.0, "other_outcomes": {"task_failure": 4}}
    assert summary["partial_score"]["passed"] == old["metric_denominators"]["atomic_goals"]["passed"] == 20
    assert summary["partial_score"]["denominator"] == 24
    assert summary["efficiency"]["tool_calls"]["mean"] == round(old["mean_tool_calls"], 2)
    assert summary["efficiency"]["input_tokens"]["mean"] == round(old["mean_input_tokens"], 2)
    assert summary["efficiency"]["output_tokens"]["mean"] == round(old["mean_output_tokens"], 2)
    assert summary["efficiency"]["elapsed_ms"]["mean"] == round(old["mean_elapsed_ms"], 2)
    missing = summarize_concise([{"case_id": "U01", "outcome": "infra_error"}], [], cases[:1])
    assert missing["pass_rate"]["denominator"] == 1
    assert missing["partial_score"]["denominator"] == len(cases[0]["expected_outcome"])
    assert missing["partial_score"]["unscored"] == len(cases[0]["expected_outcome"])


def test_h3_concise_counts_match_existing_oracle():
    dataset = load_dataset(CASES / "cases_holdout_v3.json")
    traces = CASES / "raw_current_holdout_v3_first_valid_20261023"
    rows = [evaluate_v2a(case, json.loads((traces / f"{case['case_id']}.json").read_text(encoding="utf-8")))
            for case in dataset["cases"]]
    summary = summarize_concise(rows, [], dataset["cases"])
    old = summarize(rows)
    assert summary["pass_rate"]["passed"] == old["outcomes"]["success"] == 18
    assert summary["partial_score"]["passed"] == old["metric_denominators"]["atomic_goals"]["passed"] == 32
    assert summary["partial_score"]["denominator"] == 41
    assert summary["efficiency"]["total_tokens"]["recorded_cases"] == 24


def test_fault_packet_preserves_call_result_order_and_excludes_intermediate_reasoning():
    case, trace, sha = _pair("U05")
    assert has_tool_fault(case, trace)
    trace["messages"][1]["content"] = "PRIVATE INTERNAL REASONING"
    packet = build_fault_packet(case, trace, trace_sha256=sha)
    assert packet["source_trace_sha256"] == sha
    assert [e["kind"] for e in packet["events"]] == ["tool_call", "tool_result"] * 4
    assert [e["status"] for e in packet["events"] if e["kind"] == "tool_result"] == ["error", "error", "success", "success"]
    for call, result in zip(packet["events"][::2], packet["events"][1::2]):
        assert result["call_id"] == call["call_id"]
        assert result["call_ref"] == call["ref"]
    assert packet["state_changes"]["added"][0]["scope"] == "request_owner"
    assert "PRIVATE INTERNAL REASONING" not in json.dumps(packet)
    assert packet["final_reply"]


def test_preset_fault_and_independent_todo_are_kept_separate():
    for cid in ("U11", "U12"):
        case, trace, sha = _pair(cid)
        packet = build_fault_packet(case, trace, trace_sha256=sha)
        assert packet["preset_faults"]
        assert any(e["kind"] == "tool_result" and e["status"] == "error" for e in packet["events"])
        assert any(t["title"].startswith(cid) for t in packet["state_changes"]["added"])
        assert packet["final_reply"] is not None




def test_duplicate_call_id_is_marked_ambiguous_instead_of_mispaired():
    case, trace, _ = _pair("U05")
    trace = copy.deepcopy(trace)
    repeated_id = trace["messages"][1]["tool_calls"][0]["id"]
    trace["messages"][3]["tool_calls"][0]["id"] = repeated_id
    trace["messages"][4]["tool_call_id"] = repeated_id
    packet = build_fault_packet(case, trace)
    result = packet["events"][3]
    assert result["call_ref"] is None
    assert result["pairing_issue"] == "duplicate_call_id"


def test_cli_concise_replay_is_separate_and_does_not_show_four_dimensions(tmp_path):
    out = tmp_path / "u"
    command = [sys.executable, "-m", "agent_benchmark", "--oracle-version", "project-oracle-2a-1",
               "--report-mode", "concise", "--case", "U05", "--output", str(out)]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "Pass Rate" in report and "Partial Score" in report
    assert "严格任务完成" not in report and "COMMUNICATE" not in report
    assert len((out / "fault_judge_inputs.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert json.loads((out / "results.json").read_text(encoding="utf-8"))["manifest"]["fault_evidence_version"] == "fault-evidence-1"
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert second.returncode != 0  # never overwrite historical output
