"""One-call unified Judge tests; all model responses are local mocks."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import pytest

from agent_benchmark.dataset import load_dataset
from agent_benchmark.oracle_v2a import evaluate_v2a
from agent_benchmark.unified_judge import (PROMPT_VERSION, build_unified_packet,
                                           combined_verdict, judge_unified)
from agent_benchmark.unified_report import attach_judgments, summarize_unified
from agent_benchmark.judge_debug import initial_record, finalize_record, save_record, replay

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "experiments/official_agent_eval_rebuild_20261023"
U_TRACES = ROOT / "experiments/agent_business_baseline_20261103/raw_unseen_run1"
H_TRACES = CASES / "raw_current_holdout_v3_first_valid_20261023"


def pair(cid):
    h3 = cid.startswith("H3")
    dataset = load_dataset(CASES / ("cases_holdout_v3.json" if h3 else "cases_unseen_20261103.json"))
    case = next(c for c in dataset["cases"] if c["case_id"] == cid)
    trace = json.loads(((H_TRACES if h3 else U_TRACES) / f"{cid}.json").read_text(encoding="utf-8"))
    return case, trace


def response(packet, *, task="Pass", fault=None):
    value = {"case_id": packet["case_id"],
             "task_completion": {"verdict": task, "reason": "Mock task analysis",
                                 "evidence_refs": ["state_changes", packet["final_reply"]["ref"]]},
             "review_required": False}
    if fault is not None:
        value["fault_handling"] = {"verdict": fault, "reason": "Mock fault analysis",
                                   "evidence_refs": packet["observed_tool_error_refs"]}
    return value


def test_no_fault_needs_only_task_judgment_and_one_call():
    case, trace = pair("U01")
    packet = build_unified_packet(case, trace)
    assert packet["fault_applicability"] == "none"
    seen = []
    def complete(messages):
        seen.append(messages)
        return response(packet)  # fault field may be absent when no error
    result = judge_unified(packet, complete, judge_model="mock")
    assert len(seen) == 1
    assert result["task_completion"]["verdict"] == "Pass"
    assert result["fault_handling"]["verdict"] == "NotApplicable"


def test_corrected_fault_and_completed_task_share_one_call():
    case, trace = pair("U05")
    packet = build_unified_packet(case, trace)
    assert len(packet["observed_tool_error_refs"]) == 2
    calls = []
    def complete(messages):
        calls.append(messages)
        assert json.loads(messages[1]["content"])["events"] == packet["events"]
        return response(packet, fault="Reasonable")
    result = judge_unified(packet, complete, judge_model="mock")
    assert len(calls) == 1
    assert (result["task_completion"]["verdict"],
            result["fault_handling"]["verdict"]) == ("Pass", "Reasonable")
    assert combined_verdict(evaluate_v2a(case, trace)["outcome"], result) == "Pass"


def test_oracle_failure_still_gets_fault_analysis_but_cannot_become_pass():
    case, trace = pair("U11")
    packet = build_unified_packet(case, trace)
    calls = []
    def complete(messages):
        calls.append(messages)
        return response(packet, fault="Reasonable")
    result = judge_unified(packet, complete, judge_model="mock")
    oracle = evaluate_v2a(case, trace)
    assert oracle["outcome"] == "task_failure" and len(calls) == 1
    assert result["fault_handling"]["verdict"] == "Reasonable"
    assert combined_verdict(oracle["outcome"], result) == "Fail"
    rows = attach_judgments([oracle], {"U11": result}, {"U11": packet})
    assert summarize_unified(rows, [case])["pass_rate"]["passed"] == 0


def test_wrong_final_reply_causes_task_fail_with_independent_fault_result():
    case, trace = pair("U05")
    changed = copy.deepcopy(trace)
    changed["messages"][-1]["content"] = "I created the task at the wrong time."
    packet = build_unified_packet(case, changed)
    result = judge_unified(packet, lambda _: response(packet, task="Fail", fault="Reasonable"),
                           judge_model="mock")
    assert result["task_completion"]["verdict"] == "Fail"
    assert result["fault_handling"]["verdict"] == "Reasonable"
    assert combined_verdict("success", result) == "Fail"


def test_wrong_tool_parameter_can_fail_semantics_without_changing_oracle():
    case, trace = pair("U05")
    packet = build_unified_packet(case, trace)
    result = judge_unified(packet, lambda _: response(packet, task="Fail", fault="Reasonable"),
                           judge_model="mock")
    assert combined_verdict("success", result) == "Fail"
    assert result["fault_handling"]["verdict"] == "Reasonable"


def test_fault_verdict_never_changes_task_pass_rate():
    case, trace = pair("U05")
    packet = build_unified_packet(case, trace)
    result = judge_unified(packet, lambda _: response(packet, fault="Unreasonable"),
                           judge_model="mock")
    assert combined_verdict("success", result) == "Pass"
    assert result["fault_handling"]["verdict"] == "Unreasonable"


def test_u_and_h3_share_one_fixed_prompt_version():
    from agent_benchmark.unified_judge import prompt_messages
    u_case, u_trace = pair("U05")
    h_case, h_trace = pair("H3-21")
    u, h = build_unified_packet(u_case, u_trace), build_unified_packet(h_case, h_trace)
    assert prompt_messages(u)[0] == prompt_messages(h)[0]
    assert PROMPT_VERSION == "unified-task-fault-judge-2"


def test_state_element_refs_are_shared_with_prompt_and_reject_out_of_range():
    from agent_benchmark.unified_judge import allowed_evidence_refs, prompt_messages
    case, trace = pair("U05")
    packet = build_unified_packet(case, trace)
    allowed = allowed_evidence_refs(packet)
    sent = json.loads(prompt_messages(packet)[1]["content"])
    assert sent["allowed_evidence_refs"] == allowed
    assert "state_changes.added[0]" in allowed
    assert "state_changes.added[1]" not in allowed
    valid = response(packet, fault="Reasonable")
    valid["task_completion"]["evidence_refs"] = [packet["final_reply"]["ref"],
                                                    "state_changes.added[0]"]
    assert judge_unified(packet, lambda _: valid, judge_model="mock")["task_completion"]["verdict"] == "Pass"
    invalid = copy.deepcopy(valid)
    invalid["task_completion"]["evidence_refs"][1] = "state_changes.added[1]"
    result = judge_unified(packet, lambda _: invalid, judge_model="mock")
    assert result["task_completion"]["validation_errors"] == [
        {"code": "unknown_reference", "path": "task_completion.evidence_refs[1]"}]


def test_bad_json_bad_refs_and_missing_evidence_downgrade_safely():
    case, trace = pair("U05")
    packet = build_unified_packet(case, trace)
    bad_json = judge_unified(packet, lambda _: "not json", judge_model="mock")
    assert bad_json["task_completion"]["verdict"] == "Undetermined"
    assert bad_json["fault_handling"]["verdict"] == "Undetermined"
    def failed(_):
        raise TimeoutError("mock")
    assert judge_unified(packet, failed, judge_model="mock")["review_required"]
    bad_refs = response(packet, fault="Reasonable")
    bad_refs["fault_handling"]["evidence_refs"] = ["invented"]
    result = judge_unified(packet, lambda _: bad_refs, judge_model="mock")
    assert result["task_completion"]["verdict"] == "Pass"
    assert result["fault_handling"]["verdict"] == "Undetermined"
    assert result["fault_handling"]["validation_errors"] == [
        {"code": "unknown_reference", "path": "fault_handling.evidence_refs[0]"}]
    missing = {**packet, "final_reply": None}
    assert judge_unified(missing, lambda _: pytest.fail("no call"),
                         judge_model="mock")["task_completion"]["verdict"] == "Undetermined"


@pytest.mark.parametrize(("change", "expected"), [
    (lambda value: value["task_completion"].pop("verdict"),
     {"code": "missing_field", "path": "task_completion.verdict"}),
    (lambda value: value["task_completion"].update(verdict="Done"),
     {"code": "invalid_enum", "path": "task_completion.verdict"}),
    (lambda value: value["task_completion"].update(reason=42),
     {"code": "invalid_type", "path": "task_completion.reason"}),
    (lambda value: value["task_completion"].update(evidence_refs=["invented"]),
     {"code": "unknown_reference", "path": "task_completion.evidence_refs[0]"}),
    (lambda value: value["task_completion"].update(evidence_refs=["state_changes"]),
     {"code": "missing_reply_reference", "path": "task_completion.evidence_refs"}),
    (lambda value: value["task_completion"].update(evidence_refs=["messages[9]"]),
     {"code": "missing_execution_reference", "path": "task_completion.evidence_refs"}),
])
def test_task_validation_locates_field_without_discarding_fault(change, expected):
    case, trace = pair("U05")
    packet = build_unified_packet(case, trace)
    output = response(packet, fault="Reasonable")
    change(output)
    result = judge_unified(packet, lambda _: output, judge_model="mock")
    assert result["task_completion"]["verdict"] == "Undetermined"
    assert expected in result["task_completion"]["validation_errors"]
    assert result["fault_handling"]["verdict"] == "Reasonable"
    assert result["fault_handling"]["validation_errors"] == []
    assert result["review_required"]
    assert result["response_debug"]["sha256"]
    assert "Mock task analysis" not in str(result["response_debug"])


def test_fault_validation_error_does_not_discard_valid_task():
    case, trace = pair("U05")
    packet = build_unified_packet(case, trace)
    output = response(packet, fault="Reasonable")
    output["fault_handling"]["evidence_refs"] = ["invented"]
    result = judge_unified(packet, lambda _: output, judge_model="mock")
    assert result["task_completion"]["verdict"] == "Pass"
    assert result["fault_handling"]["verdict"] == "Undetermined"


def test_restricted_debug_record_and_reason_do_not_store_secret_text():
    case, trace = pair("U05")
    packet = build_unified_packet(case, trace)
    output = response(packet, fault="Reasonable")
    output["task_completion"]["reason"] = "Bearer test-secret"
    result = judge_unified(packet, lambda _: output, judge_model="mock")
    assert "test-secret" not in json.dumps(result, ensure_ascii=False)
    assert result["response_debug"]["field_shapes"]["task_completion"]["reason"] == "str"


def test_full_debug_capture_and_offline_replay_without_model(tmp_path):
    from agent_benchmark.unified_judge import prompt_messages, allowed_evidence_refs
    case, trace = pair("U05")
    packet = build_unified_packet(case, trace)
    raw_output = response(packet, fault="Reasonable")
    raw_output["task_completion"]["evidence_refs"] = ["missing-ref"]
    raw = json.dumps(raw_output, ensure_ascii=False)
    record = initial_record(packet, prompt_messages(packet), "mock-model",
                            {"temperature": 0, "max_tokens": 8192}, raw,
                            {"http_status": 200, "usage": {"total_tokens": 10}})
    assert record["messages_sent_redacted"][0]["content"]
    assert record["messages_sent_redacted"][1]["content"]
    assert record["allowed_evidence_refs"] == allowed_evidence_refs(packet)
    assert "messages[8]" in record["allowed_evidence_refs"]
    assert record["raw_message_content_redacted"] == raw
    judgment = judge_unified(packet, lambda _: raw, judge_model="mock-model")
    saved = finalize_record(record, "success", judgment)
    path = tmp_path / "record.json"
    save_record(path, saved)
    repeated = replay(path)
    assert repeated["validation"]["task_completion"]["validation_errors"] == [
        {"code": "unknown_reference", "path": "task_completion.evidence_refs[0]"}]
    assert repeated["validation"]["fault_handling"]["validated_verdict"] == "Reasonable"
    assert repeated["fusion"]["combined_verdict"] == "Undetermined"
    assert repeated["parsed_output_redacted"]["task_completion"]["evidence_refs"] == ["missing-ref"]


def test_debug_record_redacts_credentials_and_personal_identifiers():
    from agent_benchmark.judge_debug import redact
    data = {"Authorization": "Bearer secret", "prompt_tokens": 12,
            "message": "alice@example.com 13812345678 sk-abcdefghijklmnop"}
    safe = redact(data)
    assert safe["Authorization"] == "[REDACTED]"
    assert safe["prompt_tokens"] == 12
    assert "alice@example.com" not in safe["message"]
    assert "13812345678" not in safe["message"]
    assert "sk-abcdefghijklmnop" not in safe["message"]


def test_cli_explicit_debug_capture_can_replay_without_network(monkeypatch, tmp_path):
    from agent_benchmark import __main__ as cli
    def complete(messages):
        packet = json.loads(messages[1]["content"])
        complete.diagnostics.append({"http_status": 200, "elapsed_ms": 10,
                                     "response_valid": True, "usage": {"total_tokens": 20}})
        output = response(packet, fault="Reasonable")
        output["task_completion"]["evidence_refs"] = ["not-in-packet"]
        return json.dumps(output)
    complete.diagnostics = []
    complete.request_parameters = {"temperature": 0, "max_tokens": 8192}
    monkeypatch.setattr(cli, "openai_compatible_completion", lambda **_: ("mock-model", complete))
    ignored_root = ROOT / "local_judge_debug"
    ignored_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ignored_root) as directory:
        debug_dir = Path(directory) / "capture"
        report_dir = tmp_path / "report"
        monkeypatch.setattr(sys, "argv", ["agent_benchmark", "--oracle-version",
                         "project-oracle-2a-1", "--report-mode", "unified",
                         "--judge-mode", "live", "--case", "U05", "--output",
                         str(report_dir), "--judge-debug-dir", str(debug_dir)])
        cli.main()
        saved = json.loads((debug_dir / "U05.json").read_text(encoding="utf-8"))
        assert saved["messages_sent_redacted"][0]["content"]
        assert saved["http_diagnostic"]["http_status"] == 200
        assert saved["validation"]["task_completion"]["validation_errors"] == [
            {"code": "unknown_reference", "path": "task_completion.evidence_refs[0]"}]
        assert replay(debug_dir / "U05.json")["fusion"] == saved["fusion"]
        assert len(complete.diagnostics) == 1


def test_incomplete_trace_does_not_prove_absence_of_fault():
    case, trace = pair("U01")
    trace = copy.deepcopy(trace)
    trace["status"] = "INTERRUPTED"
    packet = build_unified_packet(case, trace)
    assert packet["fault_applicability"] == "undetermined"
    assert packet["final_reply"] is None
    result = judge_unified(packet, lambda _: pytest.fail("no call"), judge_model="mock")
    assert result["fault_handling"]["verdict"] == "Undetermined"


def test_model_cannot_invent_fault_in_complete_fault_free_trace():
    case, trace = pair("U01")
    packet = build_unified_packet(case, trace)
    result = judge_unified(packet, lambda _: response(packet, fault="Reasonable"),
                           judge_model="mock")
    assert result["fault_handling"]["verdict"] == "NotApplicable"
    assert result["review_required"]


def test_missing_state_snapshot_cannot_be_inferred_as_empty_state():
    case, trace = pair("U05")
    trace = copy.deepcopy(trace)
    trace.pop("after")
    packet = build_unified_packet(case, trace)
    assert packet["state_changes"] == {"evidence": "state_missing"}
    result = judge_unified(packet, lambda _: pytest.fail("no call"), judge_model="mock")
    assert result["task_completion"]["verdict"] == "Undetermined"


def test_cli_live_mock_invokes_one_judge_per_case_including_oracle_fail(monkeypatch, tmp_path):
    from agent_benchmark import __main__ as cli
    calls = []
    def complete(messages):
        packet = json.loads(messages[1]["content"])
        calls.append(packet["case_id"])
        complete.diagnostics.append({"http_status": 200, "response_valid": True,
                                     "judge_json_valid": True, "elapsed_ms": 12.0,
                                     "exception_type": None})
        return response(packet, fault="Reasonable")
    complete.diagnostics = []
    monkeypatch.setattr(cli, "openai_compatible_completion",
                        lambda **_: ("mock-model", complete))
    out = tmp_path / "unified"
    monkeypatch.setattr(sys, "argv", ["agent_benchmark", "--oracle-version", "project-oracle-2a-1",
                    "--report-mode", "unified", "--judge-mode", "live",
                    "--cases", "U05,U11", "--output", str(out)])
    cli.main()
    assert sorted(calls) == ["U05", "U11"]
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert data["summary"]["pass_rate"]["passed"] == 1
    assert data["summary"]["pass_rate"]["failed"] == 1
    assert data["summary"]["judge_calls"] == 2
    diagnostics = [json.loads(line) for line in
                   (out / "judge_http_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [item["case_id"] for item in diagnostics] == ["U05", "U11"]
    assert all(item["http_status"] == 200 for item in diagnostics)
    assert [r["unified_judge"]["fault_handling"]["verdict"] for r in data["cases"]] == ["Reasonable", "Reasonable"]
    assert "ENV" not in (out / "REPORT.md").read_text(encoding="utf-8")
