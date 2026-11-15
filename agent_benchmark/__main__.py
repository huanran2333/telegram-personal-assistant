"""Replay CLI; live Judge calls require explicit opt in."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import digest, load_dataset
from .pipeline import evaluate, manifest, write_report
from .oracle_v2a import ORACLE_VERSION, evaluate_v2a
from .ground_truth import GROUND_TRUTH_VERSION
from .mining import candidate_from_failure
from .concise_report import REPORT_VERSION, write_concise_report
from .judge_evidence import build_fault_packet, has_tool_fault, openai_compatible_completion
from .unified_judge import (FUSION_VERSION as UNIFIED_FUSION_VERSION,
                            PROMPT_VERSION as UNIFIED_PROMPT_VERSION,
                            build_unified_packet, judge_unified)
from .unified_report import (REPORT_VERSION as UNIFIED_REPORT_VERSION,
                             attach_judgments, write_unified_report)
from .judge_debug import initial_record, finalize_record, save_record

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "experiments/official_agent_eval_rebuild_20261023/cases_unseen_20261103.json"
DEFAULT_TRACES = ROOT / "experiments/agent_business_baseline_20261103/raw_unseen_run1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Replay of existing production create_agent traces")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output", required=True, type=Path, help="new directory; never overwrites")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--case")
    select.add_argument("--cases", help="comma-separated case IDs for a focused smoke run")
    select.add_argument("--category")
    select.add_argument("--all", action="store_true")
    parser.add_argument("--compare", type=Path, help="previous results.json on the exact same dataset")
    parser.add_argument("--mine", action="store_true", help="write pending-review failure candidates separately")
    parser.add_argument("--oracle-version", choices=("legacy-replay", ORACLE_VERSION),
                        default="legacy-replay",
                        help="opt in to the versioned 2A Ground Truth/Oracle")
    parser.add_argument("--report-mode", choices=("standard", "concise", "unified"), default="standard",
                        help="opt in to concise or single-call unified Judge report")
    parser.add_argument("--judge-mode", choices=("off", "live"), default="off",
                        help="single-call unified Judge; default is offline")
    parser.add_argument("--judge-env-file", type=Path,
                        help="live mode Judge configuration; defaults to the project .env")
    parser.add_argument("--judge-debug-dir", type=Path,
                        help="explicit local capture under local_judge_debug/; unified live mode only")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("OutputAlreadyExists")
    debug_dir = None
    if args.judge_debug_dir:
        if args.report_mode != "unified" or args.judge_mode != "live":
            parser.error("JudgeDebugRequiresUnifiedLiveMode")
        debug_dir = args.judge_debug_dir.resolve()
        if not debug_dir.is_relative_to((ROOT / "local_judge_debug").resolve()):
            parser.error("JudgeDebugDirectoryMustBeLocalAndIgnored")
        if debug_dir.exists():
            parser.error("JudgeDebugDirectoryAlreadyExists")
        debug_dir.mkdir(parents=True)
    if args.report_mode in {"concise", "unified"} and args.oracle_version != ORACLE_VERSION:
        parser.error("SelectedReportRequiresOracle2A")
    if args.judge_mode == "live" and args.report_mode != "unified":
        parser.error("UnifiedLiveJudgeRequiresUnifiedReport")
    if args.report_mode == "unified" and args.compare:
        parser.error("JudgeReportComparisonRequiresSeparateProtocolReview")
    judge_model, complete = None, None
    if args.judge_mode == "live":
        try:
            judge_model, complete = openai_compatible_completion(env_path=args.judge_env_file)
        except ValueError as exc:
            parser.error(str(exc))
    dataset = load_dataset(args.dataset)
    chosen = set(args.cases.split(",")) if args.cases else None
    cases = [c for c in dataset["cases"] if
             (not args.case or c["case_id"] == args.case) and
             (chosen is None or c["case_id"] in chosen) and
             (not args.category or c["category"] == args.category)]
    if not cases:
        parser.error("NoMatchingCases")
    rows = []
    fault_packets = []
    unified_packets = []
    unified_judgments = {}
    for case in cases:
        path = args.traces / f"{case['case_id']}.json"
        trace = None
        if not path.is_file():
            row = {"case_id": case["case_id"], "category": case["category"],
                   "outcome": "infra_error", "reasons": ["MissingTrace"]}
        else:
            try:
                trace = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(trace, dict):
                    raise ValueError("TraceMustBeObject")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                row = {"case_id": case["case_id"], "category": case["category"],
                       "outcome": "infra_error", "reasons": [type(exc).__name__]}
                trace = None
            else:
                if trace.get("dataset_sha256", "").lower() != dataset["sha256"].lower():
                    row = {"case_id": case["case_id"], "category": case["category"],
                           "outcome": "invalid_case", "reasons": ["DatasetFingerprintMismatch"]}
                    trace = None
                else:
                    # An oracle bug must stop the run, never masquerade as infra failure.
                    row = (evaluate_v2a if args.oracle_version == ORACLE_VERSION
                           else evaluate)(case, trace)
        if args.report_mode == "concise" and trace is not None and has_tool_fault(case, trace):
            fault_packets.append(build_fault_packet(case, trace, trace_sha256=digest(path)))
        if (args.report_mode == "unified" and trace is not None and
                row.get("outcome") != "invalid_case" and isinstance(trace.get("messages"), list)):
            packet = build_unified_packet(case, trace, trace_sha256=digest(path))
            unified_packets.append(packet)
            if (complete is not None and args.judge_mode == "live" and
                    packet["trace_status"] in {"COMPLETED", "INTERRUPTED"} and
                    packet.get("final_reply") and
                    packet.get("state_changes", {}).get("evidence") != "state_missing"):
                diagnostic_count = len(getattr(complete, "diagnostics", []))
                debug_record = None
                debug_path = debug_dir / f"{case['case_id']}.json" if debug_dir else None
                def recording_complete(messages):
                    nonlocal debug_record
                    try:
                        raw = complete(messages)
                    except Exception:
                        debug_record = initial_record(
                            packet, messages, judge_model,
                            getattr(complete, "request_parameters", {}), None,
                            (complete.diagnostics[-1] if getattr(complete, "diagnostics", []) else None))
                        save_record(debug_path, debug_record)
                        raise
                    debug_record = initial_record(
                        packet, messages, judge_model,
                        getattr(complete, "request_parameters", {}), raw,
                        (complete.diagnostics[-1] if getattr(complete, "diagnostics", []) else None))
                    save_record(debug_path, debug_record)
                    return raw
                unified_judgments[case["case_id"]] = judge_unified(
                    packet, recording_complete if debug_dir else complete,
                    judge_model=judge_model)
                if debug_record is not None:
                    save_record(debug_path, finalize_record(
                        debug_record, row["outcome"], unified_judgments[case["case_id"]]))
                if len(getattr(complete, "diagnostics", [])) > diagnostic_count:
                    complete.diagnostics[-1]["case_id"] = case["case_id"]
        rows.append(row)
    run = manifest(dataset, args.traces, ROOT)
    run["oracle_version"] = args.oracle_version
    run["scoring_protocol_version"] = "legacy-replay"
    run["report_mode"] = args.report_mode
    if args.report_mode == "concise":
        run["concise_report_version"] = REPORT_VERSION
        run["fault_evidence_version"] = "fault-evidence-1"
    if args.report_mode == "unified":
        run["unified_report_version"] = UNIFIED_REPORT_VERSION
        run["unified_prompt_version"] = UNIFIED_PROMPT_VERSION
        run["unified_fusion_version"] = UNIFIED_FUSION_VERSION
        run["judge_mode"] = args.judge_mode
        run["judge_model"] = judge_model
        run["judge_calls"] = len(unified_judgments)
        run["unified_judge_sha256"] = digest(ROOT / "agent_benchmark/unified_judge.py")
        run["unified_report_sha256"] = digest(ROOT / "agent_benchmark/unified_report.py")
    if args.oracle_version == ORACLE_VERSION:
        run["ground_truth_version"] = GROUND_TRUTH_VERSION
        run["oracle_v2a_sha256"] = __import__("hashlib").sha256(
            (ROOT / "agent_benchmark/oracle_v2a.py").read_bytes()).hexdigest()
        run["ground_truth_sha256"] = __import__("hashlib").sha256(
            (ROOT / "agent_benchmark/ground_truth.py").read_bytes()).hexdigest()
    run["near_duplicate_review"] = dataset["near_duplicate_review"]
    run["trace_sha256_by_case"] = {c["case_id"]: __import__("hashlib").sha256((args.traces / f"{c['case_id']}.json").read_bytes()).hexdigest()
                                   for c in cases if (args.traces / f"{c['case_id']}.json").is_file()}
    if args.compare:
        prior = json.loads(args.compare.read_text(encoding="utf-8"))
        if prior["manifest"]["dataset_sha256"] != dataset["sha256"]:
            parser.error("PairedComparisonRequiresSameDatasetFingerprint")
        if prior["manifest"].get("oracle_version", "legacy-replay") != args.oracle_version:
            parser.error("PairedComparisonRequiresSameOracleVersion")
        if prior["manifest"].get("scoring_protocol_version", "legacy-replay") != "legacy-replay":
            parser.error("PairedComparisonRequiresSameScoringProtocol")
        old = {r["case_id"]: r for r in prior["cases"]}
        run["paired_comparison"] = [
            {"case_id": row["case_id"], "before": old[row["case_id"]]["outcome"], "after": row["outcome"]}
            for row in rows if row["case_id"] in old]
    if args.report_mode == "concise":
        write_concise_report(args.output, run, rows, fault_packets, cases)
    elif args.report_mode == "unified":
        rows = attach_judgments(rows, unified_judgments,
                                {p["case_id"]: p for p in unified_packets})
        write_unified_report(args.output, run, rows, cases, unified_packets)
        if complete is not None:
            diagnostics = getattr(complete, "diagnostics", [])
            (args.output / "judge_http_diagnostics.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in diagnostics),
                encoding="utf-8")
    else:
        write_report(args.output, run, rows)
    if args.mine:
        by_id = {case["case_id"]: case for case in cases}
        candidates = [candidate_from_failure(by_id[row["case_id"]], row,
                         run["trace_sha256_by_case"].get(row["case_id"]))
                      for row in rows if row["outcome"] != "success"]
        (args.output / "failure_candidates.json").write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(rows),
                      "outcomes": {name: sum(r["outcome"] == name for r in rows)
                                   for name in sorted({r["outcome"] for r in rows})}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
