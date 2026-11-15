"""Small final report for the single-call task/fault Judge protocol."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .concise_report import summarize_concise
from .unified_judge import FUSION_VERSION, PROMPT_VERSION, combined_verdict


REPORT_VERSION = "unified-judge-report-1"


def attach_judgments(rows: list[dict], judgments: dict[str, dict], packets: dict[str, dict]) -> list[dict]:
    from .unified_judge import pending_result
    result = []
    for row in rows:
        cid = row["case_id"]
        packet = packets.get(cid)
        judgment = judgments.get(cid) or (pending_result(packet, "Judge not run") if packet else None)
        verdict = (combined_verdict(row["outcome"], judgment) if judgment else
                   "Fail" if row["outcome"] == "task_failure" else "Undetermined")
        result.append({**row, "unified_judge": judgment,
                       "tool_error_observed": bool(packet and packet["observed_tool_error_refs"]),
                       "judge_status": "evaluated" if cid in judgments else "not_run",
                       "combined_verdict": verdict})
    return result


def summarize_unified(rows: list[dict], cases: list[dict]) -> dict:
    base = summarize_concise(rows, [], cases)
    valid = [r for r in rows if r.get("outcome") != "invalid_case"]
    counts = Counter(r["combined_verdict"] for r in valid)
    faults = [r for r in valid if r.get("tool_error_observed")]
    return {"selected_cases": len(rows),
            "pass_rate": {"passed": counts["Pass"], "denominator": len(valid),
                          "percent": round(100 * counts["Pass"] / len(valid), 2) if valid else None,
                          "failed": counts["Fail"], "undetermined": counts["Undetermined"],
                          "invalid_excluded": len(rows) - len(valid)},
            "partial_score": base["partial_score"],
            "efficiency": {k: base["efficiency"][k] for k in
                           ("tool_calls", "input_tokens", "output_tokens", "total_tokens", "elapsed_ms")},
            "fault_cases": len(faults),
            "judge_calls": sum(r["judge_status"] == "evaluated" for r in rows),
            "review_required": sum(bool((r.get("unified_judge") or {}).get("review_required"))
                                   for r in valid)}


def write_unified_report(output: Path, manifest: dict, rows: list[dict], cases: list[dict],
                         packets: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    summary = summarize_unified(rows, cases)
    (output / "results.json").write_text(json.dumps({"manifest": manifest, "summary": summary, "cases": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (output / "judge_inputs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in packets), encoding="utf-8")
    (output / "judge_results.jsonl").write_text("".join(json.dumps({
        "case_id": r["case_id"], "oracle_outcome": r["outcome"],
        "combined_verdict": r["combined_verdict"], "judge_status": r["judge_status"],
        "judge": r["unified_judge"]}, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ("case_id", "oracle_outcome", "task_verdict", "fault_verdict",
                  "combined_verdict", "atomic_pass", "atomic_total", "tool_calls",
                  "input_tokens", "output_tokens", "elapsed_ms")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            judge = row.get("unified_judge") or {}
            usage = row.get("usage", {})
            writer.writerow({"case_id": row["case_id"], "oracle_outcome": row["outcome"],
                             "task_verdict": judge.get("task_completion", {}).get("verdict"),
                             "fault_verdict": judge.get("fault_handling", {}).get("verdict"),
                             "combined_verdict": row["combined_verdict"],
                             "atomic_pass": sum(g.get("verdict") == "pass" for g in row.get("goals", [])),
                             "atomic_total": len(row.get("goals", [])),
                             "tool_calls": row.get("tool_calls"),
                             "input_tokens": usage.get("input_tokens"),
                             "output_tokens": usage.get("output_tokens"),
                             "elapsed_ms": row.get("elapsed_ms")})
    n = summary["selected_cases"]
    def average(name: str, unit: str = "") -> str:
        item = summary["efficiency"][name]
        return f"{item['mean'] if item['mean'] is not None else '未记录'}{unit}（{item['recorded_cases']}/{n} 有数据）"
    rate, partial = summary["pass_rate"], summary["partial_score"]
    source_models = sorted({str(row["model_id"]) for row in rows if row.get("model_id")})
    lines = ["# Agent Evaluation：统一 Judge 精简报告", "",
             "历史生产 Agent Trace 的离线 Replay；没有重新运行被测 Agent。Judge 未调用时，待定不代表 Agent 实际失败。", "",
             f"数据集 {manifest['dataset_version']}；源 Trace `{manifest['trace_directory']}`；源模型 {', '.join(source_models) or '未记录'}；Oracle {manifest['oracle_version']}；统一 Prompt {PROMPT_VERSION}；融合协议 {FUSION_VERSION}。", "",
             f"**综合 Pass Rate**：{rate['passed']}/{rate['denominator']} = {rate['percent']}%；Fail {rate['failed']}、Undetermined {rate['undetermined']}；无效案例排除 {rate['invalid_excluded']}。",
             f"**Partial Score**：{partial['passed']}/{partial['denominator']} = {partial['percent']}%；失败 {partial['failed']}、待定 {partial['undetermined']}、证据缺失未评分 {partial['unscored']}。", "",
             f"平均工具调用 {average('tool_calls')}；平均输入 Token {average('input_tokens')}、输出 Token {average('output_tokens')}、合计 Token {average('total_tokens')}；平均执行耗时 {average('elapsed_ms', ' ms')}。", "",
             f"统一 Judge 实际调用 {summary['judge_calls']} 次；需人工复核 {summary['review_required']} 例。", "",
             "## 有工具错误的案例", "",
             "| Case | 故障处理判断 | 理由与证据 |",
             "| --- | --- | --- |"]
    for row in rows:
        judge = row.get("unified_judge")
        if not judge or not row.get("tool_error_observed"):
            continue
        fault = judge["fault_handling"]
        reason = str(fault["reason"]).replace("|", "\\|").replace("\r", " ").replace("\n", " ")
        lines.append(f"| {row['case_id']} | {fault['verdict']} | {reason}（{', '.join(fault['evidence_refs']) or '待核实'}） |")
    lines += ["", "工具故障处理合理性不改变原始业务任务的完成判定。完整逐案证据与裁定见 `judge_inputs.jsonl`、`judge_results.jsonl` 和 `results.json`。模型输出须经人工样本校准；未确认的源工具 Schema 不用于历史判定。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
