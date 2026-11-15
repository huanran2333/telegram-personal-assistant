"""Opt-in concise reporting for versioned deterministic Oracle replay."""
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path



REPORT_VERSION = "concise-report-1"


def _metric(values: list[float]) -> dict:
    return {"mean": round(statistics.mean(values), 2) if values else None,
            "recorded_cases": len(values)}


def summarize_concise(rows: list[dict], fault_packets: list[dict], cases: list[dict]) -> dict:
    goals = [g for row in rows for g in row.get("goals", [])]
    applicable_goals = sum(len(case["expected_outcome"]) for case in cases)
    total = len(rows)
    passed = sum(r.get("outcome") == "success" for r in rows)
    goal_passed = sum(g.get("verdict") == "pass" for g in goals)
    def values(key):
        return [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r[key], bool)]
    def tokens(key):
        return [r["usage"][key] for r in rows
                if isinstance(r.get("usage", {}).get(key), int) and not isinstance(r["usage"][key], bool)]
    both = [r for r in rows if all(isinstance(r.get("usage", {}).get(k), int)
                                  for k in ("input_tokens", "output_tokens"))]
    return {"selected_cases": total,
            "pass_rate": {"passed": passed, "denominator": total,
                          "percent": round(100 * passed / total, 2) if total else None,
                          "other_outcomes": dict(Counter(r.get("outcome", "unknown") for r in rows if r.get("outcome") != "success"))},
            "partial_score": {"passed": goal_passed, "denominator": applicable_goals,
                              "percent": round(100 * goal_passed / applicable_goals, 2) if applicable_goals else None,
                              "failed": sum(g.get("verdict") == "fail" for g in goals),
                              "undetermined": sum(g.get("verdict") == "undetermined" for g in goals),
                              "unscored": applicable_goals - len(goals),
                              "cases_with_goals": sum(bool(r.get("goals")) for r in rows)},
            "efficiency": {"tool_calls": _metric(values("tool_calls")),
                           "model_calls": _metric(values("model_calls")),
                           "input_tokens": _metric(tokens("input_tokens")),
                           "output_tokens": _metric(tokens("output_tokens")),
                           "total_tokens": _metric([r["usage"]["input_tokens"] + r["usage"]["output_tokens"] for r in both]),
                           "elapsed_ms": _metric(values("elapsed_ms"))},
            "tool_faults": {"cases": len(fault_packets),
                            "case_ids": [p["case_id"] for p in fault_packets],
                            "judge_executed": 0, "review_pending": len(fault_packets)}}


def write_concise_report(output: Path, manifest: dict, rows: list[dict],
                         fault_packets: list[dict], cases: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    summary = summarize_concise(rows, fault_packets, cases)
    (output / "results.json").write_text(json.dumps({"manifest": manifest, "summary": summary, "cases": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (output / "fault_judge_inputs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in fault_packets), encoding="utf-8")
    sample_ids = {"U05", "U11", "U12", "H3-21", "H3-22", "H3-23", "H3-24"}
    (output / "fault_review_samples.jsonl").write_text("".join(
        json.dumps(p, ensure_ascii=False) + "\n" for p in fault_packets
        if p["case_id"] in sample_ids), encoding="utf-8")
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ("case_id", "outcome", "atomic_pass", "atomic_total", "tool_calls", "model_calls", "input_tokens", "output_tokens", "elapsed_ms", "reasons")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            usage = row.get("usage", {})
            writer.writerow({"case_id": row["case_id"], "outcome": row["outcome"],
                             "atomic_pass": sum(g.get("verdict") == "pass" for g in row.get("goals", [])),
                             "atomic_total": len(row.get("goals", [])),
                             "tool_calls": row.get("tool_calls"), "model_calls": row.get("model_calls"),
                             "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                             "elapsed_ms": row.get("elapsed_ms"), "reasons": " | ".join(row.get("reasons", []))})
    e = summary["efficiency"]
    def avg(name, unit=""):
        item = e[name]
        return f"{item['mean'] if item['mean'] is not None else '未记录'}{unit}（有数据 {item['recorded_cases']}/{summary['selected_cases']} 例）"
    model_ids = sorted({str(row["model_id"]) for row in rows if row.get("model_id")})
    lines = ["# Agent Evaluation 精简报告", "",
             "既有生产 create_agent Runner 轨迹的离线 Replay；本报告没有重新运行模型或工具。确定性任务通过率不包含最终自然语言回复的完整语义复核。", "",
             f"数据集：{manifest['dataset_version']}；指纹：`{manifest['dataset_sha256']}`。", 
             f"源 Trace 记录的模型 ID：{', '.join(model_ids) or '未记录'}；源轨迹目录：`{manifest['trace_directory']}`。",
             f"Oracle：{manifest['oracle_version']}；报告版本：{REPORT_VERSION}；故障分析证据包（尚未调用模型）。", "",
             "## 任务完成", "",
             f"**Pass Rate（确定性任务通过率）**：{summary['pass_rate']['passed']}/{summary['pass_rate']['denominator']} = {summary['pass_rate']['percent']}%。分母是全部选入 Case，包含超时、基础设施错误和证据不足；这些不会自动算通过。",
             f"**Partial Score（原子目标完成率）**：{summary['partial_score']['passed']}/{summary['partial_score']['denominator']} = {summary['partial_score']['percent']}%。分母是全部选入 Case 的适用原子目标；失败 {summary['partial_score']['failed']} 项，待定 {summary['partial_score']['undetermined']} 项，Trace 不足而未评分 {summary['partial_score']['unscored']} 项；有目标证据 {summary['partial_score']['cases_with_goals']}/{summary['selected_cases']} 例。", "",
             "## 执行效率", "",
             f"平均工具调用：{avg('tool_calls')}；平均模型调用：{avg('model_calls')}。",
             f"平均输入 Token：{avg('input_tokens')}；输出 Token：{avg('output_tokens')}；合计 Token：{avg('total_tokens')}。合计只使用输入、输出均有记录的 Case。",
             f"平均耗时：{avg('elapsed_ms', ' ms')}。各项均只对原轨迹中有数值记录的 Case 求算术平均。", "",
             "## 工具故障", "",
             f"发现预设故障或错误 ToolMessage 的 Case {summary['tool_faults']['cases']} 例：{', '.join(summary['tool_faults']['case_ids']) or '无'}。",
             "已生成 `fault_judge_inputs.jsonl`，保留调用与结果顺序及配对、业务状态差异、最终回复；代表性人工复核包见 `fault_review_samples.jsonl`。Judge 本次未运行；合理性全部待复核，不能据此判定故障处理正确。原任务失败与合理故障处理可以并存。", "",
             "## 未通过与边界", "",
             "| Case | Oracle 结果 | 原子目标 | 评分依据（现象，不等于根因） |",
             "| --- | --- | ---: | --- |"]
    for row in rows:
        if row.get("outcome") == "success":
            continue
        goals = row.get("goals", [])
        lines.append(f"| {row['case_id']} | {row['outcome']} | {sum(g.get('verdict') == 'pass' for g in goals)}/{len(goals)} | {', '.join(row.get('reasons', [])) or '证据不足'} |")
    lines += ["", "工具错误不必然是任务失败的根因。最终回复是否完整、诚实及开放式语义需要独立复核；Trace 中未记录的数据不做推断。历史人工成绩和原始 Trace 均未改写。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
