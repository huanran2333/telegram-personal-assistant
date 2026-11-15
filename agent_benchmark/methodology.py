"""Descriptive dataset profile; inferred labels never replace frozen case metadata."""
from __future__ import annotations

import argparse
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from .dataset import load_dataset


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "experiments/official_agent_eval_rebuild_20261023/cases_holdout_v3.json",
    ROOT / "experiments/official_agent_eval_rebuild_20261023/cases_unseen_20261103.json",
)


def describe(case: dict) -> dict:
    """Classify designed work, never observed model steps or historical author intent."""
    old = case["legacy"]
    kinds = [goal["kind"] for goal in case["expected_outcome"]]
    decisions = old.get("decisions", {})
    fault = bool(old.get("calendar_fault") or old.get("gmail_fault"))
    dependency = bool(old.get("order"))
    negative = case["category"] in {
        "negative_authorization", "negative_target", "readonly_negative",
        "no_duplicate", "calendar_conflict",
    } or "reject_delete" in kinds
    capabilities = set()
    for kind in kinds:
        if kind.startswith("calendar_"):
            capabilities.add("Calendar 读取")
        elif kind.startswith("gmail_"):
            capabilities.add("Gmail 读取")
        elif kind in {"delete", "reject_delete"}:
            capabilities.add("Todo 删除/审批")
        elif kind == "reminder_set":
            capabilities.add("提醒写入")
        elif kind == "create":
            capabilities.add("Todo 创建")
        elif kind == "todo_read":
            capabilities.add("Todo 查询")
    if dependency:
        capabilities.add("跨工具依赖")
    if len(kinds) > 1:
        capabilities.add("多目标")
    if fault:
        capabilities.add("故障反馈")
    if negative or not case["available_tools_or_policy"]["allowed_writes"]:
        capabilities.add("禁止多余写入")
    if decisions:
        capabilities.add("HITL")
    # One declared goal approximates one necessary tool operation; each approval
    # adds a separate human decision. This is NOT an observed model/tool count.
    planned_steps = len(kinds) + len(decisions)
    if len(decisions) >= 2 or dependency or (fault and len(kinds) > 1):
        difficulty = "困难"
    elif len(kinds) > 1 or decisions or fault or negative or "reminder_set" in kinds:
        difficulty = "中等"
    else:
        difficulty = "简单"
    length = "1–3" if planned_steps <= 3 else "4–8" if planned_steps <= 8 else "9+"
    return {"case_id": case["case_id"], "category": case["category"],
            "capabilities": sorted(capabilities), "difficulty": difficulty,
            "planned_steps": planned_steps, "length": length,
            "negative": negative, "fault": fault,
            "no_write": not case["available_tools_or_policy"]["allowed_writes"]}


def _shape(request: str) -> str:
    # Similarity is a review prompt, not proof of leakage or semantic identity.
    request = re.sub(r"[HU][0-9-]+", "<ID>", request)
    request = re.sub(r"『[^』]+』", "<TITLE>", request)
    request = re.sub(r"\d{1,4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日", "<DATE>", request)
    return re.sub(r"\s+", "", request)


def profile(paths: tuple[Path, ...] = SOURCES) -> dict:
    datasets = [load_dataset(path) for path in paths]
    rows = {dataset["version"]: [describe(case) for case in dataset["cases"]]
            for dataset in datasets}
    cross = []
    if len(datasets) == 2:
        for left in datasets[0]["cases"]:
            for right in datasets[1]["cases"]:
                a, b = _shape(left["user_request"]), _shape(right["user_request"])
                ratio = SequenceMatcher(None, a, b).ratio()
                if ratio >= 0.75:
                    cross.append({"left": left["case_id"], "right": right["case_id"],
                                  "shape_similarity": round(ratio, 3)})
    return {"datasets": [{"version": d["version"], "sha256": d["sha256"],
                          "source": Path(d["source_path"]).name,
                          "cases": len(d["cases"])} for d in datasets],
            "rows": rows, "cross_set_review_pairs": sorted(
                cross, key=lambda x: (-x["shape_similarity"], x["left"], x["right"]))}


def markdown(data: dict) -> str:
    lines = ["# H3 与 U 冻结集：设计覆盖诊断", "",
             "标签是本轮从已冻结目标、审批决定、故障注入和允许写入清单推导的诊断分类；不是历史原始标签，也不是模型实际执行步骤。",
             "预期步骤 = 声明的原子目标数 + 人工审批决定数；不含模型内部循环、工具重试与最终回复。",
             "文章的 4–8 与 8+ 在第 8 步重叠；本表用 1–3、4–8、9+ 三个互斥区间。",
             "难度规则：多审批、显式读后写依赖、故障与独立目标并存为困难；其余多目标、单审批、单故障、负面约束或提醒为中等；其余为简单。难度按业务约束而非运行结果划分。", ""]
    for dataset in data["datasets"]:
        rows = data["rows"][dataset["version"]]
        difficulties = Counter(row["difficulty"] for row in rows)
        lengths = Counter(row["length"] for row in rows)
        lines += [f"## {dataset['version']}（{len(rows)} 例）", "",
                  f"来源 `{dataset['source']}`；SHA256 `{dataset['sha256']}`。",
                  f"难度：简单 {difficulties['简单']}/{len(rows)}、中等 {difficulties['中等']}/{len(rows)}、困难 {difficulties['困难']}/{len(rows)}。文章的 30/50/20 仅供检查，不用于改写冻结集。",
                  f"设计步骤：1–3 步 {lengths['1–3']}/{len(rows)}、4–8 步 {lengths['4–8']}/{len(rows)}、9+ 步 {lengths['9+']}/{len(rows)}。",
                  f"负面约束 {sum(r['negative'] for r in rows)}/{len(rows)}；禁止所有写入 {sum(r['no_write'] for r in rows)}/{len(rows)}；注入故障 {sum(r['fault'] for r in rows)}/{len(rows)}。",
                  "", "| Case | 原始类别 | 能力维度 | 难度 | 必要步骤 | 长度 | 负面 |",
                  "| --- | --- | --- | --- | ---: | --- | --- |"]
        lines += [f"| {r['case_id']} | {r['category']} | {'、'.join(r['capabilities'])} | {r['difficulty']} | {r['planned_steps']} | {r['length']} | {'是' if r['negative'] else '否'} |"
                  for r in rows]
        lines.append("")
    pairs = data["cross_set_review_pairs"]
    lines += ["## 两套数据的独立性检查", "",
              "比较去掉案例编号、引号内标题和日期后的请求字面形状；相似度 ≥0.75 的配对仅供人工检查，不能据此判定数据泄漏。不同来源之间的语义同类任务也可能合理重复。",
              f"需人工查看的形状相似配对：{len(pairs)} 组。", ""]
    if pairs:
        lines += ["| H3 | U | 字面形状相似度 |", "| --- | --- | ---: |"]
        lines += [f"| {p['left']} | {p['right']} | {p['shape_similarity']:.3f} |" for p in pairs]
        lines.append("")
    lines += ["没有记录生成模型、独立审核者或训练数据重叠检查；不能宣称完全无污染。冻结后的开发调试会使已见案例丧失新的未见资格。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only descriptive profile of frozen evaluation datasets")
    parser.add_argument("--output", type=Path, required=True, help="new Markdown file; never overwrite")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(markdown(profile()))


if __name__ == "__main__":
    main()
