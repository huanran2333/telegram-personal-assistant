"""R7 reassessment: production-shaped route/plan audit, no provider tool execution.

This corpus exercises the production classifier and deterministic planner. It is
not an end-to-end business task benchmark; R7's existing full Todo A/B supplies
that separate layer. Only synthetic prompts and safe result metadata are saved.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from app.agent.nodes import _build_chat_llm
from app.agent.planner import analyze_route, route_request
from app.agent.provider_clients import run_with_provider_clients
from app.config import get_settings
from evals.v23_runner import RecordingStructuredLLM


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 7, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
# Expected routing and whether the production path must be a typed workflow.
# No case is artificially injected into the Todo loop after routing.
CASES = (
    ("todo_create_1", "创建待办：整理合成项目的面试笔记。", "todo", False),
    ("todo_create_2", "Add a Todo to review my synthetic interview notes tomorrow.", "todo", False),
    ("todo_read_1", "列出我今天的待办。", "todo", False),
    ("todo_read_2", "搜索标题里有合成项目的待办。", "todo", False),
    ("todo_update_1", "把合成项目面试笔记这个待办的优先级改高。", "todo", False),
    ("todo_update_2", "把合成项目面试笔记这个待办标记完成。", "todo", False),
    ("todo_reminder_1", "明晚八点提醒我整理合成项目面试笔记。", "todo", False),
    ("todo_reminder_2", "Remind me tomorrow at 20:00 to review synthetic notes.", "todo", False),
    ("calendar_read_1", "查看明天下午的日历会议。", "calendar_read", True),
    ("calendar_read_2", "Am I free on my calendar tomorrow morning?", "calendar_read", True),
    ("calendar_create_1", "明天上午十点创建一场合成项目同步会，持续一小时。", "calendar_create", True),
    ("calendar_create_2", "Create a calendar event tomorrow at 14:00 named Synthetic Review for 30 minutes.", "calendar_create", True),
    ("calendar_update_1", "把明天下午两点的合成项目同步会改到四点。", "calendar_update", True),
    ("calendar_update_2", "Move tomorrow's Synthetic Review calendar event from 14:00 to 15:00.", "calendar_update", True),
    ("gmail_search_1", "搜索 Gmail 中关于合成项目里程碑的邮件。", "email_search", True),
    ("gmail_search_2", "Find emails mentioning the synthetic benchmark report.", "email_search", True),
    ("gmail_read_1", "显示我未读的 Gmail 邮件。", "email_unread", True),
    ("gmail_read_2", "Show my unread email messages.", "email_unread", True),
    ("meeting_prep_1", "为明天的合成项目同步会做会议准备。", "meeting_prep", True),
    ("meeting_prep_2", "Prepare me for tomorrow's Synthetic Review meeting.", "meeting_prep", True),
    ("smart_schedule_1", "下周找一个两小时空档做合成项目复盘。", "smart_schedule", True),
    ("smart_schedule_2", "Find 90 free minutes next week for synthetic interview practice.", "smart_schedule", True),
    ("cross_domain_1", "查看明天下午的会议，然后创建一个跟进合成项目的待办。", "cross_domain", True),
    ("cross_domain_2", "如果明天下午日历有空，安排合成项目同步会并创建准备材料的待办。", "cross_domain", True),
    ("cross_domain_3", "明天先查日历空档，再创建一个合成复盘待办。", "cross_domain", True),
    ("chat_1", "你好。", "chat", False),
    ("chat_2", "Explain what a checkpoint is in one sentence.", "chat", False),
    ("no_tool_1", "谢谢你。", "chat", False),
    ("no_tool_2", "What is two plus two?", "chat", False),
    ("ambiguous_1", "把明天下午那个安排处理一下。", "clarify", False),
    ("ambiguous_2", "跟进一下 Alice 那件事。", "clarify", False),
)


async def run() -> dict:
    settings = get_settings()
    model = RecordingStructuredLLM(_build_chat_llm(settings.llm_provider, settings.llm_model, routing=True))
    rows = []
    for case_id, prompt, expected, expected_plan in CASES:
        started = time.perf_counter()
        before = len(model.calls)
        row = {"id": case_id, "expected_route": expected, "actual_route": None,
               "route_correct": False, "expected_typed_plan": expected_plan,
               "actual_typed_plan": None, "plan_step_domains": [], "clarified": False,
               "error_category": None, "model_calls": None, "input_tokens": None,
               "output_tokens": None, "latency_ms": None}
        try:
            route = await route_request(prompt, settings.default_timezone, llm=model, now=NOW)
            row["actual_route"] = route.intent
            if expected == "clarify":
                if route.confidence < 0.7:
                    row["clarified"] = True
                else:
                    try:
                        analyze_route(route, settings.default_timezone, now=NOW, request=prompt)
                    except ValueError:
                        row["clarified"] = True
            else:
                row["route_correct"] = route.intent == expected
                if row["route_correct"]:
                    analysis = analyze_route(route, settings.default_timezone, now=NOW, request=prompt)
                    row["actual_typed_plan"] = bool(analysis.requires_plan)
                    row["plan_step_domains"] = [step.domain for step in analysis.actions]
        except ValueError:
            row["error_category"] = "ValueError"
            row["clarified"] = expected == "clarify"
        except Exception as exc:
            row["error_category"] = type(exc).__name__
        calls = model.calls[before:]
        row["model_calls"] = len(calls)
        for field in ("input_tokens", "output_tokens"):
            row[field] = sum(call[field] for call in calls) if all(isinstance(call[field], int) for call in calls) else None
        row["latency_ms"] = round((time.perf_counter() - started) * 1000)
        rows.append(row)
        print(f"{case_id}: {row['actual_route'] or row['error_category']}, plan={row['actual_typed_plan']}", flush=True)
    explicit = [row for row in rows if row["expected_route"] != "clarify"]
    typed = [row for row in explicit if row["expected_typed_plan"]]
    return {"method": "real-model production route_request + analyze_route; synthetic requests; no business/provider tools executed",
            "model": settings.llm_model, "timestamp_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
            "rows": rows, "summary": {
                "explicit_route_correct": sum(row["route_correct"] for row in explicit),
                "explicit_route_count": len(explicit),
                "typed_plan_correct": sum(row["route_correct"] and row["actual_typed_plan"] is True for row in typed),
                "typed_plan_count": len(typed),
                "clarification_correct": sum(row["clarified"] for row in rows if row["expected_route"] == "clarify"),
                "clarification_count": len(rows) - len(explicit),
                "model_calls": sum(row["model_calls"] for row in rows),
                "route_input_tokens": sum(row["input_tokens"] for row in rows) if all(isinstance(row["input_tokens"], int) for row in rows) else None,
                "route_output_tokens": sum(row["output_tokens"] for row in rows) if all(isinstance(row["output_tokens"], int) for row in rows) else None,
                "route_latency_ms_sum": sum(row["latency_ms"] for row in rows),
            }}


def main() -> None:
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    result = asyncio.run(run_with_provider_clients(run()))
    path = ROOT / "evals" / "reports" / "r7_reassessment.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
