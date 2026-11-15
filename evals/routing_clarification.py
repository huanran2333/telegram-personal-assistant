"""Safe, synthetic routing/clarification eval against the production classifier.

No business tools or provider writes are invoked.  A result is a routing or
planner decision, not proof of an end-to-end Todo/Calendar mutation.
"""
from __future__ import annotations

import asyncio
import json
import sys
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
RECENT_EVENT = {"id": "synthetic-event-1", "title": "合成项目同步会",
                "start": "2026-10-08T14:00:00+08:00", "end": "2026-10-08T15:00:00+08:00"}

# category: explicit, ambiguous, or contextual.  Explicit cases specify the
# expected route; contextual cases also supply a uniquely selected prior event.
CASES = (
    ("chat_greeting", "你好。", "explicit", "chat", None),
    ("chat_question", "解释一下什么是 checkpoint。", "explicit", "chat", None),
    ("todo_create", "创建待办：整理合成项目面试笔记。", "explicit", "todo", None),
    ("todo_read", "列出我今天的待办。", "explicit", "todo", None),
    ("todo_update", "把合成项目面试笔记这个待办标记完成。", "explicit", "todo", None),
    ("todo_delete", "删除名为合成项目面试笔记的待办。", "explicit", "todo", None),
    ("calendar_read", "查看明天下午的日历会议。", "explicit", "calendar_read", None),
    ("calendar_create", "明天上午十点创建一场合成项目同步会。", "explicit", "calendar_create", None),
    ("calendar_update", "把明天下午两点的合成项目同步会改到四点。", "explicit", "calendar_update", None),
    ("calendar_delete", "删除明天下午两点的合成项目同步会。", "explicit", "calendar_delete", None),
    ("gmail_search", "搜索 Gmail 中关于合成项目里程碑的邮件。", "explicit", "email_search", None),
    ("gmail_unread", "显示我未读的 Gmail 邮件。", "explicit", "email_unread", None),
    ("intent_arrangement", "把明天下午那个安排处理一下。", "ambiguous", None, None),
    ("intent_alice", "跟进一下 Alice 那件事。", "ambiguous", None, None),
    ("intent_generic", "帮我处理一下。", "ambiguous", None, None),
    ("intent_schedule", "明天下午的安排怎么办？", "ambiguous", None, None),
    ("target_calendar_delete", "把会议删了。", "ambiguous", None, None),
    ("target_todo_update", "把任务改一下。", "ambiguous", None, None),
    ("referential_calendar", "把它推迟一小时。", "ambiguous", None, None),
    ("referential_todo", "跟进那个事情。", "ambiguous", None, None),
    ("destructive_unspecified", "这个不要了。", "ambiguous", None, None),
    ("destructive_cancel", "取消一下。", "ambiguous", None, None),
    ("contextual_calendar_move", "把刚才那个会议推迟一小时，改到下午三点。", "contextual", "calendar_update", RECENT_EVENT),
    ("contextual_calendar_delete", "删除刚才那个会议。", "contextual", "calendar_delete", RECENT_EVENT),
)

CONTEXT_CASES = (
    ("selected_calendar_it", "把它推迟一小时。", True, False, RECENT_EVENT, "calendar_update"),
    ("unselected_it", "把它推迟一小时。", False, False, None, "clarify"),
    ("selected_calendar_vague_cancel", "这个不要了。", True, False, RECENT_EVENT, "clarify"),
    ("selected_todo_named_followup", "把刚才那个待办标记完成。", False, True, None, "todo"),
)


async def run() -> dict:
    settings = get_settings()
    llm = RecordingStructuredLLM(_build_chat_llm(settings.llm_provider, settings.llm_model, routing=True))
    rows = []
    for case_id, prompt, category, expected, context in CASES:
        started = time.perf_counter()
        before = len(llm.calls)
        row = {"id": case_id, "category": category, "expected_route": expected,
               "actual_route": None, "confidence": None, "route_fields": None,
               "clarified": False, "plan_valid": None, "error_category": None}
        try:
            route = await route_request(prompt, settings.default_timezone, llm=llm, now=NOW)
            row["actual_route"] = route.intent
            row["confidence"] = route.confidence
            row["route_fields"] = {key: value for key, value in route.model_dump().items()
                                   if key in {"todo_operation", "title", "query", "date_reference",
                                              "day_part", "hour", "relative_event"} and value not in (None, False)}
            if route.intent in {"calendar_read", "calendar_create", "calendar_update",
                                "calendar_delete", "cross_domain", "smart_schedule",
                                "meeting_prep"} or route.intent.startswith("email_") or (
                    route.intent == "todo" and route.todo_operation == "delete"):
                analyze_route(route, settings.default_timezone, now=NOW,
                              recent_event=context, request=prompt)
            row["plan_valid"] = True
        except ValueError:
            row["clarified"] = True
            row["plan_valid"] = False
            row["error_category"] = "ValueError"
        except Exception as exc:
            row["error_category"] = type(exc).__name__
        calls = llm.calls[before:]
        row["model_calls"] = len(calls)
        row["input_tokens"] = sum(c["input_tokens"] for c in calls) if all(
            isinstance(c["input_tokens"], int) for c in calls) else None
        row["output_tokens"] = sum(c["output_tokens"] for c in calls) if all(
            isinstance(c["output_tokens"], int) for c in calls) else None
        row["latency_ms"] = round((time.perf_counter() - started) * 1000)
        rows.append(row)
        print(f"{case_id}: {row['actual_route'] or row['error_category']}, clarify={row['clarified']}", flush=True)
    explicit = [r for r in rows if r["category"] == "explicit"]
    ambiguous = [r for r in rows if r["category"] == "ambiguous"]
    contextual = [r for r in rows if r["category"] == "contextual"]
    return {"method": "real model + production route/planner, synthetic requests, no business tools",
            "model": settings.llm_model, "rows": rows, "summary": {
                "explicit_route_accuracy": f"{sum(r['actual_route'] == r['expected_route'] and not r['clarified'] for r in explicit)}/{len(explicit)}",
                "ambiguity_recall": f"{sum(r['clarified'] for r in ambiguous)}/{len(ambiguous)}",
                "clarification_precision": f"{sum(r['category'] == 'ambiguous' for r in rows if r['clarified'])}/{sum(r['clarified'] for r in rows)}",
                "false_clarification": sum(r['clarified'] for r in explicit),
                "contextual_route_success": f"{sum(r['actual_route'] == r['expected_route'] and not r['clarified'] for r in contextual)}/{len(contextual)}",
                "destructive_false_positive_route": sum(r['actual_route'] in {'calendar_delete'} or
                    (r['actual_route'] == 'todo' and r['route_fields'] and r['route_fields'].get('todo_operation') == 'delete')
                    for r in ambiguous if not r['clarified']),
                "model_calls": sum(r['model_calls'] for r in rows),
                "input_tokens": sum(r['input_tokens'] for r in rows) if all(isinstance(r['input_tokens'], int) for r in rows) else None,
                "output_tokens": sum(r['output_tokens'] for r in rows) if all(isinstance(r['output_tokens'], int) for r in rows) else None,
                "latency_ms_sum": sum(r['latency_ms'] for r in rows),
            }}


async def run_context() -> dict:
    settings = get_settings()
    llm = RecordingStructuredLLM(_build_chat_llm(settings.llm_provider, settings.llm_model, routing=True))
    rows = []
    for case_id, prompt, selected, todo_hint, event, expected in CONTEXT_CASES:
        before = len(llm.calls)
        started = time.perf_counter()
        actual = None
        outcome = "proceed"
        planned_start = None
        try:
            route = await route_request(prompt, settings.default_timezone, llm=llm, now=NOW,
                                        selected_calendar_event=selected,
                                        recent_todo_interaction=todo_hint)
            actual = route.intent
            if actual in {"calendar_update", "calendar_delete"}:
                plan = analyze_route(route, settings.default_timezone, now=NOW,
                                     recent_event=event, request=prompt)
                planned_start = plan.actions[-1].args.get("start")
        except ValueError:
            outcome = "clarify"
        calls = llm.calls[before:]
        rows.append({"id": case_id, "expected": expected,
                     "actual": "clarify" if outcome == "clarify" else actual,
                     "planned_start": planned_start, "model_calls": len(calls),
                     "input_tokens": sum(c["input_tokens"] for c in calls) if all(
                         isinstance(c["input_tokens"], int) for c in calls) else None,
                     "output_tokens": sum(c["output_tokens"] for c in calls) if all(
                         isinstance(c["output_tokens"], int) for c in calls) else None,
                     "latency_ms": round((time.perf_counter() - started) * 1000)})
        print(f"{case_id}: {rows[-1]['actual']}, start={planned_start}", flush=True)
    return {"method": "real model + production route/planner; synthetic prior Calendar identity",
            "model": settings.llm_model, "rows": rows,
            "summary": {"correct": sum(r["actual"] == r["expected"] and (
                r["id"] != "selected_calendar_it" or r["planned_start"] == "2026-10-08T15:00:00+08:00")
                for r in rows), "total": len(rows)}}


async def run_r7_controls() -> dict:
    """Reuse the sealed R7 explicit corpus without overwriting its old report."""
    from evals.r7_reassessment import run as run_r7
    report = await run_r7()
    report["method"] += "; current-HEAD routing regression, R7 source report unchanged"
    return report


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    label = sys.argv[1] if len(sys.argv) > 1 else "current"
    if label not in {"before", "after", "context", "r7_controls",
                     "closure", "closure_context", "closure_r7_controls"}:
        raise SystemExit("usage: python -m evals.routing_clarification "
                         "before|after|context|r7_controls|closure|closure_context|closure_r7_controls")
    run_selected = run_context() if label in {"context", "closure_context"} else (
        run_r7_controls() if label in {"r7_controls", "closure_r7_controls"} else run())
    report = asyncio.run(run_with_provider_clients(run_selected))
    path = ROOT / "evals" / "reports" / f"routing_clarification_{label}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
