"""R7 isolated, read-only capability-disclosure experiment.

No returned tool call is executed. The existing router is evaluated separately
from the only production agentic tool loop (Todo). The progressive arm is a
bounded prototype, not a production route or authorization policy.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import tiktoken
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.agent.calendar import discover_calendar_tools
from app.agent.graph import CALENDAR_READ_ONLY_TOOLS
from app.agent.nodes import _build_chat_llm
from app.agent.planner import route_request
from app.agent.provider_clients import run_with_provider_clients
from app.agent.tools import make_todo_tools
from app.config import get_settings


ROOT = Path(__file__).resolve().parents[1]
ROUTE_CASES = (
    ("chat_1", "你好，今天心情怎么样？", {"chat"}),
    ("chat_2", "Explain what a checkpoint is in one sentence.", {"chat"}),
    ("todo_read_1", "列出我今天的待办。", {"todo"}),
    ("todo_read_2", "Search my Todos for internship applications.", {"todo"}),
    ("todo_write_1", "创建待办：整理合成项目笔记。", {"todo"}),
    ("todo_write_2", "把合成项目笔记的优先级改成高。", {"todo"}),
    ("calendar_read_1", "查看明天下午我的日历安排。", {"calendar_read"}),
    ("calendar_read_2", "Am I free tomorrow morning on my calendar?", {"calendar_read"}),
    ("calendar_write_1", "明天上午十点在日历创建合成项目同步会。", {"calendar_create"}),
    ("calendar_write_2", "把明天下午的合成项目同步会改到四点。", {"calendar_update"}),
    ("gmail_1", "搜索 Gmail 中关于合成项目里程碑的邮件。", {"email_search"}),
    ("gmail_2", "Show my unread email messages.", {"email_unread"}),
    ("meeting_prep_1", "准备明天的合成项目同步会。", {"meeting_prep"}),
    ("meeting_prep_2", "Prepare me for tomorrow's project review meeting.", {"meeting_prep"}),
    ("smart_schedule_1", "下周找一个两小时空档做合成项目复盘。", {"smart_schedule"}),
    ("smart_schedule_2", "Find 90 free minutes next week for synthetic interview practice.", {"smart_schedule"}),
    ("rundown_1", "明天的每日简报会包含什么？", {"chat"}),
    ("rundown_2", "Explain how my daily rundown is delivered.", {"chat"}),
    ("cross_domain_1", "明天先查日历空档，再创建一个合成复盘待办。", {"cross_domain"}),
    ("cross_domain_2", "If my calendar is free tomorrow afternoon, schedule a synthetic review and add a Todo.", {"cross_domain"}),
    ("ambiguous_1", "帮我处理那个事情。", None),
    ("ambiguous_2", "Move it to later.", None),
    ("no_tool_1", "谢谢你。", {"chat"}),
    ("no_tool_2", "What is two plus two?", {"chat"}),
)

TOOL_CASES = (
    ("todo_create", "Create a Todo named synthetic review notes tomorrow", {"create_todo"}, "todo"),
    ("todo_list", "List my current Todo tasks", {"list_todos"}, "todo"),
    ("todo_search", "Search my Todos for synthetic review notes", {"search_todos"}, "todo"),
    ("todo_update", "Change the priority of synthetic review notes to high", {"search_todos", "update_todo"}, "todo"),
    ("todo_reminder", "Set a reminder for synthetic review notes tomorrow at 20:00", {"search_todos", "set_reminder"}, "todo"),
    ("todo_complete", "Mark my Todo named synthetic review notes completed", {"search_todos", "update_todo"}, "todo"),
    # Deliberately misrouted Todo: tests an explicit, bounded discovery signal.
    ("wrong_route_calendar_1", "What meetings are on my calendar tomorrow?", {"list-events", "search-events"}, "misroute"),
    ("wrong_route_calendar_2", "Check my calendar availability tomorrow afternoon.", {"get-freebusy", "list-events"}, "misroute"),
    ("no_tool_greeting", "Hello, please just say hello.", set(), "no_tool"),
    ("no_tool_arithmetic", "What is two plus two?", set(), "no_tool"),
)


@tool
def activate_calendar_read(reason: str) -> str:
    """Request Calendar read capability when this turn cannot be satisfied by Todo tools. Never request Calendar mutation. Explain the need briefly."""
    raise AssertionError("R7 evaluation must never execute a tool")


def schema_tokens(tools: list) -> int:
    codec = tiktoken.get_encoding("cl100k_base")
    serialized = json.dumps([convert_to_openai_tool(item) for item in tools],
                            ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return len(codec.encode(serialized))


def usage(response) -> dict:
    values = getattr(response, "usage_metadata", None) or {}
    return {"input": values.get("input_tokens"), "output": values.get("output_tokens"),
            "total": values.get("total_tokens")}


async def run() -> dict:
    settings = get_settings()
    discovered = await discover_calendar_tools(settings)
    calendar = [item for item in discovered if item.name in CALENDAR_READ_ONLY_TOOLS]
    if len(calendar) != 7:
        raise RuntimeError("R7 requires seven actual read-only Calendar MCP schemas")
    todo = make_todo_tools(None)
    all_tools = todo + calendar
    model = _build_chat_llm(settings.llm_provider, settings.llm_model)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "model": settings.llm_model, "provider": settings.llm_provider,
              "arms": {"ALL_TOOLS": {"schema_count": len(all_tools), "schema_tokens_estimated": schema_tokens(all_tools)},
                       "CURRENT_SCOPED": {"schema_count": len(todo), "schema_tokens_estimated": schema_tokens(todo)},
                       "PROGRESSIVE_DISCLOSURE": {"schema_count": len(todo) + 1,
                                                   "schema_tokens_estimated": schema_tokens(todo + [activate_calendar_read])}},
              "route_cases": [], "tool_cases": [], "notes": [
                  "No tool calls executed; provider usage includes prompt and schemas and cannot isolate schema tokens.",
                  "Tool-schema tokens use cl100k_base on serialized schemas: estimated, not provider billing tokens.",
                  "Progressive activation is an isolated one-expansion prototype, not a production fallback.",
              ]}
    for case_id, prompt, expected in ROUTE_CASES:
        started = time.perf_counter()
        row = {"id": case_id, "expected": sorted(expected) if expected else "clarify",
               "actual": None, "correct": False, "error_category": None,
               "latency_ms": None}
        try:
            route = await route_request(prompt, settings.default_timezone)
            row["actual"] = route.intent
            row["confidence"] = route.confidence
            row["correct"] = (route.intent in expected if expected else route.confidence < 0.7)
        except Exception as exc:
            row["error_category"] = type(exc).__name__
            row["correct"] = expected is None
        row["latency_ms"] = round((time.perf_counter() - started) * 1000)
        report["route_cases"].append(row)
        print(f"route {case_id}: {row['actual'] or row['error_category']}", flush=True)

    for case_id, prompt, expected, family in TOOL_CASES:
        row = {"id": case_id, "family": family, "expected_first": sorted(expected), "arms": {}}
        for arm in ("ALL_TOOLS", "CURRENT_SCOPED", "PROGRESSIVE_DISCLOSURE"):
            if arm == "ALL_TOOLS":
                bound = all_tools
            elif family == "no_tool":
                bound = []
            elif arm == "PROGRESSIVE_DISCLOSURE":
                bound = todo + [activate_calendar_read]
            else:
                bound = todo
            started = time.perf_counter()
            input_tokens, output_tokens = [], []
            error = None
            selected = []
            expansion = False
            prompt_messages = [SystemMessage(content=(
                "This is a read-only capability-selection evaluation. Do not pretend a tool was executed. "
                "Select the first appropriate available tool if needed. If no tool is needed, respond normally. "
                "If only Todo tools are visible but Calendar read is needed, use activate_calendar_read "
                "when it is available. Never claim Calendar writes are available.")),
                HumanMessage(content=prompt)]
            try:
                first = await (model.bind_tools(bound) if bound else model).ainvoke(prompt_messages)
                selected.extend(call["name"] for call in (getattr(first, "tool_calls", None) or []))
                input_tokens.append(usage(first)["input"])
                output_tokens.append(usage(first)["output"])
                if (arm == "PROGRESSIVE_DISCLOSURE" and selected == ["activate_calendar_read"]):
                    expansion = True
                    call = first.tool_calls[0]
                    second = await model.bind_tools(calendar).ainvoke(prompt_messages + [
                        first,
                        ToolMessage(content="Calendar read tools are now available; select the relevant read tool. No tool has been executed.",
                                    tool_call_id=call["id"]),
                    ])
                    selected.extend(item["name"] for item in (getattr(second, "tool_calls", None) or []))
                    input_tokens.append(usage(second)["input"])
                    output_tokens.append(usage(second)["output"])
            except Exception as exc:
                error = type(exc).__name__
            first_business = next((item for item in selected if item != "activate_calendar_read"), None)
            correct = (first_business in expected if expected else first_business is None)
            row["arms"][arm] = {"selected": selected, "correct_first": bool(correct),
                                "wrong_tool": first_business is not None and first_business not in expected,
                                "unnecessary_tool": not expected and first_business is not None,
                                "expanded": expansion, "loop_turns": 2 if expansion else 1,
                                "provider_input_tokens": sum(input_tokens) if all(isinstance(x, int) for x in input_tokens) else None,
                                "provider_output_tokens": sum(output_tokens) if all(isinstance(x, int) for x in output_tokens) else None,
                                "schema_tokens_estimated_first": schema_tokens(bound) if bound else 0,
                                "latency_ms": round((time.perf_counter() - started) * 1000),
                                "error_category": error}
            print(f"tool {case_id}/{arm}: {selected or '-'}", flush=True)
        report["tool_cases"].append(row)
    report["summary"] = {}
    for arm in report["arms"]:
        rows = [row["arms"][arm] for row in report["tool_cases"]]
        report["summary"][arm] = {
            "correct_first": sum(row["correct_first"] for row in rows),
            "wrong_tool": sum(row["wrong_tool"] for row in rows),
            "unnecessary_tool": sum(row["unnecessary_tool"] for row in rows),
            "expansions": sum(row["expanded"] for row in rows),
            "provider_input_tokens": sum(row["provider_input_tokens"] for row in rows)
                if all(isinstance(row["provider_input_tokens"], int) for row in rows) else None,
            "provider_output_tokens": sum(row["provider_output_tokens"] for row in rows)
                if all(isinstance(row["provider_output_tokens"], int) for row in rows) else None,
            "latency_ms": sum(row["latency_ms"] for row in rows),
            "model_calls": sum(row["loop_turns"] for row in rows),
        }
    report["summary"]["route_correct"] = sum(row["correct"] for row in report["route_cases"])
    return report


def main() -> None:
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    report = asyncio.run(run_with_provider_clients(run()))
    output = ROOT / "evals" / "reports" / "r7_capability_disclosure.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
