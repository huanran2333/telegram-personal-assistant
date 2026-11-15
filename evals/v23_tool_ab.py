"""Read-only A/B of existing Todo-tool exposure versus Todo-only binding.

Calls the configured real model and discovers current read-only Calendar MCP
schemas, but NEVER executes a returned tool call or writes to external systems.
The candidate binding exists only in this evaluator, not in production.
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.agent.calendar import discover_calendar_tools
from app.agent.graph import CALENDAR_READ_ONLY_TOOLS
from app.agent.nodes import _build_chat_llm
from app.agent.provider_clients import run_with_provider_clients
from app.agent.planner import analyze_route, route_request
from app.agent.prompt import SYSTEM_PROMPT
from app.agent.tools import make_todo_tools
from app.config import get_settings
from evals.v23_runner import ROOT, RecordingStructuredLLM

TOOL_CASES = [
    ("todo_create", "Create a Todo named synthetic review notes tomorrow", {"create_todo"}),
    ("todo_list", "List my current Todo tasks", {"list_todos"}),
    ("todo_search", "Search my Todos for synthetic review notes", {"search_todos"}),
    # A title-only update/reminder first needs ID resolution. A search call is
    # a valid first step, but complete task success is not proven by this probe.
    ("todo_update", "Change the priority of synthetic review notes to high", {"search_todos", "update_todo"}),
    ("todo_reminder", "Set a reminder for synthetic review notes tomorrow at 20:00", {"search_todos", "set_reminder"}),
]
CROSS_DOMAIN_CASES = [
    ("explicit_date", "Check my calendar on 2026-10-09 from 13:00 to 17:00, and if free "
     "schedule two hours to write a synthetic report"),
    ("relative_day", "看看周五下午有没有会，没有的话安排两小时写合成测试报告。"),
]


def schema_chars(tools: list) -> int:
    return sum(len(json.dumps(convert_to_openai_tool(tool), ensure_ascii=False)) for tool in tools)


async def run() -> dict:
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    settings = get_settings()
    todo = make_todo_tools(None)
    calendar = []
    discovery = {"status": "NOT_MEASURED", "tool_count": None, "schema_chars": None,
                 "error_category": None}
    try:
        discovered = await discover_calendar_tools(settings)
        calendar = [tool for tool in discovered if tool.name in CALENDAR_READ_ONLY_TOOLS]
        discovery.update(status="PASS", tool_count=len(calendar), schema_chars=schema_chars(calendar))
    except Exception as exc:
        discovery["error_category"] = type(exc).__name__
    all_tools = todo + calendar
    git_commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "git_commit": git_commit,
              "provider": settings.llm_provider, "model": settings.llm_model,
              "temperature": 0, "evidence_class": "real LLM, read-only synthetic tool selection",
              "calendar_discovery": discovery,
              "current": {"tool_count": len(all_tools), "schema_chars": schema_chars(all_tools)},
              "candidate": {"tool_count": len(todo), "schema_chars": schema_chars(todo)},
              "ab_valid": bool(calendar), "cases": [], "cross_domain": [], "metrics": {}}
    if not calendar:
        return report
    base = _build_chat_llm(settings.llm_provider, settings.llm_model)
    for case_id, prompt, expected_tools in TOOL_CASES:
        row = {"id": case_id, "acceptable_first_tools": sorted(expected_tools)}
        for name, tools in (("current", all_tools), ("candidate", todo)):
            started = time.perf_counter()
            try:
                response = await base.bind_tools(tools).ainvoke([
                    SystemMessage(content=SYSTEM_PROMPT.format(user_id="synthetic-eval-user")),
                    HumanMessage(content=prompt),
                ])
                calls = getattr(response, "tool_calls", []) or []
                usage = getattr(response, "usage_metadata", None) or {}
                selected = [call.get("name") for call in calls if isinstance(call, dict)]
                row[name] = {"selected_tools": selected,
                             "correct_selection": bool(selected) and selected[0] in expected_tools,
                             "wrong_selection": bool(selected) and selected[0] not in expected_tools,
                             "malformed_tool_call": bool(getattr(response, "invalid_tool_calls", [])),
                             "input_tokens": usage.get("input_tokens"),
                             "total_tokens": usage.get("total_tokens"),
                             "latency_ms": round((time.perf_counter() - started) * 1000),
                             "error_category": None}
            except Exception as exc:
                row[name] = {"selected_tools": [], "correct_selection": False,
                             "wrong_selection": None, "malformed_tool_call": None,
                             "input_tokens": None, "total_tokens": None,
                             "latency_ms": round((time.perf_counter() - started) * 1000),
                             "error_category": type(exc).__name__}
        report["cases"].append(row)
    # Cross-domain should stay in the typed productivity workflow, not the
    # narrowed Todo loop. This is a read-only routing/plan regression check.
    router = RecordingStructuredLLM(_build_chat_llm(settings.llm_provider, settings.llm_model, routing=True))
    for case_id, request in CROSS_DOMAIN_CASES:
        row = {"id": case_id, "route": None, "operations": [],
               "preserves_calendar_read": False, "no_execution": True, "error_category": None}
        try:
            route = await route_request(request, settings.default_timezone, llm=router)
            row["route"] = route.intent
            plan = analyze_route(route, settings.default_timezone, request=request)
            row["operations"] = [step.operation for step in plan.actions]
            row["preserves_calendar_read"] = any(step.domain == "calendar" and not step.mutation
                                                 for step in plan.actions)
        except Exception as exc:
            row["error_category"] = type(exc).__name__
        report["cross_domain"].append(row)
    a = [row["current"] for row in report["cases"]]
    b = [row["candidate"] for row in report["cases"]]
    report["metrics"] = {
        "first_tool_selection_current": sum(item["correct_selection"] for item in a),
        "first_tool_selection_candidate": sum(item["correct_selection"] for item in b),
        "selection_case_count": len(a),
        "malformed_current": sum(bool(item["malformed_tool_call"]) for item in a),
        "malformed_candidate": sum(bool(item["malformed_tool_call"]) for item in b),
        "provider_input_tokens_current": sum(item["input_tokens"] for item in a)
        if all(isinstance(item["input_tokens"], int) for item in a) else None,
        "provider_input_tokens_candidate": sum(item["input_tokens"] for item in b)
        if all(isinstance(item["input_tokens"], int) for item in b) else None,
        "task_success": None, "db_side_effect_correctness": None,
    }
    return report


def main() -> None:
    report = asyncio.run(run_with_provider_clients(run()))
    output = ROOT / "evals" / "reports"
    output.mkdir(parents=True, exist_ok=True)
    (output / "v23_tool_ab.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    lines = ["# V2.3 Tool Exposure A/B", "",
             f"Git: `{report['git_commit']}`; provider/model: `{report['provider']}` / `{report['model']}`; temperature: {report['temperature']}.",
             "Read-only real-LLM calls. Tool calls were inspected but never executed; DB side-effect correctness is NOT_MEASURED here.",
             f"Calendar discovery: {report['calendar_discovery']['status']}; error category: {report['calendar_discovery']['error_category'] or 'none'}",
             f"Current: {report['current']['tool_count']} schemas / {report['current']['schema_chars']} JSON chars.",
             f"Candidate: {report['candidate']['tool_count']} schemas / {report['candidate']['schema_chars']} JSON chars.",
             "", "| Case | Current selected | Candidate selected | Input tokens A/B | Latency ms A/B |", "|---|---|---|---:|---:|"]
    for row in report["cases"]:
        a, b = row["current"], row["candidate"]
        lines.append(f"| {row['id']} | {','.join(a['selected_tools']) or 'none'} | "
                     f"{','.join(b['selected_tools']) or 'none'} | "
                     f"{a['input_tokens']}/{b['input_tokens']} | {a['latency_ms']}/{b['latency_ms']} |")
    lines += ["", f"Aggregate first-tool selection: {report['metrics']['first_tool_selection_current']}/"
              f"{report['metrics']['selection_case_count']} current vs "
              f"{report['metrics']['first_tool_selection_candidate']}/"
              f"{report['metrics']['selection_case_count']} candidate.",
              f"Provider input tokens across cases: {report['metrics']['provider_input_tokens_current']} current vs "
              f"{report['metrics']['provider_input_tokens_candidate']} candidate.",
              "Task success and DB side-effect correctness: NOT_MEASURED (first tool calls were not executed).",
              "", f"Cross-domain typed routes: `{report['cross_domain']}`", "",
              "Schema characters are not provider tokens. Candidate has not been enabled in production.", ""]
    (output / "v23_tool_ab.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"tool A/B valid={report['ab_valid']} calendar_read_schemas={len(report['cases']) and report['calendar_discovery']['tool_count']}")


if __name__ == "__main__":
    main()
