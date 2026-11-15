"""Opt-in, full Todo node-loop A/B on isolated real PostgreSQL users.

The --production mode uses the same tool-selection helper as build_graph and
the same make_nodes construction. Neither mode changes the active application's
binding. All synthetic users/rows are removed after each arm, even on failure.
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import delete, select

from app.agent.calendar import MCPCalendarAdapter, discover_calendar_tools
from app.agent.actions import ActionProposal, requires_approval
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.graph import CALENDAR_READ_ONLY_TOOLS, todo_llm_tool_binding
from app.agent.nodes import make_nodes
from app.agent.planner import route_request
from app.agent.provider_clients import run_with_provider_clients
from app.agent.tools import make_todo_tools
from app.agent.workflow import build_productivity_graph, resume_workflow, start_workflow
from app.config import get_settings
from app.db import crud
from app.db.models import Todo, User
from app.db.session import AsyncSessionLocal
from evals.v23_runner import ROOT
from evals.v23_tool_ab import schema_chars


CASES = (
    ("create", "Create a Todo named V23 synthetic review notes", None),
    ("update", "Change V23 synthetic review notes priority to high", "medium"),
    ("search", "Search my Todos for V23 synthetic review notes", "medium"),
    ("deadline", "Create a Todo named V23 synthetic deadline due tomorrow", None),
    ("priority_tag", "Create a high priority Todo named V23 synthetic tagged task with tag #v23", None),
    ("multi_step", "Create V23 synthetic multi task due tomorrow, priority high, and tag #v23", None),
)


class BusyCalendar:
    """Deterministic conflict fixture; never represented as a real MCP read."""

    def __init__(self):
        self.writes = 0

    async def availability(self, start, end, timezone):
        return {"calendars": {"primary": {"busy": [{"start": start, "end": end}]}}}

    async def create_event(self, args):
        self.writes += 1
        raise AssertionError("Unexpected Calendar mutation")


async def _cross_domain(arm: str, request: str, calendar, read_type: str) -> dict:
    """Same typed production workflow for either experimental Todo binding."""
    from app.agent.planner import analyze_route
    row = {"arm": arm, "read_type": read_type, "route": None, "operations": [],
           "calendar_read": False, "proposal_requires_hitl": False,
           "mutation_before_approval": False, "rejected_without_mutation": False,
           "error_category": None}
    try:
        settings = get_settings()
        if not settings.google_calendar_owner_telegram_id:
            raise RuntimeError("Calendar owner identity is not configured")
        route = await route_request(request, settings.default_timezone)
        row["route"] = route.intent
        plan = analyze_route(route, settings.default_timezone, request=request)
        row["operations"] = [step.operation for step in plan.actions]
        if route.intent != "cross_domain":
            return row
        graph = build_productivity_graph(
            calendar, ActionExecutor(calendar, SqlAlchemyActionLedger(AsyncSessionLocal), AsyncSessionLocal),
            InMemorySaver(), calendar_owner_telegram_id=settings.google_calendar_owner_telegram_id)
        thread_id = "v23-ab-" + uuid4().hex
        result = await start_workflow(graph, "v23-ab-no-db-user", request,
                                      settings.default_timezone, thread_id,
                                      route=route, telegram_id=settings.google_calendar_owner_telegram_id)
        row["calendar_read"] = "availability" in result.get("read_results", {})
        row["mutation_before_approval"] = bool(getattr(calendar, "writes", 0))
        if result.get("__interrupt__"):
            proposal = ActionProposal.model_validate(result["proposal"])
            row["proposal_requires_hitl"] = requires_approval(proposal)
            rejected = await resume_workflow(graph, thread_id, {"decision": "reject"})
            row["rejected_without_mutation"] = (not getattr(calendar, "writes", 0) and
                                                 not rejected.get("__interrupt__"))
        else:
            row["rejected_without_mutation"] = not row["mutation_before_approval"]
    except Exception as exc:
        row["error_category"] = type(exc).__name__
    return row


async def _one_arm(case_id: str, prompt: str, seed_priority: str | None,
                   arm: str, tools: list, user_id: str, marker: str,
                   bound_tools: list | None = None) -> dict:
    started = time.perf_counter()
    row = {"case": case_id, "arm": arm, "tool_calls": [], "invalid_tool_calls": 0,
           "input_tokens": None, "total_tokens": None, "latency_ms": None,
           "final_rows": [], "task_success": False, "side_effect_count": None,
           "mutation_tool_calls": None,
           "wrong_tool_calls": [], "error_category": None}
    try:
        async with AsyncSessionLocal() as db:
            if seed_priority:
                await crud.create_todo(db, user_id, "V23 synthetic review notes", priority=seed_priority)
            await db.commit()
        initial_count = 1 if seed_priority else 0
        nodes = make_nodes(tools, llm_tools=bound_tools)
        todo_llm, execute_tools = nodes[2], nodes[3]
        messages = [HumanMessage(content=prompt)]
        input_tokens, total_tokens = [], []
        allowed = {"create": {"create_todo"}, "update": {"search_todos", "list_todos", "update_todo"},
                   "search": {"search_todos", "list_todos"}, "deadline": {"create_todo"},
                   "priority_tag": {"create_todo"}, "multi_step": {"create_todo", "update_todo", "set_reminder",
                                                           "search_todos", "list_todos"}}[case_id]
        for _ in range(7):
            state = {"messages": messages, "user_id": user_id, "telegram_id": marker,
                     "intent": "update" if case_id == "update" else "search" if case_id == "search" else "create"}
            answer = await todo_llm(state)
            response = answer["messages"][0]
            messages.append(response)
            usage = getattr(response, "usage_metadata", None)
            input_tokens.append(usage.get("input_tokens") if isinstance(usage, dict) else None)
            total_tokens.append(usage.get("total_tokens") if isinstance(usage, dict) else None)
            row["invalid_tool_calls"] += len(getattr(response, "invalid_tool_calls", []) or [])
            calls = getattr(response, "tool_calls", []) or []
            if not calls:
                break
            row["tool_calls"].extend(call["name"] for call in calls)
            row["wrong_tool_calls"].extend(call["name"] for call in calls if call["name"] not in allowed)
            tool_result = await execute_tools({**state, "messages": messages})
            messages.extend(tool_result["messages"])
        else:
            row["error_category"] = "tool_loop_limit"
        row["input_tokens"] = sum(input_tokens) if all(isinstance(x, int) for x in input_tokens) else None
        row["total_tokens"] = sum(total_tokens) if all(isinstance(x, int) for x in total_tokens) else None
        async with AsyncSessionLocal() as db:
            todos = await crud.list_todos(db, user_id, limit=None)
            row["final_rows"] = [{"title": t.title, "priority": t.priority.value,
                                   "tags": t.tags, "status": t.status.value,
                                   "due_local_date": t.due_date.astimezone(ZoneInfo(get_settings().default_timezone)).date().isoformat()
                                   if t.due_date else None} for t in todos]
        row["mutation_tool_calls"] = sum(name in {"create_todo", "update_todo", "set_reminder"}
                                         for name in row["tool_calls"])
        row["side_effect_count"] = (max(0, len(row["final_rows"]) - initial_count) +
                                    (1 if case_id == "update" and any(
                                        item["title"] == "V23 synthetic review notes" and item["priority"] == "high"
                                        for item in row["final_rows"]) else 0))
        target = next((item for item in row["final_rows"] if item["title"] == {
            "deadline": "V23 synthetic deadline", "priority_tag": "V23 synthetic tagged task",
            "multi_step": "V23 synthetic multi task"}.get(case_id, "V23 synthetic review notes")), None)
        tomorrow = (datetime.now(ZoneInfo(get_settings().default_timezone)) + timedelta(days=1)).date().isoformat()
        expected = {
            "create": bool(target) and len(row["final_rows"]) == 1,
            "update": bool(target) and target["priority"] == "high" and len(row["final_rows"]) == 1,
            "search": bool(target) and len(row["final_rows"]) == 1 and bool(row["tool_calls"]),
            "deadline": bool(target) and target["due_local_date"] == tomorrow,
            "priority_tag": bool(target) and target["priority"] == "high" and "v23" in (target["tags"] or []),
            "multi_step": bool(target) and target["due_local_date"] == tomorrow and
                          target["priority"] == "high" and "v23" in (target["tags"] or []),
        }
        expected_mutations = 0 if case_id == "search" else 1
        row["task_success"] = bool(expected[case_id] and
                                   row["mutation_tool_calls"] == expected_mutations and
                                   row["side_effect_count"] == expected_mutations and
                                   not row["wrong_tool_calls"] and
                                   not row["invalid_tool_calls"] and row["error_category"] is None)
    except Exception as exc:
        row["error_category"] = type(exc).__name__
    finally:
        row["latency_ms"] = round((time.perf_counter() - started) * 1000)
    return row


async def run(production_binding: bool = False) -> dict:
    settings = get_settings()
    discovered = await discover_calendar_tools(settings)
    read_tools = [tool for tool in discovered if tool.name in CALENDAR_READ_ONLY_TOOLS]
    if not read_tools:
        raise RuntimeError("Real Calendar MCP read schemas unavailable; no valid A/B")
    # A direct, read-only real MCP probe. The same typed cross-domain workflow
    # is used under both Todo bindings; no Calendar write is attempted here.
    calendar = MCPCalendarAdapter(discovered)
    zone = ZoneInfo(settings.default_timezone)
    day = (datetime.now(zone) + timedelta(days=3)).replace(hour=13, minute=0, second=0, microsecond=0)
    calendar_read = {"status": "NOT_MEASURED", "busy_count": None, "error_category": None}
    try:
        result = await calendar.availability(day.isoformat(), (day + timedelta(hours=2)).isoformat(),
                                             settings.default_timezone)
        calendar_read.update(status="PASS", busy_count=len(result["calendars"]["primary"]["busy"]))
    except Exception as exc:
        calendar_read["error_category"] = type(exc).__name__
    todo = make_todo_tools(AsyncSessionLocal)
    if production_binding:
        # Both arms execute the actual production node construction. Only the
        # same rollback switch used by build_graph changes LLM exposure.
        all_tools = todo + read_tools
        arms = {
            "current": (all_tools, todo_llm_tool_binding(todo, read_tools, False)),
            "candidate": (all_tools, todo_llm_tool_binding(todo, read_tools, True)),
        }
    else:
        arms = {"current": (todo + read_tools, None), "candidate": (todo, None)}
    rows = []
    for case_id, prompt, seed in CASES:
        marker = "v23-ab-" + uuid4().hex
        async with AsyncSessionLocal() as db:
            user = await crud.get_or_create_user(db, marker)
            user_id = user.id
            await db.commit()
        try:
            for arm, (tools, bound_tools) in arms.items():
                try:
                    rows.append(await _one_arm(case_id, prompt, seed, arm, tools, user_id, marker,
                                               bound_tools=bound_tools))
                finally:
                    # Same DB user_id and identical prompt are used for both
                    # arms; only isolated synthetic Todo rows are reset.
                    async with AsyncSessionLocal() as db:
                        await db.execute(delete(Todo).where(Todo.user_id == user_id))
                        await db.commit()
        finally:
            async with AsyncSessionLocal() as db:
                user = (await db.execute(select(User).where(User.telegram_id == marker))).scalar_one_or_none()
                if user is not None:
                    await db.delete(user)
                    await db.commit()
    cross_request = ("看看三天后下午有没有空，没有的话安排两小时写合成测试报告，"
                     "并创建一个四天后截止的合成测试报告待办。")
    cross_domain = []
    for arm in arms:
        cross_domain.append(await _cross_domain(arm, cross_request, calendar, "real_mcp"))
        cross_domain.append(await _cross_domain(arm, cross_request, BusyCalendar(), "synthetic_conflict"))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
            "provider": settings.llm_provider, "model": settings.llm_model,
            "evidence_class": "real LLM and isolated real PostgreSQL Todo; read-only real Calendar MCP probe",
            "production_binding_unchanged": not production_binding,
            "schema_counts": {key: len(bound or tools) for key, (tools, bound) in arms.items()},
            "schema_chars": {key: schema_chars(bound or tools) for key, (tools, bound) in arms.items()},
            "calendar_read": calendar_read, "cases": rows, "cross_domain": cross_domain}


def main() -> None:
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    production_binding = "--production" in sys.argv[1:]
    report = asyncio.run(run_with_provider_clients(run(production_binding=production_binding)))
    output = ROOT / "evals" / "reports" / ("v23_production_scoped_ab.json" if production_binding
                                            else "v23_full_tool_ab.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"full-loop A/B cases={len(report['cases'])} calendar_read={report['calendar_read']['status']}")


if __name__ == "__main__":
    main()
