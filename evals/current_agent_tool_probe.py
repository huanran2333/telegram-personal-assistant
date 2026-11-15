"""Current Agent first-step tool/schema probe; never invoke a business tool.

This is a model-behaviour diagnostic, not an end-to-end task benchmark. The
scoped arm uses the production ``todo_llm`` node and its current tool policy.
The all-schema arm receives the same messages but is deliberately hypothetical.
Only synthetic prompts and bounded metadata are written to the report.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage

from app.agent import nodes
from app.agent.capability_tools import CAPABILITY_TOOLS
from app.agent.read_tools import READ_TOOLS
from app.agent.planner import route_request
from app.agent.provider_clients import run_with_provider_clients
from app.agent.tools import make_todo_tools
from app.config import get_settings
from evals.v23_runner import RecordingStructuredLLM


ROOT = Path(__file__).resolve().parents[1]
OWNER = "itest-agent-tool-probe"
CASES = (
    ("todo_create", "创建待办：整理合成项目笔记。", "todo", None, {"create_todo"}),
    ("todo_timed", "明天下午两点到四点整理合成项目笔记，记为待办。", "todo", None, {"create_todo"}),
    ("personal_meeting", "明天下午两点开合成项目组会议。", "todo", None, {"create_todo"}),
    ("todo_list", "列出我今天的待办。", "todo", None, {"list_todos"}),
    ("calendar_events", "查看明天下午的日历事件。", "calendar_read", None, {"read_calendar_events"}),
    ("calendar_free", "明天下午我的日历有空吗？", "calendar_read", None, {"read_calendar_availability"}),
    ("gmail_unread", "查看我的未读 Gmail 邮件。", "email_unread", None, {"read_gmail_evidence"}),
    ("todo_delete", "删除名为合成项目笔记的待办。", "todo", "delete", {"prepare_action_request"}),
    ("standalone_calendar_create", "明天下午两点在 Google Calendar 创建独立事件：合成项目组会议。",
     "calendar_create", None,
     {"prepare_action_request"}),
    ("meeting_prep", "准备明天的合成项目组会议。", "meeting_prep", None, {"prepare_meeting_brief"}),
    ("read_cross_domain", "查明天下午的日历，并列出我的待办。", "read_cross_domain", None,
     {"read_calendar_events", "read_calendar_availability", "list_todos", "search_todos"}),
)


def usage(response) -> dict[str, int | None]:
    data = getattr(response, "usage_metadata", None) or {}
    return {"input": data.get("input_tokens"), "output": data.get("output_tokens")}


class CaptureModel:
    def __init__(self, base, records: list[dict]):
        self.base = base
        self.records = records

    def bind_tools(self, tools):
        bound = self.base.bind_tools(tools)
        names = [item.name for item in tools]
        records = self.records

        class Bound:
            async def ainvoke(self, messages):
                response = await bound.ainvoke(messages)
                records.append({"bound": names, "messages": messages,
                                "selected": [call["name"] for call in response.tool_calls],
                                "usage": usage(response)})
                return response

        return Bound()


def first_tool_ok(selected: list[str], expected: set[str]) -> bool:
    return bool(selected) and selected[0] in expected


def token_total(rows: list[dict], arm: str, field: str) -> int | None:
    values = [row.get(arm, {}).get("usage", {}).get(field) for row in rows]
    return sum(values) if values and all(isinstance(value, int) for value in values) else None


def route_token_total(rows: list[dict], field: str) -> int | None:
    values = [row.get("route_usage", {}).get(field) for row in rows]
    return sum(values) if values and all(isinstance(value, int) for value in values) else None


async def run() -> dict:
    settings = get_settings()
    original_builder = nodes._build_chat_llm
    original_settings = nodes.settings
    records: list[dict] = []
    all_tools = make_todo_tools(None) + list(READ_TOOLS) + list(CAPABILITY_TOOLS)

    def build(provider, model_name, routing=False):
        return CaptureModel(original_builder(provider, model_name, routing=routing), records)

    # Enable only the in-memory owner filter. No Calendar/Gmail tool is invoked.
    nodes.settings = settings.model_copy(update={
        "google_calendar_enabled": True, "google_calendar_owner_telegram_id": OWNER,
        "gmail_enabled": True, "gmail_owner_telegram_id": OWNER,
    })
    nodes._build_chat_llm = build
    try:
        todo_tools = [item for item in make_todo_tools(None) if item.name != "delete_todo"]
        todo_llm = nodes.make_nodes(
            make_todo_tools(None), llm_tools=todo_tools,
            read_tools=list(READ_TOOLS), capability_tools=list(CAPABILITY_TOOLS),
        )[2]
        rows = []
        for case_id, prompt, expected_intent, expected_operation, expected_tools in CASES:
            row = {"id": case_id, "expected_intent": expected_intent,
                   "expected_operation": expected_operation, "expected_first": sorted(expected_tools),
                   "actual_intent": None, "actual_operation": None, "route_ok": False,
                   "scoped": {}, "all_schema": {}, "error_category": None}
            try:
                route_model = RecordingStructuredLLM(
                    original_builder(settings.llm_provider, settings.llm_model, routing=True))
                route = await route_request(prompt, settings.default_timezone, llm=route_model)
                row["actual_intent"] = route.intent
                row["actual_operation"] = route.todo_operation
                row["route_ok"] = (route.intent == expected_intent and
                                   (expected_operation is None or route.todo_operation == expected_operation))
                row["route_usage"] = {
                    field: sum(call[field] for call in route_model.calls)
                    if route_model.calls and all(isinstance(call[field], int) for call in route_model.calls)
                    else None for field in ("input_tokens", "output_tokens")}
                state = {"messages": [HumanMessage(content=prompt)],
                         "intent": "delete" if route.intent == "todo" and route.todo_operation == "delete" else route.intent,
                         "precomputed_route": route.model_dump(), "active_request": prompt,
                         "active_turn_id": f"web-request:{case_id}", "request_origin": "web",
                         "user_id": OWNER, "telegram_id": OWNER, "read_tool_rounds": 0}
                before = len(records)
                await todo_llm(state)
                if len(records) != before + 1:
                    raise RuntimeError("NoSingleModelToolStep")
                scoped = records[-1]
                row["scoped"] = {"bound": scoped["bound"], "selected": scoped["selected"],
                                 "first_ok": first_tool_ok(scoped["selected"], expected_tools),
                                 "usage": scoped["usage"]}
                # Identical production messages, but all schemas visible. No calls execute.
                response = await original_builder(settings.llm_provider, settings.llm_model).bind_tools(
                    all_tools).ainvoke(scoped["messages"])
                selected = [call["name"] for call in response.tool_calls]
                row["all_schema"] = {"bound_count": len(all_tools), "selected": selected,
                                     "first_ok": first_tool_ok(selected, expected_tools),
                                     "usage": usage(response)}
            except Exception as exc:
                row["error_category"] = type(exc).__name__
            rows.append(row)
            print(f"{case_id}: route={row['actual_intent']} scoped={row['scoped'].get('selected')} "
                  f"all={row['all_schema'].get('selected')} error={row['error_category']}", flush=True)
    finally:
        nodes._build_chat_llm = original_builder
        nodes.settings = original_settings
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    return {"method": "synthetic real-model route + production first tool node; no business tool execution",
            "git_commit": commit, "model": settings.llm_model,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "rows": rows,
            "summary": {"cases": len(rows), "route_ok": sum(row["route_ok"] for row in rows),
                        "scoped_first_ok": sum(row["scoped"].get("first_ok", False) for row in rows),
                        "all_schema_first_ok": sum(row["all_schema"].get("first_ok", False) for row in rows),
                        "errors": sum(row["error_category"] is not None for row in rows),
                        "scoped_input_tokens": token_total(rows, "scoped", "input"),
                        "scoped_output_tokens": token_total(rows, "scoped", "output"),
                        "all_schema_input_tokens": token_total(rows, "all_schema", "input"),
                        "all_schema_output_tokens": token_total(rows, "all_schema", "output"),
                        "route_input_tokens": route_token_total(rows, "input_tokens"),
                        "route_output_tokens": route_token_total(rows, "output_tokens")},
            "limits": ["Single first tool step only; no tool or workflow executes.",
                       "The all-schema arm is hypothetical, not a deployable authorization policy.",
                       "One run per synthetic case; no statistical or production cost claim."]}


def main() -> None:
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = asyncio.run(run_with_provider_clients(run()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
