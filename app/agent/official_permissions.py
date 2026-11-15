"""Server-owned visibility and execution permissions for the official Agent."""
from __future__ import annotations

import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from dateparser.search import search_dates
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import select

from app.db.models import Todo
from app.agent.state import require_agent_runtime_identity
from app.agent.business_clock import business_now
from app.agent.unified_tools import (
    GUARDED_NAMES, _request_segments, _todo_time_authorized, available_tools)


def _verified_calendar_slot(state: dict, args: dict, request: str) -> bool:
    """Match a same-turn, completed Calendar candidate to this exact Todo write.

    A sibling call in the current ToolNode batch has no ToolMessage yet, so it
    cannot satisfy this check.  Historical turns cannot supply fresh evidence.
    """
    title = args.get("title")
    quote = args.get("time_source_quote")
    start_text = args.get("planned_start_str")
    end_text = args.get("planned_end_str")
    if (not all(isinstance(value, str) and value for value in
                (title, quote, start_text, end_text)) or
            quote not in request or title not in quote or
            args.get("due_date_str") or args.get("reminder_at_str")):
        return False
    try:
        start, end = datetime.fromisoformat(start_text), datetime.fromisoformat(end_text)
        if start.tzinfo is None or end.tzinfo is None or end <= start:
            return False
    except ValueError:
        return False
    messages = list(state.get("messages") or ())
    turn_id = state.get("active_turn_id")
    begin = next((index for index in range(len(messages) - 1, -1, -1)
                  if isinstance(messages[index], HumanMessage) and
                  messages[index].id == turn_id), None)
    if begin is None:
        return False
    calls = {}
    for message in messages[begin + 1:]:
        if isinstance(message, AIMessage):
            calls.update({call["id"]: call for call in message.tool_calls
                          if call["name"] == "calendar_candidate_slots"})
        if not isinstance(message, ToolMessage) or message.status != "success":
            continue
        read = calls.get(message.tool_call_id)
        if read is None or read.get("args", {}).get("title") != title:
            continue
        try:
            slots = json.loads(message.content)
            if not isinstance(slots, list):
                continue
            for slot in slots:
                if (isinstance(slot, dict) and
                        datetime.fromisoformat(slot["start"]) == start and
                        datetime.fromisoformat(slot["end"]) == end):
                    return True
        except (TypeError, ValueError, KeyError):
            continue
    return False


def _verified_todo_version(state: dict, args: dict) -> bool:
    """Only a completed same-turn Todo result can supply a write baseline."""
    target, expected = args.get("todo_id"), args.get("expected_version")
    if (not isinstance(target, str) or len(target) < 8 or
            not isinstance(expected, int) or isinstance(expected, bool) or
            expected < 1):
        return False
    messages = list(state.get("messages") or ())
    turn_id = state.get("active_turn_id")
    begin = next((index for index in range(len(messages) - 1, -1, -1)
                  if isinstance(messages[index], HumanMessage) and
                  messages[index].id == turn_id), None)
    if begin is None:
        return False
    calls = {}
    for message in messages[begin + 1:]:
        if isinstance(message, AIMessage):
            calls.update({call["id"]: call["name"] for call in message.tool_calls})
        if (not isinstance(message, ToolMessage) or message.status != "success" or
                calls.get(message.tool_call_id) not in {
                    "list_todos", "search_todos", "create_todo", "update_todo",
                    "set_reminder", "cancel_reminder"}):
            continue
        artifact = message.artifact
        versions = artifact.get("todo_versions") if isinstance(artifact, dict) else None
        if (isinstance(versions, dict) and any(
                isinstance(todo_id, str) and todo_id.startswith(target) and
                isinstance(version, int) and not isinstance(version, bool) and
                version == expected
                for todo_id, version in versions.items())):
            return True
    return False


class OfficialToolAuthorization(AgentMiddleware):
    def __init__(self, tools: list, *, settings, session_factory=None):
        super().__init__()
        self.tools = tools
        self.settings = settings
        self.session_factory = session_factory

    async def _reminder_authorized(self, state: dict, runtime, args: dict) -> bool:
        """Require one owned target and one matching time in an explicit reminder clause.

        Ambiguous or composite clauses fail closed. This is deliberately narrower
        than a general Chinese intent parser and does not authorize other Todo writes.
        """
        request = state.get("active_request")
        quote, todo_id, value = (args.get("time_source_quote"),
                                 args.get("todo_id"), args.get("remind_at_str"))
        if (self.session_factory is None or not isinstance(request, str) or
                not all(isinstance(item, str) and item for item in
                        (quote, todo_id, value)) or len(todo_id) < 8 or
                len(quote) > 120 or request.count(quote) != 1):
            return False
        principal = require_agent_runtime_identity(state, runtime)
        async with self.session_factory() as db:
            rows = (await db.execute(select(Todo).where(
                Todo.user_id == principal["user_id"],
                Todo.id.startswith(todo_id, autoescape=True)).limit(2))).scalars().all()
        if len(rows) != 1:
            return False
        title = rows[0].title
        quote_at = request.index(quote)
        segments = _request_segments(request)
        covered = [(part, begin, end) for part, begin, end in segments
                   if begin <= quote_at and quote_at + len(quote) <= end]
        if len(covered) != 1:
            return False
        clause = covered[0][0]
        if (clause.count(title) != 1 or "提醒" not in clause or
                len(re.findall(r"『[^』]+』", clause)) > 1):
            return False
        residual = clause.replace(title, "", 1)
        if (re.search(r"不要|不需|不用|取消|别|暂不|无需", residual) or
                re.search(r"会议|日历|日程|活动", residual) or
                not re.search(r"设置|设|更改|修改|调整|提醒我|提醒", residual)):
            return False
        try:
            proposed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if proposed.tzinfo is None:
                return False
            zone = ZoneInfo(self.settings.default_timezone)
            # A model may quote the whole clause. Numeric IDs in the Todo title
            # are not candidate reminder times; inspect the explicit clock only.
            clocks = re.findall(
                r"\d{4}年\s*\d{1,2}月\s*\d{1,2}日\s*\d{1,2}(?::\d{2}|[点时]\d{0,2}分?)"
                r"|\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}", quote)
            if len(clocks) > 1:
                return False
            temporal = clocks[0] if clocks else quote.replace(title, "", 1)
            found = search_dates(temporal, settings={
                "TIMEZONE": self.settings.default_timezone,
                "RETURN_AS_TIMEZONE_AWARE": True,
                "RELATIVE_BASE": business_now(zone),
            }) or []
            times = {item.astimezone(zone).replace(second=0, microsecond=0)
                     for _, item in found}
            return times == {proposed.astimezone(zone).replace(second=0, microsecond=0)}
        except (ValueError, TypeError, KeyError):
            return False

    def _allowed(self, state: dict, runtime) -> set[str]:
        principal = require_agent_runtime_identity(state, runtime)
        allowed = {tool.name for tool in available_tools(
            self.tools, telegram_id=principal["telegram_id"],
            settings=self.settings)}
        if principal["origin"] != "telegram":
            # Browser forms execute explicit writes directly. A Web Agent
            # request may read, but must not create a new HITL approval.
            allowed.difference_update(GUARDED_NAMES)
            allowed.discard("propose_todo_delete")
        return allowed

    async def awrap_model_call(self, request, handler):
        allowed = self._allowed(request.state, request.runtime)
        return await handler(request.override(
            tools=[tool for tool in request.tools if tool.name in allowed]))

    async def awrap_tool_call(self, request, handler):
        call = request.tool_call
        if call["name"] not in self._allowed(request.state, request.runtime):
            return ToolMessage(
                content="Tool unavailable for the authenticated user; nothing executed.",
                tool_call_id=call["id"], status="error")
        if call["name"] == "search_memory":
            original_request = request.state.get("active_request")
            args = call.get("args")
            reason = None
            if not isinstance(original_request, str) or not original_request or not isinstance(args, dict):
                reason = "AgentActiveRequestMissing"
            if reason is not None:
                return ToolMessage(content=f"Tool request rejected ({reason}); nothing executed.",
                                   tool_call_id=call["id"], status="error")
        if call["name"] in {"create_todo", "update_todo", "set_reminder",
                            "cancel_reminder"}:
            original_request = request.state.get("active_request")
            args = call.get("args")
            reason = None
            if not isinstance(original_request, str) or not original_request or not isinstance(args, dict):
                reason = "AgentActiveRequestMissing"
            elif call["name"] == "create_todo":
                if any(args.get(field) for field in
                       ("due_date_str", "planned_start_str", "planned_end_str", "reminder_at_str")):
                    quote = args.get("time_source_quote")
                    if (not isinstance(quote, str) or
                            not (_todo_time_authorized(original_request, quote) or
                                 _verified_calendar_slot(request.state, args, original_request))):
                        reason = "TodoTimeNotAuthorized"
            elif call["name"] == "update_todo" and any(
                    args.get(field) for field in
                    ("due_date_str", "planned_start_str", "planned_end_str")):
                quote = args.get("time_source_quote")
                if not isinstance(quote, str) or not _todo_time_authorized(original_request, quote):
                    reason = "TodoTimeNotAuthorized"
            elif call["name"] == "set_reminder":
                if not await self._reminder_authorized(request.state, request.runtime, args):
                    reason = "TodoTimeNotAuthorized"
            if reason is not None:
                detail = "本轮请求上下文无法核实。"
                next_step = "请重新确认本轮请求后重试。"
                if reason == "TodoTimeNotAuthorized":
                    quote = args.get("time_source_quote")
                    if not isinstance(quote, str) or not quote.strip():
                        detail = "缺少该待办时间或提醒的原话依据。"
                        next_step = "请准确引用本轮用户给出的时间；未给出则先询问用户。"
                    elif quote not in original_request:
                        detail = "所给时间依据不在本轮用户原话中。"
                        next_step = "请核对原话中的目标与时间，不能补造时间依据。"
                    elif call["name"] == "create_todo" and any(
                            args.get(field) for field in
                            ("planned_start_str", "planned_end_str")):
                        detail = "用户原话未直接授权该时间，且本轮没有与该待办和时段匹配的成功日历候选结果。"
                        next_step = "请先查询该待办可用时段，再根据成功的候选结果排期。"
                    else:
                        detail = "该目标的时间或提醒依据未通过核实。"
                        next_step = "请重新查询目标并核对提醒或计划时间。"
                return ToolMessage(
                    content=(f"Tool request rejected ({reason}): {detail}"
                             f"本次工具写入未执行；{next_step}"),
                    tool_call_id=call["id"], status="error")
            if call["name"] in {"update_todo", "set_reminder", "cancel_reminder"}:
                if not _verified_todo_version(request.state, args):
                    return ToolMessage(
                        content=("Tool request rejected (TodoVersionEvidenceMissing): "
                                 "本轮尚无该待办及其版本的成功查询或创建结果；"
                                 "本次写入未执行。请先查询目标，再使用返回的 ID 和版本重试。"),
                        tool_call_id=call["id"], status="error")
        return await handler(request)
