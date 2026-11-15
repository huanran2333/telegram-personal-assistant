"""Business-semantic context probes on real PostgreSQL checkpoint/profile/Todo.

All conversation facts and the DB user are synthetic. Calendar is a labeled
deterministic source fixture; this evaluator does not claim Calendar E2E.
No summarization or production context/binding policy is changed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import sys
from datetime import datetime, time as dt_time, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import select

from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.nodes import make_nodes, _build_chat_llm, chat_context_messages
from app.agent.provider_clients import run_with_provider_clients
from app.agent.email_workflow import synthesize_email
from app.agent.gmail import GmailMessage
from app.agent.planner import RoutedRequest
from app.agent.store import generate_store
from app.agent.tools import make_todo_tools
from app.agent.workflow import build_productivity_graph, resume_workflow, start_workflow
from app.config import get_settings
from app.db import crud
from app.db.models import User
from app.db.session import AsyncSessionLocal
from app.localization import load_preferences, reset_preferences, use_preferences
from evals.v23_long_context import replay_graph, snapshot, synthetic_turn
from evals.v23_runner import ROOT


class SyntheticCalendar:
    def __init__(self, event=None):
        self.event = event
        self.writes = 0

    async def search_events(self, query, start=None, end=None):
        return [self.event] if self.event else []

    async def availability(self, start, end, timezone):
        return {"calendars": {"primary": {"busy": []}}}

    async def create_event(self, args):
        self.writes += 1
        raise AssertionError("No Calendar write is allowed in semantic eval")


async def _chat_probe(nodes, store, user_id, messages, question, expected_fragment,
                      expected_language=None):
    response = await nodes[5]({"messages": messages + [HumanMessage(content=question)],
                               "user_id": user_id}, store)
    text = str(response["messages"][0].content)
    localized_aliases = {
        "Monday": ("Monday", "周一", "星期一"),
        "Wednesday": ("Wednesday", "周三", "星期三"),
        "slide": ("slide", "幻灯片", "演示文稿"),
    }
    correct = (bool(re.search(r"[\u4e00-\u9fff]", text)) if expected_language == "zh-CN"
               else any(alias.lower() in text.lower() for alias in
                        localized_aliases.get(expected_fragment, (expected_fragment,))))
    return {"correct": correct,
            "response_chars": len(text), "response_excerpt": text[:300],
            "fallback": response.get("last_action") == "chat_response_fallback",
            "history_messages_supplied_to_node": len(messages) + 1,
            "history_messages_sent_to_model": len(chat_context_messages(
                messages + [HumanMessage(content=question)]))}


async def _todo_authority(nodes, user_id, messages):
    state = {"messages": messages + [HumanMessage(content="List my Todos and their status")],
             "user_id": user_id, "telegram_id": "synthetic", "intent": "list"}
    tool_calls = []
    tool_outputs = []
    for _ in range(5):
        answer = await nodes[2](state)
        response = answer["messages"][0]
        state["messages"].append(response)
        calls = getattr(response, "tool_calls", []) or []
        if not calls:
            break
        tool_calls.extend(call["name"] for call in calls)
        results = await nodes[3](state)
        state["messages"].extend(results["messages"])
        tool_outputs.extend(str(item.content) for item in results["messages"] if isinstance(item, ToolMessage))
    return {"source": "TODO_DB", "correct": any("V23 semantic Todo" in item and
                                                   ("pending" in item.lower() or "待办" in item)
                                                   for item in tool_outputs),
            "tool_calls": tool_calls, "tool_pair_integrity":
            sum(len(getattr(item, "tool_calls", [])) for item in state["messages"] if isinstance(item, AIMessage)) ==
            sum(isinstance(item, ToolMessage) for item in state["messages"])}


async def run() -> dict:
    settings = get_settings()
    marker = "v23-semantic-" + uuid4().hex
    thread_id = marker + "-history"
    workflow_thread = marker + "-workflow"
    config = {"configurable": {"thread_id": thread_id}}
    tiers, probes = [], []
    restart_recovered = False
    user_id = None
    nodes = make_nodes(make_todo_tools(AsyncSessionLocal))
    try:
        async with AsyncSessionLocal() as db:
            user = await crud.get_or_create_user(db, marker)
            user_id = user.id
            await crud.create_todo(db, user_id, "V23 semantic Todo", priority="high")
            await db.commit()
        async with generate_store() as store:
            await store.aput(("profile", user_id), "user_profile",
                             {"assistant_language": "zh-CN", "preferences": ["concise answers"]})
            pref_token = use_preferences(await load_preferences(user_id, store))
            try:
                async with AsyncPostgresSaver.from_conn_string(settings.database_url_sync) as saver:
                    await saver.setup()
                    graph = replay_graph(saver)
                    for turn in range(1, 101):
                        items = ([HumanMessage(content="My private project codeword is cobalt-otter."),
                                  AIMessage(content="Acknowledged.")] if turn == 2 else
                                 [HumanMessage(content="The Todo is already completed."),
                                  AIMessage(content="Acknowledged.")] if turn == 15 else
                                 [HumanMessage(content="The restart-safe phrase is silver-harbor."),
                                  AIMessage(content="Noted.")] if turn == 100 else
                                 synthetic_turn(turn))
                        await graph.ainvoke({"messages": items}, config)
                        if turn == 50:
                            tiers.append(await snapshot(graph, config, turn, probe_llm=True))
                            history = (await graph.aget_state(config)).values["messages"]
                            probes.append({"turns": 50, "scenario": "old_conversation_only",
                                           "source": "CONVERSATION_HISTORY",
                                           **await _chat_probe(nodes, store, user_id, history,
                                                              "What was my project codeword?", "cobalt-otter")})
                async with AsyncPostgresSaver.from_conn_string(settings.database_url_sync) as saver:
                    graph = replay_graph(saver)
                    history = (await graph.aget_state(config)).values["messages"]
                    restart_recovered = len(history) >= 200
                    tiers.append(await snapshot(graph, config, 100, probe_llm=True))
                    probes.append({"turns": 100, "scenario": "recent_fact_after_saver_restart",
                                   "source": "RECENT_MESSAGES via CHECKPOINT",
                                   **await _chat_probe(nodes, store, user_id, history,
                                                      "What restart-safe phrase did I just mention?",
                                                      "silver-harbor")})
                    probes.append({"turns": 100, "scenario": "corrected_old_fact",
                                   "source": "CONVERSATION_HISTORY",
                                   **await _chat_probe(nodes, store, user_id, history,
                                                      "Which weekday is my project review after my correction?",
                                                      "Wednesday")})
                    for turn in range(101, 151):
                        items = ([HumanMessage(content="The temporary meeting label is orbit-echo."),
                                  AIMessage(content="Acknowledged.")] if turn == 149 else synthetic_turn(turn))
                        await graph.ainvoke({"messages": items}, config)
                    tiers.append(await snapshot(graph, config, 150, probe_llm=True))
                    history = (await graph.aget_state(config)).values["messages"]
                    probes.append({"turns": 150, "scenario": "old_conversation_only",
                                   "source": "CONVERSATION_HISTORY",
                                   **await _chat_probe(nodes, store, user_id, history,
                                                      "What was my project codeword?", "cobalt-otter")})
                    probes.append({"turns": 150, "scenario": "corrected_old_fact",
                                   "source": "CONVERSATION_HISTORY",
                                   **await _chat_probe(nodes, store, user_id, history,
                                                      "Which weekday is my project review after my correction?",
                                                      "Wednesday")})
                    probes.append({"turns": 150, "scenario": "recent_conversational_reference",
                                   "source": "RECENT_MESSAGES",
                                   **await _chat_probe(nodes, store, user_id, history,
                                                      "What temporary meeting label did I just mention?",
                                                      "orbit-echo")})
                    probes.append({"turns": 150, "scenario": "just_mentioned_fact",
                                   "source": "RECENT_MESSAGES",
                                   **await _chat_probe(nodes, store, user_id, history + [
                                       HumanMessage(content="My test codeword is blue-whale."),
                                       AIMessage(content="Noted.")],
                                       "What codeword did I just mention?", "blue-whale")})
                    probes.append({"turns": 150, "scenario": "recent_correction",
                                   "source": "RECENT_MESSAGES",
                                   **await _chat_probe(nodes, store, user_id, history + [
                                       HumanMessage(content="The review meeting is Friday."),
                                       AIMessage(content="Noted."),
                                       HumanMessage(content="Correction: it is next Monday."),
                                       AIMessage(content="Updated.")],
                                       "When is the review meeting?", "Monday")})
                    probes.append({"turns": 150, "scenario": "recent_pronoun",
                                   "source": "RECENT_MESSAGES",
                                   **await _chat_probe(nodes, store, user_id, history + [
                                       HumanMessage(content="Dr. Lin requested slide revisions."),
                                       AIMessage(content="Noted."),
                                       HumanMessage(content="She wants them before the meeting."),
                                       AIMessage(content="Understood.")],
                                       "What did she request?", "slide")})
                    probes.append({"turns": 150, "scenario": "irrelevant_recent_noise",
                                   "source": "RECENT_MESSAGES",
                                   **await _chat_probe(nodes, store, user_id, history + [
                                       HumanMessage(content="The temporary launch label is orbit-echo."),
                                       AIMessage(content="Noted."),
                                       HumanMessage(content="I will also buy coffee."),
                                       AIMessage(content="Okay."),
                                       HumanMessage(content="I will read a book."),
                                       AIMessage(content="Okay.")],
                                       "What was the temporary launch label?", "orbit-echo")})
                    probes.append({"turns": 150, "scenario": "profile_language_over_old_chat",
                                   "source": "PROFILE_STORE",
                                   **await _chat_probe(nodes, store, user_id, history + [
                                       HumanMessage(content="Please always reply in English."),
                                       AIMessage(content="Noted.")],
                                       "请用一句话问好。", "", expected_language="zh-CN")})
                    probes.append({"turns": 150, "scenario": "todo_db_over_stale_chat",
                                   **await _todo_authority(nodes, user_id, history)})
                    async with AsyncSessionLocal() as db:
                        todo = (await crud.list_todos(db, user_id, limit=None))[0]
                    probes.append({"turns": 150, "scenario": "todo_authoritative_row",
                                   "source": "TODO_DB", "correct": todo.status.value == "pending"})
                    profile = await store.aget(("profile", user_id), "user_profile")
                    probes.append({"turns": 150, "scenario": "profile_language_after_restart",
                                   "source": "PROFILE_STORE",
                                   "correct": profile.value.get("assistant_language") == "zh-CN" and
                                   (await load_preferences(user_id, store)).language == "zh-CN"})
                    source = GmailMessage(
                        message_id="synthetic-mail-1", thread_id="synthetic-thread-1",
                        sender="Colleague <colleague@example.invalid>", subject="Synthetic review date",
                        timestamp=datetime.now(timezone.utc),
                        body="The current synthetic review deadline is Thursday, not Tuesday.",
                        snippet="The current synthetic review deadline is Thursday, not Tuesday.")
                    summary = await synthesize_email(
                        [source], "What is the current synthetic review deadline?", 
                        _build_chat_llm(settings.llm_provider, settings.llm_model, routing=True))
                    probes.append({"turns": 150, "scenario": "gmail_source_over_stale_chat",
                                   "source": "GMAIL (synthetic evidence)",
                                   "correct": "synthetic-mail-1" in summary and
                                   "synthetic-thread-1" in summary and "Thursday" in summary})
                    await saver.adelete_thread(thread_id)
                # The workflow has its own typed state/checkpoint, not a copy of
                # the legacy message history. Use a long recent_context payload
                # to verify an interrupted proposal still resumes after reconnect.
                zone = ZoneInfo(settings.default_timezone)
                local_day = (datetime.now(zone) + timedelta(days=8)).date()
                event_start = datetime.combine(local_day, dt_time(14), zone)
                event = {"id": "synthetic-calendar-event", "summary": "Authoritative meeting",
                         "start": {"dateTime": event_start.isoformat()},
                         "end": {"dateTime": (event_start + timedelta(minutes=30)).isoformat()}}
                calendar = SyntheticCalendar(event)
                executor = ActionExecutor(calendar, SqlAlchemyActionLedger(AsyncSessionLocal), AsyncSessionLocal)
                async with AsyncPostgresSaver.from_conn_string(settings.database_url_sync) as saver:
                    workflow = build_productivity_graph(calendar, executor, saver,
                                                        calendar_owner_telegram_id=None)
                    read = await start_workflow(workflow, user_id, "Show my events on the stated date",
                                                settings.default_timezone, workflow_thread,
                                                route=RoutedRequest(intent="calendar_read",
                                                                    date_reference=local_day.isoformat()),
                                                recent_context=[{"role": "user", "content": "meeting canceled"}])
                    probes.append({"turns": 150, "scenario": "calendar_over_stale_chat",
                                   "source": "CALENDAR (synthetic adapter)",
                                   "correct": "Authoritative meeting" in read.get("response", "")})
                    proposal = await start_workflow(
                        workflow, user_id, "Create a synthetic meeting", settings.default_timezone,
                        workflow_thread + "-hitl",
                        route=RoutedRequest(intent="calendar_create", title="Synthetic pending meeting",
                                            date_reference=local_day.isoformat(), hour=16,
                                            duration_minutes=30),
                        recent_context=[{"role": "user", "content": f"Synthetic turn {i}"}
                                        for i in range(150)])
                    interrupted = bool(proposal.get("__interrupt__"))
                async with AsyncPostgresSaver.from_conn_string(settings.database_url_sync) as saver:
                    workflow = build_productivity_graph(calendar, executor, saver,
                                                        calendar_owner_telegram_id=None)
                    rejected = await resume_workflow(workflow, workflow_thread + "-hitl",
                                                     {"decision": "reject"}) if interrupted else {}
                    probes.append({"turns": 150, "scenario": "pending_hitl_after_restart",
                                   "source": "CHECKPOINT", "correct": interrupted and
                                   not rejected.get("__interrupt__") and calendar.writes == 0})
                    await saver.adelete_thread(workflow_thread)
                    await saver.adelete_thread(workflow_thread + "-hitl")
            finally:
                reset_preferences(pref_token)
                await store.adelete(("profile", user_id), "user_profile")
    finally:
        async with AsyncPostgresSaver.from_conn_string(settings.database_url_sync) as cleanup:
            for tid in (thread_id, workflow_thread, workflow_thread + "-hitl"):
                await cleanup.adelete_thread(tid)
        async with AsyncSessionLocal() as db:
            user = (await db.execute(select(User).where(User.telegram_id == marker))).scalar_one_or_none()
            if user is not None:
                await db.delete(user)
                await db.commit()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    return {"git_commit": commit, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_class": "real AsyncPostgresSaver/Store/Todo DB + real LLM; synthetic conversation/Calendar fixture",
            "restart_recovered": restart_recovered, "tiers": tiers, "scenarios": probes}


def main():
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    report = asyncio.run(run_with_provider_clients(run()))
    (ROOT / "evals" / "reports" / "v23_semantic_context.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"semantic context: scenarios={len(report['scenarios'])} restart={report['restart_recovered']}")


if __name__ == "__main__":
    main()
