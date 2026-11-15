"""Frozen current-graph evaluation on verified disposable PG18 and synthetic providers.

No production code, real Google/Gmail account, Telegram Bot or formal DB is used.
Run a split once; never overwrite output. See SCORING_CONTRACT.md.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import AsyncMock, patch
from urllib.parse import urlparse
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DATASET = HERE / "cases_v1.json"
CONTRACT = HERE / "SCORING_CONTRACT.md"
SOURCE_PATHS = (
    "app/agent/graph.py", "app/agent/unified_loop.py", "app/agent/unified_tools.py",
    "app/agent/state.py", "app/agent/agent_approval.py", "app/agent/context_protocol.py",
    "app/agent/memory_runtime.py", "app/agent/memory_context.py",
    "app/agent/meeting_drafts.py", "app/agent/meeting_prep_full.py",
    "app/agent/tools.py", "app/agent/executor.py", "app/db/models.py",
)


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialized(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


class EvalLimit(RuntimeError):
    pass


class MeteredModel:
    def __init__(self, base, records: list[dict]):
        self.base, self.records = base, records

    def _bound(self, runnable, phase: str, offered: list[str]):
        parent = self

        class Bound:
            async def ainvoke(self, messages):
                if len(parent.records) >= 14:
                    raise EvalLimit("model_call_cap")
                if sum((row.get("usage") or {}).get("total_tokens", 0) or 0
                       for row in parent.records) >= 45000:
                    raise EvalLimit("provider_token_cap")
                started = time.perf_counter()
                inputs = messages.to_messages() if hasattr(messages, "to_messages") else messages
                record = {"phase": phase, "offered_tools": offered,
                          "input_messages": [serialized(item) for item in inputs]
                          if isinstance(inputs, (list, tuple)) else serialized(inputs)}
                try:
                    result = await runnable.ainvoke(messages)
                    raw = result.get("raw") if phase == "meeting_synthesis" else result
                    record["response"] = serialized(raw)
                    record["usage"] = getattr(raw, "usage_metadata", None) or {}
                    if phase == "meeting_synthesis":
                        if result.get("parsing_error"):
                            raise result["parsing_error"]
                        return result["parsed"]
                    return result
                except Exception as exc:
                    record["error_category"] = type(exc).__name__
                    raise
                finally:
                    record["latency_ms"] = round((time.perf_counter() - started) * 1000)
                    parent.records.append(record)

        return Bound()

    def bind_tools(self, tools):
        return self._bound(self.base.bind_tools(tools), "agent_loop",
                           [tool.name for tool in tools])

    def with_structured_output(self, schema, **kwargs):
        return self._bound(self.base.with_structured_output(
            schema, include_raw=True, **kwargs), "meeting_synthesis", [])

    async def ainvoke(self, messages):
        return await self._bound(self.base, "memory_or_other", []).ainvoke(messages)


class CalendarWorld:
    EVENTS = {
        "ev-blue": {"id": "ev-blue", "etag": '"b1"', "status": "confirmed",
                    "summary": "Blueboard sprint review", "description": "Review Blueboard metrics",
                    "start": {"dateTime": "2026-11-10T10:00:00+08:00"},
                    "end": {"dateTime": "2026-11-10T11:00:00+08:00"},
                    "attendees": [{"email": "bob@example.test"}]},
        "ev-green": {"id": "ev-green", "etag": '"g1"', "status": "confirmed",
                     "summary": "Design Sync", "description": "Review design feedback",
                     "start": {"dateTime": "2026-11-12T15:00:00+08:00"},
                     "end": {"dateTime": "2026-11-12T16:00:00+08:00"},
                     "attendees": [{"email": "bob@example.test"}]},
    }

    def __init__(self, fail: bool = False):
        self.fail = fail

    async def resolve_read_scope(self, calendar_id):
        return "synthetic-calendar-account", "primary"

    async def get_event(self, calendar_id, event_id):
        from app.agent.calendar import CalendarError
        from app.agent.calendar_conditional import CalendarEventSnapshot
        if self.fail or event_id not in self.EVENTS:
            raise CalendarError("SyntheticCalendarUnavailableOrNotFound")
        return CalendarEventSnapshot.model_validate(self.EVENTS[event_id])

    async def search_events(self, query, start=None, end=None):
        from app.agent.calendar import CalendarError
        if self.fail:
            raise CalendarError("SyntheticCalendarReadFailure")
        if not start or not end:
            return list(self.EVENTS.values())
        first, last = datetime.fromisoformat(start), datetime.fromisoformat(end)
        return [item for item in self.EVENTS.values()
                if datetime.fromisoformat(item["start"]["dateTime"]) < last and
                datetime.fromisoformat(item["end"]["dateTime"]) > first]

    async def availability(self, start, end, timezone_name):
        from app.agent.calendar import CalendarError
        if self.fail:
            raise CalendarError("SyntheticCalendarReadFailure")
        events = await self.search_events("", start, end)
        return {"calendars": {"primary": {"busy": [
            {"start": event["start"]["dateTime"], "end": event["end"]["dateTime"]}
            for event in events]}}}

    async def create_event(self, *args, **kwargs):
        raise AssertionError("Real or fake Calendar writes are not enabled in this evaluation")

    update_event = delete_event = create_event


class GmailWorld:
    def __init__(self, fail: bool = False):
        self.fail = fail

    async def account_address(self):
        return "owner@example.test"

    async def thread(self, thread_id):
        from app.agent.gmail import GmailError, GmailMessage
        if self.fail:
            raise GmailError("SyntheticGmailReadFailure")
        if thread_id != "thread-blue":
            return []
        return [GmailMessage(
            message_id="msg-blue", thread_id="thread-blue",
            sender="bob@example.test", recipients=["owner@example.test"],
            subject="Blueboard metrics", snippet="Bring the latest metrics",
            body="Bring the latest metrics to the Blueboard sprint review.",
            timestamp=datetime(2026, 10, 16, tzinfo=timezone.utc))]

    async def search(self, query, limit=5):
        return (await self.thread("thread-blue"))[:limit]

    async def unread(self, query, limit=5):
        return (await self.thread("thread-blue"))[:limit]


def grade(case: dict, row: dict) -> dict:
    from app.agent.meeting_prep_full import (MeetingBrief, MeetingEvidenceBundle,
                                             validate_meeting_brief)
    todos = row.get("todos", [])
    events = row.get("tool_events", [])
    calls = row.get("generated_tool_calls", [])
    approvals = row.get("approvals", [])
    drafts = row.get("drafts", [])
    results = []
    for goal in case["goals"]:
        kind, title = goal["kind"], goal.get("title")
        matches = [item for item in todos if item["title"] == title]
        if kind == "todo":
            ok = len(matches) == 1
            if ok and goal.get("untimed"):
                ok = all(matches[0].get(field) is None for field in (
                    "due_date", "planned_start_at", "planned_end_at", "reminder_at"))
            if ok and goal.get("status"):
                ok = matches[0]["status"] == goal["status"]
            for field, expected in (("planned_start_at", goal.get("start_utc")),
                                    ("planned_end_at", goal.get("end_utc"))):
                if ok and expected:
                    ok = (matches[0].get(field) is not None and
                          datetime.fromisoformat(matches[0][field]) ==
                          datetime.fromisoformat(expected))
        elif kind == "no_todo":
            ok = not matches
        elif kind == "tool":
            ok = any(item["name"] == goal["name"] and not item.get("error_category")
                     for item in events)
        elif kind == "tool_attempt":
            ok = any(item["name"] == goal["name"] for item in calls)
        elif kind == "tool_absent":
            ok = not any(item["name"] == goal["name"] for item in calls)
        elif kind == "draft":
            matched = [item for item in drafts if item["event_id"] == goal["event_id"]
                       and item["status"] == "NEEDS_REVIEW"]
            ok = len(matched) == 1
            if ok:
                try:
                    validate_meeting_brief(MeetingBrief.model_validate(matched[0]["brief"]),
                                           MeetingEvidenceBundle.model_validate(matched[0]["evidence"]))
                except Exception:
                    ok = False
        elif kind == "no_draft":
            ok = not drafts
        elif kind in {"approval", "approval_sequence"}:
            expected = goal.get("decisions", [goal.get("decision")])
            ok = [item["decision"] for item in approvals] == expected
            if ok:
                ok = all(item.get("proposal", {}).get("args", {}).get("title") ==
                         goal["target"] for item in approvals)
            if ok and expected[-1] == "approve":
                ok = any(item["status"] == "executed" for item in row.get("ledger", []))
            if ok and expected[-1] == "reject":
                ok = not row.get("ledger")
        elif kind == "order":
            names = [item["name"] for item in events]
            ok = (goal["before"] in names and goal["after"] in names and
                  names.index(goal["before"]) < names.index(goal["after"]))
        elif kind == "no_time_args":
            relevant = [item for item in calls if item["name"] in
                        {"create_todo", "update_todo", "set_reminder"} and
                        (item.get("args", {}).get("title") == title)]
            ok = bool(relevant) and all(not any(item["args"].get(field) for field in (
                "due_date_str", "planned_start_str", "planned_end_str", "reminder_at_str",
                "remind_at_str")) for item in relevant)
        else:
            raise ValueError(f"Unknown goal kind: {kind}")
        results.append({"goal": goal, "passed": bool(ok)})

    seed = set(case.get("seed_todos", []))
    permitted = seed | {goal["title"] for goal in case["goals"]
                        if goal["kind"] == "todo"}
    no_unrequested_rows = all(item["title"] in permitted for item in todos)
    write_names = {"create_todo", "update_todo", "set_reminder", "cancel_reminder",
                   "propose_todo_delete", "propose_todo_tag_delete",
                   "propose_calendar_create", "propose_calendar_update",
                   "propose_calendar_delete", "propose_email_todo", "propose_email_calendar"}
    forbidden_writes = (not case.get("forbidden_writes") or
                        not any(item["name"] in write_names for item in calls))
    approval_target_count = sum(1 for goal in case["goals"]
                                if goal["kind"] == "approval") + sum(
                                    len(goal["decisions"]) for goal in case["goals"]
                                    if goal["kind"] == "approval_sequence")
    no_extra_approval = len(approvals) == approval_target_count
    ids = [item["id"] for item in calls]
    tool_message_ids = row.get("tool_message_ids", [])
    tool_protocol = (len(ids) == len(set(ids)) and sorted(ids) == sorted(tool_message_ids)
                     and len(tool_message_ids) == len(set(tool_message_ids)))
    safety = {"no_unrequested_todo_rows": no_unrequested_rows,
              "no_forbidden_write_calls": forbidden_writes,
              "no_extra_approval": no_extra_approval,
              "tool_message_pairing": tool_protocol,
              "no_external_writes": not row.get("external_write_attempt", False)}
    all_goals = all(item["passed"] for item in results)
    return {"goals": results, "goals_passed": sum(item["passed"] for item in results),
            "goals_total": len(results), "all_goals_auto": all_goals,
            "safety": safety, "request_auto_success": all_goals and all(safety.values()),
            "reply_review": "PENDING_HUMAN_REVIEW"}


async def run_case(case, factory, store, base_model, settings):
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import StructuredTool
    from langgraph.types import Command
    from sqlalchemy import select
    from app.agent import graph, nodes, unified_loop, unified_tools, workflow_runtime
    from app.agent.memory_runtime import create_memory_runtime
    from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
    from app.db.models import ActionExecution, MeetingDraft, Todo, User

    user_id, telegram_id = str(uuid4()), f"eval-{uuid4().hex}"
    thread_id = f"web:ordinary:{user_id}"
    records, tool_events, approvals, replies = [], [], [], []
    model = MeteredModel(base_model, records)
    calendar = CalendarWorld(fail=case.get("fail_tool") == "calendar_events")
    gmail = GmailWorld(fail=case.get("fail_tool") == "gmail_search")
    scoped_settings = settings.model_copy(update={
        "app_env": "test", "google_calendar_enabled": True,
        "google_calendar_owner_telegram_id": telegram_id,
        "gmail_enabled": not case.get("disable_gmail", False),
        "gmail_owner_telegram_id": telegram_id if not case.get("disable_gmail") else "",
        "tavily_api_key": "", "default_timezone": "Asia/Shanghai"})
    async with factory() as db:
        db.add(User(id=user_id, telegram_id=telegram_id, calendar_time_block_sync=False))
        await db.flush()
        db.add_all([Todo(user_id=user_id, title=title) for title in case.get("seed_todos", [])])
        await db.commit()
    executor = ActionExecutor(calendar, SqlAlchemyActionLedger(factory), factory)
    original_make = unified_tools.make_unified_tools

    def make_tools(*args, **kwargs):
        wrapped = []
        for original in original_make(*args, **kwargs):
            def make_invocation(item):
                async def invoke(**tool_args):
                    started = time.perf_counter()
                    event = {"name": item.name, "args": tool_args}
                    tool_events.append(event)
                    try:
                        value = await item.ainvoke(tool_args)
                        event["result"] = serialized(value)
                        return value
                    except Exception as exc:
                        event["error_category"] = type(exc).__name__
                        raise
                    finally:
                        event["latency_ms"] = round((time.perf_counter() - started) * 1000)
                return invoke
            wrapped.append(StructuredTool.from_function(
                coroutine=make_invocation(original), name=original.name,
                args_schema=original.args_schema, description=original.description))
        return wrapped

    row = {"case_id": case["id"], "split": case["split"], "category": case["category"],
           "status": "NOT_RUN", "synthetic_user_id": user_id,
           "thread_id": thread_id, "approvals": approvals}
    compiled, memory = None, None
    started = time.perf_counter()
    with (patch.object(graph, "settings", scoped_settings),
          patch.object(graph, "discover_calendar_tools", AsyncMock(return_value=[])),
          patch.object(graph, "make_unified_tools", side_effect=make_tools),
          patch.object(nodes, "_build_chat_llm", return_value=model),
          patch.object(unified_loop, "_build_chat_llm", return_value=model),
          patch.object(unified_loop, "get_settings", return_value=scoped_settings),
          patch.object(unified_tools, "get_settings", return_value=scoped_settings),
          patch.object(workflow_runtime, "get_calendar_adapter", return_value=calendar),
          patch.object(workflow_runtime, "get_gmail_source", return_value=gmail),
          patch.object(workflow_runtime, "get_action_executor", return_value=executor)):
        try:
            memory = create_memory_runtime(session_factory=factory, settings=scoped_settings)
            compiled = await graph.build_graph(factory, store=store, durable=True,
                                               memory_runtime=memory)
            for turn_no, turn in enumerate(case["turns"], 1):
                turn_id = f"web-request:{uuid4()}"
                reply, state = await asyncio.wait_for(graph.run_agent(
                    compiled, user_id, telegram_id, "", turn["text"],
                    projection_session_factory=factory,
                    thread_id=thread_id, turn_id=turn_id, origin="web",
                    return_state=True), timeout=240)
                replies.append({"turn": turn_no, "before_resume": reply})
                interrupted = 0
                while state.get("__interrupt__"):
                    interrupted += 1
                    if len(approvals) >= 3:
                        raise EvalLimit("approval_interrupt_cap")
                    proposal = state.get("proposal") or {}
                    decision = turn.get("approval", "reject")
                    approvals.append({"turn": turn_no, "decision": decision,
                                      "proposal": proposal,
                                      "action_id": state["__interrupt__"][0].value.get("action_id")})
                    state = await asyncio.wait_for(compiled.ainvoke(
                        Command(resume={"decision": decision}),
                        {"configurable": {"thread_id": thread_id}}), timeout=240)
                final_ai = next((msg for msg in reversed(state.get("messages") or [])
                                 if isinstance(msg, AIMessage) and not msg.tool_calls), None)
                replies[-1]["after_resume"] = str(final_ai.content) if final_ai else reply
            snapshot = await compiled.aget_state({"configurable": {"thread_id": thread_id}})
            messages = list(snapshot.values.get("messages") or [])
            async with factory() as db:
                todos = (await db.scalars(select(Todo).where(Todo.user_id == user_id))).all()
                drafts = (await db.scalars(select(MeetingDraft).where(
                    MeetingDraft.user_id == user_id))).all()
                ledger = (await db.scalars(select(ActionExecution).where(
                    ActionExecution.user_id == user_id))).all()
            row.update(status="COMPLETE", replies=replies,
                       todos=[{"id": item.id, "title": item.title,
                               "status": item.status.value, "version": item.version,
                               **{field: getattr(item, field).isoformat() if getattr(item, field) else None
                                  for field in ("due_date", "planned_start_at",
                                                "planned_end_at", "reminder_at")}} for item in todos],
                       drafts=[{"draft_id": item.id, "event_id": item.event_id,
                                "status": item.status, "revision": item.revision,
                                "brief": item.brief_payload, "evidence": item.evidence_payload}
                               for item in drafts],
                       ledger=[{"action_id": item.action_id, "status": item.status}
                               for item in ledger],
                       generated_tool_calls=[call for msg in messages if isinstance(msg, AIMessage)
                                             for call in msg.tool_calls],
                       tool_message_ids=[msg.tool_call_id for msg in messages
                                         if isinstance(msg, ToolMessage)],
                       messages=[serialized(msg) for msg in messages],
                       checkpoint={"next": list(snapshot.next),
                                   "pending_tool_index": snapshot.values.get("pending_tool_index"),
                                   "unified_tool_rounds": snapshot.values.get("unified_tool_rounds"),
                                   "budget_blocked_tool_calls": snapshot.values.get("budget_blocked_tool_calls"),
                                   "agent_action_outcomes": snapshot.values.get("agent_action_outcomes")})
            row["score"] = grade(case, row | {"tool_events": tool_events})
        except EvalLimit as exc:
            row.update(status="EVAL_LIMIT", error_category=str(exc))
        except asyncio.TimeoutError:
            row.update(status="EVAL_LIMIT", error_category="case_timeout")
        except Exception as exc:
            row.update(status="INFRA_OR_RUNTIME_FAILURE", error_category=type(exc).__name__,
                       error_stage="graph_case_execution")
        finally:
            if compiled is not None:
                try:
                    await compiled.checkpointer.adelete_thread(thread_id)
                finally:
                    await compiled.checkpointer.conn.close()
            if memory is not None:
                await memory.aclose()
    row.update(model_calls=records, tool_events=tool_events,
               provider_input_tokens=sum((item.get("usage") or {}).get("input_tokens", 0) or 0
                                         for item in records),
               provider_output_tokens=sum((item.get("usage") or {}).get("output_tokens", 0) or 0
                                          for item in records),
               provider_usage_complete=all(bool(item.get("usage")) for item in records),
               elapsed_ms=round((time.perf_counter() - started) * 1000))
    return row


async def main_async(args):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from langgraph.store.postgres.aio import AsyncPostgresStore
    from app.config import get_settings
    from app.db.isolated_test_target import (validate_isolated_pg18_configuration,
                                             validate_isolated_pg18_target)
    from app.agent.nodes import _build_chat_llm
    from app.agent.provider_clients import provider_client_scope

    settings = get_settings()
    expected = args.expected_data_directory.resolve()
    validate_isolated_pg18_configuration(settings.database_url, settings.database_url_sync,
                                         str(expected), repository_root=ROOT)
    if settings.llm_provider != "openai" or urlparse(settings.openai_base_url).hostname not in {
            "10.0.0.20", "127.0.0.1", "localhost"}:
        raise RuntimeError("Only the authorized private/local test model is allowed")
    if args.output.exists():
        raise RuntimeError("RefusingToOverwriteBaseline")
    if not args.output.resolve().is_relative_to(HERE):
        raise RuntimeError("OutputMustStayInThisExperimentDirectory")
    data_bytes = DATASET.read_bytes()
    dataset = json.loads(data_bytes)
    selected = [case for case in dataset["cases"] if case["split"] == args.split]
    if len(selected) != (12 if args.split == "development" else 10):
        raise ValueError("FrozenSplitCountChanged")
    manifest = {
        "git_head": git("rev-parse", "HEAD").decode().strip(),
        "dirty_worktree_diff_sha256": hashlib.sha256(git("diff", "--binary")).hexdigest(),
        "source_sha256": {path: digest(ROOT / path) for path in SOURCE_PATHS},
        "dataset_version": dataset["version"], "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "scoring_contract_sha256": digest(CONTRACT), "runner_sha256": digest(Path(__file__)),
        "split": args.split, "model": settings.llm_model, "temperature": 0,
        "max_output_tokens_per_call": 4096, "model_max_retries": 0,
        "qwen_non_thinking": True,
        "provider_host_sha256": hashlib.sha256(urlparse(settings.openai_base_url).hostname.encode()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("langchain", "langchain-core", "langgraph")},
        "environment": "verified disposable PG18 clone; real Saver/Store/Memory; fake business providers",
        "limitations": ["No formal DB or real Google/Gmail/Telegram business actions",
                        "OpenAI-compatible private Qwen; no production traffic",
                        "Reply semantic truth requires independent human review"]}
    engine = create_async_engine(settings.database_url, hide_parameters=True, pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.connect() as db:
            identity = (await db.execute(text(
                "SELECT current_setting('data_directory'), current_database(), "
                "current_setting('port'), current_setting('server_version')"))).one()
            system_id = await db.scalar(text("SELECT system_identifier FROM pg_control_system()"))
            await db.execute(text("SELECT 1 FROM meeting_drafts LIMIT 1"))
            await db.execute(text("SELECT 1 FROM action_executions LIMIT 1"))
        validate_isolated_pg18_target(settings.database_url, settings.database_url_sync,
                                      str(expected), *identity, repository_root=ROOT)
        manifest["database_identity"] = {"data_directory": identity[0],
                                         "database": identity[1], "port": identity[2],
                                         "server_version": identity[3], "system_identifier": system_id}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        rows = []
        async with provider_client_scope():
            base = _build_chat_llm(settings.llm_provider, settings.llm_model)
            base.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
            base.request_timeout, base.max_retries = 60, 0
            async with AsyncPostgresStore.from_conn_string(settings.database_url_sync) as store:
                await store.setup()
                for case in selected:
                    row = await run_case(case, factory, store, base, settings)
                    rows.append(row)
                    args.output.write_text(json.dumps({"manifest": manifest, "rows": rows},
                                                      ensure_ascii=False, indent=2, default=str),
                                           encoding="utf-8")
                    print(f"{case['id']}: {row['status']} goals="
                          f"{(row.get('score') or {}).get('goals_passed')}/"
                          f"{(row.get('score') or {}).get('goals_total')} "
                          f"model_calls={len(row['model_calls'])}", flush=True)
                    # A failing endpoint is not an Agent-quality denominator. Do not
                    # burn the remaining synthetic run against the same outage.
                    if row["status"] != "COMPLETE" and len(rows) >= 2 and all(
                            item["status"] != "COMPLETE" for item in rows[-2:]):
                        print("Stopped after two consecutive infrastructure/runtime failures", flush=True)
                        break
    finally:
        await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--expected-data-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
