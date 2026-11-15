"""Current create_agent, real model and checked disposable PG18; records are append-only.

Run only through run_verified.ps1. A pytest pass means the evidence was captured,
not that the agent passed its task. No real Telegram/Google/Gmail write is used.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langgraph.types import Command

from app.agent import official_read_tools, official_runtime, unified_tools, workflow_runtime
from app.agent.business_clock import isolated_evaluation_time
from app.agent.calendar import CalendarError
from app.agent.calendar_conditional import CalendarProviderError
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.gmail import GmailError, GmailMessage
from app.agent.nodes import _build_chat_llm
from app.agent.memory_runtime import create_memory_runtime
from app.agent.official_approval_view import official_decision_for_action, pending_official_approvals
from app.agent.provider_clients import provider_client_scope
from app.config import get_settings
from app.db import crud
from app.localization import bind_profile_store
from app.agent.store import generate_store
from experiments.official_agent_eval_20261020.test_official_eval import _message, _snapshot
from experiments.official_agent_eval_rebuild_20261023.scoring import score as score_v1
from experiments.official_agent_eval_rebuild_20261023.scoring_v2 import score as score_v2
from experiments.unified_multigoal_eval_20261017.run_baseline import CalendarWorld

HERE = Path(__file__).resolve().parent
DATASET_PATH = Path(os.environ.get("PPA_EVAL_DATASET", HERE / "cases_v1.json")).resolve()
if DATASET_PATH.parent != HERE or DATASET_PATH.name not in {"cases_v1.json", "cases_holdout_v2.json", "cases_holdout_v3.json", "cases_core_schedule_20261103.json", "cases_core_http403_20261103.json", "cases_unseen_20261103.json", "cases_u13_wording_20261104.json", "cases_benchmark_extension_v1.json"}:
    raise RuntimeError("EvaluationDatasetPathNotAllowed")
DATASET = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
CASES = DATASET["cases"]
score = score_v2 if DATASET_PATH.name in {"cases_holdout_v3.json", "cases_core_schedule_20261103.json", "cases_core_http403_20261103.json", "cases_unseen_20261103.json", "cases_u13_wording_20261104.json"} else score_v1
pytestmark = [pytest.mark.integration, pytest.mark.llm_integration]


class Capture(AsyncCallbackHandler):
    def __init__(self):
        self.calls = []

    async def on_chat_model_start(self, serialized, messages, **kwargs):
        if len(self.calls) >= 20:
            raise RuntimeError("EvaluationModelCallCap20")
        self.calls.append({"input": [[msg.model_dump(mode="json") for msg in batch]
                                     for batch in messages], "output": None})

    async def on_llm_end(self, response, **kwargs):
        for call in self.calls:
            if call["output"] is None:
                call["output"] = response.model_dump(mode="json")
                break


class Calendar(CalendarWorld):
    def __init__(self, fault):
        super().__init__()
        self.fault = fault
        self.read_count = 0
        self.write_calls = []
        self.EVENTS = {row["id"]: {"id": row["id"], "etag": '"synthetic"',
                       "status": "confirmed", "summary": row["summary"],
                       "description": "synthetic evaluation event",
                       "start": {"dateTime": row["start"]},
                       "end": {"dateTime": row["end"]}, "attendees": []}
                       for row in DATASET["world"]["events"]}

    def _fault(self):
        self.read_count += 1
        if self.fault == "http403":
            raise CalendarError("SyntheticCalendarRead403") from CalendarProviderError(
                "provider_unavailable", 403)
        if self.fault == "always" or (self.fault == "once" and self.read_count == 1):
            raise CalendarError("SyntheticCalendarReadFailure")

    async def search_events(self, query, start=None, end=None):
        self._fault()
        rows = await super().search_events(query, start, end)
        needle = query.casefold().strip()
        return [row for row in rows if not needle or needle in
                f"{row['summary']} {row.get('description', '')}".casefold()]

    async def get_event(self, calendar_id, event_id):
        self._fault()
        return await super().get_event(calendar_id, event_id)

    async def availability(self, start, end, timezone_name):
        self._fault()
        rows = await CalendarWorld.search_events(self, "", start, end)
        return {"calendars": {"primary": {"busy": [
            {"start": row["start"]["dateTime"], "end": row["end"]["dateTime"]}
            for row in rows]}}}

    async def create_event(self, *args, **kwargs):
        self.write_calls.append({"operation": "create", "args": str(args), "kwargs": str(kwargs)})
        raise RuntimeError("EvaluationCalendarWriteBlocked")

    async def update_event(self, *args, **kwargs):
        self.write_calls.append({"operation": "update", "args": str(args), "kwargs": str(kwargs)})
        raise RuntimeError("EvaluationCalendarWriteBlocked")

    async def delete_event(self, *args, **kwargs):
        self.write_calls.append({"operation": "delete", "args": str(args), "kwargs": str(kwargs)})
        raise RuntimeError("EvaluationCalendarWriteBlocked")


class Gmail:
    def __init__(self, fault=None):
        self.read_count = 0
        self.fault = fault
        self.rows = DATASET["world"]["mail"]

    async def account_address(self):
        return "owner@example.test"

    async def thread(self, thread_id):
        self.read_count += 1
        if self.fault == "always" or (self.fault == "once" and self.read_count == 1):
            raise GmailError("SyntheticGmailReadFailure")
        return [GmailMessage(message_id=row["id"], thread_id=row["thread_id"],
                             sender="sender@example.test", recipients=["owner@example.test"],
                             subject=row["subject"], snippet=row["body"], body=row["body"],
                             timestamp=datetime(2026, 10, 20, tzinfo=timezone.utc))
                for row in self.rows if row["thread_id"] == thread_id]

    async def search(self, query, limit=5):
        rows = [item for row in self.rows for item in await self.thread(row["thread_id"])]
        needle = query.casefold().strip()
        if needle.startswith("subject:"):
            needle = needle[len("subject:"):].strip('" ')
            return [row for row in rows if needle in row.subject.casefold()][:limit]
        return [row for row in rows if needle in
                f"{row.subject} {row.snippet}".casefold()][:limit]

    async def unread(self, query, limit=5):
        return await self.search(query, limit)


def decide(pending, case):
    preview = pending.proposal.human_readable_preview
    matches = [value for title, value in case["decisions"].items() if title in preview]
    return matches[0] if len(matches) == 1 else "reject"


def _synthetic_telegram_identity(user):
    """Match the authenticated Telegram graph context without using ingress APIs."""
    thread_id = f"user_{user.telegram_id}"
    context = {"user_id": user.id, "telegram_id": user.telegram_id,
               "chat_id": str(user.telegram_id), "origin": "telegram"}
    return thread_id, context


def _case_evaluation_time(case, *, default_timezone: str) -> tuple[datetime, str]:
    """Old date-only time_reference is not a reproducible instant."""
    value, zone_name = case.get("evaluation_now"), case.get("evaluation_timezone")
    if value is None and zone_name is None:
        # A new baseline can give an immutable old dataset an explicit clock
        # without rewriting its original case JSON. The digest binds the choice.
        expected = os.environ.get("PPA_EVAL_FIXED_DATASET_SHA256")
        if expected and expected.lower() == hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest():
            value = os.environ.get("PPA_EVAL_FIXED_NOW")
            zone_name = os.environ.get("PPA_EVAL_FIXED_TIMEZONE")
    if not isinstance(value, str) or not isinstance(zone_name, str):
        raise ValueError("EvaluationCaseRequiresExplicitAwareNowAndIanaTimezone")
    try:
        now = datetime.fromisoformat(value.replace("Z", "+00:00"))
        ZoneInfo(zone_name)
    except (ValueError, KeyError) as exc:
        raise ValueError("EvaluationCaseClockInvalid") from exc
    if now.utcoffset() is None or zone_name != default_timezone:
        raise ValueError("EvaluationCaseClockZoneMismatch")
    return now, zone_name


async def _invoke_at_evaluation_time(coroutine, *, now, zone_name, settings,
                                     deadline=None):
    with isolated_evaluation_time(now, zone_name, app_env=settings.app_env):
        limit = 500 if os.environ.get("PPA_EVAL_MODEL_PROFILE") == "no_thinking_500s" else 180
        if deadline is not None:
            limit = max(0.001, min(limit, deadline - time.perf_counter()))
        return await asyncio.wait_for(coroutine, timeout=limit)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
async def test_current_create_agent(case, pg_harness, isolated_pg18_target, monkeypatch):
    if case["split"] != os.environ.get("PPA_EVAL_SPLIT"):
        pytest.skip("Other frozen split")
    try:
        evaluation_now, evaluation_zone = _case_evaluation_time(
            case, default_timezone=get_settings().default_timezone)
    except ValueError as exc:
        pytest.fail(f"Reproducible evaluation clock required: {exc}")
    factory, new_user, _ = pg_harness
    user, bystander = await new_user(), await new_user()
    async with factory() as db:
        for seed in case["seed"]:
            if isinstance(seed, str):
                await crud.create_todo(db, user_id=user.id, title=seed)
            elif isinstance(seed, dict) and isinstance(seed.get("title"), str):
                reminder = seed.get("reminder_at")
                reminder_at = datetime.fromisoformat(reminder) if reminder else None
                if reminder_at is not None and reminder_at.utcoffset() is None:
                    pytest.fail("SeedReminderRequiresAwareTime")
                await crud.create_todo(db, user_id=user.id, title=seed["title"],
                                       reminder_at=reminder_at)
            else:
                pytest.fail("InvalidEvaluationSeed")
        await crud.create_todo(db, user_id=bystander.id, title="EVAL-FOREIGN-UNCHANGED")
        for title in case.get("foreign_seed", []):
            await crud.create_todo(db, user_id=bystander.id, title=title)
        await db.commit()
    before = await _snapshot(factory, [user.id, bystander.id])
    before["owner"] = user.id
    original = get_settings()
    if original.llm_provider != "openai" or not original.openai_base_url or not original.openai_api_key:
        pytest.fail("Authorized real-model configuration unavailable")
    settings = original.model_copy(update={
        "app_env": "test",
        "database_url_sync": isolated_pg18_target.set(drivername="postgresql").render_as_string(hide_password=False),
        "google_calendar_enabled": True, "google_calendar_owner_telegram_id": user.telegram_id,
        "gmail_enabled": True, "gmail_owner_telegram_id": user.telegram_id,
        "tavily_api_key": "", "memory_enabled": True,
        "memory_mode": "transcript_lexical"})
    calendar, gmail = Calendar(case.get("calendar_fault")), Gmail(case.get("gmail_fault"))
    monkeypatch.setattr(official_runtime, "get_settings", lambda: settings)
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    monkeypatch.setattr(official_read_tools, "get_settings", lambda: settings)
    from app.agent import graph as legacy_graph
    monkeypatch.setattr(legacy_graph, "settings", settings)
    monkeypatch.setattr(workflow_runtime, "_action_executor",
                        ActionExecutor(None, SqlAlchemyActionLedger(factory), factory))
    output = Path(os.environ["PPA_EVAL_OUTPUT"]) / f"{case['id']}.json"
    if output.exists():
        raise RuntimeError("Refusing to overwrite raw evidence")
    thread_id, context = _synthetic_telegram_identity(user)
    thread = official_runtime.official_thread_id(thread_id)
    row = {"case": case, "case_id": case["id"], "split": case["split"],
           "model": original.llm_model, "thread_id": thread,
           "model_invocation_profile": os.environ.get("PPA_EVAL_MODEL_PROFILE", "no_thinking_60s"),
           "owner": user.id, "bystander": bystander.id,
           "runtime": {"memory_enabled": settings.memory_enabled,
                       "memory_mode": settings.memory_mode,
                       "store": "AsyncPostgresStore", "saver": "AsyncPostgresSaver",
                       "calendar_gmail": "synthetic adapters",
                       "entry_source": "synthetic_telegram_direct_graph"},
           "evaluation_clock": {"now": evaluation_now.isoformat(),
                                "timezone": evaluation_zone},
           "dataset_sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),
           "before": before, "approvals": [], "status": "RUNNING"}
    capture, graph, stack = Capture(), None, AsyncExitStack()
    started = time.perf_counter()
    try:
        with (patch.object(workflow_runtime, "get_calendar_adapter", return_value=calendar),
              patch.object(workflow_runtime, "get_gmail_source", return_value=gmail),
              patch.object(official_read_tools, "get_calendar_adapter", return_value=calendar),
              patch.object(official_read_tools, "get_gmail_source", return_value=gmail)):
            async with provider_client_scope():
                store = await stack.enter_async_context(generate_store())
                bind_profile_store(store)
                memory_runtime = create_memory_runtime(session_factory=factory, settings=settings)
                await memory_runtime.preflight_storage()
                profile = os.environ.get("PPA_EVAL_MODEL_PROFILE", "no_thinking_60s")
                if profile == "production":
                    model = _build_chat_llm(
                        original.llm_provider, original.llm_model,
                        disable_qwen_thinking=True,
                        request_timeout=original.llm_primary_timeout_seconds)
                    model.callbacks = [capture]
                elif profile in {"no_thinking_60s", "no_thinking_500s"}:
                    model = _build_chat_llm(original.llm_provider, original.llm_model)
                    model.request_timeout = 500 if profile == "no_thinking_500s" else 60
                    model.max_retries, model.callbacks = 0, [capture]
                    model.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
                else:
                    raise ValueError("UnknownEvaluationModelProfile")
                graph = await official_runtime.build_graph(factory, durable=True, model=model,
                                                           store=store, memory_runtime=memory_runtime)
                _, state = await _invoke_at_evaluation_time(official_runtime.run_agent(
                    graph, context["user_id"], context["telegram_id"],
                    context["chat_id"], case["request"],
                    projection_session_factory=factory,
                    origin=context["origin"], turn_id=f"eval-{uuid4().hex}",
                    thread_id=thread_id, return_state=True),
                    now=evaluation_now, zone_name=evaluation_zone, settings=settings,
                    deadline=started + (500 if profile == "no_thinking_500s" else 180))
                config = {"configurable": {"thread_id": thread}}
                decided = set()
                for _ in range(6):
                    snapshot = await graph.aget_state(config)
                    pending = [item for item in pending_official_approvals(snapshot, user_id=user.id)
                               if item.proposal.action_id not in decided]
                    if not pending:
                        break
                    item = pending[0]
                    decision_name = decide(item, case)
                    pre = await _snapshot(factory, [user.id, bystander.id])
                    decision = official_decision_for_action(
                        snapshot, user_id=user.id, action_id=item.proposal.action_id,
                        decision=decision_name,
                        checkpoint_id=snapshot.config["configurable"]["checkpoint_id"])
                    state = await _invoke_at_evaluation_time(graph.ainvoke(
                        Command(resume=decision), config=config,
                        context=context), now=evaluation_now,
                        zone_name=evaluation_zone, settings=settings,
                        deadline=started + (500 if profile == "no_thinking_500s" else 180))
                    post = await _snapshot(factory, [user.id, bystander.id])
                    row["approvals"].append({"action_id": item.proposal.action_id,
                                             "tool_call_id": item.tool_call_id,
                                             "target": item.proposal.human_readable_preview,
                                             "decision": decision_name,
                                             "executed": any(a["action_id"] == item.proposal.action_id
                                                             and a["status"] == "executed" for a in post["actions"]),
                                             "before": pre, "after": post})
                    decided.add(item.proposal.action_id)
                snapshot = await graph.aget_state(config)
                state = snapshot.values
                row["checkpoint_id"] = snapshot.config["configurable"].get("checkpoint_id")
                row["pending_count"] = len(pending_official_approvals(snapshot, user_id=user.id))
        after = await _snapshot(factory, [user.id, bystander.id])
        messages = [_message(msg) for msg in state.get("messages", [])]
        row.update(status="COMPLETED" if not row["pending_count"] else "INCOMPLETE_APPROVAL",
                   after=after, messages=messages, model_io=capture.calls,
                   calendar_reads=calendar.read_count, gmail_reads=gmail.read_count,
                   calendar_writes=calendar.write_calls,
                   grade=(score(case, before, after, messages, row["approvals"], calendar.write_calls)
                          if DATASET_PATH.name != "cases_benchmark_extension_v1.json"
                          else {"status": "not_applicable", "reason": "new_goals_use_versioned_replay_oracle"}))
        usages = [msg.get("usage") for msg in messages if msg.get("usage")]
        row["usage"] = {"model_calls": len(capture.calls),
                        "input_tokens": sum(u.get("input_tokens", 0) for u in usages),
                        "output_tokens": sum(u.get("output_tokens", 0) for u in usages)}
    except Exception as exc:
        row.update(status="ERROR", error_category=type(exc).__name__, error=str(exc)[:500],
                   model_io=capture.calls, after=await _snapshot(factory, [user.id, bystander.id]))
        if graph is not None:
            try:
                snapshot = await graph.aget_state({"configurable": {"thread_id": thread}})
                row["messages"] = [_message(msg) for msg in snapshot.values.get("messages", [])]
                row["pending_count"] = len(pending_official_approvals(snapshot, user_id=user.id))
            except Exception as snapshot_exc:
                row["snapshot_error"] = type(snapshot_exc).__name__
    finally:
        row["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        output.write_text(json.dumps(row, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if graph is not None:
            await graph.checkpointer.conn.close()
        if official_runtime._legacy_reader is not None:
            await official_runtime._legacy_reader.checkpointer.conn.close()
        bind_profile_store(None)
        await stack.aclose()
        official_runtime.clear_graph_runtime()
        print(f"{case['id']} {row['status']} {row['elapsed_ms']}ms {output.name}")
