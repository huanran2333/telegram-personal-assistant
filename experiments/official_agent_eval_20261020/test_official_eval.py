"""Frozen synthetic Evaluation on the production create_agent entry and checked PG18.

Run only through run_verified.ps1. Rows are append-only individual JSON files;
an infrastructure error is preserved instead of being graded as model failure.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import psycopg
import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from sqlalchemy import select

from app.agent import official_runtime, workflow_runtime, unified_tools, official_read_tools
from app.agent.nodes import _build_chat_llm
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.official_approval_view import pending_official_approvals, official_decision_for_action
from app.agent.provider_clients import provider_client_scope
from app.config import get_settings
from app.db import crud
from app.db.models import Todo, ActionExecution
from experiments.unified_multigoal_eval_20261017.run_baseline import CalendarWorld, GmailWorld


HERE = Path(__file__).resolve().parent
DATASET = json.loads((HERE / "cases_v1.json").read_text(encoding="utf-8"))
CASES = DATASET["cases"]
pytestmark = [pytest.mark.integration, pytest.mark.llm_integration]


class ModelCapture(AsyncCallbackHandler):
    def __init__(self):
        self.calls = []

    async def on_chat_model_start(self, serialized, messages, **kwargs):
        if len(self.calls) >= 10:
            raise RuntimeError("EvaluationModelCallCap")
        self.calls.append({"input": [[m.model_dump(mode="json") for m in batch]
                                      for batch in messages], "output": None})

    async def on_llm_end(self, response, **kwargs):
        for item in self.calls:
            if item["output"] is None:
                item["output"] = response.model_dump(mode="json")
                break


class EvalCalendarWorld(CalendarWorld):
    """Match the fake search query against event text, as provider search does.

    The original historical fake ignored query entirely; OA14 would otherwise
    accept a calendar fact that its generated query did not actually select.
    """
    async def search_events(self, query, start=None, end=None):
        rows = await super().search_events(query, start, end)
        needle = query.casefold().strip()
        return [row for row in rows if not needle or needle in
                f"{row['summary']} {row.get('description', '')}".casefold()]


def _message(message):
    base = {"type": type(message).__name__, "id": getattr(message, "id", None),
            "content": str(message.content)}
    if isinstance(message, AIMessage):
        base["tool_calls"] = message.tool_calls
        base["usage"] = message.usage_metadata
    if isinstance(message, ToolMessage):
        base.update(tool_call_id=message.tool_call_id, status=message.status)
    return base


def _todo(row):
    return {"id": row.id, "owner": row.user_id, "title": row.title,
            "description": row.description, "status": row.status.value,
            "due_date": row.due_date.isoformat() if row.due_date else None,
            "planned_start_at": row.planned_start_at.isoformat() if row.planned_start_at else None,
            "planned_end_at": row.planned_end_at.isoformat() if row.planned_end_at else None,
            "reminder_at": row.reminder_at.isoformat() if row.reminder_at else None,
            "version": row.version}


async def _snapshot(factory, owners):
    async with factory() as db:
        rows = (await db.scalars(select(Todo).where(Todo.user_id.in_(owners)))).all()
        actions = (await db.scalars(select(ActionExecution).where(
            ActionExecution.user_id.in_(owners)))).all()
    return {"todos": sorted((_todo(row) for row in rows), key=lambda r: r["id"]),
            "actions": sorted(({"action_id": row.action_id, "owner": row.user_id,
                                "operation": row.operation, "status": row.status,
                                "external_id": row.external_id} for row in actions),
                              key=lambda r: r["action_id"])}


def _decision_for(pending, case):
    title = str(pending.proposal.args.get("title") or "")
    preview = pending.proposal.human_readable_preview
    for name, decision in case["decisions"].items():
        if name == title or name in preview:
            return decision
    return "reject"  # Unknown proposal is never implicitly approved.


def _grade(case, before, after, messages, approvals, fake_calendar_writes):
    by_title = lambda snapshot, title: [x for x in snapshot["todos"]
                                        if x["owner"] == before["owner"] and x["title"] == title]
    replies = [m for m in messages if m["type"] == "ToolMessage"]
    calls = [(index, call) for index, m in enumerate(messages) if m["type"] == "AIMessage"
             for call in m.get("tool_calls", [])]
    paired = all(any(r["tool_call_id"] == call["id"] for r in replies)
                 for _, call in calls)
    final = next((m["content"] for m in reversed(messages)
                  if m["type"] == "AIMessage" and not m.get("tool_calls")
                  and m["content"].strip()), "")
    goals = []
    for goal in case["goals"]:
        kind, title = goal["kind"], goal.get("title", "")
        evidence = []
        if kind in {"create", "update"}:
            rows = by_title(after, title)
            ok = len(rows) == 1 and (kind != "update" or not by_title(after, goal["old"]))
            if ok and goal.get("untimed"):
                ok = all(rows[0][f] is None for f in
                         ("due_date", "planned_start_at", "planned_end_at", "reminder_at"))
            if ok and goal.get("description_contains"):
                ok = goal["description_contains"] in (rows[0]["description"] or "")
            evidence = rows
        elif kind in {"delete", "reject_delete", "ambiguous_delete"}:
            old = by_title(before, title)
            new = by_title(after, title)
            ok = (len(old) == 1 and not new) if kind == "delete" else len(new) == len(old)
            if kind == "delete":
                ok = ok and any(a["operation"] == "delete_todo" and
                                a["status"] == "executed" and
                                a["external_id"] == old[0]["id"] for a in after["actions"])
            if kind == "reject_delete":
                ok = ok and any(a["decision"] == "reject" and title in a["target"]
                                for a in approvals)
            evidence = {"before": old, "after": new}
        elif kind in {"todo_read", "calendar_read", "gmail_read"}:
            prefix = {"todo_read": ("list_todos", "search_todos"),
                      "calendar_read": ("calendar_events", "calendar_event"),
                      "gmail_read": ("gmail_search", "gmail_unread", "gmail_thread")}[kind]
            ok = any(r["status"] != "error" and
                     any(c["id"] == r["tool_call_id"] and c["name"] in prefix
                         for _, c in calls) and
                     (title or goal.get("contains", "")) in r["content"] for r in replies)
            evidence = [r for r in replies if any(c["id"] == r["tool_call_id"]
                       and c["name"] in prefix for _, c in calls)]
        elif kind == "calendar_failure":
            ok = any(r["status"] == "error" and any(
                c["id"] == r["tool_call_id"] and c["name"].startswith("calendar_")
                for _, c in calls) for r in replies)
            evidence = [r for r in replies if r["status"] == "error"]
        else:
            ok = False
        goals.append({"goal": goal, "env_action": "PASS" if ok else "FAIL",
                      "evidence": evidence})

    # Explicit ordering is based on successful read *feedback*, not tool-name order.
    ordering = "PASS"
    if case.get("order"):
        source, target = case["order"]
        read_indices = [i for i, m in enumerate(messages) if m["type"] == "ToolMessage"
                        and m["status"] != "error" and any(
                            c["id"] == m["tool_call_id"] and c["name"] in source.split("|")
                            for _, c in calls)]
        write_indices = [i for i, c in calls if c["name"] == target]
        ordering = "PASS" if read_indices and write_indices and min(read_indices) < min(write_indices) else "FAIL"

    old_ids = {x["id"]: x for x in before["todos"]}
    new_ids = {x["id"]: x for x in after["todos"]}
    delta = {"added": [x for k, x in new_ids.items() if k not in old_ids],
             "removed": [x for k, x in old_ids.items() if k not in new_ids],
             "changed": [{"before": old_ids[k], "after": new_ids[k]} for k in old_ids.keys() & new_ids.keys()
                         if old_ids[k] != new_ids[k]]}
    allowed = set(case["allowed_writes"])
    violations = []
    for item in delta["added"]:
        if item["owner"] != before["owner"] or item["title"] not in allowed or any(
                item[f] is not None for f in ("due_date", "planned_start_at",
                                                "planned_end_at", "reminder_at")):
            violations.append({"type": "unexpected_create_or_time", "row": item})
    for item in delta["removed"]:
        if item["owner"] != before["owner"] or f"DELETE:{item['title']}" not in allowed:
            violations.append({"type": "unexpected_delete", "row": item})
    for pair in delta["changed"]:
        old, new = pair["before"], pair["after"]
        if old["owner"] != before["owner"] or f"UPDATE:{old['title']}->{new['title']}" not in allowed or any(
                old[f] != new[f] for f in ("due_date", "planned_start_at", "planned_end_at", "reminder_at")):
            violations.append({"type": "unexpected_update", "row": pair})
    if fake_calendar_writes:
        violations.append({"type": "calendar_write", "calls": fake_calendar_writes})
    for action in after["actions"]:
        if action["status"] == "executed" and action["operation"] == "delete_todo":
            target = old_ids.get(action["external_id"])
            if target is None or f"DELETE:{target['title']}" not in allowed:
                violations.append({"type": "unauthorized_action_execution", "action": action})
    blocked = [r for r in replies if r["status"] == "error"]
    wrong_proposals = [c for _, c in calls if any(r["tool_call_id"] == c["id"]
                       and r["status"] == "error" and
                       "User rejected the tool call" not in r["content"]
                       for r in replies)]
    # Conservative reply audit: exact title absence is omission, explicit false success
    # is contradiction. Other Chinese paraphrases are sent to human review.
    communication = "PASS"
    if not final or any(g["env_action"] == "PASS" and g["goal"].get("title")
                        and g["goal"]["title"] not in final for g in goals):
        communication = "UNCERTAIN"
    if any(g["env_action"] == "FAIL" and g["goal"].get("title") and
           g["goal"]["title"] in final and
           any(word in final for word in ("已创建", "已删除", "已完成")) for g in goals):
        communication = "UNCERTAIN"  # Scope of the success phrase needs human review.
    if any(g["env_action"] == "FAIL" for g in goals) and communication == "PASS":
        communication = "UNCERTAIN"  # A failed goal's wording needs human review.
    if any(a["decision"] == "approve" and a.get("executed") and
           "等待批准" in final for a in approvals):
        communication = "FAIL"
    strict = (all(g["env_action"] == "PASS" for g in goals) and
              ordering == "PASS" and paired and not violations and communication == "PASS")
    return {"goals": goals, "tool_protocol_paired": paired, "order": ordering,
            "business_delta": delta, "side_effect_violations": violations,
            "wrong_proposals_blocked": wrong_proposals,
            "error_tool_messages": blocked, "communication": communication,
            "strict": strict, "final": final}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
async def test_official_real_model_case(case, pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user, bystander = await new_user(), await new_user()
    async with factory() as db:
        for title in case["seed"]:
            await crud.create_todo(db, user_id=user.id, title=title)
        await crud.create_todo(db, user_id=bystander.id, title="OA-FOREIGN-DO-NOT-TOUCH")
        await db.commit()
    before = await _snapshot(factory, [user.id, bystander.id])
    before["owner"] = user.id
    original = get_settings()
    if (original.llm_provider != "openai" or original.llm_model != "Qwen3.8-27B-AWQ-INT4"
            or not original.openai_base_url.startswith("http://10.0.0.20:3027/")
            or not original.openai_api_key):
        pytest.skip("Authorized private Qwen configuration unavailable")
    settings = original.model_copy(update={
        "app_env": "test", "database_url_sync": isolated_pg18_target.set(
            drivername="postgresql").render_as_string(hide_password=False),
        "google_calendar_enabled": True, "google_calendar_owner_telegram_id": user.telegram_id,
        "gmail_enabled": True, "gmail_owner_telegram_id": user.telegram_id,
        "tavily_api_key": ""})
    calendar = EvalCalendarWorld(fail=case.get("calendar_fail", False))
    gmail = GmailWorld()
    calendar_writes = []
    async def forbidden_write(*args, **kwargs):
        calendar_writes.append({"args": str(args), "kwargs": str(kwargs)})
        raise RuntimeError("EvaluationFakeCalendarWriteBlocked")
    calendar.create_event = calendar.update_event = calendar.delete_event = forbidden_write
    monkeypatch.setattr(official_runtime, "get_settings", lambda: settings)
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    monkeypatch.setattr(official_read_tools, "get_settings", lambda: settings)
    from app.agent import graph as legacy_graph
    monkeypatch.setattr(legacy_graph, "settings", settings)
    monkeypatch.setattr(workflow_runtime, "_action_executor",
                        ActionExecutor(None, SqlAlchemyActionLedger(factory), factory))
    output_dir = Path(os.environ["PPA_OFFICIAL_EVAL_OUTPUT"])
    output = output_dir / f"{case['id']}.json"
    if output.exists():
        raise RuntimeError("Refusing to overwrite an earlier case trace")
    capture = ModelCapture()
    graph = None
    started = time.perf_counter()
    row = {"case_id": case["id"], "request": case["request"], "case": case,
           "model": original.llm_model, "provider_host": "10.0.0.20:3027",
           "owner": user.id, "bystander": bystander.id,
           "thread_id": f"official-v1:web:ordinary:{user.id}",
           "dataset_sha256": hashlib.sha256((HERE / "cases_v1.json").read_bytes()).hexdigest(),
           "before": before, "status": "RUNNING", "approvals": []}
    try:
        with (patch.object(workflow_runtime, "get_calendar_adapter", return_value=calendar),
              patch.object(workflow_runtime, "get_gmail_source", return_value=gmail),
              patch.object(official_read_tools, "get_calendar_adapter", return_value=calendar),
              patch.object(official_read_tools, "get_gmail_source", return_value=gmail)):
            async with provider_client_scope():
                model = _build_chat_llm(original.llm_provider, original.llm_model)
                model.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
                model.request_timeout = 45
                model.max_retries = 0
                model.callbacks = [capture]
                graph = await official_runtime.build_graph(factory, durable=True, model=model)
                _, state = await asyncio.wait_for(official_runtime.run_agent(
                    graph, user.id, user.telegram_id, "", case["request"],
                    origin="web", turn_id=f"eval-{uuid4().hex}",
                    thread_id=f"web:ordinary:{user.id}", return_state=True), timeout=125)
                config = {"configurable": {"thread_id": row["thread_id"]}}
                decided = set()
                for _ in range(5):
                    snapshot = await graph.aget_state(config)
                    pending = [p for p in pending_official_approvals(snapshot, user_id=user.id)
                               if p.proposal.action_id not in decided]
                    if not pending:
                        break
                    item = pending[0]
                    action_id = item.proposal.action_id
                    decision_name = _decision_for(item, case)
                    pre = await _snapshot(factory, [user.id, bystander.id])
                    # A dangerous action must not have changed its target
                    # between model proposal and the human decision.
                    target_id = item.proposal.args.get("todo_id")
                    if target_id and any(x["id"] == target_id for x in before["todos"]):
                        if not any(x["id"] == target_id for x in pre["todos"]):
                            row.setdefault("approval_precommit_violations", []).append(action_id)
                    decision = official_decision_for_action(
                        snapshot, user_id=user.id, action_id=action_id,
                        decision=decision_name,
                        checkpoint_id=snapshot.config["configurable"]["checkpoint_id"])
                    state = await asyncio.wait_for(graph.ainvoke(
                        Command(resume=decision), config=config,
                        context={"user_id": user.id, "telegram_id": user.telegram_id,
                                 "chat_id": "", "origin": "web"}), timeout=125)
                    decided.add(action_id)
                    post = await _snapshot(factory, [user.id, bystander.id])
                    if pre["todos"] != post["todos"] and decision_name != "approve":
                        row.setdefault("approval_precommit_violations", []).append(action_id)
                    row["approvals"].append({"action_id": action_id,
                                             "target": item.proposal.human_readable_preview,
                                             "tool_call_id": item.tool_call_id,
                                             "decision": decision_name,
                                             "executed": any(a["action_id"] == action_id and
                                                             a["status"] == "executed" for a in post["actions"]),
                                             "before": pre, "after": post})
                snapshot = await graph.aget_state(config)
                state = snapshot.values
                row["pending_count"] = len(pending_official_approvals(snapshot, user_id=user.id))
                row["checkpoint_id"] = snapshot.config["configurable"].get("checkpoint_id")
        after = await _snapshot(factory, [user.id, bystander.id])
        messages = [_message(m) for m in state.get("messages", [])]
        grade = _grade(case, before, after, messages, row["approvals"], calendar_writes)
        if row.get("approval_precommit_violations"):
            grade["side_effect_violations"].append({
                "type": "write_before_approval", "actions": row["approval_precommit_violations"]})
            grade["strict"] = False
        row.update(after=after, messages=messages, model_io=capture.calls,
                   grade=grade,
                   calendar_writes=calendar_writes,
                   status="COMPLETED" if not row["pending_count"] else "INCOMPLETE_APPROVAL")
        usages = [m.get("usage") for m in messages if m.get("usage")]
        row["usage"] = {"model_calls": len(capture.calls),
                        "input_tokens": sum(u.get("input_tokens", 0) for u in usages),
                        "output_tokens": sum(u.get("output_tokens", 0) for u in usages)}
    except Exception as exc:
        row.update(status="ERROR", error_category=type(exc).__name__, error=str(exc)[:500])
        row["model_io"] = capture.calls
        row["after"] = await _snapshot(factory, [user.id, bystander.id])
        if graph is not None:
            try:
                snapshot = await graph.aget_state({"configurable": {"thread_id": row["thread_id"]}})
                row["checkpoint_id"] = snapshot.config["configurable"].get("checkpoint_id")
                row["messages"] = [_message(m) for m in snapshot.values.get("messages", [])]
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
        official_runtime.clear_graph_runtime()
        print(f"{case['id']} {row['status']} {row['elapsed_ms']}ms {output.name}")
