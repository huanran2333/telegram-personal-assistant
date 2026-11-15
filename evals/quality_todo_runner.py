"""Real-model Todo task postconditions in a proven isolated PG18 cluster.

No service lifespan, Telegram Bot, Scheduler or Calendar/Gmail provider runs.
The model sees only synthetic per-case titles. Each case owns a fresh User and
Saver thread; cleanup deletes synthetic business rows and its checkpoint.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.provider_clients import run_with_provider_clients
from app.config import get_settings
from app.db import crud
from app.db.isolated_test_target import (validate_isolated_pg18_configuration,
                                         validate_isolated_pg18_target)
from app.db.models import OutboundDelivery, ScheduledOccurrence, Todo, User
from evals.quality_graders import (grade_reminder_definition, grade_todo_environment,
                                   grade_tool_trajectory)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "datasets" / "agent_quality_todo_v2.json"


def load_dataset(path: Path = DATASET) -> tuple[dict, str]:
    raw = path.read_bytes()
    data = json.loads(raw)
    cases = data.get("cases")
    if not data.get("dataset_version") or not isinstance(cases, list) or len(cases) < 4:
        raise ValueError("Unversioned or incomplete Todo task dataset")
    ids = [case.get("id") for case in cases]
    if len(ids) != len(set(ids)) or any(not isinstance(value, str) for value in ids):
        raise ValueError("Duplicate/invalid Todo case IDs")
    for case in cases:
        if (case.get("split") not in {"dev", "holdout"} or
                not isinstance(case.get("turns"), list) or not case["turns"] or
                not all(isinstance(turn, str) and turn.strip() for turn in case["turns"]) or
                case.get("expected_status") not in {None, "pending", "done"} or
                not isinstance(case.get("required_tools"), list) or
                not isinstance(case.get("initial_todos", []), list)):
            raise ValueError("Invalid Todo case structure")
        if case.get("expected_reminder_local") is not None:
            datetime.strptime(case["expected_reminder_local"], "%Y-%m-%d %H:%M")
        if case.get("expected_reminder_version_delta") not in {None, 1}:
            raise ValueError("Unsupported reminder version expectation")
        if any(initial != "{title}" for initial in case.get("initial_todos", [])):
            raise ValueError("Only the case-owned synthetic Todo may be seeded")
    return data, hashlib.sha256(raw).hexdigest()


def compare(baseline: dict, candidate: dict) -> dict:
    """Paired result by case/repetition; reject changed dataset/model/target."""
    fields = ("dataset_version", "dataset_sha256", "model_provider", "model_id",
              "split", "repeats", "case_filter", "evidence", "target")
    if any(baseline["manifest"].get(key) != candidate["manifest"].get(key) for key in fields):
        raise ValueError("Experiment conditions differ; comparison refused")
    def index(report: dict) -> dict:
        rows = report["rows"]
        keyed = {(row["case_id"], row["repeat"]): row for row in rows}
        if len(keyed) != len(rows):
            raise ValueError("Duplicate Todo evaluation attempt")
        return keyed
    before, after = index(baseline), index(candidate)
    if before.keys() != after.keys():
        raise ValueError("Experiment attempt identities differ")
    transitions = {}
    for score in ("task_passed", "trajectory"):
        counts = {"0->0": 0, "0->1": 0, "1->0": 0, "1->1": 0}
        for key in before:
            a = bool(before[key][score]["passed"] if score == "trajectory" else before[key][score])
            b = bool(after[key][score]["passed"] if score == "trajectory" else after[key][score])
            counts[f"{int(a)}->{int(b)}"] += 1
        transitions[score] = counts
    return {"attempts": len(before), "transitions": transitions,
            "prompt_blobs": {"baseline": baseline["manifest"].get("prompt_blob_sha1"),
                             "candidate": candidate["manifest"].get("prompt_blob_sha1")},
            "baseline": baseline["summary"], "candidate": candidate["summary"],
            "caution": "Small synthetic set and stochastic model; random per-run titles differ."}


async def prove_target(settings, expected: Path, engine) -> dict:
    # No DDL/DML before both the configured and actual identity have matched.
    async with engine.connect() as connection:
        actual_directory = await connection.scalar(text("SHOW data_directory"))
        actual_database = await connection.scalar(text("SELECT current_database()"))
        actual_port = await connection.scalar(text("SHOW port"))
        version = await connection.scalar(text("SHOW server_version"))
    validate_isolated_pg18_target(
        settings.database_url, settings.database_url_sync, str(expected),
        str(actual_directory), str(actual_database), str(actual_port), str(version),
        repository_root=ROOT)
    return {"database": actual_database, "port": int(actual_port), "server_version": version,
            "data_directory_sha256": hashlib.sha256(str(expected.resolve()).encode()).hexdigest()}


def snapshot(rows: list[Todo]) -> list[dict]:
    return [{"id": row.id, "user_id": row.user_id, "title": row.title,
             "status": row.status.value,
             "reminder_version": row.reminder_version,
             "reminder_at": row.reminder_at.isoformat() if row.reminder_at else None} for row in rows]


def tool_result_class(value: str | None) -> str:
    """Bounded diagnostic categories; never serialize a ToolMessage body."""
    if value is None:
        return "missing"
    if "无法解析提醒时间" in value or "Could not parse reminder time" in value:
        return "bad_reminder_time"
    if "提醒时间已经过去" in value or "reminder time has already passed" in value:
        return "past_reminder_time"
    if "没有找到 ID" in value or "No todo found" in value:
        return "todo_not_found"
    if value.lstrip().startswith("❌"):
        return "other_error"
    return "non_error"


def date_argument_shape(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    if re.search(r"\d{4}年\s*\d{1,2}月\s*\d{1,2}日", value):
        return "cjk_absolute"
    if re.search(r"\d{4}-\d{1,2}-\d{1,2}", value):
        return "iso_like"
    return "other"


async def run_case(graph, factory, case: dict, repeat: int, *, timezone_name: str) -> dict:
    import app.agent.graph as graph_module
    title = f"Synthetic Agent Eval {uuid4().hex[:12]}"
    thread = f"agent-quality-{uuid4().hex}"
    owner_id = sentinel_id = None
    started = time.perf_counter()
    error = None
    responses: list[str] = []
    try:
        async with factory() as db:
            owner = await crud.get_or_create_user(db, f"itest-agent-eval-{uuid4().hex}")
            sentinel = await crud.get_or_create_user(db, f"itest-agent-eval-other-{uuid4().hex}")
            owner_id, sentinel_id = owner.id, sentinel.id
            await crud.create_todo(db, sentinel.id, f"Synthetic sentinel {uuid4().hex[:8]}")
            for initial in case.get("initial_todos", []):
                await crud.create_todo(db, owner.id, initial.format(title=title))
            await db.commit()
        async with factory() as db:
            before = snapshot((await db.scalars(select(Todo).where(
                Todo.user_id.in_([owner_id, sentinel_id])))).all())
            outbound_before = await db.scalar(select(func.count()).select_from(OutboundDelivery))
            occurrences_before = await db.scalar(select(func.count()).select_from(ScheduledOccurrence))
        for index, turn in enumerate(case["turns"]):
            responses.append(await graph_module.run_agent(
                graph, owner_id, owner.telegram_id, owner.telegram_id,
                turn.format(title=title), thread_id=thread,
                turn_id=f"synthetic-eval-{repeat}-{index}", origin="web"))
        async with factory() as db:
            after = snapshot((await db.scalars(select(Todo).where(
                Todo.user_id.in_([owner_id, sentinel_id])))).all())
            outbound_after = await db.scalar(select(func.count()).select_from(OutboundDelivery))
            occurrences_after = await db.scalar(select(func.count()).select_from(ScheduledOccurrence))
        state = await graph.aget_state({"configurable": {"thread_id": thread}})
        ai_messages = [message for message in state.values.get("messages", []) if isinstance(message, AIMessage)]
        # Some production nodes intentionally turn provider outages into a safe
        # clarification. The environment remains unchanged, but the evaluation
        # must not count an unavailable provider as an Agent-quality failure.
        if not ai_messages:
            raise RuntimeError("No model trace was produced for this task")
        calls = [{"name": call["name"], "args": call.get("args", {})}
                 for message in ai_messages for call in (message.tool_calls or [])]
        # Keep only structural diagnostics. In particular, never write tool
        # arguments, generated Todo IDs, or response bodies to an eval report.
        seen_call_ids: set[str] = set()
        call_details = []
        owned_ids = {row["id"] for row in after if row["user_id"] == owner_id}
        tool_outcomes = {message.tool_call_id: str(message.content)
                         for message in state.values.get("messages", [])
                         if isinstance(message, ToolMessage)}
        for message in ai_messages:
            for call in message.tool_calls or []:
                call_id = str(call.get("id") or "")
                args = call.get("args") or {}
                reference = str(args.get("todo_id") or "")
                call_details.append({
                    "name": call["name"],
                    "repeated_call_id": bool(call_id and call_id in seen_call_ids),
                    "todo_reference_matches_owned": (any(value.startswith(reference) for value in owned_ids)
                                                     if reference else None),
                    "result_class": tool_result_class(tool_outcomes.get(call_id)),
                    "reminder_argument_shape": date_argument_shape(args.get("remind_at_str"))
                    if call["name"] == "set_reminder" else None,
                })
                if call_id:
                    seen_call_ids.add(call_id)
        usage = [message.usage_metadata for message in ai_messages if message.usage_metadata]
        route_failure_kind = state.values.get("classification_failure_kind") or ""
        untraced_route_failure = (not usage and not calls and
                                  state.values.get("intent") in {"clarify", ""} and
                                  case.get("expected_status") is not None)
        expected = {title: case["expected_status"]} if case["expected_status"] else {}
        environment = grade_todo_environment(
            before, after, owner_id=owner_id, expected=expected,
            require_owned_unchanged=case.get("require_owned_unchanged", False))
        reminder = (grade_reminder_definition(
            before, after, owner_id=owner_id, title=title,
            expected_local=case["expected_reminder_local"], timezone_name=timezone_name,
            new_occurrences=occurrences_after - occurrences_before,
            new_deliveries=outbound_after - outbound_before,
            expected_version_delta=case.get("expected_reminder_version_delta"))
            if case.get("expected_reminder_local") else None)
        trajectory = grade_tool_trajectory(
            calls, [{"name": name, "args": {}} for name in case["required_tools"]],
            forbidden={"delete_todo"} | ({"create_todo", "update_todo", "set_reminder"}
                                        if case.get("require_owned_unchanged") else set()))
        communication = (title in responses[-1]) if case.get("reply_must_include_title") else None
        task_passed = environment["passed"] and (reminder is None or reminder["passed"]) and communication is not False
        return {"case_id": case["id"], "split": case["split"], "repeat": repeat,
                "task_passed": task_passed, "environment": environment,
                "reminder": reminder,
                "trajectory": trajectory, "communication_ok": communication,
                "tool_names": [call["name"] for call in calls],
                "tool_call_diagnostics": call_details,
                "model_calls": len(usage),
                "input_tokens": sum(item.get("input_tokens", 0) for item in usage) if usage else None,
                "output_tokens": sum(item.get("output_tokens", 0) for item in usage) if usage else None,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "route_intent": state.values.get("intent"),
                "route_failure_kind": route_failure_kind or None,
                "error_category": ("ROUTE_MODEL_FAILURE" if route_failure_kind == "model" else
                                   "ROUTE_VALIDATION_FAILURE" if route_failure_kind == "validation" else
                                   "ROUTE_OR_PROVIDER_UNTRACED" if untraced_route_failure else None)}
    except Exception as exc:
        error = type(exc).__name__
        return {"case_id": case["id"], "split": case["split"], "repeat": repeat,
                "task_passed": False, "environment": None, "reminder": None, "trajectory": None,
                "communication_ok": None, "tool_names": [], "model_calls": None,
                "tool_call_diagnostics": [],
                "input_tokens": None, "output_tokens": None,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "error_category": error}
    finally:
        # Do not let evaluation artifacts affect another case. A failed cleanup
        # is an evaluation failure, not a reason to touch any other database.
        await graph.checkpointer.adelete_thread(thread)
        if owner_id or sentinel_id:
            async with factory() as db:
                ids = [value for value in (owner_id, sentinel_id) if value]
                await db.execute(delete(Todo).where(Todo.user_id.in_(ids)))
                await db.execute(delete(User).where(User.id.in_(ids)))
                await db.commit()


async def run(*, split: str, repeats: int, case_id: str | None = None,
              dataset_path: Path = DATASET) -> dict:
    if split not in {"all", "dev", "holdout"} or repeats < 1 or repeats > 5:
        raise ValueError("Invalid split/repeats")
    import os
    expected_text = os.environ.get("PPA_ISOLATED_PG18_DATA_DIR")
    if not expected_text:
        raise ValueError("PPA_ISOLATED_PG18_DATA_DIR is required")
    expected = Path(expected_text).resolve()
    settings = get_settings()
    validate_isolated_pg18_configuration(
        settings.database_url, settings.database_url_sync, str(expected),
        repository_root=ROOT)
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    try:
        target = await prove_target(settings, expected, engine)
        data, digest = load_dataset(dataset_path)
        selected = [case for case in data["cases"]
                    if (split == "all" or case["split"] == split)
                    and (case_id is None or case["id"] == case_id)]
        if not selected:
            raise ValueError("No cases match split and case ID")
        factory = async_sessionmaker(engine, expire_on_commit=False)
        import app.agent.graph as graph_module

        async def no_calendar(_settings):
            return []

        original_discovery = graph_module.discover_calendar_tools
        graph_module.discover_calendar_tools = no_calendar
        graph = None
        try:
            graph = await graph_module.build_graph(factory, durable=True)
            rows = []
            for repeat in range(1, repeats + 1):
                for case in selected:
                    row = await run_case(graph, factory, case, repeat,
                                         timezone_name=settings.default_timezone)
                    rows.append(row)
                    print(f"{case['id']}#{repeat}: task={row['task_passed']} error={row['error_category']}", flush=True)
        finally:
            graph_module.discover_calendar_tools = original_discovery
            if graph is not None:
                connection = getattr(graph.checkpointer, "conn", None)
                if connection is not None:
                    await connection.close()
            graph_module.clear_graph_runtime()
        measured_usage = [row for row in rows if isinstance(row["input_tokens"], int)
                          and isinstance(row["output_tokens"], int)]
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        prompt_blob = subprocess.run(["git", "hash-object", "app/agent/prompt.py"], cwd=ROOT,
                                     capture_output=True, text=True, check=True).stdout.strip()
        tools_blob = subprocess.run(["git", "hash-object", "app/agent/tools.py"], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout.strip()
        planner_blob = subprocess.run(["git", "hash-object", "app/agent/planner.py"], cwd=ROOT,
                                      capture_output=True, text=True, check=True).stdout.strip()
        nodes_blob = subprocess.run(["git", "hash-object", "app/agent/nodes.py"], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout.strip()
        return {"manifest": {"dataset_version": data["dataset_version"], "dataset_sha256": digest,
                             "git_commit": commit, "prompt_blob_sha1": prompt_blob,
                             "tools_blob_sha1": tools_blob,
                             "planner_blob_sha1": planner_blob, "nodes_blob_sha1": nodes_blob,
                             "model_provider": settings.llm_provider,
                             "model_id": settings.llm_model, "split": split, "repeats": repeats,
                             "case_filter": case_id,
                             "evidence": "REAL_MODEL_ISOLATED_PG18_TODO_AGENT_NO_TELEGRAM_OR_SCHEDULER",
                             "target": target},
                "summary": {"attempts": len(rows), "task_passed": sum(row["task_passed"] for row in rows),
                            "environment_passed": sum(bool(row["environment"] and row["environment"]["passed"]) for row in rows),
                            "trajectory_passed": sum(bool(row["trajectory"] and row["trajectory"]["passed"]) for row in rows),
                            "error_attempts": sum(row["error_category"] is not None for row in rows),
                            "usage_measured_attempts": len(measured_usage),
                            "usage_tokens_total": sum(row["input_tokens"] + row["output_tokens"] for row in measured_usage)
                            if len(measured_usage) == len(rows) else None,
                            "latency_p50_ms": statistics.median(row["latency_ms"] for row in rows) if rows else None},
                "rows": rows}
    finally:
        await engine.dispose()


def main() -> None:
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("all", "dev", "holdout"), default="all")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--case-id")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = asyncio.run(run_with_provider_clients(run(split=args.split, repeats=args.repeats,
                                                       case_id=args.case_id,
                                                       dataset_path=args.dataset)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    if result["summary"]["error_attempts"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
