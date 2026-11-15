"""Synthetic long-thread replay on the *real* AsyncPostgresSaver.

No external mutation. This measures checkpoint growth and the production Todo
context window separately. Optional real-LLM probes are not product E2E.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from sqlalchemy import text

from app.agent.nodes import todo_context_messages, _build_chat_llm
from app.agent.provider_clients import run_with_provider_clients
from app.config import get_settings
from app.db.session import async_engine
from evals.v23_runner import ROOT


class ReplayState(TypedDict):
    messages: Annotated[list, add_messages]


def replay_graph(saver):
    builder = StateGraph(ReplayState)
    builder.add_node("record", lambda _state: {"messages": []})
    builder.set_entry_point("record")
    builder.add_edge("record", END)
    return builder.compile(checkpointer=saver)


def synthetic_turn(index: int) -> list:
    if index == 1:
        question = "My project review is on Monday. Remember this statement for this thread."
    elif index == 5:
        question = "I prefer concise Chinese replies."
    elif index == 15:
        question = "Todo reference: prepare synthetic project slides before the review."
    elif index == 20:
        return [HumanMessage(content="Check synthetic Todo status"),
                AIMessage(content="", tool_calls=[{"name": "list_todos", "args": {}, "id": "eval-tool-20"}]),
                ToolMessage(content="No synthetic Todo rows", tool_call_id="eval-tool-20"),
                AIMessage(content="No synthetic tasks were found.")]
    elif index == 35:
        question = "Correction: project review is on Wednesday, not Monday."
    elif index == 45:
        question = "Calendar reference: synthetic project review, Wednesday at 14:00."
    elif index in {49, 99, 149}:
        question = "What day is my project review, and which earlier statement was corrected?"
    elif index in {48, 98, 148}:
        question = "How is that synthetic meeting preparation going?"
    else:
        question = f"Irrelevant synthetic conversation noise turn {index}; no new personal facts."
    return [HumanMessage(content=question), AIMessage(content=f"Acknowledged synthetic turn {index}.")]


def message_chars(messages: list) -> int:
    return sum(len(str(item.content)) for item in messages)


async def snapshot(graph, config, turn: int, *, probe_llm: bool) -> dict:
    state = await graph.aget_state(config)
    messages = list(state.values["messages"])
    todo_prompt = [SystemMessage(content="You are a synthetic evaluation assistant. Answer only from the supplied messages.")]
    todo_prompt += todo_context_messages(messages)
    result = {"turns": turn, "checkpoint_message_count": len(messages),
              "todo_prompt_message_count": len(todo_prompt),
              "todo_prompt_chars": message_chars(todo_prompt),
              "early_fact_in_prompt": any("review is on Monday" in str(m.content) for m in todo_prompt),
              "correction_in_prompt": any("review is on Wednesday" in str(m.content) for m in todo_prompt),
              "recent_reference_in_prompt": any("meeting preparation" in str(m.content) for m in todo_prompt),
              "tool_pair_integrity_checkpoint": None,
              "tool_pair_integrity_prompt": None,
              "checkpoint_serialized_bytes": None,
              "thread_checkpoint_storage_bytes": None,
              "provider_input_tokens": None, "provider_total_tokens": None,
              "llm_latency_ms": None, "llm_error_category": None,
              "model_corrected_fact": None, "model_stale_fact": None}
    def pair_integrity(items):
        calls = {call["id"] for m in items if isinstance(m, AIMessage)
                 for call in getattr(m, "tool_calls", [])}
        results = {m.tool_call_id for m in items if isinstance(m, ToolMessage)}
        return calls == results
    result["tool_pair_integrity_checkpoint"] = pair_integrity(messages)
    result["tool_pair_integrity_prompt"] = pair_integrity(todo_prompt)
    async with async_engine.connect() as connection:
        result["checkpoint_serialized_bytes"] = await connection.scalar(text(
            "SELECT pg_column_size(c.checkpoint) + COALESCE((SELECT octet_length(b.blob) "
            "FROM checkpoint_blobs b WHERE b.thread_id = c.thread_id "
            "AND b.checkpoint_ns = c.checkpoint_ns AND b.channel = 'messages' "
            "AND b.version = c.checkpoint->'channel_versions'->>'messages'), 0) "
            "FROM checkpoints c WHERE c.thread_id = :thread "
            "ORDER BY c.checkpoint_id DESC LIMIT 1"
        ), {"thread": config["configurable"]["thread_id"]})
        result["thread_checkpoint_storage_bytes"] = await connection.scalar(text(
            "SELECT (SELECT COALESCE(SUM(pg_column_size(checkpoint)), 0) "
            "FROM checkpoints WHERE thread_id = :thread) + "
            "(SELECT COALESCE(SUM(octet_length(blob)), 0) "
            "FROM checkpoint_blobs WHERE thread_id = :thread) + "
            "(SELECT COALESCE(SUM(octet_length(blob)), 0) "
            "FROM checkpoint_writes WHERE thread_id = :thread)"
        ), {"thread": config["configurable"]["thread_id"]})
    if probe_llm:
        settings = get_settings()
        model = _build_chat_llm(settings.llm_provider, settings.llm_model, routing=True)
        started = time.perf_counter()
        try:
            answer = await model.ainvoke(todo_prompt)
            usage = getattr(answer, "usage_metadata", None)
            if isinstance(usage, dict):
                result["provider_input_tokens"] = usage.get("input_tokens")
                result["provider_total_tokens"] = usage.get("total_tokens")
            content = str(answer.content).lower()
            result["model_corrected_fact"] = "wednesday" in content
            result["model_stale_fact"] = "monday" in content and "wednesday" not in content
        except Exception as exc:
            result["llm_error_category"] = type(exc).__name__
        result["llm_latency_ms"] = round((time.perf_counter() - started) * 1000)
    return result


async def run(*, probe_llm: bool, turns: int = 150) -> dict:
    if turns < 150:
        raise ValueError("Baseline requires 50, 100 and one longer tier")
    settings = get_settings()
    thread_id = "v23-replay-" + uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    results = []
    try:
        async with AsyncPostgresSaver.from_conn_string(settings.database_url_sync) as saver:
            await saver.setup()
            graph = replay_graph(saver)
            for turn in range(1, 101):
                await graph.ainvoke({"messages": synthetic_turn(turn)}, config)
                if turn == 50:
                    results.append(await snapshot(graph, config, turn, probe_llm=probe_llm))
        # A new saver/graph is a process-connection restart, not MemorySaver.
        async with AsyncPostgresSaver.from_conn_string(settings.database_url_sync) as saver:
            graph = replay_graph(saver)
            state = await graph.aget_state(config)
            restart_recovered = len(state.values.get("messages", [])) >= 200
            results.append(await snapshot(graph, config, 100, probe_llm=probe_llm))
            for turn in range(101, turns + 1):
                await graph.ainvoke({"messages": synthetic_turn(turn)}, config)
            results.append(await snapshot(graph, config, turns, probe_llm=probe_llm))
            await saver.adelete_thread(thread_id)
    finally:
        # A failure should not leave a synthetic thread behind.
        async with AsyncPostgresSaver.from_conn_string(settings.database_url_sync) as cleanup:
            await cleanup.adelete_thread(thread_id)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    return {"metadata": {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                          "git_commit": commit, "evidence_class": "real PostgreSQL / synthetic replay",
                          "llm_probe": probe_llm, "provider": settings.llm_provider if probe_llm else None,
                          "model": settings.llm_model if probe_llm else None},
            "checkpointer": "AsyncPostgresSaver", "restart_recovered": restart_recovered,
            "tiers": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-llm", action="store_true")
    args = parser.parse_args()
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    report = asyncio.run(run_with_provider_clients(run(probe_llm=args.probe_llm)))
    output = ROOT / "evals" / "reports"
    output.mkdir(parents=True, exist_ok=True)
    (output / "v23_long_context.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# V2.3 Long-Context Baseline", "",
             "Real AsyncPostgresSaver with synthetic turns; not full Telegram/Agent E2E.",
             f"Restart recovery: {report['restart_recovered']}", "",
             "| Turns | Checkpoint messages | Latest checkpoint + messages blob bytes | Total thread storage bytes | Todo prompt messages | Prompt chars | Provider input tokens |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["tiers"]:
        lines.append(f"| {row['turns']} | {row['checkpoint_message_count']} | "
                     f"{row['checkpoint_serialized_bytes']} | {row['thread_checkpoint_storage_bytes']} | {row['todo_prompt_message_count']} | "
                     f"{row['todo_prompt_chars']} | {row['provider_input_tokens'] or 'UNAVAILABLE'} |")
    lines += ["", "Checkpoint growth does not imply LLM prompt growth. This probe uses the production Todo message-window helper.", ""]
    (output / "v23_long_context.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"long-context: tiers={len(report['tiers'])} restart={report['restart_recovered']}")


if __name__ == "__main__":
    main()
