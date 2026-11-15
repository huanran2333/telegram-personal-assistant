"""One isolated, scripted-model Agent→ToolNode→PostgreSQL Todo smoke.

The model's tool call is predetermined. This proves graph/tool/persistence wiring,
not real LLM tool selection. No Bot, sender, scheduler, Google or model API starts.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import secrets
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def verify(port: int) -> dict:
    from ops import local_staging

    data, private, _, admin, app = local_staging.read_private(port)
    local_staging.verify_server(data, port, admin)
    env = local_staging.safe_environment(port, admin, app)
    env.update({
        "LLM_PROVIDER": "openai", "LLM_MODEL": "scripted-agent-smoke",
        "OPENAI_BASE_URL": "http://127.0.0.1:65433/v1",
        "OPENAI_API_KEY": "placeholder-not-a-secret",
        "LLM_FALLBACK_1_BASE_URL": "", "LLM_FALLBACK_1_MODEL": "",
        "LLM_FALLBACK_1_API_KEY": "", "LLM_FALLBACK_2_BASE_URL": "",
        "LLM_FALLBACK_2_MODEL": "", "LLM_FALLBACK_2_API_KEY": "",
        "MEMORY_ENABLED": "false", "TAVILY_API_KEY": "",
    })
    os.environ.update(env)

    import psycopg
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from sqlalchemy import select

    from app.agent.official_graph import build_official_agent
    from app.agent.official_runtime import run_agent
    from app.config import get_settings
    from app.db.models import ActionExecution, Todo, User
    from app.db.session import AsyncSessionLocal
    from app.telegram.durable import admit_receipt, claim_receipt
    from app.telegram.processing_context import receipt_scope
    from experiments.official_agent_migration_20261020.test_official_runtime_contract import ScriptedModel

    get_settings.cache_clear()
    settings = get_settings()
    if (settings.telegram_bot_token or settings.google_calendar_enabled or
            settings.gmail_enabled or settings.scheduler_enabled or
            settings.tavily_api_key or settings.memory_enabled):
        raise RuntimeError("External integration unexpectedly enabled")

    suffix = uuid4().hex[:12]
    title = f"Isolated Agent smoke {suffix}"
    user_message = f"请创建一个待办，标题是{title}，不设置时间和提醒。"
    telegram_number = 900_000_000 + secrets.randbelow(100_000_000)
    telegram_id = str(telegram_number)
    thread_id = f"user_{telegram_id}"
    user_id = str(uuid4())
    call_id = f"smoke-create-{uuid4().hex}"
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[{
            "name": "create_todo", "args": {"title": title}, "id": call_id,
        }]),
        AIMessage(content=f"已创建待办：{title}。"),
    ])

    async with AsyncSessionLocal() as db:
        db.add(User(id=user_id, telegram_id=telegram_id,
                    first_name="Synthetic Agent Smoke", calendar_time_block_sync=False))
        await db.commit()
        before = len((await db.scalars(select(Todo).where(Todo.user_id == user_id))).all())

    receipt = await admit_receipt(
        AsyncSessionLocal, bot_identity=99990001,
        data={"update_id": secrets.randbelow(2**31), "message": {
            "message_id": 1, "date": 0,
            "from": {"id": telegram_number, "first_name": "Synthetic"},
            "chat": {"id": telegram_number, "type": "private"},
            "text": user_message,
        }}, graph_kind="telegram_session", thread_id=thread_id,
    )
    fence = await claim_receipt(AsyncSessionLocal, receipt.id, "isolated-agent-smoke")
    if not fence:
        raise RuntimeError("Synthetic receipt could not be claimed")

    connection = await psycopg.AsyncConnection.connect(
        settings.database_url_sync, autocommit=True)
    try:
        saver = AsyncPostgresSaver(connection)
        await saver.setup()
        graph, _ = build_official_agent(
            model=model, session_factory=AsyncSessionLocal,
            settings=settings, checkpointer=saver, interrupt_mode="per_call",
        )
        with receipt_scope(receipt.id, fence):
            reply, result = await run_agent(
                graph, user_id, telegram_id, telegram_id, user_message,
                thread_id=thread_id, origin="telegram", return_state=True,
            )
        snapshot = await graph.aget_state({"configurable": {
            "thread_id": f"official-v1:{thread_id}"}})
        calls = [call for message in result["messages"]
                 if isinstance(message, AIMessage) for call in message.tool_calls]
        outputs = [message for message in result["messages"]
                   if isinstance(message, ToolMessage)]
        humans = [message for message in result["messages"]
                  if isinstance(message, HumanMessage)]
        async with AsyncSessionLocal() as db:
            rows = (await db.scalars(select(Todo).where(Todo.user_id == user_id))).all()
            actions = (await db.scalars(select(ActionExecution).where(
                ActionExecution.user_id == user_id))).all()
        if not (before == 0 and len(rows) == 1 and rows[0].title == title and
                len(calls) == 1 and calls[0]["name"] == "create_todo" and
                calls[0]["id"] == call_id and len(outputs) == 1 and
                outputs[0].tool_call_id == call_id and outputs[0].status != "error" and
                len(actions) == 1 and actions[0].operation == "create_todo" and
                actions[0].status == "executed" and
                actions[0].result.get("receipt_id") == receipt.id and
                len(model._received) == 2 and
                any("create_todo" in names for names in model._bound_tool_names) and
                any(message.id == f"telegram-receipt:{receipt.id}" for message in humans) and
                (snapshot.config or {}).get("configurable", {}).get("checkpoint_id") and
                title in reply and not result.get("__interrupt__")):
            raise AssertionError("Agent, tool, business state or checkpoint mismatch")
        report = {
            "kind": "scripted_model_agent_smoke", "model_api_calls": 0,
            "origin": "synthetic_telegram_without_bot", "receipt_id": receipt.id,
            "thread_id": f"official-v1:{thread_id}", "todo_count_before": before,
            "todo_count_after": len(rows), "todo_id": rows[0].id,
            "tool_calls": calls,
            "tool_results": [{"tool_call_id": item.tool_call_id,
                              "status": item.status, "content": str(item.content)}
                             for item in outputs],
            "action_execution_id": actions[0].action_id,
            "action_execution_status": actions[0].status,
            "assistant_reply": reply, "checkpoint_saved": True,
            "telegram_sent": False,
        }
        target = private / f"agent-smoke-{suffix}.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"result": "passed", "report": str(target), "todo_before": before,
                "todo_after": len(rows), "tool": calls[0]["name"],
                "checkpoint_saved": True, "model_api_calls": 0}
    finally:
        await connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=56520)
    args = parser.parse_args()
    if sys.platform != "win32":
        raise RuntimeError("The isolated staging recipe was verified only on Windows")
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print(json.dumps(asyncio.run(verify(args.port)), ensure_ascii=False))


if __name__ == "__main__":
    main()
