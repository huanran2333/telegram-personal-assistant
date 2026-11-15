"""Reminder permission and side effects through the production create_agent graph."""

from datetime import datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

import psycopg
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import select

from app.agent.official_graph import build_official_agent
from app.agent.official_runtime import run_agent
from app.config import get_settings
from app.db import crud
from app.db.models import Todo
from experiments.official_agent_migration_20261020.test_official_runtime_contract import ScriptedModel

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["allowed", "numeric_title_quote", "denied",
                                   "wrong_time", "wrong_owner"])
async def test_reminder_graph_binds_owner_target_time_and_preserves_plan(
        pg_harness, isolated_pg18_target, case):
    factory, new_user, _ = pg_harness
    owner, other = await new_user(), await new_user()
    zone = ZoneInfo("Asia/Shanghai")
    start = datetime(2026, 11, 14, 13, 0, tzinfo=zone)
    end = start + timedelta(minutes=45)
    async with factory() as db:
        own = await crud.create_todo(db, owner.id,
                                     ("E2E-20261103-提醒闭环-R2-2352" if case == "numeric_title_quote"
                                      else f"提醒图验收-{uuid4().hex[:8]}"),
                                     planned_start_at=start, planned_end_at=end)
        foreign = await crud.create_todo(db, other.id, "旁观者待办")
        await db.commit()
        own_id, title, foreign_id = own.id, own.title, foreign.id

    target_id = foreign_id[:8] if case == "wrong_owner" else own_id[:8]
    base = f"给已有待办『{title}』设置2026年11月12日08:30（北京时间）的提醒，只设提醒，不修改标题或计划时间。"
    text = (f"不要给已有待办『{title}』设置2026年11月12日08:30的提醒。"
            if case == "denied" else base)
    quote = ("2026年11月12日08:30的提醒" if case == "denied" else
             f"给已有待办『{title}』设置2026年11月12日08:30（北京时间）的提醒"
             if case == "numeric_title_quote" else
             "2026年11月12日08:30（北京时间）的提醒")
    value = "2026-11-12T09:30:00+08:00" if case == "wrong_time" else "2026-11-12T08:30:00+08:00"
    call_id = f"reminder-{case}"
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[{"name": "list_todos",
                                            "id": f"read-{case}", "args": {}}]),
        AIMessage(content="", tool_calls=[{"name": "set_reminder", "id": call_id,
            "args": {"todo_id": target_id, "remind_at_str": value,
                     "time_source_quote": quote, "expected_version": 1}}]),
        AIMessage(content="结果已核对。"),
    ])
    settings = get_settings().model_copy(update={"default_timezone": "Asia/Shanghai"})
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(model=model, session_factory=factory,
                                        settings=settings, checkpointer=saver)
        _, state = await run_agent(
            graph, owner.id, owner.telegram_id, "", text,
            origin="web", turn_id=f"reminder-{uuid4().hex}",
            thread_id=f"web:ordinary:{owner.id}", return_state=True)
        result = [m for m in state["messages"] if isinstance(m, ToolMessage)
                  and m.tool_call_id == call_id]
        read = [m for m in state["messages"] if isinstance(m, ToolMessage)
                and m.tool_call_id == f"read-{case}"]
        assert len(read) == 1
        assert read[0].artifact["todo_versions"][own_id] == 1
        assert len(result) == 1
        assert result[0].status == ("success" if case in {"allowed", "numeric_title_quote"}
                                    else "error")
        async with factory() as db:
            own_row = await db.scalar(select(Todo).where(Todo.id == own_id))
            foreign_row = await db.scalar(select(Todo).where(Todo.id == foreign_id))
            assert own_row.title == title
            assert own_row.planned_start_at == start and own_row.planned_end_at == end
            assert own_row.reminder_at == (datetime(2026, 11, 12, 8, 30,
                                                    tzinfo=zone) if case in {"allowed", "numeric_title_quote"}
                                       else None)
            assert foreign_row.reminder_at is None
    finally:
        await conn.close()
