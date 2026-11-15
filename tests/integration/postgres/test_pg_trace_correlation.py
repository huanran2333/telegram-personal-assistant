"""Real Postgres checkpoint proves diagnostic origin survives a process boundary."""
import asyncio
import sys
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.agent.executor import ActionExecutor
from app.agent.planner import RoutedRequest
from app.agent.workflow import build_productivity_graph, resume_workflow, start_workflow
from app.config import get_settings
from app.trace import begin_trace, current_trace_id, end_trace, set_correlation

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def event_loop_policy():
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.get_event_loop_policy()


class ReadOnlyCalendar:
    async def availability(self, start, end, timezone):
        return {"calendars": {"primary": {"busy": []}}}


class NoWriteLedger:
    async def begin(self, proposal):
        raise AssertionError("Reject must not enter the executor")

    async def complete(self, proposal, result):
        raise AssertionError("Reject must not write")

    async def fail(self, proposal, message):
        raise AssertionError("Reject must not write")


def graph(saver):
    calendar = ReadOnlyCalendar()
    return build_productivity_graph(calendar, ActionExecutor(calendar, NoWriteLedger(), None), saver)


@pytest.mark.asyncio
async def test_origin_trace_survives_real_postgres_interrupt_restart_and_reject(pg_harness):
    thread_id = f"workflow_itest_trace_{uuid4().hex}"
    route = RoutedRequest(intent="calendar_create", title="synthetic trace test",
                          date_reference="2026-11-20", hour=16, duration_minutes=30)
    token = begin_trace("synthetic-user")
    try:
        origin = current_trace_id()
        set_correlation(thread_id=thread_id)
        async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as saver:
            await saver.setup()
            first = await start_workflow(graph(saver), "synthetic-user", "synthetic request",
                                         "Asia/Shanghai", thread_id, route=route)
            assert first["__interrupt__"][0].value["kind"] == "action_proposal"
            assert (await graph(saver).aget_state({"configurable": {"thread_id": thread_id}})).values[
                "trace_origin_id"] == origin
    finally:
        end_trace(token)

    callback_token = begin_trace("synthetic-user")
    try:
        callback_trace = current_trace_id()
        assert callback_trace != origin
        async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as fresh_saver:
            restarted = graph(fresh_saver)
            state = await restarted.aget_state({"configurable": {"thread_id": thread_id}})
            assert state.values["trace_origin_id"] == origin
            set_correlation(origin_trace_id=state.values["trace_origin_id"], thread_id=thread_id)
            final = await resume_workflow(restarted, thread_id, {"decision": "reject"})
            assert final["proposal"]["status"] == "rejected"
            assert (await restarted.aget_state({"configurable": {"thread_id": thread_id}})).values[
                "trace_origin_id"] == origin
            await fresh_saver.adelete_thread(thread_id)
    finally:
        end_trace(callback_token)
