"""Production graph construction persists and reloads a thread on real PostgreSQL."""
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from app.agent import graph as graph_module
from app.agent.store import generate_store
from app.agent.provider_clients import provider_client_scope

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_real_postgres_production_graph_reconnect_restores_checkpoint(pg_harness, monkeypatch):
    session_factory, _, _ = pg_harness
    thread_id = f"r1-durable-{uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}
    async with provider_client_scope(), generate_store() as store:
        first = await graph_module.build_graph(session_factory, store=store)
        assert first.checkpointer is not None
        assert first.store is store
        await first.aupdate_state(config, {"messages": [HumanMessage(content="synthetic durability probe")]},
                                  as_node="agent_model")
        assert (await first.aget_state(config)).values["messages"][-1].content == "synthetic durability probe"
        await first.checkpointer.conn.close()

    async with provider_client_scope(), generate_store() as restarted_store:
        restarted = await graph_module.build_graph(session_factory, store=restarted_store)
        try:
            assert restarted.checkpointer is not None
            assert restarted.store is restarted_store
            restored = await restarted.aget_state(config)
            assert restored.values["messages"][-1].content == "synthetic durability probe"
        finally:
            await restarted.checkpointer.adelete_thread(thread_id)
            await restarted.checkpointer.conn.close()
