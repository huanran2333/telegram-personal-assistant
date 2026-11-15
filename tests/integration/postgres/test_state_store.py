from typing import Annotated, TypedDict
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import text

from app.agent.nodes import _merge_profile, chat_response_messages, load_profile_context
from app.agent.store import generate_store
from app.config import get_settings
from app.localization import load_preferences, save_language


pytestmark = pytest.mark.integration


class CounterState(TypedDict):
    count: int


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def build_chat_replay(saver):
    builder = StateGraph(ChatState)
    builder.add_node("record", lambda _state: {"messages": []})
    builder.set_entry_point("record")
    builder.add_edge("record", END)
    return builder.compile(checkpointer=saver)


@pytest.mark.asyncio
async def test_real_postgres_chat_recent_context_survives_saver_restart(pg_harness):
    thread_id = f"itest-chat-context-{uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}
    try:
        async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as saver:
            await saver.setup()
            graph = build_chat_replay(saver)
            await graph.ainvoke({"messages": [HumanMessage(content="My codeword is blue-whale."),
                                               AIMessage(content="Noted.")]}, config)
        async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as fresh_saver:
            graph = build_chat_replay(fresh_saver)
            persisted = (await graph.aget_state(config)).values["messages"]
            assembled = chat_response_messages(
                persisted + [HumanMessage(content="What codeword did I just mention?")])
            assert [item.content for item in assembled[1:]] == [
                "My codeword is blue-whale.", "Noted.", "What codeword did I just mention?"]
            assert len(assembled) == 4
    finally:
        async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as cleanup:
            await cleanup.adelete_thread(thread_id)


def build_counter_graph(saver):
    builder = StateGraph(CounterState)
    builder.add_node("increment", lambda state: {"count": state["count"] + 1})
    builder.set_entry_point("increment")
    builder.add_edge("increment", END)
    return builder.compile(checkpointer=saver)


@pytest.mark.asyncio
async def test_real_postgres_checkpoint_restart(pg_harness):
    _, _, engine = pg_harness
    thread_id = f"itest-checkpoint-{uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as saver:
        await saver.setup()
        graph = build_counter_graph(saver)
        assert (await graph.ainvoke({"count": 4}, config))["count"] == 5

    async with engine.connect() as connection:
        count = await connection.scalar(text("SELECT count(*) FROM checkpoints WHERE thread_id = :id"), {"id": thread_id})
        assert count > 0

    async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as fresh_saver:
        fresh_graph = build_counter_graph(fresh_saver)
        assert (await fresh_graph.aget_state(config)).values["count"] == 5
        await fresh_saver.adelete_thread(thread_id)


@pytest.mark.asyncio
async def test_real_postgres_store_restart(pg_harness):
    _, new_user, _ = pg_harness
    user = await new_user()
    namespace = ("profile", user.id)
    key = "user_profile"

    async with generate_store() as store:
        await store.aput(namespace, key, {"preferences": ["concise answers"]})

    async with generate_store() as fresh_store:
        item = await fresh_store.aget(namespace, key)
        assert item.value == {"preferences": ["concise answers"]}
        assert "concise answers" in await load_profile_context(fresh_store, user.id)
        merged = _merge_profile(item.value, {"preferences": ["evening reminders"]})
        await fresh_store.aput(namespace, key, merged)

    async with generate_store() as third_store:
        assert (await third_store.aget(namespace, key)).value == {
            "preferences": ["concise answers", "evening reminders"]
        }
        await third_store.adelete(namespace, key)


@pytest.mark.asyncio
async def test_real_postgres_profile_isolation(pg_harness):
    _, new_user, _ = pg_harness
    first, second = await new_user(), await new_user()
    first_namespace = ("profile", first.id)
    second_namespace = ("profile", second.id)

    async with generate_store() as store:
        await store.aput(first_namespace, "user_profile", {"preferences": ["dummy A"]})

    async with generate_store() as fresh_store:
        assert (await fresh_store.aget(first_namespace, "user_profile")).value["preferences"] == ["dummy A"]
        assert await fresh_store.aget(second_namespace, "user_profile") is None
        assert await load_profile_context(fresh_store, second.id) == ""
        await fresh_store.adelete(first_namespace, "user_profile")


@pytest.mark.asyncio
async def test_real_postgres_assistant_language_survives_store_reconnect_and_user_isolation(pg_harness):
    _, new_user, _ = pg_harness
    first, second = await new_user(), await new_user()
    first_key = ("profile", first.id)
    try:
        async with generate_store() as store:
            await store.aput(first_key, "user_profile", {"preferences": ["concise answers"]})
            assert (await save_language(first.id, "zh-CN", store)).language == "zh-CN"
        async with generate_store() as fresh_store:
            assert (await load_preferences(first.id, fresh_store)).language == "zh-CN"
            assert (await load_preferences(first.id, fresh_store)).concise
            assert (await load_preferences(second.id, fresh_store)).language == "en"
            assert (await save_language(first.id, "en", fresh_store)).language == "en"
        async with generate_store() as restarted_store:
            assert (await load_preferences(first.id, restarted_store)).language == "en"
            assert (await load_preferences(second.id, restarted_store)).language == "en"
    finally:
        async with generate_store() as cleanup_store:
            await cleanup_store.adelete(first_key, "user_profile")
