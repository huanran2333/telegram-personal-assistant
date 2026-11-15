"""R4.1 deterministic observations of the *current* boundary, not R4.2 implementation.

These tests deliberately use no database, Telegram, or production migration.
They establish why the frozen single-flight and chunk-ledger contracts are needed.
"""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from app.agent.graph import run_agent
from app.telegram import sender


class OverlapGraph:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.entered_twice = asyncio.Event()
        self.release = asyncio.Event()
        self.thread_ids = []

    async def aget_state(self, config):
        return SimpleNamespace(values={})

    async def ainvoke(self, state, config, *, context=None):
        assert context == {"user_id": "synthetic-user",
                           "telegram_id": "synthetic-telegram-user",
                           "chat_id": "synthetic-chat", "origin": "telegram"}
        self.thread_ids.append(config["configurable"]["thread_id"])
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active == 2:
            self.entered_twice.set()
        try:
            await asyncio.wait_for(self.release.wait(), timeout=2)
            return {"messages": [AIMessage(content="ok")]}
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_current_same_thread_invocations_overlap_without_server_queue():
    graph = OverlapGraph()
    args = (graph, "synthetic-user", "synthetic-telegram-user", "synthetic-chat")
    first = asyncio.create_task(run_agent(*args, "first"))
    second = asyncio.create_task(run_agent(*args, "second"))
    try:
        await asyncio.wait_for(graph.entered_twice.wait(), timeout=2)
        assert graph.thread_ids == ["user_synthetic-telegram-user"] * 2
        assert graph.max_active == 2
    finally:
        graph.release.set()
        await asyncio.gather(first, second)


@pytest.mark.asyncio
async def test_current_sender_partial_chunk_failure_has_no_durable_chunk_result(monkeypatch):
    class PartialBot:
        def __init__(self):
            self.calls = []

        async def send_message(self, **kwargs):
            self.calls.append(kwargs["text"])
            if len(self.calls) == 2:
                raise TimeoutError("synthetic second chunk timeout")
            return SimpleNamespace(message_id=101, chat_id=kwargs["chat_id"])

    bot = PartialBot()
    monkeypatch.setattr(sender, "get_bot", lambda: bot)
    with pytest.raises(TimeoutError):
        await sender.send_message("synthetic-chat", "x" * 4097)
    assert [len(chunk) for chunk in bot.calls] == [4096, 1]
    # Current sender throws without returning the successful first Message;
    # R4.2 must persist per-chunk result before claiming logical SENT.
