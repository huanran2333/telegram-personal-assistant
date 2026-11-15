"""G2 deterministic cross-entry lock and fail-closed source rules."""

from contextlib import asynccontextmanager

import pytest

from app.telegram import durable


@pytest.mark.asyncio
async def test_telegram_receipt_adds_exact_original_workflow_guard(monkeypatch):
    observed = []

    @asynccontextmanager
    async def fake_guard(_url, graph_kind, thread_id):
        observed.append((graph_kind, thread_id))
        yield

    monkeypatch.setattr(durable, "thread_invocation_guard", fake_guard)
    async with durable.telegram_workflow_guard(
            "postgresql://synthetic", "telegram_session", "telegram_user_12345"):
        assert observed == [("workflow", "workflow_user_12345"),
                            ("workflow", "user_12345"),
                            ("workflow", "official-v1:user_12345")]
    async with durable.telegram_workflow_guard(
            "postgresql://synthetic", "workflow", "workflow_user_12345"):
        pass
    assert len(observed) == 3
    async with durable.telegram_workflow_guard(
            "postgresql://synthetic", "telegram_session", "arbitrary_thread"):
        pass
    assert len(observed) == 3
