"""Telegram parsing/duplicate guards only; not a real Telegram E2E claim."""
import json
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from telegram import Bot, Update

from app.telegram import webhook


@pytest.fixture(autouse=True)
def no_new_agent_checkpoint_for_legacy_callback_tests(monkeypatch):
    monkeypatch.setattr(webhook, "get_graph", AsyncMock(return_value=SimpleNamespace(
        aget_state=AsyncMock(return_value=None))))


@pytest.mark.asyncio
async def test_natural_language_action_uses_only_current_agent_reply(monkeypatch):
    """A new Telegram text turn has no outer route or workflow handoff."""
    from app.telegram.processing_context import receipt_scope

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args):
            return None

        async def commit(self):
            return None

    monkeypatch.setattr(webhook, "AsyncSessionLocal", Session)
    monkeypatch.setattr(webhook, "get_bot", lambda: Bot(
        token="123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"))
    monkeypatch.setattr(webhook, "get_user_by_telegram_id", AsyncMock(return_value=None))
    monkeypatch.setattr(webhook, "get_or_create_user", AsyncMock(
        return_value=SimpleNamespace(id="synthetic-user")))
    monkeypatch.setattr(webhook, "_replay_completed_resume", AsyncMock(return_value=False))
    monkeypatch.setattr(webhook, "send_typing", AsyncMock())
    monkeypatch.setattr(webhook, "get_graph", AsyncMock(return_value=SimpleNamespace(
        aget_state=AsyncMock(return_value=None))))
    agent = AsyncMock(return_value=("Agent result", {"pending_agent_proposal": {}}))
    monkeypatch.setattr(webhook, "run_agent", agent)
    ordinary_proof = AsyncMock()
    delivered = AsyncMock()
    monkeypatch.setattr(webhook, "send_message", delivered)
    monkeypatch.setattr(webhook, "_record_checkpoint_outcome", ordinary_proof)
    update = {"update_id": 77, "message": {"message_id": 1, "date": 0,
              "from": {"id": 7, "is_bot": False, "first_name": "Synthetic"},
              "chat": {"id": 7, "type": "private"}, "text": "删除合成任务"}}

    with receipt_scope("receipt-1", "synthetic-fence"):
        assert await webhook.process_admitted_update(update) == {"ok": True}
    agent.assert_awaited_once()
    # New natural-language turns enter the unified Agent without an outer
    # single-intent model route; the exact action is chosen inside its loop.
    assert "route" not in agent.await_args.kwargs
    ordinary_proof.assert_awaited_once()
    delivered.assert_awaited_once_with(7, "Agent result")


@pytest.mark.asyncio
async def test_admission_commit_failure_returns_retryable_503_without_processing(monkeypatch):
    update_id = 89898912
    payload = {
        "update_id": update_id,
        "message": {"message_id": 1, "date": 0,
                    "from": {"id": 7, "is_bot": False, "first_name": "Tester"},
                    "chat": {"id": 7, "type": "private"}, "text": "Hello"},
    }
    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode(),
                "more_body": False}
    from app.telegram import durable
    async def admission_failure(*_args, **_kwargs):
        raise RuntimeError("synthetic database outage")
    monkeypatch.setattr(durable, "admit_receipt", admission_failure)
    class Recovery:
        bot_identity = 123456
        is_running = True
        admission_lock = asyncio.Lock()
        def kick(self, _receipt_id):
            raise AssertionError("Must not process after failed commit")
    request = Request({"type": "http", "method": "POST", "path": "/webhook",
                       "app": SimpleNamespace(state=SimpleNamespace(
                           runtime=SimpleNamespace(ready=True), telegram_durable=Recovery())),
                       "headers": [(b"x-telegram-bot-api-secret-token",
                                    webhook.webhook_secret_token().encode())]}, receive)
    with pytest.raises(HTTPException) as error:
        await webhook.telegram_webhook(request)
    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_shutdown_between_first_check_and_admission_rejects_new_custody(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    runtime = SimpleNamespace(ready=True)
    calls = []

    async def receive():
        entered.set()
        await release.wait()
        return {"type": "http.request", "body": json.dumps({
            "update_id": 451, "message": {"message_id": 1, "date": 0,
            "from": {"id": 7, "is_bot": False, "first_name": "Synthetic"},
            "chat": {"id": 7, "type": "private"}, "text": "synthetic"}
        }).encode(), "more_body": False}

    async def forbidden(*_args, **_kwargs):
        calls.append("admitted")

    from app.telegram import durable
    monkeypatch.setattr(durable, "admit_receipt", forbidden)

    class Recovery:
        bot_identity = 123456
        is_running = True
        admission_lock = asyncio.Lock()

    request = Request({"type": "http", "method": "POST", "path": "/webhook",
                       "app": SimpleNamespace(state=SimpleNamespace(
                           runtime=runtime, telegram_durable=Recovery())),
                       "headers": [(b"x-telegram-bot-api-secret-token",
                                    webhook.webhook_secret_token().encode())]}, receive)
    task = asyncio.create_task(webhook.telegram_webhook(request))
    await entered.wait()
    runtime.ready = False
    release.set()
    with pytest.raises(HTTPException) as error:
        await task
    assert error.value.status_code == 503 and calls == []


@pytest.mark.asyncio
async def test_webhook_rejects_missing_secret_header(monkeypatch):
    # Authentication precedes runtime/body access when a Bot is configured.
    monkeypatch.setattr(webhook, "settings", SimpleNamespace(telegram_bot_token="synthetic-test-token"))
    monkeypatch.setattr(webhook, "webhook_secret_token", lambda: "synthetic-secret")

    async def receive():
        raise AssertionError("unauthenticated webhook body must not be read")

    request = Request({"type": "http", "method": "POST", "path": "/webhook", "headers": []}, receive)
    with pytest.raises(HTTPException) as exc:
        await webhook.telegram_webhook(request)
    assert exc.value.status_code == 403


def test_callback_update_is_bound_to_bot():
    bot = Bot(token="123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")
    update = Update.de_json({
        "update_id": 42,
        "callback_query": {
            "id": "callback-1",
            "from": {"id": 7, "is_bot": False, "first_name": "Tester"},
            "chat_instance": "instance-1",
            "data": "act:action-1:approve",
            "message": {"message_id": 1, "date": 0,
                        "chat": {"id": 7, "type": "private"}, "text": "Proposal"},
        },
    }, bot)
    assert update.callback_query.get_bot() is bot


@pytest.mark.asyncio
async def test_malformed_callback_is_answered():
    replies = []

    async def answer(text, show_alert):
        replies.append((text, show_alert))

    callback = SimpleNamespace(data="act:bad:unexpected", answer=answer)
    await webhook._handle_action_callback(callback)
    assert replies == [("Invalid action button", True)]


@pytest.mark.asyncio
async def test_completed_callback_is_answered_without_resuming(monkeypatch):
    replies = []

    async def answer(text=None, show_alert=False):
        replies.append((text, show_alert))

    class Graph:
        async def aget_state(self, config):
            return SimpleNamespace(values={"proposal": {"action_id": "action-1"}}, next=())

    async def get_graph():
        return Graph()

    async def forbidden_resume(*args, **kwargs):
        raise AssertionError("completed action must not resume")

    monkeypatch.setattr(webhook, "get_productivity_graph", get_graph)
    monkeypatch.setattr(webhook, "get_calendar_create_coordinator", lambda: None)
    monkeypatch.setattr(webhook, "resume_workflow", forbidden_resume)
    callback = SimpleNamespace(data="act:action-1:approve",
                               from_user=SimpleNamespace(id=7),
                               message=SimpleNamespace(chat_id=7),
                               answer=answer)
    await webhook._handle_action_callback(callback)
    assert replies == [("This action has already been processed.", True)]


@pytest.mark.asyncio
async def test_duplicate_callback_while_calendar_continuation_waits_is_not_resume(monkeypatch):
    replies, delivered = [], []

    class Graph:
        async def aget_state(self, _config):
            return SimpleNamespace(values={"proposal": {"action_id": "calendar-wait"}},
                                   next=("wait_for_calendar",))

    class Store:
        async def load_create(self, _action_id):
            return SimpleNamespace(outcome="UNCERTAIN"), None

    async def graph():
        return Graph()

    async def answer(text=None, show_alert=False):
        replies.append((text, show_alert))

    async def deliver(_callback, text):
        delivered.append(text)

    async def forbidden_resume(*_args, **_kwargs):
        raise AssertionError("Repeated approval cannot wake dependent work")

    monkeypatch.setattr(webhook, "get_productivity_graph", graph)
    monkeypatch.setattr(webhook, "get_calendar_create_coordinator",
                        lambda: SimpleNamespace(store=Store()))
    monkeypatch.setattr(webhook, "resume_workflow", forbidden_resume)
    monkeypatch.setattr(webhook, "_durable_callback_reply", deliver)
    callback = SimpleNamespace(data="act:calendar-wait:approve",
                               from_user=SimpleNamespace(id=7),
                               message=SimpleNamespace(chat_id=7), answer=answer)
    await webhook._handle_action_callback(callback)
    assert len(replies) == len(delivered) == 1
    assert "not yet confirmed" in replies[0][0]
    assert delivered[0] == replies[0][0]


@pytest.mark.asyncio
async def test_duplicate_calendar_create_callback_reports_unresolved_evidence(monkeypatch):
    replies, delivered = [], []

    class Graph:
        async def aget_state(self, _config):
            return SimpleNamespace(values={"proposal": {"action_id": "calendar-a"}}, next=())

    class Store:
        async def load_create(self, action_id):
            assert action_id == "calendar-a"
            return (SimpleNamespace(operation="create_event", outcome="UNCERTAIN"),
                    SimpleNamespace(status="executing"))

    async def graph():
        return Graph()

    async def answer(text=None, show_alert=False):
        replies.append((text, show_alert))

    async def deliver(_callback, text):
        delivered.append(text)

    async def forbidden_resume(*_args, **_kwargs):
        raise AssertionError("A duplicate callback cannot resume the action")

    monkeypatch.setattr(webhook, "get_productivity_graph", graph)
    monkeypatch.setattr(webhook, "get_calendar_create_coordinator",
                        lambda: SimpleNamespace(store=Store()))
    monkeypatch.setattr(webhook, "_durable_callback_reply", deliver)
    monkeypatch.setattr(webhook, "resume_workflow", forbidden_resume)
    callback = SimpleNamespace(data="act:calendar-a:approve",
                               from_user=SimpleNamespace(id=7),
                               message=SimpleNamespace(chat_id=7), answer=answer)
    await webhook._handle_action_callback(callback)
    assert "not yet confirmed" in replies[0][0]
    assert replies[0][1] and delivered == [replies[0][0]]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["update_event", "delete_event"])
async def test_duplicate_calendar_modification_callback_reports_unresolved_evidence(
        monkeypatch, operation):
    replies, delivered = [], []

    class Graph:
        async def aget_state(self, _config):
            return SimpleNamespace(values={"proposal": {"action_id": "calendar-mod"}}, next=())

    class Store:
        async def load_create(self, action_id):
            assert action_id == "calendar-mod"
            return (SimpleNamespace(operation=operation, outcome="UNCERTAIN"),
                    SimpleNamespace(status="executing"))

    async def graph():
        return Graph()

    async def answer(text=None, show_alert=False):
        replies.append((text, show_alert))

    async def deliver(_callback, text):
        delivered.append(text)

    async def forbidden_resume(*_args, **_kwargs):
        raise AssertionError("Duplicate callback cannot reapprove a modification")

    monkeypatch.setattr(webhook, "get_productivity_graph", graph)
    monkeypatch.setattr(webhook, "get_calendar_create_coordinator", lambda: None)
    monkeypatch.setattr(webhook, "get_calendar_modify_coordinator",
                        lambda: SimpleNamespace(store=Store()))
    monkeypatch.setattr(webhook, "_durable_callback_reply", deliver)
    monkeypatch.setattr(webhook, "resume_workflow", forbidden_resume)
    callback = SimpleNamespace(data="act:calendar-mod:approve",
                               from_user=SimpleNamespace(id=7),
                               message=SimpleNamespace(chat_id=7), answer=answer)
    await webhook._handle_action_callback(callback)
    assert "not yet confirmed" in replies[0][0]
    assert replies[0][1] and delivered == [replies[0][0]]


@pytest.mark.asyncio
async def test_expired_action_callback_is_answered(monkeypatch):
    replies = []

    async def answer(text, show_alert):
        replies.append((text, show_alert))

    class Graph:
        async def aget_state(self, config):
            return SimpleNamespace(values={"proposal": None}, next=())

    async def get_graph():
        return Graph()

    monkeypatch.setattr(webhook, "get_productivity_graph", get_graph)
    callback = SimpleNamespace(data="act:expired:approve",
                               from_user=SimpleNamespace(id=7),
                               message=SimpleNamespace(chat_id=7),
                               answer=answer)
    await webhook._handle_action_callback(callback)
    assert replies == [("This action is no longer pending.", True)]


@pytest.mark.asyncio
async def test_slot_callback_resumes_only_matching_pending_choice(monkeypatch):
    replies, resumes, deliveries = [], [], []

    async def answer(text=None, show_alert=False):
        replies.append((text, show_alert))

    class Graph:
        async def aget_state(self, config):
            assert config["configurable"]["thread_id"] == "workflow_user_7"
            return SimpleNamespace(
                values={"slot_selection_id": "selection-1", "slot_candidates": [{"start": "one"}],
                        "user_id": "db-user-7"}, next=("choose_slot",),
            )

    async def get_graph():
        return Graph()

    async def resume(graph, thread_id, payload):
        resumes.append((thread_id, payload))
        return {"response": "proposal prepared"}

    async def deliver(chat_id, thread_id, result):
        deliveries.append((chat_id, thread_id, result))

    monkeypatch.setattr(webhook, "get_productivity_graph", get_graph)
    monkeypatch.setattr(webhook, "resume_workflow", resume)
    monkeypatch.setattr(webhook, "_send_workflow_result", deliver)
    callback = SimpleNamespace(data="slot:selection-1:0", from_user=SimpleNamespace(id=7),
                               message=SimpleNamespace(chat_id=7), answer=answer)
    await webhook._handle_slot_callback(callback)
    assert replies == [(None, False)]
    assert resumes == [("workflow_user_7", {"selection_index": 0})]
    assert deliveries[0][2]["response"] == "proposal prepared"

    callback.data = "slot:old-selection:0"
    await webhook._handle_slot_callback(callback)
    assert replies[-1] == ("This slot choice is no longer pending.", True)
    assert len(resumes) == 1
