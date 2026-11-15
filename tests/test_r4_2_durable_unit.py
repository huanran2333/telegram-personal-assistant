"""E3 deterministic R4.2 faults; no Bot API or PostgreSQL claim."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.telegram.durable import normalize_update, render_chunks
from app.telegram import webhook
from telegram import Bot, Update
from langchain_core.messages import AIMessage, HumanMessage
from app.telegram.processing_context import current_run, receipt_scope


def _text_update(update_id=1, text="hello"):
    return {"update_id": update_id, "message": {
        "message_id": 1, "date": 0,
        "from": {"id": 7, "is_bot": False, "first_name": "Synthetic"},
        "chat": {"id": 7, "type": "private"}, "text": text}}


@pytest.mark.asyncio
async def test_process_owned_bot_is_closed_and_cleared(monkeypatch):
    from app.telegram import sender

    class FakeBot:
        def __init__(self):
            self.closed = 0

        async def shutdown(self):
            self.closed += 1

    first = FakeBot()
    monkeypatch.setattr(sender, "_bot", first)
    await sender.close_bot()
    assert first.closed == 1
    assert sender._bot is None
    await sender.close_bot()
    assert first.closed == 1


def test_replay_envelope_is_typed_and_excludes_unneeded_payload():
    data = _text_update(text="private synthetic request")
    data["message"]["photo"] = [{"file_id": "unused-private-attachment"}]
    data["message"]["contact"] = {"phone_number": "never-store"}
    update_id, envelope, kind, digest = normalize_update(data)
    assert update_id == 1 and kind == "text" and len(digest) == 64
    assert envelope["message"]["text"] == "private synthetic request"
    assert "photo" not in envelope["message"]
    assert "contact" not in envelope["message"]
    bot = Bot(token="123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")
    restored = Update.de_json(envelope, bot)
    assert restored.message.text == "private synthetic request"


def test_callback_and_voice_typed_envelopes_can_be_rehydrated():
    bot = Bot(token="123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")
    callback = {"update_id": 2, "callback_query": {
        "id": "synthetic-query", "from": {"id": 7, "is_bot": False,
        "first_name": "Synthetic"}, "chat_instance": "synthetic-instance",
        "data": "act:action-1:reject", "message": {"message_id": 3,
        "date": 0, "chat": {"id": 7, "type": "private"}, "text": "private old proposal"}}}
    _, stored, kind, _ = normalize_update(callback)
    assert kind == "callback" and "private old proposal" not in str(stored)
    assert Update.de_json(stored, bot).callback_query.data == "act:action-1:reject"
    voice = _text_update(update_id=3)
    del voice["message"]["text"]
    voice["message"]["voice"] = {"file_id": "synthetic-file", "file_unique_id": "synthetic-unique",
                                    "duration": 1}
    _, stored, kind, _ = normalize_update(voice)
    assert kind == "voice"
    assert Update.de_json(stored, bot).message.voice.file_id == "synthetic-file"


@pytest.mark.parametrize("bad", [None, True, -1, "1"])
def test_invalid_update_identity_rejected(bad):
    data = _text_update()
    data["update_id"] = bad
    with pytest.raises(ValueError):
        normalize_update(data)


def test_chunk_boundaries_and_atomic_keyboard_or_markup():
    text = "甲" * 8200
    chunks = render_chunks(text, parse_mode=None, keyboard=None)
    assert [len(chunk) for chunk in chunks] == [4096, 4096, 8]
    assert "".join(chunks) == text
    with pytest.raises(ValueError, match="FormattedDeliveryTooLong"):
        render_chunks(text, parse_mode="HTML", keyboard=None)
    actionable = render_chunks(text, parse_mode=None, keyboard={"inline_keyboard": []})
    assert actionable[:-1] == chunks
    assert len(actionable[-1]) <= 4096
    assert actionable[-1].startswith(text[:3500])


def test_receipt_outcome_must_precede_reply_and_cannot_change():
    with receipt_scope("synthetic-receipt", "synthetic-fence"):
        run = current_run()
        assert run.outcome_kind is None
        run.record_outcome("no_business", "start_command")
        assert run.next_reply() == 0
        with pytest.raises(ValueError, match="InvalidBusinessOutcome"):
            run.record_outcome("ordinary_checkpoint", "unverified-checkpoint")


@pytest.mark.asyncio
async def test_sender_refuses_user_visible_reply_without_business_proof(monkeypatch):
    from app.telegram import sender

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("no intent may be written without a proof")

    monkeypatch.setattr("app.telegram.durable.enqueue_reply", forbidden)
    with receipt_scope("synthetic-receipt", "synthetic-fence"):
        from app.telegram.durable import DurableSafetyError
        with pytest.raises(DurableSafetyError, match="MissingBusinessOutcomeProof"):
            await sender.send_message(7, "synthetic reply")
        assert current_run().reply_ordinal == 0


@pytest.mark.asyncio
async def test_callback_ui_ack_failure_does_not_abort_business_resume(monkeypatch):
    class Callback:
        data = "act:action-1:reject"
        from_user = SimpleNamespace(id=7)
        message = SimpleNamespace(chat_id=7)
        async def answer(self, *_args, **_kwargs):
            raise TimeoutError("synthetic ephemeral UI timeout")

    class Graph:
        async def aget_state(self, _config):
            return SimpleNamespace(
                values={"proposal": {"action_id": "action-1"}, "user_id": "synthetic-user"},
                next=("approval_gate",))

    resumed = []
    async def graph():
        return Graph()
    async def resume(_graph, _thread, decision):
        resumed.append(decision)
        return {"response": "rejected"}
    async def deliver(*_args):
        return None
    monkeypatch.setattr(webhook, "get_productivity_graph", graph)
    monkeypatch.setattr(webhook, "get_graph", AsyncMock(return_value=SimpleNamespace(
        aget_state=AsyncMock(return_value=None))))
    monkeypatch.setattr(webhook, "resume_workflow", resume)
    monkeypatch.setattr(webhook, "_send_workflow_result", deliver)
    await webhook._handle_action_callback(Callback())
    assert resumed == [{"decision": "reject"}]


@pytest.mark.asyncio
async def test_ordinary_checkpoint_final_result_replayed_without_new_invoke():
    from app.agent.graph import run_agent

    class Graph:
        async def aget_state(self, _config):
            return SimpleNamespace(values={"messages": [
                HumanMessage(id="telegram-receipt:synthetic-receipt", content="create task"),
                AIMessage(content="task created")], "user_id": "synthetic-user"}, next=())

        async def ainvoke(self, *_args, **_kwargs):
            raise AssertionError("a finalized Telegram receipt must not reinvoke")

    with receipt_scope("synthetic-receipt", "synthetic-fence"):
        result = await run_agent(Graph(), "synthetic-user", "synthetic-telegram",
                                 "synthetic-chat", "create task")
    assert result == "task created"


@pytest.mark.asyncio
async def test_typed_workflow_checkpoint_final_result_replayed_without_new_invoke():
    from app.agent.workflow import start_workflow

    class Graph:
        async def aget_state(self, _config):
            return SimpleNamespace(values={
                "transcript_turn_id": "telegram-receipt:synthetic-receipt",
                "response": "verified availability"}, next=(), tasks=())

        async def ainvoke(self, *_args, **_kwargs):
            raise AssertionError("a finalized Telegram workflow must not reinvoke")

    with receipt_scope("synthetic-receipt", "synthetic-fence"):
        result = await start_workflow(Graph(), "synthetic-user", "check calendar", "UTC",
                                      "synthetic-workflow")
    assert result["response"] == "verified availability"
