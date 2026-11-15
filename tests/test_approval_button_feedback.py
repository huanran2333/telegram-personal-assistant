from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.telegram.approval_markup import _resolved
from app.telegram import webhook
from app.telegram.webhook import _terminal_approval_notice


def test_saver_terminal_status_controls_stale_button_notice():
    assert "workbench" in _terminal_approval_notice({
        "proposal": {"status": "rejected"}, "approval_web_request_id": "run"}).lower()
    assert "already rejected" in _terminal_approval_notice({
        "proposal": {"status": "rejected"}}).lower()
    assert "already complete" in _terminal_approval_notice({
        "proposal": {"status": "executed"}}).lower()
    assert "unsuccessfully" in _terminal_approval_notice({
        "proposal": {"status": "failed"}}).lower()


def test_only_terminal_or_replaced_saver_proposal_may_be_edited():
    def state(action, status, next_nodes=()):
        return SimpleNamespace(values={"proposal": {
            "action_id": action, "status": status}}, next=next_nodes)

    assert not _resolved(state("one", "proposed", ("approval_gate",)), "one")
    assert not _resolved(state("one", "approved", ("wait_for_calendar",)), "one")
    assert _resolved(state("one", "rejected"), "one")
    assert _resolved(state("one", "executed"), "one")
    assert _resolved(state("two", "proposed", ("approval_gate",)), "one")
    assert not _resolved(None, "one")


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.parametrize("from_workbench", [True, False])
async def test_stale_button_after_reject_alerts_without_resuming(
        monkeypatch, decision, from_workbench):
    notices = []

    class Graph:
        async def aget_state(self, _config):
            return SimpleNamespace(values={
                "proposal": {"action_id": "action-1", "status": "rejected"},
                "approval_web_request_id": "web-1" if from_workbench else None}, next=())

    async def graph():
        return Graph()

    async def forbidden_resume(*_args, **_kwargs):
        raise AssertionError("A terminal Saver proposal must not resume")

    async def answer(text=None, show_alert=False):
        notices.append((text, show_alert))

    async def durable_reply(_callback, _text):
        pass

    monkeypatch.setattr(webhook, "get_productivity_graph", graph)
    monkeypatch.setattr(webhook, "get_user_by_telegram_id", AsyncMock(
        return_value=SimpleNamespace(id="synthetic-user")))
    monkeypatch.setattr(webhook, "get_graph", AsyncMock(return_value=SimpleNamespace(
        aget_state=AsyncMock(return_value=None))))
    monkeypatch.setattr(webhook, "get_calendar_create_coordinator", lambda: None)
    monkeypatch.setattr(webhook, "get_calendar_modify_coordinator", lambda: None)
    monkeypatch.setattr(webhook, "resume_workflow", forbidden_resume)
    monkeypatch.setattr(webhook, "_durable_callback_reply", durable_reply)
    callback = SimpleNamespace(
        data=f"act:action-1:{decision}", from_user=SimpleNamespace(id=7),
        message=SimpleNamespace(chat_id=7), answer=answer)
    await webhook._handle_action_callback(callback)
    expected = ("This approval was rejected in the workbench; no action was repeated."
                if from_workbench else "This approval was already rejected; no action was repeated.")
    assert notices == [(expected, True)]
