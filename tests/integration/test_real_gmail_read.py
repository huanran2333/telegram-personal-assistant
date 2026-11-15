"""Opt-in, read-only Gmail API acceptance. Never prints mailbox contents."""
import os
from pathlib import Path
from email.utils import parseaddr

import pytest

from app.agent.actions import requires_approval
from app.agent.email_workflow import propose_email_action, synthesize_email
from app.agent.gmail import MAX_BODY_CHARS, OAuthGmailSource
from app.agent.nodes import _build_chat_llm
from app.config import get_settings
from app.localization import ResponsePreferences, reset_preferences, use_preferences, draft_matches_language


pytestmark = [pytest.mark.gmail_integration, pytest.mark.asyncio]


def _test_subject() -> str:
    return os.getenv("GMAIL_E2E_SUBJECT", "PPA V2 Gmail E2E")


def _test_query() -> str:
    return f'subject:"{_test_subject()}"'


async def test_real_gmail_account_search_unread_and_thread():
    settings = get_settings()
    if not settings.gmail_enabled or not settings.gmail_token_file:
        pytest.skip("Gmail integration is not configured")
    token_file = Path(settings.gmail_token_file)
    if not token_file.is_file():
        pytest.skip("Separate Gmail read-only OAuth authorization has not completed")

    source = OAuthGmailSource(token_file)
    assert "@" in await source.account_address()
    query = _test_query()
    messages = await source.search(query, limit=2)
    unread = await source.unread(query, limit=2)
    assert len(messages) <= 2 and len(unread) <= 2
    for message in messages + unread:
        assert message.message_id and message.thread_id
        assert len(message.body) <= MAX_BODY_CHARS
    if messages:
        thread = await source.thread(messages[0].thread_id)
        assert thread and all(item.thread_id == messages[0].thread_id for item in thread)


async def test_real_gmail_test_message_to_sourced_summary():
    settings = get_settings()
    if not settings.gmail_enabled or not settings.gmail_token_file:
        pytest.skip("Gmail integration is not configured")
    token_file = Path(settings.gmail_token_file)
    if not token_file.is_file():
        pytest.skip("Separate Gmail read-only OAuth authorization has not completed")

    source = OAuthGmailSource(token_file)
    messages = await source.search(_test_query(), limit=5)
    if not messages:
        pytest.skip("User has not sent the harmless Gmail E2E message")
    selected = max(messages, key=lambda item: item.timestamp)
    assert _test_subject() in selected.subject
    llm = _build_chat_llm(settings.llm_provider, settings.llm_model, routing=True)
    result = await synthesize_email(
        [selected], "Summarize only this test email; do not send",
        llm)
    if f"Gmail message {selected.message_id}; thread {selected.thread_id}" not in result:
        pytest.fail("Real Gmail summary omitted its source IDs")
    if "Automated summary could not be source-verified" in result:
        pytest.fail("Real LLM used the safe source-list fallback instead of a verified summary")


async def test_real_gmail_test_message_draft_preview_only():
    settings = get_settings()
    if not settings.gmail_enabled or not settings.gmail_token_file:
        pytest.skip("Gmail integration is not configured")
    source = OAuthGmailSource(Path(settings.gmail_token_file))
    messages = await source.search(_test_query(), limit=5)
    if not messages:
        pytest.skip("User has not sent the harmless Gmail E2E message")
    selected = max(messages, key=lambda item: item.timestamp)
    assert _test_subject() in selected.subject
    account = await source.account_address()
    if parseaddr(selected.sender)[1].lower() == account:
        pytest.skip("Test message is self-sent; a reply draft needs a different sender")
    llm = _build_chat_llm(settings.llm_provider, settings.llm_model, routing=True)
    result = await synthesize_email(
        [selected], "Draft a brief reply to this one test email; do not send",
        llm, draft=True, account_address=account)
    assert f"Gmail message {selected.message_id}; thread {selected.thread_id}" in result
    assert "Draft preview only — NOT SENT" in result


async def test_real_english_email_keeps_english_draft_under_chinese_assistant():
    subject = os.getenv("GMAIL_LANG_E2E_SUBJECT")
    if not subject:
        pytest.skip("Set GMAIL_LANG_E2E_SUBJECT to a harmless English test email marker")
    settings = get_settings()
    if not settings.gmail_enabled or not settings.gmail_token_file:
        pytest.skip("Gmail integration is not configured")
    source = OAuthGmailSource(Path(settings.gmail_token_file))
    messages = await source.search(f'subject:"{subject}"', limit=5)
    if not messages:
        messages = await source.search(subject, limit=5)
    if not messages:
        pytest.fail("Dedicated English localization message was not found")
    selected = max(messages, key=lambda item: item.timestamp)
    assert subject in selected.subject
    account = await source.account_address()
    if parseaddr(selected.sender)[1].lower() == account:
        pytest.fail("Localization test email must come from a different account")
    llm = _build_chat_llm(settings.llm_provider, settings.llm_model, routing=True)
    token = use_preferences(ResponsePreferences(language="zh-CN"))
    try:
        result = await synthesize_email(
            [selected], "根据这封来信拟一份简短回复，只预览，不发送。",
            llm, draft=True, account_address=account,
        )
    finally:
        reset_preferences(token)
    assert selected.message_id in result and selected.thread_id in result
    assert "以下是未发送的回复草稿" in result
    draft_body = result.rsplit("\n\n", 1)[-1]
    assert draft_matches_language(draft_body, "en")


async def test_real_gmail_test_message_to_todo_proposal_without_execution():
    settings = get_settings()
    if not settings.gmail_enabled or not settings.gmail_token_file:
        pytest.skip("Gmail integration is not configured")
    source = OAuthGmailSource(Path(settings.gmail_token_file))
    messages = await source.search(_test_query(), limit=5)
    if not messages:
        pytest.skip("User has not sent the harmless Gmail E2E message")
    selected = max(messages, key=lambda item: item.timestamp)
    assert _test_subject() in selected.subject
    llm = _build_chat_llm(settings.llm_provider, settings.llm_model, routing=True)
    proposal = await propose_email_action(
        selected, "Make a Todo to confirm the task progress requested in this email",
        "create_todo", settings.default_timezone, llm)
    assert proposal.args["source_message_id"] == selected.message_id
    assert proposal.args["source_thread_id"] == selected.thread_id
    assert requires_approval(proposal)
