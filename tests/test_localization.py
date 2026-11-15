import asyncio
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.agent.email_workflow import synthesize_email
from app.agent.gmail import GmailMessage
from app.agent.meeting_prep import MeetingPrepSnapshot, format_meeting_prep
from app.agent.workflow import format_calendar_events
from app.localization import (MESSAGES, ResponsePreferences, current_preferences, draft_language,
                              language_request, load_preferences, profile_preferences,
                              proposal_preview, reset_preferences, safe_error, save_language, tr,
                              use_preferences)
from app.scheduler.rundown import FactSnapshot, PriorityItem, format_rundown
from app.db import crud
from app.telegram import sender


class ProfileStore:
    def __init__(self):
        self.values = {}

    async def aget(self, namespace, key):
        value = self.values.get((namespace, key))
        return SimpleNamespace(value=value) if value is not None else None

    async def aput(self, namespace, key, value):
        self.values[(namespace, key)] = value


def test_explicit_language_parsing_and_no_auto_inference():
    assert language_request("/language zh-CN") == ("set", "zh-CN")
    assert language_request("/language en") == ("set", "en")
    assert language_request("以后用中文回复我") == ("set", "zh-CN")
    assert language_request("以后用英文回复我") == ("set", "en")
    assert language_request("switch to Chinese") == ("set", "zh-CN")
    assert language_request("reply in English from now on") == ("set", "en")
    assert language_request("我现在设置的回复语言是什么？") == ("query", None)
    assert language_request("请帮我翻译成英文") is None
    assert language_request("你好") is None


@pytest.mark.asyncio
async def test_language_persists_in_profile_and_isolates_users():
    store = ProfileStore()
    store.values[(("profile", "a"), "user_profile")] = {"preferences": ["concise answers"]}
    assert (await load_preferences("a", store)).language == "en"
    await save_language("a", "zh-CN", store)
    assert store.values[(("profile", "a"), "user_profile")]["preferences"] == ["concise answers"]
    assert (await load_preferences("a", store)) == ResponsePreferences("zh-CN", True)
    assert (await load_preferences("b", store)).language == "en"
    await save_language("a", "en", store)
    assert (await load_preferences("a", store)).language == "en"
    with pytest.raises(ValueError):
        await save_language("a", "fr", store)


@pytest.mark.asyncio
async def test_strict_response_preference_load_distinguishes_outage_from_empty():
    class BrokenStore:
        async def aget(self, *_args):
            raise ConnectionError("private store error")

    assert (await load_preferences("a", ProfileStore(), strict=True)).language == "en"
    with pytest.raises(RuntimeError, match="Profile Store is unavailable"):
        await load_preferences("a", BrokenStore(), strict=True)


def test_explicit_language_overrides_legacy_free_text_and_context_is_scoped():
    assert profile_preferences({"preferences": ["以后请用简短中文回答"]}).language == "zh-CN"
    assert profile_preferences({"assistant_language": "en", "preferences": ["以后请用简短中文回答"]}) == ResponsePreferences("en", True)
    token = use_preferences(ResponsePreferences("zh-CN"))
    try:
        assert tr("approve") == "批准"
        assert "Simplified Chinese" in current_preferences().llm_instruction
    finally:
        reset_preferences(token)
    assert tr("approve") == "Approve"


def test_locale_catalog_is_complete_and_ambiguity_is_localized():
    assert set(MESSAGES["en"]) == set(MESSAGES["zh-CN"])
    token = use_preferences(ResponsePreferences("zh-CN"))
    try:
        answer = safe_error(ValueError(
            "Multiple meetings match (ev-1 15:00, ev-2 16:00). Please give a title and local time."))
        assert "ev-1 15:00, ev-2 16:00" in answer
        assert "Multiple meetings match" not in answer
    finally:
        reset_preferences(token)


@pytest.mark.asyncio
async def test_concurrent_user_contexts_do_not_leak():
    async def task(language):
        token = use_preferences(ResponsePreferences(language))
        try:
            await asyncio.sleep(0)
            return tr("action_rejected")
        finally:
            reset_preferences(token)
    zh, en = await asyncio.gather(task("zh-CN"), task("en"))
    assert "已拒绝" in zh and en == "Action rejected; nothing was changed."


def test_calendar_meeting_and_rundown_deterministic_chinese():
    token = use_preferences(ResponsePreferences("zh-CN", True))
    try:
        assert format_calendar_events([], "Asia/Shanghai") == "没有找到日程。"
        assert "全天" in format_calendar_events([{"summary": "会议", "start": {"date": "2026-10-03"}}], "Asia/Shanghai")
        snapshot = MeetingPrepSnapshot(
            user_id="a", event={"id": "ev1", "summary": "论文讨论",
                                  "start": {"dateTime": "2026-10-03T15:00:00+08:00"}},
            todos=[], profile={"assistant_language": "zh-CN"}, recent_turns=[], timezone="Asia/Shanghai")
        assert format_meeting_prep(snapshot).startswith("会议准备：论文讨论")
        assert "缺失信息" in format_meeting_prep(snapshot)
        facts = FactSnapshot(user_id="a", local_date=date(2026, 10, 3), timezone="Asia/Shanghai",
                             source_errors={"Calendar": "read failed"})
        item = PriorityItem(source_id="todo:t1", title="修改论文", reason="高优先级；今天截止",
                            suggestion="确认进展", score=90)
        result = format_rundown(facts, [item])
        assert "今日重点" in result and "日历 暂时不可用" in result
        assert "Top priorities" not in result
    finally:
        reset_preferences(token)


@pytest.mark.asyncio
async def test_telegram_proposal_buttons_translate_but_callback_data_stays_stable(monkeypatch):
    sent = []

    class Bot:
        async def send_message(self, **kwargs):
            sent.append(kwargs)

    monkeypatch.setattr(sender, "get_bot", lambda: Bot())
    proposal = {"action_id": "action-1", "operation": "create_event", "domain": "calendar",
                "args": {"title": "讨论", "start": "2026-10-09T15:00:00+08:00",
                         "end": "2026-10-09T15:30:00+08:00",
                         "source_message_id": "m1", "source_thread_id": "t1"},
                "human_readable_preview": "Create event", "reason": "Requested slot is free"}
    token = use_preferences(ResponsePreferences("zh-CN"))
    try:
        await sender.send_action_proposal(1, proposal)
    finally:
        reset_preferences(token)
    buttons = sent[0]["reply_markup"].inline_keyboard[0]
    assert [button.text for button in buttons] == ["批准", "编辑", "拒绝"]
    assert [button.callback_data for button in buttons] == [
        "act:action-1:approve", "act:action-1:edit", "act:action-1:reject"]
    assert "依据 Gmail 邮件 m1；会话 t1" in sent[0]["text"]
    assert "Requested slot is free" not in sent[0]["text"]


@pytest.mark.asyncio
async def test_smart_slot_button_localizes_weekday_without_changing_callback(monkeypatch):
    sent = []

    class Bot:
        async def send_message(self, **kwargs):
            sent.append(kwargs)

    monkeypatch.setattr(sender, "get_bot", lambda: Bot())
    token = use_preferences(ResponsePreferences("zh-CN"))
    try:
        await sender.send_slot_choices(1, "selection-1", [
            {"start": "2026-10-05T12:00:00+08:00", "end": "2026-10-05T13:00:00+08:00"}])
    finally:
        reset_preferences(token)
    assert "已核实的空闲时段" in sent[0]["text"]
    button = sent[0]["reply_markup"].inline_keyboard[0][0]
    assert "周一" in button.text
    assert button.callback_data == "slot:selection-1:0"


class EnglishDraftLLM:
    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        system = messages[0].content
        assert "Assistant-to-user output language: Simplified Chinese" in system
        assert "external recipient-facing draft must be in English" in system
        return {"facts": [{"message_id": "m1", "thread_id": "t1",
                           "evidence_quote": "Please review the draft",
                           "summary": "发件人希望审阅草稿。"}],
                "draft_reply": "Thank you. I will review the draft before Friday."}


@pytest.mark.asyncio
async def test_chinese_assistant_ui_preserves_english_gmail_draft():
    mail = GmailMessage(message_id="m1", thread_id="t1", sender="peer@example.com",
                        subject="Project review", body="Please review the draft before Friday.")
    assert draft_language("Please draft a reply", mail.body) == "en"
    assert draft_language("请用中文回复这封邮件", mail.body) == "zh-CN"
    token = use_preferences(ResponsePreferences("zh-CN"))
    try:
        answer = await synthesize_email([mail], "根据邮件拟一份回复，只预览", EnglishDraftLLM(),
                                        draft=True, account_address="me@example.com")
    finally:
        reset_preferences(token)
    assert "以下是未发送的回复草稿" in answer
    assert "收件人： peer@example.com" in answer
    assert "发件人希望审阅草稿" in answer
    assert "Thank you. I will review the draft before Friday." in answer
    assert "NOT SENT" not in answer


@pytest.mark.asyncio
async def test_reminder_dispatch_uses_owners_chinese_profile(session_factory, monkeypatch,
                                                            proactive_test_runtime):
    import app.scheduler.reminders as reminders

    sent = []

    async def fake_send(chat_id, text, parse_mode=None):
        sent.append(text)

    async def chinese_profile(user_id):
        return ResponsePreferences("zh-CN")

    proactive_test_runtime(session_factory, fake_send)
    monkeypatch.setattr(reminders, "load_preferences", chinese_profile)
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "locale-reminder-user")
        todo = await crud.create_todo(db, user.id, "提交论文",
                                      due_date=datetime.now(timezone.utc) + timedelta(days=1),
                                      priority="high")
        await crud.update_todo(db, todo.id, user.id,
                               reminder_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        await db.commit()
    await reminders.check_and_send_reminders()
    assert len(sent) == 1
    assert "提醒" in sent[0] and "截止：" in sent[0]
    assert "优先级： 高" in sent[0] and "Reminder" not in sent[0]


@pytest.mark.asyncio
async def test_actionable_rundown_delivery_uses_owners_chinese_profile(session_factory, monkeypatch):
    import app.scheduler.daily_brief as daily_brief

    sent = []

    class Calendar:
        async def search_events(self, query, start=None, end=None):
            return [{"id": "meeting-1", "summary": "项目讨论",
                     "start": {"dateTime": datetime.fromisoformat(start).replace(hour=15).isoformat()},
                     "end": {"dateTime": datetime.fromisoformat(start).replace(hour=16).isoformat()}}]

    async def chinese_profile(user_id):
        return ResponsePreferences("zh-CN")

    async def send(chat_id, text):
        sent.append(text)

    monkeypatch.setattr(daily_brief, "load_preferences", chinese_profile)
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "locale-rundown-user")
        await crud.create_todo(db, user.id, "修改论文",
                               due_date=datetime.now(timezone.utc) + timedelta(hours=1),
                               priority="high")
        await db.commit()
    await daily_brief.send_daily_briefs(Calendar(), session_factory, send, target_user_id=user.id)
    assert len(sent) == 1
    assert "今日重点" in sent[0] and "项目讨论" in sent[0] and "修改论文" in sent[0]
    assert "Top priorities" not in sent[0]
