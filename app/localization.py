"""One per-user assistant-language boundary for prompts and deterministic UI.

The ContextVar is request-local, never the persistence authority. Each Telegram
request/callback and proactive delivery reloads the profile Store. Internal
action decisions and external email content do not use translated UI values.
"""
from __future__ import annotations

import contextvars
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langgraph.store.base import BaseStore


SUPPORTED_LANGUAGES = frozenset({"en", "zh-CN"})
_store: BaseStore | None = None


@dataclass(frozen=True)
class ResponsePreferences:
    language: str = "en"
    concise: bool = False

    @property
    def llm_instruction(self) -> str:
        language = "Simplified Chinese (zh-CN)" if self.language == "zh-CN" else "English"
        style = "Keep the answer concise." if self.concise else "Use a natural length."
        return (f"Assistant-to-user output language: {language}. {style} "
                "This explicit preference overrides the language of the latest user message, "
                "retrieved pages, Calendar data, Todo titles and email evidence. "
                "Do not translate source titles, quotes, identifiers or external email drafts. "
                "For a draft to an external recipient, follow the separate draft-language rule.")


_current: contextvars.ContextVar[ResponsePreferences] = contextvars.ContextVar(
    "response_preferences", default=ResponsePreferences()
)


def current_preferences() -> ResponsePreferences:
    return _current.get()


def use_preferences(preferences: ResponsePreferences):
    return _current.set(preferences)


def reset_preferences(token) -> None:
    _current.reset(token)


def bind_profile_store(store: BaseStore | None) -> None:
    global _store
    _store = store


def profile_preferences(profile: dict[str, Any] | None) -> ResponsePreferences:
    profile = profile or {}
    language = profile.get("assistant_language")
    if language not in SUPPORTED_LANGUAGES:
        language = None
    raw = profile.get("preferences") or []
    entries = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    concise = False
    legacy_language = None
    for entry in entries:
        if not isinstance(entry, str):
            continue
        lowered = entry.lower()
        if re.search(r"简短|简洁|精简|扼要|concise|brief|short", lowered):
            concise = True
        elif re.search(r"详细|详尽|detailed|verbose", lowered):
            concise = False
        if re.search(r"中文|汉语|chinese", lowered):
            legacy_language = "zh-CN"
        elif re.search(r"英文|英语|english", lowered):
            legacy_language = "en"
    return ResponsePreferences(language=language or legacy_language or "en", concise=concise)


async def load_preferences(user_id: str, store: BaseStore | None = None,
                           *, strict: bool = False) -> ResponsePreferences:
    source = store or _store
    if not source or not user_id:
        if strict:
            raise RuntimeError("Profile Store is unavailable")
        return ResponsePreferences()
    try:
        item = await source.aget(("profile", user_id), "user_profile")
    except Exception:
        if strict:
            raise RuntimeError("Profile Store is unavailable") from None
        return ResponsePreferences()
    return profile_preferences(item.value if item else None)


async def save_language(user_id: str, language: str, store: BaseStore | None = None) -> ResponsePreferences:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError("unsupported assistant language")
    source = store or _store
    if source is None or not user_id:
        raise RuntimeError("Profile Store is unavailable")
    key = ("profile", user_id)
    item = await source.aget(key, "user_profile")
    profile = dict(item.value if item and isinstance(item.value, dict) else {})
    profile["assistant_language"] = language
    await source.aput(key, "user_profile", profile)
    return profile_preferences(profile)


def language_request(message: str) -> tuple[str, str | None] | None:
    """Only explicit setting/query utterances, not inferred user locale."""
    raw = message.strip()
    command = re.fullmatch(r"/language(?:@[\w_]+)?(?:\s+(\S+))?", raw, re.I)
    if command:
        if command.group(1) is None:
            return ("query", None)
        return ("set", command.group(1))
    lowered = raw.lower().rstrip("?？。.! ")
    if re.fullmatch(r"(?:我现在设置的回复语言是什么|现在(?:的)?回复语言是什么|what is my (?:current )?(?:reply|response|assistant) language)", lowered):
        return ("query", None)
    if re.fullmatch(r"(?:以后(?:请)?用中文(?:回复我|回答我|回复|回答)|请以后用中文(?:回复我|回答我)|switch to chinese|reply in chinese from now on)", lowered):
        return ("set", "zh-CN")
    if re.fullmatch(r"(?:以后(?:请)?用英文(?:回复我|回答我|回复|回答)|请以后用英文(?:回复我|回答我)|switch to english|reply in english from now on)", lowered):
        return ("set", "en")
    return None


MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "language_set": "Assistant response language set to English.",
        "language_query": "Assistant response language: {language}.",
        "language_invalid": "Choose /language zh-CN or /language en.",
        "language_store_unavailable": "Could not save the language setting; profile storage is unavailable.",
        "start": "👋 Hi! I'm your personal assistant. Tell me what to add to your todo list, or ask me anything!",
        "help": "🤖 Personal Assistant\n\nTry: create a Todo, list tasks, check Calendar, find a free slot, prepare a meeting, search Gmail, or change language with /language zh-CN or /language en.",
        "voice_error": "Sorry, I couldn't understand your voice message.",
        "request_unclear": "I could not safely understand the request. Please rephrase it.",
        "model_unavailable": "The model service is temporarily unavailable. Nothing was changed; please try again later.",
        "agent_error": "Sorry, something went wrong. Please try again.",
        "memory_error": "Sorry, I'm having trouble accessing my memory right now. Please try again later.",
        "memory_unavailable": "Conversation memory is unavailable; nothing was changed.",
        "memory_target_missing": "I could not identify the remembered fact. Please name it more precisely; nothing was changed.",
        "memory_target_ambiguous": "More than one remembered fact may match. Please specify which one; nothing was changed.",
        "memory_forgotten": "I removed that remembered fact and its searchable conversation evidence. This did not delete any Todo, Calendar event, or email.",
        "memory_corrected": "I corrected that remembered fact. Current Todo, Calendar, Gmail, and Profile records were not changed.",
        "done": "Done.",
        "processed": "I processed your request.",
        "approve": "Approve", "edit": "Edit", "reject": "Reject",
        "invalid_button": "Invalid action button", "expired_action": "This action is no longer pending.",
        "processed_action": "This action has already been processed.",
        "approval_already_rejected": "This approval was already rejected; no action was repeated.",
        "approval_web_rejected": "This approval was rejected in the workbench; no action was repeated.",
        "approval_already_completed": "This approval is already complete; no action was repeated.",
        "approval_already_failed": "This approval has ended unsuccessfully; no action was repeated.",
        "approval_replaced": "This proposal was updated; review the latest proposal instead.",
        "proposal_heading": "Proposed action:", "reason_heading": "Reason:",
        "proposal_create_event": "Create calendar event ‘{title}’ from {start} to {end}",
        "proposal_update_event": "Update calendar event ‘{title}’", "proposal_delete_event": "Delete calendar event ‘{title}’",
        "proposal_update_interval": "New time: {start} to {end}",
        "proposal_create_todo": "Create Todo ‘{title}’", "proposal_delete_todo": "Delete Todo ‘{title}’",
        "proposal_delete_todos_by_tag": "Delete {count} Todos tagged #{tag}",
        "proposal_todo_item": "- {title} [{id}]",
        "proposal_reason": "This change requires your approval before execution.",
        "proposal_source": "From Gmail message {message_id}; thread {thread_id}",
        "slot_heading": "Verified free slots (choose one; no event has been created):",
        "edit_request": "Send the change you want, and I will show a new proposal.",
        "action_completed": "Action completed.", "action_duplicate": "Action was already processed.",
        "action_failed": "Action failed: {error}", "action_rejected": "Action rejected; nothing was changed.",
        "action_unconfirmed": "Calendar outcome is not yet confirmed. Do not request a replacement event; this action is being reconciled.",
        "action_review_required": "Calendar outcome is not confirmed and needs review. Do not submit a replacement action.",
        "cross_domain_done": "Calendar event and Todo/reminder created.",
        "cross_domain_partial": "Calendar event created, but Todo/reminder failed: {error}",
        "no_events": "No events found.", "untitled": "Untitled", "all_day": "All day", "time_unavailable": "time unavailable",
        "calendar_busy": "Calendar is busy for the requested time.", "calendar_free": "Calendar is free for the requested time.",
        "calendar_read_error": "Calendar read returned no usable result.",
        "calendar_unavailable": "Calendar unavailable: {error}. No calendar action was proposed.",
        "calendar_not_connected": "Calendar is not connected for this user; no action was executed.",
        "calendar_ambiguous": "Multiple matching events were found. Please provide a more specific title or time.",
        "calendar_not_found": "No matching Calendar event was found. Please check the title or time.",
        "todo_ambiguous": "I found zero or multiple matching Todos. Please provide an exact title or task ID.",
        "todo_tag_many": "Too many matching Todos to preview safely; narrow the request.",
        "todo_tag_none": "No Todos matched that tag; nothing was deleted.",
        "slot_busy": "That time is busy, so I did not create a calendar event.",
        "slot_none": "No free slot matched those constraints. Try a wider range or different hours.",
        "slot_invalid": "That slot selection is invalid or expired.",
        "slot_expired": "This slot choice is no longer pending.",
        "slot_read_failed": "Calendar unavailable: {error}; no free time was inferred.",
        "plan_failed": "I could not safely create an action proposal.",
        "route_legacy": "This request belongs on the existing Todo/chat path.",
        "email_no_match": "No matching Gmail messages found.",
        "email_no_subject": "(no subject)",
        "email_read_error": "Email workflow read failed: {error}. No action was proposed.",
        "email_analysis_error": "Email analysis failed; no action was executed. Please try again.",
        "email_summary_unverified": "Automated summary could not be source-verified; showing the original message list instead.",
        "email_draft_invalid": "The draft language or source quote could not be verified; no draft was shown.",
        "email_draft_label": "Draft preview only — NOT SENT", "to_label": "To:", "source_label": "Source:",
        "evidence_label": "Evidence:", "thread_label": "thread", "gmail_message_label": "Gmail message",
        "calendar_label": "Calendar:", "calendar_verified_free": "verified free in the requested window",
        "calendar_busy_intervals": "busy intervals: {intervals}",
        "rundown_greeting": "Good morning, today ({date}, {timezone}):",
        "top_priorities": "Top priorities", "next_step": "Next step: {step}",
        "rundown_empty": "No due work or Calendar events were found in the available sources.",
        "reminders_heading": "Reminders", "conflicts_heading": "Calendar conflicts",
        "conflict_line": "{first} overlaps {second}",
        "source_unavailable": "{source} unavailable ({error}); this rundown omits that source.",
        "rundown_todo_reason": "{priority} priority; {due}",
        "due_tomorrow": "due tomorrow", "rundown_todo_step": "Review progress on {title}.",
        "rundown_meeting": "scheduled meeting", "rundown_related": "related unfinished Todo: {title} [{id}]",
        "rundown_overlap": "overlaps another Calendar event", "rundown_soon": "starts within 3 hours",
        "rundown_prepare": "Prepare the unfinished task {todo} before {event}.",
        "rundown_agenda": "Check the agenda for {title}.",
        "rundown_review_conflict": "Review the Calendar overlap for {title}.",
        "source_calendar": "Calendar", "source_todo": "Todo",
        "reminder_heading": "Reminder", "due_label": "Due:", "priority_label": "Priority:",
        "no_deadline": "no deadline", "reminder_date": "{month_name} {day:02d} at {hour:02d}:{minute:02d} {timezone}",
        "daily_greeting": "Good morning, today:", "schedule_heading": "Schedule", "tasks_heading": "Tasks",
        "calendar_tasks_only": "Calendar unavailable ({error}); showing tasks only.",
        "no_calendar_events": "No calendar events found.", "no_due_tasks": "No due or overdue tasks.",
        "overdue": "overdue", "due_today": "due today",
        "meeting_title": "Meeting prep: {title}", "meeting_when": "When: {time}",
        "meeting_source": "Calendar source: {id}", "meeting_attendees": "Attendees: {names}",
        "meeting_related": "Related work:", "meeting_context": "Known context:",
        "meeting_points": "Suggested discussion points:", "meeting_missing": "Missing context: {details}",
        "meeting_no_todo": "No related Todo found.", "meeting_no_history": "No related conversation found.",
        "meeting_todo_line": "{title} [{id}]: {status}, {priority}, due {due}",
        "meeting_user_said": "User said ({source}): {text}",
        "meeting_no_prior": "No related prior discussion is stored; do not assume what attendees previously said.",
        "status_pending": "pending", "status_in_progress": "in_progress", "status_done": "done",
        "priority_high": "high", "priority_medium": "medium", "priority_low": "low",
        "meeting_confirm": "Confirm the goal and decisions needed for {title}. [Calendar {id}]",
        "meeting_progress": "Review progress and blockers for {title}. [Todo {id}]",
        "meeting_followup": "Ask which preparation tasks should follow this meeting. [Calendar {id}]",
        "meeting_no_older": "Only the cited recent user turns are available; no older discussion history was inferred.",
        "meeting_profile_unavailable": "Profile unavailable; preferences were omitted.",
        "meeting_grounding_unavailable": "I could not verify a grounded meeting brief. No agenda facts were invented.",
        "meeting_no_match": "No matching Calendar meeting was found; I did not invent an agenda.",
        "meeting_many": "Multiple meetings match ({choices}). Please give a title and local time.",
        "web_none": "I couldn't find reliable web results for that request.",
        "web_summarize_error": "I found results but could not summarize them.",
        "web_unavailable": "Sorry, web search is unavailable right now. Please try again later.",
        "chat_fallback": "Sorry, I had trouble generating a response. Could you rephrase that?",
        "clarification": "I'm not sure what you meant by '{message}'.\n\n{question}\n\nPlease respond with your choice or rephrase.",
        "safe_error": "I could not safely complete that request. Please clarify and try again.",
        "todo_bad_due": "❌ Could not parse due date: '{value}'",
        "todo_bad_reminder": "❌ Could not parse reminder time: '{value}'",
        "todo_past_reminder": "❌ The reminder time has already passed. Please choose a future time.",
        "todo_completed_reminder": "❌ This Todo is completed; no reminder was set. Reopen it before setting a reminder.",
        "todo_no_due": "no due date", "none": "none",
        "todo_created": "✅ Created todo [{id}]: '{title}' | priority: {priority} | due: {due} | reminder: {reminder} | tags: {tags}",
        "todo_empty": "📭 No todos found matching your criteria.", "todo_heading": "📋 Your todos:\n",
        "todo_line": "{icon} [{id}] {title}\n   status: {status} | priority: {priority} | 📅 {due} {tags}",
        "todo_not_found": "❌ No todo found with id starting with '{id}'",
        "todo_many": "❌ Multiple todos match; please use the task ID shown in the list.",
        "todo_update_error": "❌ Could not update todo '{id}'",
        "todo_updated": "✏️ Updated todo [{id}]: '{title}' → status: {status}",
        "todo_deleted_tag": "🗑️ Deleted {count} todo(s) tagged #{tag}",
        "todo_deleted": "🗑️ Deleted todo '{title}'", "todo_delete_failed": "❌ Delete failed",
        "todo_delete_args": "❌ Provide either todo_id or tag to delete.",
        "todo_search_empty": "🔍 No todos found matching '{query}'",
        "todo_search_heading": "🔍 Found {count} todo(s) matching '{query}':\n",
        "todo_search_line": "• [{id}] {title} | {status} | due: {due}",
        "todo_reminder_set": "⏰ Reminder set for '{title}' at {time}",
        "todo_reminder_cancelled": "The pending reminder for '{title}' was cancelled; the item and its planned time remain.",
        "todo_version_conflict": "This Todo changed since it was read; this operation was not executed. Read it again and retry with its current version.",
        "tool_unavailable": "❌ Tool not available in this environment.",
        "calendar_write_blocked": "❌ Calendar writes require an approved action proposal.",
        "calendar_user_unavailable": "Calendar unavailable for this user; no availability was inferred.",
        "todo_delete_blocked": "❌ Todo deletion requires an approved action proposal.",
        "tool_blocked": "❌ For create requests, only todo creation and calendar conflict checks are allowed.",
        "tool_error": "❌ Tool error: {error}",
        "calendar_conflict_heading": "Conflicting calendar event(s):",
        "calendar_conflict_prompt": "⚠️ Calendar conflict detected!\n\nAn event overlaps with {title} at {time}.\n{details}\nDo you still want to create this task? Reply yes or no.",
    },
    "zh-CN": {
        "language_set": "已将助手回复语言设为简体中文。",
        "language_query": "当前助手回复语言：{language}。",
        "language_invalid": "请选择 /language zh-CN 或 /language en。",
        "language_store_unavailable": "无法保存语言设置：个人偏好存储暂时不可用。",
        "start": "👋 你好！我是你的个人效率助手。你可以让我管理待办、日程、提醒，或直接提问。",
        "help": "🤖 个人效率助手\n\n可以试试：创建或查询待办、查看日程、寻找空档、准备会议、搜索邮件。使用 /language zh-CN 或 /language en 切换回复语言。",
        "voice_error": "抱歉，我没听清这条语音，请重试。",
        "request_unclear": "我无法安全地理解这个请求，请换种说法。",
        "model_unavailable": "模型服务暂时不可用，本次没有修改任何内容，请稍后重试。",
        "agent_error": "抱歉，处理时出了问题，请稍后重试。",
        "memory_error": "暂时无法读取你的个人资料，请稍后重试。",
        "memory_unavailable": "对话记忆暂时不可用；没有修改任何内容。",
        "memory_target_missing": "无法确定你指的是哪条记忆。请说得更具体一些；没有修改任何内容。",
        "memory_target_ambiguous": "有多条记忆可能匹配。请明确是哪一条；没有修改任何内容。",
        "memory_forgotten": "已移除该记忆及其可搜索的对话证据；未删除待办、日程或邮件。",
        "memory_corrected": "已更正该记忆；未修改待办、日程、邮件或个人资料记录。",
        "done": "完成。", "processed": "已处理你的请求。",
        "approve": "批准", "edit": "编辑", "reject": "拒绝",
        "invalid_button": "无效的操作按钮", "expired_action": "该操作已不再等待审批。",
        "processed_action": "该操作已经处理过。",
        "approval_already_rejected": "该审批已被拒绝，无需重复操作。",
        "approval_web_rejected": "该审批已在工作台中被拒绝，无需重复操作。",
        "approval_already_completed": "该审批已完成，无需重复操作。",
        "approval_already_failed": "该审批已结束但未成功，无需重复操作。",
        "approval_replaced": "该提案已更新，请查看最新提案。",
        "proposal_heading": "待审批操作：", "reason_heading": "原因：",
        "proposal_create_event": "创建日程「{title}」：{start} 至 {end}",
        "proposal_update_event": "修改日程「{title}」", "proposal_delete_event": "删除日程「{title}」",
        "proposal_update_interval": "新时间：{start} 至 {end}",
        "proposal_create_todo": "创建待办「{title}」", "proposal_delete_todo": "删除待办「{title}」",
        "proposal_delete_todos_by_tag": "删除 {count} 条带 #{tag} 标签的待办",
        "proposal_todo_item": "- {title} [{id}]",
        "proposal_reason": "执行前需要你确认。",
        "proposal_source": "依据 Gmail 邮件 {message_id}；会话 {thread_id}",
        "slot_heading": "已核实的空闲时段（请选择；尚未创建日程）：",
        "edit_request": "请发送修改要求，我会重新给出待审批提案。",
        "action_completed": "操作已完成。", "action_duplicate": "该操作已经处理过。",
        "action_failed": "操作失败：{error}", "action_rejected": "已拒绝操作，未作任何修改。",
        "action_unconfirmed": "日程操作结果尚未确认。请勿重新提交相同操作；系统正在核对这次操作。",
        "action_review_required": "日程操作结果尚未确认，需要人工复核。请勿重新提交相同操作。",
        "cross_domain_done": "日程、待办和提醒已创建。",
        "cross_domain_partial": "日程已创建，但待办或提醒失败：{error}",
        "no_events": "没有找到日程。", "untitled": "未命名日程", "all_day": "全天", "time_unavailable": "时间不可用",
        "calendar_busy": "请求的时段已有安排。", "calendar_free": "请求的时段空闲。",
        "calendar_read_error": "日程读取结果不可用。",
        "calendar_unavailable": "日历不可用：{error}。未提出日历写入操作。",
        "calendar_not_connected": "此用户未连接日历，未执行任何操作。",
        "calendar_ambiguous": "找到多个匹配日程，请提供更具体的标题或时间。",
        "calendar_not_found": "没有找到匹配的日程，请核对标题或时间。",
        "todo_ambiguous": "找到零个或多个匹配待办，请提供准确标题或任务 ID。",
        "todo_tag_many": "匹配待办过多，请缩小范围后再预览。",
        "todo_tag_none": "没有匹配该标签的待办，未删除任何内容。",
        "slot_busy": "该时段已有安排，因此没有创建日程。",
        "slot_none": "没有符合条件的空闲时段，请扩大日期范围或调整时间。",
        "slot_invalid": "所选时段无效或已过期。",
        "slot_expired": "该时段选择已不再等待确认。",
        "slot_read_failed": "日历不可用：{error}；不能据此判断空闲。",
        "plan_failed": "无法安全地生成操作提案。",
        "route_legacy": "该请求应由待办或聊天流程处理。",
        "email_no_match": "没有找到匹配的 Gmail 邮件。",
        "email_no_subject": "（无主题）",
        "email_read_error": "邮件流程读取失败：{error}。未提出操作。",
        "email_analysis_error": "邮件分析失败，未执行任何操作，请重试。",
        "email_summary_unverified": "自动摘要无法核对来源，改为显示原始邮件列表。",
        "email_draft_invalid": "无法核对草稿语言或来源引用，因此未显示草稿。",
        "email_draft_label": "以下是未发送的回复草稿", "to_label": "收件人：", "source_label": "来源：",
        "evidence_label": "证据：", "thread_label": "会话", "gmail_message_label": "Gmail 邮件",
        "calendar_label": "日历：", "calendar_verified_free": "请求时段已核实为空闲",
        "calendar_busy_intervals": "忙碌时段：{intervals}",
        "rundown_greeting": "早上好，今天（{date}，{timezone}）：",
        "top_priorities": "今日重点", "next_step": "建议下一步：{step}",
        "rundown_empty": "可用来源中没有今天到期的任务或日程。",
        "reminders_heading": "提醒", "conflicts_heading": "日程冲突",
        "conflict_line": "{first} 与 {second} 时间重叠",
        "source_unavailable": "{source} 暂时不可用（{error}）；本次概览未包含该来源。",
        "rundown_todo_reason": "{priority}优先级；{due}",
        "due_tomorrow": "明天截止", "rundown_todo_step": "确认「{title}」的进展。",
        "rundown_meeting": "已安排的会议", "rundown_related": "相关未完成待办：「{title}」[{id}]",
        "rundown_overlap": "与其他日程冲突", "rundown_soon": "3 小时内开始",
        "rundown_prepare": "在「{event}」前准备未完成的任务「{todo}」。",
        "rundown_agenda": "检查「{title}」的会议议程。",
        "rundown_review_conflict": "检查「{title}」的日程冲突。",
        "source_calendar": "日历", "source_todo": "待办",
        "reminder_heading": "提醒", "due_label": "截止：", "priority_label": "优先级：",
        "no_deadline": "无截止日期", "reminder_date": "{month}月{day:02d}日 {hour:02d}:{minute:02d} {timezone}",
        "daily_greeting": "早上好，今天：", "schedule_heading": "日程", "tasks_heading": "任务",
        "calendar_tasks_only": "日历暂时不可用（{error}）；仅显示任务。",
        "no_calendar_events": "没有找到日程。", "no_due_tasks": "没有今天到期或已逾期的任务。",
        "overdue": "已逾期", "due_today": "今天截止",
        "meeting_title": "会议准备：{title}", "meeting_when": "时间：{time}",
        "meeting_source": "日程来源：{id}", "meeting_attendees": "参与者：{names}",
        "meeting_related": "相关任务：", "meeting_context": "已知背景：",
        "meeting_points": "讨论建议：", "meeting_missing": "缺失信息：{details}",
        "meeting_no_todo": "未找到相关待办。", "meeting_no_history": "无相关对话记录。",
        "meeting_todo_line": "{title} [{id}]：{status}，优先级{priority}，截止 {due}",
        "meeting_user_said": "用户曾说（{source}）：{text}",
        "meeting_no_prior": "没有更早的讨论记录，不推测参与者此前说过什么。",
        "status_pending": "待办", "status_in_progress": "进行中", "status_done": "已完成",
        "priority_high": "高", "priority_medium": "中", "priority_low": "低",
        "meeting_confirm": "确认本次会议目标与待决事项。[日程 {id}]",
        "meeting_progress": "确认「{title}」的进展与阻碍。[待办 {id}]",
        "meeting_followup": "询问会后需要跟进什么。[日程 {id}]",
        "meeting_no_older": "仅引用上述用户消息；不推测更早的讨论。",
        "meeting_profile_unavailable": "个人资料不可用，未应用偏好。",
        "meeting_grounding_unavailable": "无法核实有来源依据的会议准备内容，因此未编造议程事实。",
        "meeting_no_match": "没有找到匹配的日历会议；未编造议程。",
        "meeting_many": "有多个匹配会议（{choices}），请提供标题和当地时间。",
        "web_none": "没有找到可靠的网页结果。", "web_summarize_error": "找到了结果，但暂时无法可靠地总结。",
        "web_unavailable": "网页搜索暂时不可用，请稍后重试。",
        "chat_fallback": "暂时无法生成回答，请换种说法。",
        "clarification": "我不确定“{message}”的意思。\n\n{question}\n\n请说明你的选择或换种说法。",
        "safe_error": "无法安全地完成该请求，请补充信息后重试。",
        "todo_bad_due": "❌ 无法解析截止日期：“{value}”",
        "todo_bad_reminder": "❌ 无法解析提醒时间：“{value}”",
        "todo_past_reminder": "❌ 提醒时间已经过去，请选择未来时间。",
        "todo_completed_reminder": "❌ 待办已完成，本次未设置提醒；如需提醒，请先重新打开该待办。",
        "todo_no_due": "无截止日期", "none": "无",
        "todo_created": "✅ 已创建待办 [{id}]：「{title}」｜优先级：{priority}｜截止：{due}｜提醒：{reminder}｜标签：{tags}",
        "todo_empty": "📭 没有符合条件的待办。", "todo_heading": "📋 你的待办：\n",
        "todo_line": "{icon} [{id}] {title}\n   状态：{status}｜优先级：{priority}｜📅 {due} {tags}",
        "todo_not_found": "❌ 没有找到 ID 或标题匹配“{id}”的待办。",
        "todo_many": "❌ 匹配多个待办，请使用列表中的任务 ID。",
        "todo_update_error": "❌ 无法更新待办“{id}”。",
        "todo_updated": "✏️ 已更新待办 [{id}]：「{title}」→ 状态：{status}",
        "todo_deleted_tag": "🗑️ 已删除 {count} 条带 #{tag} 标签的待办。",
        "todo_deleted": "🗑️ 已删除待办「{title}」。", "todo_delete_failed": "❌ 删除失败。",
        "todo_delete_args": "❌ 请提供待办 ID 或标签。",
        "todo_search_empty": "🔍 没有找到匹配“{query}”的待办。",
        "todo_search_heading": "🔍 找到 {count} 条匹配“{query}”的待办：\n",
        "todo_search_line": "• [{id}] {title}｜{status}｜截止：{due}",
        "todo_reminder_set": "⏰ 已为「{title}」设置提醒：{time}",
        "todo_reminder_cancelled": "已取消「{title}」尚未发送的提醒；事项和排期保留。",
        "todo_version_conflict": "待办在读取后已发生变化；本次操作未执行。请重新查询，并使用最新版本重试。",
        "tool_unavailable": "❌ 当前环境不支持该工具。",
        "calendar_write_blocked": "❌ 日历写入必须先经过操作提案审批。",
        "calendar_user_unavailable": "该用户无法访问日历，因此未判断空闲状态。",
        "todo_delete_blocked": "❌ 删除待办必须先经过操作提案审批。",
        "tool_blocked": "❌ 创建请求仅允许待办创建和日历冲突检查。",
        "tool_error": "❌ 工具调用失败：{error}",
        "calendar_conflict_heading": "有冲突的日程：",
        "calendar_conflict_prompt": "⚠️ 发现日程冲突！\n\n「{title}」在 {time} 与已有日程重叠。\n{details}\n仍要创建这条任务吗？请回复 yes 或 no。",
    },
}


def tr(key: str, *, preferences: ResponsePreferences | None = None, **values: Any) -> str:
    language = (preferences or current_preferences()).language
    try:
        template = MESSAGES[language][key]
    except KeyError as exc:
        raise KeyError(f"Missing localization key {language}.{key}") from exc
    return template.format(**values)


def safe_error(exc: Exception | str) -> str:
    """Never expose English transport/LLM errors as localized user prose."""
    value = str(exc)
    if current_preferences().language == "en":
        return value
    for source, key in {
        "No matching Calendar meeting": "meeting_no_match",
        "Multiple meetings match": "meeting_many",
        "No matching Gmail": "email_no_match",
        "Calendar is not connected": "calendar_not_connected",
    }.items():
        if value.startswith(source):
            if key == "meeting_many":
                candidates = re.search(r"\(([^)]*)\)", value)
                return tr(key, choices=candidates.group(1) if candidates else "")
            return tr(key)
    return tr("safe_error")


def format_date(value: datetime, timezone: str, *, with_day: bool = False) -> str:
    local = value.astimezone(ZoneInfo(timezone)) if value.tzinfo else value
    stamp = local.strftime("%Y-%m-%d %H:%M")
    if not with_day:
        return stamp
    day = local.weekday()
    label = (["周一", "周二", "周三", "周四", "周五", "周六", "周日"] if current_preferences().language == "zh-CN"
             else ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])[day]
    return f"{label} {stamp}"


def proposal_preview(proposal: dict[str, Any]) -> str:
    operation = proposal.get("operation", "")
    args = proposal.get("args", {})
    key = {
        "create_event": "proposal_create_event", "update_event": "proposal_update_event",
        "delete_event": "proposal_delete_event", "create_todo": "proposal_create_todo",
        "create_todo_from_email": "proposal_create_todo", "delete_todo": "proposal_delete_todo",
        "delete_todos_by_tag": "proposal_delete_todos_by_tag",
    }.get(operation)
    if key is None:
        return tr("plan_failed")
    title = args.get("title") or args.get("display_title") or args.get("summary") or args.get("eventId") or ""
    preview = tr(key, title=title, start=args.get("start", ""), end=args.get("end", ""),
                 count=len(args.get("todo_ids", [])), tag=args.get("tag", ""))
    if operation == "update_event" and args.get("start") and args.get("end"):
        preview += "\n" + tr("proposal_update_interval", start=args["start"], end=args["end"])
    if operation == "delete_todos_by_tag":
        preview += "\n" + "\n".join(tr("proposal_todo_item", title=title, id=todo_id[:8])
                                  for title, todo_id in zip(args.get("todo_titles", []), args.get("todo_ids", [])))
    if args.get("source_message_id"):
        preview += "\n" + tr("proposal_source", message_id=args["source_message_id"],
                              thread_id=args.get("source_thread_id", ""))
    return preview


def draft_language(request: str, source_text: str) -> str:
    """Recipient-facing language is independent of assistant_language."""
    lowered = request.lower()
    if re.search(r"(?:草稿|回复|回信|draft|reply).{0,24}(?:英文|英语|english)|(?:用英文|in english).{0,24}(?:草稿|回复|回信|draft|reply)", lowered):
        return "en"
    if re.search(r"(?:草稿|回复|回信|draft|reply).{0,24}(?:中文|汉语|chinese)|(?:用中文|in chinese).{0,24}(?:草稿|回复|回信|draft|reply)", lowered):
        return "zh-CN"
    chinese = len(re.findall(r"[\u4e00-\u9fff]", source_text))
    latin = len(re.findall(r"[A-Za-z]", source_text))
    return "zh-CN" if chinese > latin else "en"


def draft_matches_language(text: str | None, language: str) -> bool:
    if not text or not text.strip():
        return False
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return chinese >= 2 if language == "zh-CN" else chinese <= max(3, latin // 10)
