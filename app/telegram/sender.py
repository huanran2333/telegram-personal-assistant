"""
Telegram sender — wraps python-telegram-bot's Bot for sending messages.
"""
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.constants import ParseMode
from app.config import get_settings
from app.trace import emit_trace
from app.localization import current_preferences, format_date, proposal_preview, tr

settings = get_settings()

_bot: Bot = None


def get_bot() -> Bot:
    global _bot
    if _bot is None:
        _bot = Bot(token=settings.telegram_bot_token)
    return _bot


async def close_bot() -> None:
    """Release the process-owned PTB HTTP clients before its event loop ends."""
    global _bot
    bot, _bot = _bot, None
    if bot is not None:
        await bot.shutdown()


async def _queue_interactive(chat_id, text: str, *, kind: str,
                             parse_mode: str | None = None,
                             keyboard: InlineKeyboardMarkup | None = None) -> bool:
    from app.telegram.processing_context import current_run
    run = current_run()
    if run is None:
        return False
    from app.db.session import AsyncSessionLocal
    from app.telegram.durable import DurableSafetyError, enqueue_reply
    if run.outcome_kind is None or run.outcome_ref is None or run.required_reply_count is None:
        raise DurableSafetyError("MissingBusinessOutcomeProof")
    if run.required_reply_count > 1 and kind != "hitl_proposal":
        raise DurableSafetyError("InvalidMultiReplyKind")
    try:
        await enqueue_reply(
            AsyncSessionLocal, receipt_id=run.receipt_id,
            fence_token=run.fence_token,
            reply_ordinal=run.next_reply(), kind=kind, chat_id=str(chat_id),
            locale=current_preferences().language, text=text, parse_mode=parse_mode,
            keyboard=keyboard.to_dict() if keyboard else None,
            outcome_kind=run.outcome_kind, outcome_ref=run.outcome_ref,
            required_reply_count=run.required_reply_count)
    except DurableSafetyError:
        raise
    except Exception as exc:
        # A failed intent write is never a user-facing business failure. Do
        # not let webhook's broad Agent-error handler replace a successful
        # action with a misleading error message at a new reply ordinal.
        raise DurableSafetyError("ReplyIntentUnavailable") from exc
    emit_trace("delivery_intent", operation=kind, success=True)
    return True


async def send_message(chat_id: str | int, text: str, parse_mode: str = None) -> None:
    """Send a plain text message to a Telegram chat."""
    if await _queue_interactive(chat_id, text, kind="interactive_reply", parse_mode=parse_mode):
        return
    bot = get_bot()
    # Telegram max message length is 4096 chars; split if needed
    max_len = 4096
    if len(text) <= max_len:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
    else:
        for i in range(0, len(text), max_len):
            await bot.send_message(chat_id=chat_id, text=text[i:i + max_len], parse_mode=parse_mode)
    emit_trace("delivery", tool="telegram.send_message", success=True)


async def send_typing(chat_id: str | int) -> None:
    """Send 'typing...' action."""
    bot = get_bot()
    await bot.send_chat_action(chat_id=chat_id, action="typing")


async def send_action_proposal(chat_id: str | int, proposal: dict,
                               *, completed_summary: str = "") -> None:
    """Render a durable action id as compact callback data, never raw args."""
    action_id = proposal["action_id"]
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton(tr("approve"), callback_data=f"act:{action_id}:approve"),
        InlineKeyboardButton(tr("edit"), callback_data=f"act:{action_id}:edit"),
        InlineKeyboardButton(tr("reject"), callback_data=f"act:{action_id}:reject"),
    ]])
    rendered = f"{tr('proposal_heading')}\n{proposal_preview(proposal)}\n\n{tr('reason_heading')} {tr('proposal_reason')}"
    if completed_summary:
        prefix_budget = max(0, min(1000, 4096 - len(rendered) - 2))
        if prefix_budget:
            rendered = completed_summary[:prefix_budget] + "\n\n" + rendered
    if not await _queue_interactive(chat_id, rendered, kind="hitl_proposal", keyboard=keyboard):
        await get_bot().send_message(chat_id=chat_id, text=rendered, reply_markup=keyboard)
    emit_trace("delivery", operation="action_proposal", tool="telegram.inline_keyboard", success=True,
               action_id=action_id)


async def send_official_action_proposals(chat_id: str | int, proposals: list[dict]) -> None:
    """One durable Telegram card per independent official interrupt.

    Callback data contains only the stable business action ID. The webhook
    re-reads Saver to bind it to the exact still-pending interrupt ID.
    """
    if not proposals:
        raise ValueError("OfficialApprovalEmpty")
    if len({item.get("action_id") for item in proposals}) != len(proposals):
        raise ValueError("OfficialApprovalDuplicateAction")
    for proposal in proposals:
        action_id = proposal["action_id"]
        preview = proposal_preview(proposal)
        rendered = f"{tr('proposal_heading')}\n{preview}"
        row = [
            InlineKeyboardButton(tr("approve"), callback_data=f"act:{action_id}:approve"),
            InlineKeyboardButton(tr("reject"), callback_data=f"act:{action_id}:reject"),
        ]
        if len(rendered) > 4096 or any(len(button.callback_data.encode("utf-8")) > 64
                                        for button in row):
            raise ValueError("OfficialApprovalTelegramPayloadTooLarge")
        keyboard = InlineKeyboardMarkup([row])
        if not await _queue_interactive(chat_id, rendered, kind="hitl_proposal", keyboard=keyboard):
            await get_bot().send_message(chat_id=chat_id, text=rendered, reply_markup=keyboard)
    emit_trace("delivery", operation="official_action_proposals",
               tool="telegram.inline_keyboard", success=True,
               approval_count=len(proposals))


async def send_workbench_launch(chat_id: str | int, url: str) -> None:
    """A real Telegram web_app button carried by the sealed R4 reply path."""
    from app.telegram.durable import DurableSafetyError
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("打开工作台", web_app=WebAppInfo(url=url))
    ]])
    from app.telegram.workbench_launch import WORKBENCH_LAUNCH_TEXT
    if not await _queue_interactive(chat_id, WORKBENCH_LAUNCH_TEXT,
                                    kind="workbench_launch", keyboard=keyboard):
        raise DurableSafetyError("WorkbenchLaunchRequiresDurableReceipt")


async def send_slot_choices(chat_id: str | int, selection_id: str, candidates: list[dict]) -> None:
    """One compact callback per verified slot; no event parameters in callback_data."""
    rows = []
    for index, candidate in enumerate(candidates):
        from datetime import datetime
        start = datetime.fromisoformat(candidate["start"])
        end = datetime.fromisoformat(candidate["end"])
        timezone = get_settings().default_timezone
        label = f"{format_date(start, timezone, with_day=True)}–{format_date(end, timezone)[-5:]}"
        rows.append([InlineKeyboardButton(label, callback_data=f"slot:{selection_id}:{index}")])
    keyboard = InlineKeyboardMarkup(rows)
    if not await _queue_interactive(chat_id, tr("slot_heading"), kind="slot_choices", keyboard=keyboard):
        await get_bot().send_message(chat_id=chat_id, text=tr("slot_heading"), reply_markup=keyboard)
    emit_trace("delivery", operation="slot_selection", tool="telegram.inline_keyboard", success=True)
