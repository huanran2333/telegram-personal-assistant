"""Opt-in private-model classification regression; synthetic text only."""
import pytest

from app.agent.nodes import _build_chat_llm
from app.agent.planner import classify_core_intent
from app.config import get_settings


pytestmark = [pytest.mark.llm_integration, pytest.mark.asyncio]


@pytest.mark.parametrize(("utterance", "intent", "operation"), [
    ("把「合成月报」的提醒改为2032年2月10日早上7点。", "todo", "update"),
    ("已经有待办「合成清单」，请在2032年5月12日18点提醒我做它。", "todo", "update"),
    ("取消「合成报告」原提醒，改到2032年6月1日下午两点。", "todo", "update"),
    ("新增待办「合成试验」，2032年8月10日提醒我。", "todo", "create"),
    ("完成待办「合成试验」。", "todo", "update"),
    ("查看2032年8月10日的日历安排。", "calendar_read", None),
])
async def test_core_route_distinguishes_existing_reminder_from_new_task(utterance, intent, operation):
    settings = get_settings()
    model = _build_chat_llm(settings.llm_provider, settings.llm_model)
    route = await classify_core_intent(utterance, model)
    assert (route.intent, route.todo_operation) == (intent, operation)
    if intent == "todo":
        assert route.title
