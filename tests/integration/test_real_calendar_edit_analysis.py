"""Opt-in real LLM extraction of a pending edit; no Calendar side effect."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.agent.actions import ActionProposal, RiskLevel
from app.agent.planner import apply_edit
from app.config import get_settings

pytestmark = pytest.mark.llm_integration


@pytest.mark.asyncio
@pytest.mark.parametrize("instruction", ["时间改为五点半", "时间改为下午五点半开始"])
async def test_real_llm_reanalyzes_five_thirty_edit(instruction):
    timezone = get_settings().default_timezone
    original_start = datetime(2026, 10, 1, 17, 0, tzinfo=ZoneInfo(timezone))
    original_end = original_start + timedelta(minutes=30)
    original = ActionProposal(
        domain="calendar", operation="create_event",
        args={"title": "测试会议", "start": original_start.isoformat(),
              "end": original_end.isoformat(), "timeZone": timezone,
              "user_id": "edit-analysis-only"},
        human_readable_preview="Create calendar event",
        reason="Pending proposal", risk_level=RiskLevel.EXTERNAL_WRITE,
    )
    revised = await apply_edit(original, instruction, timezone)
    assert revised.action_id != original.action_id
    assert revised.args["start"] == original_start.replace(minute=30).isoformat()
    assert revised.args["end"] == original_start.replace(hour=18).isoformat()
    assert revised.args["title"] == "测试会议"
    assert original.args["start"] == original_start.isoformat()
