"""Safety checks for the synthetic unified-Agent evaluation environment."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from app.agent.unified_tools import AgentToolScope, agent_tool_scope
from evals.unified_agent_acceptance import SyntheticWorld, USER, run_case


class SequenceModel:
    def __init__(self, *responses):
        self.responses = list(responses)

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        return self.responses.pop(0)


def synthetic_settings(monkeypatch):
    from app.agent import unified_loop, unified_tools
    settings = SimpleNamespace(
        google_calendar_enabled=True, google_calendar_owner_telegram_id="synthetic-agent-eval-owner",
        gmail_enabled=True, gmail_owner_telegram_id="synthetic-agent-eval-owner",
        tavily_api_key="", default_timezone="Asia/Shanghai")
    monkeypatch.setattr(unified_loop, "get_settings", lambda: settings)
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    return settings


@pytest.fixture
def legacy_meeting_contract(monkeypatch):
    """Historical v1 runner contract only, not new-request tool disclosure.

    Full Saver compatibility is exercised separately in test_agent_p1_graph.
    Do not report this fixture as current end-to-end acceptance.
    """
    from app.agent import unified_loop, unified_tools
    original = unified_tools.available_tools
    monkeypatch.setattr(unified_loop, "available_tools", lambda *a, **kw:
                        original(*a, **{**kw, "include_legacy": True}))


def test_dataset_cases_have_stable_unique_ids_and_synthetic_origin():
    path = Path(__file__).resolve().parents[1] / "evals/datasets/unified_agent_acceptance_v1.json"
    dataset = json.loads(path.read_text(encoding="utf-8"))
    assert dataset["source"].startswith("synthetic")
    ids = [case["id"] for case in dataset["cases"]]
    assert len(ids) == len(set(ids))
    assert {"cross_read_three_domains", "read_then_approval_partial_summary",
            "meeting_prep_grounding"} <= set(ids)


@pytest.mark.asyncio
async def test_synthetic_tools_reject_wrong_owner_and_invalid_anchor():
    world = SyntheticWorld()
    wrong = AgentToolScope("another-user", "owner", "synthetic", "turn", "web", "thread")
    with agent_tool_scope(wrong):
        with pytest.raises(PermissionError, match="SyntheticOwnerMismatch"):
            await world.invoke("calendar_event", {"event_id": "event-synthetic-1"})
    correct = AgentToolScope(USER, "owner", "synthetic", "turn", "web", "thread")
    with agent_tool_scope(correct):
        exact = await world.invoke("calendar_event", {"event_id": "event-synthetic-1"})
        assert json.loads(exact)["id"] == "event-synthetic-1"
        await world.invoke("meeting_evidence", {"event_id": "wrong"})
        assert "MeetingAnchorMismatch" in world.errors


@pytest.mark.asyncio
async def test_synthetic_calendar_slot_is_checked_before_todo_postcondition():
    world = SyntheticWorld(busy=False)
    correct = AgentToolScope(USER, "owner", "synthetic", "turn", "web", "thread")
    with agent_tool_scope(correct):
        await world.invoke("calendar_availability", {
            "start": "2026-11-15T15:00:00+08:00",
            "end": "2026-11-15T16:00:00+08:00"})
        await world.invoke("create_todo", {"title": "合成复盘", "planned_start_str": "tomorrow"})
    assert "AvailabilityWindowMismatch" in world.errors
    assert "TodoScheduleMismatch" in world.errors
    assert not any(todo["title"] == "合成复盘" for todo in world.todos)


@pytest.mark.asyncio
async def test_evaluator_keeps_prior_read_at_approval_handoff(monkeypatch):
    settings = synthetic_settings(monkeypatch)
    model = SequenceModel(
        AIMessage(content="", tool_calls=[{
            "name": "calendar_events", "id": "read-1",
            "args": {"query": "", "start": "2026-11-15T00:00:00+08:00",
                     "end": "2026-11-15T23:59:59+08:00"}}]),
        AIMessage(content="", tool_calls=[{
            "name": "propose_todo_delete", "id": "approval-1",
            "args": {"target": "合成待办"}}]))
    result = await run_case({
        "id": "read_then_approval_partial_summary", "prompt": "先查日历再提议删除合成待办",
        "required_tools": ["calendar_events", "propose_todo_delete"],
        "expected_handoff": True, "expected_terms": ["合成项目组会议"]}, model, settings)
    assert result["checks"]["required_tools"]
    assert result["checks"]["handoff"]
    assert result["checks"]["expected_terms"]
    assert result["final_response"] == ""
    assert "合成项目组会议" in result["handoff_summary"]


@pytest.mark.asyncio
async def test_meeting_final_claims_fail_closed_without_submission(monkeypatch, legacy_meeting_contract):
    settings = synthetic_settings(monkeypatch)
    model = SequenceModel(
        AIMessage(content="", tool_calls=[{
            "name": "meeting_evidence", "id": "evidence-1",
            "args": {"event_id": "event-synthetic-1"}}]),
        AIMessage(content="合成项目组会议：参会人承诺未经证实。"))
    result = await run_case({
        "id": "meeting_prep_grounding", "prompt": "准备合成项目组会议",
        "required_tools": ["meeting_evidence"],
        "expected_terms": ["合成项目组会议"]}, model, settings)
    assert result["checks"]["required_tools"]
    assert not result["checks"]["expected_terms"]
    assert not result["checks"]["meeting_claim_validation"]
    assert "未通过来源校验" in result["final_response"]


@pytest.mark.asyncio
async def test_validated_meeting_claim_overrides_unchecked_final_text(monkeypatch, legacy_meeting_contract):
    settings = synthetic_settings(monkeypatch)
    model = SequenceModel(
        AIMessage(content="", tool_calls=[{
            "name": "meeting_evidence", "id": "evidence-1",
            "args": {"event_id": "event-synthetic-1"}}]),
        AIMessage(content="", tool_calls=[{
            "name": "submit_meeting_brief", "id": "submit-1",
            "args": {"event_id": "event-synthetic-1", "claims": [{
                "kind": "SOURCE_FACT", "text": "会议名称", "source_keys": ["C1"],
                "fact_field": "title", "fact_value": "合成项目组会议"}]}}]),
        AIMessage(content="大家已经同意转账"))
    result = await run_case({
        "id": "meeting_prep_grounding", "prompt": "准备合成项目组会议",
        "required_tools": ["meeting_evidence", "submit_meeting_brief"],
        "expected_terms": ["合成项目组会议"],
        "forbidden_terms": ["大家已经同意转账"]}, model, settings)
    assert result["checks"]["meeting_claim_validation"]
    assert result["checks"]["expected_terms"]
    assert result["checks"]["forbidden_terms"]
    assert result["meeting_submission_attempts"] == 1


@pytest.mark.asyncio
async def test_invalid_meeting_claims_stop_after_three_attempts(monkeypatch, legacy_meeting_contract):
    settings = synthetic_settings(monkeypatch)
    invalid = {"event_id": "event-synthetic-1", "claims": [{
        "kind": "SOURCE_FACT", "text": "编造标题", "source_keys": ["C1"],
        "fact_field": "title", "fact_value": "编造标题"}]}
    model = SequenceModel(
        AIMessage(content="", tool_calls=[{
            "name": "meeting_evidence", "id": "evidence-1",
            "args": {"event_id": "event-synthetic-1"}}]),
        *[AIMessage(content="", tool_calls=[{
            "name": "submit_meeting_brief", "id": f"invalid-{index}",
            "args": invalid}]) for index in range(3)])
    result = await run_case({
        "id": "meeting_prep_grounding", "prompt": "准备合成项目组会议",
        "required_tools": ["meeting_evidence"],
        "forbidden_terms": ["编造标题"]}, model, settings)
    assert result["meeting_submission_attempts"] == 3
    assert result["observed_meeting_validation_calls"] == 3
    assert not result["checks"]["meeting_claim_validation"]
    assert "Meeting claim rejected" in result["final_response"]
    assert "编造标题" not in result["final_response"]
