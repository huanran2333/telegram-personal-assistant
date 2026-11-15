from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
import pytest

from app.agent.transcript import _projection_rows, search_history
from app.agent.graph import run_agent
from app.telegram import webhook
from evals.r2_memory_benchmark import corpus, current_ids, summary_batch_ranges


def test_projection_excludes_tool_and_system_payloads():
    messages = [SystemMessage(content="SECRET SYSTEM"), HumanMessage(id="u1", content="synthetic question"),
                AIMessage(id="call", content="", tool_calls=[{"name":"x","args":{},"id":"tool-1"}]),
                ToolMessage(content="PRIVATE EXTERNAL BODY", tool_call_id="tool-1"),
                AIMessage(id="a1", content="safe answer")]
    rows = _projection_rows("user", "thread", messages)
    assert [r["message_id"] for r in rows] == ["u1", "a1"]
    assert all("PRIVATE" not in r["search_text"] and "SECRET" not in r["search_text"] for r in rows)
    assert all(r["source_ref"].startswith("checkpoint:thread:") for r in rows)


@pytest.mark.asyncio
async def test_history_search_requires_scope_and_bounded_k():
    for arguments in ({"user_id":"", "thread_id":"t", "query":"q"},
                      {"user_id":"u", "thread_id":"", "query":"q"},
                      {"user_id":"u", "thread_id":"t", "query":" ", "top_k":100}):
        with pytest.raises(ValueError):
            await search_history(None, **arguments)


def test_matched_corpus_includes_chinese_corrections_and_authority():
    turns, cases = corpus()
    assert len(turns) >= 160
    assert len([case for case in cases if case.gold]) >= 20
    assert {"PROFILE_STORE", "TODO_DB", "CALENDAR", "GMAIL"} <= {case.source_of_truth for case in cases}
    assert any("蓝鲸" in case.query for case in cases)
    assert len(current_ids(turns)) >= 3


def test_summary_trigger_counts_messages_not_characters():
    turns, _ = corpus()
    own = [turn for turn in turns if (turn.user_id, turn.thread_id) ==
           ("synthetic-a", "synthetic-main")]
    ranges = summary_batch_ranges(own)
    assert len(ranges) == 3
    assert ranges[0][0] == 0 and ranges[-1][1] == len(own)-8
    assert ranges[0][1] > 30


@pytest.mark.asyncio
async def test_projection_failure_never_leaks_content_or_retries_business(caplog):
    private = "synthetic-private-message-marker"
    class Graph:
        calls = 0
        async def aget_state(self, _config):
            return type("Snapshot", (), {"values": {}})()
        async def ainvoke(self, _state, config, *, context=None):
            assert context == {"user_id": "safe-user", "telegram_id": "safe-telegram",
                               "chat_id": "safe-chat", "origin": "telegram"}
            self.calls += 1
            return {"messages": [HumanMessage(id="u", content=private),
                                 AIMessage(id="a", content="completed") ]}
    def broken_projection_factory():
        raise RuntimeError("synthetic-private-database-payload")
    graph = Graph()
    with caplog.at_level("INFO"):
        response = await run_agent(graph, "safe-user", "safe-telegram", "safe-chat", private,
                                   projection_session_factory=broken_projection_factory)
    assert response == "completed"
    assert graph.calls == 1
    assert "RuntimeError" in caplog.text
    assert private not in caplog.text
    assert "synthetic-private-database-payload" not in caplog.text


@pytest.mark.asyncio
async def test_workflow_projection_failure_does_not_block_completed_response(monkeypatch, caplog):
    sent = []

    async def broken_graph():
        raise RuntimeError("synthetic-private-projection-failure")

    async def record_send(chat_id, message):
        sent.append((chat_id, message))

    monkeypatch.setattr(webhook, "get_productivity_graph", broken_graph)
    monkeypatch.setattr(webhook, "send_message", record_send)
    with caplog.at_level("WARNING"):
        await webhook._send_workflow_result(11, "synthetic-thread", {"response": "done"})
    assert sent == [(11, "done")]
    assert "synthetic-private-projection-failure" not in caplog.text
