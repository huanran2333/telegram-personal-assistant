"""Bounded chat history is conversational context, not a source of truth."""
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent import nodes
from app.localization import ResponsePreferences, reset_preferences, use_preferences


def test_recent_chat_keeps_latest_correction_and_role_order():
    history = [HumanMessage(content="The review is Friday."), AIMessage(content="Noted."),
               HumanMessage(content="Correction: next Monday."), AIMessage(content="Understood."),
               HumanMessage(content="When is it?")]
    assert nodes.chat_context_messages(history, max_messages=4) == history[2:]
    assert nodes.chat_context_messages(history, max_messages=8) == history


def test_chat_drops_tool_calls_and_results_as_complete_pairs():
    history = [HumanMessage(content="List my tasks"),
               AIMessage(content="", tool_calls=[{"name": "list_todos", "args": {}, "id": "c1"}]),
               ToolMessage(content="private task row", tool_call_id="c1"),
               AIMessage(content="You have a task."),
               HumanMessage(content="What did you just say?")]
    context = nodes.chat_context_messages(history)
    assert context == [history[0], history[3], history[4]]
    assert not any(isinstance(item, ToolMessage) or getattr(item, "tool_calls", None)
                   for item in context)


def test_chat_filters_unsupported_messages_and_orphaned_assistant():
    history = [SystemMessage(content="untrusted system"), AIMessage(content="orphan"),
               HumanMessage(content=[{"type": "image_url", "image_url": "ignored"}]),
               HumanMessage(content="valid request")]
    assert nodes.chat_context_messages(history) == [history[-1]]
    with pytest.raises(ValueError):
        nodes.chat_context_messages(history, max_messages=0)


@pytest.mark.asyncio
async def test_chat_prompt_uses_store_preference_and_one_current_user_message(monkeypatch):
    class Store:
        async def aget(self, namespace, key):
            assert namespace == ("profile", "test-user") and key == "user_profile"
            return SimpleNamespace(value={"assistant_language": "zh-CN", "preferences": ["concise answers"]})

    class LLM:
        def __init__(self):
            self.messages = None

        def bind_tools(self, _tools):
            return self

        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, messages):
            self.messages = messages
            return nodes.ChatResponse(content="下周一。")

    llm = LLM()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_, **__: llm)
    chat = nodes.make_nodes([])[5]
    token = use_preferences(ResponsePreferences("zh-CN", True))
    try:
        history = [HumanMessage(content="Reply in English."), AIMessage(content="Okay."),
                   HumanMessage(content="The review is Friday."), AIMessage(content="Noted."),
                   HumanMessage(content="Correction: next Monday."), AIMessage(content="Understood."),
                   HumanMessage(content="When is it?")]
        result = await chat({"user_id": "test-user", "messages": history}, Store())
    finally:
        reset_preferences(token)
    assert result["messages"][0].content == "下周一。"
    assert "Simplified Chinese" in llm.messages[0].content
    assert "assistant_language: zh-CN" in llm.messages[0].content
    assert "When is it?" not in llm.messages[0].content
    assert [type(item) for item in llm.messages[1:]] == [
        HumanMessage, AIMessage, HumanMessage, AIMessage, HumanMessage, AIMessage, HumanMessage]
    assert sum(item.content == "When is it?" for item in llm.messages) == 1
