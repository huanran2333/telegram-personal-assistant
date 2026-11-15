import pytest
from langchain_core.messages import HumanMessage

from app.agent import nodes
from app.agent.nodes import _extract_profile_update, _merge_profile


def test_profile_preference_parsing():
    update = _extract_profile_update("I prefer concise answers.")
    assert update == {"preferences": ["concise answers"]}
    assert _merge_profile({"preferences": ["verbose"]}, update)["preferences"] == ["verbose", "concise answers"]


def test_chinese_response_preference_is_not_an_interest():
    assert _extract_profile_update("我喜欢简洁的回答。") == {"preferences": ["简洁的回答"]}
    assert _extract_profile_update("以后请尽量简短回答。") == {"preferences": ["简短回答"]}


@pytest.mark.asyncio
async def test_later_chat_loads_profile_into_real_response_prompt(monkeypatch):
    class Store:
        async def aget(self, namespace, key):
            assert namespace == ("profile", "test-user")
            assert key == "user_profile"
            return type("Item", (), {"value": {"preferences": ["简洁的回答"]}})()

    class LLM:
        def __init__(self):
            self.prompt = ""

        def bind_tools(self, tools):
            return self

        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, messages):
            self.prompt = messages[0].content
            return nodes.ChatResponse(content="好的。")

    llm = LLM()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_, **__: llm)
    chat_response = nodes.make_nodes([])[5]
    result = await chat_response({"user_id": "test-user", "messages": [HumanMessage(content="解释一下")]} , Store())
    assert "preferences: ['简洁的回答']" in llm.prompt
    assert result["messages"][0].content == "好的。"
