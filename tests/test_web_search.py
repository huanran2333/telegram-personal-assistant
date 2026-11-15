import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import nodes


class SearchLLM:
    def __init__(self):
        self.messages = None

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        assert schema is nodes.ProfileAwareSearchQuery
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        if messages[0].content.startswith("Rewrite a web search query"):
            return nodes.ProfileAwareSearchQuery(query="sushi restaurants Shanghai")
        return AIMessage(content="根据来源，结果是 X：https://example.org/result")


def web_node(monkeypatch, llm):
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_, **__: llm)
    return nodes.make_nodes([])[4]


@pytest.mark.asyncio
async def test_web_search_uses_tavily_results_for_llm_answer(monkeypatch):
    llm = SearchLLM()

    class Tavily:
        def __init__(self, api_key):
            pass

        def search(self, **kwargs):
            assert kwargs["query"] == "最近的消息"
            return {"results": [{"title": "Result", "url": "https://example.org/result",
                                 "content": "Relevant source excerpt"}]}

    monkeypatch.setattr(nodes, "TavilyClient", Tavily)
    result = await web_node(monkeypatch, llm)({
        "messages": [HumanMessage(content="最近的消息")], "entities": {"query": "最近的消息"},
    })
    assert "https://example.org/result" in result["messages"][0].content
    assert "Relevant source excerpt" in llm.messages[1].content


@pytest.mark.asyncio
async def test_web_search_empty_results_does_not_make_up_answer(monkeypatch):
    llm = SearchLLM()

    class Tavily:
        def __init__(self, api_key):
            pass

        def search(self, **kwargs):
            return {"results": []}

    monkeypatch.setattr(nodes, "TavilyClient", Tavily)
    result = await web_node(monkeypatch, llm)({"messages": [HumanMessage(content="rare query")], "entities": {}})
    assert "couldn't find" in result["messages"][0].content
    assert llm.messages is None


@pytest.mark.asyncio
async def test_profile_aware_search_uses_relevant_preference(monkeypatch):
    llm = SearchLLM()
    captured = {}

    class Store:
        async def aget(self, namespace, key):
            assert namespace == ("profile", "user-a")
            return type("Item", (), {"value": {"interests": ["sushi"]}})()

    class Tavily:
        def __init__(self, api_key):
            pass

        def search(self, **kwargs):
            captured["query"] = kwargs["query"]
            return {"results": [{"title": "Sushi", "url": "https://example.org/result",
                                 "content": "Sushi place"}]}

    monkeypatch.setattr(nodes, "TavilyClient", Tavily)
    result = await web_node(monkeypatch, llm)(
        {"user_id": "user-a", "messages": [HumanMessage(content="上海有哪些餐厅推荐？")],
         "entities": {"query": "Shanghai restaurants"}}, Store(),
    )
    assert captured["query"] == "sushi restaurants Shanghai"
    assert "https://example.org/result" in result["messages"][0].content
