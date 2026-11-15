from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.nodes import todo_context_messages


def test_long_tool_turn_retains_user_query_and_complete_tool_pairs():
    history = [HumanMessage(content="Create a reminder for my paper")]
    for index in range(6):
        call_id = f"call-{index}"
        history.append(AIMessage(content="", tool_calls=[{
            "name": "list_todos", "args": {}, "id": call_id,
        }]))
        history.append(ToolMessage(content="[]", tool_call_id=call_id))

    context = todo_context_messages(history)
    assert context == history
    assert isinstance(context[0], HumanMessage)
    assert len(context) == 13


def test_short_tool_turn_keeps_existing_recent_window():
    history = [HumanMessage(content="older"), AIMessage(content="okay"),
               HumanMessage(content="new request"), AIMessage(content="working")]
    assert todo_context_messages(history) == history
