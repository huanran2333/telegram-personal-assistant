"""Single model-facing Todo tools over plain business operations."""
from __future__ import annotations


from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from app.agent.state import AgentRuntimeContext, require_agent_runtime_identity
from app.agent.tools import make_todo_operations
from app.localization import tr


def make_official_todo_tools(db_session_factory) -> list:
    operations = make_todo_operations(db_session_factory)
    writes = {"create_todo", "update_todo", "set_reminder", "cancel_reminder"}

    def trusted_request(runtime: ToolRuntime[AgentRuntimeContext]) -> tuple[str, str]:
        user_id = require_agent_runtime_identity(runtime.state, runtime)["user_id"]
        state = runtime.state
        request = state.get("active_request")
        if not isinstance(request, str) or not request:
            raise PermissionError("AgentActiveRequestMissing")
        return user_id, request

    async def invoke(name: str, runtime: ToolRuntime[AgentRuntimeContext], **args):
        user_id, _ = trusted_request(runtime)
        if name in writes:
            args["tool_call_id"] = runtime.tool_call_id
        result = await operations[name](**args, user_id=user_id)
        if name in writes:
            failures = {tr("todo_not_found", id=args.get("todo_id", "")),
                        tr("todo_many"), tr("todo_past_reminder"),
                        tr("todo_completed_reminder"),
                        tr("todo_version_conflict"),
                        tr("todo_update_error", id=args.get("todo_id", ""))}
            for value in (args.get("due_date_str"), args.get("planned_start_str"),
                          args.get("planned_end_str")):
                if value:
                    failures.add(tr("todo_bad_due", value=value))
            for value in (args.get("reminder_at_str"), args.get("remind_at_str")):
                if value:
                    failures.add(tr("todo_bad_reminder", value=value))
            if result in failures:
                return ToolMessage(content=result, tool_call_id=runtime.tool_call_id,
                                   status="error")
        versions = getattr(result, "todo_versions", None)
        if isinstance(versions, dict):
            return ToolMessage(
                content=str(result), tool_call_id=runtime.tool_call_id,
                artifact={"todo_versions": versions})
        return result

    @tool
    async def create_todo(
        title: str, runtime: ToolRuntime[AgentRuntimeContext],
        description: str = "", due_date_str: str = "", planned_start_str: str = "",
        planned_end_str: str = "", reminder_at_str: str = "",
        priority: str = "medium", tags: str = "", time_source_quote: str = "",
    ) -> str | ToolMessage:
        """Create one Todo. For a user-specified time, quote this Todo's exact user time words in time_source_quote. For a requested free-slot schedule, call calendar_candidate_slots in an earlier model turn, wait for its result, then use one exact returned start/end and quote the user's request to schedule this title. Never guess a free slot or borrow another goal's time."""
        return await invoke("create_todo", runtime, title=title, description=description,
                            due_date_str=due_date_str, planned_start_str=planned_start_str,
                            planned_end_str=planned_end_str, reminder_at_str=reminder_at_str,
                            priority=priority, tags=tags)

    @tool
    async def list_todos(runtime: ToolRuntime[AgentRuntimeContext], status: str = "",
                         priority: str = "", tags: str = "", period: str = "",
                         due_date_str: str = "") -> str:
        """List this user's Todos, optionally filtered by status, priority, tags, or period."""
        return await invoke("list_todos", runtime, status=status, priority=priority,
                            tags=tags, period=period, due_date_str=due_date_str)

    @tool
    async def update_todo(
        todo_id: str, expected_version: int,
        runtime: ToolRuntime[AgentRuntimeContext], title: str = "",
        status: str = "", priority: str = "", due_date_str: str = "",
        planned_start_str: str = "", planned_end_str: str = "", tags: str = "",
        time_source_quote: str = "",
    ) -> str | ToolMessage:
        """Update an owned Todo. First read it with list_todos or search_todos, then pass its displayed version as expected_version. A stale version is rejected. Cite this Todo's user time words for time changes."""
        return await invoke("update_todo", runtime, todo_id=todo_id, title=title,
                            status=status, priority=priority, due_date_str=due_date_str,
                            planned_start_str=planned_start_str,
                            planned_end_str=planned_end_str, tags=tags,
                            expected_version=expected_version)

    @tool
    async def search_todos(query: str, runtime: ToolRuntime[AgentRuntimeContext]) -> str:
        """Search this user's Todos by query."""
        return await invoke("search_todos", runtime, query=query)

    @tool
    async def set_reminder(todo_id: str, remind_at_str: str, expected_version: int,
                           runtime: ToolRuntime[AgentRuntimeContext],
                           time_source_quote: str = "") -> str | ToolMessage:
        """Set a Todo reminder using an authorized time. First read the Todo or use the version returned by its creation; pass that version as expected_version. A stale version is rejected."""
        return await invoke("set_reminder", runtime, todo_id=todo_id,
                            remind_at_str=remind_at_str,
                            expected_version=expected_version)

    @tool
    async def cancel_reminder(todo_id: str, expected_version: int,
                              runtime: ToolRuntime[AgentRuntimeContext]) -> str | ToolMessage:
        """Cancel one owned Todo's reminder. First read it and pass its current version as expected_version; a stale version is rejected."""
        return await invoke("cancel_reminder", runtime, todo_id=todo_id,
                            expected_version=expected_version)

    return [create_todo, list_todos, update_todo, search_todos,
            set_reminder, cancel_reminder]
