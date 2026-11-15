"""Read-only R5.1 pinned Calendar MCP schema probe; prints no credentials."""
import asyncio
import json

from app.agent.calendar import discover_calendar_tools
from app.config import get_settings


async def main():
    tools = {tool.name: tool for tool in await discover_calendar_tools(get_settings())}
    selected = {}
    for name in ("create-event", "get-event", "update-event", "delete-event"):
        selected[name] = sorted(tools[name].args)
    print(json.dumps({"package": "@cocal/google-calendar-mcp@2.7.0",
                      "tool_schema_fields": selected}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
