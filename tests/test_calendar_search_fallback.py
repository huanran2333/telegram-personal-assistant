"""A provider text-search miss must not hide an event visible in the same date window."""
import pytest

from app.agent.calendar import MCPCalendarAdapter


class FakeTool:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        return {"events": self.events}


START = "2026-10-20T00:00:00+08:00"
END = "2026-10-21T00:00:00+08:00"


@pytest.mark.asyncio
async def test_exact_title_can_be_found_when_provider_text_search_misses():
    search = FakeTool("search-events", [])
    listing = FakeTool("list-events", [
        {"id": "e1", "summary": "在线验收-日历-20261019", "status": "confirmed"},
        {"id": "e2", "summary": "unrelated", "status": "confirmed"},
    ])
    events = await MCPCalendarAdapter([search, listing]).search_events(
        "在线验收-日历-20261019", START, END)
    assert [item["id"] for item in events] == ["e1"]
    assert listing.calls[0]["timeMin"] == START
    assert listing.calls[0]["timeMax"] == END


@pytest.mark.asyncio
async def test_provider_search_hit_does_not_trigger_extra_list_read():
    search = FakeTool("search-events", [{"id": "e1", "summary": "Found"}])
    listing = FakeTool("list-events", [])
    assert (await MCPCalendarAdapter([search, listing]).search_events("Found", START, END))[0]["id"] == "e1"
    assert listing.calls == []


@pytest.mark.asyncio
async def test_unbounded_search_miss_does_not_scan_calendar_history():
    search = FakeTool("search-events", [])
    listing = FakeTool("list-events", [{"id": "e1", "summary": "Found"}])
    assert await MCPCalendarAdapter([search, listing]).search_events("Found") == []
    assert listing.calls == []
