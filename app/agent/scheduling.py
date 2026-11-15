"""Bounded, deterministic search for free Calendar slots.

The model extracts constraints; this module, not the model, decides whether a
slot is actually free. All intervals are timezone-aware and Google freebusy is
treated as an authoritative read that may fail closed.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, model_validator

from app.agent.calendar import CalendarError


class SchedulingConstraints(BaseModel):
    title: str = Field(min_length=1)
    duration_minutes: int = Field(ge=15, le=480)
    range_start: datetime
    range_end: datetime
    timezone: str
    preferred_day_parts: list[str] = Field(default_factory=list)
    preferred_weekdays: list[int] = Field(default_factory=list)
    excluded_weekdays: list[int] = Field(default_factory=list)
    earliest_hour: int = Field(default=9, ge=0, le=23)
    latest_hour: int = Field(default=21, ge=1, le=24)

    @model_validator(mode="after")
    def valid_window(self):
        ZoneInfo(self.timezone)
        if self.range_start.tzinfo is None or self.range_end.tzinfo is None:
            raise ValueError("Scheduling range must be timezone-aware")
        if any(not 0 <= weekday <= 6 for weekday in
               self.preferred_weekdays + self.excluded_weekdays):
            raise ValueError("Weekdays must be Monday=0 through Sunday=6")
        return self


def verified_free_interval(start: datetime, end: datetime, freebusy: dict[str, Any]) -> bool:
    """Return false for overlap; raise if Google did not provide valid availability."""
    primary = freebusy.get("calendars", {}).get("primary") if isinstance(freebusy, dict) else None
    if not isinstance(primary, dict) or primary.get("errors") or not isinstance(primary.get("busy"), list):
        raise CalendarError("Calendar availability was incomplete; no free slot can be inferred.")
    try:
        for item in primary["busy"]:
            left, right = datetime.fromisoformat(item["start"]), datetime.fromisoformat(item["end"])
            if not left.tzinfo or not right.tzinfo or right <= left:
                raise ValueError("invalid interval")
            if start < right and end > left:
                return False
    except (KeyError, TypeError, ValueError) as exc:
        raise CalendarError("Calendar availability contained an invalid busy interval.") from exc
    return True


def find_candidate_slots(constraints: SchedulingConstraints, freebusy: dict[str, Any],
                         now: datetime | None = None, limit: int = 3) -> list[dict[str, str]]:
    zone = ZoneInfo(constraints.timezone)
    start = constraints.range_start.astimezone(zone)
    end = constraints.range_end.astimezone(zone)
    current = (now or datetime.now(zone)).astimezone(zone)
    if end <= start or end - start > timedelta(days=15) or constraints.latest_hour <= constraints.earliest_hour:
        raise ValueError("Scheduling date range or working hours are invalid.")
    primary = freebusy.get("calendars", {}).get("primary") if isinstance(freebusy, dict) else None
    if not isinstance(primary, dict) or primary.get("errors") or not isinstance(primary.get("busy"), list):
        raise CalendarError("Calendar availability was incomplete; no free slot can be inferred.")
    busy = []
    try:
        for interval in primary["busy"]:
            left = datetime.fromisoformat(interval["start"])
            right = datetime.fromisoformat(interval["end"])
            if not left.tzinfo or not right.tzinfo or right <= left:
                raise ValueError("reversed interval")
            busy.append((left.astimezone(zone), right.astimezone(zone)))
    except (KeyError, TypeError, ValueError) as exc:
        raise CalendarError("Calendar availability contained an invalid busy interval.") from exc

    duration = timedelta(minutes=constraints.duration_minutes)
    candidates: list[tuple[int, datetime, datetime]] = []
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < end:
        if day.weekday() not in constraints.excluded_weekdays:
            lower = max(day + timedelta(hours=constraints.earliest_hour), start, current + timedelta(minutes=5))
            upper = min(day + timedelta(hours=constraints.latest_hour), end)
            # Half-hour grid is a transparent product rule, not an LLM guess.
            minute_of_day = lower.hour * 60 + lower.minute
            grid_minute = ((minute_of_day + 29) // 30) * 30
            cursor = day + timedelta(minutes=grid_minute)
            while cursor + duration <= upper:
                slot_end = cursor + duration
                if all(slot_end <= left or cursor >= right for left, right in busy):
                    # Noon is a separate lunch boundary, not an afternoon slot.
                    # A preferred afternoon begins at 13:00 local time.
                    part = ("morning" if cursor.hour < 12 else
                            "midday" if cursor.hour < 13 else
                            "afternoon" if cursor.hour < 18 else "evening")
                    score = (2 if part in constraints.preferred_day_parts else 0) + (
                        1 if cursor.weekday() in constraints.preferred_weekdays else 0)
                    candidates.append((score, cursor, slot_end))
                cursor += timedelta(minutes=30)
        day += timedelta(days=1)

    candidates.sort(key=lambda item: (-item[0], item[1]))
    chosen: list[tuple[datetime, datetime]] = []
    # Prefer distinct days, then fill with non-overlapping slots if needed.
    for distinct_day in (True, False):
        for _, slot_start, slot_end in candidates:
            if len(chosen) >= limit:
                break
            if distinct_day and any(previous.date() == slot_start.date() for previous, _ in chosen):
                continue
            if any(slot_start < previous_end and slot_end > previous for previous, previous_end in chosen):
                continue
            chosen.append((slot_start, slot_end))
    return [{"start": left.isoformat(), "end": right.isoformat(), "timezone": constraints.timezone}
            for left, right in chosen]
