"""Synthetic Calendar read task evaluation; never connects to Google or PostgreSQL.

The real structured router runs against the configured private model. A tiny
stateful fake Calendar supplies one synthetic event and records every read or
write attempt. This is Agent-behavior evidence, not Google E2E evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from app.agent.provider_clients import run_with_provider_clients
from app.agent.workflow import build_productivity_graph, start_workflow
from app.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "datasets" / "agent_quality_calendar_read_v1.json"
_EVENT_DAY = "2030-01-15"
_EVENT_TITLE = "Synthetic bluebird review"
_EVENT = {
    "id": "synthetic-bluebird-event", "summary": _EVENT_TITLE,
    "start": {"dateTime": "2030-01-15T09:00:00+08:00"},
    "end": {"dateTime": "2030-01-15T09:30:00+08:00"},
}


def load_dataset(path: Path = DATASET) -> tuple[dict, str]:
    raw = path.read_bytes()
    data = json.loads(raw)
    ids = set()
    if not data.get("dataset_version") or not isinstance(data.get("cases"), list):
        raise ValueError("Invalid Calendar evaluation dataset")
    for case in data["cases"]:
        if (not isinstance(case, dict) or case.get("id") in ids or
                case.get("split") not in {"dev", "holdout"} or
                case.get("expected_day") not in {_EVENT_DAY, "2030-01-16"} or
                not isinstance(case.get("expected_event"), bool) or
                not isinstance(case.get("owner_allowed"), bool) or
                not isinstance(case.get("prompt"), str) or
                "calendar" not in case["prompt"].lower() or
                "2030" not in case["prompt"]):
            raise ValueError("Only fixed synthetic Calendar cases are permitted")
        if case["expected_event"] and case["expected_day"] != _EVENT_DAY:
            raise ValueError("Synthetic event date mismatch")
        ids.add(case["id"])
    return data, hashlib.sha256(raw).hexdigest()


class SyntheticCalendar:
    def __init__(self):
        self.searches: list[dict] = []
        self.availability_reads = 0
        self.write_attempts = 0

    async def search_events(self, query: str, start: str | None = None,
                            end: str | None = None) -> list[dict]:
        self.searches.append({"query": query, "start": start, "end": end})
        if not start or not end:
            return []
        instant = datetime.fromisoformat(_EVENT["start"]["dateTime"])
        return [_EVENT] if datetime.fromisoformat(start) <= instant < datetime.fromisoformat(end) else []

    async def availability(self, start: str, end: str, timezone: str) -> dict:
        self.availability_reads += 1
        return {"calendars": {"primary": {"busy": []}}}

    async def create_event(self, *args, **kwargs):
        self.write_attempts += 1
        raise AssertionError("Calendar evaluation forbids writes")

    async def update_event(self, *args, **kwargs):
        self.write_attempts += 1
        raise AssertionError("Calendar evaluation forbids writes")

    async def delete_event(self, *args, **kwargs):
        self.write_attempts += 1
        raise AssertionError("Calendar evaluation forbids writes")


class NoBusinessExecution:
    @staticmethod
    def session_factory():
        raise AssertionError("Calendar read evaluation forbids database access")

    async def execute(self, *args, **kwargs):
        raise AssertionError("Calendar read evaluation forbids action execution")


def grade_case(case: dict, result: dict, calendar: SyntheticCalendar) -> dict:
    analysis = result.get("analysis") or {}
    steps = analysis.get("actions") or []
    event_step = [step for step in steps if step.get("operation") == "search_events"]
    range_correct = False
    if len(event_step) == 1:
        args = event_step[0].get("args") or {}
        try:
            start = datetime.fromisoformat(args["start"])
            end = datetime.fromisoformat(args["end"])
            range_correct = (start.date().isoformat() == case["expected_day"] and
                             end.date().toordinal() - start.date().toordinal() == 1 and
                             start.utcoffset().total_seconds() == 8 * 3600)
        except (KeyError, TypeError, ValueError, AttributeError):
            pass
    reads = result.get("read_results") or {}
    events = reads.get("events") or []
    owner_allowed = case["owner_allowed"]
    expected_ids = [_EVENT["id"]] if owner_allowed and case["expected_event"] else []
    correct_events = [item.get("id") for item in events] == expected_ids
    response = result.get("response") or ""
    response_ok = (_EVENT_TITLE in response and "09:00" in response
                   if expected_ids else _EVENT_TITLE not in response)
    authorization_ok = (len(calendar.searches) == 1 if owner_allowed else
                        len(calendar.searches) == 0 and bool(reads.get("error")))
    safe = ("__interrupt__" not in result and not result.get("proposal") and
            calendar.write_attempts == 0 and calendar.availability_reads == 0)
    passed = (analysis.get("mode") == "calendar_read" and len(steps) == 1 and
              range_correct and correct_events and response_ok and authorization_ok and safe)
    return {"passed": passed, "route_and_plan": analysis.get("mode") == "calendar_read" and range_correct,
            "environment": correct_events and authorization_ok,
            "communication": response_ok, "no_side_effect": safe,
            "owner_blocked": not owner_allowed and authorization_ok}


async def run_case(case: dict, repeat: int) -> dict:
    started = time.perf_counter()
    calendar = SyntheticCalendar()
    graph = build_productivity_graph(
        calendar, NoBusinessExecution(), InMemorySaver(),
        calendar_owner_telegram_id="synthetic-calendar-owner")
    try:
        result = await start_workflow(
            graph, "synthetic-calendar-user", case["prompt"], "Asia/Shanghai",
            f"calendar-quality-{uuid4().hex}",
            telegram_id="synthetic-calendar-owner" if case["owner_allowed"] else "synthetic-other-user")
        score = grade_case(case, result, calendar)
        return {"case_id": case["id"], "repeat": repeat, **score,
                "read_calls": len(calendar.searches), "write_attempts": calendar.write_attempts,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "error_category": None}
    except Exception as exc:
        return {"case_id": case["id"], "repeat": repeat,
                "passed": False, "read_calls": len(calendar.searches),
                "write_attempts": calendar.write_attempts,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "error_category": type(exc).__name__}


async def run(*, split: str = "all", repeats: int = 1) -> dict:
    if split not in {"all", "dev", "holdout"} or not 1 <= repeats <= 5:
        raise ValueError("Invalid Calendar evaluation split/repeats")
    data, digest = load_dataset()
    cases = [case for case in data["cases"] if split == "all" or case["split"] == split]
    if not cases:
        raise ValueError("No Calendar evaluation cases")
    settings = get_settings()
    rows = []
    for repeat in range(1, repeats + 1):
        for case in cases:
            rows.append(await run_case(case, repeat))
    return {"manifest": {"dataset_version": data["dataset_version"],
                         "dataset_sha256": digest, "split": split, "repeats": repeats,
                         "model_provider": settings.llm_provider, "model_id": settings.llm_model,
                         "evidence": data["evidence"],
                         "token_usage": "NOT_MEASURED"},
            "summary": {"attempts": len(rows), "passed": sum(row["passed"] for row in rows),
                        "errors": sum(bool(row["error_category"]) for row in rows)},
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("all", "dev", "holdout"), default="all")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = asyncio.run(run_with_provider_clients(run(split=args.split, repeats=args.repeats)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    if report["summary"]["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
