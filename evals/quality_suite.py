"""Versioned, synthetic, real-model route/plan evaluation.

This is intentionally a *route/plan slice*, not a full task-success benchmark.
No database, Telegram, Calendar, or Gmail provider is opened by this runner.
The dataset hash, split, repeats, model and raw per-case denominators travel
with each report so a regression cannot be hidden by a changed corpus.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from app.agent.planner import (RouteModelError, RoutedRequest, analyze_route,
                               normalize_weekday_phrases, validate_email_thread_id_source,
                               preserve_email_subject_identifier, validate_email_action_intent,
                               preserve_explicit_calendar_date, validate_route_clarity,
                               route_request)
from app.agent.provider_clients import run_with_provider_clients
from app.config import get_settings
from evals.v23_runner import RecordingStructuredLLM

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "datasets" / "agent_quality_v3.json"
REQUIRED = {"id", "split", "category", "prompt", "expected_intent", "expected_typed_plan"}
SPLITS = {"dev", "holdout"}


class ObservedRouteLLM(RecordingStructuredLLM):
    """Record only non-sensitive route shape before production validation."""

    def __init__(self, base):
        super().__init__(base)
        self.route_shapes: list[dict[str, Any]] = []
        self.raw_routes: list[RoutedRequest] = []

    def with_structured_output(self, schema):
        bound = super().with_structured_output(schema)
        owner = self

        class ObservedBound:
            async def ainvoke(self, messages):
                parsed = await bound.ainvoke(messages)
                if schema is RoutedRequest:
                    value = RoutedRequest.model_validate(parsed)
                    owner.raw_routes.append(value)
                    owner.route_shapes.append({
                        "intent": value.intent,
                        "todo_operation": value.todo_operation,
                        "has_calendar_date": bool(value.date_reference or value.calendar_date_text),
                        "has_todo_title": bool(value.todo_title or value.title),
                        "has_calendar_read": value.read_kind in {"events", "availability"},
                        "confidence": value.confidence,
                    })
                return parsed

        return ObservedBound()


def validation_failure_stage(route: RoutedRequest, prompt: str) -> str | None:
    """Replay only the pure production validators for a synthetic failed case."""
    validators = (
        normalize_weekday_phrases, validate_email_thread_id_source,
        preserve_email_subject_identifier, validate_email_action_intent,
        preserve_explicit_calendar_date, validate_route_clarity,
    )
    value = route
    for validator in validators:
        try:
            value = validator(value, prompt)
        except ValueError:
            return validator.__name__
    return None


def load_dataset(path: Path = DATASET) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    corpus = json.loads(raw)
    if not isinstance(corpus, dict) or not isinstance(corpus.get("cases"), list):
        raise ValueError("Invalid quality dataset")
    if not corpus.get("dataset_version") or len(corpus["cases"]) < 20:
        raise ValueError("Unversioned or insufficient quality dataset")
    ids: set[str] = set()
    for case in corpus["cases"]:
        if not isinstance(case, dict) or not REQUIRED.issubset(case):
            raise ValueError("Quality case has missing fields")
        if not isinstance(case["id"], str) or case["id"] in ids:
            raise ValueError("Quality case ID is invalid or duplicated")
        ids.add(case["id"])
        if case["split"] not in SPLITS or not isinstance(case["prompt"], str) or not case["prompt"].strip():
            raise ValueError("Quality case split/prompt is invalid")
        if case["expected_intent"] not in {
            "clarify", "clarify_or_safe_refusal", "chat", "todo", "calendar_read",
            "calendar_create", "calendar_update", "calendar_delete", "email_search",
            "email_unread", "meeting_prep", "smart_schedule", "cross_domain",
        }:
            raise ValueError("Quality case intent is unsupported")
        if not isinstance(case["expected_typed_plan"], bool):
            raise ValueError("Quality case expected_typed_plan must be boolean")
        if "expected_plan_error" in case and not isinstance(case["expected_plan_error"], bool):
            raise ValueError("Quality case expected_plan_error must be boolean")
    return corpus, hashlib.sha256(raw).hexdigest()


def grade_route(case: dict[str, Any], *, route: Any = None,
                route_error: BaseException | None = None,
                plan_error: BaseException | None = None,
                plan: Any = None) -> dict[str, Any]:
    """Judge route, argument and plan independently; error != clarify on provider failure."""
    expected = case["expected_intent"]
    if route_error is not None:
        actual = "provider_error" if isinstance(route_error, RouteModelError) else "clarify" if isinstance(route_error, ValueError) else "runtime_error"
    else:
        actual = getattr(route, "intent", None) or "runtime_error"
    if expected == "clarify_or_safe_refusal":
        route_ok = actual in {"clarify", "chat"}
    else:
        route_ok = actual == expected
    operation_expected = case.get("expected_todo_operation")
    operation_actual = getattr(route, "todo_operation", None) if route is not None else None
    operation_ok = (operation_actual == operation_expected) if operation_expected else None
    expected_plan = case["expected_typed_plan"]
    plan_actual = bool(getattr(plan, "requires_plan", False)) if plan is not None else None
    if route_ok and expected not in {"clarify", "clarify_or_safe_refusal"}:
        plan_ok = (isinstance(plan_error, ValueError) if case.get("expected_plan_error")
                   else plan_error is None and plan_actual == expected_plan)
    else:
        plan_ok = None
    return {
        "actual_intent": actual,
        "route_ok": route_ok,
        "operation_expected": operation_expected,
        "operation_actual": operation_actual,
        "operation_ok": operation_ok,
        "plan_expected": expected_plan,
        "plan_actual": plan_actual,
        "plan_ok": plan_ok,
        "expected_safe_block": bool(case.get("expected_plan_error")),
        "failure_category": type(route_error or plan_error).__name__ if route_error or plan_error else None,
        "failure_stage": "route" if route_error else "plan" if plan_error else None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def metric(field: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
        values = [row[field] for row in selected if isinstance(row.get(field), bool)]
        passed = sum(values)
        return {"passed": passed, "measured": len(values),
                "rate": passed / len(values) if values else None}

    eligible = [row for row in rows if row.get("actual_intent") != "provider_error"]
    by_category = {}
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        by_category[category] = metric("route_ok", [row for row in selected if row.get("actual_intent") != "provider_error"])
    usage = [row["input_tokens"] + row["output_tokens"] for row in rows
             if isinstance(row.get("input_tokens"), int) and isinstance(row.get("output_tokens"), int)]
    return {
        "attempts": len(rows),
        "status": "INFRA_FAILURE" if rows and not eligible else "PARTIAL_INFRA_FAILURE" if len(eligible) != len(rows) else "COMPLETE",
        "provider_error_attempts": len(rows) - len(eligible),
        "route": metric("route_ok", eligible),
        "todo_operation": metric("operation_ok", eligible),
        "typed_plan": metric("plan_ok", eligible),
        "by_category": by_category,
        "failures": dict(Counter(row["failure_category"] for row in rows
                                 if row.get("failure_category") and not
                                 (row.get("expected_safe_block") and row.get("plan_ok")) and not
                                 (row.get("expected_intent") == "clarify" and row.get("route_ok")))),
        "expected_safe_blocks": sum(bool(row.get("expected_safe_block") and row.get("plan_ok")) for row in rows),
        "model_calls": sum(row.get("model_calls") or 0 for row in rows),
        "token_usage": {"measured_attempts": len(usage), "total": sum(usage) if len(usage) == len(rows) else None,
                        "measured_subtotal": sum(usage) if usage else None},
        "latency_p50_ms": statistics.median(row["latency_ms"] for row in rows) if rows else None,
        "task_success": "NOT_MEASURED",
        "calendar_external_effect": "NOT_RUN",
        "gmail_external_effect": "NOT_RUN",
    }


async def run(*, split: str = "all", repeats: int = 1, path: Path = DATASET,
              case_id: str | None = None) -> dict[str, Any]:
    if split not in SPLITS | {"all"} or repeats < 1 or repeats > 10:
        raise ValueError("Invalid split or repeats")
    corpus, digest = load_dataset(path)
    settings = get_settings()
    from app.agent.nodes import _build_chat_llm
    model = ObservedRouteLLM(_build_chat_llm(settings.llm_provider, settings.llm_model, routing=True))
    now = datetime.fromisoformat(corpus["fixed_now"])
    if now.tzinfo is None:
        raise ValueError("Dataset fixed_now must be timezone-aware")
    cases = [case for case in corpus["cases"] if split == "all" or case["split"] == split]
    if case_id is not None:
        cases = [case for case in cases if case["id"] == case_id]
        if not cases:
            raise ValueError("Unknown case ID in selected split")
    rows = []
    for repeat in range(1, repeats + 1):
        for case in cases:
            started = time.perf_counter()
            before = len(model.calls)
            before_routes = len(model.route_shapes)
            before_raw_routes = len(model.raw_routes)
            route = plan = route_error = plan_error = None
            try:
                route = await route_request(case["prompt"], corpus["timezone"], llm=model, now=now)
            except Exception as exc:
                route_error = exc
            if route is not None and case["expected_intent"] not in {"clarify", "clarify_or_safe_refusal"}:
                try:
                    plan = analyze_route(route, corpus["timezone"], now=now, request=case["prompt"])
                except Exception as exc:
                    plan_error = exc
            grade = grade_route(case, route=route, route_error=route_error,
                                plan_error=plan_error, plan=plan)
            calls = model.calls[before:]
            row = {"case_id": case["id"], "split": case["split"], "category": case["category"],
                   "expected_intent": case["expected_intent"],
                   "repeat": repeat, **grade,
                   "pre_validation_route_shapes": model.route_shapes[before_routes:],
                   "validation_failure_stage": (
                       validation_failure_stage(model.raw_routes[before_raw_routes], case["prompt"])
                       if isinstance(route_error, ValueError) and len(model.raw_routes) > before_raw_routes
                       else None),
                   "model_calls": len(calls),
                   "input_tokens": sum(c["input_tokens"] for c in calls) if calls and all(isinstance(c.get("input_tokens"), int) for c in calls) else None,
                   "output_tokens": sum(c["output_tokens"] for c in calls) if calls and all(isinstance(c.get("output_tokens"), int) for c in calls) else None,
                   "latency_ms": round((time.perf_counter() - started) * 1000)}
            rows.append(row)
            print(f"{case['id']}#{repeat}: {grade['actual_intent']} route={grade['route_ok']}", flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()
    return {"manifest": {"dataset_version": corpus["dataset_version"], "dataset_sha256": digest,
                          "git_commit": commit, "split": split, "repeats": repeats,
                          "case_filter": case_id,
                          "model_provider": settings.llm_provider, "model_id": settings.llm_model,
                          "fixed_now": corpus["fixed_now"], "timezone": corpus["timezone"],
                          "evidence": "REAL_MODEL_SYNTHETIC_ROUTE_PLAN_NO_BUSINESS_EXECUTION"},
            "summary": summarize(rows), "rows": rows}


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    fields = ("dataset_version", "dataset_sha256", "split", "repeats", "case_filter", "model_provider",
              "model_id", "fixed_now", "timezone", "evidence")
    if any(baseline["manifest"].get(key) != candidate["manifest"].get(key) for key in fields):
        raise ValueError("Experiment conditions differ; paired comparison refused")
    def index(report):
        return {(row["case_id"], row["repeat"]): row for row in report["rows"]}
    a, b = index(baseline), index(candidate)
    if a.keys() != b.keys() or len(a) != len(baseline["rows"]) or len(b) != len(candidate["rows"]):
        raise ValueError("Experiment attempt identities differ")
    transitions = Counter(f"{int(bool(a[key]['route_ok']))}->{int(bool(b[key]['route_ok']))}" for key in a)
    return {"attempts": len(a), "route_transitions": dict(transitions),
            "baseline": baseline["summary"], "candidate": candidate["summary"],
            "new_failures": [case for case in sorted(a) if a[case]["route_ok"] and not b[case]["route_ok"]],
            "fixed_failures": [case for case in sorted(a) if not a[case]["route_ok"] and b[case]["route_ok"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("all", "dev", "holdout"), default="all")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--case-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run_with_provider_clients(run(split=args.split, repeats=args.repeats,
                                                       case_id=args.case_id)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    if result["summary"]["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
