"""Deterministic, side-effect-free task/trajectory/memory graders.

State snapshots must be collected from the *isolated* environment by the
runner. A text claim of completion never substitutes for a state predicate.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo


def grade_tool_trajectory(actual: list[dict[str, Any]],
                          required: list[dict[str, Any]],
                          *, forbidden: set[str] | None = None,
                          ordered: bool = True) -> dict[str, Any]:
    """Required calls and argument subset are checked; extra calls are visible.

    A subset matcher is appropriate for tools that add safe optional arguments.
    It is diagnostic, never a substitute for an environment postcondition.
    """
    forbidden = forbidden or set()
    if any(not isinstance(call, dict) or not isinstance(call.get("name"), str)
           or not isinstance(call.get("args", {}), dict) for call in actual + required):
        raise ValueError("Tool calls require name and dict args")

    def matches(observed: dict[str, Any], reference: dict[str, Any]) -> bool:
        return (observed["name"] == reference["name"] and
                all(observed.get("args", {}).get(key) == value
                    for key, value in reference.get("args", {}).items()))

    matched_indices: set[int] = set()
    cursor = 0
    for reference in required:
        candidates = range(cursor, len(actual)) if ordered else range(len(actual))
        index = next((i for i in candidates if i not in matched_indices and matches(actual[i], reference)), None)
        if index is None:
            break
        matched_indices.add(index)
        if ordered:
            cursor = index + 1
    forbidden_seen = [call["name"] for call in actual if call["name"] in forbidden]
    return {"passed": len(matched_indices) == len(required) and not forbidden_seen,
            "required_matched": len(matched_indices), "required_total": len(required),
            "forbidden_seen": forbidden_seen,
            "extra_calls": len(actual) - len(matched_indices),
            "duplicate_names": {name: count for name, count in Counter(call["name"] for call in actual).items() if count > 1}}


def grade_memory_retrieval(hit_ids: list[str], expected_ids: set[str],
                           *, forbidden_ids: set[str] | None = None, k: int = 3) -> dict[str, Any]:
    if k < 1 or len(hit_ids) != len(set(hit_ids)):
        raise ValueError("k must be positive and hit IDs unique")
    if not expected_ids:
        raise ValueError("Memory recall requires positive evidence IDs")
    selected = hit_ids[:k]
    correct = sum(hit in expected_ids for hit in selected)
    forbidden = sorted(set(selected) & (forbidden_ids or set()))
    return {"hits": len(selected), "correct": correct, "relevant_total": len(expected_ids),
            "recall_at_k": correct / len(expected_ids),
            "precision_at_k": correct / len(selected) if selected else 0.0,
            "forbidden_hit_ids": forbidden, "safe": not forbidden}


def grade_todo_environment(before: list[dict[str, Any]], after: list[dict[str, Any]],
                           *, owner_id: str, expected: dict[str, str],
                           forbidden_other_owner_change: bool = True,
                           require_owned_unchanged: bool = False) -> dict[str, Any]:
    """Score synthetic Todo title->status postconditions and cross-owner safety.

    Snapshots carry UUID, owner, title and status (no private user data).
    Expected titles are dataset-owned synthetic strings, not model replies.
    """
    def normalize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if any(not {"id", "user_id", "title", "status"}.issubset(row) for row in rows):
            raise ValueError("Todo snapshot lacks authoritative fields")
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate Todo IDs in snapshot")
        return {row["id"]: row for row in rows}

    old, new = normalize(before), normalize(after)
    owned = [row for row in new.values() if row["user_id"] == owner_id]
    matched = {title: [row for row in owned if row["title"] == title and row["status"] == status]
               for title, status in expected.items()}
    changed_other = [row_id for row_id in old.keys() | new.keys()
                     if (old.get(row_id) or new.get(row_id))["user_id"] != owner_id
                     and old.get(row_id) != new.get(row_id)]
    changed_owned = [row_id for row_id in old.keys() | new.keys()
                     if (old.get(row_id) or new.get(row_id))["user_id"] == owner_id
                     and old.get(row_id) != new.get(row_id)]
    return {"passed": all(len(rows) == 1 for rows in matched.values())
            and (not forbidden_other_owner_change or not changed_other)
            and (not require_owned_unchanged or not changed_owned),
            "postconditions_passed": sum(len(rows) == 1 for rows in matched.values()),
            "postconditions_total": len(expected),
            "other_owner_changed": bool(changed_other),
            "owned_changed": bool(changed_owned)}


def grade_reminder_definition(before: list[dict[str, Any]], after: list[dict[str, Any]], *, owner_id: str,
                              title: str, expected_local: str, timezone_name: str,
                              new_occurrences: int, new_deliveries: int,
                              expected_version_delta: int | None = None) -> dict[str, Any]:
    """Judge the persisted definition, not a model claim or a Telegram send.

    A future reminder must not itself create an R4 occurrence or delivery while
    the scheduler and Bot runtime are intentionally absent from evaluation.
    """
    expected = datetime.strptime(expected_local, "%Y-%m-%d %H:%M")
    matches = [row for row in after if row["user_id"] == owner_id and row["title"] == title]
    actual_local = None
    if len(matches) == 1 and matches[0].get("reminder_at"):
        actual = datetime.fromisoformat(matches[0]["reminder_at"])
        if actual.tzinfo is None:
            raise ValueError("Reminder snapshot must include timezone")
        actual_local = actual.astimezone(ZoneInfo(timezone_name)).replace(tzinfo=None)
    previous = [row for row in before if row["user_id"] == owner_id and row["title"] == title]
    version_delta = (matches[0]["reminder_version"] - previous[0]["reminder_version"]
                     if len(matches) == len(previous) == 1 else None)
    version_ok = expected_version_delta is None or version_delta == expected_version_delta
    return {"passed": actual_local == expected and new_occurrences == 0 and new_deliveries == 0
            and version_ok,
            "definition_matches": actual_local == expected,
            "reminder_version_delta": version_delta, "version_delta_matches": version_ok,
            "new_occurrences": new_occurrences, "new_deliveries": new_deliveries}
