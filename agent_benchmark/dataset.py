"""Validate and adapt existing immutable evaluation datasets."""
from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


class InvalidDataset(ValueError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise InvalidDataset("DatasetRequiresCases")
    version = raw.get("version")
    source_sha256 = digest(path)
    if not isinstance(version, str) or not version:
        raise InvalidDataset("DatasetVersionRequired")
    ids: set[str] = set()
    requests: set[str] = set()
    request_list: list[tuple[str, str]] = []
    near_duplicates: list[tuple[str, str]] = []
    cases = []
    for old in raw["cases"]:
        if not isinstance(old, dict):
            raise InvalidDataset("CaseMustBeObject")
        case_id, request = old.get("id"), old.get("request")
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{2,64}", case_id):
            raise InvalidDataset("InvalidCaseId")
        if case_id in ids:
            raise InvalidDataset(f"DuplicateCaseId:{case_id}")
        ids.add(case_id)
        if not isinstance(request, str) or not request.strip():
            raise InvalidDataset(f"EmptyRequest:{case_id}")
        normalized = re.sub(r"\s+", "", request).casefold()
        if normalized in requests:
            raise InvalidDataset(f"DuplicateRequest:{case_id}")
        for prior_id, prior_request in request_list:
            if SequenceMatcher(None, normalized, prior_request).ratio() >= 0.95:
                near_duplicates.append((prior_id, case_id))
        requests.add(normalized)
        request_list.append((case_id, normalized))
        split = old.get("split")
        if split not in {"development", "holdout"}:
            raise InvalidDataset(f"InvalidSplit:{case_id}")
        category = old.get("stratum")
        if not isinstance(category, str) or not category:
            raise InvalidDataset(f"MissingCategory:{case_id}")
        goals, allowed = old.get("goals"), old.get("allowed_writes")
        if not isinstance(goals, list) or not goals or not all(isinstance(g, dict) and g.get("kind") for g in goals):
            raise InvalidDataset(f"MissingGoals:{case_id}")
        if not isinstance(allowed, list) or not all(isinstance(w, str) for w in allowed):
            raise InvalidDataset(f"InvalidAllowedWrites:{case_id}")
        for goal in goals:
            if goal["kind"] == "create" and goal.get("title") not in allowed:
                raise InvalidDataset(f"CreateNotAllowed:{case_id}")
        tags = [category, *(goal["kind"] for goal in goals)]
        cases.append({
            "case_id": case_id, "dataset_version": version, "split": split,
            "source": {"type": "legacy_curated", "file": path.name,
                       "dataset_sha256": source_sha256,
                       "generation_model": "not_recorded", "review_status": raw.get("status", "not_recorded")},
            "category": category, "difficulty": "not_recorded",
            "user_request": request,
            "initial_state": {"seed": old.get("seed", []),
                              "foreign_seed": old.get("foreign_seed", []),
                              "world": raw.get("world", {}),
                              "time_reference_metadata": raw.get("time_reference"),
                              "evaluation_now": old.get("evaluation_now"),
                              "evaluation_timezone": old.get("evaluation_timezone")},
            "available_tools_or_policy": {"allowed_writes": allowed},
            "expected_outcome": goals,
            "required_constraints": {"order": old.get("order"), "decisions": old.get("decisions", {})},
            "oracle_spec": {"kind": "legacy_state_and_tool_results",
                            "manual_expectation": old.get("manual_expectation")},
            "tags": sorted(set(tags)), "legacy": old,
        })
    return {"version": version, "sha256": source_sha256, "cases": cases,
            "near_duplicate_review": near_duplicates,
            "source_path": str(path.resolve())}
