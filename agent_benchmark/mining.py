"""Propose review-only variants; never mutate development or frozen datasets."""
from __future__ import annotations

import copy
import re


def candidate_from_failure(case: dict, row: dict, trace_hash: str | None) -> dict:
    candidate = {"origin_case_id": case["case_id"],
                 "failure_pattern": row.get("reasons", []),
                 "source_trace_sha256": trace_hash,
                 "status": "pending_human_review",
                 "required_checks": ["ground_truth", "near_duplicate", "tags", "sensitive_data"],
                 "variant": None}
    if case["source"]["type"] != "legacy_curated":
        return candidate
    # A single literal-title substitution gives a bounded proposal, not truth.
    creates = [g for g in case["expected_outcome"] if g["kind"] == "create"]
    if len(creates) != 1:
        return candidate
    title = creates[0].get("title")
    if not isinstance(title, str) or case["user_request"].count(title) != 1:
        return candidate
    if re.search(r"(?:@|https?://|\b(?:token|secret|password|api.key)\b)",
                 case["user_request"], re.IGNORECASE):
        return candidate
    variant = copy.deepcopy(case)
    replacement = title + "-变体待审核"
    variant["case_id"] = case["case_id"] + "-candidate-1"
    variant["dataset_version"] = "candidate-unreleased"
    variant["split"] = "candidate"
    variant["user_request"] = case["user_request"].replace(title, replacement)
    variant["initial_state"] = {"requires_human_curation": True}
    variant["oracle_spec"] = {"kind": "pending_human_ground_truth"}
    variant["expected_outcome"][case["expected_outcome"].index(creates[0])]["title"] = replacement
    variant["available_tools_or_policy"]["allowed_writes"] = [
        replacement if value == title else value
        for value in variant["available_tools_or_policy"]["allowed_writes"]]
    variant.pop("legacy", None)
    variant["source"] = {"type": "deterministic_failure_variant", "generation_model": "none",
                         "review_status": "pending_human_review"}
    candidate["variant"] = variant
    return candidate
