"""Typed durable-memory identity and evidence-policy unit tests."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.agent.structured_memory import (
    EvidenceRef, MemoryEvidenceError, ProjectFact, UserFact,
    _verified_refs, memory_key, parse_payload,
)


def test_typed_memory_covers_core_kinds_and_rejects_unknown_fields():
    examples = [
        {"kind": "user_fact", "subject": "user", "attribute": "research_area", "value": "agents"},
        {"kind": "project_fact", "project_key": "Aurora", "attribute": "pages", "value": "52"},
        {"kind": "decision", "project_key": "Aurora", "question": "DB", "outcome": "PostgreSQL"},
        {"kind": "alias", "entity_kind": "project", "entity_key": "Aurora", "label": "nickname", "alias": "Blue Whale"},
        {"kind": "open_loop", "project_key": "Aurora", "loop_key": "review", "item": "Review slides"},
        {"kind": "relationship", "person_key": "Teacher", "relation": "advisor", "related_entity": "Aurora"},
        {"kind": "episode_reference", "event_key": "meeting-1", "description": "Discussed scope"},
    ]
    assert {parse_payload(example).kind for example in examples} == {example["kind"] for example in examples}
    with pytest.raises(ValidationError):
        parse_payload({**examples[1], "unexpected": "not allowed"})


def test_memory_key_is_stable_for_correction_and_opaque():
    old = ProjectFact(project_key="蓝鲸", attribute=" Pages ", value="47")
    new = ProjectFact(project_key="蓝鲸", attribute="pages", value="52")
    assert memory_key(old) == memory_key(new)
    assert "蓝鲸" not in memory_key(old)
    assert memory_key(old) != memory_key(UserFact(subject="user", attribute="pages", value="52"))


def test_evidence_requires_owned_source_and_exact_quote():
    ref = EvidenceRef(source_type="checkpoint", source_id="m1",
                      occurred_at=datetime.now(timezone.utc), evidence_quote="52 pages")
    with pytest.raises(MemoryEvidenceError):
        _verified_refs([ref], {})
    with pytest.raises(MemoryEvidenceError):
        _verified_refs([ref], {("checkpoint", "m1"): "47 pages"})
    verified = _verified_refs([ref], {("checkpoint", "m1"): "Aurora has 52 pages"})
    assert verified[0]["source_id"] == "m1"
    assert "52 pages" not in str(verified)
