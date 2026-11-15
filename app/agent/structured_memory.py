"""Typed, source-backed durable knowledge lifecycle.

Adaptation of LangMem's schema-based insert/update/delete and existing-memory
comparison to transactional PostgreSQL. Profile Store and domain databases
remain higher authorities. Model output is only a candidate; this module
requires trusted source IDs and exact evidence before persistence.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MemorySuppression, StructuredMemory, StructuredMemoryEmbedding


class MemoryEvidenceError(ValueError):
    """Candidate fact lacks source evidence within the authenticated scope."""


class MemorySuppressedError(ValueError):
    """A forgotten key cannot be reintroduced by background extraction."""


class MemoryAuthorityError(ValueError):
    """A conversation candidate attempts to duplicate a higher-authority fact."""


class _Fact(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class UserFact(_Fact):
    kind: Literal["user_fact"] = "user_fact"
    subject: str = Field(min_length=1, max_length=160)
    attribute: str = Field(min_length=1, max_length=160)
    value: str = Field(min_length=1, max_length=1000)


class ProjectFact(_Fact):
    kind: Literal["project_fact"] = "project_fact"
    project_key: str = Field(min_length=1, max_length=160)
    attribute: str = Field(min_length=1, max_length=160)
    value: str = Field(min_length=1, max_length=1500)


class Decision(_Fact):
    kind: Literal["decision"] = "decision"
    project_key: str = Field(min_length=1, max_length=160)
    question: str = Field(min_length=1, max_length=300)
    outcome: str = Field(min_length=1, max_length=1500)


class Alias(_Fact):
    kind: Literal["alias"] = "alias"
    entity_kind: Literal["project", "todo", "person", "meeting"]
    entity_key: str = Field(min_length=1, max_length=160)
    label: str = Field(min_length=1, max_length=80)
    alias: str = Field(min_length=1, max_length=160)


class OpenLoop(_Fact):
    kind: Literal["open_loop"] = "open_loop"
    project_key: str = Field(min_length=1, max_length=160)
    loop_key: str = Field(min_length=1, max_length=160)
    item: str = Field(min_length=1, max_length=1000)
    state: Literal["open", "resolved"] = "open"
    due_hint: str | None = Field(default=None, max_length=160)


class Relationship(_Fact):
    kind: Literal["relationship"] = "relationship"
    person_key: str = Field(min_length=1, max_length=160)
    relation: str = Field(min_length=1, max_length=160)
    related_entity: str = Field(min_length=1, max_length=160)


class EpisodeReference(_Fact):
    kind: Literal["episode_reference"] = "episode_reference"
    event_key: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=1200)


MemoryPayload = Annotated[
    Union[UserFact, ProjectFact, Decision, Alias, OpenLoop, Relationship, EpisodeReference],
    Field(discriminator="kind"),
]
_PAYLOAD_ADAPTER = TypeAdapter(MemoryPayload)


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    source_type: Literal["checkpoint", "gmail", "calendar", "todo", "profile", "manual"]
    source_id: str = Field(min_length=1, max_length=256)
    thread_id: str | None = Field(default=None, max_length=160)
    occurred_at: datetime
    evidence_quote: str = Field(min_length=1, max_length=500)


def parse_payload(value: dict | _Fact) -> _Fact:
    return _PAYLOAD_ADAPTER.validate_python(value)


def validate_conversation_candidate(payload: dict | _Fact, source_text: str) -> _Fact:
    """Reject source-backed candidates outside conversational-memory authority.

    LangMem's extraction instructions guide the model, but are not an
    enforcement boundary. This schema-aware gate protects Profile and mutable
    domain state before a candidate can enter canonical SQL memory. It is not
    an intent router and never mutates those higher-authority stores.
    """
    fact = parse_payload(payload)
    source = _norm(source_text)
    # A past conversation line is evidence, not a standing instruction for the
    # memory manager. Reject the prompt-control/secret-exfiltration class of
    # imperatives before any typed candidate can become durable knowledge.
    instruction_attack = (
        re.search(r"\b(ignore|override|disregard|bypass)\b.{0,80}"
                  r"\b(instructions?|prompts?|rules?|policy|policies)\b", source),
        re.search(r"\b(store|remember|send|leak|exfiltrate|log)\b.{0,80}"
                  r"\b(secrets?|credentials?|tokens?|passwords?|api keys?)\b", source),
        re.search(r"(忽略|覆盖|绕过).{0,40}(指令|提示词|规则|策略)", source),
        re.search(r"(存储|记住|发送|泄露).{0,40}(秘密|凭据|令牌|密码|密钥)", source),
    )
    if any(instruction_attack):
        raise MemoryAuthorityError("Operational instructions are not durable facts")
    if isinstance(fact, UserFact):
        identity = _norm(f"{fact.subject} {fact.attribute}")
        profile_terms = (
            "assistant", "回复", "回答", "语言", "language", "preference",
            "偏好", "简短", "简洁", "姓名", "名字", "name", "location", "job",
        )
        if any(term in identity for term in profile_terms):
            raise MemoryAuthorityError("Profile Store owns this user preference")
    # These clauses identify mutable domain state, not arbitrary mentions of
    # a task/calendar/email in a project rationale or decision.
    domain_state = (
        (r"\b(todo|task)\b|待办|任务", r"\b(currently|status|marked|complete|completed|pending|done)\b|已完成|待办|进行中"),
        (r"\bcalendar\b|日历|日程", r"\b(scheduled|time|status|cancelled|at [0-9])\b|时间|已取消"),
        (r"\b(gmail|email)\b|邮件", r"\b(unread|read|received|sent|body)\b|未读|已读|收到|发出"),
    )
    if isinstance(fact, (UserFact, EpisodeReference, OpenLoop)) and any(
            re.search(subject, source, flags=re.I) and
            re.search(state, source, flags=re.I)
            for subject, state in domain_state):
        raise MemoryAuthorityError("Current domain state belongs to its source")
    # Exact codes and numbers are especially dangerous to invent. A model may
    # paraphrase prose, but every alphanumeric identifier/number in the typed
    # fact must be found in its cited source.
    canonical = _canonical_text(fact)
    exact_tokens = re.findall(r"(?<!\w)(?:[A-Za-z]+[-_])?\d+(?:[.-]\d+)*(?!\w)", canonical)
    if any(token.casefold() not in source for token in exact_tokens):
        raise MemoryEvidenceError("Exact value is absent from source text")
    return fact


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def memory_key(payload: _Fact) -> str:
    """Opaque stable identity for one mutable semantic assertion slot."""
    if isinstance(payload, UserFact):
        parts = (payload.kind, payload.subject, payload.attribute)
    elif isinstance(payload, ProjectFact):
        parts = (payload.kind, payload.project_key, payload.attribute)
    elif isinstance(payload, Decision):
        parts = (payload.kind, payload.project_key, payload.question)
    elif isinstance(payload, Alias):
        parts = (payload.kind, payload.entity_kind, payload.entity_key, payload.label)
    elif isinstance(payload, OpenLoop):
        parts = (payload.kind, payload.project_key, payload.loop_key)
    elif isinstance(payload, Relationship):
        parts = (payload.kind, payload.person_key, payload.relation)
    elif isinstance(payload, EpisodeReference):
        parts = (payload.kind, payload.event_key)
    else:
        raise TypeError("Unsupported structured memory schema")
    return hashlib.sha256("\x1f".join(_norm(part) for part in parts).encode("utf-8")).hexdigest()


def _canonical_text(payload: _Fact) -> str:
    return " ".join(str(value) for key, value in payload.model_dump().items()
                    if key != "kind" and value is not None)


def _verified_refs(refs: list[EvidenceRef], source_catalog: dict[tuple[str, str], str]) -> list[dict]:
    if not refs:
        raise MemoryEvidenceError("Durable memory requires at least one source")
    result = []
    for ref in refs:
        content = source_catalog.get((ref.source_type, ref.source_id))
        if content is None or ref.evidence_quote not in content:
            raise MemoryEvidenceError("Source quote is absent from the authenticated evidence set")
        if ref.occurred_at.tzinfo is None:
            raise MemoryEvidenceError("Source timestamp requires timezone")
        result.append({
            "source_type": ref.source_type,
            "source_id": ref.source_id,
            **({"thread_id": ref.thread_id} if ref.thread_id else {}),
            "occurred_at": ref.occurred_at.isoformat(),
            # Preserve an evidence fingerprint, not another copy of private text.
            "quote_sha256": hashlib.sha256(ref.evidence_quote.encode("utf-8")).hexdigest(),
        })
    return list({(item["source_type"], item["source_id"]): item for item in result}.values())


async def upsert_memory(
    db: AsyncSession, *, user_id: str, payload: dict | _Fact,
    refs: list[EvidenceRef], source_catalog: dict[tuple[str, str], str],
    explicit_restore: bool = False, expires_at: datetime | None = None,
) -> StructuredMemory:
    """Create, reinforce or correct one typed assertion in caller transaction.

    A model cannot write this table directly: source identifiers and supporting
    quotes must be verified against a caller-supplied, user-scoped evidence set.
    Explicit restore is reserved for an authenticated user correction path.
    """
    if not user_id:
        raise ValueError("user_id is required")
    fact = parse_payload(payload)
    if isinstance(fact, UserFact) and _norm(fact.attribute) in {
            "name", "location", "job", "assistant_language", "language", "preference", "preferences"}:
        raise ValueError("Profile Store owns this user attribute")
    if expires_at is not None and expires_at.tzinfo is None:
        raise ValueError("expires_at requires timezone")
    key = memory_key(fact)
    verified = _verified_refs(refs, source_catalog)
    suppression = await db.scalar(select(MemorySuppression).where(
        MemorySuppression.user_id == user_id, MemorySuppression.memory_key == key).with_for_update())
    if suppression is not None:
        if not explicit_restore:
            raise MemorySuppressedError("Forgotten memory cannot be reintroduced automatically")
        await db.delete(suppression)
        await db.flush()

    current = await db.scalar(select(StructuredMemory).where(
        StructuredMemory.user_id == user_id,
        StructuredMemory.memory_key == key,
        StructuredMemory.status == "current").with_for_update())
    now = datetime.now(timezone.utc)
    payload_json = fact.model_dump(mode="json")
    if current is not None and current.payload == payload_json:
        prior = {(ref["source_type"], ref["source_id"]): ref for ref in current.source_refs}
        for ref in verified:
            prior[(ref["source_type"], ref["source_id"])] = ref
        current.source_refs = list(prior.values())
        current.row_version += 1
        current.updated_at = now
        await db.flush()
        return current

    lineage = current.lineage_version + 1 if current else 1
    supersedes_id = current.id if current else None
    if current is None:
        latest = await db.scalar(select(StructuredMemory).where(
            StructuredMemory.user_id == user_id,
            StructuredMemory.memory_key == key,
        ).order_by(StructuredMemory.lineage_version.desc()).limit(1).with_for_update())
        if latest is not None:
            lineage = latest.lineage_version + 1
            supersedes_id = latest.id
    if current is not None:
        current.status = "superseded"
        current.updated_at = now
        current.row_version += 1
        await db.execute(delete(StructuredMemoryEmbedding).where(
            StructuredMemoryEmbedding.user_id == user_id,
            StructuredMemoryEmbedding.memory_id == current.id))
        await db.flush()
    row = StructuredMemory(
        user_id=user_id, memory_key=key, kind=fact.kind,
        payload=payload_json, canonical_text=_canonical_text(fact),
        source_refs=verified, status="current", lineage_version=lineage,
        supersedes_id=supersedes_id, expires_at=expires_at,
        created_at=now, updated_at=now,
    )
    db.add(row)
    await db.flush()
    return row


async def forget_memory(db: AsyncSession, *, user_id: str, memory_id: str,
                        reason: Literal["explicit_forget", "source_removed"] = "explicit_forget") -> bool:
    """Tombstone current fact and suppress automatic resurrection."""
    row = await db.scalar(select(StructuredMemory).where(
        StructuredMemory.id == memory_id,
        StructuredMemory.user_id == user_id,
        StructuredMemory.status == "current").with_for_update())
    if row is None:
        return False
    now = datetime.now(timezone.utc)
    row.status = "tombstoned"
    row.tombstoned_at = now
    row.updated_at = now
    row.row_version += 1
    await db.execute(delete(StructuredMemoryEmbedding).where(
        StructuredMemoryEmbedding.user_id == user_id,
        StructuredMemoryEmbedding.memory_id == row.id))
    await db.execute(pg_insert(MemorySuppression).values(
        user_id=user_id, memory_key=row.memory_key,
        source_memory_id=row.id, reason=reason, suppressed_at=now,
    ).on_conflict_do_update(
        index_elements=[MemorySuppression.user_id, MemorySuppression.memory_key],
        set_={"source_memory_id": row.id, "reason": reason, "suppressed_at": now},
    ))
    await db.flush()
    return True


async def remove_source(db: AsyncSession, *, user_id: str,
                        source_type: str, source_id: str) -> int:
    """Revoke one provenance link; tombstone facts with no remaining evidence."""
    rows = (await db.scalars(select(StructuredMemory).where(
        StructuredMemory.user_id == user_id,
        StructuredMemory.status == "current").with_for_update())).all()
    changed = 0
    for row in rows:
        remaining = [ref for ref in row.source_refs if not (
            ref["source_type"] == source_type and ref["source_id"] == source_id)]
        if len(remaining) == len(row.source_refs):
            continue
        changed += 1
        if remaining:
            row.source_refs = remaining
            row.row_version += 1
            row.updated_at = datetime.now(timezone.utc)
        else:
            await forget_memory(db, user_id=user_id, memory_id=row.id, reason="source_removed")
    await db.flush()
    return changed


async def current_memories(db: AsyncSession, *, user_id: str,
                           kind: str | None = None, limit: int = 100) -> list[StructuredMemory]:
    if not user_id or not 1 <= limit <= 500:
        raise ValueError("Valid user scope and bounded limit required")
    statement = select(StructuredMemory).where(
        StructuredMemory.user_id == user_id,
        StructuredMemory.status == "current",
        (StructuredMemory.expires_at.is_(None) |
         (StructuredMemory.expires_at > datetime.now(timezone.utc))),
    )
    if kind is not None:
        statement = statement.where(StructuredMemory.kind == kind)
    return (await db.scalars(statement.order_by(StructuredMemory.updated_at.desc()).limit(limit))).all()
