from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import (
    String, Text, DateTime, Enum, ForeignKey, Boolean, JSON, UniqueConstraint, Index, Integer, BigInteger, CheckConstraint, text
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
import enum
import uuid

class Base(DeclarativeBase):
    pass

class Priority(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"

class TodoStatus(str, enum.Enum):
    pending = "pending"
    in_progress = "in_progress"
    done = "done"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower().replace(" ", "_")
            alias_map = {
                "completed": "done",
                "complete": "done",
                "finished": "done",
                "in_progress": "in_progress",
                "in progress": "in_progress",
                "started": "in_progress",
                "doing": "in_progress",
                "pending": "pending",
                "todo": "pending",
                "open": "pending",
            }
            normalized = alias_map.get(normalized, normalized)
            if normalized in cls._value2member_map_:
                return cls(normalized)
        return super()._missing_(value)

def gen_uuid() -> str:
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    telegram_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    first_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    preferences: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    # Per-user preference for Google time-block projection. Existing false
    # values are preserved; the default for newly added users is on.
    calendar_time_block_sync: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    todos: Mapped[List["Todo"]] = relationship("Todo", back_populates="user", cascade="all, delete-orphan")


class WebCredential(Base):
    __tablename__ = "web_credentials"

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                  default=lambda: datetime.now(timezone.utc))


class WebSession(Base):
    __tablename__ = "web_sessions"
    __table_args__ = (Index("ix_web_session_owner_expiry", "user_id", "expires_at"),)

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    credential_version: Mapped[int] = mapped_column(Integer, nullable=False)
    auth_source: Mapped[str] = mapped_column(String(16), nullable=False, default="password")
    telegram_bot_identity: Mapped[Optional[int]] = mapped_column(BigInteger)
    telegram_user_id: Mapped[Optional[str]] = mapped_column(String(64))
    csrf_token: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                  default=lambda: datetime.now(timezone.utc))


class TelegramMiniAppLaunchTicket(Base):
    """Ten-minute repeatable launch binding; signed initData proves each login.

    consumed_at records first redemption for audit, not one-use authority.
    """
    __tablename__ = "telegram_miniapp_launch_tickets"
    __table_args__ = (
        Index("ix_miniapp_ticket_expiry", "expires_at"),
    )

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    bot_identity: Mapped[int] = mapped_column(BigInteger, nullable=False)
    telegram_user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    consumed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                  default=lambda: datetime.now(timezone.utc))


class WebLoginNonce(Base):
    __tablename__ = "web_login_nonces"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    csrf_token: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WebLoginThrottle(Base):
    __tablename__ = "web_login_throttles"

    client_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    blocked_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    window_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WebRequest(Base):
    """Durable Web custody/result locator; Saver and domain rows stay authoritative."""
    __tablename__ = "web_requests"
    __table_args__ = (
        UniqueConstraint("user_id", "operation", "idempotency_key", name="uq_web_request_key"),
        Index("ix_web_request_recovery", "state", "lease_until"),
        CheckConstraint("state IN ('ACCEPTED','RUNNING','WAITING_APPROVAL','WAITING_EDIT',"
                        "'WAITING_EXTERNAL','COMPLETED','FAILED')", name="ck_web_request_state"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    operation: Mapped[str] = mapped_column(String(32), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(96), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    request_body: Mapped[dict] = mapped_column(JSON, nullable=False)
    thread_id: Mapped[str] = mapped_column(String(160), nullable=False)
    turn_id: Mapped[str] = mapped_column(String(160), nullable=False, unique=True)
    route_payload: Mapped[Optional[dict]] = mapped_column(JSON)
    graph_kind: Mapped[Optional[str]] = mapped_column(String(32))
    state: Mapped[str] = mapped_column(String(24), nullable=False, default="ACCEPTED")
    result: Mapped[Optional[dict]] = mapped_column(JSON)
    safe_error: Mapped[Optional[str]] = mapped_column(String(80))
    lease_owner: Mapped[Optional[str]] = mapped_column(String(80))
    fence_token: Mapped[Optional[str]] = mapped_column(String(36))
    lease_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                  default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                  default=lambda: datetime.now(timezone.utc))


class WebMutation(Base):
    """HTTP idempotency proof committed in the same transaction as a form mutation."""
    __tablename__ = "web_mutations"
    __table_args__ = (UniqueConstraint("user_id", "operation", "idempotency_key",
                                         name="uq_web_mutation_key"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    operation: Mapped[str] = mapped_column(String(48), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(96), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    result: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                  default=lambda: datetime.now(timezone.utc))


class MeetingDraft(Base):
    """Owner-scoped content resource; independent of Agent checkpoints and action approval."""

    __tablename__ = "meeting_drafts"
    __table_args__ = (
        UniqueConstraint("user_id", "turn_id", "event_id", name="uq_meeting_draft_turn_event"),
        Index("ix_meeting_draft_owner_updated", "user_id", "updated_at"),
        CheckConstraint("status IN ('NEEDS_REVIEW','ACCEPTED','CANCELED','STALE')",
                        name="ck_meeting_draft_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    turn_id: Mapped[str] = mapped_column(String(160), nullable=False)
    event_id: Mapped[str] = mapped_column(String(256), nullable=False)
    calendar_account: Mapped[str] = mapped_column(String(256), nullable=False)
    calendar_id: Mapped[str] = mapped_column(String(256), nullable=False)
    event_etag: Mapped[str] = mapped_column(String(256), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="NEEDS_REVIEW")
    rendered: Mapped[str] = mapped_column(Text, nullable=False)
    brief_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    evidence_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    revisions: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    last_decision: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                  default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                  default=lambda: datetime.now(timezone.utc))

class Todo(Base):
    __tablename__ = "todos"
    __table_args__ = (CheckConstraint(
        "(planned_start_at IS NULL AND planned_end_at IS NULL) OR "
        "(planned_start_at IS NOT NULL AND planned_end_at > planned_start_at)",
        name="ck_todo_planned_window"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # A deadline and a notification are distinct user commitments.  Keep the
    # former even when a reminder is changed.
    due_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # A plan to work on the item, not its deadline or notification time.
    planned_start_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    planned_end_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    reminder_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    priority: Mapped[Priority] = mapped_column(Enum(Priority), default=Priority.medium)
    status: Mapped[TodoStatus] = mapped_column(Enum(TodoStatus), default=TodoStatus.pending)
    tags: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    # ``reminder_sent`` is retained for old deployments; new dispatch logic
    # uses the timestamps below to claim work safely across scheduler workers.
    reminder_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    reminder_claimed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    reminder_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    reminder_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reminder_state: Mapped[str] = mapped_column(String(12), nullable=False, default="NONE")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now() )
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now() , onupdate=datetime.now() )

    user: Mapped["User"] = relationship("User", back_populates="todos")


class TodoCalendarMirror(Base):
    """Durable desired Google projection; survives deletion of its Todo."""

    __tablename__ = "todo_calendar_mirrors"
    __table_args__ = (
        Index("ix_todo_calendar_mirror_recovery", "status", "next_attempt_at", "lease_until"),
        UniqueConstraint("account_identity", "calendar_id", "event_id",
                         name="uq_todo_calendar_remote_identity"),
    )

    todo_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    calendar_id: Mapped[str] = mapped_column(String(256), nullable=False, default="primary", server_default="primary")
    account_identity: Mapped[Optional[str]] = mapped_column(String(256))
    event_id: Mapped[Optional[str]] = mapped_column(String(256))
    etag: Mapped[Optional[str]] = mapped_column(String(512))
    desired_revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=text("1"))
    synced_revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    desired_operation: Mapped[str] = mapped_column(String(8), nullable=False, default="UPSERT", server_default="UPSERT")
    desired_payload: Mapped[Optional[dict]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="PENDING", server_default="PENDING")
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    next_attempt_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    lease_token: Mapped[Optional[str]] = mapped_column(String(64))
    lease_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_error_category: Mapped[Optional[str]] = mapped_column(String(48))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                  default=lambda: datetime.now(timezone.utc),
                                                  server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                  default=lambda: datetime.now(timezone.utc),
                                                  server_default=text("CURRENT_TIMESTAMP"))


class ActionExecution(Base):
    """Durable execution ledger for approved actions and direct Todo tool calls."""

    __tablename__ = "action_executions"

    action_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    domain: Mapped[str] = mapped_column(String(64), nullable=False)
    operation: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="executing")
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    external_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class CalendarDependentContinuation(Base):
    """Locator only; the Saver checkpoint remains the approved workflow authority."""

    __tablename__ = "calendar_dependent_continuations"
    __table_args__ = (
        CheckConstraint("state IN ('WAITING','READY','COMPLETED','REVIEW_REQUIRED')",
                        name="ck_calendar_continuation_state"),
        Index("ix_calendar_continuation_recovery", "state", "lease_until"),
    )

    parent_action_id: Mapped[str] = mapped_column(String(64), ForeignKey("action_executions.action_id"), primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, default=gen_uuid)
    thread_id: Mapped[str] = mapped_column(String(160), nullable=False)
    checkpoint_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    child_action_id: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    state: Mapped[str] = mapped_column(String(24), nullable=False, default="WAITING")
    lease_owner: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    fence_token: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    lease_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class CalendarActionEvidence(Base):
    """R5 approved-version and remote-attempt evidence; never approval authority."""

    __tablename__ = "calendar_action_evidence"
    __table_args__ = (
        CheckConstraint("operation IN ('create_event','update_event','delete_event')",
                        name="ck_calendar_evidence_operation"),
        CheckConstraint("outcome IN ('NOT_STARTED','IN_FLIGHT','CONFIRMED',"
                        "'FAILED_DEFINITE','UNCERTAIN','PARKED_REVIEW')",
                        name="ck_calendar_evidence_outcome"),
        CheckConstraint("operation = 'create_event' OR approved_baseline_etag IS NOT NULL",
                        name="ck_calendar_evidence_approved_etag"),
        Index("ix_calendar_evidence_recovery", "outcome", "lease_until"),
    )

    action_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    operation: Mapped[str] = mapped_column(String(32), nullable=False)
    account_identity: Mapped[str] = mapped_column(String(128), nullable=False)
    calendar_id: Mapped[str] = mapped_column(String(256), nullable=False)
    event_id: Mapped[str] = mapped_column(String(1024), nullable=False)
    approved_args_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    approved_baseline_etag: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    approved_baseline: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    desired_postcondition: Mapped[dict] = mapped_column(JSON, nullable=False)
    immutable_body: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    private_action_marker: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False, default="NOT_STARTED")
    owner: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    fence_token: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    lease_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dispatch_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    failure_category: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    confirmed_external_id: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    confirmed_etag: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    confirmation_source: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  default=lambda: datetime.now(timezone.utc))


class TelegramInboundReceipt(Base):
    """Durable custody of one bot-scoped Telegram update; not a delivery receipt."""

    __tablename__ = "telegram_inbound_receipts"
    __table_args__ = (
        UniqueConstraint("bot_identity", "update_id", name="uq_telegram_bot_update"),
        Index("ix_telegram_receipt_recovery", "state", "next_attempt_at", "lease_until"),
        Index("ix_telegram_receipt_thread", "graph_kind", "thread_id", "received_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    bot_identity: Mapped[int] = mapped_column(BigInteger, nullable=False)
    update_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    envelope: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    envelope_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="RECEIVED")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lease_owner: Mapped[Optional[str]] = mapped_column(String(80))
    fence_token: Mapped[Optional[str]] = mapped_column(String(36))
    lease_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    failure_category: Mapped[Optional[str]] = mapped_column(String(80))
    graph_kind: Mapped[Optional[str]] = mapped_column(String(24))
    thread_id: Mapped[Optional[str]] = mapped_column(String(160))
    action_id: Mapped[Optional[str]] = mapped_column(String(64))
    origin_trace_id: Mapped[Optional[str]] = mapped_column(String(64))
    outcome_kind: Mapped[Optional[str]] = mapped_column(String(32))
    outcome_ref: Mapped[Optional[str]] = mapped_column(String(256))
    required_reply_count: Mapped[Optional[int]] = mapped_column(Integer)
    resume_from_checkpoint_id: Mapped[Optional[str]] = mapped_column(String(128))
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  default=lambda: datetime.now(timezone.utc))
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    terminal_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    payload_redacted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class TelegramApprovalMarkupSync(Base):
    """Retryable UI projection for an approval or expiring Mini App button.

    action_id is the original action ID for HITL, or a launch ticket verifier
    hash for workbench buttons. It is never a second approval/login authority.
    """

    __tablename__ = "telegram_approval_markup_sync"
    delivery_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("outbound_deliveries.id"), primary_key=True)
    action_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fence_token: Mapped[Optional[str]] = mapped_column(String(36))
    lease_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    failure_category: Mapped[Optional[str]] = mapped_column(String(80))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class TelegramThreadLease(Base):
    """Cross-worker single-flight admission for a LangGraph thread."""

    __tablename__ = "telegram_thread_leases"
    graph_kind: Mapped[str] = mapped_column(String(24), primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    receipt_id: Mapped[str] = mapped_column(String(36), nullable=False)
    lease_owner: Mapped[str] = mapped_column(String(80), nullable=False)
    fence_token: Mapped[str] = mapped_column(String(36), nullable=False)
    lease_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class OutboundDelivery(Base):
    """One immutable user-visible Telegram reply intent."""

    __tablename__ = "outbound_deliveries"
    __table_args__ = (
        UniqueConstraint("receipt_id", "reply_ordinal", name="uq_outbound_receipt_ordinal"),
        UniqueConstraint("occurrence_id", name="uq_outbound_occurrence"),
        CheckConstraint("(receipt_id IS NULL) <> (occurrence_id IS NULL)",
                        name="ck_outbound_one_origin"),
        Index("ix_outbound_delivery_state", "state", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    receipt_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("telegram_inbound_receipts.id"))
    occurrence_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("scheduled_occurrences.id"))
    reply_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    chat_id: Mapped[str] = mapped_column(String(64), nullable=False)
    locale: Mapped[str] = mapped_column(String(16), nullable=False)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parse_mode: Mapped[Optional[str]] = mapped_column(String(24))
    keyboard: Mapped[Optional[dict]] = mapped_column(JSON)
    payload_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=lambda: datetime.now(timezone.utc))
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    terminal_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    payload_redacted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class ScheduledOccurrence(Base):
    """Stable proactive identity; APScheduler only wakes discovery workers."""

    __tablename__ = "scheduled_occurrences"
    __table_args__ = (
        UniqueConstraint("occurrence_key", name="uq_scheduled_occurrence_key"),
        Index("ix_scheduled_occurrence_recovery", "state", "next_attempt_at", "lease_until"),
        Index("ix_scheduled_occurrence_user", "user_id", "kind", "planned_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    occurrence_key: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    bot_identity: Mapped[int] = mapped_column(BigInteger, nullable=False)
    chat_id: Mapped[str] = mapped_column(String(64), nullable=False)
    local_date: Mapped[Optional[str]] = mapped_column(String(10))
    timezone_name: Mapped[str] = mapped_column(String(80), nullable=False)
    planned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    todo_id: Mapped[Optional[str]] = mapped_column(String(36))
    reminder_at_snapshot: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    reminder_version_snapshot: Mapped[Optional[int]] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lease_owner: Mapped[Optional[str]] = mapped_column(String(80))
    fence_token: Mapped[Optional[str]] = mapped_column(String(36))
    lease_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    failure_category: Mapped[Optional[str]] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  default=lambda: datetime.now(timezone.utc))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class OutboundDeliveryChunk(Base):
    """A single Bot API send; logical delivery is SENT only if every chunk is confirmed."""

    __tablename__ = "outbound_delivery_chunks"
    delivery_id: Mapped[str] = mapped_column(String(36), ForeignKey("outbound_deliveries.id"), primary_key=True)
    chunk_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lease_owner: Mapped[Optional[str]] = mapped_column(String(80))
    fence_token: Mapped[Optional[str]] = mapped_column(String(36))
    lease_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    send_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    failure_category: Mapped[Optional[str]] = mapped_column(String(80))
    telegram_chat_id: Mapped[Optional[str]] = mapped_column(String(64))
    telegram_message_id: Mapped[Optional[int]] = mapped_column(BigInteger)


class OutboundDeliveryAttempt(Base):
    """Immutable attempt identity and uncertain remote outcome audit."""

    __tablename__ = "outbound_delivery_attempts"
    __table_args__ = (UniqueConstraint("delivery_id", "chunk_index", "attempt_no",
                                       name="uq_outbound_chunk_attempt"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    delivery_id: Mapped[str] = mapped_column(String(36), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False)
    fence_token: Mapped[str] = mapped_column(String(36), nullable=False)
    state: Mapped[str] = mapped_column(String(20), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=lambda: datetime.now(timezone.utc))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    failure_category: Mapped[Optional[str]] = mapped_column(String(80))
    telegram_message_id: Mapped[Optional[int]] = mapped_column(BigInteger)


class TelegramBusinessOperation(Base):
    """Migration-only mapping for retired Telegram Todo proofs; never runtime DDL."""

    __tablename__ = "telegram_business_operations"
    receipt_id: Mapped[str] = mapped_column(String(36), ForeignKey("telegram_inbound_receipts.id"), primary_key=True)
    operation_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    args_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    result: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=lambda: datetime.now(timezone.utc))


class DailyBriefDelivery(Base):
    """One row per user/local date prevents duplicate proactive sends."""

    __tablename__ = "daily_brief_deliveries"

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    brief_date: Mapped[str] = mapped_column(String(10), primary_key=True)
    claimed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class TranscriptProjection(Base):
    """Rebuildable search index; checkpoints, not this table, own conversation state."""

    __tablename__ = "transcript_projection"
    __table_args__ = (
        UniqueConstraint("user_id", "thread_id", "message_id", name="uq_transcript_source"),
        Index("ix_transcript_scope_time", "user_id", "thread_id", "occurred_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(160), nullable=False)
    message_id: Mapped[str] = mapped_column(String(160), nullable=False)
    group_id: Mapped[str] = mapped_column(String(160), nullable=False)
    role: Mapped[str] = mapped_column(String(24), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    search_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_ref: Mapped[str] = mapped_column(String(256), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    supersedes_message_id: Mapped[Optional[str]] = mapped_column(String(160))
    supersedes_thread_id: Mapped[Optional[str]] = mapped_column(String(160))
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class TranscriptSuppression(Base):
    """Privacy tombstone authority; prevents derived-index rebuild resurrection."""

    __tablename__ = "transcript_suppressions"
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    message_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    supersedes_message_id: Mapped[Optional[str]] = mapped_column(String(160))
    supersedes_thread_id: Mapped[Optional[str]] = mapped_column(String(160))
    suppressed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                     default=lambda: datetime.now(timezone.utc))


class ConversationSummary(Base):
    """Non-authoritative LangMem cursor over retained checkpoint history.

    A successful model response and cursor advance are committed together. The
    full Saver history remains unchanged and is the source for rebuilding.
    """

    __tablename__ = "conversation_summaries"

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    summary_id: Mapped[str] = mapped_column(String(36), nullable=False, default=gen_uuid)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # LangMem's cursor IDs identify atomic completed turns. This separate list
    # preserves each underlying Human/final-AI source ID for audit/rebuild.
    summarized_message_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    covered_source_message_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    last_summarized_message_id: Mapped[Optional[str]] = mapped_column(String(160))
    first_source_message_id: Mapped[Optional[str]] = mapped_column(String(160))
    last_source_message_id: Mapped[Optional[str]] = mapped_column(String(160))
    model_name: Mapped[Optional[str]] = mapped_column(String(160))
    provider: Mapped[Optional[str]] = mapped_column(String(64))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=lambda: datetime.now(timezone.utc))
    last_failure_category: Mapped[Optional[str]] = mapped_column(String(80))
    last_failure_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class StructuredMemory(Base):
    """Canonical, source-backed durable knowledge; never a domain DB authority."""

    __tablename__ = "structured_memories"
    __table_args__ = (
        UniqueConstraint("user_id", "memory_key", "lineage_version", name="uq_memory_lineage"),
        Index("uq_memory_current", "user_id", "memory_key", unique=True,
              postgresql_where=text("status = 'current'")),
        Index("ix_memory_scope_status", "user_id", "status", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    memory_key: Mapped[str] = mapped_column(String(512), nullable=False)
    kind: Mapped[str] = mapped_column(String(40), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    canonical_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_refs: Mapped[list] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="current")
    lineage_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    row_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    supersedes_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("structured_memories.id"))
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=lambda: datetime.now(timezone.utc))
    tombstoned_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class MemorySuppression(Base):
    """Explicit forget authority; prevents re-extraction from old transcript."""

    __tablename__ = "memory_suppressions"
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    memory_key: Mapped[str] = mapped_column(String(512), primary_key=True)
    source_memory_id: Mapped[Optional[str]] = mapped_column(String(36))
    reason: Mapped[str] = mapped_column(String(32), nullable=False)
    suppressed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                     default=lambda: datetime.now(timezone.utc))


class MemoryConsolidationCursor(Base):
    """Per-thread source cursor committed only after memory lifecycle changes."""

    __tablename__ = "memory_consolidation_cursors"
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    last_source_message_id: Mapped[Optional[str]] = mapped_column(String(160))
    last_source_occurred_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_success_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_failure_category: Mapped[Optional[str]] = mapped_column(String(80))
    last_failure_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class TranscriptEmbedding(Base):
    """Rebuildable semantic index; never a transcript or fact authority."""

    __tablename__ = "transcript_embeddings"
    __table_args__ = (
        UniqueConstraint("user_id", "thread_id", "message_id", "model_id",
                         name="uq_transcript_embedding_source_model"),
        Index("ix_embedding_scope_model", "user_id", "thread_id", "model_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(160), nullable=False)
    message_id: Mapped[str] = mapped_column(String(160), nullable=False)
    model_id: Mapped[str] = mapped_column(String(200), nullable=False)
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[list] = mapped_column(JSON, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=lambda: datetime.now(timezone.utc))


class StructuredMemoryEmbedding(Base):
    """Rebuildable semantic projection of canonical structured knowledge."""

    __tablename__ = "structured_memory_embeddings"
    __table_args__ = (
        UniqueConstraint("user_id", "memory_id", "model_id",
                         name="uq_structured_embedding_source_model"),
        Index("ix_structured_embedding_scope_model", "user_id", "model_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    memory_id: Mapped[str] = mapped_column(String(36), nullable=False)
    model_id: Mapped[str] = mapped_column(String(200), nullable=False)
    provider: Mapped[str] = mapped_column(String(80), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[list] = mapped_column(JSON, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=lambda: datetime.now(timezone.utc))
