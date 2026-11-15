from datetime import datetime, timezone

import pytest

from app.db.legacy_migration_dry_run import (
    EXCLUDED_TABLES, OWNER_TABLES, TARGETS, _map_row,
)


def test_archived_execution_tables_cannot_enter_owner_whitelist():
    assert not set(OWNER_TABLES) & set(EXCLUDED_TABLES)
    assert {"checkpoints", "action_executions", "scheduled_occurrences",
            "outbound_deliveries", "calendar_action_evidence"} <= set(EXCLUDED_TABLES)
    assert TARGETS["production"][1:] == (
        "5432", "personal_assistant", "personal_assistant_app")


def test_legacy_todo_mapping_preserves_identity_and_deadline_without_send():
    created = datetime(2026, 9, 30, tzinfo=timezone.utc)
    original = {"id": "original-id", "user_id": "original-owner",
                "due_date": created, "reminder_at": None,
                "status": "pending", "created_at": created}
    mapped = _map_row("todos", original)
    assert original.get("version") is None
    assert {key: mapped[key] for key in original} == original
    assert (mapped["version"], mapped["reminder_version"],
            mapped["reminder_state"]) == (1, 0, "NONE")


def test_legacy_active_reminder_requires_review():
    with pytest.raises(RuntimeError, match="Active legacy reminder"):
        _map_row("todos", {"reminder_at": datetime.now(timezone.utc)})
