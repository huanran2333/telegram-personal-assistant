import json
import logging

from app.trace import (begin_trace, current_trace_id, emit_trace, end_trace,
                       set_correlation, set_user)


def test_trace_correlates_stages_without_raw_secret_or_payload(caplog):
    caplog.set_level(logging.INFO, logger="app.turn_trace")
    token = begin_trace()
    try:
        set_user("user-1")
        emit_trace("route", route="smart_schedule", token="should-not-appear",
                   raw_payload="private-message", success=True)
        emit_trace("read", tool="calendar.get-freebusy", success=True)
    finally:
        end_trace(token)
    records = [json.loads(record.message) for record in caplog.records
               if record.name == "app.turn_trace"]
    assert len(records) == 3
    assert len({record["trace_id"] for record in records}) == 1
    assert records[-1]["stage"] == "request_end"
    assert records[0]["user_id"] == "user-1"
    assert "should-not-appear" not in str(records)
    assert "private-message" not in str(records)


def test_trace_allowlist_rejects_sensitive_or_malformed_fields(caplog):
    caplog.set_level(logging.INFO, logger="app.turn_trace")
    token = begin_trace()
    try:
        first = current_trace_id()
        set_correlation(origin_trace_id="a" * 32, thread_id="workflow_user_123456789")
        emit_trace("proposal", operation="create_event", action_id="b" * 32,
                   email_body="PRIVATE_EMAIL_BODY", args={"token": "PRIVATE_TOKEN"},
                   external_id="PRIVATE EVENT TITLE with spaces", token_usage=None,
                   error_category="OAuth failed: PRIVATE_TOKEN",
                   route="sk-private_token_value")
    finally:
        end_trace(token)
    records = [json.loads(record.message) for record in caplog.records
               if record.name == "app.turn_trace"]
    assert records[0]["trace_id"] == first
    assert records[0]["origin_trace_id"] == "a" * 32
    assert records[0]["thread_ref"] != "workflow_user_123456789"
    assert records[-1]["trace_id"] == first
    assert "token_usage" not in records[0]
    assert "external_id" not in records[0]
    assert "error_category" not in records[0]
    assert "route" not in records[0]
    assert "PRIVATE" not in str(records)
