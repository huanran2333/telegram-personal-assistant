"""Offline contract checks for the copy-only legacy proof backfill."""
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select

from app.db.migrations.telegram_todo_execution_backfill import backfill
from app.db.models import (ActionExecution, TelegramBusinessOperation,
                           TelegramInboundReceipt, User)
from app.telegram.durable import telegram_todo_action_id


def _database():
    engine = create_engine("sqlite://")
    User.metadata.create_all(engine, tables=[
        User.__table__, TelegramInboundReceipt.__table__,
        TelegramBusinessOperation.__table__, ActionExecution.__table__])
    return engine


def test_action_identity_is_stable_per_receipt_and_call():
    receipt = str(uuid4())
    action = telegram_todo_action_id(receipt, "create_todo", "call-1")
    assert len(action) <= 64 and action.startswith(f"tgtodo:{receipt}:")
    assert action == telegram_todo_action_id(receipt, "create_todo", "call-1")
    assert action != telegram_todo_action_id(receipt, "create_todo", "call-2")
    assert action != telegram_todo_action_id(str(uuid4()), "create_todo", "call-1")
    assert action != telegram_todo_action_id(receipt, "update_todo", "call-1")


def test_backfill_copies_once_without_overwriting_an_execution():
    engine = _database()
    user_id, receipt_id = str(uuid4()), str(uuid4())
    with engine.begin() as db:
        db.execute(User.__table__.insert().values(id=user_id, telegram_id="70001"))
        db.execute(TelegramInboundReceipt.__table__.insert().values(
            id=receipt_id, bot_identity=11, update_id=21, state="FAILED",
            envelope={"message": {"from": {"id": 70001}}}))
        db.execute(TelegramBusinessOperation.__table__.insert().values(
            receipt_id=receipt_id, operation_key="create_todo:call-1",
            args_sha256="a" * 64, status="done", result={"text": "created once"}))
        assert backfill(db) == {"copied": 1, "already_present": 0, "skipped_terminal": 0}
        assert backfill(db) == {"copied": 0, "already_present": 1, "skipped_terminal": 0}
        proof = db.execute(select(ActionExecution.__table__).where(
            ActionExecution.action_id == telegram_todo_action_id(
                receipt_id, "create_todo", "call-1"))).mappings().one()
        assert proof["user_id"] == user_id
        assert proof["result"]["text"] == "created once"
    engine.dispose()


def test_backfill_refuses_unresolved_active_proof():
    engine = _database()
    receipt_id = str(uuid4())
    with engine.begin() as db:
        db.execute(TelegramInboundReceipt.__table__.insert().values(
            id=receipt_id, bot_identity=11, update_id=22, state="FAILED",
            envelope={"message": {"from": {"id": 70002}}}))
        db.execute(TelegramBusinessOperation.__table__.insert().values(
            receipt_id=receipt_id, operation_key="create_todo:call-1",
            args_sha256="b" * 64, status="done", result={"text": "created"}))
        with pytest.raises(RuntimeError, match="ActiveTelegramProofOwnerUnknown"):
            backfill(db)
        assert db.execute(select(ActionExecution.action_id)).first() is None
    engine.dispose()
