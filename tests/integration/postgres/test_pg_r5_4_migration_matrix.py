"""R5.4 E4 frozen migration paths in rollback-only isolated PG18 schemas."""
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.db.models import Base, ActionExecution, TranscriptProjection, User
from app.db.migrations.r4_2_durable_ingress import apply as apply_r4_2
from app.db.migrations.r4_2_outcome_proof import apply as apply_r4_2_proof
from app.db.migrations.r4_3_proactive_occurrences import (
    apply as apply_r4_3, create_occurrence_table)
from app.db.migrations.r4_4_terminal_payload_redaction import apply as apply_r4_4
from app.db.migrations.r5_1_calendar_action_evidence import apply as apply_r5_1
from app.db.migrations.r5_2_create_plan import apply as apply_r5_2
from app.db.migrations.r5_3_conditional_modification import apply as apply_r5_3
from app.db.migrations.r5_4_dependent_continuation import apply as apply_r5_4

pytestmark = pytest.mark.integration


def _columns(sync_connection):
    return set(sync_connection.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = 'calendar_action_evidence'"
    )).scalars())


@pytest.mark.asyncio
async def test_pg18_fresh_and_historical_r5_upgrade_are_idempotent(pg_harness):
    _, _, engine = pg_harness  # fixture independently validates major/data dir/database
    for shape in ("fresh", "pre_r5_1", "r5_1", "r5_2"):
        schema = "r54_" + uuid4().hex[:16]
        async with engine.connect() as connection:
            transaction = await connection.begin()
            try:
                await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
                await connection.execute(text(f'SET LOCAL search_path = "{schema}", public'))

                def migrate(sync):
                    sync = sync.execution_options(schema_translate_map={None: schema})
                    Base.metadata.create_all(sync, tables=[User.__table__], checkfirst=True)
                    create_occurrence_table(sync)
                    apply_r4_2(sync)
                    apply_r4_2_proof(sync)
                    apply_r4_3(sync)
                    apply_r4_4(sync)
                    if shape == "fresh":
                        apply_r5_1(sync)
                        apply_r5_2(sync)
                        apply_r5_3(sync)
                        Base.metadata.create_all(sync, tables=[ActionExecution.__table__],
                                                 checkfirst=True)
                        apply_r5_4(sync)
                        Base.metadata.create_all(sync, checkfirst=True)
                    else:
                        Base.metadata.create_all(sync, tables=[ActionExecution.__table__,
                                                               TranscriptProjection.__table__],
                                                 checkfirst=True)
                        sync.execute(text(
                            "INSERT INTO users (id,telegram_id,created_at,updated_at) "
                            "VALUES ('r54-user','r54-synthetic',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"))
                        sync.execute(text(
                            "INSERT INTO telegram_inbound_receipts "
                            "(id,bot_identity,update_id,state,attempts,received_at) "
                            "VALUES ('r54-receipt',1,54,'PROCESSED',1,CURRENT_TIMESTAMP)"))
                        sync.execute(text(
                            "INSERT INTO transcript_projection "
                            "(id,user_id,thread_id,message_id,group_id,role,occurred_at,"
                            "search_text,source_ref,content_sha256) VALUES "
                            "('r54-transcript','r54-user','r54-thread','r54-message',"
                            "'r54-group','human',CURRENT_TIMESTAMP,'synthetic','checkpoint:r54',"
                            "'synthetic-sha')"))
                        if shape != "pre_r5_1":
                            apply_r5_1(sync)
                            sync.execute(text(
                                "INSERT INTO calendar_action_evidence "
                                "(action_id,user_id,operation,account_identity,calendar_id,event_id,"
                                "approved_args_fingerprint,approved_baseline_etag,desired_postcondition) "
                                "VALUES ('r54-action','r54-user','update_event','normal',"
                                "'test-calendar','test-event','synthetic-fingerprint','\"v1\"','{}')"))
                        if shape == "r5_2":
                            apply_r5_2(sync)
                        apply_r5_1(sync)
                        apply_r5_2(sync)
                        apply_r5_3(sync)
                        apply_r5_4(sync)
                        Base.metadata.create_all(sync, checkfirst=True)
                    # Every frozen revision can run twice without changing
                    # existing R3/R4 relations or the historical action row.
                    for revision in (apply_r5_1, apply_r5_2, apply_r5_3, apply_r5_4):
                        revision(sync)
                    assert {"immutable_body", "private_action_marker",
                            "confirmed_etag"} <= _columns(sync)
                    assert sync.scalar(text(
                        "SELECT count(*) FROM app_schema_migrations "
                        "WHERE revision LIKE '20261007_r5_%'")) == 4
                    assert sync.scalar(text(
                        "SELECT count(*) FROM information_schema.tables "
                        "WHERE table_schema = current_schema() "
                        "AND table_name = 'calendar_dependent_continuations'")) == 1
                    if shape in {"r5_1", "r5_2"}:
                        assert sync.scalar(text(
                            "SELECT approved_baseline_etag FROM calendar_action_evidence "
                            "WHERE action_id='r54-action'")) == '"v1"'
                    assert sync.scalar(text(
                        "SELECT count(*) FROM telegram_inbound_receipts")) == (
                            0 if shape == "fresh" else 1)
                    if shape != "fresh":
                        assert sync.scalar(text(
                            "SELECT search_text FROM transcript_projection "
                            "WHERE id='r54-transcript'")) == "synthetic"

                await connection.run_sync(migrate)
            finally:
                # Schema and all synthetic rows never escape this transaction.
                await transaction.rollback()
