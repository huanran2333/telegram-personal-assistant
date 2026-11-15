"""Fault injection for the production durability/readiness boundary."""
from contextlib import asynccontextmanager
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app import main
from app.agent import graph as agent_graph
from app.runtime import RuntimeCapabilities
from app.telegram import webhook


class FakeConnection:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class FakeSaver:
    def __init__(self, conn):
        self.conn = conn

    async def setup(self):
        return None


class FakeBot:
    def __init__(self):
        self.registrations = []

    async def set_webhook(self, **kwargs):
        self.registrations.append(kwargs)


class FakeDurableTelegramRuntime:
    """Explicit synthetic dependency for R1 startup tests, not a fallback."""
    def __init__(self, _session_factory, _bot, *, on_recovery_failure=None,
                 retention_days=30):
        self.bot_identity = None
        self.is_running = False
        self.admission_lock = asyncio.Lock()

    async def verify_bot_identity(self):
        self.bot_identity = 123456
        return self.bot_identity

    async def start(self):
        self.is_running = True

    async def stop(self):
        self.is_running = False

    async def quiesce(self):
        self.is_running = False

    async def drain(self, deadline):
        return None


class FakeProactiveRuntime:
    """Explicit R4.3 startup dependency injection for R1 fault tests."""
    def __init__(self, *_args, **_kwargs):
        self.running = False

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False

    async def quiesce(self):
        self.running = False

    async def drain(self, deadline):
        return None


@pytest.fixture
def startup_harness(monkeypatch):
    calls = {"scheduler_start": 0, "scheduler_pause": 0, "scheduler_stop": 0,
             "agent_build": 0, "workflow_build": 0}
    bot = FakeBot()
    connections = []

    async def init_db():
        return None

    async def redact_due_payloads(*args, **kwargs):
        return SimpleNamespace(receipts=0, deliveries=0)

    @asynccontextmanager
    async def store():
        yield object()

    async def connect(*args, **kwargs):
        conn = FakeConnection()
        connections.append(conn)
        return conn

    async def build_graph(**kwargs):
        calls["agent_build"] += 1
        return SimpleNamespace(checkpointer=FakeSaver(FakeConnection()))

    async def build_workflow(*args, **kwargs):
        calls["workflow_build"] += 1
        return SimpleNamespace(checkpointer=args[1])

    async def recover_continuations():
        return 0

    async def preflight_continuations():
        return None

    def start_scheduler():
        calls["scheduler_start"] += 1

    def stop_scheduler():
        calls["scheduler_stop"] += 1

    def pause_scheduler():
        calls["scheduler_pause"] += 1

    monkeypatch.setattr(main, "init_db", init_db)
    monkeypatch.setattr("app.telegram.retention.redact_due_payloads", redact_due_payloads)
    monkeypatch.setattr(main, "generate_store", store)
    monkeypatch.setattr(main.psycopg.AsyncConnection, "connect", connect)
    monkeypatch.setattr(main, "AsyncPostgresSaver", FakeSaver)
    monkeypatch.setattr(main, "build_graph", build_graph)
    # This fixture supplies a synthetic graph, not a PostgreSQL Saver with
    # legacy checkpoints. Cutover readiness is verified in its own tests.
    monkeypatch.setattr(
        "app.agent.official_runtime.assert_legacy_approvals_cleared",
        AsyncMock(return_value=None))
    monkeypatch.setattr(main, "build_productivity_runtime", build_workflow)
    monkeypatch.setattr("app.agent.workflow_runtime.get_calendar_continuation_runtime",
                        lambda: SimpleNamespace(recover_due=recover_continuations,
                                                preflight=preflight_continuations))
    monkeypatch.setattr(main, "start_scheduler", start_scheduler)
    monkeypatch.setattr(main, "stop_scheduler", stop_scheduler)
    monkeypatch.setattr(main, "pause_scheduler", pause_scheduler)
    monkeypatch.setattr(main, "register_daily_brief", lambda scheduler: None)
    monkeypatch.setattr(main, "webhook_secret_token", lambda: "synthetic-secret")
    monkeypatch.setattr(main, "DurableTelegramRuntime", FakeDurableTelegramRuntime)
    monkeypatch.setattr(main, "ProactiveRuntime", FakeProactiveRuntime)
    monkeypatch.setattr("app.telegram.sender.get_bot", lambda: bot)
    monkeypatch.setattr(main, "settings", SimpleNamespace(
        database_url_sync="postgresql://synthetic.invalid/db",
        google_calendar_owner_telegram_id="synthetic-owner",
        telegram_bot_token="synthetic-bot-token",
        telegram_webhook_url="https://example.invalid/webhook",
        telegram_live_start_preflight_ack=True,
        scheduler_enabled=True,
        reminder_allowed_telegram_ids="",
        daily_rundown_allowed_telegram_ids="",
        default_timezone="UTC",
        daily_brief_hour=8,
        messaging_terminal_retention_days=30,
    ))
    return calls, bot, connections


@pytest.mark.asyncio
async def test_live_entry_blocks_startup_when_old_approval_is_pending(
        startup_harness, monkeypatch):
    blocked = AsyncMock(side_effect=RuntimeError("LegacyApprovalsRequireCancellation"))
    monkeypatch.setattr(
        "app.agent.official_runtime.assert_legacy_approvals_cleared", blocked)
    async with main.lifespan(main.app):
        assert not main.app.state.runtime.ready
        assert not main.app.state.runtime.components["agent_graph"]
    blocked.assert_awaited_once()


@pytest.mark.asyncio
async def test_durable_startup_is_ready_only_after_all_capabilities(startup_harness):
    calls, bot, connections = startup_harness
    async with main.lifespan(main.app):
        runtime = main.app.state.runtime
        assert runtime.ready
        assert all(runtime.components.values())
        assert len(bot.registrations) == 1
        assert (await main.health()).status_code == 200
    assert connections[0].closed
    assert calls["scheduler_stop"] == 1
    assert calls["scheduler_pause"] == 1
    assert not main.app.state.runtime.ready


@pytest.mark.asyncio
async def test_continuation_preflight_failure_keeps_webhook_not_ready(startup_harness,
                                                                     monkeypatch):
    _, bot, _ = startup_harness

    async def fail_preflight():
        raise RuntimeError("synthetic continuation storage failure")

    async def recover():
        raise AssertionError("Recovery must not start after failed preflight")

    monkeypatch.setattr("app.agent.workflow_runtime.get_calendar_continuation_runtime",
                        lambda: SimpleNamespace(preflight=fail_preflight,
                                                recover_due=recover))
    async with main.lifespan(main.app):
        runtime = main.app.state.runtime
        assert not runtime.ready
        assert not runtime.components["calendar_continuation"]
        assert not runtime.components["productivity_workflow"]
        assert not bot.registrations


@pytest.mark.asyncio
async def test_shutdown_orders_ingress_scheduler_quiesce_drain_before_clients(
        startup_harness, monkeypatch):
    """A live task owns provider/DB resources through the drain boundary."""
    events = []
    drain_deadlines = []
    entered, leave, drain_started, release = (asyncio.Event() for _ in range(4))

    def pause_scheduler():
        events.append("scheduler_pause")

    def stop_scheduler():
        events.append("scheduler_stop")

    async def ingress_quiesce(self):
        assert not main.app.state.runtime.ready
        events.append("ingress_quiesce")
        self.is_running = False

    async def proactive_quiesce(self):
        events.append("proactive_quiesce")

    async def ingress_drain(self, deadline):
        events.append("ingress_drain")
        drain_deadlines.append(deadline)
        drain_started.set()
        await release.wait()

    async def proactive_drain(self, deadline):
        events.append("proactive_drain")
        drain_deadlines.append(deadline)

    class Owner:
        async def aclose(self):
            events.append("provider_close")

    async def close_bot():
        events.append("bot_close")

    monkeypatch.setattr(main, "stop_scheduler", stop_scheduler)
    monkeypatch.setattr(main, "pause_scheduler", pause_scheduler)
    monkeypatch.setattr(FakeDurableTelegramRuntime, "quiesce", ingress_quiesce)
    monkeypatch.setattr(FakeDurableTelegramRuntime, "drain", ingress_drain)
    monkeypatch.setattr(FakeProactiveRuntime, "quiesce", proactive_quiesce)
    monkeypatch.setattr(FakeProactiveRuntime, "drain", proactive_drain)
    monkeypatch.setattr(main, "ProviderClientOwner", Owner)
    monkeypatch.setattr("app.telegram.sender.close_bot", close_bot)

    async def lifespan_task():
        async with main.lifespan(main.app):
            entered.set()
            await leave.wait()

    task = asyncio.create_task(lifespan_task())
    await entered.wait()
    leave.set()
    await drain_started.wait()
    assert (await main.health()).status_code == 503
    assert events[:3] == ["scheduler_pause", "ingress_quiesce", "proactive_quiesce"]
    assert "provider_close" not in events and "bot_close" not in events
    release.set()
    await task
    assert events.index("ingress_drain") < events.index("provider_close")
    assert events.index("proactive_drain") < events.index("provider_close")
    assert events.index("ingress_drain") < events.index("scheduler_stop")
    assert events.index("proactive_drain") < events.index("scheduler_stop")
    assert events.index("scheduler_stop") < events.index("provider_close")
    assert events.index("provider_close") < events.index("bot_close")
    assert len(drain_deadlines) == 2 and drain_deadlines[0] == drain_deadlines[1]
    assert drain_deadlines[0] > asyncio.get_running_loop().time() - 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_stage,backend,expected_ready", [
    (None, "pgvector_halfvec_exact", True),
    ("embedding", "pgvector_halfvec_exact", True),
    ("storage", "pgvector_halfvec_exact", False),
    (None, "json_bounded", True),
])
async def test_memory_capabilities_are_explicit_and_never_silent_recent_only(
    startup_harness, monkeypatch, failed_stage, backend, expected_ready,
):
    _, bot, _ = startup_harness
    main.settings.memory_enabled = True
    main.settings.memory_mode = "legacy_r3"

    class Memory:
        orchestrator = object()
        vector_backend = backend
        closed = False

        async def preflight_storage(self):
            if failed_stage == "storage":
                raise RuntimeError("synthetic private body must not be logged")

        async def preflight_embedding(self):
            if failed_stage == "embedding":
                raise ConnectionError("synthetic private body must not be logged")

        async def maintenance_tick(self):
            return {"processed_threads": 0}

        async def aclose(self):
            self.closed = True

    memory = Memory()
    monkeypatch.setattr(main, "create_memory_runtime", lambda **_kwargs: memory)

    async with main.lifespan(main.app):
        runtime = main.app.state.runtime
        assert runtime.ready is expected_ready
        assert runtime.components["memory_core"] is (failed_stage != "storage")
        assert runtime.components["memory_semantic"] is (
            failed_stage is None and backend == "pgvector_halfvec_exact")
        assert runtime.roles["memory_semantic"] == "DEGRADED_ALLOWED"
        assert len(bot.registrations) == int(expected_ready)
        if failed_stage == "embedding" or backend == "json_bounded":
            assert runtime.full_ready is False
            assert (await main.health()).status_code == 200
        if failed_stage == "storage":
            assert (await main.health()).status_code == 503
            assert "synthetic private body" not in str(runtime.snapshot())
    assert memory.closed


@pytest.mark.asyncio
async def test_memory_maintenance_runs_without_proactive_scheduler(startup_harness, monkeypatch):
    calls, _, _ = startup_harness
    main.settings.memory_enabled = True
    main.settings.memory_mode = "legacy_r3"
    main.settings.scheduler_enabled = False
    monkeypatch.setattr(main, "MEMORY_MAINTENANCE_INTERVAL_SECONDS", 0.01)
    ticked = asyncio.Event()

    class Memory:
        orchestrator = object()
        vector_backend = "pgvector_halfvec_exact"

        async def preflight_storage(self):
            pass

        async def preflight_embedding(self):
            pass

        async def maintenance_tick(self):
            ticked.set()
            return {"processed_threads": 0}

        async def aclose(self):
            pass

    monkeypatch.setattr(main, "create_memory_runtime", lambda **_kwargs: Memory())
    async with main.lifespan(main.app):
        await asyncio.wait_for(ticked.wait(), timeout=1)
        assert main.app.state.runtime.components["memory_maintenance"] is True
        assert calls["scheduler_start"] == 0


@pytest.mark.asyncio
async def test_source_only_memory_starts_without_embedding_or_background_extraction(
        startup_harness, monkeypatch):
    _, bot, _ = startup_harness
    main.settings.memory_enabled = True
    main.settings.memory_mode = "transcript_lexical"

    class Memory:
        orchestrator = object()
        vector_backend = "transcript_lexical"

        async def preflight_storage(self):
            return None

        async def preflight_embedding(self):
            raise AssertionError("embedding must not be probed")

        async def maintenance_tick(self):
            raise AssertionError("LangMem maintenance must not start")

        async def aclose(self):
            return None

    monkeypatch.setattr(main, "create_memory_runtime", lambda **_kwargs: Memory())
    async with main.lifespan(main.app):
        runtime = main.app.state.runtime
        assert runtime.ready and runtime.full_ready
        assert "memory_semantic" not in runtime.components
        assert "memory_maintenance" not in runtime.components
        assert bot.registrations


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_component,stage", [
    ("agent_checkpointer", "saver_connect"),
    ("agent_checkpointer", "saver_setup"),
    ("agent_checkpointer", "graph_compile"),
])
async def test_agent_saver_failures_keep_process_live_and_webhook_unregistered(
    startup_harness, monkeypatch, failed_component, stage,
):
    calls, bot, _ = startup_harness

    async def failed_graph(**kwargs):
        calls["agent_build"] += 1
        raise agent_graph.DurableGraphInitializationError(stage, "SyntheticFailure")

    monkeypatch.setattr(main, "build_graph", failed_graph)
    async with main.lifespan(main.app):
        runtime = main.app.state.runtime
        assert not runtime.ready
        assert runtime.snapshot()["alive"] is True
        assert runtime.failures[failed_component] == {"stage": stage, "error_category": "SyntheticFailure"}
        assert runtime.components["productivity_workflow"] is True
        assert runtime.components["scheduler"] is True  # DB/Store-backed proactive path is independent.
        assert not bot.registrations
        assert (await main.health()).status_code == 503


@pytest.mark.asyncio
async def test_database_failure_skips_dependent_runtime(startup_harness, monkeypatch):
    calls, bot, _ = startup_harness

    async def fail_db():
        raise ConnectionError("postgresql://private:password@host/db")

    monkeypatch.setattr(main, "init_db", fail_db)
    async with main.lifespan(main.app):
        runtime = main.app.state.runtime
        assert runtime.failures["database"]["error_category"] == "ConnectionError"
        assert calls["agent_build"] == calls["workflow_build"] == calls["scheduler_start"] == 0
        assert not bot.registrations


@pytest.mark.asyncio
async def test_store_failure_skips_dependent_runtime(startup_harness, monkeypatch):
    calls, bot, _ = startup_harness

    @asynccontextmanager
    async def failed_store():
        raise RuntimeError("secret value must not be logged")
        yield

    monkeypatch.setattr(main, "generate_store", failed_store)
    async with main.lifespan(main.app):
        runtime = main.app.state.runtime
        assert runtime.components["database"]
        assert runtime.failures["store"]["error_category"] == "RuntimeError"
        assert calls["agent_build"] == calls["workflow_build"] == calls["scheduler_start"] == 0
        assert not bot.registrations


@pytest.mark.asyncio
async def test_workflow_failure_blocks_ingress_but_not_independent_scheduler(startup_harness, monkeypatch):
    calls, bot, connections = startup_harness

    async def failed_workflow(*args, **kwargs):
        calls["workflow_build"] += 1
        raise RuntimeError("private OAuth material")

    monkeypatch.setattr(main, "build_productivity_runtime", failed_workflow)
    async with main.lifespan(main.app):
        runtime = main.app.state.runtime
        assert not runtime.components["workflow_checkpointer"]  # connection was closed after build failure
        assert not runtime.components["productivity_workflow"]
        assert runtime.components["scheduler"]
        assert not runtime.ready
        assert not bot.registrations
        assert connections[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["saver_connect", "saver_construct", "saver_setup"])
async def test_workflow_saver_failures_block_readiness(startup_harness, monkeypatch, stage):
    calls, bot, connections = startup_harness
    original_connect = main.psycopg.AsyncConnection.connect
    original_saver = main.AsyncPostgresSaver

    async def fail_connect(*args, **kwargs):
        if stage == "saver_connect":
            raise ConnectionError("private connection string")
        return await original_connect(*args, **kwargs)

    class FailSaver(original_saver):
        def __init__(self, conn):
            if stage == "saver_construct":
                raise RuntimeError("private connection string")
            super().__init__(conn)

        async def setup(self):
            if stage == "saver_setup":
                raise RuntimeError("private connection string")

    monkeypatch.setattr(main.psycopg.AsyncConnection, "connect", fail_connect)
    monkeypatch.setattr(main, "AsyncPostgresSaver", FailSaver)
    async with main.lifespan(main.app):
        runtime = main.app.state.runtime
        assert runtime.failures["workflow_checkpointer"]["stage"] == stage
        assert runtime.components["agent_graph"]
        assert not runtime.ready
        assert not bot.registrations
        assert calls["workflow_build"] == 0
        if connections:
            assert connections[0].closed


@pytest.mark.asyncio
async def test_scheduler_failure_blocks_full_product_readiness(startup_harness, monkeypatch):
    calls, bot, _ = startup_harness

    def failed_scheduler():
        calls["scheduler_start"] += 1
        raise RuntimeError("private scheduler detail")

    monkeypatch.setattr(main, "start_scheduler", failed_scheduler)
    async with main.lifespan(main.app):
        assert main.app.state.runtime.failures["scheduler"]["error_category"] == "RuntimeError"
        assert calls["scheduler_stop"] == 1
        assert not main.app.state.runtime.ready
        assert not bot.registrations


@pytest.mark.asyncio
async def test_proactive_custody_failure_blocks_scheduler_and_webhook(startup_harness, monkeypatch):
    calls, bot, _ = startup_harness

    class FailedProactive(FakeProactiveRuntime):
        async def start(self):
            raise ConnectionError("postgresql://private-credential")

    monkeypatch.setattr(main, "ProactiveRuntime", FailedProactive)
    async with main.lifespan(main.app):
        runtime = main.app.state.runtime
        assert not runtime.ready
        assert runtime.failures["proactive_delivery"] == {
            "stage": "initialize", "error_category": "ConnectionError"}
        assert calls["scheduler_start"] == 0
        assert not bot.registrations
        assert "private-credential" not in str(runtime.snapshot())


@pytest.mark.asyncio
async def test_explicitly_disabled_scheduler_keeps_interactive_bot_ready(startup_harness, monkeypatch):
    calls, bot, _ = startup_harness
    monkeypatch.setattr(main.settings, "scheduler_enabled", False)
    async with main.lifespan(main.app):
        assert main.app.state.runtime.ready
        assert not main.app.state.runtime.full_ready
        assert main.app.state.runtime.roles["scheduler"] == "DEGRADED_ALLOWED"
        assert main.app.state.runtime.failures["scheduler"]["error_category"] == "Disabled"
        assert calls["scheduler_start"] == 0
        assert len(bot.registrations) == 1


@pytest.mark.asyncio
async def test_live_bot_start_requires_explicit_queue_preflight_ack(startup_harness, monkeypatch):
    calls, bot, _ = startup_harness
    monkeypatch.setattr(main.settings, "telegram_live_start_preflight_ack", False)
    async with main.lifespan(main.app):
        runtime = main.app.state.runtime
        assert not runtime.ready
        assert not runtime.components["telegram_ingress"]
        assert not runtime.components["interactive_outbox"]
        assert not runtime.components["scheduler"]
        assert not bot.registrations
        assert calls["scheduler_start"] == 0


@pytest.mark.asyncio
async def test_webhook_registration_failure_keeps_process_live_but_not_ready(startup_harness):
    calls, bot, _ = startup_harness

    async def failed_registration(**kwargs):
        raise ConnectionError("private bot credential")

    bot.set_webhook = failed_registration
    async with main.lifespan(main.app):
        runtime = main.app.state.runtime
        assert all(value for name, value in runtime.components.items() if name != "telegram_webhook")
        assert runtime.failures["telegram_webhook"] == {
            "stage": "register", "error_category": "ConnectionError"}
        assert (await main.health()).status_code == 503


@pytest.mark.asyncio
async def test_not_ready_ingress_does_not_read_update_or_execute_business(monkeypatch):
    main.app.state.runtime = RuntimeCapabilities()
    async def receive():
        raise AssertionError("not-ready update body must not be read")

    request = Request({"type": "http", "method": "POST", "path": "/webhook", "app": main.app,
                       "headers": [(b"x-telegram-bot-api-secret-token",
                                    webhook.webhook_secret_token().encode())]}, receive)
    with pytest.raises(HTTPException) as exc:
        await webhook.telegram_webhook(request)
    assert exc.value.status_code == 503
    assert exc.value.headers["Retry-After"] == "5"


@pytest.mark.asyncio
async def test_repaired_dependency_requires_restart_then_becomes_ready(startup_harness, monkeypatch):
    calls, bot, _ = startup_harness
    original_build = main.build_graph

    async def failed_graph(**kwargs):
        raise agent_graph.DurableGraphInitializationError("saver_setup", "SyntheticFailure")

    monkeypatch.setattr(main, "build_graph", failed_graph)
    async with main.lifespan(main.app):
        assert not main.app.state.runtime.ready
    monkeypatch.setattr(main, "build_graph", original_build)
    async with main.lifespan(main.app):
        assert main.app.state.runtime.ready
        assert len(bot.registrations) == 1


def test_runtime_snapshot_contains_only_safe_categories(caplog):
    state = RuntimeCapabilities()
    state.mark_failed("database", "connect", RuntimeError("postgresql://user:password@host"))
    rendered = str(state.snapshot()) + caplog.text
    assert "password" not in rendered
    assert state.snapshot()["failures"]["database"] == {
        "stage": "connect", "error_category": "RuntimeError"}
