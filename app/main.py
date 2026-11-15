"""
FastAPI application entrypoint.
Handles startup (DB init, graph build, scheduler) and the Telegram webhook.
"""
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
import psycopg

from fastapi import FastAPI, Response, status
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db.session import init_db, AsyncSessionLocal
from app.agent.official_runtime import build_graph, clear_graph_runtime, close_legacy_reader
from app.agent.graph import DurableGraphInitializationError
from app.agent.nodes import LLMConfigurationError
from app.agent.store import generate_store
from app.localization import bind_profile_store
from app.agent.workflow_runtime import (build_productivity_runtime, clear_productivity_runtime,
                                        get_calendar_create_coordinator,
                                        get_calendar_modify_coordinator)
from app.telegram.webhook import router as webhook_router, webhook_secret_token
from app.scheduler.reminders import (scheduler, start_scheduler, stop_scheduler,
                                     pause_scheduler, bind_proactive_runtime,
                                     render_reminder_occurrence)
from app.scheduler.daily_brief import register_daily_brief, render_daily_occurrence
from app.scheduler.rundown import synthesize_order
from app.scheduler.proactive import ProactiveRuntime
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.runtime import RuntimeCapabilities
from app.agent.memory_runtime import create_memory_runtime
from app.agent.provider_clients import ProviderClientOwner, bind_process_owner
from app.telegram.durable_runtime import DurableTelegramRuntime
from app.web.api import router as web_router
from app.web.todos import router as web_todos_router
from app.web.reminders import router as web_reminders_router
from app.web.calendar import router as web_calendar_router
from app.web.today import router as web_today_router
from app.web.auth import validate_web_configuration
from app.web.static import WorkbenchStaticFiles
from app.db.models import WebCredential, WebRequest
from sqlalchemy import func, select
from app.trace import diagnostic_error


MEMORY_MAINTENANCE_INTERVAL_SECONDS = 120

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("Starting up Personal Assistant...")
    logger.info("Application event loop: %s", type(asyncio.get_running_loop()).__name__)
    runtime = RuntimeCapabilities()
    runtime.register("telegram_ingress", "REQUIRED")
    runtime.register("interactive_outbox", "REQUIRED")
    runtime.register("proactive_delivery", "REQUIRED")
    runtime.register("messaging_retention", "REQUIRED")
    runtime.register("calendar_continuation", "REQUIRED")
    if not settings.scheduler_enabled:
        # Interactive Telegram ingress remains usable without opting into
        # proactive sends. Readiness still exposes the disabled components.
        runtime.roles["scheduler"] = "DEGRADED_ALLOWED"
        runtime.roles["proactive_delivery"] = "DEGRADED_ALLOWED"
    if getattr(settings, "web_enabled", False):
        # Web direct business APIs need their own authentication readiness.
        # Agent execution remains exclusively on Telegram Receipt workers.
        runtime.register("web_auth", "DEGRADED_ALLOWED")
    app.state.runtime = runtime
    app.state.telegram_durable = None
    app.state.web_ready = False
    provider_owner = ProviderClientOwner()
    bind_process_owner(provider_owner)
    clear_graph_runtime()
    clear_productivity_runtime()
    workflow_conn = None
    agent_graph = None
    store_cm = None
    store = None
    memory_runtime = None
    memory_enabled = bool(getattr(settings, "memory_enabled", False))
    telegram_durable = None
    proactive_runtime = None
    maintenance_tasks: set[asyncio.Task] = set()
    calendar_recovery_stop = asyncio.Event()
    memory_maintenance_stop = asyncio.Event()

    async def calendar_action_recovery_loop(coordinator, *, required_component: str | None = None) -> None:
        # The DB evidence/lease remains authority; this loop is only wakeup.
        while not calendar_recovery_stop.is_set():
            try:
                await coordinator.recover_due()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("runtime component=calendar_action_recovery stage=sweep category=%s",
                               diagnostic_error(exc))
                if required_component:
                    runtime.mark_failed(required_component, "recovery", exc)
            try:
                await asyncio.wait_for(calendar_recovery_stop.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                pass

    async def memory_maintenance_job():
        # APScheduler owns the wakeup; lifespan owns the active provider/DB
        # call so clients cannot close while maintenance is still running.
        task = asyncio.create_task(memory_runtime.maintenance_tick())
        maintenance_tasks.add(task)
        try:
            return await task
        finally:
            if task.done():
                maintenance_tasks.discard(task)

    async def memory_maintenance_loop():
        # Memory indexing/consolidation is independent of proactive Telegram
        # delivery. A disabled reminder scheduler must not silently disable
        # long-term memory writes.
        while not memory_maintenance_stop.is_set():
            try:
                await asyncio.wait_for(memory_maintenance_stop.wait(),
                                       timeout=MEMORY_MAINTENANCE_INTERVAL_SECONDS)
                break
            except asyncio.TimeoutError:
                pass
            try:
                await memory_maintenance_job()
                runtime.mark_ready("memory_maintenance")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                runtime.mark_failed("memory_maintenance", "tick", exc)

    async def drain_maintenance(deadline: float) -> None:
        tasks = {task for task in maintenance_tasks if not task.done()}
        if tasks:
            _, pending = await asyncio.wait(
                tasks, timeout=max(0.0, deadline - asyncio.get_running_loop().time()))
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # 1. Init database tables
    try:
        await init_db()
        runtime.mark_ready("database")
    except Exception as e:
        runtime.mark_failed("database", "initialize", e)

    # A restored database must reapply the current terminal privacy policy
    # before any worker can claim or dispatch a message. Deletion suppression
    # from a backup older than the deletion requires external restore input.
    if runtime.components["database"]:
        try:
            from app.telegram.retention import redact_due_payloads
            await redact_due_payloads(
                AsyncSessionLocal,
                retention_days=settings.messaging_terminal_retention_days)
            runtime.mark_ready("messaging_retention")
        except Exception as exc:
            runtime.mark_failed("messaging_retention", "startup", exc)
    else:
        runtime.mark_failed("messaging_retention", "dependency", "DatabaseUnavailable")

    # Establish the authenticated bot's stable identity before model/Store
    # construction. The later recovery start reuses this verified identity;
    # it never derives the receipt namespace from a token or username.
    if (settings.telegram_bot_token
            and not getattr(settings, "telegram_live_start_preflight_ack", False)):
        runtime.mark_failed("telegram_ingress", "configuration", "PreflightNotAcknowledged")
    if (runtime.components["database"] and settings.telegram_bot_token
            and getattr(settings, "telegram_live_start_preflight_ack", False)):
        try:
            from app.telegram.sender import get_bot
            telegram_durable = DurableTelegramRuntime(
                AsyncSessionLocal, get_bot(),
                retention_days=settings.messaging_terminal_retention_days,
                on_recovery_failure=lambda exc: (
                    runtime.mark_failed("telegram_ingress", "recovery", exc),
                    runtime.mark_failed("interactive_outbox", "recovery", exc),
                    runtime.mark_failed("messaging_retention", "recovery", exc)))
            await telegram_durable.verify_bot_identity()
        except Exception as exc:
            runtime.mark_failed("telegram_ingress", "bot_identity", exc)

    # 2. Store is a required authority for per-user profile/language.
    if runtime.components["database"]:
        try:
            store_cm = generate_store()
            store = await store_cm.__aenter__()
            bind_profile_store(store)
            runtime.mark_ready("store")
        except Exception as exc:
            runtime.mark_failed("store", "initialize", exc)
    else:
        runtime.mark_failed("store", "dependency", "DatabaseUnavailable")

    # The active source-only mode needs Store and Transcript storage, not an
    # embedding service or background extraction. Legacy R3 is explicit.
    if memory_enabled:
        runtime.register("memory_core", "REQUIRED")
        if settings.memory_mode == "legacy_r3":
            runtime.register("memory_semantic", "DEGRADED_ALLOWED")
            runtime.register("memory_maintenance", "OPTIONAL_BACKGROUND")
        if runtime.components["database"] and runtime.components["store"]:
            try:
                memory_runtime = create_memory_runtime(
                    session_factory=AsyncSessionLocal, settings=settings)
                await memory_runtime.preflight_storage()
                runtime.mark_ready("memory_core")
                if settings.memory_mode == "legacy_r3":
                    try:
                        await memory_runtime.preflight_embedding()
                        if memory_runtime.vector_backend == "pgvector_halfvec_exact":
                            runtime.mark_ready("memory_semantic")
                        else:
                            runtime.mark_failed("memory_semantic", "backend",
                                                "BoundedFallbackOnly")
                    except Exception as exc:
                        runtime.mark_failed("memory_semantic", "synthetic_probe", exc)
            except Exception as exc:
                runtime.mark_failed("memory_core", "initialize", exc)
        else:
            runtime.mark_failed("memory_core", "dependency", "DatabaseOrStoreUnavailable")

    if (memory_runtime is not None and runtime.components["memory_core"] and
            settings.memory_mode == "legacy_r3"):
        memory_task = asyncio.create_task(memory_maintenance_loop())
        maintenance_tasks.add(memory_task)
        memory_task.add_done_callback(maintenance_tasks.discard)
        runtime.mark_ready("memory_maintenance")

    # 3. Ordinary conversation graph must retain its PostgreSQL Saver.
    if runtime.components["store"] and (not memory_enabled or runtime.components["memory_core"]):
        try:
            agent_graph = await build_graph(
                db_session_factory=AsyncSessionLocal, store=store,
                **({"memory_runtime": memory_runtime} if memory_runtime else {}))
            # A development-labeled process can still serve the real Bot or
            # Mini App. Gate the cutover by live entry availability, not by
            # APP_ENV alone, so old pending approvals are never orphaned.
            if (getattr(settings, "telegram_live_start_preflight_ack", False) or
                    getattr(settings, "web_enabled", False)):
                from app.agent.official_runtime import assert_legacy_approvals_cleared
                await assert_legacy_approvals_cleared(AsyncSessionLocal)
            if agent_graph.checkpointer is None:
                raise DurableGraphInitializationError("graph_verify", "MissingCheckpointer")
            runtime.mark_ready("agent_checkpointer")
            runtime.mark_ready("agent_graph")
        except DurableGraphInitializationError as exc:
            runtime.mark_failed("agent_checkpointer", exc.stage, exc.category)
            runtime.mark_failed("agent_graph", "dependency", "CheckpointerUnavailable")
        except LLMConfigurationError as exc:
            runtime.mark_failed("agent_graph", "llm_config", exc)
            runtime.mark_failed("agent_checkpointer", "not_initialized", "GraphUnavailable")
        except Exception as exc:
            runtime.mark_failed("agent_graph", "build", exc)
            runtime.mark_failed("agent_checkpointer", "not_initialized", "GraphUnavailable")
    else:
        runtime.mark_failed("agent_checkpointer", "dependency", "StoreUnavailable")
        runtime.mark_failed("agent_graph", "dependency", "StoreUnavailable")

    # 4. Typed HITL workflow has a separate Saver connection. Verify it too.
    if (runtime.components["database"] and runtime.components["store"] and
            (not memory_enabled or runtime.components["memory_core"])):
        stage = "saver_connect"
        try:
            workflow_conn = await psycopg.AsyncConnection.connect(
                settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://"),
                autocommit=True,
            )
            stage = "saver_construct"
            workflow_checkpointer = AsyncPostgresSaver(workflow_conn)
            stage = "saver_setup"
            await workflow_checkpointer.setup()
            runtime.mark_ready("workflow_checkpointer")
            stage = "workflow_build"
            workflow_graph = await build_productivity_runtime(
                AsyncSessionLocal, workflow_checkpointer, store=store,
                calendar_owner_telegram_id=settings.google_calendar_owner_telegram_id,
                **({"memory_orchestrator": memory_runtime.orchestrator} if memory_runtime else {}))
            if workflow_graph.checkpointer is None:
                raise RuntimeError("Typed workflow compiled without a checkpointer")
            from app.agent.workflow_runtime import get_calendar_continuation_runtime
            continuation_runtime = get_calendar_continuation_runtime()
            if continuation_runtime is None:
                raise RuntimeError("CalendarContinuationUnavailable")
            await continuation_runtime.preflight()
            runtime.mark_ready("calendar_continuation")
            runtime.mark_ready("productivity_workflow")
            for coordinator in (get_calendar_create_coordinator(),
                                get_calendar_modify_coordinator()):
                if coordinator is None:
                    continue
                recovery_task = asyncio.create_task(calendar_action_recovery_loop(coordinator))
                maintenance_tasks.add(recovery_task)
                recovery_task.add_done_callback(maintenance_tasks.discard)
            recovery_task = asyncio.create_task(
                calendar_action_recovery_loop(
                    continuation_runtime, required_component="calendar_continuation"))
            maintenance_tasks.add(recovery_task)
            recovery_task.add_done_callback(maintenance_tasks.discard)
            if (getattr(settings, "google_calendar_enabled", False) and
                    getattr(settings, "google_calendar_owner_telegram_id", "")):
                # Only opted-in users can produce jobs. This worker never sends
                # Telegram messages and does not backfill older Todos.
                from app.agent.todo_calendar_sync import TodoCalendarSync
                from app.agent.workflow_runtime import get_calendar_adapter
                sync = TodoCalendarSync(AsyncSessionLocal, get_calendar_adapter(),
                                        settings.google_calendar_owner_telegram_id)
                sync_task = asyncio.create_task(sync.run(calendar_recovery_stop))
                maintenance_tasks.add(sync_task)
                sync_task.add_done_callback(maintenance_tasks.discard)
        except Exception as exc:
            runtime.mark_failed("calendar_continuation", stage, exc)
            runtime.mark_failed("productivity_workflow", stage, exc)
            if workflow_conn is not None:
                await workflow_conn.close()
                workflow_conn = None
            runtime.mark_failed("workflow_checkpointer", stage, exc)
            clear_productivity_runtime()
    else:
        runtime.mark_failed("calendar_continuation", "dependency", "StoreOrDatabaseUnavailable")
        runtime.mark_failed("workflow_checkpointer", "dependency", "StoreOrDatabaseUnavailable")
        runtime.mark_failed("productivity_workflow", "dependency", "StoreOrDatabaseUnavailable")

    # Web admission only needs the bound credential and shared dependencies.
    # Historical Web approval requests are reported, never resumed here.
    if getattr(settings, "web_enabled", False):
        try:
            validate_web_configuration(settings)
            if not all(runtime.components.get(name, False) for name in (
                    "database", "store", "agent_graph", "productivity_workflow")):
                raise RuntimeError("WebGraphDependencyUnavailable")
            async with AsyncSessionLocal() as db:
                bound = await db.scalar(select(WebCredential.user_id).limit(1))
                legacy_pending = await db.scalar(select(func.count()).select_from(WebRequest).where(
                    WebRequest.operation.in_(("approval", "edit_instruction")),
                    WebRequest.state.in_(("ACCEPTED", "RUNNING", "WAITING_EXTERNAL"))))
            if not bound:
                raise RuntimeError("WebCredentialNotInitialized")
            if legacy_pending:
                logger.warning("retired_web_approvals stage=startup pending=%d", legacy_pending)
            runtime.mark_ready("web_auth")
            app.state.web_ready = True
        except Exception as exc:
            runtime.mark_failed("web_auth", "initialize", exc)
            app.state.web_ready = False

    # 5. Preflight and recover proactive custody before enabling scheduler
    # wakeups. Telegram send failures remain durable backlog, not lost jobs.
    if not settings.scheduler_enabled:
        runtime.mark_failed("scheduler", "configuration", "Disabled")
        runtime.mark_failed("proactive_delivery", "configuration", "Disabled")
    elif not (runtime.components["database"] and runtime.components["messaging_retention"]
              and runtime.components["store"]
              and telegram_durable is not None and telegram_durable.bot_identity is not None):
        runtime.mark_failed("scheduler", "dependency", "DatabaseStoreOrTelegramUnavailable")
        runtime.mark_failed("proactive_delivery", "dependency", "DatabaseStoreOrTelegramUnavailable")
    else:
        try:
            from app.agent.workflow_runtime import get_calendar_adapter
            from app.telegram.sender import get_bot
            def chat_scope(value: str) -> frozenset[str] | None:
                return frozenset(part.strip() for part in value.split(",")
                                 if part.strip()) if value.strip() else None
            proactive_runtime = ProactiveRuntime(
                AsyncSessionLocal, get_bot(), bot_identity=telegram_durable.bot_identity,
                timezone_name=settings.default_timezone, daily_hour=settings.daily_brief_hour,
                daily_renderer=lambda occurrence: render_daily_occurrence(
                    occurrence, calendar=get_calendar_adapter(),
                    synthesizer=synthesize_order,
                    calendar_owner_telegram_id=settings.google_calendar_owner_telegram_id),
                reminder_renderer=render_reminder_occurrence,
                daily_enabled=getattr(settings, "daily_rundown_enabled", False),
                reminder_allowed_chat_ids=chat_scope(
                    settings.reminder_allowed_telegram_ids),
                daily_allowed_chat_ids=chat_scope(
                    settings.daily_rundown_allowed_telegram_ids),
                on_failure=lambda exc: (
                    runtime.mark_failed("proactive_delivery", "recovery", exc),
                    runtime.mark_failed("scheduler", "dependency", "ProactiveRecoveryUnavailable")),
            )
            await proactive_runtime.start()
            bind_proactive_runtime(proactive_runtime)
            runtime.mark_ready("proactive_delivery")
            start_scheduler()
            if getattr(settings, "daily_rundown_enabled", False):
                register_daily_brief(scheduler)
            runtime.mark_ready("scheduler")
        except Exception as exc:
            runtime.mark_failed("scheduler", "initialize", exc)
            runtime.mark_failed("proactive_delivery", "initialize", exc)
            try:
                stop_scheduler()
            except Exception as stop_exc:
                logger.warning("runtime component=scheduler stage=cleanup category=%s",
                               type(stop_exc).__name__)
            bind_proactive_runtime(None)
            if proactive_runtime is not None:
                await proactive_runtime.stop()
                proactive_runtime = None

    # 6. Verified bot identity, committed receipt/outbox schema and an active
    # recovery loop are required before this process can ACK a webhook.
    if (runtime.components["database"] and runtime.components["messaging_retention"]
            and telegram_durable is not None
            and telegram_durable.bot_identity is not None):
        try:
            await telegram_durable.start()
            app.state.telegram_durable = telegram_durable
            runtime.mark_ready("telegram_ingress")
            runtime.mark_ready("interactive_outbox")
        except Exception as exc:
            runtime.mark_failed("telegram_ingress", "initialize", exc)
            runtime.mark_failed("interactive_outbox", "initialize", exc)
            if telegram_durable is not None:
                await telegram_durable.stop()
                telegram_durable = None
    else:
        runtime.mark_failed("telegram_ingress", "dependency", "DatabaseOrTelegramUnavailable")
        runtime.mark_failed("interactive_outbox", "dependency", "DatabaseOrTelegramUnavailable")

    # 7. Register only after *all* local durable capabilities are ready.
    if not settings.telegram_bot_token or not settings.telegram_webhook_url:
        runtime.mark_failed("telegram_webhook", "configuration", "NotConfigured")
    elif not runtime.required_ready_except("telegram_webhook"):
        runtime.mark_failed("telegram_webhook", "preflight", "RuntimeUnavailable")
    else:
        try:
            from app.telegram.sender import get_bot
            await get_bot().set_webhook(url=settings.telegram_webhook_url,
                                        secret_token=webhook_secret_token(),
                                        read_timeout=20, connect_timeout=20)
            runtime.mark_ready("telegram_webhook")
        except Exception as exc:
            runtime.mark_failed("telegram_webhook", "register", exc)

    logger.info("runtime overall_status=%s recovery=restart_required",
                "ready" if runtime.ready else "degraded")


    try:
        yield
    finally:
        # Provider HTTP resources must close before the application loop exits,
        # even when a request raises during lifespan teardown.
        try:
            logger.info("Shutting down...")
            # Close public admission first; a request already inside its
            # admission lock may commit custody, but no later one may enter.
            runtime.mark_failed("telegram_webhook", "shutdown", "Stopping")
            runtime.mark_failed("telegram_ingress", "shutdown", "Stopping")
            runtime.mark_failed("interactive_outbox", "shutdown", "Stopping")
            try:
                pause_scheduler()
            except Exception as exc:
                logger.warning("runtime component=scheduler stage=shutdown category=%s",
                               type(exc).__name__)
            runtime.mark_failed("scheduler", "shutdown", "Stopping")
            runtime.mark_failed("proactive_delivery", "shutdown", "Stopping")
            runtime.mark_failed("calendar_continuation", "shutdown", "Stopping")
            app.state.web_ready = False
            if getattr(settings, "web_enabled", False):
                runtime.mark_failed("web_auth", "shutdown", "Stopping")
            bind_proactive_runtime(None)
            calendar_recovery_stop.set()
            memory_maintenance_stop.set()
            # Quiesce new claims before draining. APScheduler's coroutine-job
            # cancellation does not own the shielded proactive tick task.
            await asyncio.gather(*(
                runtime_worker.quiesce() for runtime_worker in
                (telegram_durable, proactive_runtime) if runtime_worker is not None))
            drain_seconds = getattr(settings, "shutdown_drain_seconds", 30.0)
            deadline = asyncio.get_running_loop().time() + drain_seconds
            await asyncio.gather(
                *(runtime_worker.drain(deadline) for runtime_worker in
                  (telegram_durable, proactive_runtime) if runtime_worker is not None),
                drain_maintenance(deadline))
            try:
                stop_scheduler()
            except Exception as exc:
                logger.warning("runtime component=scheduler stage=close category=%s",
                               type(exc).__name__)
            app.state.telegram_durable = None
            await close_legacy_reader()
            clear_graph_runtime()
            clear_productivity_runtime()
            bind_profile_store(None)
            try:
                if store is not None and store_cm is not None:
                    await store_cm.__aexit__(None, None, None)
                    logger.info("Custom store shut down.")
            except Exception as exc:
                logger.warning("runtime component=store stage=shutdown category=%s",
                               type(exc).__name__)
            if workflow_conn is not None:
                await workflow_conn.close()
            if agent_graph is not None and agent_graph.checkpointer is not None:
                await agent_graph.checkpointer.conn.close()
            if memory_runtime is not None:
                await memory_runtime.aclose()
        finally:
            bind_process_owner(None)
            try:
                await provider_owner.aclose()
            finally:
                from app.telegram.sender import close_bot
                await close_bot()

app = FastAPI(
    title="Personal Assistant",
    description="Telegram-based personal assistant with AI-powered todo management",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(webhook_router)
app.include_router(web_router)
app.include_router(web_todos_router)
app.include_router(web_reminders_router)
app.include_router(web_calendar_router)
app.include_router(web_today_router)
_web_dist = Path(__file__).resolve().parents[1] / "web" / "dist"
if settings.web_enabled and _web_dist.is_dir():
    # Static assets contain no private data; every business API remains
    # session/owner scoped. One origin keeps Secure cookies and CSRF coherent.
    app.mount("/workbench", WorkbenchStaticFiles(directory=_web_dist, html=True), name="workbench")

@app.get("/")
async def root():
    return {"status": "ok", "service": "personal-assistant"}

@app.get("/health")
async def health():
    runtime = getattr(app.state, "runtime", RuntimeCapabilities())
    payload = runtime.snapshot()
    return JSONResponse(
        {"status": ("ready" if payload["full_ready"] else "degraded")
         if runtime.ready else "not_ready", **payload},
        status_code=status.HTTP_200_OK if runtime.ready else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@app.get("/health/messaging")
async def messaging_health():
    """Aggregate-only R4 custody visibility; never expose recipient or payload."""
    runtime = getattr(app.state, "runtime", RuntimeCapabilities())
    if not runtime.ready:
        return JSONResponse(
            {"status": "dependency_not_ready", "runtime_ready": False,
             "capabilities": runtime.snapshot()},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    try:
        from app.telegram.operational_snapshot import messaging_snapshot
        snapshot = await messaging_snapshot(AsyncSessionLocal)
    except Exception as exc:
        # A live process whose custody store cannot be queried cannot claim
        # messaging observability/availability. Never return SQL or payloads.
        logger.warning("messaging_snapshot stage=query category=%s", diagnostic_error(exc))
        return JSONResponse(
            {"status": "dependency_not_ready", "runtime_ready": False,
             "error_category": "StatusQueryUnavailable"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return JSONResponse({"status": snapshot["status"], "runtime_ready": True,
                         "messaging": snapshot})

@app.head("/")
async def health_head():
    return Response(status_code=200)
