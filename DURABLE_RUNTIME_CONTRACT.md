# R1 — Production Durable Runtime Contract

Status: implemented and fault-injection tested on 2026-10-04. Scope: startup, readiness, Telegram ingress and restart recovery only. No Conversation Memory, durable Telegram receipt, Rundown outbox, Calendar remote-ID reconciliation or new product workflow was implemented.

## Root cause and selected contract

**Before:** `app/agent/graph.py::build_graph` caught Saver connect/setup/compile errors and returned `builder.compile()` without a checkpointer. `app/main.py` then marked `agent=True`; `/health` tested only `database && agent`. The typed workflow and scheduler could be unavailable while `/health` said ready. Telegram registration happened before scheduler startup, and a previously registered webhook had no overall runtime guard.

**After:** this process may remain HTTP-live for diagnosis, but full-product readiness requires every configured production capability below. The ordinary graph's default path either compiles with a verified `AsyncPostgresSaver` or raises a stage-labelled error; it never silently downgrades. The typed workflow has a separate verified Saver. Startup registers the Telegram webhook only after local required capabilities are ready. A webhook registered by an earlier process is still protected: authenticated requests receive **503 before JSON is read, update ID is claimed, callback is answered, or any business path runs**. [Telegram's Bot API](https://core.telegram.org/bots/api#setwebhook) says non-2xx webhook requests are retried for a reasonable number of attempts, not forever; `Retry-After: 5` is advisory, not a delivery guarantee. R2's durable inbound receipt remains separate work.

| Capability | Ready condition | Depends on | Degraded behavior |
|---|---|---|---|
| `database` | SQLAlchemy `init_db()` succeeds | PostgreSQL | No Store, graph, workflow or scheduler startup |
| `store` | `AsyncPostgresStore` context and setup succeed | database | No Agent/workflow/scheduler; saved profile/language cannot silently default for production ingress |
| `agent_checkpointer` | ordinary `AsyncPostgresSaver` connects, constructs and sets up; compiled graph retains it | database, store | ordinary graph unavailable; no non-persistent production fallback |
| `agent_graph` | ordinary graph compiles with Saver | agent_checkpointer, Store, LLM configuration | no chat/Todo route exposed to Telegram |
| `workflow_checkpointer` | typed workflow Saver connects, constructs, sets up and remains open | database, store | no recoverable HITL thread |
| `productivity_workflow` | typed graph compiles with Saver | workflow_checkpointer, Store | Calendar/Gmail/cross-domain/HITL unavailable |
| `scheduler` | Reminder and Rundown registration/start succeed | database, store, Telegram token | full product not ready; no in-memory-only reminder substitute |
| `telegram_webhook` | configured URL/token and successful registration **after** all preceding components | all above | old webhook deliveries get 503; no normal business processing |

`/health` returns HTTP 200 only when all eight capabilities are true; otherwise 503. Its JSON separates `alive: true` from `ready: false`, includes booleans and safe `{stage,error_category}` values, and says `recovery: restart_required`. It never includes a DSN, token, raw exception message, Telegram payload or user content. This is one bounded capability model, not a general dependency framework. A running process is not proof that it can honor conversation continuity or resume an interrupted action. The current `/` and HEAD liveness routes remain available; no extra `/live`/`/ready` endpoints were added.

### Scheduler dependency choice

Reminder sweeps use persisted PostgreSQL Todo claims; Rundown uses DB Todo and Store language, with Calendar source errors disclosed. Neither needs the ordinary chat Saver to dispatch safely. Thus scheduler **may continue** when the ordinary graph fails, but overall `ready=false` and webhook rejects new work. It is **not** started without DB, Store or Telegram sender configuration. Because Reminder/Rundown are existing product capabilities, `SCHEDULER_ENABLED=false` deliberately leaves full-product readiness false; it is suitable for isolated local work, not a claim of complete deployment readiness. A Calendar MCP read failure does not itself invalidate durable-runtime readiness: the workflow remains durable and source-unavailable handling belongs to its domain policy.

### Failure matrix

Every row keeps the process alive and `/health` at 503. Startup errors are logged only as component, stage and exception *type*; shutdown clears graph holders so a later startup cannot expose a stale compiled graph or Calendar adapter.

| Injected failure | Capability state and request behavior | Scheduler | Recovery |
|---|---|---|---|
| DB unavailable | `database=false`; all dependents unavailable; old webhook gets 503 | not started | repair DB, restart |
| ordinary Saver connect/construct/setup | `agent_checkpointer=false`, `agent_graph=false`; no no-Saver compile | may run if DB/Store/Telegram are healthy | repair, restart |
| ordinary graph compile with Saver | same; failed connection closed | may run | repair, restart |
| Store init/setup | `store=false`; Agent/workflow/scheduler not started | not started | repair, restart |
| typed Saver connect/construct/setup | workflow capabilities false, connection closed; no callback resume | may run with Calendar source unavailable if its dependencies hold | repair, restart |
| typed workflow build/compile | workflow and its now-closed Saver marked unavailable | may run independently | repair, restart |
| scheduler init | `scheduler=false`; no webhook registration | stopped if partially started | repair, restart |
| Telegram registration | `telegram_webhook=false`; old endpoint still guards with 503 | may run independently | repair tunnel/API, restart |

The supported recovery path is **operator restart** after repairing the dependency. This avoids hot-swapping two graphs, Savers, Store context and in-flight Telegram callbacks in one process. Startup creates fresh runtime state; shutdown marks ingress unavailable before closing Store/Savers. There is no automatic reinitialization claim.

### Explicit non-production fallback

`build_graph(..., durable=False)` is available only when `APP_ENV` is `development` or `test`, selected by the caller for isolated evaluation. It does not attempt a Saver connection. The default `durable=True` always requires PostgreSQL Saver; a connect/setup/compile exception cannot enter the explicit branch. `APP_ENV=production` and other non-allowlisted environments reject non-durable construction. `tests/test_scoped_todo_binding.py` now explicitly selects this evaluation mode instead of relying on a simulated database outage to reach an implicit fallback. Production `main.lifespan` never passes `durable=False`.

## Evidence and limits

- `tests/test_graph_durability.py`: Saver connect, constructor, setup and compile failures, no fallback; explicit test/development path versus production rejection.
- `tests/test_durable_runtime.py`: normal capability completion, DB/Store/ordinary Saver/typed Saver/workflow/scheduler/webhook failures, no webhook registration, 503 before update-body read, restart-to-ready, safe diagnostics.
- `tests/test_health.py`: live versus ready HTTP response. Existing scoped-binding test updated to explicit evaluation mode.
- `tests/integration/postgres/test_pg_durable_runtime.py`: actual production `build_graph` with real PostgreSQL Saver **and** Store; a synthetic `HumanMessage` is persisted via `aupdate_state`, both connections closed, graph rebuilt with fresh Saver/Store connections, same thread restored, then deleted. No LLM, Calendar, Gmail or Telegram write is involved.
- Existing real-PostgreSQL `test_state_store.py` and `test_pg_trace_correlation.py` cover Store reconnect and pending HITL Reject across a Saver reconnect. These use synthetic requests and no external mutations.

Test counts and exact run commands belong to this Gate's final handoff. This work does **not** claim a live Telegram 503/retry experiment, a real PostgreSQL outage drill, or a new Calendar/Gmail E2E. Telegram's finite retry budget means prolonged downtime may still lose pending updates; durable inbound receipt is a separate remediation.

## INTERVIEW_EXPLANATION

1. **Liveness is not readiness.** The HTTP process can answer diagnostics while required persistence is unavailable. Reporting 200 ready would invite real user traffic that cannot meet the Assistant's recovery contract.
2. **Compile success is not durable Agent success.** A LangGraph object without a checkpointer can respond now but cannot restore the same thread after restart. The old catch-and-compile path hid that semantic change.
3. **Saver is a correctness dependency.** Chat continuity and, more critically, `interrupt`/`Command(resume)` depend on persisted thread state. A missing Saver is not a performance degradation: it can orphan a pending approval or break action continuity.
4. **No silent degraded mode.** A read-only or stateless mode would need a separately specified user contract and route policy. Keeping the old chat/Todo route active while Calendar/HITL is unavailable is not such a mode. The ingress guard fails closed for the complete Telegram product.
5. **Explicit test fallback is useful.** Unit tests may inspect tool binding without PostgreSQL. They should ask for an ephemeral graph deliberately, not simulate a production failure and hope it downgrades.
6. **Failure injection proves the contract.** Tests fail each initialization boundary, assert precise safe capability state, observe 503 before the update is parsed, and rebuild after a simulated operator restart. A real PostgreSQL test separately proves checkpoint persistence/reload. The three layers—unit fault injection, real DB persistence, and historical Telegram E2E—are not conflated.

`DURABLE_RUNTIME_CONTRACT_DECISION` is set in the final Gate result after all relevant tests complete, not inferred merely from this design document.
