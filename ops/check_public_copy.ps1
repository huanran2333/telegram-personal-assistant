param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Python)) { throw "Python virtual environment not found: $Python" }
# Safety overrides apply only to this process and its children. No live Bot,
# production database, Google OAuth or paid model call is needed by this suite.
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:APP_ENV = "development"
$env:LLM_PROVIDER = "openai"
$env:LLM_MODEL = "placeholder-model"
$env:OPENAI_BASE_URL = "http://127.0.0.1:65433/v1"
$env:OPENAI_API_KEY = "placeholder-not-a-secret"
$env:DATABASE_URL = "postgresql+asyncpg://release_example:invalid@127.0.0.1:65432/release_example"
$env:DATABASE_URL_SYNC = "postgresql://release_example:invalid@127.0.0.1:65432/release_example"
$env:TELEGRAM_BOT_TOKEN = ""
$env:TELEGRAM_WEBHOOK_URL = ""
$env:TELEGRAM_LIVE_START_PREFLIGHT_ACK = "false"
$env:SCHEDULER_ENABLED = "false"
$env:DAILY_RUNDOWN_ENABLED = "false"
$env:GOOGLE_CALENDAR_ENABLED = "false"
$env:GMAIL_ENABLED = "false"
$env:WEB_ENABLED = "false"
$env:TAVILY_API_KEY = ""
$suite = @(
    "tests/test_local_staging.py",
    "tests/test_durable_runtime.py",
    "tests/test_r4_2_durable_unit.py",
    "tests/test_official_cross_entry_versions.py",
    "tests/test_official_calendar_error_boundary.py",
    "tests/test_official_calendar_slot_boundary.py",
    "tests/test_official_guarded_tool_feedback.py",
    "tests/test_official_model_failover.py",
    "tests/test_official_preflight_read_feedback.py",
    "tests/test_official_read_retry.py",
    "tests/test_official_reminder_authorization.py",
    "tests/test_official_runtime_write_failure_probe.py",
    "tests/test_official_todo_create_intent.py",
    "tests/test_official_tool_normalization.py",
    "tests/test_r5_1_calendar_evidence.py",
    "tests/test_r5_3_calendar_modify.py",
    "tests/test_reminders.py",
    "tests/test_scoped_todo_binding.py",
    "tests/test_todo_calendar_transport.py",
    "tests/test_todos.py",
    "tests/test_web_boundary.py",
    "tests/test_web_calendar_read.py"
)
& $Python -m pytest -q --tb=short @suite
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Push-Location web
try {
    if (-not (Test-Path -LiteralPath "node_modules")) {
        throw "Install locked frontend dependencies first: npm ci"
    }
    npm test
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    npm run build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally { Pop-Location }