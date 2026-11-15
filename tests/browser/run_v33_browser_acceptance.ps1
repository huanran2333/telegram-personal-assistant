# Core and read-only real Calendar UI acceptance on a dedicated synthetic PG18.
# No provider write or Telegram sender is enabled. Never target the operator DB.
param([ValidateSet('calendar', 'core', 'g3')][string]$Mode = 'calendar')
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $repo
$localTools = (Resolve-Path '.local-tools').Path
$cluster = Join-Path $localTools 'v33_calendar_pg18_data'
$pg = 'C:\Program Files\PostgreSQL\18\bin'
$port = 55577
if (-not (Test-Path -LiteralPath (Join-Path $cluster 'PG_VERSION'))) {
    if (Test-Path -LiteralPath $cluster) { throw 'Uninitialized dedicated PG path exists; refusing overwrite' }
    & (Join-Path $pg 'initdb.exe') -D $cluster -U postgres --auth-local=trust --auth-host=trust --encoding=UTF8 --no-instructions
    if ($LASTEXITCODE -ne 0) { throw 'Dedicated PG18 initdb failed' }
}
if ((Get-Content -LiteralPath (Join-Path $cluster 'PG_VERSION')).Trim() -ne '18') {
    throw 'Dedicated cluster is not PostgreSQL 18'
}

& (Join-Path $pg 'pg_ctl.exe') -D $cluster -o "-h 127.0.0.1 -p $port" -l '.local-tools/v32-browser/v33-calendar-pg.log' start
if ($LASTEXITCODE -ne 0) { throw 'Dedicated PG18 start failed' }
$server = $null
try {
    $exists = & (Join-Path $pg 'psql.exe') -U postgres -h 127.0.0.1 -p $port -d postgres -Atc "SELECT 1 FROM pg_database WHERE datname='assistant'"
    if ($exists -ne '1') {
        & (Join-Path $pg 'createdb.exe') -U postgres -h 127.0.0.1 -p $port assistant
        if ($LASTEXITCODE -ne 0) { throw 'Dedicated assistant DB creation failed' }
    }
    foreach ($name in @('DATABASE_URL', 'DATABASE_URL_SYNC')) {
        $line = (Select-String -LiteralPath '.env' -Pattern "^$name=").Line
        if (-not $line) { throw "Missing local $name" }
        $value = $line.Substring($line.IndexOf('=') + 1).Trim().Trim('"')
        if ($value -notmatch '@(127\.0\.0\.1|localhost):\d+/assistant$') {
            throw "Refusing non-local $name target"
        }
        $dedicated = $value -replace ':\d+/assistant$', ":$port/assistant"
        [Environment]::SetEnvironmentVariable($name, $dedicated, 'Process')
    }
    $env:PYTHONPATH = $repo + ';' + (Resolve-Path '.local-tools/v32site').Path
    $env:PPA_ISOLATED_PG18_DATA_DIR = $cluster
    $env:V32_E2E_PASSWORD = 'synthetic-' + [guid]::NewGuid().ToString('N')
    $env:V32_E2E_TAG = 'v33closure'
    $env:V32_E2E_RESET_SYNTHETIC_PASSWORD = 'true'
    $env:V32_EDGE_PATH = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
    $nodePackages = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'
    if (-not (Test-Path -LiteralPath $nodePackages)) { throw 'Playwright runtime packages unavailable' }
    $env:NODE_PATH = $nodePackages
    $env:WEB_ENABLED = 'true'
    $env:WEB_PUBLIC_ORIGIN = 'https://127.0.0.1:18443'
    $env:SECRET_KEY = 'synthetic-v33-calendar-readonly-only'
    $env:TELEGRAM_BOT_TOKEN = 'synthetic-disabled-token'
    $env:TELEGRAM_WEBHOOK_URL = ''
    $env:GOOGLE_CALENDAR_ENABLED = if ($Mode -eq 'calendar') { 'true' } else { 'false' }
    $env:GOOGLE_CALENDAR_OWNER_TELEGRAM_ID = 'v32-browser-synthetic-' + $env:V32_E2E_TAG
    $env:GMAIL_ENABLED = 'false'
    $env:SCHEDULER_ENABLED = 'false'
    $env:MEMORY_ENABLED = 'false'

    python tests/browser/prepare_v32_e2e.py
    if ($LASTEXITCODE -ne 0) { throw 'Synthetic fixture or PG identity verification failed' }
    $server = Start-Process -FilePath python -ArgumentList 'tests/browser/v33_local_server.py' -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput '.local-tools/v32-browser/v33-calendar-out.log' `
        -RedirectStandardError '.local-tools/v32-browser/v33-calendar-err.log'
    $ready = $false
    for ($i = 0; $i -lt 80; $i++) {
        try {
            $response = Invoke-WebRequest -Uri 'https://127.0.0.1:18443/workbench/' -SkipCertificateCheck -TimeoutSec 2
            if ($response.StatusCode -eq 200) { $ready = $true; break }
        } catch {}
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { throw 'Local HTTPS SPA not ready' }
    if ($Mode -eq 'calendar') {
        node tests/browser/v33_calendar_readonly.cjs
        if ($LASTEXITCODE -ne 0) { throw 'Real Calendar read-only Edge acceptance failed' }
    } elseif ($Mode -eq 'g3') {
        node tests/browser/g3_five_page_e2e.cjs
        if ($LASTEXITCODE -ne 0) { throw 'G3 five-page Edge acceptance failed' }
    } else {
        node tests/browser/v32_workbench_e2e.cjs
        if ($LASTEXITCODE -ne 0) { throw 'Real-model Web workbench Edge acceptance failed' }
        node tests/browser/v33_workbench_smoke.cjs
        if ($LASTEXITCODE -ne 0) { throw 'Core Web workbench Edge smoke failed' }
    }
} finally {
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
    & (Join-Path $pg 'pg_ctl.exe') -D $cluster -m fast stop
}
