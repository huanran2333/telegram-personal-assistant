# One bounded public-TLS Edge run against the existing synthetic PG18 cluster.
# Start the foreground Funnel separately; this script never uses the real Bot.
param([Parameter(Mandatory = $true)][string]$Origin)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $repo
if ($Origin -notmatch '^https://[a-zA-Z0-9.-]+\.ts\.net$') {
    throw 'Expected exact trusted Funnel origin'
}
$cluster = (Resolve-Path -LiteralPath '.local-tools/g4_public_pg18_data').Path
if ((Get-Content -LiteralPath (Join-Path $cluster 'PG_VERSION')).Trim() -ne '18') {
    throw 'Dedicated synthetic PG18 cluster missing'
}
if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Refusing to replace an existing port-8000 service'
}
$async = (Get-Content -LiteralPath '.env' | Where-Object { $_ -match '^DATABASE_URL=' } |
    Select-Object -First 1) -replace '^DATABASE_URL=', ''
$sync = (Get-Content -LiteralPath '.env' | Where-Object { $_ -match '^DATABASE_URL_SYNC=' } |
    Select-Object -First 1) -replace '^DATABASE_URL_SYNC=', ''
if ($async -notmatch '@(127\.0\.0\.1|localhost):55432/assistant$' -or
    $sync -notmatch '@(127\.0\.0\.1|localhost):55432/assistant$') {
    throw 'Expected historical local credential source, not a production URL'
}
$env:DATABASE_URL = $async -replace ':55432/assistant$', ':55588/assistant'
$env:DATABASE_URL_SYNC = $sync -replace ':55432/assistant$', ':55588/assistant'
$env:PPA_ISOLATED_PG18_DATA_DIR = $cluster
$env:PPA_G4_SYNTHETIC_PUBLIC = 'YES'
$env:V32_E2E_TAG = 'g4public'
$env:V32_E2E_PASSWORD = 'synthetic-' + [guid]::NewGuid().ToString('N')
$env:V32_E2E_RESET_SYNTHETIC_PASSWORD = 'true'
$env:V32_E2E_ORIGIN = $Origin
$env:G3_FORCE_STALE_CACHE = '1'
$env:V32_EDGE_PATH = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
$nodeModules = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'
if (-not (Test-Path -LiteralPath (Join-Path $nodeModules 'playwright'))) {
    throw 'Playwright runtime unavailable'
}
$env:NODE_PATH = $nodeModules
$env:PYTHONPATH = $repo
$env:WEB_ENABLED = 'true'
$env:WEB_PUBLIC_ORIGIN = $Origin
$env:SECRET_KEY = 'synthetic-' + [guid]::NewGuid().ToString('N') +
    [guid]::NewGuid().ToString('N')
$env:TELEGRAM_BOT_TOKEN = ''
$env:TELEGRAM_WEBHOOK_URL = ''
$env:TELEGRAM_LIVE_START_PREFLIGHT_ACK = 'false'
$env:SCHEDULER_ENABLED = 'false'
$env:DAILY_RUNDOWN_ENABLED = 'false'
$env:GOOGLE_CALENDAR_ENABLED = 'false'
$env:GMAIL_ENABLED = 'false'
$env:MEMORY_ENABLED = 'false'
$env:LLM_PROVIDER = 'openai'
$env:LLM_MODEL = 'synthetic-unused'
$env:OPENAI_BASE_URL = 'http://127.0.0.1:1/v1'
$env:OPENAI_API_KEY = 'synthetic-unused'
$env:LANGCHAIN_OPENAI_TCP_KEEPALIVE = '0'
$env:HTTP_PROXY = ''
$env:HTTPS_PROXY = ''
$env:ALL_PROXY = ''
$python = (Resolve-Path -LiteralPath '.venv/Scripts/python.exe').Path
& $python tests/browser/prepare_v32_e2e.py
if ($LASTEXITCODE -ne 0) { throw 'Synthetic fixture preparation failed' }
$stamp = Get-Date -Format yyyyMMddHHmmss
$out = Join-Path $repo ".local-tools/g4b_guard/g4_public_stale_$stamp.out.log"
$err = Join-Path $repo ".local-tools/g4b_guard/g4_public_stale_$stamp.err.log"
if ((Test-Path -LiteralPath $out) -or (Test-Path -LiteralPath $err)) {
    throw 'Refusing to overwrite existing local logs'
}
$server = Start-Process -FilePath $python -ArgumentList 'tests/browser/g4_public_synthetic_server.py' `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $out -RedirectStandardError $err
try {
    $ready = $false
    for ($i = 0; $i -lt 80; $i++) {
        try {
            # With the real Bot deliberately disabled, aggregate /health may
            # be degraded; the SPA must still serve the synthetic Web slice.
            $response = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/workbench/' -TimeoutSec 2
            if ($response.StatusCode -eq 200) { $ready = $true; break }
        } catch {}
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { throw 'Synthetic public backend not ready' }
    node tests/browser/g3_five_page_e2e.cjs
    if ($LASTEXITCODE -ne 0) { throw 'Public Edge stale-version E2E failed' }
} finally {
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
}
