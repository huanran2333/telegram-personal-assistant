# Dot-source in the intended local PowerShell process after reviewing G4-B
# preflight. This only configures that process; it never starts the app/Bot.
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$credentialFile = Join-Path $repo '.local-tools/g4b_guard/pg_app_dpapi.xml'
if (-not (Test-Path -LiteralPath $credentialFile)) {
    throw 'Local DPAPI application credential unavailable'
}
$credential = Import-Clixml -LiteralPath $credentialFile
if ($credential.UserName -ne 'personal_assistant_app') {
    throw 'Application role identity mismatch'
}
$escapedPassword = [Uri]::EscapeDataString(
    $credential.GetNetworkCredential().Password)
$authority = "personal_assistant_app:$escapedPassword@127.0.0.1:5432/personal_assistant"
$env:DATABASE_URL = "postgresql+asyncpg://$authority"
$env:DATABASE_URL_SYNC = "postgresql://$authority"
$env:APP_ENV = 'production'
# Interactive messages and proactive messages have separate authorization.
# Both stay off until the next gate's queue/schedule review and explicit opt-in.
$env:TELEGRAM_LIVE_START_PREFLIGHT_ACK = 'false'
$env:SCHEDULER_ENABLED = 'false'
$env:DAILY_RUNDOWN_ENABLED = 'false'
$escapedPassword = $null
$authority = $null
$credential = $null
Write-Output 'Production database target loaded; Bot and proactive startup remain gated'
