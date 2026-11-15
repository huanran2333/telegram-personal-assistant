# One attended session; app, Funnel and Webhook are scoped to this invocation.
param(
    [switch]$CheckOnly,
    [switch]$CleanupOnly,
    [switch]$UseLocalProductionCredential
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python environment is missing: $python"
}

$changedEnvironment = @()
if ($UseLocalProductionCredential) {
    $changedEnvironment = @(
        'DATABASE_URL', 'DATABASE_URL_SYNC', 'APP_ENV',
        'TELEGRAM_LIVE_START_PREFLIGHT_ACK', 'SCHEDULER_ENABLED',
        'DAILY_RUNDOWN_ENABLED', 'SECRET_KEY'
    )
}
$originalEnvironment = @{}
foreach ($name in $changedEnvironment) {
    $originalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
Push-Location $repo
try {
    if ($UseLocalProductionCredential) {
        . (Join-Path $PSScriptRoot 'load_personal_assistant_production_env.ps1')
        $secretFile = Join-Path $repo '.local-tools/g4b_guard/pa_secret_dpapi.xml'
        if (-not (Test-Path -LiteralPath $secretFile -PathType Leaf)) {
            throw 'Local DPAPI application secret unavailable'
        }
        $secretCredential = Import-Clixml -LiteralPath $secretFile
        if ($secretCredential -isnot [PSCredential]) {
            throw 'Local DPAPI application secret has an unexpected type'
        }
        $secretValue = $secretCredential.GetNetworkCredential().Password
        if ($secretValue.Length -lt 32) {
            throw 'Local DPAPI application secret is too short'
        }
        $env:SECRET_KEY = $secretValue
        $secretValue = $null
        $secretCredential = $null
    }
    $runnerArgs = @('-m', 'app.on_demand_runner')
    if ($CheckOnly) { $runnerArgs += '--check-only' }
    if ($CleanupOnly) { $runnerArgs += '--cleanup-only' }
    & $python @runnerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "On-demand launcher exited with code $LASTEXITCODE"
    }
}
finally {
    foreach ($name in $changedEnvironment) {
        [Environment]::SetEnvironmentVariable($name, $originalEnvironment[$name], 'Process')
    }
    Pop-Location
}
