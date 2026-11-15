# One-time local PG18 provisioning. No application, Bot, or worker is started.
# The operator first creates a DPAPI-protected postgres PSCredential in the
# ignored .local-tools/g4b_guard directory using Get-Credential/Export-Clixml.
param()
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$secrets = Join-Path $repo '.local-tools/g4b_guard'
$adminFile = Join-Path $secrets 'pg_admin_dpapi.xml'
$appFile = Join-Path $secrets 'pg_app_dpapi.xml'
$psql = 'C:\Program Files\PostgreSQL\18\bin\psql.exe'
$createdb = 'C:\Program Files\PostgreSQL\18\bin\createdb.exe'
if (-not (Test-Path -LiteralPath $adminFile) -or
    (Test-Path -LiteralPath $appFile)) {
    throw 'Missing admin credential or app credential already exists; refusing overwrite'
}
$admin = Import-Clixml -LiteralPath $adminFile
if ($admin.UserName -ne 'postgres') { throw 'Administrator credential username mismatch' }
$env:PGPASSWORD = $admin.GetNetworkCredential().Password
try {
    $identity = & $psql -w -h 127.0.0.1 -p 5432 -U postgres -d postgres -X -Atqc `
        "SELECT current_user,current_database(),current_setting('server_version'),current_setting('data_directory'),current_setting('port')"
    if ($LASTEXITCODE -ne 0) { throw 'System PG18 admin authentication failed' }
    $parts = $identity.Split('|')
    if ($parts.Count -ne 5 -or $parts[0] -ne 'postgres' -or
        $parts[1] -ne 'postgres' -or $parts[2] -ne '18.6' -or
        $parts[3] -ne 'C:/Program Files/PostgreSQL/18/data' -or
        $parts[4] -ne '5432') { throw 'System PostgreSQL identity mismatch' }
    $exists = & $psql -w -h 127.0.0.1 -p 5432 -U postgres -d postgres -X -Atqc `
        "SELECT (SELECT count(*) FROM pg_database WHERE datname='personal_assistant')::text||'|'||(SELECT count(*) FROM pg_roles WHERE rolname='personal_assistant_app')::text"
    if ($LASTEXITCODE -ne 0 -or $exists -ne '0|0') {
        throw 'Dedicated database or role already exists; inspect before retry'
    }

    [byte[]]$entropy = New-Object byte[] 48
    [System.Security.Cryptography.RandomNumberGenerator]::Fill($entropy)
    $password = [Convert]::ToBase64String($entropy).TrimEnd('=') -replace '\+', '-' -replace '/', '_'
    $secure = ConvertTo-SecureString $password -AsPlainText -Force
    $app = [pscredential]::new('personal_assistant_app', $secure)
    $app | Export-Clixml -LiteralPath $appFile

    & $psql -w -h 127.0.0.1 -p 5432 -U postgres -d postgres -X -v ON_ERROR_STOP=1 -c `
        'CREATE ROLE personal_assistant_app NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS'
    if ($LASTEXITCODE -ne 0) { throw 'Dedicated role creation failed' }
    # Feed the generated secret on stdin, never in argv, Git, or process logs.
    "ALTER ROLE personal_assistant_app WITH LOGIN PASSWORD '$password';" |
        & $psql -w -h 127.0.0.1 -p 5432 -U postgres -d postgres -X -v ON_ERROR_STOP=1 -f -
    if ($LASTEXITCODE -ne 0) { throw 'Dedicated role password setup failed' }
    & $createdb -h 127.0.0.1 -p 5432 -U postgres -O personal_assistant_app -E UTF8 personal_assistant
    if ($LASTEXITCODE -ne 0) { throw 'Dedicated database creation failed' }
    & $psql -w -h 127.0.0.1 -p 5432 -U postgres -d postgres -X -v ON_ERROR_STOP=1 -c `
        'REVOKE CONNECT ON DATABASE personal_assistant FROM PUBLIC'
    if ($LASTEXITCODE -ne 0) { throw 'Dedicated database CONNECT restriction failed' }
    Write-Output 'DEDICATED_PG18_DATABASE_PROVISIONED'
} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
    $password = $null
    $secure = $null
    $admin = $null
    $app = $null
}
