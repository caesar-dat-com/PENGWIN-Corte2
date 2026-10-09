param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$cfgDemo = Get-Content -LiteralPath (Join-Path $projectRoot 'config_datos.local.json') -Raw -Encoding utf8 | ConvertFrom-Json
$runtimeDemo = Join-Path $projectRoot 'salidas/cierre/demo_runtime'
$serverScript = Join-Path $PSScriptRoot 'servir_dashboard.py'
$tunnelExe = Join-Path $projectRoot 'tools/cloudflared.exe'
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw "El puerto $Port ya está ocupado; no se reemplaza ese proceso." }
if (-not (Test-Path -LiteralPath $tunnelExe)) { throw 'Falta tools/cloudflared.exe, herramienta oficial de Cloudflare.' }
if (Test-Path -LiteralPath (Join-Path $runtimeDemo 'procesos.json')) { throw 'Existe registro de una demo anterior. Revisar y detenerla antes de iniciar otra.' }
New-Item -ItemType Directory -Path $runtimeDemo -Force | Out-Null
$serverDemo = Start-Process -FilePath $cfgDemo.python -ArgumentList @('-B', ('"' + $serverScript + '"'), '--port', $Port) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeDemo 'servidor.log') -RedirectStandardError (Join-Path $runtimeDemo 'servidor_error.log')
$serverConnection = $null
for ($demoAttempt=0; $demoAttempt -lt 30; $demoAttempt++) {
    Start-Sleep -Milliseconds 300
    $serverConnection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($serverConnection) { break }
}
if (-not $serverConnection) { throw 'No inició el servidor local. Revisar servidor_error.log.' }
$tunnelDemo = Start-Process -FilePath $tunnelExe -ArgumentList @('tunnel','--no-autoupdate','--protocol','http2','--url',"http://127.0.0.1:$Port") -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeDemo 'cloudflare.log') -RedirectStandardError (Join-Path $runtimeDemo 'cloudflare_error.log')
$demoState = [ordered]@{server_pid=$serverConnection.OwningProcess; tunnel_pid=$tunnelDemo.Id; port=$Port; server_script=$serverScript; tunnel_exe=$tunnelExe; started_utc=[DateTime]::UtcNow.ToString('o'); scope='Solo dashboard estático; enlace público temporal autorizado por el usuario'; status='starting'}
$demoState | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimeDemo 'procesos.json') -Encoding utf8
$demoState | ConvertTo-Json
