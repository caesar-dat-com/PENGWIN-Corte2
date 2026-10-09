$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$statePath=Join-Path $projectRoot 'salidas/cierre/demo_runtime/procesos.json'
$stateDemo=Get-Content -LiteralPath $statePath -Raw -Encoding utf8 | ConvertFrom-Json
foreach ($kindDemo in @('tunnel','server')) {
    $demoProcessId=if ($kindDemo -eq 'tunnel') {$stateDemo.tunnel_pid} else {$stateDemo.server_pid}
    $demoProcess=Get-CimInstance Win32_Process -Filter "ProcessId=$demoProcessId" -ErrorAction SilentlyContinue
    if (-not $demoProcess) {continue}
    $expectedDemo=if ($kindDemo -eq 'tunnel') {$stateDemo.tunnel_exe} else {$stateDemo.server_script}
    if (-not $demoProcess.CommandLine -or -not $demoProcess.CommandLine.Contains($expectedDemo)) {throw "El PID $demoProcessId no coincide con la demo; no se detiene."}
    Stop-Process -Id $demoProcessId
}
$stateDemo.status='stopped'
$stateDemo | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
Write-Output 'Demo detenida; la URL temporal deja de estar disponible.'
