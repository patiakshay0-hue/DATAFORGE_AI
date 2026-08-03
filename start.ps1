# start.ps1
# Run with:  powershell -ExecutionPolicy Bypass -File start.ps1

$root = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }

# --- Free our ports (only real listeners; never touch PID 0/4) -----------------
function Stop-PortListener($port) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($procId in ($conns.OwningProcess | Select-Object -Unique)) {
        if ($procId -and $procId -gt 4) {
            Write-Host "Freeing port $port (PID $procId)" -ForegroundColor DarkYellow
            try { Stop-Process -Id $procId -Force -ErrorAction Stop } catch {}
        }
    }
}
Stop-PortListener 5000
Stop-PortListener 3000
Start-Sleep 1

# --- Sanity checks -------------------------------------------------------------
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "python not found on PATH." -ForegroundColor Red; exit 1
}
if (-not (Test-Path "$root\frontend\node_modules")) {
    Write-Host "frontend\node_modules missing. Run: npm install --prefix frontend" -ForegroundColor Red; exit 1
}

# --- Backend -------------------------------------------------------------------
Write-Host "Starting backend..." -ForegroundColor Cyan
$be = Start-Process -NoNewWindow -PassThru -FilePath "python" `
    -ArgumentList "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--reload" `
    -WorkingDirectory "$root\backend"

# --- Frontend ------------------------------------------------------------------
# NOTE: must be npm.cmd / npx.cmd, not "npx". Start-Process resolves the bare name
# to the extensionless Unix shell script shipped by Node, which Windows cannot
# execute ("%1 is not a valid Win32 application").
Write-Host "Starting frontend..." -ForegroundColor Cyan
$fe = Start-Process -NoNewWindow -PassThru -FilePath "npm.cmd" `
    -ArgumentList "run", "dev" `
    -WorkingDirectory "$root\frontend"

Write-Host "Backend  -> http://localhost:5000  (docs: /docs)" -ForegroundColor Green
Write-Host "Frontend -> http://localhost:3000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop both." -ForegroundColor DarkGray

# --- Wait ----------------------------------------------------------------------
$procs = @($be, $fe) | Where-Object { $_ -and -not $_.HasExited }
try {
    if ($procs) { Wait-Process -Id $procs.Id }
    else { Write-Host "Neither process started." -ForegroundColor Red; exit 1 }
}
finally {
    foreach ($p in @($be, $fe)) {
        if ($p -and -not $p.HasExited) {
            try { Stop-Process -Id $p.Id -Force -ErrorAction Stop } catch {}
        }
    }
    Stop-PortListener 5000
    Stop-PortListener 3000
}
