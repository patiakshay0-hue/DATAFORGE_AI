# start.ps1
$root = Split-Path $MyInvocation.MyCommand.Path

# Kill any leftover processes on our ports
netstat -ano | Select-String ":5000" | ForEach-Object { ($_ -split '\s+')[-1] } | Where-Object { $_ -ne '' } | Select-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
netstat -ano | Select-String ":3000" | ForEach-Object { ($_ -split '\s+')[-1] } | Where-Object { $_ -ne '' } | Select-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
Start-Sleep 1

# Backend
Write-Host "Starting backend..." -ForegroundColor Cyan
$be = Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m uvicorn main:app --host 0.0.0.0 --port 5000 --reload" -PassThru -WorkingDirectory "$root\backend"

# Frontend
Write-Host "Starting frontend..." -ForegroundColor Cyan
$fe = Start-Process -NoNewWindow -FilePath "npx" -ArgumentList "vite" -PassThru -WorkingDirectory "$root\frontend"

Write-Host "Backend running on http://localhost:5000" -ForegroundColor Green
Write-Host "Frontend running on http://localhost:3000" -ForegroundColor Green

Wait-Process -Id $be.Id, $fe.Id


# run this file using "powershell -ExecutionPolicy Bypass -File start.ps1"