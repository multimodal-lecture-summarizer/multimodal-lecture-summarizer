# Multimodal Lecture Summarizer - PowerShell Launcher
$OutputEncoding = [System.Text.Encoding]::UTF8

$script:RootDir = $PSScriptRoot
if (-not $script:RootDir) { $script:RootDir = Get-Location }
Set-Location $script:RootDir

$script:Processes = @()

function Show-Menu {
    Clear-Host
    Write-Host "===================================================" -ForegroundColor Cyan
    Write-Host "    MULTIMODAL LECTURE SUMMARIZER - LAUNCHER       " -ForegroundColor Yellow
    Write-Host "===================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  [1] Local Dev Mode (Backend + Celery Worker + Frontend)" -ForegroundColor Green
    Write-Host "  [2] Stop All Local Services" -ForegroundColor Yellow
    Write-Host "  [3] Docker Compose Mode (All services in Docker)" -ForegroundColor Magenta
    Write-Host "  [4] Reset DB & R2 Storage" -ForegroundColor Red
    Write-Host "  [5] Install / Update Dependencies" -ForegroundColor Blue
    Write-Host "  [6] Exit (Stop services & Quit)" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "===================================================" -ForegroundColor Cyan
}

function Stop-LocalDev {
    Write-Host "`nStopping all launched local services..." -ForegroundColor Yellow
    
    foreach ($proc in $script:Processes) {
        if ($proc -and -not $proc.HasExited) {
            try {
                taskkill /PID $proc.Id /T /F 2>$null
            } catch {}
        }
    }
    $script:Processes = @()

    # Also kill by window titles if spawned via cmd/powershell
    taskkill /FI "WINDOWTITLE eq MLS_Backend_API*" /F /T 2>$null
    taskkill /FI "WINDOWTITLE eq MLS_Celery_Worker*" /F /T 2>$null
    taskkill /FI "WINDOWTITLE eq MLS_Frontend*" /F /T 2>$null

    Write-Host "All services stopped successfully." -ForegroundColor Green
}

function Check-EnvFiles {
    if (-not (Test-Path "$script:RootDir\backend\.env")) {
        if (Test-Path "$script:RootDir\backend\.env.example") {
            Write-Host "Creating backend\.env from example..." -ForegroundColor Yellow
            Copy-Item "$script:RootDir\backend\.env.example" "$script:RootDir\backend\.env"
        }
    }
    if (-not (Test-Path "$script:RootDir\.env")) {
        if (Test-Path "$script:RootDir\.env.example") {
            Write-Host "Creating .env from example..." -ForegroundColor Yellow
            Copy-Item "$script:RootDir\.env.example" "$script:RootDir\.env"
        }
    }
}

function Check-VirtualEnv {
    if (-not (Test-Path "$script:RootDir\backend\venv")) {
        Write-Host "Creating virtual environment backend\venv..." -ForegroundColor Yellow
        python -m venv "$script:RootDir\backend\venv"
        Write-Host "Installing requirements..." -ForegroundColor Yellow
        & "$script:RootDir\backend\venv\Scripts\pip.exe" install -r "$script:RootDir\backend\requirements.txt"
        & "$script:RootDir\backend\venv\Scripts\pip.exe" install -r "$script:RootDir\ai_workers\requirements.txt"
    }
}

function Check-NodeModules {
    if (-not (Test-Path "$script:RootDir\frontend\node_modules")) {
        Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
        Set-Location "$script:RootDir\frontend"
        npm install
        Set-Location "$script:RootDir"
    }
}

function Start-LocalDev {
    Check-EnvFiles
    Check-VirtualEnv
    Check-NodeModules

    Stop-LocalDev

    Write-Host "`nLaunching Backend API..." -ForegroundColor Green
    $p1 = Start-Process powershell -WorkingDirectory "$script:RootDir\backend" -ArgumentList "-NoExit", "-Command", "`$host.ui.RawUI.WindowTitle='MLS_Backend_API'; .\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000" -PassThru

    Write-Host "Launching Celery Worker..." -ForegroundColor Green
    $p2 = Start-Process powershell -WorkingDirectory "$script:RootDir" -ArgumentList "-NoExit", "-Command", "`$host.ui.RawUI.WindowTitle='MLS_Celery_Worker'; `$env:PYTHONPATH='.'; `$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'; .\backend\venv\Scripts\python.exe -m celery -A ai_workers.core.celery_app worker --loglevel=info --pool=solo --concurrency=1" -PassThru

    Write-Host "Launching Frontend..." -ForegroundColor Green
    $p3 = Start-Process powershell -WorkingDirectory "$script:RootDir\frontend" -ArgumentList "-NoExit", "-Command", "`$host.ui.RawUI.WindowTitle='MLS_Frontend'; npm run dev" -PassThru

    $script:Processes = @($p1, $p2, $p3)

    Write-Host "`n===================================================" -ForegroundColor Cyan
    Write-Host "All services started in separate windows!" -ForegroundColor Green
    Write-Host "- Backend API: http://127.0.0.1:8000 (Docs: http://127.0.0.1:8000/docs)" -ForegroundColor Gray
    Write-Host "- Frontend:    http://localhost:5173" -ForegroundColor Gray
    Write-Host "===================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press [K] + Enter to Stop all services now" -ForegroundColor Yellow
    Write-Host "Press [M] + Enter to Keep running and return to Main Menu" -ForegroundColor Gray

    $opt = Read-Host "Option (K/M)"
    if ($opt -eq 'K' -or $opt -eq 'k') {
        Stop-LocalDev
    }
}

function Start-Docker {
    Write-Host "`nStarting Docker Compose..." -ForegroundColor Magenta
    docker compose version 2>$null
    if ($LASTEXITCODE -eq 0) {
        docker compose up --build
    } else {
        docker-compose up --build
    }
}

function Invoke-Reset {
    Write-Host "`nResetting Database & R2 Storage..." -ForegroundColor Red
    if (Test-Path "$script:RootDir\backend\.venv\Scripts\python.exe") {
        Set-Location "$script:RootDir\backend"
        & .venv\Scripts\python.exe reset_r2_and_db.py
        Set-Location "$script:RootDir"
    } else {
        python "$script:RootDir\backend\reset_r2_and_db.py"
    }
}

function Install-Deps {
    Write-Host "`nUpdating Python dependencies..." -ForegroundColor Blue
    if (-not (Test-Path "$script:RootDir\backend\.venv")) {
        python -m venv "$script:RootDir\backend\.venv"
    }
    & "$script:RootDir\backend\.venv\Scripts\pip.exe" install -r "$script:RootDir\backend\requirements.txt"
    & "$script:RootDir\backend\.venv\Scripts\pip.exe" install -r "$script:RootDir\ai_workers\requirements.txt"

    Write-Host "Updating Frontend dependencies..." -ForegroundColor Blue
    Set-Location "$script:RootDir\frontend"
    npm install
    Set-Location "$script:RootDir"
    Write-Host "Finished updating dependencies." -ForegroundColor Green
}

# Param support (e.g. .\run_win.ps1 -Dev)
param (
    [switch]$Dev,
    [switch]$Stop,
    [switch]$Docker,
    [switch]$Reset,
    [switch]$Install
)

if ($Dev) { Start-LocalDev; exit }
if ($Stop) { Stop-LocalDev; exit }
if ($Docker) { Start-Docker; exit }
if ($Reset) { Invoke-Reset; exit }
if ($Install) { Install-Deps; exit }

do {
    Show-Menu
    $choice = Read-Host "Select an option (1-6)"
    switch ($choice) {
        '1' { Start-LocalDev; break }
        '2' { Stop-LocalDev; Read-Host "Press Enter to return to menu..."; break }
        '3' { Start-Docker; Read-Host "Press Enter to return to menu..."; break }
        '4' { Invoke-Reset; Read-Host "Press Enter to return to menu..."; break }
        '5' { Install-Deps; Read-Host "Press Enter to return to menu..."; break }
        '6' { Stop-LocalDev; exit }
        default { Write-Host "Invalid option!" -ForegroundColor Red; Start-Sleep -Seconds 1 }
    }
} while ($choice -ne '6')
