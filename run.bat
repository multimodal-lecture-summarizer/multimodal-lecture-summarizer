@echo off
chcp 65001 >nul
title Multimodal Lecture Summarizer Launcher

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

:MENU
cls
echo ===================================================
echo     MULTIMODAL LECTURE SUMMARIZER - LAUNCHER
echo ===================================================
echo.
echo   [1] Local Dev Mode (Backend + Celery Worker + Frontend)
echo   [2] Stop All Local Services
echo   [3] Docker Compose Mode (All services in Docker)
echo   [4] Reset DB ^& R2 Storage
echo   [5] Install / Update Dependencies
echo   [6] Exit (Stop services ^& Quit)
echo.
echo ===================================================
set /p CHOICE="Select an option (1-6): "

if "%CHOICE%"=="1" goto LOCAL_DEV
if "%CHOICE%"=="2" goto STOP_SERVICES
if "%CHOICE%"=="3" goto DOCKER_MODE
if "%CHOICE%"=="4" goto RESET_SYSTEM
if "%CHOICE%"=="5" goto INSTALL_DEPS
if "%CHOICE%"=="6" goto EXIT_CLEAN
goto MENU

:STOP_SERVICES
echo.
echo Stopping all running local services (MLS_Backend, MLS_Worker, MLS_Frontend)...
taskkill /FI "WINDOWTITLE eq MLS_Backend_API*" /F /T >nul 2>&1
wmic process where "name='cmd.exe' and commandline like '%%MLS_Backend_API%%'" call terminate >nul 2>&1
taskkill /FI "WINDOWTITLE eq MLS_Celery_Worker*" /F /T >nul 2>&1
wmic process where "name='cmd.exe' and commandline like '%%MLS_Celery_Worker%%'" call terminate >nul 2>&1
taskkill /FI "WINDOWTITLE eq MLS_Frontend*" /F /T >nul 2>&1
wmic process where "name='cmd.exe' and commandline like '%%MLS_Frontend%%'" call terminate >nul 2>&1
echo Services stopped successfully.
timeout /t 2 >nul
goto MENU

:EXIT_CLEAN
echo.
echo Stopping all running services before exit...
taskkill /FI "WINDOWTITLE eq MLS_Backend_API*" /F /T >nul 2>&1
wmic process where "name='cmd.exe' and commandline like '%%MLS_Backend_API%%'" call terminate >nul 2>&1
taskkill /FI "WINDOWTITLE eq MLS_Celery_Worker*" /F /T >nul 2>&1
wmic process where "name='cmd.exe' and commandline like '%%MLS_Celery_Worker%%'" call terminate >nul 2>&1
taskkill /FI "WINDOWTITLE eq MLS_Frontend*" /F /T >nul 2>&1
wmic process where "name='cmd.exe' and commandline like '%%MLS_Frontend%%'" call terminate >nul 2>&1
echo Goodbye!
exit /b 0

:LOCAL_DEV
echo.
echo [1/3] Checking environment files...
if not exist "backend\.env" (
    if exist "backend\.env.example" (
        echo Creating backend\.env from example...
        copy "backend\.env.example" "backend\.env"
    )
)
if not exist ".env" (
    if exist ".env.example" (
        echo Creating .env from example...
        copy ".env.example" ".env"
    )
)

if not exist "backend\.venv" (
    echo Virtual environment backend\.venv not found. Creating virtualenv...
    python -m venv backend\.venv
    echo Installing dependencies...
    backend\.venv\Scripts\pip.exe install -r backend\requirements.txt
    backend\.venv\Scripts\pip.exe install -r ai_workers\requirements.txt
)

if not exist "frontend\node_modules" (
    echo frontend\node_modules not found. Running npm install...
    cd /d "%PROJECT_ROOT%frontend"
    call npm install
    cd /d "%PROJECT_ROOT%"
)

echo.
echo Stopping any existing service instances first...
taskkill /FI "WINDOWTITLE eq MLS_Backend_API*" /F /T >nul 2>&1
wmic process where "name='cmd.exe' and commandline like '%%MLS_Backend_API%%'" call terminate >nul 2>&1
taskkill /FI "WINDOWTITLE eq MLS_Celery_Worker*" /F /T >nul 2>&1
wmic process where "name='cmd.exe' and commandline like '%%MLS_Celery_Worker%%'" call terminate >nul 2>&1
taskkill /FI "WINDOWTITLE eq MLS_Frontend*" /F /T >nul 2>&1
wmic process where "name='cmd.exe' and commandline like '%%MLS_Frontend%%'" call terminate >nul 2>&1

echo Starting services in separate windows...
echo Starting Backend API...
start "MLS_Backend_API" /d "%PROJECT_ROOT%backend" cmd /k "title MLS_Backend_API && ..\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000"

echo Starting Celery AI Worker...
start "MLS_Celery_Worker" /d "%PROJECT_ROOT%" cmd /k "title MLS_Celery_Worker && set PYTHONPATH=. && set CUBLAS_WORKSPACE_CONFIG=:4096:8 && backend\.venv\Scripts\python.exe -m celery -A ai_workers.core.celery_app worker --loglevel=info --pool=solo --concurrency=1"

echo Starting Frontend...
start "MLS_Frontend" /d "%PROJECT_ROOT%frontend" cmd /k "title MLS_Frontend && npm run dev"

echo.
echo ===================================================
echo All services launched!
echo - Backend API: http://127.0.0.1:8000 (Docs: http://127.0.0.1:8000/docs)
echo - Frontend:    http://localhost:5173
echo ===================================================
echo.
echo Press [K] + Enter to Stop all services now
echo Press [M] + Enter to Keep running and return to Main Menu
echo.
set /p DEV_OPT="Option (K/M): "

if /i "%DEV_OPT%"=="K" goto STOP_SERVICES
goto MENU

:DOCKER_MODE
echo.
echo Starting system with Docker Compose...
docker compose version >nul 2>&1
if %errorlevel% equ 0 (
    docker compose up --build
) else (
    docker-compose up --build
)
pause
goto MENU

:RESET_SYSTEM
echo.
echo Resetting Database and Cloudflare R2 Storage...
if exist "backend\.venv\Scripts\python.exe" (
    cd /d "%PROJECT_ROOT%backend"
    .venv\Scripts\python.exe reset_r2_and_db.py
    cd /d "%PROJECT_ROOT%"
) else (
    python backend\reset_r2_and_db.py
)
pause
goto MENU

:INSTALL_DEPS
echo.
echo Installing/Updating dependencies...
if not exist "backend\.venv" (
    python -m venv backend\.venv
)
backend\.venv\Scripts\pip.exe install -r backend\requirements.txt
backend\.venv\Scripts\pip.exe install -r ai_workers\requirements.txt
cd /d "%PROJECT_ROOT%frontend"
call npm install
cd /d "%PROJECT_ROOT%"
echo Dependencies installation finished.
pause
goto MENU
