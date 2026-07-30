#!/usr/bin/env bash

# Multimodal Lecture Summarizer - Ubuntu / Linux Launcher
set -e

# Change to repo root directory if script is run from elsewhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COLOR_BLUE='\033[0;34m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_RED='\033[0;31m'
COLOR_MAGENTA='\033[0;35m'
COLOR_NC='\033[0m' # No Color

check_python() {
    if command -v python3.10 &>/dev/null; then
        PYTHON_CMD="python3.10"
    elif command -v python3.11 &>/dev/null; then
        PYTHON_CMD="python3.11"
    else
        PYTHON_CMD="python3"
    fi
}

check_env_files() {
    if [ ! -f "backend/.env" ]; then
        if [ -f "backend/.env.example" ]; then
            echo -e "${COLOR_YELLOW}Creating backend/.env from example...${COLOR_NC}"
            cp backend/.env.example backend/.env
        fi
    fi
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            echo -e "${COLOR_YELLOW}Creating .env from example...${COLOR_NC}"
            cp .env.example .env
        fi
    fi
}

setup_venv() {
    check_python
    if [ ! -d "backend/.venv" ]; then
        echo -e "${COLOR_YELLOW}Creating virtual environment backend/.venv with $PYTHON_CMD...${COLOR_NC}"
        $PYTHON_CMD -m venv backend/.venv
        echo -e "${COLOR_YELLOW}Installing Python requirements...${COLOR_NC}"
        backend/.venv/bin/pip install --upgrade pip
        backend/.venv/bin/pip install -r backend/requirements.txt
        backend/.venv/bin/pip install -r ai_workers/requirements.txt
    fi
}

setup_frontend() {
    if [ ! -d "frontend/node_modules" ]; then
        echo -e "${COLOR_YELLOW}Installing frontend node_modules...${COLOR_NC}"
        (cd frontend && npm install)
    fi
}

run_dev() {
    check_env_files
    setup_venv
    setup_frontend

    echo -e "${COLOR_GREEN}Starting Local Development Mode...${COLOR_NC}"

    PIDS=()

    cleanup() {
        echo -e "\n${COLOR_YELLOW}Stopping all services...${COLOR_NC}"
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done
        wait 2>/dev/null || true
        echo -e "${COLOR_GREEN}All services stopped.${COLOR_NC}"
        exit 0
    }

    trap cleanup SIGINT SIGTERM EXIT

    echo -e "${COLOR_BLUE}[1/3] Starting Backend API (Uvicorn)...${COLOR_NC}"
    (cd backend && ../backend/.venv/bin/python -m uvicorn app.main:app --reload --port 8000) &
    PIDS+=($!)

    echo -e "${COLOR_BLUE}[2/3] Starting Celery AI Worker...${COLOR_NC}"
    PYTHONPATH=. backend/.venv/bin/python -m celery -A ai_workers.core.celery_app worker --loglevel=info --concurrency=2 &
    PIDS+=($!)

    echo -e "${COLOR_BLUE}[3/3] Starting Frontend (React)...${COLOR_NC}"
    (cd frontend && npm run dev) &
    PIDS+=($!)

    echo -e "${COLOR_GREEN}Services started successfully! Press Ctrl+C to stop all.${COLOR_NC}"
    echo -e "- Backend API: http://127.0.0.1:8000 (Docs: http://127.0.0.1:8000/docs)"
    echo -e "- Frontend:    http://localhost:5173"

    wait
}

run_docker() {
    echo -e "${COLOR_MAGENTA}Starting Docker Compose...${COLOR_NC}"
    if command -v docker &>/dev/null && docker compose version &>/dev/null; then
        docker compose up --build
    else
        docker-compose up --build
    fi
}

reset_system() {
    echo -e "${COLOR_RED}Resetting DB & Cloudflare R2 storage...${COLOR_NC}"
    if [ -f "backend/.venv/bin/python" ]; then
        (cd backend && .venv/bin/python reset_r2_and_db.py)
    else
        check_python
        (cd backend && $PYTHON_CMD reset_r2_and_db.py)
    fi
}

install_deps() {
    check_env_files
    setup_venv
    setup_frontend
    echo -e "${COLOR_GREEN}Dependencies updated successfully.${COLOR_NC}"
}

show_menu() {
    echo -e "${COLOR_BLUE}===================================================${COLOR_NC}"
    echo -e "${COLOR_YELLOW}    MULTIMODAL LECTURE SUMMARIZER - LAUNCHER       ${COLOR_NC}"
    echo -e "${COLOR_BLUE}===================================================${COLOR_NC}"
    echo -e "  ${COLOR_GREEN}1) Local Dev Mode (Backend + Celery Worker + Frontend)${COLOR_NC}"
    echo -e "  ${COLOR_MAGENTA}2) Docker Compose Mode (All services in Docker)${COLOR_NC}"
    echo -e "  ${COLOR_RED}3) Reset DB & R2 Storage${COLOR_NC}"
    echo -e "  ${COLOR_BLUE}4) Install / Update Dependencies${COLOR_NC}"
    echo -e "  5) Exit"
    echo -e "${COLOR_BLUE}===================================================${COLOR_NC}"
    read -p "Select an option (1-5): " CHOICE
    case "$CHOICE" in
        1) run_dev ;;
        2) run_docker ;;
        3) reset_system ;;
        4) install_deps ;;
        5) exit 0 ;;
        *) echo -e "${COLOR_RED}Invalid option!${COLOR_NC}" ; show_menu ;;
    esac
}

# Handle command line arguments
case "$1" in
    --dev|-d)
        run_dev
        ;;
    --docker|-c)
        run_docker
        ;;
    --reset|-r)
        reset_system
        ;;
    --install|-i)
        install_deps
        ;;
    *)
        show_menu
        ;;
esac
