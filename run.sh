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

create_venv() {
    check_python
    echo -e "${COLOR_YELLOW}Creating virtual environment backend/.venv with $PYTHON_CMD...${COLOR_NC}"
    $PYTHON_CMD -m venv backend/.venv
}

install_python_deps() {
    echo -e "${COLOR_YELLOW}Installing/updating Python requirements...${COLOR_NC}"
    backend/.venv/bin/python -m pip install --upgrade pip
    backend/.venv/bin/python -m pip install -r backend/requirements.txt
    backend/.venv/bin/python -m pip install -r ai_workers/requirements.txt
}

setup_venv() {
    if [ ! -d "backend/.venv" ]; then
        create_venv
        install_python_deps
    fi
}

setup_frontend() {
    if [ ! -d "frontend/node_modules" ]; then
        echo -e "${COLOR_YELLOW}Installing frontend node_modules...${COLOR_NC}"
        (cd frontend && npm install)
    fi
}

stop_services() {
    echo -e "${COLOR_YELLOW}Stopping local services...${COLOR_NC}"
    pkill -f "uvicorn app.main:app" || true
    pkill -f "celery -A ai_workers.core.celery_app" || true
    pkill -f "npm run dev" || true
    pkill -f "vite" || true
    echo -e "${COLOR_GREEN}Services stopped.${COLOR_NC}"
}

detect_gui_terminal() {
    if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
        return 1
    fi
    # Check for common terminals, especially those used in XRDP (like xfce4-terminal)
    for term in gnome-terminal xfce4-terminal mate-terminal lxterminal konsole xterm x-terminal-emulator; do
        if command -v "$term" &>/dev/null; then
            GUI_TERMINAL="$term"
            return 0
        fi
    done
    return 1
}

start_service_terminal() {
    local title="$1"
    local working_directory="$2"
    local service_command="$3"
    
    # We want the window to stay open after the command finishes so the user can see any errors
    local bash_cmd="cd \"$working_directory\" && $service_command ; echo ; echo \"[$title] Process exited. Press Enter to close.\"; read"

    case "$GUI_TERMINAL" in
        gnome-terminal)
            gnome-terminal --title="$title" -- bash -c "$bash_cmd" &
            ;;
        xfce4-terminal)
            xfce4-terminal --title="$title" -e "bash -c '$bash_cmd'" &
            ;;
        mate-terminal)
            mate-terminal --title="$title" -e "bash -c '$bash_cmd'" &
            ;;
        lxterminal)
            lxterminal --title="$title" -e "bash -c '$bash_cmd'" &
            ;;
        konsole)
            konsole --title "$title" -e bash -c "$bash_cmd" &
            ;;
        xterm)
            xterm -title "$title" -e bash -c "$bash_cmd" &
            ;;
        x-terminal-emulator)
            x-terminal-emulator -T "$title" -e "bash -c '$bash_cmd'" &
            ;;
    esac
}

run_dev() {
    check_env_files
    setup_venv
    setup_frontend

    echo -e "${COLOR_GREEN}Starting Local Development Mode...${COLOR_NC}"

    if detect_gui_terminal; then
        echo -e "${COLOR_GREEN}Opening Backend, Worker, and Frontend in separate terminals using ${GUI_TERMINAL}...${COLOR_NC}"
        
        start_service_terminal "MLS_Backend_API" "$SCRIPT_DIR/backend" \
            "../backend/.venv/bin/python -m uvicorn app.main:app --reload --port 8000"
            
        start_service_terminal "MLS_Celery_Worker" "$SCRIPT_DIR" \
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. backend/.venv/bin/python -m celery -A ai_workers.core.celery_app worker --loglevel=info --pool=solo --concurrency=1"
            
        start_service_terminal "MLS_Frontend" "$SCRIPT_DIR/frontend" "npm run dev"
        
        echo -e "${COLOR_GREEN}Three service terminals opened.${COLOR_NC}"
        echo -e "- Backend API: http://127.0.0.1:8000 (Docs: http://127.0.0.1:8000/docs)"
        echo -e "- Frontend:    http://localhost:5173"
        echo -e "${COLOR_YELLOW}Returning to menu...${COLOR_NC}"
        return
    fi

    echo -e "${COLOR_YELLOW}No supported graphical terminal detected; using combined terminal mode.${COLOR_NC}"
    
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
    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. backend/.venv/bin/python -m celery -A ai_workers.core.celery_app worker --loglevel=info --pool=solo --concurrency=1 &
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
    if [ ! -d "backend/.venv" ]; then
        create_venv
    fi
    install_python_deps
    echo -e "${COLOR_YELLOW}Installing/updating frontend dependencies...${COLOR_NC}"
    (cd frontend && npm install)
    echo -e "${COLOR_GREEN}Dependencies updated successfully.${COLOR_NC}"
}

show_menu() {
    echo -e "${COLOR_BLUE}===================================================${COLOR_NC}"
    echo -e "${COLOR_YELLOW}    MULTIMODAL LECTURE SUMMARIZER - LAUNCHER       ${COLOR_NC}"
    echo -e "${COLOR_BLUE}===================================================${COLOR_NC}"
    echo -e "  ${COLOR_GREEN}1) Local Dev Mode (Backend + Celery Worker + Frontend)${COLOR_NC}"
    echo -e "  ${COLOR_YELLOW}2) Stop All Local Services${COLOR_NC}"
    echo -e "  ${COLOR_MAGENTA}3) Docker Compose Mode (All services in Docker)${COLOR_NC}"
    echo -e "  ${COLOR_RED}4) Reset DB & R2 Storage${COLOR_NC}"
    echo -e "  ${COLOR_BLUE}5) Install / Update Dependencies${COLOR_NC}"
    echo -e "  6) Exit (Stop services & Quit)"
    echo -e "${COLOR_BLUE}===================================================${COLOR_NC}"
    read -p "Select an option (1-6): " CHOICE
    case "$CHOICE" in
        1) run_dev ; show_menu ;;
        2) stop_services ; show_menu ;;
        3) run_docker ;;
        4) reset_system ; show_menu ;;
        5) install_deps ; show_menu ;;
        6) stop_services ; exit 0 ;;
        *) echo -e "${COLOR_RED}Invalid option!${COLOR_NC}" ; show_menu ;;
    esac
}

# Handle command line arguments
case "$1" in
    --dev|-d)
        run_dev
        ;;
    --stop|-s)
        stop_services
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
