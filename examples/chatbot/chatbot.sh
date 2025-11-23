#!/bin/bash
set -euo pipefail

# Set PYTHONPATH to project root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

REDIS_SERVICE_NAME="redis-server"
REDIS_DAEMON_CMD="redis-server --daemonize yes"
CHATBOT_SCRIPT="app.py"

# Logging
log()  { echo -e "\033[1;34m[INFO]\033[0m  $1"; }   # Blue
warn() { echo -e "\033[1;33m[WARN]\033[0m  $1" >&2; }   # Yellow
fail() { echo -e "\033[1;31m[ERROR]\033[0m $1" >&2; exit 1; }   # Red

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

ensure_redis_installed() {
    if ! command_exists redis-server; then
        log "Redis is not installed. Installing..."
        sudo apt update && sudo apt install -y redis-server || fail "Failed to install Redis."
        log "Redis installed successfully."
    else
        log "Redis is already installed."
    fi
}

ensure_redis_running() {
    if ! pgrep -x "redis-server" >/dev/null; then
        log "Redis is not running. Starting..."
        $REDIS_DAEMON_CMD || fail "Failed to start Redis."
        sleep 1  # Give it time to start
    else
        log "Redis is already running."
    fi
}

check_redis_ping() {
    local response
    response=$(redis-cli ping || echo "FAIL")
    if [[ "$response" != "PONG" ]]; then
        fail "Redis is not responding to PING. Check logs."
    fi
    log "Redis responded to PING."
}

start_chatbot() {
    if [[ ! -f "$CHATBOT_SCRIPT" ]]; then
        fail "Chatbot script '$CHATBOT_SCRIPT' not found."
    fi
    log "Starting chatbot..."
    exec python3 "$CHATBOT_SCRIPT"
}

# Main
main() {
    ensure_redis_installed
    ensure_redis_running
    check_redis_ping
    start_chatbot
}

main "$@"

# chmod +x start_chatbot.sh
# Usage: ./chatbot.sh