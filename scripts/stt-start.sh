#!/usr/bin/env bash
# Start the hold-to-talk voice-to-text daemon in the background.
#
# Launches stt-daemon.py via python3 with stdout/stderr captured to
# $TMPDIR/stt-daemon.log. Refuses to start a second copy if one is
# already running. PID is recorded next to this script.
#
# macOS counterpart of stt-start.ps1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAEMON="$SCRIPT_DIR/stt-daemon.py"
PID_FILE="$SCRIPT_DIR/stt-daemon.pid"
LOG_DIR="${TMPDIR:-/tmp}"
LOG_FILE="$LOG_DIR/stt-daemon.log"
ERR_FILE="$LOG_DIR/stt-daemon.err.log"

if [ ! -f "$DAEMON" ]; then
    echo "Daemon script not found: $DAEMON" >&2
    exit 1
fi

# Already running?
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "STT daemon already running (PID $OLD_PID)."
        exit 0
    fi
    # PID file is stale — clean up before relaunching.
    rm -f "$PID_FILE"
fi

# Fallback scan: someone may have launched the daemon without the PID file.
if pgrep -fl 'stt-daemon\.py' >/dev/null 2>&1; then
    EXISTING=$(pgrep -f 'stt-daemon\.py' | head -1)
    echo "STT daemon already running (PID $EXISTING, no PID file)."
    exit 0
fi

export PYTHONIOENCODING=utf-8

# Launch detached. nohup + & + disown keeps the daemon alive after the
# launching shell exits. Output goes to log files like the Windows version.
nohup python3 -u "$DAEMON" >"$LOG_FILE" 2>"$ERR_FILE" &
DAEMON_PID=$!
disown "$DAEMON_PID" 2>/dev/null || true

echo "$DAEMON_PID" >"$PID_FILE"

echo "STT daemon started (PID $DAEMON_PID)."
echo "Log: $LOG_FILE"
echo "Allow ~10-30s for model load + Metal warmup before first trigger key."
echo "First run also downloads the model (~1.5 GB for large-v3-turbo)."
