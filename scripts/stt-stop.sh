#!/usr/bin/env bash
# Stop the hold-to-talk voice-to-text daemon.
#
# Reads the recorded PID file first, falls back to scanning processes
# for stt-daemon.py if the file is missing or stale.
#
# macOS counterpart of stt-stop.ps1.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/stt-daemon.pid"

killed=0
primary_pid=""

if [ -f "$PID_FILE" ]; then
    primary_pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$primary_pid" ] && kill -0 "$primary_pid" 2>/dev/null; then
        kill "$primary_pid" 2>/dev/null || true
        # Wait briefly, escalate to SIGKILL if still alive.
        for _ in 1 2 3 4 5; do
            kill -0 "$primary_pid" 2>/dev/null || break
            sleep 0.2
        done
        if kill -0 "$primary_pid" 2>/dev/null; then
            kill -9 "$primary_pid" 2>/dev/null || true
        fi
        echo "Stopped daemon (PID $primary_pid)."
        killed=1
    fi
    rm -f "$PID_FILE"
fi

# Give the OS a beat so the fallback scan doesn't pick the just-killed PID.
sleep 0.25

# Fallback: any orphan python running stt-daemon.py that wasn't already killed.
ORPHANS=$(pgrep -f 'stt-daemon\.py' 2>/dev/null || true)
for pid in $ORPHANS; do
    if [ "$pid" != "$primary_pid" ]; then
        kill "$pid" 2>/dev/null || true
        echo "Stopped orphan daemon (PID $pid)."
        killed=1
    fi
done

if [ "$killed" = "0" ]; then
    echo "STT daemon was not running."
fi
