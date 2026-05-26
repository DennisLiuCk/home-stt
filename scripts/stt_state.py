"""Shared daemon state file for the system tray and other observers.

The daemon writes a small JSON file on each state transition; the tray
(or any external tool) polls this file to display current status.

State file location: same temp dir as daemon log files.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

STATE_FILE = Path(tempfile.gettempdir()) / "stt-daemon-state.json"

# Valid states
IDLE = "idle"
RECORDING = "recording"
PROCESSING = "processing"


def write_state(
    state: str,
    *,
    last_text: str | None = None,
    last_lang: str | None = None,
    edit_mode: bool = False,
) -> None:
    """Atomically write daemon state to the shared file."""
    data = {
        "state": state,
        "ts": time.time(),
        "edit_mode": edit_mode,
    }
    if last_text is not None:
        data["last_text"] = last_text[:200]
    if last_lang is not None:
        data["last_lang"] = last_lang
    tmp = STATE_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(str(tmp), str(STATE_FILE))
    except OSError:
        pass


def _daemon_alive() -> bool:
    """Check if the daemon process is alive via PID file."""
    pid_file = Path(__file__).resolve().parent / "stt-daemon.pid"
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return False
    if sys.platform == "win32":
        import subprocess
        try:
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True, text=True, timeout=2,
            )
            return str(pid) in r.stdout
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, PermissionError):
        return False


def read_state() -> dict | None:
    """Read current daemon state. Uses PID file to detect stopped daemon."""
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError):
        data = None

    if not _daemon_alive():
        if data is not None:
            data["state"] = "stopped"
        else:
            return {"state": "stopped", "ts": 0}

    return data


def cleanup() -> None:
    """Remove the state file on daemon exit."""
    try:
        STATE_FILE.unlink(missing_ok=True)
    except OSError:
        pass
