"""Shared daemon state file for the system tray and other observers.

The daemon writes a small JSON file on each state transition; the tray
(or any external tool) polls this file to display current status.

State file location: same temp dir as daemon log files.
"""
from __future__ import annotations

import json
import os
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


def read_state() -> dict | None:
    """Read current daemon state. Returns None if file doesn't exist or is stale."""
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        age = time.time() - data.get("ts", 0)
        if age > 30:
            data["state"] = "stopped"
        return data
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def cleanup() -> None:
    """Remove the state file on daemon exit."""
    try:
        STATE_FILE.unlink(missing_ok=True)
    except OSError:
        pass
