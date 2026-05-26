"""System tray icon for home-stt daemon.

Shows daemon state (idle / recording / processing / stopped) via a
coloured icon in the Windows system tray or macOS menu bar. Right-click
menu provides Start / Stop / Status / Recent / Quit.

Requires: pystray, Pillow (Windows) or rumps (macOS — future).
Launch via `home-stt tray`.
"""
from __future__ import annotations

import subprocess
import sys
import time
import threading
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from stt_state import read_state

# Colours per state (RGBA)
_COLOURS = {
    "idle":       (120, 120, 120, 255),  # grey
    "recording":  (220, 40,  40,  255),  # red
    "processing": (220, 180, 30,  255),  # amber
    "stopped":    (60,  60,  60,  255),  # dark grey
}

_LABELS = {
    "idle":       "home-stt: idle",
    "recording":  "home-stt: recording…",
    "processing": "home-stt: processing…",
    "stopped":    "home-stt: stopped",
}


def _make_icon(state: str):
    """Generate a 64x64 icon: filled circle on transparent background."""
    from PIL import Image, ImageDraw
    colour = _COLOURS.get(state, _COLOURS["stopped"])
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([8, 8, 56, 56], fill=colour)
    if state == "recording":
        draw.ellipse([22, 22, 42, 42], fill=(255, 255, 255, 200))
    return img


def _home_stt_cmd(*args: str) -> None:
    """Run a home-stt CLI subcommand in background."""
    cmd = [sys.executable, str(SCRIPTS_DIR / "home_stt.py")] + list(args)
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        pass


def _on_start(icon, item):
    _home_stt_cmd("start")


def _on_stop(icon, item):
    _home_stt_cmd("stop")


def _on_restart(icon, item):
    _home_stt_cmd("restart")


def _on_status(icon, item):
    if sys.platform == "win32":
        subprocess.Popen(
            ["cmd", "/c", "start", "cmd", "/k",
             sys.executable, str(SCRIPTS_DIR / "home_stt.py"), "status"],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        _home_stt_cmd("status")


def _on_quit(icon, item):
    icon.stop()


def _make_recording_frame(phase: int):
    """Generate a recording-animation frame with pulsing inner ring."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([8, 8, 56, 56], fill=_COLOURS["recording"])
    # Pulse: inner circle radius oscillates between 8 and 14
    r = 8 + (phase % 6) if phase < 6 else 14 - (phase % 6)
    cx, cy = 32, 32
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, 200))
    return img


def _poll_state(icon, interval: float = 0.3) -> None:
    """Background thread: poll daemon state and update icon + tooltip.

    Features:
    - Icon colour changes per state (grey/red/amber/dark)
    - Recording: icon pulses with animation frames
    - Transcription complete: toast notification with the text
    """
    prev_state = None
    anim_phase = 0
    while icon.visible:
        data = read_state()
        state = data["state"] if data else "stopped"

        if state == "recording":
            icon.icon = _make_recording_frame(anim_phase)
            icon.title = "home-stt: recording…"
            anim_phase = (anim_phase + 1) % 12
        elif state != prev_state:
            icon.icon = _make_icon(state)
            icon.title = _LABELS.get(state, f"home-stt: {state}")
            anim_phase = 0
            # Toast notification when transcription completes
            if prev_state in ("processing", "recording") and state == "idle" and data:
                text = data.get("last_text", "")
                if text:
                    try:
                        label = "Voice-Edit" if data.get("edit_mode") else "Dictate"
                        icon.notify(text[:150], f"home-stt {label}")
                    except Exception:
                        pass

        # Auto-quit when daemon stops (so `home-stt stop` closes both)
        if state == "stopped" and prev_state is not None and prev_state != "stopped":
            icon.notify("Daemon stopped", "home-stt")
            time.sleep(2)
            icon.stop()
            return

        prev_state = state
        time.sleep(interval)


def main() -> None:
    try:
        import pystray
    except ImportError:
        print("home-stt tray: pystray not installed. "
              "Install with: pip install pystray Pillow", file=sys.stderr)
        sys.exit(1)

    from pystray import MenuItem as Item

    menu = pystray.Menu(
        Item("Start", _on_start),
        Item("Stop", _on_stop),
        Item("Restart", _on_restart),
        pystray.Menu.SEPARATOR,
        Item("Status", _on_status),
        pystray.Menu.SEPARATOR,
        Item("Quit Tray", _on_quit),
    )

    data = read_state()
    initial_state = data["state"] if data else "stopped"

    icon = pystray.Icon(
        name="home-stt",
        icon=_make_icon(initial_state),
        title=_LABELS.get(initial_state, "home-stt"),
        menu=menu,
    )

    def _on_setup(icon):
        icon.visible = True
        _poll_state(icon)

    icon.run(setup=_on_setup)


if __name__ == "__main__":
    main()
