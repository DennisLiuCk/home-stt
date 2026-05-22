"""
macOS Pasteboard implementation.

  - Clipboard write: `pbcopy` via subprocess (UTF-8 stdin).
  - Paste keystroke: Cmd+V via `osascript -e 'tell application "System Events"
    to keystroke "v" using command down'`. We deliberately AVOID pynput's
    `keyboard.Controller` here because Accessibility permission attaches to
    the running binary — under pyenv that path can drift between Python
    versions, and the silent-failure mode when permission is missing is
    really hard to diagnose (clipboard gets the text, paste keystroke is
    just dropped). osascript's keystrokes attach to a stable system binary
    (/usr/bin/osascript -> System Events), and the first invocation pops a
    clear macOS dialog asking for Accessibility access. Permission grant
    is one-time and survives Python version switches.
  - Trigger key: Right Option (`Key.alt_r`) by default. On Apple keyboards
    Right Option is unmapped to OS shortcuts in most contexts, making it a
    safer hold-to-talk key than Cmd / Ctrl.

Required macOS permissions:

  - **Input Monitoring** (on the Python binary) — for the global keyboard
    listener (pynput.keyboard.Listener uses CGEventTap).
  - **Accessibility** (on `osascript` / System Events — auto-prompted on
    first paste) — for sending the Cmd+V keystroke.
  - **Microphone** (on the Python binary) — for `sounddevice` audio capture.

Find the real Python binary path (NOT the pyenv shim) via:
    python3 -c "import sys; print(sys.executable)"
"""
from __future__ import annotations

import subprocess
import sys

from pynput.keyboard import Key

from stt_platform import Pasteboard


# AppleScript that asks System Events to send Cmd+V to the focused process.
# System Events is a stable macOS scripting bridge; its Accessibility grant
# does not depend on which Python binary we happen to be running under.
_PASTE_APPLESCRIPT = (
    'tell application "System Events" '
    'to keystroke "v" using command down'
)


class MacOSPasteboard(Pasteboard):
    default_trigger_keys = {Key.alt_r}

    def set_text(self, text: str) -> None:
        proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        proc.communicate(input=text.encode("utf-8"))

    def paste(self) -> None:
        # ~30-50ms fork + AppleScript exec overhead vs. ~1ms for in-process
        # CGEventPost, but reliability >> latency here — the previous in-
        # process path silently dropped keystrokes when Accessibility was
        # granted to the wrong binary (common under pyenv).
        proc = subprocess.run(
            ["osascript", "-e", _PASTE_APPLESCRIPT],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            # error 1002 = errAEEventNotPermitted: System Events doesn't have
            # Accessibility permission. Surface a single, actionable line to
            # the main log instead of letting it disappear into err.log.
            if "1002" in err or "not permitted" in err.lower() or "不允許" in err:
                print(
                    "[stt] paste failed: macOS denied osascript/System Events the "
                    "Accessibility permission needed to send Cmd+V. Grant it in "
                    "系統設定 → 隱私權與安全性 → 輔助使用 (add 'System Events').",
                    file=sys.stderr, flush=True,
                )
                print(
                    "[stt] paste blocked — text is on clipboard, press Cmd+V manually for now.",
                    flush=True,
                )
            else:
                print(f"[stt] paste failed: {err}",
                      file=sys.stderr, flush=True)
