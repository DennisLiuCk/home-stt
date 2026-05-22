"""
macOS Pasteboard implementation.

  - Clipboard write: `pbcopy` via subprocess (UTF-8 stdin).
  - Paste keystroke: TWO PATHS, chosen at __init__:

    1. **Quartz CGEvent @ kCGAnnotatedSessionEventTap** (preferred when
       available). Posts Cmd+V at the event tap downstream of all IME
       event taps, so Chinese input methods (注音, 拼音, 倉頡, 嘸蝦米, RIME,
       …) cannot intercept the keystroke. Requires Accessibility
       permission for THIS Python binary; we detect via
       AXIsProcessTrusted() at __init__ time.

    2. **osascript via System Events** (fallback when Python lacks
       Accessibility). The historical v0.2.0 path. Reliable when no IME
       is active; some Chinese IMEs intercept the synthesized keystroke
       because System Events' event tap is upstream of the IME tap.

    The hybrid lets users who haven't granted Accessibility to Python
    keep working (osascript path), while users who do grant it get
    IME-immune paste automatically.

  - Trigger key: Right Option (`Key.alt_r`) by default. On Apple keyboards
    Right Option is unmapped to OS shortcuts in most contexts, making it a
    safer hold-to-talk key than Cmd / Ctrl.

Required macOS permissions for the running Python binary:

  - **Input Monitoring** — for the global keyboard listener
    (pynput.keyboard.Listener uses CGEventTap).
  - **Accessibility** — for the Quartz CGEvent paste path (IME-safe).
    If absent, paste falls back to osascript, which also needs
    Accessibility — but on `/usr/bin/osascript` / System Events, NOT on
    the Python binary. The first osascript run pops a system dialog
    asking to allow that.
  - **Microphone** — for sounddevice audio capture.

Find the real Python binary path (NOT the pyenv shim) via:
    python3 -c "import sys; print(sys.executable)"
"""
from __future__ import annotations

import subprocess
import sys

from pynput.keyboard import Key

from stt_platform import Pasteboard


# AppleScript fallback that asks System Events to send Cmd+V to the focused
# process. System Events is a stable macOS scripting bridge; its
# Accessibility grant does not depend on which Python binary we are using.
_PASTE_APPLESCRIPT = (
    'tell application "System Events" '
    'to keystroke "v" using command down'
)


class MacOSPasteboard(Pasteboard):
    default_trigger_keys = {Key.alt_r}

    # Virtual key codes for the Quartz path. Constants don't change.
    _CMD_KEYCODE = 55  # kVK_Command (left command — either side works)
    _V_KEYCODE   = 9   # kVK_ANSI_V

    def __init__(self) -> None:
        # Decide the paste path ONCE at construction. Re-probing on every
        # paste would be wasted work; if the user grants Accessibility
        # later they can restart the daemon to upgrade the path.
        self._has_ax = self._probe_ax_trust()
        if self._has_ax:
            # Lazy-import Quartz only if we'll actually use it — saves
            # ~30ms startup on Intel Macs / unprivileged daemons that fall
            # through to osascript.
            from Quartz import (
                CGEventCreateKeyboardEvent,
                CGEventPost,
                CGEventSetFlags,
                kCGAnnotatedSessionEventTap,
                kCGEventFlagMaskCommand,
            )

            self._cg_create = CGEventCreateKeyboardEvent
            self._cg_set_flags = CGEventSetFlags
            self._cg_post = CGEventPost
            self._cmd_mask = kCGEventFlagMaskCommand
            self._event_tap = kCGAnnotatedSessionEventTap

    @staticmethod
    def _probe_ax_trust() -> bool:
        """Silent check whether THIS process has Accessibility permission.
        Uses the non-prompting variant — we don't want to spam the user
        with a system dialog every daemon start; we just report status via
        describe_paste_path()."""
        try:
            from ApplicationServices import AXIsProcessTrusted

            return bool(AXIsProcessTrusted())
        except Exception:
            return False

    def describe_paste_path(self) -> str:
        if self._has_ax:
            return "Quartz CGEvent @ AnnotatedSessionEventTap (IME-safe)"
        return (
            "osascript via System Events (Chinese IMEs may intercept Cmd+V; "
            "grant Accessibility to this Python binary to switch to the "
            "IME-safe Quartz path)"
        )

    def set_text(self, text: str) -> bool:
        proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        proc.communicate(input=text.encode("utf-8"))
        if proc.returncode != 0:
            print(
                f"[stt] pbcopy failed (rc={proc.returncode})",
                file=sys.stderr, flush=True,
            )
            return False
        return True

    def paste(self) -> bool:
        if self._has_ax:
            return self._paste_via_quartz()
        return self._paste_via_osascript()

    # -- Path 1: Quartz CGEvent (preferred, IME-safe) -----------------------

    def _paste_via_quartz(self) -> bool:
        """Send Cmd+V via four CGEvents (Cmd-down, V-down, V-up, Cmd-up)
        posted at kCGAnnotatedSessionEventTap. This tap sits AFTER the
        IME's event tap, so Chinese / Japanese / Korean IMEs do not get a
        chance to consume the keystroke. CGEventPost returns void, so we
        cannot directly verify delivery — we trust the AXIsProcessTrusted
        check from __init__: if that was True, post will be honoured."""
        # Cmd down — no flag yet (no key is "modified" by Cmd being pressed).
        cmd_down = self._cg_create(None, self._CMD_KEYCODE, True)
        self._cg_post(self._event_tap, cmd_down)
        # V down WITH Cmd flag set on the event itself. Apps that check
        # [NSEvent modifierFlags] see Cmd held; apps that rely on event
        # ordering see Cmd-down first then V-down.
        v_down = self._cg_create(None, self._V_KEYCODE, True)
        self._cg_set_flags(v_down, self._cmd_mask)
        self._cg_post(self._event_tap, v_down)
        # V up still flagged so the press/release pair share the same flag.
        v_up = self._cg_create(None, self._V_KEYCODE, False)
        self._cg_set_flags(v_up, self._cmd_mask)
        self._cg_post(self._event_tap, v_up)
        # Cmd up.
        cmd_up = self._cg_create(None, self._CMD_KEYCODE, False)
        self._cg_post(self._event_tap, cmd_up)
        return True

    # -- Path 2: osascript fallback (no Accessibility for Python) -----------

    def _paste_via_osascript(self) -> bool:
        try:
            proc = subprocess.run(
                ["osascript", "-e", _PASTE_APPLESCRIPT],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            # System Events occasionally hangs after macOS upgrades or
            # Stage Manager glitches. Without timeout this would block the
            # transcription thread forever.
            print(
                "[stt] paste timed out (5s) — System Events not responding; "
                "text is on clipboard, press Cmd+V manually.",
                flush=True,
            )
            return False
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            # error 1002 = errAEEventNotPermitted: System Events doesn't have
            # Accessibility permission. Match across locales since osascript
            # localizes stderr to the system language.
            err_lower = err.lower()
            denied = (
                "1002" in err
                or "not permitted" in err_lower
                or "not allowed" in err_lower
                or "不允許" in err     # zh-Hant
                or "不允许" in err     # zh-Hans
                or "許可されて" in err   # ja
                or "non autorisé" in err_lower  # fr
                or "no permitido" in err_lower  # es
                or "nicht erlaubt" in err_lower  # de
            )
            if denied:
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
                print(
                    "[stt] paste blocked — text is on clipboard, press Cmd+V manually.",
                    flush=True,
                )
            return False
        return True
