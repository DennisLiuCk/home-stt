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

import logging
import subprocess
import sys

from pynput.keyboard import Key

from stt_platform import Pasteboard

logger = logging.getLogger("stt.platform")


# AppleScript fallback that asks System Events to send Cmd+V to the focused
# process. System Events is a stable macOS scripting bridge; its
# Accessibility grant does not depend on which Python binary we are using.
_PASTE_APPLESCRIPT = (
    'tell application "System Events" '
    'to keystroke "v" using command down'
)

# v0.7.5: Cmd+C parallel for voice-edit selection capture. Same scripting
# bridge, same Accessibility constraints as the paste path.
_COPY_APPLESCRIPT = (
    'tell application "System Events" '
    'to keystroke "c" using command down'
)


# v0.7.2: cached NSPasteboard reference. Lazy-loaded on first set_text()
# call rather than at module import — keeps the daemon startable on
# systems where AppKit import is slow or fails (CI without graphics
# stack, headless tests). If AppKit isn't available we fall through to
# the pbcopy subprocess path used pre-v0.7.2.
_NS_PASTEBOARD = None
_NS_PASTEBOARD_TYPE = None
_NS_PASTEBOARD_PROBED = False


def _try_load_nspasteboard() -> bool:
    """Probe whether NSPasteboard via PyObjC AppKit is available. Memoised.

    PyObjC's AppKit is a transitive dep through Quartz (which the IME-safe
    paste path imports), but AppKit specifically is its own subpackage and
    may be missing on minimal installs. Falling back to pbcopy keeps the
    daemon working on those — at the cost of the ~20-50 ms subprocess
    spawn that NSPasteboard was meant to eliminate.
    """
    global _NS_PASTEBOARD, _NS_PASTEBOARD_TYPE, _NS_PASTEBOARD_PROBED
    if _NS_PASTEBOARD_PROBED:
        return _NS_PASTEBOARD is not None
    _NS_PASTEBOARD_PROBED = True
    try:
        from AppKit import NSPasteboard, NSPasteboardTypeString
        _NS_PASTEBOARD = NSPasteboard.generalPasteboard()
        _NS_PASTEBOARD_TYPE = NSPasteboardTypeString
        return True
    except Exception as e:
        # Don't print at module import — only at first use, so users who
        # never actually paste don't see noise.
        logger.warning("clipboard: AppKit NSPasteboard unavailable (%s); "
                       "falling back to pbcopy subprocess (slower but functional)", e)
        return False


def _set_clipboard_via_pbcopy(text: str) -> bool:
    """Legacy v0.7.1 path: pbcopy subprocess. Kept as fallback when
    NSPasteboard / AppKit isn't available on the host."""
    proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    proc.communicate(input=text.encode("utf-8"))
    if proc.returncode != 0:
        logger.warning("pbcopy failed (rc=%d)", proc.returncode)
        return False
    return True


def _set_clipboard_via_nspasteboard(text: str) -> bool:
    """v0.7.2 fast path: direct NSPasteboard via PyObjC.

    setString_forType_ is synchronous — by the time it returns, the
    pasteboard's changeCount has incremented and the OS-side pasteboard
    daemon has the new contents. No settle-sleep needed.
    """
    try:
        # clearContents() must precede setString_forType_ per Apple docs;
        # without it, multiple representations from a prior write can
        # linger and shadow the new value depending on paste-side query.
        _NS_PASTEBOARD.clearContents()
        ok = _NS_PASTEBOARD.setString_forType_(text, _NS_PASTEBOARD_TYPE)
        if not ok:
            logger.warning("clipboard: NSPasteboard setString returned False; "
                          "falling back to pbcopy for this write")
            return _set_clipboard_via_pbcopy(text)
        return True
    except Exception as e:
        logger.warning("clipboard: NSPasteboard failed (%s); "
                       "falling back to pbcopy for this write", e)
        return _set_clipboard_via_pbcopy(text)


def _get_clipboard_via_nspasteboard() -> str | None:
    """v0.7.5: read clipboard via PyObjC for voice-edit selection capture.

    Returns the string, or None on empty/non-text clipboard or any error.
    Falls back to `pbpaste` subprocess when AppKit isn't importable.
    """
    if _try_load_nspasteboard():
        try:
            value = _NS_PASTEBOARD.stringForType_(_NS_PASTEBOARD_TYPE)
            if value is None:
                return None  # empty or non-text format
            return str(value)
        except Exception as e:
            logger.warning("clipboard: NSPasteboard read failed (%s); "
                          "falling back to pbpaste for this read", e)
    # pbpaste fallback path — works without AppKit
    try:
        proc = subprocess.run(
            ["pbpaste"], capture_output=True, text=True, check=False, timeout=2,
        )
    except subprocess.TimeoutExpired:
        logger.warning("clipboard: pbpaste timed out (2s)")
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout
    return out if out else None


def _clipboard_has_nontext_via_nspasteboard() -> bool:
    """v0.8.0: True iff the pasteboard holds a non-empty set of types but no
    readable string type (image, file URLs, RTF-only) — i.e. get_text()
    returns None while the pasteboard is not actually empty. Used by
    voice-edit to avoid clobbering such content. Returns False when AppKit
    is unavailable or on any error ('assume safe' — the pbpaste fallback
    cannot introspect types, so we never block voice-edit on a probe miss)."""
    if not _try_load_nspasteboard():
        return False
    try:
        types = _NS_PASTEBOARD.types()
        if types is None or len(types) == 0:
            return False  # empty pasteboard — safe to overwrite
        # A readable string is present → get_text() round-trips it; not
        # "non-text" for our purposes.
        if _NS_PASTEBOARD.stringForType_(_NS_PASTEBOARD_TYPE) is not None:
            return False
        return True
    except Exception as e:
        logger.warning("clipboard: NSPasteboard type probe failed (%s)", e)
        return False


class MacOSPasteboard(Pasteboard):
    default_trigger_keys = {Key.alt_r}
    # v0.7.5 voice-edit default — Right Command. Symmetric to Right
    # Option dictate trigger (both next to space bar), exists on all
    # Mac keyboards including MacBook (unlike F13 which only exists
    # on Magic Keyboard with Numpad), and doesn't interfere with
    # Option-dead-key composition (Left Option does — would block
    # typing é/è/ñ etc.).
    default_edit_trigger_keys = {Key.cmd_r}

    # Virtual key codes for the Quartz path. Constants don't change.
    _CMD_KEYCODE = 55  # kVK_Command (left command — either side works)
    _V_KEYCODE   = 9   # kVK_ANSI_V
    _C_KEYCODE   = 8   # kVK_ANSI_C  (v0.7.5 voice-edit selection capture)

    def __init__(self) -> None:
        # Decide the paste path ONCE at construction. Re-probing on every
        # paste would be wasted work; if the user grants Accessibility
        # later they can restart the daemon to upgrade the path.
        self._has_ax = self._probe_ax_trust()
        if self._has_ax:
            # Lazy-import Quartz only if we'll actually use it — saves
            # ~30ms startup on unprivileged daemons that fall through to
            # osascript.
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
        """Place text on the system pasteboard.

        v0.7.2: prefers NSPasteboard via PyObjC (~1 ms, synchronous).
        Falls back to `pbcopy` subprocess (~20-50 ms, also synchronous
        after process exit) when AppKit isn't importable — minimal installs,
        custom Python builds without PyObjC, etc. Probe cost paid once
        on first use, then memoised.
        """
        if _try_load_nspasteboard():
            return _set_clipboard_via_nspasteboard(text)
        return _set_clipboard_via_pbcopy(text)

    def paste(self) -> bool:
        if self._has_ax:
            return self._send_cmd_combo_via_quartz(self._V_KEYCODE)
        return self._paste_via_osascript()

    def get_text(self) -> str | None:
        """v0.7.5: read pasteboard for voice-edit selection capture."""
        return _get_clipboard_via_nspasteboard()

    def has_nontext_content(self) -> bool:
        """v0.8.0: True iff the pasteboard holds non-text content (image,
        file URLs, RTF-only) so voice-edit can decline rather than destroy
        it. False when AppKit is unavailable (the pbpaste fallback can't
        introspect types — degrade to never-block)."""
        return _clipboard_has_nontext_via_nspasteboard()

    def clipboard_seqno(self) -> int | None:
        """v0.7.5: NSPasteboard.changeCount monotonically increases on
        every modification. Returns None only if AppKit isn't available
        — in that case voice-edit falls back to a 'no selection detected'
        result, which is the safe degradation."""
        if not _try_load_nspasteboard():
            return None
        try:
            return int(_NS_PASTEBOARD.changeCount())
        except Exception as e:
            logger.warning("clipboard: NSPasteboard.changeCount failed (%s)", e)
            return None

    def simulate_copy(self) -> bool:
        """v0.7.5: Send Cmd+C so the focused app puts its selection on the
        pasteboard. Mirrors `paste()` — Quartz IME-safe path when
        Accessibility granted, osascript fallback otherwise."""
        if self._has_ax:
            return self._send_cmd_combo_via_quartz(self._C_KEYCODE)
        return self._copy_via_osascript()

    # -- Path 1: Quartz CGEvent (preferred, IME-safe) -----------------------

    def _send_cmd_combo_via_quartz(self, letter_keycode: int) -> bool:
        """Shared Cmd+letter CGEvent sender — used by paste() (Cmd+V) and
        simulate_copy() (Cmd+C). v0.7.5 extracted from the original
        _paste_via_quartz; behaviour unchanged for paste path.

        Posts at kCGAnnotatedSessionEventTap (after the IME's event tap)
        so Chinese / Japanese / Korean IMEs do not get a chance to consume
        the keystroke. CGEventPost returns void; we trust the
        AXIsProcessTrusted check from __init__."""
        # Cmd down — no flag yet (no key is "modified" by Cmd being pressed).
        cmd_down = self._cg_create(None, self._CMD_KEYCODE, True)
        self._cg_post(self._event_tap, cmd_down)
        # Letter down WITH Cmd flag set on the event. Apps that check
        # [NSEvent modifierFlags] see Cmd held; apps that rely on event
        # ordering see Cmd-down first then letter-down.
        letter_down = self._cg_create(None, letter_keycode, True)
        self._cg_set_flags(letter_down, self._cmd_mask)
        self._cg_post(self._event_tap, letter_down)
        # Letter up still flagged so the press/release pair share the same flag.
        letter_up = self._cg_create(None, letter_keycode, False)
        self._cg_set_flags(letter_up, self._cmd_mask)
        self._cg_post(self._event_tap, letter_up)
        # Cmd up.
        cmd_up = self._cg_create(None, self._CMD_KEYCODE, False)
        self._cg_post(self._event_tap, cmd_up)
        return True

    # -- Path 2: osascript fallback (no Accessibility for Python) -----------

    def _copy_via_osascript(self) -> bool:
        """v0.7.5: parallel to _paste_via_osascript, sends Cmd+C via
        System Events. Same Accessibility requirements on the osascript
        binary (NOT on the Python binary). Same error-detection logic
        as paste — refactored to share _run_osascript helper."""
        return self._run_osascript(_COPY_APPLESCRIPT, action="copy")

    def _paste_via_osascript(self) -> bool:
        return self._run_osascript(_PASTE_APPLESCRIPT, action="paste")

    def _run_osascript(self, script: str, *, action: str) -> bool:
        """Shared osascript Cmd+letter runner. `action` is "paste" or "copy"
        — used to compose user-facing messages so the hint for paste says
        "press Cmd+V manually" while the copy variant says "voice-edit:
        selection capture failed" (Cmd+C from the daemon side has no
        manual-recovery — the user can just try again)."""
        key_letter = "V" if action == "paste" else "C"
        manual_hint = (
            "text is on clipboard, press Cmd+V manually"
            if action == "paste"
            else "voice-edit: selection capture failed, try again"
        )
        try:
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            # System Events occasionally hangs after macOS upgrades or
            # Stage Manager glitches. Without timeout this would block the
            # transcription thread forever.
            logger.warning("%s timed out (5s) — System Events not "
                          "responding; %s.", action, manual_hint)
            return False
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
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
                logger.warning(
                    "%s failed: macOS denied osascript/System Events the "
                    "Accessibility permission needed to send Cmd+%s. Grant "
                    "it in 系統設定 → 隱私權與安全性 → 輔助使用 (add 'System "
                    "Events').", action, key_letter,
                )
                logger.info("%s blocked — %s.", action, manual_hint)
            else:
                logger.warning("%s failed: %s", action, err)
                logger.info("%s blocked — %s.", action, manual_hint)
            return False
        return True
