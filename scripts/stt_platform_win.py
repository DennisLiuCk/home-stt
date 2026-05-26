"""
Windows Pasteboard implementation.

Extracted from the v0.1.0 single-file daemon. Behaviour is byte-for-byte
identical to v0.1.0 — only the module location changed:

  - NVIDIA cuDNN/cuBLAS DLL discovery (so CTranslate2 finds them) —
    `register_native_libs()`.
  - Clipboard write via PowerShell `Set-Clipboard` with forced UTF-8 stdin —
    `set_text()`.
  - Paste via raw SendInput Ctrl+V — `paste()`. Ctrl is a system modifier
    that IMEs (Bopomofo etc.) never intercept, so the whole clipboard
    contents land atomically.
  - SendInput UNICODE typing fallback (`type_text`) kept module-level for
    debugging / experimentation; not used in the main paste path because
    the Bopomofo IME swallows characters after CJK punctuation like 、.

Module-level Win32 setup is safe here because `build_pasteboard()` only
imports this module when `sys.platform == "win32"`.
"""
from __future__ import annotations

import ctypes
import logging
import os
import site
import subprocess
import sys
from ctypes import wintypes

from pynput.keyboard import Key

logger = logging.getLogger("stt.platform")

from stt_platform import Pasteboard


# ---------------------------------------------------------------------------
# NVIDIA DLL discovery — must run BEFORE the STT backend imports faster_whisper
# (CTranslate2 looks up cuDNN / cuBLAS at import time). The pip wheels for
# `nvidia-cudnn-cu12` etc. ship the .so/.dll in site-packages/nvidia/<lib>/bin/
# rather than on PATH, so we add each one explicitly via os.add_dll_directory()
# and also prepend to PATH (belt-and-braces).
# ---------------------------------------------------------------------------
def _register_nvidia_dlls() -> int:
    bin_dirs: list[str] = []
    roots = [site.getusersitepackages()] + list(site.getsitepackages())
    for sp in roots:
        nv = os.path.join(sp, "nvidia")
        if not os.path.isdir(nv):
            continue
        for sub in os.listdir(nv):
            bin_dir = os.path.join(nv, sub, "bin")
            if os.path.isdir(bin_dir):
                bin_dirs.append(bin_dir)
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(bin_dir)
                    except Exception:
                        pass
    if bin_dirs:
        os.environ["PATH"] = (os.pathsep.join(bin_dirs)
                              + os.pathsep + os.environ.get("PATH", ""))
    return len(bin_dirs)


# ---------------------------------------------------------------------------
# Win32 SendInput structures + unicode typing helper.
# pynput.Controller.type() falls back to virtual-key presses for ASCII
# letters, which the Bopomofo IME swallows as zhuyin keystrokes. The
# `type_text` helper forces every character through KEYEVENTF_UNICODE so
# the IME layer never sees them. Retained as a fallback; the main path uses
# clipboard + Ctrl+V (see paste()) because IME also interferes with type
# after CJK punctuation like 、.
# ---------------------------------------------------------------------------
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_INPUT_KEYBOARD    = 1
_KEYEVENTF_UNICODE = 0x0004
_KEYEVENTF_KEYUP   = 0x0002
_ULONG_PTR = (ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8
              else ctypes.c_ulong)

# ---------------------------------------------------------------------------
# v0.7.2: direct Win32 clipboard API. Replaces the v0.7.1 path that spawned
# `powershell.exe -Command "Set-Clipboard ..."` per paste — cold powershell
# startup was 100-300 ms per call, plus the daemon then slept 150 ms
# afterwards to let the clipboard publish settle. Total ~250-450 ms of
# pure latency per dictation, dominating short-utterance UX.
#
# Direct API: ~1-5 ms. Synchronous — SetClipboardData returns only after
# the OS-side clipboard daemon publishes the new contents, so no
# settle-sleep needed (daemon trims sleep to 20 ms for keystroke timing).
#
# CF_UNICODETEXT contract: data must be a GMEM_MOVEABLE handle to a
# UTF-16 LE buffer with a terminating null wchar. Clipboard takes
# ownership of the handle on successful SetClipboardData — so we MUST
# GlobalFree on the failure path but NOT on success. Mixing this up
# either leaks one handle per paste (success path freeing) or causes the
# clipboard daemon to read freed memory on the failure path. The retry
# loop on OpenClipboard handles the common case where another process
# (clipboard manager, paste helper) holds the lock briefly.
# ---------------------------------------------------------------------------
_GMEM_MOVEABLE  = 0x0002
_CF_UNICODETEXT = 13

_kernel32.GlobalAlloc.restype  = ctypes.c_void_p
_kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
_kernel32.GlobalLock.restype   = ctypes.c_void_p
_kernel32.GlobalLock.argtypes  = [ctypes.c_void_p]
_kernel32.GlobalUnlock.restype = ctypes.c_int
_kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
_kernel32.GlobalFree.restype   = ctypes.c_void_p
_kernel32.GlobalFree.argtypes  = [ctypes.c_void_p]

_user32.OpenClipboard.restype     = ctypes.c_int
_user32.OpenClipboard.argtypes    = [wintypes.HWND]
_user32.EmptyClipboard.restype    = ctypes.c_int
_user32.EmptyClipboard.argtypes   = []
_user32.SetClipboardData.restype  = ctypes.c_void_p
_user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
_user32.CloseClipboard.restype    = ctypes.c_int
_user32.CloseClipboard.argtypes   = []

# v0.7.5: read-side of the clipboard for voice-edit selection capture.
# GetClipboardData returns a handle to the global memory; sequence number
# bumps on every modification (no OpenClipboard needed, so it's a cheap
# poll). Both APIs documented at MSDN under "Clipboard Functions".
_user32.GetClipboardData.restype           = ctypes.c_void_p
_user32.GetClipboardData.argtypes          = [ctypes.c_uint]
_user32.GetClipboardSequenceNumber.restype = wintypes.DWORD
_user32.GetClipboardSequenceNumber.argtypes = []


def _set_clipboard_text(text: str, *, retries: int = 5,
                        retry_delay_s: float = 0.01) -> bool:
    """Write text to the system clipboard via direct Win32 API.

    Returns True iff the clipboard now contains exactly `text`. False on
    any failure — caller (WindowsPasteboard.set_text) prints the
    user-facing error line; this helper logs the low-level errno to
    stderr for debugging without polluting the main log.
    """
    # UTF-16 LE + terminating null wchar. Encode then append two null
    # bytes (one null wchar = two bytes in UTF-16). Python encodes
    # surrogate pairs correctly for chars outside the BMP.
    encoded = text.encode("utf-16-le") + b"\x00\x00"
    size = len(encoded)

    h_mem = _kernel32.GlobalAlloc(_GMEM_MOVEABLE, size)
    if not h_mem:
        err = ctypes.get_last_error()
        logger.warning("clipboard: GlobalAlloc(%d) failed (errno=%d)", size, err)
        return False

    ptr = _kernel32.GlobalLock(h_mem)
    if not ptr:
        err = ctypes.get_last_error()
        _kernel32.GlobalFree(h_mem)
        logger.warning("clipboard: GlobalLock failed (errno=%d)", err)
        return False
    try:
        ctypes.memmove(ptr, encoded, size)
    finally:
        _kernel32.GlobalUnlock(h_mem)

    # OpenClipboard can transiently fail when another process holds the
    # clipboard (clipboard managers, paste tools). Brief retry — the
    # contention window is typically <10 ms.
    import time as _time
    opened = False
    for attempt in range(retries):
        if _user32.OpenClipboard(None):
            opened = True
            break
        _time.sleep(retry_delay_s)
    if not opened:
        err = ctypes.get_last_error()
        _kernel32.GlobalFree(h_mem)
        logger.warning("clipboard: OpenClipboard failed after %d attempts "
                       "(errno=%d)", retries, err)
        return False

    try:
        _user32.EmptyClipboard()
        if not _user32.SetClipboardData(_CF_UNICODETEXT, h_mem):
            err = ctypes.get_last_error()
            _kernel32.GlobalFree(h_mem)  # ownership not transferred on failure
            logger.warning("clipboard: SetClipboardData failed (errno=%d)", err)
            return False
        # Success — clipboard now owns h_mem. Do NOT GlobalFree.
        return True
    finally:
        _user32.CloseClipboard()


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", _ULONG_PTR)]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", _ULONG_PTR)]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("ii", _INPUTUNION)]


def type_text(text: str) -> None:
    """Fallback typing path: SendInput + KEYEVENTF_UNICODE per character.
    Currently NOT used in the main path because the Bopomofo IME enters a
    punctuation buffer after CJK punctuation like 、 (U+3001) and swallows
    every character that follows in the same batch."""
    inputs: list[_INPUT] = []
    for ch in text:
        cp = ord(ch)
        units = ((cp,) if cp <= 0xFFFF
                 else (0xD800 + ((cp - 0x10000) >> 10),
                       0xDC00 + ((cp - 0x10000) & 0x3FF)))
        for u in units:
            for flag in (0, _KEYEVENTF_KEYUP):
                inp = _INPUT()
                inp.type = _INPUT_KEYBOARD
                inp.ii.ki.wVk = 0
                inp.ii.ki.wScan = u
                inp.ii.ki.dwFlags = _KEYEVENTF_UNICODE | flag
                inputs.append(inp)
    if not inputs:
        return
    n = len(inputs)
    arr = (_INPUT * n)(*inputs)
    _user32.SendInput(n, arr, ctypes.sizeof(_INPUT))


# Virtual-key codes for the Ctrl+V combo.
_VK_CONTROL = 0x11
_VK_V       = 0x56
_VK_C       = 0x43  # v0.7.5: Ctrl+C for voice-edit selection capture


def _get_clipboard_text(*, retries: int = 5,
                        retry_delay_s: float = 0.01) -> str | None:
    """Read the current clipboard's CF_UNICODETEXT contents.

    Returns the string or None on any failure (empty clipboard, non-text
    format, transient OpenClipboard contention). Mirrors `_set_clipboard_text`'s
    retry policy — clipboard managers / paste tools may briefly hold the
    lock so retry ~50 ms total before giving up.

    Caller (voice-edit selection capture) treats None as 'no readable
    selection' — same as 'app had no selection to copy'.
    """
    import time as _time
    opened = False
    for _ in range(retries):
        if _user32.OpenClipboard(None):
            opened = True
            break
        _time.sleep(retry_delay_s)
    if not opened:
        return None

    try:
        h_mem = _user32.GetClipboardData(_CF_UNICODETEXT)
        if not h_mem:
            # Clipboard empty or holds a non-text format (image, file list,
            # RTF only). Not an error — voice-edit will treat as 'no
            # selection' and fail-beep.
            return None
        ptr = _kernel32.GlobalLock(h_mem)
        if not ptr:
            return None
        try:
            # wstring_at copies a NUL-terminated UTF-16 string out of the
            # global handle. CF_UNICODETEXT is contractually NUL-terminated
            # so this is safe; otherwise we'd need a length probe.
            return ctypes.wstring_at(ptr)
        finally:
            _kernel32.GlobalUnlock(h_mem)
    finally:
        _user32.CloseClipboard()


class WindowsPasteboard(Pasteboard):
    default_trigger_keys = {Key.alt_gr, Key.ctrl_r}
    # v0.7.5 voice-edit default — F13 is almost universally present on
    # full-size Win keyboards and unbound from OS shortcuts. TKL / laptop
    # users without F13 override via EDIT_TRIGGER_KEYS in stt-daemon.py.
    default_edit_trigger_keys = {Key.f13}

    def register_native_libs(self) -> int:
        return _register_nvidia_dlls()

    def set_text(self, text: str) -> bool:
        """Place text on the Windows clipboard via direct Win32 API.

        v0.7.2: replaced PowerShell `Set-Clipboard` subprocess (100-300
        ms cold start, async publish) with `_set_clipboard_text` ctypes
        path (~1-5 ms, synchronous). UTF-16 LE encoding native to the
        CF_UNICODETEXT contract — no cp950 / zh-TW locale mojibake risk
        that the PowerShell path needed `[Console]::InputEncoding =
        UTF8` to work around.
        """
        return _set_clipboard_text(text)

    def paste(self) -> bool:
        """Send Ctrl+V via raw SendInput. Ctrl is a system modifier and IMEs
        don't intercept Ctrl-combos, so a single Ctrl+V pastes the whole
        clipboard content atomically (no per-character IME interference)."""
        return self._send_ctrl_combo(_VK_V, label="Ctrl+V")

    def get_text(self) -> str | None:
        """v0.7.5: read clipboard for voice-edit selection capture."""
        return _get_clipboard_text()

    def clipboard_seqno(self) -> int | None:
        """v0.7.5: GetClipboardSequenceNumber bumps on every modification.
        Cheap to poll (no OpenClipboard required). Returns None only if
        the ctypes call raises — practically never on a working Win32
        runtime."""
        try:
            return int(_user32.GetClipboardSequenceNumber())
        except Exception as e:
            logger.warning("clipboard: GetClipboardSequenceNumber failed (%s)", e)
            return None

    def simulate_copy(self) -> bool:
        """v0.7.5: Send Ctrl+C via SendInput so the focused app puts its
        selection on the clipboard. Mirrors `paste()` — same IME-immunity
        argument (Ctrl is a system modifier IMEs don't intercept)."""
        return self._send_ctrl_combo(_VK_C, label="Ctrl+C")

    def _send_ctrl_combo(self, vk_letter: int, *, label: str) -> bool:
        """Shared helper for Ctrl+letter SendInput (Ctrl+V for paste,
        Ctrl+C for voice-edit selection capture). Extracted v0.7.5 — the
        4-event sequence is identical, only the letter VK differs."""
        sequence = [
            (_VK_CONTROL, False),
            (vk_letter,   False),
            (vk_letter,   True),
            (_VK_CONTROL, True),
        ]
        inputs: list[_INPUT] = []
        for vk, up in sequence:
            inp = _INPUT()
            inp.type = _INPUT_KEYBOARD
            inp.ii.ki.wVk = vk
            inp.ii.ki.wScan = 0
            inp.ii.ki.dwFlags = _KEYEVENTF_KEYUP if up else 0
            inputs.append(inp)
        n = len(inputs)
        arr = (_INPUT * n)(*inputs)
        sent = _user32.SendInput(n, arr, ctypes.sizeof(_INPUT))
        if sent != n:
            logger.warning("SendInput partial: sent %d/%d events (%s failed)",
                          sent, n, label)
            return False
        return True
