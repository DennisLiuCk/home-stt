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
import os
import site
import subprocess
import sys
from ctypes import wintypes

from pynput.keyboard import Key

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
        print(f"[stt] clipboard: GlobalAlloc({size}) failed (errno={err})",
              file=sys.stderr, flush=True)
        return False

    ptr = _kernel32.GlobalLock(h_mem)
    if not ptr:
        err = ctypes.get_last_error()
        _kernel32.GlobalFree(h_mem)
        print(f"[stt] clipboard: GlobalLock failed (errno={err})",
              file=sys.stderr, flush=True)
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
        print(f"[stt] clipboard: OpenClipboard failed after {retries} "
              f"attempts (errno={err})", file=sys.stderr, flush=True)
        return False

    try:
        _user32.EmptyClipboard()
        if not _user32.SetClipboardData(_CF_UNICODETEXT, h_mem):
            err = ctypes.get_last_error()
            _kernel32.GlobalFree(h_mem)  # ownership not transferred on failure
            print(f"[stt] clipboard: SetClipboardData failed (errno={err})",
                  file=sys.stderr, flush=True)
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


class WindowsPasteboard(Pasteboard):
    default_trigger_keys = {Key.alt_gr, Key.ctrl_r}

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
        sequence = [
            (_VK_CONTROL, False),
            (_VK_V,       False),
            (_VK_V,       True),
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
            print(f"[stt] SendInput partial: sent {sent}/{n} events "
                  f"(text on clipboard, press Ctrl+V manually)",
                  file=sys.stderr, flush=True)
            return False
        return True
