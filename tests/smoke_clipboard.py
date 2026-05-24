"""Standalone smoke test for the v0.7.2 Win32 ctypes clipboard helper.

NOT a pytest test — calling it touches the real system clipboard, which
would be a destructive side effect in CI. Run manually to verify the
new path works on the live machine:

    python tests/smoke_clipboard.py

What it does:
  1. Snapshot the current clipboard text (so we can restore at end).
  2. Round-trip 5 strings through `_set_clipboard_text` + a Win32
     GetClipboardData read-back. Includes zh-TW, mixed zh-en, identifier
     fidelity, surrogate-pair emoji, and a 1000-char stress string.
  3. Restore original clipboard (best effort — image/RTF clipboards
     can't be restored from this script).
  4. Print pass/fail per case.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from stt_platform_win import _set_clipboard_text  # noqa: E402


# Win32 helpers for read-back (separate from the daemon's write path).
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_CF_UNICODETEXT = 13

_user32.OpenClipboard.restype = ctypes.c_int
_user32.OpenClipboard.argtypes = [wintypes.HWND]
_user32.GetClipboardData.restype = ctypes.c_void_p
_user32.GetClipboardData.argtypes = [ctypes.c_uint]
_user32.IsClipboardFormatAvailable.restype = ctypes.c_int
_user32.IsClipboardFormatAvailable.argtypes = [ctypes.c_uint]
_user32.CloseClipboard.restype = ctypes.c_int
_user32.CloseClipboard.argtypes = []
_kernel32.GlobalLock.restype = ctypes.c_void_p
_kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
_kernel32.GlobalUnlock.restype = ctypes.c_int
_kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]


def _read_clipboard_text() -> str | None:
    """Read CF_UNICODETEXT from clipboard. None if no text-format data."""
    if not _user32.IsClipboardFormatAvailable(_CF_UNICODETEXT):
        return None
    if not _user32.OpenClipboard(None):
        return None
    try:
        handle = _user32.GetClipboardData(_CF_UNICODETEXT)
        if not handle:
            return None
        ptr = _kernel32.GlobalLock(handle)
        if not ptr:
            return None
        try:
            return ctypes.wstring_at(ptr)
        finally:
            _kernel32.GlobalUnlock(handle)
    finally:
        _user32.CloseClipboard()


CASES = [
    ("ascii", "hello world"),
    ("zh_tw_basic", "幫我 review 這個 Python function"),
    ("zh_tw_punctuation", "好的，這個 commit 之後直接 push 到 main，OK？"),
    ("identifier_fidelity", "_USE_TORCH_COMPILE = False  # see text_polisher.py:189"),
    ("emoji_surrogate_pair", "v0.7.2 ships 🚀 cross-platform 中英 🌏"),
    ("stress_1000_chars", "中" * 500 + "x" * 500),
]


def main() -> int:
    print("=" * 60)
    print("v0.7.2 ctypes clipboard smoke test")
    print("=" * 60)
    original = _read_clipboard_text()
    print(f"[snapshot] original clipboard: "
          f"{repr(original)[:80]}{'...' if original and len(original) > 80 else ''}")
    print()
    failures = []
    for name, text in CASES:
        write_ok = _set_clipboard_text(text)
        if not write_ok:
            print(f"[FAIL] {name}: _set_clipboard_text returned False")
            failures.append(name)
            continue
        read_back = _read_clipboard_text()
        match = read_back == text
        status = "PASS" if match else "FAIL"
        print(f"[{status}] {name}: len={len(text)} chars, "
              f"round-trip {'matches' if match else 'MISMATCH'}")
        if not match:
            print(f"        wrote: {text[:60]!r}{'...' if len(text) > 60 else ''}")
            print(f"        read:  {read_back[:60]!r}{'...' if read_back and len(read_back) > 60 else ''}")
            failures.append(name)
    print()
    print("=" * 60)
    if original is not None:
        _set_clipboard_text(original)
        print(f"[restore] clipboard restored to original "
              f"({len(original)} chars)")
    else:
        print("[restore] original clipboard was non-text (image/RTF/empty); "
              "leaving the last test string in place")
    print("=" * 60)
    if failures:
        print(f"FAILED: {len(failures)}/{len(CASES)} — {failures}")
        return 1
    print(f"ALL {len(CASES)} CASES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
