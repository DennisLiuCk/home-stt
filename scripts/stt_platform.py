"""
Platform abstraction layer.

The daemon stays platform-agnostic; per-OS concerns (clipboard write, paste
keystroke simulation, default global trigger keys, native-lib registration)
live behind the `Pasteboard` interface. Concrete implementations live in
`stt_platform_win.py` and `stt_platform_mac.py` and are lazy-imported by
`build_pasteboard()` so that Windows-only ctypes never load on macOS, and
vice versa.

Add a new platform by:
  1. Creating `stt_platform_<os>.py` with a `Pasteboard` subclass.
  2. Adding a branch in `build_pasteboard()`.
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod


class Pasteboard(ABC):
    """Per-platform clipboard write + paste simulation + trigger keys.

    Subclasses set `default_trigger_keys` to a set of `pynput.keyboard.Key`
    members (or characters) that the daemon listens to as hold-to-record
    triggers.
    """

    default_trigger_keys: set  # must be set by subclasses

    @abstractmethod
    def set_text(self, text: str) -> bool:
        """Place `text` on the system clipboard. Return True iff the write
        is confirmed successful (subprocess returncode == 0 or equivalent).
        On False the daemon skips paste() and the success beep, and logs a
        clipboard-write failure — implementations should print an
        actionable diagnostic before returning False."""

    @abstractmethod
    def paste(self) -> bool:
        """Simulate the paste keystroke (Ctrl+V on Win, Cmd+V on Mac) so the
        focused application receives the clipboard contents as if the user
        had typed it. Return True iff the keystroke was delivered. On False
        the daemon skips the success beep/log and treats the text as
        'available on clipboard, user must paste manually' — implementations
        should print an actionable diagnostic to the main log before
        returning False (e.g. macOS Accessibility-permission hint)."""

    def register_native_libs(self) -> int:
        """Optional: register native libs needed by STT backends (NVIDIA
        cuDNN/cuBLAS DLLs on Windows). Returns the number of paths added.
        Default no-op for platforms without DLL search-path quirks."""
        return 0

    def describe_paste_path(self) -> str:
        """Optional one-line summary of which paste mechanism is active, for
        the daemon's startup log. Empty string means "no extra detail" —
        platforms with multiple paste paths (e.g. macOS Quartz vs.
        osascript) override this to surface which one was chosen."""
        return ""


def build_pasteboard() -> Pasteboard:
    """Factory dispatching on `sys.platform`.

    Lazy-imports the concrete module so e.g. Windows ctypes are never loaded
    on macOS — otherwise the daemon would fail to import at all on Mac.
    """
    if sys.platform == "win32":
        from stt_platform_win import WindowsPasteboard
        return WindowsPasteboard()
    if sys.platform == "darwin":
        from stt_platform_mac import MacOSPasteboard
        return MacOSPasteboard()
    raise NotImplementedError(
        f"home-stt has no Pasteboard implementation for sys.platform={sys.platform!r}. "
        f"Implement Pasteboard in stt_platform_<os>.py and add a branch to build_pasteboard()."
    )
