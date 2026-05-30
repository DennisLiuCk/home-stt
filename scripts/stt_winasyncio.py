"""Shared Windows asyncio noise suppression.

On Windows, anything serving over asyncio's ProactorEventLoop (Gradio/uvicorn
today; any future local server) logs a spurious traceback when a client drops
a connection abruptly: the loop tries to ``shutdown()`` an already-closed
socket and raises ``ConnectionResetError [WinError 10054]`` from
``_ProactorBasePipeTransport._call_connection_lost``. The request has already
completed — the error is harmless but floods the console (the web UI's
gr.Timer polling triggers it every couple of seconds). See CPython issue
#83413: https://github.com/python/cpython/issues/83413

This module is stdlib-only on purpose: it pulls in no gradio/torch, so any
component — including the lightweight daemon side — can import and apply the
guard without dragging in heavy dependencies.
"""
from __future__ import annotations

import sys


def silence_proactor_connection_reset() -> None:
    """Wrap the Proactor transport so it ignores benign connection-reset noise.

    No-op off Windows (only the ProactorEventLoop is affected). Idempotent —
    re-applying does not stack wrappers. The wrapper swallows only
    ConnectionResetError / ConnectionAbortedError; every other exception still
    propagates to asyncio's normal handling.
    """
    if sys.platform != "win32":
        return
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        from functools import wraps
    except ImportError:
        return

    func = _ProactorBasePipeTransport._call_connection_lost
    if getattr(func, "_stt_patched", False):
        return

    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except (ConnectionResetError, ConnectionAbortedError):
            pass

    _wrapper._stt_patched = True
    _ProactorBasePipeTransport._call_connection_lost = _wrapper
