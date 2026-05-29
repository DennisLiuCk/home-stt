"""Tests for the home_stt CLI helpers.

Focus: the v0.8.0 daemon-identity hardening in _process_alive. Existence of
a PID is no longer sufficient — the process command line must reference
stt-daemon.py — but a failed command-line read must fall back to
assume-alive so a query hiccup never reports a genuinely-running daemon as
'stopped'. conftest.py puts scripts/ on sys.path, so `import home_stt` works.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import home_stt


def _fake_existence(monkeypatch, alive: bool, pid: int):
    """Make the EXISTENCE half of _process_alive report `alive` on either
    platform, without touching the command-line identity query."""
    if sys.platform == "win32":
        out = str(pid) if alive else "INFO: No tasks are running ..."
        monkeypatch.setattr(
            home_stt.subprocess, "run",
            lambda *a, **k: MagicMock(stdout=out, returncode=0),
        )
    else:
        def kill(p, sig):
            if not alive:
                raise ProcessLookupError()
        monkeypatch.setattr(home_stt.os, "kill", kill)


def test_process_alive_true_when_cmdline_is_daemon(monkeypatch):
    _fake_existence(monkeypatch, alive=True, pid=12345)
    monkeypatch.setattr(home_stt, "_process_cmdline",
                        lambda pid: "python -u C:/x/scripts/stt-daemon.py")
    assert home_stt._process_alive(12345) is True


def test_process_alive_false_when_cmdline_is_unrelated(monkeypatch):
    """PID exists but belongs to a recycled, unrelated process → 'stopped'."""
    _fake_existence(monkeypatch, alive=True, pid=12345)
    monkeypatch.setattr(home_stt, "_process_cmdline",
                        lambda pid: "python -u some_other_app.py")
    assert home_stt._process_alive(12345) is False


def test_process_alive_falls_back_to_existence_when_cmdline_unknown(monkeypatch):
    """If the command line can't be read (None/empty — CIM unavailable, venv
    launcher, timeout), assume alive rather than risk a false 'stopped'."""
    _fake_existence(monkeypatch, alive=True, pid=12345)
    monkeypatch.setattr(home_stt, "_process_cmdline", lambda pid: None)
    assert home_stt._process_alive(12345) is True


def test_process_alive_false_when_process_missing(monkeypatch):
    _fake_existence(monkeypatch, alive=False, pid=99999)
    # cmdline should not even be consulted, but stub it so a regression that
    # reorders the checks can't accidentally hit the real subprocess.
    monkeypatch.setattr(home_stt, "_process_cmdline", lambda pid: None)
    assert home_stt._process_alive(99999) is False
