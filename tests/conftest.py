"""Pytest config + shared fixtures for home-stt tests.

Loads scripts/stt-daemon.py (which has a hyphen and isn't importable via
plain `import stt-daemon`) as the `stt_daemon` module so individual tests
can reference `daemon._on_press` etc. directly.

Tests that need a clean module-global state should depend on the
`fresh_daemon` fixture, which resets _buffer / _recording / _processing /
_active_trigger / _recording_samples between tests.

Custom CLI options:
  --run-polish-bench    Run the polish-quality regression bench. Requires
                        the polish model loaded (~8 GB VRAM Win / ~4 GB
                        RSS Mac). Off by default to keep CI fast and
                        model-free. Equivalent env var:
                        HOME_STT_RUN_POLISH_BENCH=1.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"


# Make scripts/ importable so stt-daemon's `from stt_platform import ...`
# / `from text_polisher import ...` succeed.
sys.path.insert(0, str(_SCRIPTS_DIR))


def pytest_addoption(parser):
    parser.addoption(
        "--run-polish-bench", action="store_true", default=False,
        help="Run polish-quality regression bench (loads ~8 GB model)",
    )


def _load_daemon_module():
    """Load scripts/stt-daemon.py as the `stt_daemon` module.

    The hyphen in the filename makes plain `import stt-daemon` a
    SyntaxError, so we go through importlib's filespec API. Cached at
    module scope below so multiple fixtures share one instance.
    """
    spec = importlib.util.spec_from_file_location(
        "stt_daemon", str(_SCRIPTS_DIR / "stt-daemon.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Module-level: loaded once per pytest session.
daemon = _load_daemon_module()


@pytest.fixture
def fresh_daemon():
    """Reset daemon module-global state to a known-clean baseline.

    Pytest does NOT reload modules between tests by default — module
    globals from one test leak into the next. For the state-machine
    tests this would mean a previous test's leftover `_recording=True`
    or `_buffer=[...]` corrupts the next test. Fixture restores the
    daemon to its post-import baseline (all idle, empty buffer) and
    yields the module reference.
    """
    daemon._buffer = []
    daemon._recording = False
    daemon._active_trigger = None
    daemon._processing = False
    daemon._recording_samples = 0
    yield daemon
    # Post-test cleanup: same reset (defensive — a test that left state
    # half-set should not leak into the next test even if assertion failed).
    daemon._buffer = []
    daemon._recording = False
    daemon._active_trigger = None
    daemon._processing = False
    daemon._recording_samples = 0
