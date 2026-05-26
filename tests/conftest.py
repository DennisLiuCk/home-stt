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
import logging
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"


# Make scripts/ importable so stt-daemon's `from stt_platform import ...`
# / `from text_polisher import ...` succeed.
sys.path.insert(0, str(_SCRIPTS_DIR))

# Configure the "stt" logger so pytest's caplog fixture captures messages.
# No StreamHandler needed — pytest's log capture plugin handles it.
_stt_logger = logging.getLogger("stt")
_stt_logger.setLevel(logging.DEBUG)
_stt_logger.propagate = True


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

# Capture original module-default for ENCODER_PIPELINING so the
# fresh_daemon fixture can restore it between tests. v0.7.3 ships with
# False (bench-first save — see scripts/stt-daemon.py config comment).
# Tests that exercise the streaming framework set it to True locally;
# the auto-restore here keeps that local mutation from leaking into
# the next test. Capturing at import time (not hard-coding False)
# means a future module-default flip doesn't require touching this file.
_ORIGINAL_ENCODER_PIPELINING = daemon.ENCODER_PIPELINING

# v0.7.5: same pattern for EDIT_TRIGGER_KEYS — voice-edit state-machine
# tests set this to {Key.f13} locally; restore between tests.
_ORIGINAL_EDIT_TRIGGER_KEYS = daemon.EDIT_TRIGGER_KEYS

# Mute beeps globally for the test session. Several tests exercise
# _on_press which calls _play_beep, which calls sd.query_devices() +
# sd.play(). On headless CI runners (macOS GitHub Actions in particular)
# there's no audio device — sounddevice / PortAudio segfault with
# Abort trap (SIGABRT, exit 134) before Python's try/except in
# _play_beep can catch it (native crash, not a Python exception).
# Tests don't care about audio feedback; just disable for safety.
daemon.BEEPS_ENABLED = False


def _reset_daemon_state():
    """Single source of truth for the clean-state baseline used by
    fresh_daemon (pre and post test). Extracted so additions to the
    v0.8.0 encoder state etc. don't drift between pre/post halves."""
    import queue as _queue
    # v0.8.0: kill any encoder worker thread that may have been spawned
    # by the previous test BEFORE resetting state, so its goroutine-like
    # loop sees the stop signal and exits cleanly. Without this, a
    # background worker can still write to module globals after we've
    # "reset" them, corrupting the next test.
    daemon._encoder_stop_event.set()
    prior_thread = daemon._encoder_thread
    if prior_thread is not None and prior_thread.is_alive():
        prior_thread.join(timeout=2.0)
    daemon._encoder_stop_event.clear()

    daemon._buffer = []
    daemon._recording = False
    daemon._active_trigger = None
    daemon._processing = False
    daemon._recording_samples = 0
    # v0.7.5: voice-edit per-recording state. Reset between tests so a
    # test that sets _edit_mode=True (via mocked _on_press flow) doesn't
    # leak into the next test's release dispatch.
    daemon._edit_mode = False
    daemon._edit_selection = None
    daemon._edit_original_clipboard = None
    # v0.7.3: restore module-default ENCODER_PIPELINING after any test
    # mutation (streaming tests flip it True for their body).
    daemon.ENCODER_PIPELINING = _ORIGINAL_ENCODER_PIPELINING
    # v0.7.5: restore EDIT_TRIGGER_KEYS module-default (voice-edit tests
    # set it to a sentinel like {Key.f13}).
    daemon.EDIT_TRIGGER_KEYS = _ORIGINAL_EDIT_TRIGGER_KEYS
    # v0.8.0 encoder-pipelining state. Mirrors what _on_press does on a
    # real recording start, so tests that exercise the audio callback
    # without going through _on_press first don't hit stale flags.
    daemon._encoder_thread = None
    daemon._encoder_handle = None
    daemon._encoder_active = False
    daemon._encoder_failed = False
    daemon._encoder_consecutive_failures = 0
    daemon._encoder_use_batch_fallback = False
    daemon._encoder_silence_run_samples = 0
    daemon._encoder_residual_samples = None
    # Drain the queue (fresh Queue() would also work but reusing the
    # existing one avoids re-binding the module attribute that the
    # daemon's own functions captured at import time).
    while True:
        try:
            daemon._encoder_queue.get_nowait()
        except _queue.Empty:
            break
    # _backend default stays None (was lazily set by main() in v0.7.2;
    # v0.8.0 made it nullable so the audio callback can early-out when
    # no backend is wired). Tests that need a backend set their own
    # mock via _install_inert_mocks or equivalent.


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
    _reset_daemon_state()
    yield daemon
    # Post-test cleanup: same reset (defensive — a test that left state
    # half-set should not leak into the next test even if assertion failed).
    _reset_daemon_state()
