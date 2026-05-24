"""State-machine + silence-trim + config regression tests.

Verifies the behaviours v0.7.2 introduced or fixed. Runs in CI without
GPU or model loading — backend / polisher / pasteboard are mocked.

Covered:
  - Config regression: __version__, MIN_AUDIO_SEC, MAX_AUDIO_SEC,
    POLISH_LANGUAGES (catches accidental rollback in future commits).
  - _trim_silence: returns identity on tight clips, trims silence-padded
    speech, empties pure silence.
  - _on_press: state transitions, ignores non-trigger / re-press while
    holding.
  - _on_release: drain delay actually fires (≥50 ms), aborts cleanly
    when a new press arrives during the drain window.
  - _audio_callback: gates appends on _recording, enforces MAX_AUDIO_SEC.
  - _transcribe_and_emit: busy path explicitly clears buffer + logs
    (v0.7.2 C2 — previously silently merged into next utterance).
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from pynput.keyboard import Key


# ---------------------------------------------------------------------------
# Config / regression markers
# ---------------------------------------------------------------------------

def test_version_bumped(fresh_daemon):
    assert fresh_daemon.__version__ == "0.7.2"


def test_polish_languages_is_zh_only(fresh_daemon):
    """v0.7.2 C4: narrowed from {zh, ja, ko} to {zh}.

    POLISH_PROMPT is Chinese-only; ja/ko transcripts through it have
    zero rule-level constraint. Re-expand only when per-language prompts
    land. This test catches accidental rollback.
    """
    assert fresh_daemon.POLISH_LANGUAGES == {"zh"}


def test_min_audio_sec_015(fresh_daemon):
    """v0.7.2 O7: dropped 0.3 → 0.15 to accept 「好」「對」「是」 (~0.25s)."""
    assert fresh_daemon.MIN_AUDIO_SEC == 0.15


def test_max_audio_sec_120(fresh_daemon):
    """v0.7.2 C3: 120s ceiling against stuck-key buffer growth."""
    assert fresh_daemon.MAX_AUDIO_SEC == 120


# ---------------------------------------------------------------------------
# _trim_silence
# ---------------------------------------------------------------------------

def test_trim_silence_tight_clip_passes_through(fresh_daemon):
    """A clip with no leading/trailing silence should be near-identity."""
    sr = fresh_daemon.SAMPLE_RATE
    # 0.5s of noise at ~-15 dBFS (well above -50 dBFS threshold).
    speech = (np.random.randn(int(sr * 0.5)) * 0.2).astype(np.float32)
    trimmed = fresh_daemon._trim_silence(speech)
    # Should preserve roughly the full clip — within 100ms margin either side.
    assert len(trimmed) >= len(speech) * 0.9


def test_trim_silence_pads_removed(fresh_daemon):
    """Leading + trailing 1s silence should be trimmed away."""
    sr = fresh_daemon.SAMPLE_RATE
    silence = np.zeros(sr, dtype=np.float32)
    speech = (np.random.randn(sr) * 0.2).astype(np.float32)
    clip = np.concatenate([silence, speech, silence])
    trimmed = fresh_daemon._trim_silence(clip)
    # 1s speech + 2 × 100ms margins ≈ 1.2s; trimming should be tight.
    assert 0.8 < len(trimmed) / sr < 1.5


def test_trim_silence_all_silence_returns_empty(fresh_daemon):
    """Pure silence → empty array (caller treats as 'too short' and skips)."""
    sr = fresh_daemon.SAMPLE_RATE
    silence = np.zeros(sr * 2, dtype=np.float32)
    trimmed = fresh_daemon._trim_silence(silence)
    assert len(trimmed) == 0


def test_trim_silence_short_clip_no_op(fresh_daemon):
    """Clips below 100ms are too short to meaningfully trim — pass through."""
    sr = fresh_daemon.SAMPLE_RATE
    tiny = np.zeros(int(sr * 0.05), dtype=np.float32)  # 50ms
    trimmed = fresh_daemon._trim_silence(tiny)
    assert len(trimmed) == len(tiny)


# ---------------------------------------------------------------------------
# _on_press
# ---------------------------------------------------------------------------

def test_on_press_starts_recording(fresh_daemon):
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    # Pre-poison _recording_samples to confirm the press resets it
    fresh_daemon._recording_samples = 999
    fresh_daemon._buffer = [np.zeros(100, dtype=np.float32)]
    fresh_daemon._on_press(Key.alt_r)
    assert fresh_daemon._recording is True
    assert fresh_daemon._active_trigger == Key.alt_r
    assert fresh_daemon._recording_samples == 0
    assert fresh_daemon._buffer == []


def test_on_press_ignores_non_trigger(fresh_daemon):
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    fresh_daemon._on_press(Key.shift)
    assert fresh_daemon._recording is False
    assert fresh_daemon._active_trigger is None


def test_on_press_ignores_when_already_holding(fresh_daemon):
    """A second press while first is held should no-op (cf. OS key-repeat)."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r, Key.ctrl_r}
    fresh_daemon._on_press(Key.alt_r)
    fresh_daemon._on_press(Key.ctrl_r)
    assert fresh_daemon._active_trigger == Key.alt_r
    assert fresh_daemon._recording is True


# ---------------------------------------------------------------------------
# _on_release — drain delay (C1 regression)
# ---------------------------------------------------------------------------

def _install_inert_mocks(daemon_mod):
    """Replace backend / polisher / pasteboard with mocks so the
    transcribe thread spawned by _on_release returns immediately."""
    daemon_mod._backend = MagicMock()
    daemon_mod._backend.transcribe.return_value = ("", "zh")
    daemon_mod._pasteboard = MagicMock()
    daemon_mod._pasteboard.set_text.return_value = True
    daemon_mod._pasteboard.paste.return_value = True
    daemon_mod._polisher = MagicMock()
    daemon_mod._polisher.polish.return_value = ""


def test_on_release_has_drain_delay(fresh_daemon):
    """C1: _on_release must NOT flip _recording=False immediately —
    PortAudio's next 50ms callback needs the window to land the trailing
    block (otherwise the final phoneme of every utterance is clipped)."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_inert_mocks(fresh_daemon)
    fresh_daemon._on_press(Key.alt_r)
    start = time.monotonic()
    fresh_daemon._on_release(Key.alt_r)
    elapsed = time.monotonic() - start
    # Drain is 80ms in the daemon; assert ≥50ms here because Windows
    # time.sleep default resolution is ~15.6ms (multimedia timer tick)
    # so even sleep(0.08) can return at ~62-78ms on Win. The functional
    # requirement is ≥1 PortAudio callback period (50ms) — that's what
    # this test guards. Whether the actual drain is 50 or 80 ms is a
    # tuning decision; either is correct as long as ≥50.
    assert elapsed >= 0.05, f"expected ≥50ms drain, got {elapsed*1000:.0f}ms"
    assert fresh_daemon._recording is False


def test_on_release_aborts_if_new_press_during_drain(fresh_daemon, capfd):
    """C1 corner case: user re-presses during the drain window.
    _on_release should leave _recording=True (so the new press
    continues capturing) and skip its transcribe spawn — the new
    press will spawn its own when it eventually releases."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_inert_mocks(fresh_daemon)
    fresh_daemon._on_press(Key.alt_r)

    release_thread = threading.Thread(
        target=fresh_daemon._on_release, args=(Key.alt_r,),
    )
    release_thread.start()
    time.sleep(0.02)  # 20ms into the 60ms drain
    fresh_daemon._on_press(Key.alt_r)  # new press while still draining
    release_thread.join(timeout=1.0)

    assert fresh_daemon._recording is True, "new press should keep capturing"
    assert fresh_daemon._active_trigger == Key.alt_r
    out, err = capfd.readouterr()
    assert "aborted" in (out + err).lower()


def test_on_release_ignores_non_active_key(fresh_daemon):
    """Releasing a key that isn't the active trigger should no-op
    (e.g. user holds alt_r, taps ctrl_r — only the alt_r release matters)."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r, Key.ctrl_r}
    _install_inert_mocks(fresh_daemon)
    fresh_daemon._on_press(Key.alt_r)
    # Release ctrl_r (never pressed as active) — should no-op fast
    start = time.monotonic()
    fresh_daemon._on_release(Key.ctrl_r)
    elapsed = time.monotonic() - start
    assert elapsed < 0.01, "non-active release should return immediately"
    assert fresh_daemon._recording is True  # still capturing alt_r
    assert fresh_daemon._active_trigger == Key.alt_r


# ---------------------------------------------------------------------------
# _audio_callback
# ---------------------------------------------------------------------------

def test_audio_callback_appends_only_when_recording(fresh_daemon):
    """Callback should append to _buffer only while _recording==True."""
    chunk = np.zeros((100, 1), dtype=np.float32)

    # Not recording → no append
    fresh_daemon._recording = False
    fresh_daemon._audio_callback(chunk, 100, None, None)
    assert fresh_daemon._buffer == []

    # Recording → append
    fresh_daemon._recording = True
    fresh_daemon._audio_callback(chunk, 100, None, None)
    assert len(fresh_daemon._buffer) == 1
    assert fresh_daemon._recording_samples == 100


def test_audio_callback_auto_stops_at_max_audio_sec(fresh_daemon, capfd):
    """C3: a chunk that pushes _recording_samples past
    MAX_AUDIO_SEC * SAMPLE_RATE should force _recording=False."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    fresh_daemon._on_press(Key.alt_r)
    # Set _processing=True so the spawned transcribe thread early-returns
    # (we don't care about the transcribe; just the auto-stop behaviour).
    fresh_daemon._processing = True
    sr = fresh_daemon.SAMPLE_RATE
    # One big chunk = full MAX_AUDIO_SEC worth of samples. Real PortAudio
    # callbacks are 50ms but the threshold check is purely sample-count.
    big = np.zeros((sr * fresh_daemon.MAX_AUDIO_SEC, 1), dtype=np.float32)
    fresh_daemon._audio_callback(big, big.shape[0], None, None)
    assert fresh_daemon._recording is False
    out, err = capfd.readouterr()
    assert "auto-stop" in (out + err).lower()


# ---------------------------------------------------------------------------
# _transcribe_and_emit — busy path (C2 regression)
# ---------------------------------------------------------------------------

def test_transcribe_busy_clears_buffer_and_logs(fresh_daemon, capfd):
    """C2: pre-v0.7.2 the busy early-return silently left captured audio
    in _buffer where the NEXT transcribe call would merge it with the
    next utterance. Now: explicitly drop, log, and clear."""
    fresh_daemon._processing = True  # simulate previous transcribe in flight
    sr = fresh_daemon.SAMPLE_RATE
    # Pre-load buffer with 100 × 50ms chunks = 5s
    chunk_size = int(sr * 0.05)
    fresh_daemon._buffer = [
        np.zeros((chunk_size, 1), dtype=np.float32) for _ in range(100)
    ]
    fresh_daemon._recording_samples = chunk_size * 100

    fresh_daemon._transcribe_and_emit()

    assert fresh_daemon._buffer == [], "busy path must clear buffer"
    assert fresh_daemon._recording_samples == 0
    out, err = capfd.readouterr()
    combined = (out + err).lower()
    assert "busy" in combined
    assert "drop" in combined  # "dropped 5.00s..."


def test_transcribe_busy_with_empty_buffer_is_silent(fresh_daemon, capfd):
    """The busy log should only fire if there was actually audio to drop —
    a no-op busy bail (e.g. release with empty buffer) shouldn't spam."""
    fresh_daemon._processing = True
    fresh_daemon._buffer = []
    fresh_daemon._recording_samples = 0

    fresh_daemon._transcribe_and_emit()

    out, err = capfd.readouterr()
    # Should NOT log "busy ... dropped" when there was nothing to drop.
    assert "busy" not in (out + err).lower()
