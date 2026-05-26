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

import logging
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from pynput.keyboard import Key

import stt_streaming


# ---------------------------------------------------------------------------
# Config / regression markers
# ---------------------------------------------------------------------------

def test_version_bumped(fresh_daemon):
    assert fresh_daemon.__version__ == "0.7.5"


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
# v0.8.0 config + STTBackend streaming ABC contract
# ---------------------------------------------------------------------------

def test_encoder_pipelining_config_defaults(fresh_daemon):
    """v0.7.3 (post-v0.8.0 bench-first save): ENCODER_PIPELINING ships
    DISABLED. Day 13-14 bench (tmp/bench_v080_latency.py) showed encoder
    pipelining saves only ~3% on RTX 5080 + Qwen3-ASR-0.6B — decoder
    dominates ~95% of post-release time, encoder ~0.2s (plan estimated
    3-5s, off by 15-25x). Framework preserved for future revisit when
    decoder side gets faster. Flip to True only with new bench evidence.

    Tunables stay at their plan-spec values so re-enable is trivial."""
    assert fresh_daemon.ENCODER_PIPELINING is False
    assert stt_streaming.ENCODER_CHUNK_SEC == 5.0
    assert stt_streaming.ENCODER_QUEUE_MAX == 200
    assert stt_streaming.ENCODER_FINALIZE_TIMEOUT == 8.0
    assert stt_streaming.ENCODER_FAILURE_BUDGET == 3
    # Option C silence-detect fallback threshold (still relevant when
    # streaming is re-enabled).
    assert stt_streaming.ENCODER_SILENCE_FALLBACK_SEC == 2.0


def test_sttbackend_default_supports_streaming_false(fresh_daemon):
    """Default opt-out: existing batch-only backends (FasterWhisper,
    MlxWhisper) need ZERO changes for v0.8.0 — they inherit
    supports_streaming()=False and the daemon routes them through the
    v0.7.2 batch path."""
    class _Stub(fresh_daemon.STTBackend):
        name = "stub"
        def transcribe(self, samples):
            return ("", "")
    s = _Stub()
    assert s.supports_streaming() is False


def test_sttbackend_default_streaming_methods_raise(fresh_daemon):
    """Backends that DO opt in (supports_streaming()=True) but forget
    to override one of the four streaming methods should fail loudly,
    not silently produce empty output. NotImplementedError carries
    the backend name for fast diagnosis."""
    import numpy as np
    class _Stub(fresh_daemon.STTBackend):
        name = "stub"
        def transcribe(self, samples):
            return ("", "")
    s = _Stub()
    with pytest.raises(NotImplementedError, match="stub"):
        s.start_encoder()
    with pytest.raises(NotImplementedError, match="stub"):
        s.push_chunk(object(), np.zeros(16000, dtype=np.float32))
    with pytest.raises(NotImplementedError, match="stub"):
        s.finalize(object(), np.zeros(0, dtype=np.float32))
    with pytest.raises(NotImplementedError, match="stub"):
        s.abort(object())


def test_existing_backends_inherit_no_streaming(fresh_daemon):
    """v0.8.0 must NOT accidentally flip FasterWhisper / MlxWhisper into
    streaming mode — they don't have the encoder API. The Qwen3
    backends will override to True (separate test once impl lands)."""
    # FasterWhisperBackend / MlxWhisperBackend don't load models on
    # supports_streaming() — it's just a method check on the class.
    from stt_backends import FasterWhisperBackend, MlxWhisperBackend
    assert FasterWhisperBackend.supports_streaming(
        FasterWhisperBackend.__new__(FasterWhisperBackend)
    ) is False
    assert MlxWhisperBackend.supports_streaming(
        MlxWhisperBackend.__new__(MlxWhisperBackend)
    ) is False


# ---------------------------------------------------------------------------
# _trim_silence
# ---------------------------------------------------------------------------

def test_trim_silence_tight_clip_passes_through(fresh_daemon):
    """A clip with no leading/trailing silence should be near-identity."""
    sr = fresh_daemon.SAMPLE_RATE
    # 0.5s of noise at ~-15 dBFS (well above -50 dBFS threshold).
    speech = (np.random.randn(int(sr * 0.5)) * 0.2).astype(np.float32)
    trimmed = fresh_daemon._trim_silence(speech, sr)
    # Should preserve roughly the full clip — within 100ms margin either side.
    assert len(trimmed) >= len(speech) * 0.9


def test_trim_silence_pads_removed(fresh_daemon):
    """Leading + trailing 1s silence should be trimmed away."""
    sr = fresh_daemon.SAMPLE_RATE
    silence = np.zeros(sr, dtype=np.float32)
    speech = (np.random.randn(sr) * 0.2).astype(np.float32)
    clip = np.concatenate([silence, speech, silence])
    trimmed = fresh_daemon._trim_silence(clip, sr)
    # 1s speech + 2 × 100ms margins ≈ 1.2s; trimming should be tight.
    assert 0.8 < len(trimmed) / sr < 1.5


def test_trim_silence_all_silence_returns_empty(fresh_daemon):
    """Pure silence → empty array (caller treats as 'too short' and skips)."""
    sr = fresh_daemon.SAMPLE_RATE
    silence = np.zeros(sr * 2, dtype=np.float32)
    trimmed = fresh_daemon._trim_silence(silence, sr)
    assert len(trimmed) == 0


def test_trim_silence_short_clip_no_op(fresh_daemon):
    """Clips below 100ms are too short to meaningfully trim — pass through."""
    sr = fresh_daemon.SAMPLE_RATE
    tiny = np.zeros(int(sr * 0.05), dtype=np.float32)  # 50ms
    trimmed = fresh_daemon._trim_silence(tiny, sr)
    assert len(trimmed) == len(tiny)


# ---------------------------------------------------------------------------
# _on_press
# ---------------------------------------------------------------------------

def test_on_press_starts_recording(fresh_daemon):
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    # Pre-poison _recording_samples to confirm the press resets it
    fresh_daemon._st.recording_samples = 999
    fresh_daemon._st.buffer = [np.zeros(100, dtype=np.float32)]
    fresh_daemon._on_press(Key.alt_r)
    assert fresh_daemon._st.recording is True
    assert fresh_daemon._st.active_trigger == Key.alt_r
    assert fresh_daemon._st.recording_samples == 0
    assert fresh_daemon._st.buffer == []


def test_on_press_ignores_non_trigger(fresh_daemon):
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    fresh_daemon._on_press(Key.shift)
    assert fresh_daemon._st.recording is False
    assert fresh_daemon._st.active_trigger is None


def test_on_press_ignores_when_already_holding(fresh_daemon):
    """A second press while first is held should no-op (cf. OS key-repeat)."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r, Key.ctrl_r}
    fresh_daemon._on_press(Key.alt_r)
    fresh_daemon._on_press(Key.ctrl_r)
    assert fresh_daemon._st.active_trigger == Key.alt_r
    assert fresh_daemon._st.recording is True


# ---------------------------------------------------------------------------
# _on_release — drain delay (C1 regression)
# ---------------------------------------------------------------------------

def _install_inert_mocks(daemon_mod, streaming: bool = False,
                         finalize_text: str = "stream output",
                         finalize_lang: str = "zh"):
    """Replace backend / polisher / pasteboard with mocks so the
    transcribe thread spawned by _on_release returns immediately.

    v0.8.0 (`streaming` kwarg): when True, the backend mock reports
    supports_streaming()=True and the four streaming methods
    (start_encoder/push_chunk/finalize/abort) are stub-able via the
    returned MagicMock. Default False keeps v0.7.2-style test behaviour
    — _audio_callback won't lazy-spawn an encoder thread.

    v0.7.3 (bench-first save): module-default ENCODER_PIPELINING is now
    False. Installing mocks unconditionally flips it to True so the
    streaming framework is exercisable; the conftest fresh_daemon
    fixture restores it to the module default at test teardown.
    """
    daemon_mod._backend = MagicMock()
    daemon_mod._backend.transcribe.return_value = ("", "zh")
    daemon_mod._backend.supports_streaming.return_value = bool(streaming)
    daemon_mod._backend.start_encoder.return_value = MagicMock(name="EncoderHandle")
    daemon_mod._backend.finalize.return_value = (finalize_text, finalize_lang)
    daemon_mod._pasteboard = MagicMock()
    daemon_mod._pasteboard.set_text.return_value = True
    daemon_mod._pasteboard.paste.return_value = True
    daemon_mod._polisher = MagicMock()
    daemon_mod._polisher.polish.return_value = ""
    # Enable the streaming framework irrespective of the module's shipped
    # default — tests that DON'T want it can override after this call.
    daemon_mod.ENCODER_PIPELINING = True
    stt_streaming.ENCODER_PIPELINING = True


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
    assert fresh_daemon._st.recording is False


def test_on_release_aborts_if_new_press_during_drain(fresh_daemon, caplog):
    """C1 corner case: user re-presses during the drain window.
    _on_release should leave _recording=True (so the new press
    continues capturing) and skip its transcribe spawn — the new
    press will spawn its own when it eventually releases."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_inert_mocks(fresh_daemon)
    fresh_daemon._on_press(Key.alt_r)

    with caplog.at_level(logging.DEBUG, logger="stt"):
        release_thread = threading.Thread(
            target=fresh_daemon._on_release, args=(Key.alt_r,),
        )
        release_thread.start()
        time.sleep(0.02)  # 20ms into the 60ms drain
        fresh_daemon._on_press(Key.alt_r)  # new press while still draining
        release_thread.join(timeout=1.0)

    assert fresh_daemon._st.recording is True, "new press should keep capturing"
    assert fresh_daemon._st.active_trigger == Key.alt_r
    assert "aborted" in caplog.text.lower()


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
    assert fresh_daemon._st.recording is True  # still capturing alt_r
    assert fresh_daemon._st.active_trigger == Key.alt_r


# ---------------------------------------------------------------------------
# _audio_callback
# ---------------------------------------------------------------------------

def test_audio_callback_appends_only_when_recording(fresh_daemon):
    """Callback should append to _buffer only while _recording==True."""
    chunk = np.zeros((100, 1), dtype=np.float32)

    # Not recording → no append
    fresh_daemon._st.recording = False
    fresh_daemon._audio_callback(chunk, 100, None, None)
    assert fresh_daemon._st.buffer == []

    # Recording → append
    fresh_daemon._st.recording = True
    fresh_daemon._audio_callback(chunk, 100, None, None)
    assert len(fresh_daemon._st.buffer) == 1
    assert fresh_daemon._st.recording_samples == 100


def test_audio_callback_auto_stops_at_max_audio_sec(fresh_daemon, caplog):
    """C3: a chunk that pushes _recording_samples past
    MAX_AUDIO_SEC * SAMPLE_RATE should force _recording=False."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    fresh_daemon._on_press(Key.alt_r)
    # Set _processing=True so the spawned transcribe thread early-returns
    # (we don't care about the transcribe; just the auto-stop behaviour).
    fresh_daemon._st.processing = True
    sr = fresh_daemon.SAMPLE_RATE
    # One big chunk = full MAX_AUDIO_SEC worth of samples. Real PortAudio
    # callbacks are 50ms but the threshold check is purely sample-count.
    big = np.zeros((sr * fresh_daemon.MAX_AUDIO_SEC, 1), dtype=np.float32)
    with caplog.at_level(logging.DEBUG, logger="stt"):
        fresh_daemon._audio_callback(big, big.shape[0], None, None)
    assert fresh_daemon._st.recording is False
    assert "auto-stop" in caplog.text.lower()


# ---------------------------------------------------------------------------
# _transcribe_and_emit — busy path (C2 regression)
# ---------------------------------------------------------------------------

def test_transcribe_busy_clears_buffer_and_logs(fresh_daemon, caplog):
    """C2: pre-v0.7.2 the busy early-return silently left captured audio
    in _buffer where the NEXT transcribe call would merge it with the
    next utterance. Now: explicitly drop, log, and clear."""
    fresh_daemon._st.processing = True  # simulate previous transcribe in flight
    sr = fresh_daemon.SAMPLE_RATE
    # Pre-load buffer with 100 × 50ms chunks = 5s
    chunk_size = int(sr * 0.05)
    fresh_daemon._st.buffer = [
        np.zeros((chunk_size, 1), dtype=np.float32) for _ in range(100)
    ]
    fresh_daemon._st.recording_samples = chunk_size * 100

    with caplog.at_level(logging.DEBUG, logger="stt"):
        fresh_daemon._transcribe_and_emit()

    assert fresh_daemon._st.buffer == [], "busy path must clear buffer"
    assert fresh_daemon._st.recording_samples == 0
    combined = caplog.text.lower()
    assert "busy" in combined
    assert "drop" in combined  # "dropped 5.00s..."


def test_transcribe_busy_with_empty_buffer_is_silent(fresh_daemon, caplog):
    """The busy log should only fire if there was actually audio to drop —
    a no-op busy bail (e.g. release with empty buffer) shouldn't spam."""
    fresh_daemon._st.processing = True
    fresh_daemon._st.buffer = []
    fresh_daemon._st.recording_samples = 0

    with caplog.at_level(logging.DEBUG, logger="stt"):
        fresh_daemon._transcribe_and_emit()

    # Should NOT log "busy ... dropped" when there was nothing to drop.
    assert "busy" not in caplog.text.lower()


# ===========================================================================
# v0.8.0 press-time encoder pipelining
# ===========================================================================
#
# Test plan from `plan-v0.8.0` and Option C silence-detection extension.
# All tests use MagicMock backends (no GPU, no model load). Thread-spawning
# tests rely on the conftest fresh_daemon fixture to auto-join the worker
# on teardown so they don't leak between tests.

def _voiced_chunk(n: int) -> np.ndarray:
    """50 ms-ish voiced-looking chunk: above the -50 dBFS silence
    threshold so the silence-fallback detector doesn't fire."""
    return (np.ones((n, 1), dtype=np.float32) * 0.1)


def _silent_chunk(n: int) -> np.ndarray:
    return np.zeros((n, 1), dtype=np.float32)


def test_encoder_lazy_spawn_on_first_chunk(fresh_daemon):
    """Press alone doesn't spawn (it just resets state). The encoder
    thread is spawned by the FIRST audio_callback that lands while
    _recording. Short taps shorter than the first 50 ms callback would
    never even reach spawn — important for keeping cost off the
    keyboard-listener thread."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_inert_mocks(fresh_daemon, streaming=True)
    fresh_daemon._on_press(Key.alt_r)
    # Right after press, no encoder spawned yet
    assert fresh_daemon._encoder._thread is None
    assert fresh_daemon._encoder.active is False

    # First callback should trigger lazy spawn
    fresh_daemon._audio_callback(_voiced_chunk(800), 800, None, None)
    assert fresh_daemon._encoder.active is True
    assert fresh_daemon._encoder._thread is not None
    assert fresh_daemon._backend.start_encoder.call_count == 1

    # Second callback should NOT re-spawn
    fresh_daemon._audio_callback(_voiced_chunk(800), 800, None, None)
    assert fresh_daemon._backend.start_encoder.call_count == 1


def test_encoder_no_spawn_when_supports_streaming_false(fresh_daemon):
    """Backends that don't opt in (FasterWhisper, MlxWhisper) must
    NOT have an encoder thread spawned for them — that would crash
    with NotImplementedError on push_chunk."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_inert_mocks(fresh_daemon, streaming=False)
    fresh_daemon._on_press(Key.alt_r)
    fresh_daemon._audio_callback(_voiced_chunk(800), 800, None, None)
    assert fresh_daemon._encoder.active is False
    assert fresh_daemon._encoder._thread is None
    assert fresh_daemon._backend.start_encoder.call_count == 0


def test_encoder_dual_write_to_buffer_and_queue(fresh_daemon):
    """Each audio chunk MUST land in BOTH _buffer (for batch fallback)
    AND _encoder_queue (for the worker). Single-write would break
    fallback semantics — a streaming failure mid-recording would lose
    the audio that had already been pushed to queue but not yet flushed
    to buffer."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_inert_mocks(fresh_daemon, streaming=True)
    fresh_daemon._on_press(Key.alt_r)
    fresh_daemon._audio_callback(_voiced_chunk(800), 800, None, None)
    fresh_daemon._audio_callback(_voiced_chunk(800), 800, None, None)
    # buffer: at least 2 chunks
    assert len(fresh_daemon._st.buffer) == 2
    # queue: at least 2 chunks (worker may have already pulled some;
    # this is racy. assert queue+worker_pulled >= 2 indirectly via
    # backend.push_chunk total calls. Since chunks are < ENCODER_CHUNK_SEC,
    # worker shouldn't have pushed yet — both should still be in queue.
    # But threads race. Just verify the queue API received them: count by
    # popping until empty.)
    fresh_daemon._encoder._stop_event.set()
    fresh_daemon._encoder._thread.join(timeout=2.0)
    # After worker exits, the residual catches anything left in queue
    # plus accumulator. Residual should be ≈ 1600 samples (2 × 800).
    assert fresh_daemon._encoder.residual_samples is not None
    assert fresh_daemon._encoder.residual_samples.shape[0] == 1600


def test_release_signals_stop_event_before_transcribe_spawn(fresh_daemon):
    """_on_release MUST set _encoder_stop_event BEFORE spawning the
    transcribe thread, so the worker sees the signal promptly (its loop
    polls every 100 ms) and the transcribe thread doesn't block on a
    worker that's still happily encoding new chunks."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_inert_mocks(fresh_daemon, streaming=True)
    fresh_daemon._on_press(Key.alt_r)
    fresh_daemon._audio_callback(_voiced_chunk(800), 800, None, None)
    assert fresh_daemon._encoder.active is True
    assert not fresh_daemon._encoder._stop_event.is_set()
    fresh_daemon._on_release(Key.alt_r)
    # After release returns, stop_event MUST be set
    assert fresh_daemon._encoder._stop_event.is_set()


def test_encoder_crash_sets_failure_flag(fresh_daemon, caplog):
    """Worker's push_chunk raising → worker catches, sets
    _encoder_failed=True, increments _encoder_consecutive_failures.
    _transcribe_and_emit reads the flag and falls back to batch."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_inert_mocks(fresh_daemon, streaming=True)
    fresh_daemon._backend.push_chunk.side_effect = RuntimeError("boom")
    with caplog.at_level(logging.DEBUG, logger="stt"):
        fresh_daemon._on_press(Key.alt_r)
        sr = fresh_daemon.SAMPLE_RATE
        big = _voiced_chunk(int(sr * 5.1))
        fresh_daemon._audio_callback(big, big.shape[0], None, None)
        fresh_daemon._encoder._stop_event.set()
        fresh_daemon._encoder._thread.join(timeout=2.0)
    assert fresh_daemon._encoder.failed is True
    assert fresh_daemon._encoder.consecutive_failures >= 1
    assert "encoder worker crashed" in caplog.text.lower()


def test_finalize_timeout_falls_back_to_batch(fresh_daemon, caplog, monkeypatch):
    """If the encoder worker is stuck (e.g. mid-forward on a slow GPU)
    when _transcribe_and_emit tries to join it, the join must time out
    within ENCODER_FINALIZE_TIMEOUT seconds and the batch path takes
    over. We shorten the timeout for test speed."""
    monkeypatch.setattr(stt_streaming, "ENCODER_FINALIZE_TIMEOUT", 0.2)
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_inert_mocks(fresh_daemon, streaming=True)

    release_hang = threading.Event()
    def hang_until_event(handle, samples):
        release_hang.wait(timeout=30)
    fresh_daemon._backend.push_chunk.side_effect = hang_until_event

    try:
        with caplog.at_level(logging.DEBUG, logger="stt"):
            fresh_daemon._on_press(Key.alt_r)
            sr = fresh_daemon.SAMPLE_RATE
            big = _voiced_chunk(int(sr * 5.1))
            fresh_daemon._audio_callback(big, big.shape[0], None, None)
            time.sleep(0.15)
            fresh_daemon._on_release(Key.alt_r)
            time.sleep(3.0)
        combined = caplog.text.lower()
        assert ("encoder join timed out" in combined
                or "encoder finalize" in combined)
        # Batch transcribe must have been called as fallback
        assert fresh_daemon._backend.transcribe.call_count >= 1
    finally:
        # Always release the hung worker so it can exit cleanly and not
        # leak into subsequent tests' leaked-worker check.
        release_hang.set()
        if fresh_daemon._encoder._thread is not None:
            fresh_daemon._encoder._thread.join(timeout=2.0)


def test_consecutive_failure_suppresses_spawn(fresh_daemon):
    """After ENCODER_FAILURE_BUDGET (=3) consecutive crashes, the next
    recording's _audio_callback skips the encoder spawn entirely — daemon
    goes straight to batch path for the next utterance."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_inert_mocks(fresh_daemon, streaming=True)
    # Pre-poison counter to the budget
    fresh_daemon._encoder.consecutive_failures = stt_streaming.ENCODER_FAILURE_BUDGET
    fresh_daemon._on_press(Key.alt_r)
    fresh_daemon._audio_callback(_voiced_chunk(800), 800, None, None)
    assert fresh_daemon._encoder.active is False
    assert fresh_daemon._backend.start_encoder.call_count == 0


def test_short_tap_encoder_finalize_with_tail(fresh_daemon):
    """Hold for ~200 ms (< ENCODER_CHUNK_SEC): worker spawns + accumulates
    but never push_chunks (not enough audio). Finalize gets all 200 ms as
    `tail_samples`. The mock finalize returns canned text; we just verify
    the path completes without crash + finalize WAS called with non-zero
    tail."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_inert_mocks(fresh_daemon, streaming=True)
    fresh_daemon._on_press(Key.alt_r)
    # 4 × 50 ms chunks = 200 ms total
    for _ in range(4):
        fresh_daemon._audio_callback(_voiced_chunk(800), 800, None, None)
    # Skip the 80 ms drain by setting stop_event directly + manually
    # invoking _transcribe_and_emit. (Calling _on_release would also work
    # but pads test runtime with the drain sleep.)
    fresh_daemon._encoder._stop_event.set()
    fresh_daemon._encoder._thread.join(timeout=2.0)
    with fresh_daemon._st.lock:
        fresh_daemon._st.recording = False
    fresh_daemon._transcribe_and_emit()
    # push_chunk should NOT have been called (200 ms < 5 s chunk size).
    # finalize SHOULD have been called with the 200 ms tail (~3200 samples).
    assert fresh_daemon._backend.push_chunk.call_count == 0
    assert fresh_daemon._backend.finalize.call_count == 1
    call_args = fresh_daemon._backend.finalize.call_args
    tail = call_args[0][1]  # second positional arg
    assert isinstance(tail, np.ndarray)
    assert tail.shape[0] == 3200  # 4 × 800


def test_processing_flag_independent_of_encoder_active(fresh_daemon):
    """_encoder_active=True from a still-running encoder must NOT block
    a new press from starting a new recording. Only _processing
    (decoder running) gates re-entry per v0.7.2 semantics. v0.8.0 docs
    explicitly preserve this — encoder running while user presses
    again happens during back-to-back utterances."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_inert_mocks(fresh_daemon, streaming=True)
    # Simulate state: encoder active from a previous utterance
    fresh_daemon._encoder.active = True
    fresh_daemon._encoder._thread = MagicMock()  # pretend worker alive
    # New press should still succeed (sets _active_trigger, _recording=True)
    fresh_daemon._on_press(Key.alt_r)
    assert fresh_daemon._st.recording is True
    assert fresh_daemon._st.active_trigger == Key.alt_r
    # _on_press DOES reset _encoder_active to False (new recording starts
    # its own encoder). The old worker thread reference is overwritten
    # too — but we'd never get into this race in real usage because
    # _processing gates the previous recording's decoder.
    assert fresh_daemon._encoder.active is False
    assert fresh_daemon._encoder._thread is None


def test_silence_detection_triggers_batch_fallback(fresh_daemon):
    """Option C: mid-utterance silence ≥ ENCODER_SILENCE_FALLBACK_SEC
    sets _encoder_use_batch_fallback=True. The encoder lazy-spawn check
    has this flag in its conditional, so once set, no encoder spawns
    even on subsequent voiced audio. _transcribe_and_emit reads the
    flag and routes to batch."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_inert_mocks(fresh_daemon, streaming=True)
    fresh_daemon._on_press(Key.alt_r)
    sr = fresh_daemon.SAMPLE_RATE
    # Feed silence until the threshold is crossed. Use one big silent
    # chunk = ENCODER_SILENCE_FALLBACK_SEC + safety margin.
    silent = _silent_chunk(int(sr * (stt_streaming.ENCODER_SILENCE_FALLBACK_SEC + 0.5)))
    fresh_daemon._audio_callback(silent, silent.shape[0], None, None)
    assert fresh_daemon._encoder.use_batch_fallback is True
    # Now feed voiced audio — should NOT spawn encoder (fallback already
    # decided for this recording).
    fresh_daemon._audio_callback(_voiced_chunk(800), 800, None, None)
    assert fresh_daemon._encoder.active is False
    assert fresh_daemon._backend.start_encoder.call_count == 0


def test_100_press_release_cycles_no_deadlock(fresh_daemon):
    """Rapid press/release stress test with mocked backend. Verifies the
    new _encoder_* state + worker thread lifecycle doesn't deadlock or
    leak threads when exercised at high frequency. Inert mocks make each
    cycle essentially zero-cost."""
    import gc
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_inert_mocks(fresh_daemon, streaming=True)

    # Patch ENCODER_FINALIZE_TIMEOUT short so any wedged worker doesn't
    # add multi-second waits to the stress loop. Same for the drain sleep
    # in _on_release — we'll skip it by manually doing the state flips
    # to keep the test fast.
    for _ in range(100):
        fresh_daemon._on_press(Key.alt_r)
        # Feed one chunk so encoder lazy-spawns
        fresh_daemon._audio_callback(_voiced_chunk(800), 800, None, None)
        # Skip the 80 ms drain — flip state directly + join worker
        with fresh_daemon._st.lock:
            fresh_daemon._st.active_trigger = None
            fresh_daemon._st.recording = False
        if fresh_daemon._encoder.active:
            fresh_daemon._encoder._stop_event.set()
            if fresh_daemon._encoder._thread is not None:
                fresh_daemon._encoder._thread.join(timeout=0.5)
            with fresh_daemon._st.lock:
                fresh_daemon._encoder.active = False
        fresh_daemon._encoder._stop_event.clear()

    # After 100 cycles, no leftover thread should be alive (the fresh
    # fixture would also catch this on teardown, but we want explicit
    # confirmation in the body).
    gc.collect()
    alive = [t for t in threading.enumerate()
             if t.name.startswith("Thread") and t.is_alive()
             and t is not threading.current_thread()]
    # Filter out pytest's own helper threads (best-effort: name prefix).
    # Any worker still around would have been spawned by us; we joined
    # them all. So 0 expected from this test.
    leaked = [t for t in alive
              if getattr(t, "_target", None) is not None
              and getattr(t._target, "__name__", "") == "_encoder_worker"]
    assert leaked == [], f"leaked encoder workers: {leaked}"


# ---------------------------------------------------------------------------
# v0.7.5 voice-edit mode
# ---------------------------------------------------------------------------

def _install_edit_mocks(daemon_mod, *, seqno_bumps: bool = True,
                        selection_text: str = "selected text",
                        original_clipboard: str = "original clipboard",
                        edit_result: str | None = "edited text",
                        asr_instruction: str = "make it formal"):
    """Voice-edit-specific mock setup. Builds on _install_inert_mocks.

    seqno_bumps=True simulates a successful selection capture (clipboard
    sequence number bumps after simulate_copy → daemon reads new
    selection); False simulates no selection (seqno unchanged → daemon
    aborts).

    edit_result=None simulates a polish.edit() failure (e.g. model
    returned empty/None); the daemon should restore the original
    clipboard and play the fail beep instead of pasting.
    """
    _install_inert_mocks(daemon_mod)
    pb = daemon_mod._pasteboard
    # get_text is called twice per capture: first to save original
    # clipboard, then to read the selection after simulate_copy.
    pb.get_text.side_effect = [original_clipboard, selection_text]
    if seqno_bumps:
        pb.clipboard_seqno.side_effect = [100, 101]  # before, after
    else:
        pb.clipboard_seqno.side_effect = [100, 100]  # unchanged
    pb.simulate_copy.return_value = True
    daemon_mod._polisher.edit.return_value = edit_result
    daemon_mod._backend.transcribe.return_value = (asr_instruction, "zh")
    # Enable voice-edit explicitly per test (fresh_daemon resets to None).
    daemon_mod.EDIT_TRIGGER_KEYS = {Key.f13}


def test_edit_trigger_distinct_from_dictate_trigger(fresh_daemon):
    """Pasteboard's per-platform default_edit_trigger_keys must not
    overlap with default_trigger_keys (dictate) — overlap would create
    routing ambiguity in `_on_press` (both is_edit and is_dictate true).

    Checks the REAL pasteboard subclass for the current platform (Win
    or Mac), not a mock — this is a per-platform contract assertion."""
    from stt_platform import build_pasteboard
    pb = build_pasteboard()
    assert pb.default_edit_trigger_keys.isdisjoint(
        pb.default_trigger_keys
    ), (
        f"default_edit_trigger_keys={pb.default_edit_trigger_keys} "
        f"overlaps with default_trigger_keys={pb.default_trigger_keys} "
        f"on {type(pb).__name__}"
    )


def test_edit_press_captures_selection(fresh_daemon):
    """Press of edit trigger → daemon round-trips clipboard, captures
    selection text, populates _edit_* globals, starts recording."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_edit_mocks(fresh_daemon, seqno_bumps=True,
                        selection_text="hello world",
                        original_clipboard="prev clipboard")

    fresh_daemon._on_press(Key.f13)

    assert fresh_daemon._st.edit_mode is True
    assert fresh_daemon._st.edit_selection == "hello world"
    assert fresh_daemon._st.edit_original_clipboard == "prev clipboard"
    assert fresh_daemon._st.recording is True
    assert fresh_daemon._st.active_trigger == Key.f13
    # simulate_copy was actually called
    fresh_daemon._pasteboard.simulate_copy.assert_called_once()


def test_edit_press_aborts_when_no_selection(fresh_daemon, caplog):
    """If the focused app has no selection (seqno doesn't bump after
    simulate_copy), the daemon should NOT start recording, edit state
    stays clean, fail beep would have played."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_edit_mocks(fresh_daemon, seqno_bumps=False)

    with caplog.at_level(logging.DEBUG, logger="stt"):
        fresh_daemon._on_press(Key.f13)

    assert fresh_daemon._st.recording is False
    assert fresh_daemon._st.edit_mode is False
    assert fresh_daemon._st.edit_selection is None
    assert fresh_daemon._st.active_trigger is None
    assert "no selection" in caplog.text.lower()


def test_edit_press_routes_release_to_transcribe_and_emit_edit(fresh_daemon,
                                                                 monkeypatch):
    """End-to-end routing: edit press + release → spawns thread targeting
    `_transcribe_and_emit_edit` with (selection, original_clipboard)."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_edit_mocks(fresh_daemon, selection_text="some selection",
                        original_clipboard="prev")

    invocations = []

    def fake_edit(selection, original):
        invocations.append(("edit", selection, original))

    def fake_polish():
        invocations.append(("polish",))

    monkeypatch.setattr(fresh_daemon, "_transcribe_and_emit_edit", fake_edit)
    monkeypatch.setattr(fresh_daemon, "_transcribe_and_emit", fake_polish)

    fresh_daemon._on_press(Key.f13)
    fresh_daemon._on_release(Key.f13)
    # _on_release spawns a daemon thread; give it time to invoke fake_edit
    time.sleep(0.3)

    assert len(invocations) == 1, (
        f"expected exactly 1 spawn, got {invocations}"
    )
    assert invocations[0] == ("edit", "some selection", "prev")


def test_dictate_press_unaffected_by_edit_keys(fresh_daemon, monkeypatch):
    """Regression guard: pressing a regular dictate trigger should NOT
    trigger any voice-edit behaviour (no clipboard read, no simulate_copy,
    routes to _transcribe_and_emit)."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_edit_mocks(fresh_daemon)  # also sets EDIT_TRIGGER_KEYS

    invocations = []
    monkeypatch.setattr(fresh_daemon, "_transcribe_and_emit",
                        lambda: invocations.append("polish"))
    monkeypatch.setattr(fresh_daemon, "_transcribe_and_emit_edit",
                        lambda s, o: invocations.append("edit"))

    fresh_daemon._on_press(Key.alt_r)
    fresh_daemon._on_release(Key.alt_r)
    time.sleep(0.3)

    assert invocations == ["polish"]
    assert fresh_daemon._st.edit_mode is False
    # Dictate path must NOT have touched the clipboard or simulated Cmd+C
    fresh_daemon._pasteboard.simulate_copy.assert_not_called()
    fresh_daemon._pasteboard.get_text.assert_not_called()


def test_edit_clipboard_restored_on_success(fresh_daemon):
    """End-to-end: after a successful edit, the daemon must have:
    1. set_text(edit_result) — pasted the LLM output
    2. set_text(original_clipboard) — restored the pre-edit clipboard"""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_edit_mocks(fresh_daemon, selection_text="the selection",
                        original_clipboard="user's saved clipboard",
                        edit_result="LLM output")

    fresh_daemon._on_press(Key.f13)
    # Pre-load buffer with non-silent audio so _trim_silence doesn't
    # strip everything. 16000 samples of low-amplitude noise = 1s.
    fresh_daemon._st.buffer.append(
        np.full(16000, 0.05, dtype=np.float32)
    )
    fresh_daemon._st.recording_samples = 16000
    fresh_daemon._on_release(Key.f13)
    time.sleep(0.5)  # let edit thread run

    set_text_calls = [c.args[0] for c
                      in fresh_daemon._pasteboard.set_text.call_args_list]
    assert "LLM output" in set_text_calls, (
        f"paste of edit result missing: {set_text_calls}"
    )
    assert "user's saved clipboard" in set_text_calls, (
        f"clipboard restore missing: {set_text_calls}"
    )
    # Restore must come AFTER paste
    paste_idx = set_text_calls.index("LLM output")
    restore_idx = set_text_calls.index("user's saved clipboard")
    assert restore_idx > paste_idx, "restore must run after paste"


def test_edit_clipboard_restored_on_polish_failure(fresh_daemon):
    """polisher.edit returning None → daemon aborts the paste but MUST
    still restore the original clipboard (finally block guarantees this)."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_edit_mocks(fresh_daemon, selection_text="sel",
                        original_clipboard="orig", edit_result=None)

    fresh_daemon._on_press(Key.f13)
    fresh_daemon._st.buffer.append(
        np.full(16000, 0.05, dtype=np.float32)
    )
    fresh_daemon._st.recording_samples = 16000
    fresh_daemon._on_release(Key.f13)
    time.sleep(0.5)

    set_text_calls = [c.args[0] for c
                      in fresh_daemon._pasteboard.set_text.call_args_list]
    assert "orig" in set_text_calls, (
        f"clipboard restore missing on polish failure: {set_text_calls}"
    )


def test_edit_polisher_called_with_selection_and_instruction(fresh_daemon):
    """polisher.edit must be invoked with (selection, asr_instruction).
    Regression guard against arg-order swap that would silently treat
    the instruction as the selection."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_edit_mocks(
        fresh_daemon,
        selection_text="my code",
        asr_instruction="add docstring",
        edit_result="my code\n# docstring",
    )

    fresh_daemon._on_press(Key.f13)
    fresh_daemon._st.buffer.append(
        np.full(16000, 0.05, dtype=np.float32)
    )
    fresh_daemon._st.recording_samples = 16000
    fresh_daemon._on_release(Key.f13)
    time.sleep(0.5)

    fresh_daemon._polisher.edit.assert_called_once_with(
        "my code", "add docstring",
    )


def test_edit_busy_path_drops_and_restores(fresh_daemon):
    """If a previous transcribe is still running (_processing=True) when
    voice-edit release fires, the daemon drops the audio AND still
    restores the original clipboard (the finally block guarantees this
    so the user's pre-edit clipboard is never silently lost on busy)."""
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_edit_mocks(fresh_daemon)
    fresh_daemon._st.processing = True  # simulate busy

    # Call _transcribe_and_emit_edit directly — bypass the thread spawn
    # to avoid the daemon thread timing variance.
    fresh_daemon._transcribe_and_emit_edit("sel", "orig")

    set_text_calls = [c.args[0] for c
                      in fresh_daemon._pasteboard.set_text.call_args_list]
    assert "orig" in set_text_calls, (
        f"busy-path restore missing: {set_text_calls}"
    )
    # Polisher.edit must NOT have been called (busy-drop happens before)
    fresh_daemon._polisher.edit.assert_not_called()


def test_edit_press_skips_capture_on_key_repeat(fresh_daemon):
    """v0.7.5 hotfix regression guard: OS key-repeat fires _on_press on
    every repeat tick (~24×/s on Windows for F13). Without the cheap
    _active_trigger early-return BEFORE _capture_selection, every repeat
    would invoke a 100 ms selection capture + simulate_copy + fail beep +
    log spam (24×/s of all of that). Live-log evidence on 2026-05-24
    showed 24 'no selection captured' messages per single F13 press.

    This test asserts that the second _on_press (simulating a repeat
    tick) does NOT invoke simulate_copy / get_text / play a fail beep.
    """
    fresh_daemon.TRIGGER_KEYS = {Key.alt_r}
    _install_edit_mocks(fresh_daemon, seqno_bumps=True)

    # First press — real start of voice-edit
    fresh_daemon._on_press(Key.f13)
    assert fresh_daemon._st.active_trigger == Key.f13
    capture_calls_after_first = fresh_daemon._pasteboard.simulate_copy.call_count
    assert capture_calls_after_first == 1, (
        f"first press should simulate_copy once, got "
        f"{capture_calls_after_first}"
    )

    # Second press — OS key-repeat. Must be a cheap no-op (no extra
    # simulate_copy, no extra get_text, no fail beep).
    fresh_daemon._on_press(Key.f13)
    capture_calls_after_repeat = fresh_daemon._pasteboard.simulate_copy.call_count
    assert capture_calls_after_repeat == 1, (
        f"key-repeat must NOT re-invoke simulate_copy; "
        f"call count went from {capture_calls_after_first} to "
        f"{capture_calls_after_repeat}"
    )
