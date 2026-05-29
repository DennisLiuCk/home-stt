"""Audio helpers extracted from stt-daemon.py.

Contains:
  - post_process()          — Simplified→Traditional Chinese + CJK↔ASCII spacing
  - _detect_output_samplerate() — query default output device sample rate
  - _play_beep()            — short sine-wave audio feedback tone
  - _trim_silence()         — RMS-based leading/trailing silence trimmer
"""
from __future__ import annotations

import logging
import re

import numpy as np
import sounddevice as sd
from opencc import OpenCC

logger = logging.getLogger("stt.audio")


# ---------------------------------------------------------------------------
# Text post-processing (backend-agnostic)
# ---------------------------------------------------------------------------
_s2twp = OpenCC("s2twp")
_CJK   = r"[㐀-鿿]"
_AW    = r"[A-Za-z0-9]"
# Precompiled once at import (matches the codebase convention used for
# log-parsing regexes elsewhere); post_process runs on every utterance.
_RE_CJK_AW = re.compile(f"({_CJK})({_AW})")
_RE_AW_CJK = re.compile(f"({_AW})({_CJK})")


def post_process(text: str) -> str:
    """Simplified -> Taiwan-traditional (with TW phrase mapping via s2twp),
    then add spaces at CJK <-> ASCII edges. Idempotent."""
    text = _s2twp.convert(text)
    text = _RE_CJK_AW.sub(r"\1 \2", text)
    text = _RE_AW_CJK.sub(r"\1 \2", text)
    return text


# ---------------------------------------------------------------------------
# Audio feedback (cross-platform)
# ---------------------------------------------------------------------------
def _detect_output_samplerate() -> int:
    """Return the default output device's preferred sample rate, falling
    back to 44.1 kHz if querying fails. Cached at module load."""
    try:
        return int(sd.query_devices(kind="output")["default_samplerate"])
    except Exception:
        return 44100


# Populated lazily by _get_beep_sr() on first beep — keeps PortAudio /
# CoreAudio out of the import path so the daemon can start on macOS
# LaunchAgent (pre-login session, audio HAL not yet ready) or other
# headless contexts without emitting noisy errors before any [stt] log
# line. By the time the first beep fires (on first key press) the
# listener / audio stream are already up, so HAL is in a known state.
_BEEP_SR: int | None = None


def _get_beep_sr() -> int:
    global _BEEP_SR
    if _BEEP_SR is None:
        _BEEP_SR = _detect_output_samplerate()
    return _BEEP_SR


def _play_beep(freq_hz: float,
               duration_ms: int,
               volume: float,
               enabled: bool) -> None:
    if not enabled:
        return
    try:
        sr = _get_beep_sr()
        n = int(sr * duration_ms / 1000)
        if n <= 0:
            return
        t = np.arange(n) / sr
        wave = (volume * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)

        # Raised-cosine fade in/out (~15 ms each) — smoother than linear,
        # eliminates click pops at start/end of the tone.
        fade = min(int(sr * 0.015), n // 2)
        if fade > 0:
            ramp = (0.5 * (1 - np.cos(np.pi * np.arange(fade) / fade))).astype(np.float32)
            wave[:fade]  *= ramp
            wave[-fade:] *= ramp[::-1]

        # Silence pad at the start: CoreAudio's first-buffer transient
        # (when sd.play() opens a new OutputStream) lands in this silence
        # rather than on top of the sine attack, so the user hears one
        # clean "ding" instead of "click-ding".
        pad = np.zeros(int(sr * 0.005), dtype=np.float32)
        wave = np.concatenate([pad, wave])

        sd.play(wave, samplerate=sr)
    except Exception as e:
        # Beep is purely cosmetic — never let it break transcription.
        logger.warning("beep failed: %s", e)


# ---------------------------------------------------------------------------
# Silence trimming
# ---------------------------------------------------------------------------
def _trim_silence(samples: np.ndarray,
                  sample_rate: int = 16000,
                  threshold_dbfs: float = -50.0,
                  frame_ms: int = 30,
                  margin_ms: int = 100) -> np.ndarray:
    if len(samples) < sample_rate * 0.1:
        return samples  # too short to meaningfully trim
    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = len(samples) // frame_size
    if n_frames == 0:
        return samples
    frames = samples[:n_frames * frame_size].reshape(n_frames, frame_size)
    # RMS per frame, vectorised.
    # frames is already float32; np.square keeps the whole RMS computation in
    # float32 (avoids a full float64 copy + a float64 square temporary) — the
    # result only feeds a coarse fixed -50 dBFS gate, so float64 buys nothing.
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    above = rms > threshold
    if not above.any():
        return samples[:0]  # all silence — return empty array (correct dtype)
    first = int(above.argmax())
    last = n_frames - int(above[::-1].argmax()) - 1
    margin = int(sample_rate * margin_ms / 1000)
    start = max(0, first * frame_size - margin)
    end = min(len(samples), (last + 1) * frame_size + margin)
    return samples[start:end]
