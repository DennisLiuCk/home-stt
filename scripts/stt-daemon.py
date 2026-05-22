"""
Hold-to-talk voice → text daemon.

Hold the trigger key (Right Alt/AltGr or Right Ctrl on Windows; Right Option
on macOS) to record from the default microphone. Release to:
  1. Transcribe via the active STT backend (default: faster-whisper on
     Windows/Linux/Intel-Mac, mlx-whisper on Apple Silicon).
  2. Convert simplified Chinese to Taiwan-traditional via OpenCC.
  3. Insert spaces at zh ↔ en/digit boundaries.
  4. Place the text on the system clipboard AND simulate Ctrl+V / Cmd+V to
     paste it into the focused window (atomic paste — IME cannot interrupt).

Stdin/stdout is forced to UTF-8 so simplified-Chinese characters can be
logged on a zh-TW Windows locale (default cp950 cannot encode them).

────────────────────────────────────────────────────────────────────────
Backend abstraction
────────────────────────────────────────────────────────────────────────
  The STT engine is hidden behind the `STTBackend` interface so the rest
  of the pipeline (mic capture → post-processing → clipboard+paste) stays
  the same when swapping engines. Switch by changing `STT_BACKEND` below
  and adding a class. See `build_backend()` for the dispatch table.

  Implemented:
    - faster-whisper (Whisper large-v3-turbo via CTranslate2 — CPU / CUDA)
    - mlx-whisper    (Whisper large-v3-turbo via Apple MLX — Metal native)
  Planned (roadmap):
    - sense-voice  (Alibaba FunASR SenseVoice-Small — fast, small, multilang)
    - paraformer   (Alibaba FunASR Paraformer-zh — Chinese SOTA)

────────────────────────────────────────────────────────────────────────
Platform abstraction
────────────────────────────────────────────────────────────────────────
  Clipboard write / paste-keystroke simulation / default global trigger keys
  live behind the `Pasteboard` interface in `stt_platform.py`. Adding a
  third platform (Linux X11 / Wayland) means adding `stt_platform_linux.py`
  and a branch in `build_pasteboard()` — the daemon itself does not change.
"""
from __future__ import annotations

import platform as _host_platform
import re
import sys
import threading
import time
from abc import ABC, abstractmethod

# ---------------------------------------------------------------------------
# Stdout: force UTF-8 — Whisper may output simplified Chinese before OpenCC
# runs (during partial logs), and the default cp950 codec on zh-TW Windows
# can't encode those characters.
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


import numpy as np
import sounddevice as sd
from opencc import OpenCC
from pynput import keyboard
from pynput.keyboard import Key

from stt_platform import Pasteboard, build_pasteboard


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
__version__ = "0.2.0"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_RATE      = 16000
MIN_AUDIO_SEC    = 0.3                 # taps shorter than this are ignored

# STT backend defaults are platform-aware. Apple Silicon gets MLX (Metal-
# native large-v3-turbo, latency comparable to NVIDIA float16). Everything
# else (Windows, Linux, Intel Mac, Rosetta) defaults to faster-whisper,
# which auto-falls-back to CPU int8 when CUDA is unavailable.
# Override by hardcoding STT_BACKEND below.
if sys.platform == "darwin" and _host_platform.machine() == "arm64":
    _DEFAULT_BACKEND = "mlx-whisper"
else:
    _DEFAULT_BACKEND = "faster-whisper"

STT_BACKEND      = _DEFAULT_BACKEND
# Model identifier passed to the backend. Interpretation is backend-specific:
#   faster-whisper:  Whisper model name ("large-v3-turbo", "medium", ...)
#   mlx-whisper:     short name or HF repo id (auto-resolves "large-v3-turbo"
#                    to "mlx-community/whisper-large-v3-turbo")
#   sense-voice:     ModelScope ID, e.g. "iic/SenseVoiceSmall" (planned)
STT_MODEL        = "large-v3-turbo"

# Set of pynput Key/character triggers to listen for as hold-to-record keys.
# `None` means "use the platform default" (Windows: Right Alt + Right Ctrl;
# macOS: Right Option). Override with e.g. `{Key.f13}` to lock to one key.
TRIGGER_KEYS: set | None = None

# Audio feedback — short sine-wave tones at trigger-press / paste-done
# so the user knows when recording starts and when transcription has
# landed. Cross-platform: relies only on sounddevice (already a dep).
BEEPS_ENABLED    = True
BEEP_START_HZ    = 880                 # A5, "bright" — start of recording
BEEP_END_HZ      = 660                 # E5, "calmer" — paste done
BEEP_DURATION_MS = 80
BEEP_VOLUME      = 0.15                # 0.0–1.0; keep low to avoid mic bleed


# ---------------------------------------------------------------------------
# Text post-processing (backend-agnostic)
# ---------------------------------------------------------------------------
_s2tw = OpenCC("s2tw")
_CJK  = r"[㐀-鿿]"
_AW   = r"[A-Za-z0-9]"


def post_process(text: str) -> str:
    """Simplified → Taiwan-traditional, then add spaces at CJK ↔ ASCII edges."""
    text = _s2tw.convert(text)
    text = re.sub(f"({_CJK})({_AW})", r"\1 \2", text)
    text = re.sub(f"({_AW})({_CJK})", r"\1 \2", text)
    return text


# ---------------------------------------------------------------------------
# Audio feedback (cross-platform — uses sounddevice which we already need
# for capture). Generates a short sine wave and plays it non-blocking via
# sd.play().
#
# A few subtleties that bit us on macOS:
#   - The mic stream is open at SAMPLE_RATE (16 kHz), but the default OS
#     output device is usually 44.1/48 kHz. If we hand sd.play() a 16 kHz
#     buffer, CoreAudio resamples it, and the resampling artifacts on a
#     short (~80 ms) tone are audible as a "broken" / "double-ding" sound.
#     Fix: generate the wave at the OUTPUT device's native rate.
#   - sd.play() opens a fresh OutputStream every call, and on macOS the
#     stream-open transient is a brief audible click. Fix: prepend a tiny
#     (~5 ms) silence pad so the click happens during silence, not on top
#     of the sine attack.
#   - Linear fades produce harsher edges than half-cosine ramps; using a
#     raised-cosine envelope makes the tone sound "tighter".
# ---------------------------------------------------------------------------
def _detect_output_samplerate() -> int:
    """Return the default output device's preferred sample rate, falling
    back to 44.1 kHz if querying fails. Cached at module load."""
    try:
        return int(sd.query_devices(kind="output")["default_samplerate"])
    except Exception:
        return 44100


_BEEP_SR = _detect_output_samplerate()


def _play_beep(freq_hz: float,
               duration_ms: int = BEEP_DURATION_MS,
               volume: float = BEEP_VOLUME) -> None:
    if not BEEPS_ENABLED:
        return
    try:
        sr = _BEEP_SR
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
        print(f"[stt] beep failed: {e}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# STT Backend abstraction
#
# Concrete backends own model loading, warmup, and inference. The pipeline
# only needs `transcribe(samples) -> (text, language)` and an optional
# `warmup()` to pay one-time JIT costs up front.
# ---------------------------------------------------------------------------
class STTBackend(ABC):
    """Interface for speech-to-text engines. Implementations decide their
    own model + device + precision; the daemon just hands them raw float32
    audio at 16 kHz mono and expects a (text, language) tuple back."""

    name: str = "abstract"

    @abstractmethod
    def transcribe(self, samples: np.ndarray) -> tuple[str, str]:
        """Run inference on a 1-D float32 PCM array (16 kHz, mono).
        Returns (raw_text, language_code). Backends do NOT post-process
        text (no s2tw, no spacing) — that's done downstream."""

    def warmup(self) -> None:
        """Optional: do a dummy inference so the first real request is fast.
        Default is no-op; backends override if they have a cold start."""

    @property
    def device_label(self) -> str:
        """Short human label like 'CUDA (float16)' for startup logging."""
        return "unknown"


class FasterWhisperBackend(STTBackend):
    """OpenAI Whisper via CTranslate2 (faster-whisper). Tries CUDA float16
    first, falls back to CPU int8 if CUDA initialisation fails."""

    name = "faster-whisper"

    def __init__(self, model_name: str):
        # Lazy import — keeps `faster_whisper` out of the import graph when
        # a different backend is selected.
        from faster_whisper import WhisperModel

        try:
            self._model = WhisperModel(model_name, device="cuda",
                                       compute_type="float16")
            self._device_label = "CUDA (float16)"
        except Exception as e:
            print(f"[stt] CUDA load failed ({e}); falling back to CPU int8.",
                  flush=True)
            self._model = WhisperModel(model_name, device="cpu",
                                       compute_type="int8")
            self._device_label = "CPU (int8)"

    @property
    def device_label(self) -> str:
        return self._device_label

    def transcribe(self, samples: np.ndarray) -> tuple[str, str]:
        segments, info = self._model.transcribe(
            samples,
            beam_size=1,                       # greedy — 3-5x faster, near-identical quality
            condition_on_previous_text=False,  # avoids cross-segment carryover
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        text = "".join(s.text for s in segments).strip()
        return text, info.language

    def warmup(self) -> None:
        # Cold-start CUDA JIT compile on Blackwell can take ~10s; once
        # cached it's sub-second. Pay it now so the user's first
        # hold-to-talk doesn't eat it.
        warm_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        list(self._model.transcribe(warm_audio, beam_size=1,
                                    vad_filter=False)[0])


class MlxWhisperBackend(STTBackend):
    """Whisper via Apple MLX (Metal-native).

    Accepts either a short Whisper model name like ``large-v3-turbo`` (auto-
    resolved to ``mlx-community/whisper-large-v3-turbo``) or a fully-
    qualified HuggingFace repo id. Apple Silicon only — on Intel Mac /
    Rosetta the pipeline falls back to ``faster-whisper`` via the default
    backend picker in the Config section.
    """

    name = "mlx-whisper"

    def __init__(self, model_name: str):
        import mlx_whisper  # lazy import (Apple Silicon only)

        self._mlx_whisper = mlx_whisper
        if "/" not in model_name:
            model_name = f"mlx-community/whisper-{model_name}"
        self._model_name = model_name
        self._device_label = "Apple Silicon (Metal, MLX)"

    @property
    def device_label(self) -> str:
        return self._device_label

    def transcribe(self, samples: np.ndarray) -> tuple[str, str]:
        result = self._mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=self._model_name,
            condition_on_previous_text=False,
            verbose=None,
        )
        text = result.get("text", "").strip()
        language = result.get("language", "") or ""
        return text, language

    def warmup(self) -> None:
        # First call materialises model weights + Metal kernels. Once warm,
        # subsequent transcribes are sub-second on Apple Silicon turbo.
        warm_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self._mlx_whisper.transcribe(
            warm_audio,
            path_or_hf_repo=self._model_name,
            verbose=None,
        )


def build_backend(name: str, model: str) -> STTBackend:
    """Factory. To add a new backend: implement STTBackend in a new class,
    add a branch here, and update STT_BACKEND in the Config section."""
    if name == "faster-whisper":
        return FasterWhisperBackend(model)
    if name == "mlx-whisper":
        return MlxWhisperBackend(model)
    # ── Future backends ────────────────────────────────────────────────
    # elif name == "sense-voice":
    #     return SenseVoiceBackend(model)
    # elif name == "paraformer":
    #     return ParaformerBackend(model)
    raise ValueError(f"Unknown STT backend: {name!r}")


# ---------------------------------------------------------------------------
# Audio capture state
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
_buffer: list[np.ndarray] = []
_recording = False
_active_trigger = None   # which TRIGGER_KEYS member is currently held, or None
_processing = False


def _audio_callback(indata, frames, time_info, status) -> None:
    if status:
        print(f"[stt] audio status: {status}", file=sys.stderr, flush=True)
    with _state_lock:
        if _recording:
            _buffer.append(indata.copy())


# ---------------------------------------------------------------------------
# Transcription pipeline (backend-agnostic)
# ---------------------------------------------------------------------------
_backend: STTBackend       # set in main()
_pasteboard: Pasteboard    # set in main()


def _transcribe_and_emit() -> None:
    global _processing, _buffer
    with _state_lock:
        if _processing:
            return
        _processing = True
        chunks = _buffer
        _buffer = []
    try:
        if not chunks:
            return
        samples = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
        if len(samples) < SAMPLE_RATE * MIN_AUDIO_SEC:
            print(f"[stt] too short ({len(samples)/SAMPLE_RATE:.2f}s)",
                  flush=True)
            return
        t0 = time.time()
        raw, language = _backend.transcribe(samples)
        elapsed = time.time() - t0
        if not raw:
            print(f"[stt] empty ({language}, {elapsed:.2f}s)", flush=True)
            return
        text = post_process(raw)

        # Set clipboard, then paste — atomic, no per-char IME drama.
        # Tiny sleep lets the clipboard write settle before the keystroke
        # (otherwise the paste keystroke can race ahead and paste empty/stale
        # content).
        _pasteboard.set_text(text)
        time.sleep(0.15)
        _pasteboard.paste()
        _play_beep(BEEP_END_HZ)

        try:
            print(f"[stt] {language} {elapsed:.2f}s -> {text}", flush=True)
        except Exception:
            print(f"[stt] inserted ({elapsed:.2f}s); log encoding failed",
                  flush=True)
    except Exception as e:
        print(f"[stt] error: {e}", file=sys.stderr, flush=True)
    finally:
        with _state_lock:
            _processing = False


# ---------------------------------------------------------------------------
# Keyboard hooks
# ---------------------------------------------------------------------------
def _on_press(key) -> None:
    global _recording, _active_trigger
    if key not in TRIGGER_KEYS:
        return
    with _state_lock:
        if _active_trigger is not None:
            return  # another trigger is already held (or OS key-repeat)
        _active_trigger = key
        _recording = True
        _buffer.clear()
    print(f"[stt] REC ({key})", flush=True)
    _play_beep(BEEP_START_HZ)


def _on_release(key) -> None:
    global _recording, _active_trigger
    if key not in TRIGGER_KEYS:
        return
    with _state_lock:
        if _active_trigger != key:
            return  # releasing a non-active trigger (e.g. tap of the other one)
        _active_trigger = None
        _recording = False
    print("[stt] processing...", flush=True)
    threading.Thread(target=_transcribe_and_emit, daemon=True).start()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    global _backend, _pasteboard, TRIGGER_KEYS

    _pasteboard = build_pasteboard()
    n_native = _pasteboard.register_native_libs()

    print(f"[stt] home-stt v{__version__} starting", flush=True)
    print(f"[stt] platform: {sys.platform} ({_host_platform.machine()}) | "
          f"native libs registered: {n_native}", flush=True)
    print(f"[stt] backend: {STT_BACKEND} | model: {STT_MODEL}", flush=True)

    _backend = build_backend(STT_BACKEND, STT_MODEL)

    if TRIGGER_KEYS is None:
        TRIGGER_KEYS = _pasteboard.default_trigger_keys

    print(f"[stt] warming up on {_backend.device_label}...", flush=True)
    t0 = time.time()
    _backend.warmup()
    trigger_labels = ", ".join(str(k) for k in TRIGGER_KEYS)
    print(f"[stt] warmup {time.time()-t0:.1f}s — hold {trigger_labels} to record.",
          flush=True)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=_audio_callback,
        blocksize=int(SAMPLE_RATE * 0.05),  # 50 ms chunks
    )
    stream.start()
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()
    try:
        listener.join()
    except KeyboardInterrupt:
        print("[stt] bye", flush=True)
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
