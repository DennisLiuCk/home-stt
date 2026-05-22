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
__version__ = "0.3.0"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_RATE      = 16000
MIN_AUDIO_SEC    = 0.3                 # taps shorter than this are ignored

# STT backend + model defaults per platform.
#   Apple Silicon (arm64): Qwen3-ASR-0.6B via mlx-qwen3-asr. Strong Chinese
#       punctuation + native zh-en code-switching beat Whisper turbo for
#       our 80%-zh + tech-loanword usage. Default since v0.3.0; v0.2.0/0.2.1
#       default was mlx-whisper large-v3-turbo, still available via
#       STT_BACKEND="mlx-whisper".
#   Windows / Linux / Intel Mac / Rosetta: faster-whisper large-v3-turbo
#       (CUDA float16 when available, CPU int8 fallback).
# Override by hardcoding STT_BACKEND / STT_MODEL below.
if sys.platform == "darwin" and _host_platform.machine() == "arm64":
    _DEFAULT_BACKEND = "qwen3-asr"
    _DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
else:
    _DEFAULT_BACKEND = "faster-whisper"
    _DEFAULT_MODEL = "large-v3-turbo"

STT_BACKEND      = _DEFAULT_BACKEND
# Model identifier passed to the backend. Interpretation is backend-specific:
#   faster-whisper:  Whisper model name ("large-v3-turbo", "medium", ...)
#   mlx-whisper:     short name or HF repo id (auto-resolves "large-v3-turbo"
#                    to "mlx-community/whisper-large-v3-turbo")
#   qwen3-asr:       HF repo id ("Qwen/Qwen3-ASR-0.6B" / "Qwen/Qwen3-ASR-1.7B")
#                    or short aliases "0.6B" / "1.7B" (case-insensitive). Anything
#                    unrecognised falls back to the 0.6B variant.
#   sense-voice:     ModelScope ID, e.g. "iic/SenseVoiceSmall" (planned)
STT_MODEL        = _DEFAULT_MODEL

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
               duration_ms: int = BEEP_DURATION_MS,
               volume: float = BEEP_VOLUME) -> None:
    if not BEEPS_ENABLED:
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
            # Lock temperature to greedy decoding. mlx_whisper's default
            # schedule (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) retries up to 6 times
            # on low-confidence clips (background noise, mumbles, single
            # English word) — that can push latency from ~0.3s to ~2s.
            # Matches FasterWhisperBackend's beam_size=1 no-retry behaviour.
            temperature=0.0,
            condition_on_previous_text=False,
            verbose=None,
        )
        text = (result.get("text") or "").strip()
        language = result.get("language") or ""
        return text, language

    def warmup(self) -> None:
        # First call materialises model weights + Metal kernels. Once warm,
        # subsequent transcribes are sub-second on Apple Silicon turbo.
        warm_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self._mlx_whisper.transcribe(
            warm_audio,
            path_or_hf_repo=self._model_name,
            temperature=0.0,
            verbose=None,
        )


class Qwen3AsrBackend(STTBackend):
    """Qwen3-ASR via mlx-qwen3-asr (Apple Silicon Metal-native).

    Alibaba's Qwen3-ASR, released 2026-01 under Apache-2.0. Two model sizes:
        Qwen/Qwen3-ASR-0.6B  — default, ~1.2 GB fp16, 92ms TTFT
        Qwen/Qwen3-ASR-1.7B  — higher accuracy, ~3.4 GB fp16

    Strengths vs Whisper turbo for this daemon's typical usage:
      - Native Chinese punctuation (Qwen3 LLM backbone — text generation is
        a first-class objective, unlike Whisper which trained on subtitles
        where punctuation is inconsistent).
      - Native code-switching: zh + occasional English proper nouns
        (Python, MLX, async, function, …) handled in one pass without
        the model needing to flip language mid-sentence.
      - 52 languages + 22 Chinese dialects in the training mix.

    Accepts:
      - Fully qualified HF repo ids ("Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B")
      - Short aliases ("0.6B", "1.7B", case-insensitive)
      - Anything else falls back to the 0.6B default — so a user with
        STT_MODEL = "large-v3-turbo" who just flips STT_BACKEND still
        gets a working setup without editing two lines.
    """

    name = "qwen3-asr"

    # Map mlx-qwen3-asr's human-readable language names back to the ISO-ish
    # short codes used elsewhere in the daemon log ("zh", "en", ...).
    _LANG_NORM = {
        "chinese":    "zh",
        "english":    "en",
        "japanese":   "ja",
        "korean":     "ko",
        "french":     "fr",
        "german":     "de",
        "spanish":    "es",
        "portuguese": "pt",
        "russian":    "ru",
        "italian":    "it",
        "arabic":     "ar",
    }

    def __init__(self, model_name: str):
        import mlx_qwen3_asr  # lazy import (Apple Silicon only)

        self._mqa = mlx_qwen3_asr
        self._model_name = self._resolve_model_name(model_name)
        self._device_label = "Apple Silicon (Metal, MLX) — Qwen3-ASR"

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        # Already a HF repo id — pass through (covers community forks too).
        if "/" in model_name:
            return model_name
        m = model_name.lower()
        if "1.7b" in m or "1_7" in m:
            return "Qwen/Qwen3-ASR-1.7B"
        if "0.6b" in m or "0_6" in m:
            return "Qwen/Qwen3-ASR-0.6B"
        # User probably has STT_MODEL = "large-v3-turbo" left over from the
        # Whisper path; pick the smaller Qwen3-ASR variant by default.
        return "Qwen/Qwen3-ASR-0.6B"

    @property
    def device_label(self) -> str:
        return self._device_label

    def transcribe(self, samples: np.ndarray) -> tuple[str, str]:
        result = self._mqa.transcribe(
            samples,
            model=self._model_name,
            verbose=False,
        )
        text = (getattr(result, "text", "") or "").strip()
        raw_lang = (getattr(result, "language", "") or "").strip().lower()
        # Normalise "Chinese" → "zh", "English" → "en", etc. Falls back to
        # the first two letters of whatever the model returned so unknown
        # languages still produce something sensible in the log line.
        language = self._LANG_NORM.get(raw_lang, raw_lang[:2] if raw_lang else "")
        return text, language

    def warmup(self) -> None:
        # First call downloads weights (~1.2 GB for 0.6B) on first run and
        # materialises Metal kernels. Once warm, the next transcribe is
        # near the ~92ms time-to-first-token claimed by the upstream MLX
        # port.
        warm_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self._mqa.transcribe(
            warm_audio,
            model=self._model_name,
            verbose=False,
        )


def build_backend(name: str, model: str) -> STTBackend:
    """Factory. To add a new backend: implement STTBackend in a new class,
    add a branch here, and update STT_BACKEND in the Config section."""
    if name == "faster-whisper":
        return FasterWhisperBackend(model)
    if name == "mlx-whisper":
        return MlxWhisperBackend(model)
    if name == "qwen3-asr":
        return Qwen3AsrBackend(model)
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
        # content). Both set_text and paste return False on failure; we
        # suppress the success beep + use a different log line so the user
        # sees one consistent signal of what actually happened.
        if not _pasteboard.set_text(text):
            print(f"[stt] {language} {elapsed:.2f}s clipboard write failed — "
                  f"'{text}' NOT inserted",
                  file=sys.stderr, flush=True)
            return
        time.sleep(0.15)
        paste_ok = _pasteboard.paste()
        if paste_ok:
            _play_beep(BEEP_END_HZ)

        try:
            if paste_ok:
                print(f"[stt] {language} {elapsed:.2f}s -> {text}", flush=True)
            else:
                # paste() already printed a user-facing 'paste blocked' line
                # to the main log; we record the transcript itself so it's
                # discoverable even when auto-paste didn't fire.
                print(f"[stt] {language} {elapsed:.2f}s clipboard-only -> {text}",
                      flush=True)
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
    paste_desc = _pasteboard.describe_paste_path()
    if paste_desc:
        print(f"[stt] paste path: {paste_desc}", flush=True)

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
