"""
Hold-to-talk voice → text daemon.

Hold the trigger key (Right Alt/AltGr or Right Ctrl on Windows; Right Option
on macOS) to record from the default microphone. Release to:
  1. Transcribe via the active STT backend (default: qwen3-asr on both
     Apple Silicon (v0.3.0+) and Windows/Linux (v0.6.0+); faster-whisper
     and mlx-whisper remain available as switchable fallbacks).
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
from text_polisher import TextPostProcessor, build_polisher


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
__version__ = "0.6.0"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_RATE      = 16000
MIN_AUDIO_SEC    = 0.3                 # taps shorter than this are ignored

# STT backend + model defaults per platform.
#   macOS (Apple Silicon only as of v0.4.0): Qwen3-ASR-0.6B via mlx-qwen3-asr.
#       Strong Chinese punctuation + native zh-en code-switching beat Whisper
#       turbo for our 80%-zh + tech-loanword usage. Default since v0.3.0;
#       v0.2.x default was mlx-whisper large-v3-turbo, still available via
#       STT_BACKEND="mlx-whisper".
#   Windows / Linux (v0.6.0+): Qwen3-ASR-0.6B via qwen-asr (PyTorch +
#       transformers, CUDA bfloat16). Same model as macOS for consistent
#       behaviour. faster-whisper large-v3-turbo remains available as a
#       fallback for low-VRAM machines or when the PyTorch CUDA wheel is
#       not installed — set STT_BACKEND="faster-whisper" to use it.
# Intel Mac (darwin x86_64) support was dropped in v0.4.0 — the platform is
# rare enough now that the maintenance + docs cost outweighs the benefit.
# Pin to v0.3.0 or earlier for Intel Mac.
# Override by hardcoding STT_BACKEND / STT_MODEL below.
if sys.platform == "darwin":
    if _host_platform.machine() != "arm64":
        raise SystemExit(
            "home-stt no longer supports Intel Mac (darwin "
            f"{_host_platform.machine()}) since v0.4.0. macOS support is "
            "Apple Silicon (arm64) only. Pin to v0.3.0 or earlier if you "
            "need Intel Mac."
        )
    _DEFAULT_BACKEND = "qwen3-asr"
    _DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
    # MLX 4-bit quantised variant — ~2.5 GB disk / ~4 GB RSS on Apple Silicon.
    _DEFAULT_POLISH_MODEL = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit"
else:
    _DEFAULT_BACKEND = "qwen3-asr"
    _DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
    # Default polish model for Win/Linux: Qwen2.5-1.5B-Instruct (~3 GB VRAM).
    # Picked empirically over Qwen3-4B-Instruct-2507 (~8 GB) on bf16 because
    # the 4B has two real problems on this task:
    #   (a) decode-bound — long inputs (~100-char Chinese) take ~5-6 s polish
    #       on RTX 5080 (memory-bandwidth limited at 8 GB of weights);
    #   (b) over-eager edits — the 2507 SFT recipe translates EN keywords
    #       (commit→提交, push→推送) and substitutes plausible Chinese for
    #       words it doesn't recognise (ASR mistake 拷滅 → 拷貝), despite
    #       explicit prompt rules.
    # Qwen2.5-1.5B attacks both: ~2.5× faster decode + older more conservative
    # SFT recipe + smaller capacity (less ingrained EN↔zh association). Pure
    # instruct, no hybrid-thinking — sidesteps the <think>-tag-leak risk of
    # the entire Qwen3-Instruct family. Alternatives if quality regresses
    # for your usage (override POLISH_MODEL below):
    #   - "Qwen/Qwen3-4B-Instruct-2507" — original v0.6.0 plan default;
    #     stronger but suffers (a) + (b) above.
    #   - "Qwen/Qwen3-1.7B" with enable_thinking=False — better than 1.5B on
    #     C-Eval/IFEval, thinking-tag-leak risk inherited.
    #   - "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4" — ~1 GB VRAM (needs
    #     auto-gptq). Day-3 production target once bf16 quality validated.
    _DEFAULT_POLISH_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

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

# ---------------------------------------------------------------------------
# Optional polish stage (v0.5.0+): runs ASR output through a small local
# instruction-tuned LLM that removes filler words (呃、嗯、就是、那個、然後),
# fixes immediate repetitions (「我我我覺得」→「我覺得」), and otherwise
# preserves the speaker's meaning. The polished text is what gets pasted.
#
# Defaults to Qwen3-4B-Instruct-2507-MLX-4bit (~2.5 GB on disk, ~3-4 GB
# RSS once loaded). The 2507 build is a pure instruction-tuned variant
# (no chain-of-thought trace) — newer Qwen3.5 thinking models are not a
# good fit for this single-step polish task. POLISH_LANGUAGES gates which
# detected-language transcripts get polished, because small Chinese-strong
# instruction LLMs eagerly translate pure-English text into Chinese even
# with an explicit "preserve English" instruction.
#
# Failure modes (mlx-lm missing, model load OOM) degrade silently to a
# NoopPolisher — the daemon continues to work with raw ASR output.
# ---------------------------------------------------------------------------
POLISH_ENABLED   = True
POLISH_MODEL     = _DEFAULT_POLISH_MODEL
POLISH_LANGUAGES = {"zh", "ja", "ko"}
# Polish prompt — lean version. The bf16 4B Qwen3-Instruct over-edits when
# given loose instructions (translates English keywords, substitutes
# "looks-similar" words, restructures sentences). Earlier iteration loaded
# the prompt with detailed rules + 3 few-shot examples (~600 chars) which
# fixed correctness but tripled prefill cost on every polish call. This
# lean form keeps the essential bans + one example. Trade-off accepted:
# polish may occasionally over-edit on edge cases, but per-call prefill
# is much cheaper (~150 chars → ~100 tokens vs 600 → 400 tokens).
POLISH_PROMPT    = (
    "把口語逐字稿做最小修飾。\n"
    "只移除贅字(呃、嗯、就是、那個、然後、嘛、啊)、修立即重複(我我我→我)、補必要標點。\n"
    "嚴禁:翻譯英文(commit/push/function 等保留)、改動詞、替換陌生詞(看似錯字也照樣輸出)、加新詞、改句式。\n"
    "中文一律繁體。只輸出修飾後文字,不解釋、不加引號、不加前綴。\n"
    "\n"
    "範例:\n"
    "輸入:呃我覺得這個 Python function 可以再優化\n"
    "輸出:我覺得這個 Python function 可以再優化"
)

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
#
# `s2twp` (not plain `s2tw`) does character-level Simplified→Traditional
# conversion PLUS phrase-level mapping to Taiwan vocabulary:
#   软件→軟體 (not 軟件), 视频→影片 (not 視頻), 异步→非同步 (not 異步),
#   函数→函式 (not 函數), 代码→程式碼 (not 代碼). For a TW-target daemon
# this is strictly better than s2tw. Cost difference is ~0.1ms (microsecond
# range either way).
#
# post_process() is called TWICE per transcribe:
#   1. on raw ASR output (Qwen3-ASR-0.6B outputs simplified natively — so
#      polish always sees clean traditional + TW-vocab input, less risk of
#      the small polish model being primed into a simplified output register)
#   2. on polish output (deterministic backstop — polish models occasionally
#      leak simplified glyphs back in despite the prompt rule; OpenCC is
#      cheaper and more reliable than fighting that with prompt engineering)
# Both calls are idempotent: re-running on already-traditional+spaced text
# is a no-op.
# ---------------------------------------------------------------------------
_s2twp = OpenCC("s2twp")
_CJK   = r"[㐀-鿿]"
_AW    = r"[A-Za-z0-9]"


def post_process(text: str) -> str:
    """Simplified → Taiwan-traditional (with TW phrase mapping via s2twp),
    then add spaces at CJK ↔ ASCII edges. Idempotent."""
    text = _s2twp.convert(text)
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
        text (no s2twp, no spacing) — that's done downstream."""

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
    qualified HuggingFace repo id. Apple Silicon only — other platforms
    should use ``faster-whisper``.
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
    """Qwen3-ASR — Apple Silicon goes via mlx-qwen3-asr (Metal-native);
    Windows / Linux go via qwen-asr (PyTorch + transformers + NVIDIA CUDA).

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

    # Map upstream's human-readable language names back to the ISO-ish short
    # codes used elsewhere in the daemon log ("zh", "en", ...). Both the
    # MLX and Torch impls go through this normaliser via the outer class.
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
        self._model_name = self._resolve_model_name(model_name)
        if sys.platform == "darwin" and _host_platform.machine() == "arm64":
            self._impl = _Qwen3MlxImpl(self._model_name)
        else:
            self._impl = _Qwen3TorchImpl(self._model_name)

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
        return self._impl.device_label

    def transcribe(self, samples: np.ndarray) -> tuple[str, str]:
        result = self._impl.transcribe(samples)
        text = (result.get("text") or "").strip()
        raw_lang = (result.get("language") or "").strip().lower()
        # Normalise "Chinese" → "zh", "English" → "en", etc. Falls back to
        # the first two letters of whatever the model returned so unknown
        # languages still produce something sensible in the log line.
        language = self._LANG_NORM.get(raw_lang, raw_lang[:2] if raw_lang else "")
        return text, language

    def warmup(self) -> None:
        self._impl.warmup()


class _Qwen3MlxImpl:
    """Apple Silicon path — mlx-qwen3-asr (Metal native)."""

    def __init__(self, model_name: str):
        import mlx_qwen3_asr  # lazy import (Apple Silicon only)

        self._mqa = mlx_qwen3_asr
        self._model_name = model_name
        self.device_label = "Apple Silicon (Metal, MLX) — Qwen3-ASR"

    def transcribe(self, samples: np.ndarray) -> dict:
        result = self._mqa.transcribe(
            samples,
            model=self._model_name,
            verbose=False,
        )
        return {
            "text": getattr(result, "text", "") or "",
            "language": getattr(result, "language", "") or "",
        }

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


class _Qwen3TorchImpl:
    """Windows / Linux path — qwen-asr (PyTorch + transformers + NVIDIA CUDA).

    Hard-requires CUDA. On CPU the Qwen3-ASR-0.6B is ~30-60× slower per
    transcribe — way over the hold-to-talk perceptual budget. Mirroring
    TorchLocalLlmPolisher's pattern, __init__ raises RuntimeError if CUDA
    is unavailable so the daemon's build_backend_with_fallback can route
    to faster-whisper (which has its own working CPU int8 fallback inside
    its __init__). Refusing CPU here is much better UX than silently
    delivering 30-60 s transcribes.
    """

    def __init__(self, model_name: str):
        # Heavy lazy imports — only paid on Win/Linux + qwen3-asr backend.
        import torch
        from qwen_asr import Qwen3ASRModel

        if not torch.cuda.is_available():
            raise RuntimeError(
                "_Qwen3TorchImpl requires CUDA. On CPU the Qwen3-ASR "
                "model is 30-60x slower per transcribe — unusable for "
                "hold-to-talk UX. Install torch with CUDA support, or "
                "set STT_BACKEND = 'faster-whisper' which has a working "
                "CPU int8 fallback."
            )

        device = "cuda:0"
        dtype = torch.bfloat16
        try:
            # torch returns e.g. "NVIDIA GeForce RTX 5080" — already has
            # vendor prefix; don't add a second "NVIDIA ".
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA device"
        self.device_label = f"{gpu_name} (bfloat16) — Qwen3-ASR"

        self._model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )
        self._model_name = model_name

    def transcribe(self, samples: np.ndarray) -> dict:
        results = self._model.transcribe(
            audio=(samples, SAMPLE_RATE),
            language=None,
        )
        if not results:
            return {"text": "", "language": ""}
        r = results[0]
        return {
            "text": getattr(r, "text", "") or "",
            "language": getattr(r, "language", "") or "",
        }

    def warmup(self) -> None:
        warm_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self._model.transcribe(
            audio=(warm_audio, SAMPLE_RATE),
            language=None,
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


def build_backend_with_fallback() -> STTBackend:
    """Try the configured STT backend; on ImportError (missing package) or
    CUDA OOM, fall back to faster-whisper with a loud actionable stderr
    message. If that also fails, exit cleanly — the daemon can't run
    without an STT backend, and a half-initialised state is worse than a
    clear-cut exit. Mirrors build_polisher's degrade-gracefully pattern."""
    try:
        return build_backend(STT_BACKEND, STT_MODEL)
    except (ImportError, ModuleNotFoundError) as e:
        print(
            f"[stt] backend '{STT_BACKEND}' missing required package: "
            f"{e}. Falling back to faster-whisper. To enable "
            f"{STT_BACKEND}, see README -> Windows 安裝步驟 (install "
            f"torch+CUDA wheel before `pip install qwen-asr`).",
            file=sys.stderr, flush=True,
        )
    except Exception as e:
        # Three actionable failure classes (mirrors build_polisher):
        #   - CUDA OOM: hint to free polish VRAM or pick smaller STT model
        #   - DLL load failure: hint to install NVIDIA cuDNN/cuBLAS wheels
        #   - Other: surface raw exception
        msg = str(e)
        # isinstance check beats string-typed class name compare — future
        # torch renames/wraps won't silently break OOM detection.
        try:
            import torch as _torch
            oom_cls = getattr(_torch.cuda, "OutOfMemoryError", None)
        except Exception:
            oom_cls = None
        is_oom = (
            (oom_cls is not None and isinstance(e, oom_cls))
            or "out of memory" in msg.lower()
        )
        is_dll = isinstance(e, OSError) and any(
            s in msg.lower() for s in ("dll", "cudart", "cudnn", "cublas")
        )
        if is_oom:
            hint = (
                " (CUDA OOM — try POLISH_ENABLED = False to free ~3-8 GB, "
                "or pick a smaller STT_MODEL)"
            )
        elif is_dll:
            hint = (
                " (CUDA DLL load failed — install nvidia-cudnn-cu12 + "
                "nvidia-cublas-cu12 or reinstall torch with CUDA wheel)"
            )
        else:
            hint = ""
        print(
            f"[stt] backend '{STT_BACKEND}' init failed: {e}{hint}. "
            f"Falling back to faster-whisper.",
            file=sys.stderr, flush=True,
        )

    try:
        return build_backend("faster-whisper", "large-v3-turbo")
    except Exception as e:
        print(
            f"[stt] fatal: faster-whisper fallback also failed: {e}. "
            f"Daemon cannot continue — install dependencies per README "
            f"-> Windows 安裝步驟 and restart.",
            file=sys.stderr, flush=True,
        )
        raise SystemExit(1)


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
_polisher: TextPostProcessor  # set in main()


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
        pre_polish = text  # captured to log diff when polish edits substantively

        # Optional polish stage (text → text). Gated on detected language
        # because small Chinese-strong instruction LLMs translate pure-
        # English text even with an explicit "preserve English" prompt.
        # POLISH_LANGUAGES whitelists which language codes trigger polish.
        #
        # `polish_edited` is computed on the polish output BEFORE the
        # OpenCC backstop — otherwise edits that the backstop normalises
        # away (e.g. polish leaked 简, s2twp converted back) would be
        # invisible in the diff log even though polish DID change text.
        polish_elapsed = 0.0
        polish_edited = False
        if language in POLISH_LANGUAGES:
            t1 = time.time()
            polished = _polisher.polish(text)
            polish_elapsed = time.time() - t1
            polish_edited = polished != text
            # Polish models can leak simplified glyphs back in despite the
            # "中文一律繁體" prompt rule (especially smaller models with
            # weaker instruction following). Re-run post_process as a
            # deterministic μs-cost backstop — guarantees clipboard output
            # is always TW-traditional with consistent CJK/ASCII spacing,
            # regardless of which polish model is loaded or how strict it is.
            text = post_process(polished)

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
            timing = f"{elapsed:.2f}s"
            # 5ms threshold suppresses NoopPolisher's microsecond runtime
            # — without it the log shows "+polish 0.00s" even when polish
            # was disabled / fell back to Noop, falsely suggesting polish
            # ran.
            if polish_elapsed > 0.005:
                timing += f"+polish {polish_elapsed:.2f}s"
            # Print the raw (pre-polish, post-OpenCC) text on a preceding
            # line when polish substantively edited it — lets the user diff
            # what polish changed vs what ASR produced. Gated on
            # polish_edited (computed before the OpenCC backstop) so edits
            # that the backstop normalises away are still surfaced.
            if polish_edited:
                print(f"[stt] {language} raw   -> {pre_polish}", flush=True)
            if paste_ok:
                print(f"[stt] {language} {timing} -> {text}", flush=True)
            else:
                # paste() already printed a user-facing 'paste blocked' line
                # to the main log; we record the transcript itself so it's
                # discoverable even when auto-paste didn't fire.
                print(f"[stt] {language} {timing} clipboard-only -> {text}",
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

    global _polisher
    _polisher = build_polisher(POLISH_ENABLED, POLISH_MODEL, POLISH_PROMPT)
    print(f"[stt] polish: {_polisher.device_label}", flush=True)

    _backend = build_backend_with_fallback()

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
