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

import logging
import os
import platform as _host_platform
import queue
import sys
import threading
import time
from typing import Any

logger = logging.getLogger("stt")

# v0.7.1: PyTorch CUDA allocator hint — reduces fragmentation from the
# polish KV cache + tokenizer scratch allocations that happen on every
# polish call. Must be set BEFORE torch loads via build_polisher /
# build_backend (both lazy-import torch). `setdefault` so explicit
# environment override still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
from pynput import keyboard
from pynput.keyboard import Key

from stt_audio import post_process, _play_beep, _trim_silence
from stt_backends import STTBackend, build_backend, build_backend_with_fallback
from stt_platform import Pasteboard, build_pasteboard
from stt_state import write_state as _write_state, cleanup as _cleanup_state, IDLE, RECORDING, PROCESSING
from text_polisher import TextPostProcessor, build_polisher


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
__version__ = "0.7.5"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_RATE      = 16000
# v0.7.2: lowered from 0.3 → 0.15. Mandarin one-syllable replies
# (「好」「對」「是」, ~0.25 s) were being silent-rejected by the 0.3 s
# threshold. 0.15 s sits above typical key-bounce (~10-20 ms) and the
# minimum deliberate human press (~80 ms) while preserving short
# affirmatives.
MIN_AUDIO_SEC    = 0.15                # taps shorter than this are ignored
# v0.7.2: cap on a single recording. Stuck triggers (key-repeat anomaly,
# RDP disconnect mid-hold, kernel hang) would otherwise grow _buffer
# unbounded. 120 s is well past any reasonable hold-to-talk session;
# beyond this, _audio_callback force-releases and spawns transcribe.
MAX_AUDIO_SEC    = 120                 # hard ceiling — auto-release on stuck key

# Microphone device — None = system default. Set via config.toml or
# HOME_STT_MIC_DEVICE env var. Accepts device name (str) or index (int).
MIC_DEVICE: str | int | None = None

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
    # Default polish model for Win/Linux: Qwen3-4B-Instruct-2507 (~8 GB
    # VRAM bf16). Aligned with macOS default (same model in MLX 4-bit)
    # so polish output is consistent cross-platform.
    #
    # v0.6.0 shipped Qwen2.5-1.5B-Instruct as an intermediate diagnostic
    # default while characterising latency vs quality trade-offs. v0.7.0
    # reverts to Qwen3-4B-Instruct-2507 after a structured 18-case bench
    # against Qwen2.5-{0.5B, 1.5B}-Instruct + Qwen3-4B-Instruct-2507
    # found that Qwen3-4B:
    #   - Faithfully preserves English keywords (Qwen2.5-1.5B silently
    #     swapped `commit` → `push` — a wrong-verb production bug)
    #   - Does NOT mutate facts (Qwen2.5-1.5B flipped "INT4 反而更慢"
    #     to "INT4 反而更快" — semantic-reversal hallucination)
    #   - Preserves identifiers (Qwen2.5-0.5B dropped the `_` prefix
    #     from `_USE_TORCH_COMPILE`)
    #   - Preserves subjects (Qwen2.5-0.5B swapped 「幫我」→「幫你」)
    # Cost: ~50% slower per-call (long polish ~3.6 s vs ~2.3 s on RTX 5080),
    # ~5 GB more VRAM. Acceptable trade for these quality wins on
    # NVIDIA ≥ 12 GB cards. v0.7.0 release notes carry the full
    # investigation; bnb-INT4 + torch.compile + flash-attn were also
    # measured (all neutral-to-negative on Windows for this workload —
    # see TorchLocalLlmPolisher class-level toggles in text_polisher.py).
    #
    # Alternatives if VRAM-constrained or latency-sensitive (override
    # POLISH_MODEL below):
    #   - "Qwen/Qwen2.5-1.5B-Instruct" — ~3 GB VRAM, faster but
    #     instruction-following weaker (see Balanced preset in README)
    #   - "Qwen/Qwen2.5-0.5B-Instruct" — ~1 GB VRAM, fastest but
    #     identifier/subject errors observed
    #   - POLISH_ENABLED = False — disable entirely
    _DEFAULT_POLISH_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

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
# Default polish models per platform (v0.7.0+ unified on Qwen3-4B-Instruct-2507):
#   macOS Apple Silicon: lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit
#                        (~2.5 GB disk, ~3-4 GB RSS via MLX/Metal)
#   Windows / Linux:     Qwen/Qwen3-4B-Instruct-2507
#                        (~8 GB disk, ~8 GB VRAM bf16 via PyTorch CUDA)
# The 2507 build is a pure instruction-tuned variant (no chain-of-thought
# trace) — Qwen3.5 thinking models are not a good fit for this single-step
# polish task. POLISH_LANGUAGES gates which detected-language transcripts
# get polished, because small Chinese-strong instruction LLMs eagerly
# translate pure-English text into Chinese even with an explicit "preserve
# English" instruction.
#
# Failure modes (mlx-lm missing, model load OOM) degrade silently to a
# NoopPolisher — the daemon continues to work with raw ASR output.
# ---------------------------------------------------------------------------
POLISH_ENABLED   = True
POLISH_MODEL     = _DEFAULT_POLISH_MODEL
# v0.7.2: narrowed from {zh, ja, ko} to {zh} only. POLISH_PROMPT is
# written entirely in Chinese and only anchors Chinese behaviour
# (「中文一律繁體」「禁翻譯英文」). A ja or ko transcript through the
# same prompt has zero rule-level constraint — the 4B model is free to
# rewrite, translate, or hallucinate without violating any instruction
# it can parse. Restrict to zh until a per-language prompt path lands.
# To re-enable ja/ko: write a per-language POLISH_PROMPT dispatch and
# expand this set in lockstep.
POLISH_LANGUAGES = {"zh"}
# Polish prompt — lean version. The bf16 4B Qwen3-Instruct over-edits when
# given loose instructions (translates English keywords, substitutes
# "looks-similar" words, restructures sentences). Earlier iteration loaded
# the prompt with detailed rules + 3 few-shot examples (~600 chars) which
# fixed correctness but tripled prefill cost on every polish call. This
# lean form keeps the essential bans + two examples. Trade-off accepted:
# polish may occasionally over-edit on edge cases, but per-call prefill
# is much cheaper (~210 chars → ~140 tokens vs 600 → 400 tokens).
#
# v0.7.4: addressed asymmetric punctuation rule (補標點 but no 禁刪標點)
# that let Qwen3-4B drop句末「。」between sentences. Symptom from live
# stt-daemon.log on 2026-05-24: ASR raw "...小問題。我發現..." → polished
# "...小問題 我發現..." (period replaced with space, two sentences merged).
# Three reinforcing changes — single-axis fixes proved insufficient in
# bench (negative constraint alone left 4/5 punct cases still failing):
#   (a) Front-loaded positive constraint "原有標點(。？！，)完整保留"
#       on line 1 where the model pays most attention,
#   (b) Negative constraint "刪除或替換原有標點" in 嚴禁 line,
#   (c) Second few-shot example showing period-preservation behavior on
#       multi-sentence input — strongest signal for instruction-tuned
#       models per the prior in-context-learning literature.
# Regression-guarded by 5 punctuation_preservation cases in
# tests/fixtures/polish_cases.json. Triggered by user noticing「過去有
# 標點符號,現在沒有」during dictation.
POLISH_PROMPT    = (
    "把口語逐字稿做最小修飾。原有標點(。？！，)完整保留。\n"
    "只移除贅字(呃、嗯、就是、那個、然後、嘛、啊)、修立即重複(我我我→我)、補必要標點。\n"
    "嚴禁:翻譯英文(commit/push/function 等保留)、改動詞、替換陌生詞(看似錯字也照樣輸出)、加新詞、改句式、刪除或替換原有標點。\n"
    "中文一律繁體。只輸出修飾後文字,不解釋、不加引號、不加前綴。\n"
    "\n"
    "範例 1:\n"
    "輸入:呃我覺得這個 Python function 可以再優化\n"
    "輸出:我覺得這個 Python function 可以再優化\n"
    "\n"
    "範例 2:\n"
    "輸入:我剛剛測試了一下。發現一個問題。\n"
    "輸出:我剛剛測試了一下。發現一個問題。"
)

# Set of pynput Key/character triggers to listen for as hold-to-record keys.
# `None` means "use the platform default" (Windows: Right Alt + Right Ctrl;
# macOS: Right Option). Override with e.g. `{Key.f13}` to lock to one key.
TRIGGER_KEYS: set | None = None

# ---------------------------------------------------------------------------
# v0.7.5 voice-edit mode (⌥+E hotkey, clipboard round-trip selection capture)
#
# Hold an edit trigger key → daemon captures the current text selection
# via clipboard round-trip → user speaks an instruction → polish LLM
# applies the instruction to the selection (different prompt: EDIT_PROMPT
# in text_polisher.py) → result replaces the selection via paste →
# original clipboard restored.
#
# `None` here means "use platform default" — Win: {Key.f13}, Mac:
# {Key.cmd_r} (Right Command, symmetric to Right Option dictate trigger;
# defaults defined per-platform in stt_platform_{win,mac}.py to mirror
# the existing default_trigger_keys pattern). Set explicitly to an
# empty set `set()` to DISABLE voice-edit entirely.
#
# Caveat: F13 only exists on full-size Win keyboards — TKL / laptop users
# override via config.toml or `home-stt config --set-trigger`.
# Right Command exists on all Mac keyboards including MacBook.
EDIT_TRIGGER_KEYS: set | None = None
SELECTION_CAPTURE_WAIT_S  = 0.1   # post-Cmd+C wait before checking seqno

# ---------------------------------------------------------------------------
# Press-time encoder pipelining framework (built for v0.8.0, shipped
# DISABLED in v0.7.3 after a bench-first save).
#
# Goal was 50% release-to-text latency reduction by running the ASR
# encoder in a background thread while the user holds the trigger, so
# only the decoder + tail encoder runs on the post-release critical path.
# Spike (`tmp/spike_torch_encoder.py`) verified chunked encoding produces
# text within Lev≤4 of batch on real speech; Day 13-14 latency bench
# (`tmp/bench_v080_latency.py`) measured the actual saving:
#
#     sample              audio   batch   stream   saved   pct  Lev
#     sample.wav          20.0s   2.83s   2.88s   -0.06s   -2%    2
#     sample_english      20.0s   2.82s   2.85s   -0.03s   -1%    0
#     sample_long         40.0s   6.99s   6.81s   +0.18s   +3%   21
#     sample_silence-mid  30.0s   4.54s   4.47s   +0.07s   +2%   21
#
# Root cause: original plan estimated encoder forward at 3-5s for 40s
# audio on RTX 5080 + Qwen3-ASR-0.6B; reality is ~0.2s. Decoder dominates
# ~95% of post-release time. Pipelining the encoder saves ≤0.2s — not
# worth the Lev=21 long-form text drift it introduces (chunk-boundary
# noise: punctuation + homophone choice + occasional word substitution;
# semantic content preserved but char-level differs by ~10%).
#
# Shipped DISABLED so the daemon's runtime behaviour matches v0.7.2.
# Framework code (StreamingQwen3ASRModel, _encoder_worker, all state +
# tests) preserved for future re-evaluation when the decoder side gets
# faster (Qwen3-ASR-FP8 if Alibaba ships it, smaller decoder variant,
# speculative decoding with a viable draft model, or llama.cpp+GGUF Q8_0
# swap — the original v0.8.0 plan candidate B that's the real path to
# 2-3x decoder speedup).
#
# To re-enable for testing / future work:
#   ENCODER_PIPELINING = True
ENCODER_PIPELINING            = False  # see comment above — null-result ship
ENCODER_CHUNK_SEC             = 5.0    # encoder forward every N s of buffered audio
ENCODER_QUEUE_MAX             = 200    # ~10s of 50ms ticks; lag safety cap
ENCODER_FINALIZE_TIMEOUT      = 8.0    # max join wait before fallback to batch
ENCODER_FAILURE_BUDGET        = 3      # consecutive failures before disabling next utterance
ENCODER_SILENCE_FALLBACK_SEC  = 2.0    # mid-utterance silence ≥ N s → batch fallback

# Audio feedback — short sine-wave tones at trigger-press / paste-done
# so the user knows when recording starts and when transcription has
# landed. Cross-platform: relies only on sounddevice (already a dep).
BEEPS_ENABLED    = True
BEEP_START_HZ    = 880                 # A5, "bright" — start of recording
BEEP_END_HZ      = 660                 # E5, "calmer" — paste done
BEEP_FAIL_HZ     = 220                 # A3, "dull" — v0.7.5 voice-edit
                                       #   abort (no selection / polish.edit
                                       #   failure). Distinct from press +
                                       #   end so user can audibly tell
                                       #   things went wrong.
BEEP_DURATION_MS = 80
BEEP_VOLUME      = 0.15                # 0.0–1.0; keep low to avoid mic bleed


# ---------------------------------------------------------------------------
# Audio capture state
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
_buffer: list[np.ndarray] = []
_recording = False
_active_trigger = None   # which TRIGGER_KEYS member is currently held, or None
_processing = False
# v0.7.2: running sample count for the current _buffer. Tracked to enforce
# MAX_AUDIO_SEC without re-summing every callback (which would be O(N) per
# 50 ms tick on long recordings). Reset whenever _buffer is cleared / swapped.
_recording_samples = 0

# v0.8.0: press-time encoder pipelining state. All access guarded by
# _state_lock (the bool flags + counters), except _encoder_queue +
# _encoder_stop_event which are already thread-safe primitives. See plan
# in ~/.claude/plans/v0-8-0-architectural-polymorphic-ullman.md for the
# end-to-end lifecycle. Reset together in _on_press; never set outside
# _on_press / _audio_callback / _encoder_worker / _transcribe_and_emit.
_encoder_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=ENCODER_QUEUE_MAX)
_encoder_thread: threading.Thread | None = None
_encoder_handle: Any = None
_encoder_stop_event = threading.Event()
_encoder_active = False            # True iff worker is alive and handle valid
_encoder_failed = False            # set by worker on exception, or by finalize-timeout
_encoder_consecutive_failures = 0  # ENCODER_FAILURE_BUDGET → suppress next utterance
_encoder_use_batch_fallback = False  # Option C: mid-silence detected → use batch path
_encoder_silence_run_samples = 0   # rolling silence sample count for Option C
_encoder_residual_samples: np.ndarray | None = None  # tail audio worker left for finalize

# v0.7.5 voice-edit per-recording state. Snapshotted at press time —
# `_on_release` reads these under _state_lock then routes to
# `_transcribe_and_emit_edit` instead of `_transcribe_and_emit`. All three
# are cleared at the top of `_on_press` and re-populated if the press is
# an edit-trigger.
_edit_mode = False                  # True iff this recording is voice-edit
_edit_selection: str | None = None  # the captured selection text
_edit_original_clipboard: str | None = None  # to restore in finally

# Silence threshold for the mid-utterance fallback detector. Matches the
# -50 dBFS that _trim_silence uses, so user-visible behaviour is consistent
# ("daemon thought you stopped speaking" → fallback).
_ENCODER_SILENCE_THRESHOLD = 10.0 ** (-50.0 / 20.0)




def _encoder_worker(handle: Any) -> None:
    """v0.8.0: drain `_encoder_queue` into ENCODER_CHUNK_SEC slabs and feed
    them through `_backend.push_chunk(handle, slab)` while the user is
    still holding the trigger. Sets `_encoder_residual_samples` to the
    remaining tail (< ENCODER_CHUNK_SEC) for finalize to encode.

    Failure modes:
      - Any exception in `push_chunk` → log + set `_encoder_failed = True`
        + increment `_encoder_consecutive_failures`. `_transcribe_and_emit`
        detects the flag and transparently falls back to the batch path.
      - `_encoder_queue.Empty` after stop_event is the normal exit path.

    Aborts cleanly (never via interrupt — PyTorch/MLX forwards are
    uninterruptible). Worst case: a 5 s slab is in flight when the user
    releases, the worker finishes that forward (≤ ~1 s on RTX 5080), THEN
    checks `_encoder_stop_event` and exits. `_transcribe_and_emit`'s
    8 s join timeout bounds the wait either way.
    """
    global _encoder_failed, _encoder_consecutive_failures, _encoder_residual_samples
    chunk_size = int(SAMPLE_RATE * ENCODER_CHUNK_SEC)
    accumulator: list[np.ndarray] = []
    accumulated_n = 0
    try:
        while not _encoder_stop_event.is_set():
            try:
                # Short timeout so we re-check stop_event regularly even
                # during long silence (when no new chunks arrive).
                chunk = _encoder_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            accumulator.append(chunk)
            accumulated_n += chunk.shape[0]
            if accumulated_n >= chunk_size:
                concat = np.concatenate(accumulator, axis=0).flatten().astype(np.float32)
                slab = concat[:chunk_size]
                _backend.push_chunk(handle, slab)
                # Keep remainder for next slab.
                if concat.shape[0] > chunk_size:
                    accumulator = [concat[chunk_size:]]
                    accumulated_n = concat.shape[0] - chunk_size
                else:
                    accumulator = []
                    accumulated_n = 0
        # Stop event set — drain any remaining queue items into accumulator
        # so finalize sees the full tail. Don't push_chunk: finalize is the
        # designated point for the final (possibly partial) slab.
        while True:
            try:
                chunk = _encoder_queue.get_nowait()
            except queue.Empty:
                break
            accumulator.append(chunk)
            accumulated_n += chunk.shape[0]
        if accumulator:
            residual = np.concatenate(accumulator, axis=0).flatten().astype(np.float32)
        else:
            residual = np.zeros(0, dtype=np.float32)
        with _state_lock:
            _encoder_residual_samples = residual
    except Exception as e:
        # Don't print full traceback — log a one-liner and let the
        # batch-fallback path produce the user-visible transcript.
        logger.warning(f"encoder worker crashed: {type(e).__name__}: {e}")
        with _state_lock:
            _encoder_failed = True
            _encoder_consecutive_failures += 1


def _audio_callback(indata, frames, time_info, status) -> None:
    """PortAudio callback fired every ~50 ms with a fresh chunk of float32
    samples. Three responsibilities (in order, all under lock or
    thread-safe primitives):

      1. Append to `_buffer` (v0.7.2 — full-audio fallback path).
      2. Enforce MAX_AUDIO_SEC stuck-key cap (v0.7.2).
      3. v0.8.0: dual-write the chunk to `_encoder_queue` for the
         press-time encoder worker; lazy-spawn the worker on the first
         chunk if the backend supports streaming AND we haven't burned
         the failure budget; track silence runs for Option C fallback.

    Spawning a Python thread from PortAudio's audio thread is safe (it's
    not realtime-blocking).
    """
    global _recording, _recording_samples
    global _encoder_thread, _encoder_handle, _encoder_active
    global _encoder_silence_run_samples, _encoder_use_batch_fallback
    global _encoder_failed, _encoder_consecutive_failures
    if status:
        logger.info("audio status: %s", status)

    # Per-callback chunk stats. RMS uses float64 to avoid bf16/fp32 round
    # cancellation on very-low-amplitude room tone.
    chunk = indata.copy()
    chunk_rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
    is_silent = chunk_rms < _ENCODER_SILENCE_THRESHOLD

    auto_stop = False
    spawn_encoder = False
    push_to_encoder = False
    with _state_lock:
        if not _recording:
            return
        _buffer.append(chunk)
        _recording_samples += chunk.shape[0]
        if _recording_samples >= SAMPLE_RATE * MAX_AUDIO_SEC:
            _recording = False
            auto_stop = True

        # Option C: silence-run tracking. Set fallback flag once we cross
        # the threshold; once set, stays set for this recording (one
        # detection is enough — chunked encoder will degrade gracefully).
        if is_silent:
            _encoder_silence_run_samples += chunk.shape[0]
            if (_encoder_silence_run_samples >=
                    SAMPLE_RATE * ENCODER_SILENCE_FALLBACK_SEC
                    and not _encoder_use_batch_fallback):
                _encoder_use_batch_fallback = True
                # No need to actively stop the encoder worker — let it
                # keep encoding; finalize will route to batch instead.
        else:
            _encoder_silence_run_samples = 0

        # v0.8.0: encoder lazy-spawn decision. Only on the FIRST chunk of
        # a recording, only if (a) pipelining enabled, (b) backend opts in,
        # (c) we haven't burned the failure budget for this session,
        # (d) silence-fallback hasn't already been triggered for this
        # recording (e.g. user held key 3 s before speaking).
        if (not _encoder_active
                and ENCODER_PIPELINING
                and _backend is not None
                and _backend.supports_streaming()
                and _encoder_consecutive_failures < ENCODER_FAILURE_BUDGET
                and not _encoder_use_batch_fallback):
            spawn_encoder = True
        elif _encoder_active:
            push_to_encoder = True

    if spawn_encoder:
        try:
            _encoder_handle = _backend.start_encoder()
            _encoder_stop_event.clear()
            _encoder_thread = threading.Thread(
                target=_encoder_worker, args=(_encoder_handle,), daemon=True,
            )
            _encoder_thread.start()
            with _state_lock:
                _encoder_active = True
            # Push the first chunk to the queue too — worker is now ready.
            push_to_encoder = True
        except Exception as e:
            logger.warning(f"encoder spawn failed: {type(e).__name__}: {e}; "
                           f"will use batch path")
            with _state_lock:
                _encoder_failed = True
                _encoder_consecutive_failures += 1

    if push_to_encoder:
        try:
            _encoder_queue.put_nowait(chunk)
        except queue.Full:
            # Lag: worker can't keep up. _buffer still has the full audio,
            # so finalize / batch fallback will still produce a transcript.
            # Set the fallback flag so finalize doesn't trust accumulated
            # partial encoder state.
            with _state_lock:
                _encoder_use_batch_fallback = True

    if auto_stop:
        logger.warning(f"auto-stop at {MAX_AUDIO_SEC}s — released stuck trigger")
        threading.Thread(target=_transcribe_and_emit, daemon=True).start()


# ---------------------------------------------------------------------------
# Transcription pipeline (backend-agnostic)
# ---------------------------------------------------------------------------
_backend: STTBackend | None = None         # set in main()
_pasteboard: Pasteboard | None = None      # set in main()
_polisher: TextPostProcessor | None = None  # set in main()


def _abort_encoder_quiet() -> None:
    """Helper: signal stop, briefly join the worker, abort the backend
    handle. Idempotent and exception-safe — used by every early-return /
    fallback path in _transcribe_and_emit to release GPU buffers cleanly.
    Caller must hold the _processing semaphore or know the encoder is
    not concurrently being read elsewhere."""
    global _encoder_active
    if not _encoder_active:
        return
    _encoder_stop_event.set()
    if _encoder_thread is not None and _encoder_thread.is_alive():
        _encoder_thread.join(timeout=2.0)
    if _encoder_handle is not None and _backend is not None:
        try:
            _backend.abort(_encoder_handle)
        except Exception as e:
            logger.warning(f"encoder abort raised (ignored): "
                           f"{type(e).__name__}: {e}")
    with _state_lock:
        _encoder_active = False


def _transcribe_and_emit() -> None:
    global _processing, _buffer, _recording_samples
    global _encoder_active, _encoder_failed, _encoder_consecutive_failures
    with _state_lock:
        if _processing:
            # v0.7.2: previously this was a silent early-return that left
            # the captured audio in _buffer — where the NEXT successful
            # transcribe would scoop it up and merge it with the next
            # utterance ("I said A, then B during processing; my next
            # transcript looks like B+C concatenated"). The user got no
            # diagnostic. Now: explicitly drop, log, and clear.
            dropped_chunks = len(_buffer)
            dropped_sec = _recording_samples / SAMPLE_RATE
            _buffer = []
            _recording_samples = 0
            if dropped_chunks > 0:
                logger.info(f"busy — dropped {dropped_sec:.2f}s of captured "
                            f"audio ({dropped_chunks} blocks; previous transcribe "
                            f"still running)")
            return
        _processing = True
        chunks = _buffer
        _buffer = []
        _recording_samples = 0
        # v0.8.0: snapshot encoder state under the same lock acquisition
        # for a coherent routing decision. After this point, the audio
        # callback's writes to these flags only affect the NEXT recording.
        snap_encoder_active = _encoder_active
        snap_encoder_failed = _encoder_failed
        snap_encoder_use_batch = _encoder_use_batch_fallback
    text = None
    language = None
    try:
        if not chunks:
            _abort_encoder_quiet()
            return
        samples = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
        raw_sec = len(samples) / SAMPLE_RATE

        # Decide path: streaming if encoder spawned cleanly, didn't crash,
        # and no silence-fallback was triggered for this recording.
        use_streaming = (
            snap_encoder_active
            and not snap_encoder_failed
            and not snap_encoder_use_batch
        )

        raw: str = ""
        language: str = ""
        elapsed: float = 0.0
        path_label = "batch"  # for the log line

        if use_streaming:
            # Join the worker; if it doesn't exit within
            # ENCODER_FINALIZE_TIMEOUT, abort and fall back.
            t0 = time.time()
            if _encoder_thread is not None:
                _encoder_thread.join(timeout=ENCODER_FINALIZE_TIMEOUT)
            join_elapsed = time.time() - t0
            if _encoder_thread is not None and _encoder_thread.is_alive():
                logger.warning(f"encoder join timed out after "
                               f"{join_elapsed:.1f}s — falling back to batch path")
                _abort_encoder_quiet()
                with _state_lock:
                    _encoder_failed = True
                    _encoder_consecutive_failures += 1
                use_streaming = False
            else:
                # Worker exited; re-read the failure flag in case it
                # crashed mid-flight (worker sets _encoder_failed under
                # lock; callback may also have set it for queue overflow).
                with _state_lock:
                    if _encoder_failed or _encoder_use_batch_fallback:
                        # Worker reported a failure OR callback flagged a
                        # late silence/lag event — abort and use batch.
                        use_streaming = False

        if use_streaming:
            tail = _encoder_residual_samples
            if tail is None:
                tail = np.zeros(0, dtype=np.float32)
            t0 = time.time()
            try:
                raw, language = _backend.finalize(_encoder_handle, tail)
                elapsed = time.time() - t0
                path_label = "stream"
                with _state_lock:
                    _encoder_consecutive_failures = 0  # success → reset
                    _encoder_active = False
            except Exception as e:
                logger.warning(f"encoder finalize raised "
                               f"{type(e).__name__}: {e}; falling back to batch")
                try:
                    _backend.abort(_encoder_handle)
                except Exception:
                    pass
                with _state_lock:
                    _encoder_failed = True
                    _encoder_consecutive_failures += 1
                    _encoder_active = False
                use_streaming = False

        if not use_streaming:
            # Batch fallback — v0.7.2 path. First make sure any encoder
            # worker that might still be alive is released cleanly so we
            # don't leak GPU buffers.
            _abort_encoder_quiet()
            # v0.7.2: RMS silence trim before ASR. Qwen3-ASR is LLM-backbone
            # and hallucinates on long silence (model card known issue).
            trimmed = _trim_silence(samples, SAMPLE_RATE)
            if len(trimmed) < SAMPLE_RATE * MIN_AUDIO_SEC:
                # Could be (a) too-short tap or (b) all-silence after trim.
                trim_sec = len(trimmed) / SAMPLE_RATE
                if trim_sec < raw_sec * 0.5 and raw_sec >= MIN_AUDIO_SEC:
                    logger.info(f"silent — trimmed {raw_sec:.2f}s → "
                                f"{trim_sec:.2f}s; mic muted or very quiet?")
                else:
                    logger.info(f"too short ({raw_sec:.2f}s)")
                return
            t0 = time.time()
            raw, language = _backend.transcribe(trimmed)
            elapsed = time.time() - t0
            # path_label already "batch"

        if not raw:
            logger.info(f"empty ({language}, {elapsed:.2f}s, {path_label})")
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
            logger.error(f"{language} {elapsed:.2f}s clipboard write failed — "
                         f"'{text}' NOT inserted")
            return
        # v0.7.2: was 0.15 s, dropped to 0.02 s. The original sleep was
        # primarily compensating for PowerShell Set-Clipboard / pbcopy
        # subprocess publishing the clipboard contents asynchronously —
        # set_text could return before the OS-side clipboard daemon
        # finished. v0.7.2 ships direct Win32 OpenClipboard / NSPasteboard
        # paths that publish synchronously, so the dominant remaining
        # need is letting the keyboard listener / paste keystroke synth
        # land in a settled focus window. 20 ms is plenty.
        time.sleep(0.02)
        paste_ok = _pasteboard.paste()
        if paste_ok:
            _play_beep(BEEP_END_HZ, BEEP_DURATION_MS, BEEP_VOLUME, BEEPS_ENABLED)

        try:
            timing = f"{elapsed:.2f}s"
            # 5ms threshold suppresses NoopPolisher's microsecond runtime
            # — without it the log shows "+polish 0.00s" even when polish
            # was disabled / fell back to Noop, falsely suggesting polish
            # ran.
            if polish_elapsed > 0.005:
                timing += f"+polish {polish_elapsed:.2f}s"
            # v0.8.0: tag streaming path explicitly so user can verify
            # encoder pipelining actually kicked in vs silent batch
            # fallback. Batch is the v0.7.2-equivalent default — keep
            # logs quiet for it (no tag).
            path_tag = " (stream)" if path_label == "stream" else ""
            # Print the raw (pre-polish, post-OpenCC) text on a preceding
            # line when polish substantively edited it — lets the user diff
            # what polish changed vs what ASR produced. Gated on
            # polish_edited (computed before the OpenCC backstop) so edits
            # that the backstop normalises away are still surfaced.
            if polish_edited:
                logger.info(f"{language} raw   -> {pre_polish}")
            if paste_ok:
                logger.info(f"{language} {timing}{path_tag} -> {text}")
            else:
                # paste() already printed a user-facing 'paste blocked' line
                # to the main log; we record the transcript itself so it's
                # discoverable even when auto-paste didn't fire.
                logger.info(f"{language} {timing}{path_tag} "
                            f"clipboard-only -> {text}")
        except Exception:
            logger.info(f"inserted ({elapsed:.2f}s); log encoding failed")
    except Exception as e:
        logger.error(f"{e}")
    finally:
        with _state_lock:
            _processing = False
        _write_state(IDLE, last_text=text if text else None,
                     last_lang=language if language else None)


# ---------------------------------------------------------------------------
# v0.7.5 voice-edit mode — selection capture + edit-path transcribe
# ---------------------------------------------------------------------------
def _capture_selection(pb) -> tuple[str, str | None] | None:
    """Synchronous clipboard round-trip to capture the focused app's
    current text selection.

    Returns (selection, original_clipboard) on success, or None if no
    selection was captured (seqno unchanged after simulate_copy — the
    focused app either had no selection or doesn't expose one to Cmd+C).
    `original_clipboard` is the pre-capture clipboard text (may be None
    if empty / non-text); the caller is responsible for restoring it via
    pb.set_text in their finally block.

    Timing: ~110 ms sync (SELECTION_CAPTURE_WAIT_S = 100 ms + ~10 ms of
    OS calls). Per plan §G this is hidden behind the user's natural
    "start speaking" gesture latency, so the press feels instant.

    Why daemon-side timing not pasteboard-side: the 100 ms wait is policy
    (depends on host responsiveness) and may need to become a 5-poll loop
    on a slow machine. That belongs at the orchestration layer, not in
    the platform abstraction.
    """
    original = pb.get_text()
    seqno_before = pb.clipboard_seqno()
    if not pb.simulate_copy():
        # SendInput / Quartz / osascript already logged the specific
        # failure. Caller will play fail beep + log voice-edit context.
        return None
    time.sleep(SELECTION_CAPTURE_WAIT_S)
    seqno_after = pb.clipboard_seqno()
    if seqno_before is not None and seqno_after is not None:
        if seqno_after == seqno_before:
            # No clipboard mutation → no selection in the focused app
            # (or simulate_copy hit a no-op modal). Restore not needed
            # (we never overwrote anything).
            return None
    selection = pb.get_text()
    if selection is None or not selection.strip():
        return None
    return selection, original


def _transcribe_and_emit_edit(selection: str,
                              original_clipboard: str | None) -> None:
    """v0.7.5 voice-edit transcribe path. Parallel to _transcribe_and_emit
    but: (a) uses _polisher.edit(selection, instruction) instead of
    .polish(instruction), (b) pastes the LLM's edit result, (c) restores
    the original clipboard in finally.

    Reads audio from _buffer at lock acquisition (same pattern as
    _transcribe_and_emit). selection/original_clipboard are passed in by
    _on_release (snapshotted under lock at the release moment).

    Skips encoder pipelining (edit recordings are typically short
    instructions — pipelining win is negligible). Skips POLISH_LANGUAGES
    gating (the LLM is given an EXPLICIT language-handling rule via
    EDIT_PROMPT, so all language combinations route here)."""
    global _processing, _buffer, _recording_samples
    with _state_lock:
        if _processing:
            dropped_chunks = len(_buffer)
            dropped_sec = _recording_samples / SAMPLE_RATE
            _buffer = []
            _recording_samples = 0
            if dropped_chunks > 0:
                logger.info(f"busy — dropped {dropped_sec:.2f}s of voice-edit "
                            f"audio ({dropped_chunks} blocks)")
            # Still need to restore the original clipboard even on busy-
            # drop, because _on_press already overwrote it via simulate_copy.
            _try_restore_clipboard(original_clipboard, context="busy-drop")
            return
        _processing = True
        chunks = _buffer
        _buffer = []
        _recording_samples = 0
    try:
        if not chunks:
            logger.info("voice-edit: empty audio (no instruction)")
            return
        samples_arr = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
        raw_sec = len(samples_arr) / SAMPLE_RATE
        trimmed = _trim_silence(samples_arr, SAMPLE_RATE)
        if len(trimmed) < SAMPLE_RATE * MIN_AUDIO_SEC:
            logger.info(f"voice-edit: too short ({raw_sec:.2f}s) — "
                        f"no instruction captured")
            return
        # ASR transcribes the spoken instruction
        t0 = time.time()
        instruction_raw, language = _backend.transcribe(trimmed)
        asr_elapsed = time.time() - t0
        if not instruction_raw.strip():
            logger.info(f"voice-edit: empty transcript ({language}, "
                        f"{asr_elapsed:.2f}s)")
            return
        # OpenCC normalisation on the instruction (Qwen3-ASR outputs
        # simplified natively — same logic as polish path).
        instruction = post_process(instruction_raw)

        # LLM applies instruction to selection
        t1 = time.time()
        edited = _polisher.edit(selection, instruction)
        edit_elapsed = time.time() - t1
        if edited is None:
            _play_beep(BEEP_FAIL_HZ, BEEP_DURATION_MS, BEEP_VOLUME, BEEPS_ENABLED)
            logger.warning(f"voice-edit: polish.edit returned None "
                           f"(instruction: {instruction!r})")
            return
        edited = post_process(edited)  # backstop simplified→traditional

        # Paste result
        if not _pasteboard.set_text(edited):
            logger.error(f"voice-edit: clipboard write failed — "
                         f"'{edited}' NOT inserted")
            return
        time.sleep(0.02)
        paste_ok = _pasteboard.paste()
        if paste_ok:
            _play_beep(BEEP_END_HZ, BEEP_DURATION_MS, BEEP_VOLUME, BEEPS_ENABLED)
            # Compact log: instruction + before/after lets user diff in
            # the daemon log without re-deriving from clipboard history.
            logger.info(f"voice-edit ({language}, {asr_elapsed:.2f}s+"
                        f"edit {edit_elapsed:.2f}s) "
                        f"instr: {instruction} | "
                        f"before: {selection} | "
                        f"after: {edited}")
        else:
            logger.warning(f"voice-edit: paste keystroke failed — '{edited}' "
                           f"on clipboard, press Ctrl+V/Cmd+V manually")
    except Exception as e:
        logger.error(f"voice-edit error: {e}")
    finally:
        # Restore original clipboard regardless of success/failure path.
        # User's clipboard history should look like nothing happened
        # beyond the paste of the edit result (which is then immediately
        # replaced — so user keeps their pre-edit clipboard intact).
        _try_restore_clipboard(original_clipboard, context="post-edit")
        with _state_lock:
            _processing = False
        _write_state(IDLE, edit_mode=True)


def _try_restore_clipboard(original: str | None, *, context: str) -> None:
    """Best-effort clipboard restore for voice-edit cleanup. Logs a
    warning on failure but does NOT raise — failed restore means user
    loses their pre-edit clipboard, an acceptable rare degradation."""
    if original is None:
        return  # nothing to restore (clipboard was empty pre-capture)
    try:
        if not _pasteboard.set_text(original):
            logger.warning(f"voice-edit: clipboard restore failed ({context})")
    except Exception as e:
        logger.warning(f"voice-edit: clipboard restore raised ({context}): "
                       f"{e}")


# ---------------------------------------------------------------------------
# Keyboard hooks
# ---------------------------------------------------------------------------
def _on_press(key) -> None:
    global _recording, _active_trigger, _recording_samples
    global _encoder_thread, _encoder_handle, _encoder_active, _encoder_failed
    global _encoder_use_batch_fallback, _encoder_silence_run_samples
    global _encoder_residual_samples
    global _edit_mode, _edit_selection, _edit_original_clipboard

    # v0.7.5: route both trigger key sets. EDIT_TRIGGER_KEYS may be None
    # (= disabled) or a set; same for TRIGGER_KEYS. The two sets are
    # expected to be disjoint — overlap would be ambiguous (we'd default
    # to dictate via the `elif` below).
    is_edit = bool(EDIT_TRIGGER_KEYS) and key in EDIT_TRIGGER_KEYS
    is_dictate = bool(TRIGGER_KEYS) and key in TRIGGER_KEYS and not is_edit
    if not (is_edit or is_dictate):
        return

    # v0.7.5 hotfix: cheap _active_trigger check BEFORE any expensive work
    # (selection capture takes ~100 ms). Windows OS fires _on_press on
    # every key-repeat tick while a key is held (~24×/s for F13) — without
    # this early-return, edit triggers would invoke _capture_selection on
    # every repeat: 24 × 100 ms wasted per second + 24 × failed-beep
    # cacophony + 24 × log-flood of "no selection captured" + 24 × Ctrl+C
    # injected into the focused app. Dictate path already had this early
    # return below (inside the lock); we needed to hoist it above the
    # selection-capture step.
    with _state_lock:
        if _active_trigger is not None:
            return  # OS key-repeat — first press already started recording

    # For edit, capture selection BEFORE acquiring state lock (again).
    # The 100 ms blocking sleep inside _capture_selection should not hold
    # _state_lock — the audio callback acquires _state_lock on every
    # 50 ms tick, and blocking it for 100 ms would back-pressure
    # PortAudio. (When _recording=False the callback early-returns without
    # locking, so this is only a defence-in-depth — recording hasn't
    # started yet.)
    captured: tuple[str, str | None] | None = None
    if is_edit:
        captured = _capture_selection(_pasteboard)
        if captured is None:
            _play_beep(BEEP_FAIL_HZ, BEEP_DURATION_MS, BEEP_VOLUME, BEEPS_ENABLED)
            logger.info("voice-edit: no selection captured")
            return  # DO NOT start recording
        # Verify polisher actually has edit capability — NoopPolisher's
        # edit() returns None unconditionally, so starting a recording
        # we know can't succeed would just waste 1-5s of user effort.
        if _polisher.__class__.__name__ == "NoopPolisher":
            _play_beep(BEEP_FAIL_HZ, BEEP_DURATION_MS, BEEP_VOLUME, BEEPS_ENABLED)
            logger.warning("voice-edit: POLISH_ENABLED is False — voice-edit "
                           "requires polish (LLM does the editing). Enable polish "
                           "or unset EDIT_TRIGGER_KEYS.")
            _try_restore_clipboard(captured[1], context="noop-polisher")
            return

    with _state_lock:
        if _active_trigger is not None:
            # Race: another trigger pressed during the ~100 ms selection
            # capture window. For edit, restore the captured clipboard
            # we modified via simulate_copy.
            if captured is not None:
                _try_restore_clipboard(captured[1], context="active-trigger")
            return
        _active_trigger = key
        _recording = True
        _write_state(RECORDING, edit_mode=bool(captured))
        _buffer.clear()
        # v0.7.2: reset sample counter for MAX_AUDIO_SEC tracking. Without
        # this, a previous transcribe that left a stale count + a new press
        # could falsely trip the cap and auto-stop the new recording early.
        _recording_samples = 0
        # v0.7.5: edit state — cleared every press, populated only if this
        # press is an edit trigger. Snapshot read in _on_release.
        _edit_mode = False
        _edit_selection = None
        _edit_original_clipboard = None
        if captured is not None:
            _edit_mode = True
            _edit_selection, _edit_original_clipboard = captured
        # v0.8.0: reset all per-recording encoder state. _encoder_consecutive_failures
        # is INTENTIONALLY NOT reset — it persists across utterances so 3
        # back-to-back failures suppress streaming for the 4th. A successful
        # finalize in _transcribe_and_emit resets it to 0.
        _encoder_active = False
        _encoder_failed = False
        _encoder_use_batch_fallback = False
        _encoder_silence_run_samples = 0
        _encoder_residual_samples = None
        _encoder_handle = None
        _encoder_thread = None
        _encoder_stop_event.clear()
        # Drain any leftover items from a prior (crashed?) recording's queue.
        while True:
            try:
                _encoder_queue.get_nowait()
            except queue.Empty:
                break
    edit_tag = " [edit]" if is_edit else ""
    logger.info(f"REC ({key}){edit_tag}")
    _play_beep(BEEP_START_HZ, BEEP_DURATION_MS, BEEP_VOLUME, BEEPS_ENABLED)


def _on_release(key) -> None:
    global _recording, _active_trigger
    # v0.7.5: accept release of either trigger key set.
    is_edit_key = bool(EDIT_TRIGGER_KEYS) and key in EDIT_TRIGGER_KEYS
    is_dictate_key = bool(TRIGGER_KEYS) and key in TRIGGER_KEYS
    if not (is_edit_key or is_dictate_key):
        return
    with _state_lock:
        if _active_trigger != key:
            return  # releasing a non-active trigger (e.g. tap of the other one)
        _active_trigger = None
        # v0.7.2: do NOT flip _recording=False here yet. PortAudio fires
        # the callback every 50 ms; if we stop capture immediately, the
        # in-flight 0-50 ms audio block that arrives between user release
        # and the next callback gets discarded by the `if _recording:` gate.
        # Real-world symptom: trailing phoneme clipped ("...這個 function"
        # → "...這個 functio"). Drain below.
    # Drain window: keep capturing for ~80 ms so the callback can land
    # the post-release block. Sleep is outside the lock — audio callback
    # continues to acquire the lock and append normally.
    # Why 80 ms (not 50 ms = exactly one PortAudio period)? Windows
    # time.sleep default resolution is ~15.6 ms (one multimedia timer
    # tick) and `sleep(0.05)` can return as early as ~47 ms. Padding to
    # 80 ms guarantees ≥ 1 callback period elapsed before we stop.
    # On macOS sleep precision is sub-millisecond so the extra 30 ms is
    # cheap perceived latency — bounded delay before transcribe starts.
    time.sleep(0.08)
    abort = False
    # v0.7.5: snapshot edit state under lock for coherent routing decision.
    edit_mode_snap = False
    edit_selection_snap: str | None = None
    edit_original_snap: str | None = None
    with _state_lock:
        # If user pressed a trigger again during the drain, _active_trigger
        # is non-None. Leave _recording=True (the new press wants to
        # continue capturing) and SKIP this release's transcribe spawn —
        # the new press will spawn its own when it eventually releases.
        if _active_trigger is not None:
            abort = True
        else:
            _recording = False
            _write_state(PROCESSING, edit_mode=_edit_mode)
            edit_mode_snap = _edit_mode
            edit_selection_snap = _edit_selection
            edit_original_snap = _edit_original_clipboard
    if abort:
        logger.info("release aborted by new press during drain")
        return
    # v0.8.0: signal the encoder worker (if any) to drain remaining queue
    # and exit. _transcribe_and_emit will join the worker before calling
    # finalize. Setting the event BEFORE the spawn ensures the worker
    # sees it on its next iteration even if _transcribe_and_emit is still
    # being scheduled.
    if _encoder_active:
        _encoder_stop_event.set()
    if edit_mode_snap:
        logger.info("voice-edit processing...")
        threading.Thread(
            target=_transcribe_and_emit_edit,
            args=(edit_selection_snap, edit_original_snap),
            daemon=True,
        ).start()
    else:
        logger.info("processing...")
        threading.Thread(target=_transcribe_and_emit, daemon=True).start()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    """Configure the 'stt' logger hierarchy.

    Format matches the legacy `[stt] <message>` pattern so existing log
    parsers (home_stt.py status) continue to work. Log level can be
    overridden via HOME_STT_LOG_LEVEL env var."""
    level_name = os.environ.get("HOME_STT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = logging.Formatter("[stt] %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(fmt)
    stdout_handler.addFilter(lambda r: r.levelno < logging.WARNING)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(fmt)

    root = logging.getLogger("stt")
    root.setLevel(level)
    root.addHandler(stdout_handler)
    root.addHandler(stderr_handler)


def _resolve_mic_device(spec: str | int | None) -> int | None:
    """Resolve a mic device spec to a sounddevice device index.

    None → system default (returns None).
    int  → passed through as device index.
    str  → substring match against input device names.
    """
    if spec is None:
        return None
    if isinstance(spec, int):
        return spec
    name = str(spec).lower()
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and name in dev["name"].lower():
            logger.info(f"mic: matched '{dev['name']}' (index {idx})")
            return idx
    logger.warning(f"mic: no input device matching '{spec}', using system default")
    return None


def main() -> None:
    global _backend, _pasteboard, TRIGGER_KEYS, EDIT_TRIGGER_KEYS

    _setup_logging()

    # Load user config (TOML file + env overrides) and apply to module globals.
    import stt_config as _cfg
    _user_cfg = _cfg.load_config()
    _this = sys.modules[__name__]
    _cfg.apply_to_module(_user_cfg, _this)
    _cf_path = _cfg.config_path()
    if _cf_path.exists():
        logger.info(f"config: {_cf_path}")

    _pasteboard = build_pasteboard()
    n_native = _pasteboard.register_native_libs()

    logger.info(f"home-stt v{__version__} starting")
    logger.info(f"platform: {sys.platform} ({_host_platform.machine()}) | "
                f"native libs registered: {n_native}")
    logger.info(f"backend: {STT_BACKEND} | model: {STT_MODEL}")
    paste_desc = _pasteboard.describe_paste_path()
    if paste_desc:
        logger.info(f"paste path: {paste_desc}")

    global _polisher
    _polisher = build_polisher(POLISH_ENABLED, POLISH_MODEL, POLISH_PROMPT)
    logger.info(f"polish: {_polisher.device_label}")

    _backend = build_backend_with_fallback(STT_BACKEND, STT_MODEL, SAMPLE_RATE)

    if TRIGGER_KEYS is None:
        TRIGGER_KEYS = _pasteboard.default_trigger_keys
    # v0.7.5: apply EDIT_TRIGGER_KEYS default. `None` = use platform
    # default (Win: F13, Mac: Right Cmd — defined in WindowsPasteboard
    # / MacOSPasteboard subclasses). Explicit `set()` = disabled.
    if EDIT_TRIGGER_KEYS is None:
        EDIT_TRIGGER_KEYS = _pasteboard.default_edit_trigger_keys

    logger.info(f"warming up on {_backend.device_label}...")
    t0 = time.time()
    _backend.warmup()
    trigger_labels = ", ".join(str(k) for k in TRIGGER_KEYS)
    if EDIT_TRIGGER_KEYS:
        edit_labels = ", ".join(str(k) for k in EDIT_TRIGGER_KEYS)
        # v0.7.5: voice-edit requires polish (NoopPolisher.edit returns
        # None). Warn at startup so user isn't surprised when the fail
        # beep plays on every edit press.
        if _polisher.__class__.__name__ == "NoopPolisher":
            logger.warning(f"EDIT_TRIGGER_KEYS set ({edit_labels}) but "
                           f"polish is disabled — voice-edit will fail-beep on every "
                           f"press. Enable polish or unset EDIT_TRIGGER_KEYS.")
        logger.info(f"warmup {time.time()-t0:.1f}s — hold {trigger_labels} "
                    f"to dictate, hold {edit_labels} to voice-edit.")
    else:
        logger.info(f"warmup {time.time()-t0:.1f}s — hold {trigger_labels} "
                    f"to record.")

    _mic_id = _resolve_mic_device(MIC_DEVICE)
    stream = sd.InputStream(
        device=_mic_id,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=_audio_callback,
        blocksize=int(SAMPLE_RATE * 0.05),  # 50 ms chunks
    )
    stream.start()
    _write_state(IDLE)
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()
    try:
        listener.join()
    except KeyboardInterrupt:
        logger.info("bye")
    finally:
        stream.stop()
        stream.close()
        _cleanup_state()


if __name__ == "__main__":
    main()
