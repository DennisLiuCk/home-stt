"""STT backend abstraction — ABC + concrete implementations + factory.

Extracted from stt-daemon.py (v0.7.5) so backend code lives in its own
module. The daemon imports `STTBackend`, `build_backend`, and
`build_backend_with_fallback` from here.
"""
from __future__ import annotations

import logging
import platform as _host_platform
import sys
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger("stt.backends")

_DEFAULT_SAMPLE_RATE = 16000


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
    sample_rate: int = _DEFAULT_SAMPLE_RATE

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

    # ----------------------------------------------------------------
    # v0.8.0: optional streaming/encoder-pipelining API
    #
    # The default implementations below let existing batch-only backends
    # (FasterWhisperBackend, MlxWhisperBackend) opt out automatically —
    # they need ZERO changes for the v0.8.0 daemon refactor. Only
    # Qwen3AsrBackend overrides supports_streaming()→True and implements
    # the four streaming methods.
    #
    # Lifecycle:
    #     handle = backend.start_encoder()       # on first audio chunk
    #     backend.push_chunk(handle, slab)       # every ENCODER_CHUNK_SEC
    #     ...
    #     text, lang = backend.finalize(handle, tail)   # on release
    #         # tail = leftover audio not yet encoded; finalize must
    #         # encode it as the last slab before running the decoder.
    #
    # Or, on any failure / abort:
    #     backend.abort(handle)                  # release GPU buffers
    #
    # The handle type is opaque to the daemon (typed `Any`). Backends
    # define their own dataclass (e.g. holding a list of hidden-state
    # tensors). The daemon never touches it.
    #
    # All methods MUST be safe to call from a non-main thread — the
    # encoder runs on a dedicated worker thread (see _encoder_worker
    # in this file).
    # ----------------------------------------------------------------

    def supports_streaming(self) -> bool:
        """Override + return True to opt into press-time encoder pipelining.

        Default False — daemon will use the existing batch `transcribe(samples)`
        path for this backend, same behaviour as v0.7.2.
        """
        return False

    def start_encoder(self) -> Any:
        """Return an opaque handle that subsequent push_chunk / finalize /
        abort calls thread through. Called once at the start of each
        recording (on first audio block, lazy)."""
        raise NotImplementedError(
            f"{self.name}: start_encoder requires supports_streaming()=True"
        )

    def push_chunk(self, handle: Any, samples: np.ndarray) -> None:
        """Run the encoder forward on a ~ENCODER_CHUNK_SEC slab of audio
        and accumulate the resulting hidden states on `handle`. Called
        many times per recording, on the worker thread."""
        raise NotImplementedError(
            f"{self.name}: push_chunk requires supports_streaming()=True"
        )

    def finalize(self, handle: Any, tail_samples: np.ndarray) -> tuple[str, str]:
        """Encode the residual `tail_samples` as the final slab, concatenate
        all accumulated hidden states, run the decoder, return
        (raw_text, language_code) — matching the existing `transcribe()`
        contract. Called once per recording on the transcribe thread.

        `tail_samples` may be empty (length 0) if the user released right
        at a chunk boundary."""
        raise NotImplementedError(
            f"{self.name}: finalize requires supports_streaming()=True"
        )

    def abort(self, handle: Any) -> None:
        """Release any GPU buffers / state on `handle` without running the
        decoder. Called when the daemon decides to fall back to the batch
        path mid-recording (encoder crashed, silence-detected, etc.).
        MUST NOT raise — best-effort cleanup only."""
        raise NotImplementedError(
            f"{self.name}: abort requires supports_streaming()=True"
        )


class FasterWhisperBackend(STTBackend):
    """OpenAI Whisper via CTranslate2 (faster-whisper). Tries CUDA float16
    first, falls back to CPU int8 if CUDA initialisation fails."""

    name = "faster-whisper"

    def __init__(self, model_name: str, sample_rate: int = _DEFAULT_SAMPLE_RATE):
        from faster_whisper import WhisperModel
        self.sample_rate = sample_rate

        try:
            self._model = WhisperModel(model_name, device="cuda",
                                       compute_type="float16")
            self._device_label = "CUDA (float16)"
        except Exception as e:
            logger.info(f"CUDA load failed ({e}); falling back to CPU int8.")
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
        warm_audio = np.zeros(self.sample_rate, dtype=np.float32)
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

    def __init__(self, model_name: str, sample_rate: int = _DEFAULT_SAMPLE_RATE):
        import mlx_whisper  # lazy import (Apple Silicon only)

        self.sample_rate = sample_rate
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
        warm_audio = np.zeros(self.sample_rate, dtype=np.float32)
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

    def __init__(self, model_name: str, sample_rate: int = _DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._model_name = self._resolve_model_name(model_name)
        if sys.platform == "darwin" and _host_platform.machine() == "arm64":
            self._impl = _Qwen3MlxImpl(self._model_name, sample_rate)
        else:
            self._impl = _Qwen3TorchImpl(self._model_name, sample_rate)

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

    def _normalise_result(self, result: dict) -> tuple[str, str]:
        text = (result.get("text") or "").strip()
        raw_lang = (result.get("language") or "").strip().lower()
        # Normalise "Chinese" → "zh", "English" → "en", etc. Falls back to
        # the first two letters of whatever the model returned so unknown
        # languages still produce something sensible in the log line.
        language = self._LANG_NORM.get(raw_lang, raw_lang[:2] if raw_lang else "")
        return text, language

    def transcribe(self, samples: np.ndarray) -> tuple[str, str]:
        return self._normalise_result(self._impl.transcribe(samples))

    def warmup(self) -> None:
        self._impl.warmup()

    # ----------------------------------------------------------------
    # v0.8.0 streaming delegation. Outer wrapper just routes to the
    # platform impl (MLX or Torch) which makes its own decision about
    # whether streaming is wired. Lang normalisation via _LANG_NORM
    # mirrors transcribe().
    # ----------------------------------------------------------------
    def supports_streaming(self) -> bool:
        return self._impl.supports_streaming()

    def start_encoder(self) -> Any:
        return self._impl.start_encoder()

    def push_chunk(self, handle: Any, samples: np.ndarray) -> None:
        self._impl.push_chunk(handle, samples)

    def finalize(self, handle: Any, tail_samples: np.ndarray) -> tuple[str, str]:
        return self._normalise_result(self._impl.finalize(handle, tail_samples))

    def abort(self, handle: Any) -> None:
        self._impl.abort(handle)


class _Qwen3MlxImpl:
    """Apple Silicon path — mlx-qwen3-asr (Metal native)."""

    def __init__(self, model_name: str, sample_rate: int = _DEFAULT_SAMPLE_RATE):
        import mlx_qwen3_asr  # lazy import (Apple Silicon only)

        self._mqa = mlx_qwen3_asr
        self._model_name = model_name
        self._sample_rate = sample_rate
        self.device_label = "Apple Silicon (Metal, MLX) — Qwen3-ASR"

    def supports_streaming(self) -> bool:
        # v0.8.0 ships Torch path only. MLX deferred to v0.8.1 — see
        # tasks list. Returning False here means the daemon's batch
        # transcribe path runs unchanged on Mac (zero v0.7.2 regression).
        return False

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
        warm_audio = np.zeros(self._sample_rate, dtype=np.float32)
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

    def __init__(self, model_name: str, sample_rate: int = _DEFAULT_SAMPLE_RATE):
        import torch
        self._sample_rate = sample_rate
        # v0.8.0: use the streaming-capable subclass. transcribe()
        # behaviour is inherited unchanged from Qwen3ASRModel; the
        # streaming methods are additive. If the wrapper import fails
        # (qwen-asr upstream refactor broke our private-attr access),
        # caller can catch ImportError and fall back to the original
        # Qwen3ASRModel — but for now the spike-verified path is
        # production code.
        from qwen3_asr_streaming import StreamingQwen3ASRModel

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
        self._model = StreamingQwen3ASRModel.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )
        self._model_name = model_name
        # Probe whether streaming is actually wired (upstream attr
        # availability check). If it returns False, supports_streaming()
        # below also reports False and the daemon stays on batch path —
        # zero v0.7.2 regression.
        self._streaming_ok = StreamingQwen3ASRModel._streaming_supported(self._model)
        stream_tag = " + streaming" if self._streaming_ok else ""
        self.device_label = f"{gpu_name} (bfloat16) — Qwen3-ASR{stream_tag}"
        if not self._streaming_ok:
            logger.warning("qwen-asr streaming attrs missing — encoder "
                           "pipelining disabled for this session, daemon will "
                           "use batch path (v0.7.2 behaviour)")

    def supports_streaming(self) -> bool:
        # Both ENCODER_PIPELINING (daemon-side switch) AND _streaming_ok
        # (upstream-API probe) gate the streaming path; either one False
        # routes to batch transcribe.
        return self._streaming_ok

    def start_encoder(self) -> Any:
        return self._model.start_encoder()

    def push_chunk(self, handle: Any, samples: np.ndarray) -> None:
        self._model.encode_chunk(handle, samples)

    def finalize(self, handle: Any, tail_samples: np.ndarray) -> dict:
        text, language = self._model.finalize_with_features(handle, tail_samples)
        return {"text": text, "language": language}

    def abort(self, handle: Any) -> None:
        self._model.abort(handle)

    def transcribe(self, samples: np.ndarray) -> dict:
        results = self._model.transcribe(
            audio=(samples, self._sample_rate),
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
        warm_audio = np.zeros(self._sample_rate, dtype=np.float32)
        self._model.transcribe(
            audio=(warm_audio, self._sample_rate),
            language=None,
        )


def build_backend(name: str, model: str, sample_rate: int = _DEFAULT_SAMPLE_RATE) -> STTBackend:
    """Factory. To add a new backend: implement STTBackend in a new class,
    add a branch here, and update STT_BACKEND in the Config section."""
    if name == "faster-whisper":
        return FasterWhisperBackend(model, sample_rate)
    if name == "mlx-whisper":
        return MlxWhisperBackend(model, sample_rate)
    if name == "qwen3-asr":
        return Qwen3AsrBackend(model, sample_rate)
    # ── Future backends ────────────────────────────────────────────────
    # elif name == "sense-voice":
    #     return SenseVoiceBackend(model)
    # elif name == "paraformer":
    #     return ParaformerBackend(model)
    raise ValueError(f"Unknown STT backend: {name!r}")


def build_backend_with_fallback(backend_name: str, model_name: str,
                                sample_rate: int = _DEFAULT_SAMPLE_RATE) -> STTBackend:
    """Try the configured STT backend; on ImportError (missing package) or
    CUDA OOM, fall back to faster-whisper with a loud actionable stderr
    message. If that also fails, exit cleanly — the daemon can't run
    without an STT backend, and a half-initialised state is worse than a
    clear-cut exit. Mirrors build_polisher's degrade-gracefully pattern."""
    try:
        return build_backend(backend_name, model_name, sample_rate)
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(
            f"backend '{backend_name}' missing required package: "
            f"{e}. Falling back to faster-whisper. To enable "
            f"{backend_name}, see README -> Windows 安裝步驟 (install "
            f"torch+CUDA wheel before `pip install qwen-asr`)."
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
        logger.warning(
            f"backend '{backend_name}' init failed: {e}{hint}. "
            f"Falling back to faster-whisper."
        )

    try:
        return build_backend("faster-whisper", "large-v3-turbo", sample_rate)
    except Exception as e:
        logger.critical(
            f"faster-whisper fallback also failed: {e}. "
            f"Daemon cannot continue — install dependencies per README "
            f"-> Windows 安裝步驟 and restart."
        )
        raise SystemExit(1)
