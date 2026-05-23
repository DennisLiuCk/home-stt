"""
Optional text post-processing layer.

Sits between the ASR backend's output (after s2tw + CJK/ASCII spacing) and
the clipboard write. The default `NoopPolisher` returns the input unchanged
— so existing setups without polish enabled behave identically.

Two real impls share the same prompt + behaviour, chosen by `build_polisher`
based on platform:

  - `MlxLocalLlmPolisher` — Apple Silicon, via `mlx-lm` (Metal native).
  - `TorchLocalLlmPolisher` — Windows / Linux, via `transformers` +
    PyTorch + NVIDIA CUDA (bfloat16). Refuses to run on CPU because a
    multi-billion-param polish model on CPU is 30-60× over the
    hold-to-talk perception budget.

Both impls run the same polish prompt that:

  - removes filler words (呃, 嗯, 就是, 那個, 然後, …)
  - fixes immediate repetitions / slips of tongue (「我我我覺得」→「我覺得」)
  - preserves speaker meaning and sentence style
  - outputs the polished string only (no quotes, no explanation)

Design notes:

  - Failures (package missing, model load OOM, generate exception) degrade
    silently to NoopPolisher — the daemon must continue working with raw
    ASR output. `build_polisher` differentiates ImportError vs CUDA OOM
    vs other so the user gets an actionable next step in the startup log.
    The startup line `[stt] polish: …` shows which polisher is active.
  - Polish runs synchronously inside `_transcribe_and_emit`, adding
    ~200-500 ms latency on Apple Silicon and ~500-1000 ms on NVIDIA CUDA
    bfloat16 for short clips (1-3 sentences).
  - The interface is intentionally text-in / text-out for now. A future
    multimodal polisher (Qwen3-Omni-style) that wants audio context
    can extend the signature with a default-None audio kwarg.
"""
from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod


def _format_polish_user_msg(text: str) -> str:
    """Wrap ASR text as data, not as a user-role request.

    Strong instruction-tuned models (esp. unquantised bf16, less so 4-bit
    quantised) otherwise treat question-shaped ASR output (e.g. "幫我
    review 這個 function") as a real conversation request and ANSWER it
    instead of polishing — despite the system prompt's rules. Wrapping
    with an explicit "請修飾以下逐字稿:" prefix re-frames the input as
    data to be transformed, which the model then handles correctly.

    Applied uniformly to both MLX (Mac) and Torch (Win/Linux) impls so
    the same POLISH_MODEL produces the same output across platforms.
    """
    return f"請修飾以下逐字稿:\n{text}"


class TextPostProcessor(ABC):
    """Polish raw ASR text. Implementations may transform, leave unchanged,
    or fail-safely return the input. Must NEVER raise — failure modes are
    surfaced via the printed `[stt] polish failed: …` line and falling
    back to the input text."""

    @abstractmethod
    def polish(self, text: str) -> str:
        """Return polished text. Return input unchanged on any failure."""

    @property
    def device_label(self) -> str:
        """Short label for the startup `[stt] polish: …` log line."""
        return "unknown"


class NoopPolisher(TextPostProcessor):
    """Identity polisher. Used when POLISH_ENABLED is False or when the
    MLX-based polisher failed to initialise (e.g. mlx-lm missing)."""

    @property
    def device_label(self) -> str:
        return "disabled (raw ASR output)"

    def polish(self, text: str) -> str:
        return text


class MlxLocalLlmPolisher(TextPostProcessor):
    """MLX-LM-backed text polisher for Apple Silicon.

    Loads the model + tokenizer at __init__ (so the daemon pays the cost
    once at startup, not per transcription). On each polish call it
    formats a chat-style prompt via the tokenizer's chat template and
    runs greedy generation up to `max_tokens` new tokens.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        max_tokens: int = 256,
    ) -> None:
        # Lazy import — keeps mlx_lm out of the import graph for users who
        # disabled polish or run on a non-Apple-Silicon platform.
        from mlx_lm import generate, load

        self._generate = generate
        self._model, self._tokenizer = load(model_name)
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._model_name = model_name

    @property
    def device_label(self) -> str:
        return f"{self._model_name} (MLX, ≤{self._max_tokens} tok)"

    def polish(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text
        try:
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": _format_polish_user_msg(text)},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            t0 = time.time()
            response = self._generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self._max_tokens,
                verbose=False,
            )
            elapsed = time.time() - t0
            polished = (response or "").strip()
            if not polished:
                return text
            # Best-effort guard: some models occasionally wrap output in
            # quotes despite the prompt. Strip a single matched pair.
            if (polished.startswith('"') and polished.endswith('"')) or \
               (polished.startswith("「") and polished.endswith("」")):
                polished = polished[1:-1].strip() or polished
            # Stash the last polish duration so the caller can log it.
            self.last_elapsed = elapsed
            return polished
        except Exception as e:
            print(f"[stt] polish failed, returning raw: {e}",
                  file=sys.stderr, flush=True)
            return text


class TorchLocalLlmPolisher(TextPostProcessor):
    """PyTorch + transformers + NVIDIA CUDA polisher for Windows / Linux.

    VRAM budget: Qwen3-4B-Instruct-2507 bfloat16 needs ~8 GB on GPU. Combined
    with Qwen3-ASR-0.6B (~1.5 GB), the daemon wants ~10 GB VRAM total. Cards
    with < 10 GB should either set POLISH_ENABLED = False, or swap to a
    smaller polish model (Qwen2.5-1.5B-Instruct ~3 GB bf16). build_polisher
    catches CUDA OOM at init and prints the actionable swap-or-disable hint.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        max_tokens: int = 256,
    ) -> None:
        # Heavy lazy imports — only paid on Win/Linux + polish enabled.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            # A 4B model on CPU is ~30-60s/polish — way over the
            # hold-to-talk perceptual budget. Refuse rather than silently
            # making every transcription unusable; build_polisher's
            # exception handler catches this and falls back to NoopPolisher
            # with an install-hint log line.
            raise RuntimeError(
                "TorchLocalLlmPolisher requires CUDA. On CPU a multi-B "
                "polish model is far over the daemon's perceptual budget. "
                "Install torch with CUDA support, or set POLISH_ENABLED = "
                "False."
            )

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        self._model.eval()
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._model_name = model_name
        try:
            # torch returns e.g. "NVIDIA GeForce RTX 5080" — already has
            # vendor prefix; don't add a second "NVIDIA ".
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA device"
        self._device_label = (
            f"{model_name} (PyTorch bfloat16 @ {gpu_name}, "
            f"≤{max_tokens} tok)"
        )

    @property
    def device_label(self) -> str:
        return self._device_label

    def polish(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text
        try:
            # Wrapper logic (request → data framing) lives in
            # _format_polish_user_msg at module scope and is shared with
            # MlxLocalLlmPolisher for cross-platform output consistency.
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": _format_polish_user_msg(text)},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda:0")
            with self._torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_tokens,
                    # Greedy (do_sample=False) is deterministic + ~30%
                    # faster than sampling for this single-step polish task.
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            polished = self._tokenizer.decode(
                new_tokens, skip_special_tokens=True,
            ).strip()
            if not polished:
                return text
            # Best-effort guard: some models occasionally wrap output in
            # quotes despite the prompt. Strip a single matched pair.
            if (polished.startswith('"') and polished.endswith('"')) or \
               (polished.startswith("「") and polished.endswith("」")):
                polished = polished[1:-1].strip() or polished
            return polished
        except Exception as e:
            print(f"[stt] polish failed, returning raw: {e}",
                  file=sys.stderr, flush=True)
            return text


def build_polisher(
    enabled: bool,
    model_name: str,
    system_prompt: str,
) -> TextPostProcessor:
    """Factory. Dispatches on `sys.platform` like build_pasteboard /
    build_backend. Falls back to NoopPolisher on any init failure, with an
    error-type-specific hint logged to stderr so the user knows whether to
    reinstall, swap models, or disable polish."""
    if not enabled:
        return NoopPolisher()
    try:
        import platform as _platform
        if sys.platform == "darwin" and _platform.machine() == "arm64":
            return MlxLocalLlmPolisher(model_name, system_prompt)
        return TorchLocalLlmPolisher(model_name, system_prompt)
    except (ImportError, ModuleNotFoundError) as e:
        print(
            f"[stt] polish disabled — required package missing for "
            f"{model_name}: {e}. Install torch+CUDA and transformers "
            f"(see README → Windows 安裝步驟), or set POLISH_ENABLED = "
            f"False to silence this.",
            file=sys.stderr, flush=True,
        )
        return NoopPolisher()
    except Exception as e:
        # Three actionable failure classes:
        #   - CUDA OOM: torch.cuda.OutOfMemoryError on newer torch, plain
        #     RuntimeError carrying 'CUDA out of memory' on older. Hint:
        #     swap to a smaller model.
        #   - DLL load failure: OSError raised when torch.cuda.is_available()
        #     or from_pretrained triggers loading cudart/cudnn/cublas DLLs
        #     that aren't installed or are on a wrong CUDA version. Hint:
        #     install the missing NVIDIA wheels / reinstall torch.
        #   - Other: surface raw exception.
        msg = str(e)
        is_oom = (
            "out of memory" in msg.lower()
            or type(e).__name__ == "OutOfMemoryError"
        )
        is_dll = isinstance(e, OSError) and any(
            s in msg.lower() for s in ("dll", "cudart", "cudnn", "cublas")
        )
        if is_oom:
            print(
                f"[stt] polish disabled — CUDA OOM loading {model_name}: "
                f"{e}. Try POLISH_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct' "
                f"(~3 GB VRAM), or set POLISH_ENABLED = False.",
                file=sys.stderr, flush=True,
            )
        elif is_dll:
            print(
                f"[stt] polish disabled — CUDA DLL load failed for "
                f"{model_name}: {e}. Install nvidia-cudnn-cu12 + "
                f"nvidia-cublas-cu12 (Windows), or reinstall torch with "
                f"the CUDA wheel (see README -> Windows 安裝步驟).",
                file=sys.stderr, flush=True,
            )
        else:
            print(
                f"[stt] polish disabled — could not initialise "
                f"{model_name}: {e}",
                file=sys.stderr, flush=True,
            )
        return NoopPolisher()
