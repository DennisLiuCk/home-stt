"""
Optional text post-processing layer.

Sits between the ASR backend's output (after s2tw + CJK/ASCII spacing) and
the clipboard write. The default `NoopPolisher` returns the input unchanged
— so existing setups without polish enabled behave identically.

The `MlxLocalLlmPolisher` uses `mlx-lm` to run a small instruction-tuned
LLM (default `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, ~900 MB) with a
polish prompt that:

  - removes filler words (呃, 嗯, 就是, 那個, 然後, …)
  - fixes immediate repetitions / slips of tongue (「我我我覺得」→「我覺得」)
  - preserves speaker meaning and sentence style
  - outputs the polished string only (no quotes, no explanation)

Design notes:

  - Failures (mlx-lm not installed, model load OOM, generate exception)
    degrade silently to NoopPolisher — the daemon must continue working
    with raw ASR output. The startup log line `[stt] polish: …` shows
    which polisher is active.
  - Polish runs synchronously inside `_transcribe_and_emit`, adding
    ~200-500 ms latency on Apple Silicon for short clips (1-3 sentences).
    Larger polish models can blow the daemon's hold-to-talk perception
    budget; stick to ≤2B params on a 16 GB Mac.
  - The interface is intentionally text-in / text-out for now. A future
    multimodal polisher (Qwen3-Omni-style) that wants audio context
    can extend the signature with a default-None audio kwarg.
"""
from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod


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
                {"role": "user", "content": text},
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


def build_polisher(
    enabled: bool,
    model_name: str,
    system_prompt: str,
) -> TextPostProcessor:
    """Factory. Falls back to NoopPolisher if mlx-lm is unavailable, the
    model fails to download / load, or `enabled` is False."""
    if not enabled:
        return NoopPolisher()
    try:
        return MlxLocalLlmPolisher(model_name, system_prompt)
    except Exception as e:
        print(
            f"[stt] polish disabled — could not initialise "
            f"{model_name}: {e}",
            file=sys.stderr, flush=True,
        )
        return NoopPolisher()
