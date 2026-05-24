"""
Optional text post-processing layer.

Sits between the ASR backend's output (after s2twp + CJK/ASCII spacing) and
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
        # v0.7.2: bumped 256 → 512. Old cap silent-truncated polish on the
        # README "超長" stress case (~280-char zh, ~280 tokens with Qwen3
        # tokenizer). 512 covers the realistic worst case while still
        # acting as a runaway safety cap. Memory cost is negligible (~25
        # MB extra KV during decode for the larger ceiling, vs 8 GB
        # model weights). Per-call latency unchanged — max_new_tokens is
        # a cap, the model stops at EOS naturally on short outputs.
        max_tokens: int = 512,
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
            response = self._generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self._max_tokens,
                verbose=False,
            )
            polished = (response or "").strip()
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


class TorchLocalLlmPolisher(TextPostProcessor):
    """PyTorch + transformers + NVIDIA CUDA polisher for Windows / Linux.

    VRAM budget: Qwen3-4B-Instruct-2507 bfloat16 needs ~8 GB on GPU. Combined
    with Qwen3-ASR-0.6B (~1.5 GB), the daemon wants ~10 GB VRAM total. Cards
    with < 10 GB should either set POLISH_ENABLED = False, or swap to a
    smaller polish model (Qwen2.5-1.5B-Instruct ~3 GB bf16). build_polisher
    catches CUDA OOM at init and prints the actionable swap-or-disable hint.
    """

    # v0.7.0: attention impl preference order. Auto-falls back if a higher
    # candidate isn't available. flash_attention_2 needs `pip install
    # flash-attn`; sdpa is PyTorch 2.0+ built-in; eager is the last-resort
    # legacy path.
    _PREFERRED_ATTN = ("flash_attention_2", "sdpa", "eager")

    # v0.7.0: 4-bit NF4 quantization via bitsandbytes. Toggle True after
    # `pip install bitsandbytes` is confirmed working. Cuts VRAM ~75%
    # nominally. Measured 2026-05-23 on RTX 5080 + Qwen2.5-1.5B-Instruct:
    # **67% SLOWER** on long polish (5.44s → 9.07s) and quality drift vs
    # bf16 baseline. Root cause: per-layer dequant kernel overhead exceeds
    # memory-bandwidth savings on small-model + fast-GPU + short-batch
    # workloads (INT4's sweet spot is big-model + slow-GPU). bnb's
    # Blackwell sm_120 kernels are also not yet tuned (FutureWarning on
    # init re: deprecated torch APIs). Keep False until either model is
    # larger (≥7B) or bnb gets Blackwell-aware kernels.
    _USE_4BIT_QUANT = False

    # v0.7.0: torch.compile the model after load. On LINUX adds ~30-60s
    # startup warmup but ~20-30% decode improvement once warm. On WINDOWS
    # this is silently no-op: torch.compile's inductor backend depends on
    # `triton`, which has no official Windows wheel (only the community
    # `triton-windows` fork). Without triton, inductor falls back to
    # aot_eager / no-op — measured 2026-05-23 on RTX 5080: 0% improvement,
    # load +1.7s. Keep False on Windows; experiment on Linux in a v0.7.x
    # followup.
    _USE_TORCH_COMPILE = False

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        # v0.7.2: bumped 256 → 512. See MlxLocalLlmPolisher comment for
        # rationale. The dynamic budget calc in polish() also caps to
        # this value, so it acts as the hard safety ceiling against
        # runaway generation. Bump it higher (e.g. 1024) only if you
        # routinely dictate multi-paragraph utterances.
        max_tokens: int = 512,
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
        # v0.7.1: cuDNN benchmark mode picks fastest kernel per shape on
        # first use, then reuses. Decode steps all share the same kernel
        # shape (1 token forward) so benefit accrues fast. Marginal but
        # free; harmless if shapes vary because cuDNN just re-picks.
        torch.backends.cudnn.benchmark = True
        # Pinned device id used by from_pretrained AND polish's input
        # tensor move — kept in one place so they stay in sync. To run
        # on a different physical GPU, set CUDA_VISIBLE_DEVICES env var
        # (remaps the chosen GPU to cuda:0 inside the process).
        self._device = "cuda:0"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        # v0.7.0: Resolve best available attention impl. Use try/import,
        # not the transformers helper — helper's API path varies across
        # versions. flash_attention_2 needs `pip install flash-attn`;
        # sdpa is PyTorch 2.0+ built-in.
        attn = "eager"
        for candidate in self._PREFERRED_ATTN:
            if candidate == "flash_attention_2":
                try:
                    import flash_attn  # noqa: F401
                    attn = candidate
                    break
                except ImportError:
                    continue
            else:
                attn = candidate
                break

        load_kwargs = dict(
            dtype=torch.bfloat16,
            device_map=self._device,
            attn_implementation=attn,
        )
        if self._USE_4BIT_QUANT:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            # quantization_config overrides dtype — drop to avoid warning
            load_kwargs.pop("dtype")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs,
        )
        self._model.eval()

        if self._USE_TORCH_COMPILE:
            # mode="reduce-overhead" tuned for autoregressive decode;
            # falls back to default mode if unsupported by the model.
            self._model = torch.compile(self._model, mode="reduce-overhead")

        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._model_name = model_name
        # Resolve generate's pad/stop token once. For Qwen2/3 chat models the
        # chat template terminates at <|im_end|>; some community variants
        # report tokenizer.eos_token_id as a list or use a different base-EOS
        # that doesn't match the chat boundary. Pin to <|im_end|> when
        # available (so generate() stops at chat boundary), else fall back
        # to a scalar eos.
        self._pad_token_id = self._resolve_pad_token_id()
        try:
            # torch returns e.g. "NVIDIA GeForce RTX 5080" — already has
            # vendor prefix; don't add a second "NVIDIA ".
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA device"
        quant_label = "INT4-NF4" if self._USE_4BIT_QUANT else "bfloat16"
        compile_label = " + torch.compile" if self._USE_TORCH_COMPILE else ""
        self._device_label = (
            f"{model_name} (PyTorch {quant_label} + {attn}{compile_label} "
            f"@ {gpu_name}, ≤{max_tokens} tok, PLD+prefix-cache)"
        )

        # v0.7.1: pre-build GenerationConfig once. Per-call construction
        # is ~1-2 ms overhead; we run polish many times. Also enables
        # Prompt Lookup Decoding (PLD) — for "output ≈ input with
        # minimal edits" tasks (polish is the textbook case), PLD
        # typically delivers 1.8-2.5x decode speedup on Chinese-tokenized
        # text. Lossless: the verifier is still the full model.
        # `prompt_lookup_num_tokens=10` is the apoorvumang reference
        # sweet spot; lower (5) is safer on adversarial inputs, higher
        # (15) wins more on long unchanged spans.
        from transformers import GenerationConfig
        self._gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self._pad_token_id,
            prompt_lookup_num_tokens=10,
        )

        # v0.7.1: pre-compute KV cache for the static system block. The
        # POLISH_PROMPT (+ chat template scaffolding around it) is
        # identical across every polish call, so paying its ~200-token
        # prefill cost once at startup saves it on every subsequent call
        # (~30-50 ms per call on RTX 5080). Cache is deep-copied per
        # request because DynamicCache mutates during decode.
        self._build_prefix_cache()

    def _build_prefix_cache(self) -> None:
        """Pre-fill DynamicCache with the static system message + chat
        template overhead. Stores `self._prefix_cache` and
        `self._prefix_len` for reuse in `polish()`. Idempotent if called
        multiple times (overwrites previous cache)."""
        from transformers import DynamicCache
        sys_msg = [{"role": "system", "content": self._system_prompt}]
        prefix_text = self._tokenizer.apply_chat_template(
            sys_msg, tokenize=False, add_generation_prompt=False,
        )
        prefix_inputs = self._tokenizer(
            prefix_text, return_tensors="pt", add_special_tokens=False,
        ).to(self._device)
        cache = DynamicCache()
        with self._torch.no_grad():
            result = self._model(
                input_ids=prefix_inputs.input_ids,
                past_key_values=cache,
                use_cache=True,
            )
        self._prefix_cache = result.past_key_values
        self._prefix_len = prefix_inputs.input_ids.shape[1]

    def _resolve_pad_token_id(self) -> int:
        """Pin pad/stop to <|im_end|> when present (Qwen2/3 chat models all
        terminate there). Fall back to a scalar form of eos_token_id —
        defensively unwrap list form some community fine-tunes report.
        Without this, model swaps that report eos as a list trip
        generate()'s 'pad_token_id must be int' ValueError; swaps where
        base eos != chat terminator run past chat boundary to max_tokens."""
        im_end = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
        unk = getattr(self._tokenizer, "unk_token_id", None)
        if im_end is not None and im_end != unk:
            return im_end
        eos = self._tokenizer.eos_token_id
        return eos[0] if isinstance(eos, list) else eos

    @property
    def device_label(self) -> str:
        return self._device_label

    def polish(self, text: str) -> str:
        import copy
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
            # add_special_tokens=False: the chat template already inserts
            # any BOS / role-marker tokens the model expects. For Qwen
            # tokenizers add_bos_token defaults False so this is a no-op
            # today, but Llama-family tokenizers auto-prepend <s> — without
            # this flag, a future POLISH_MODEL swap to Llama-derived weights
            # would silently produce double-BOS prompts and degrade quality
            # with no error.
            inputs = self._tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False,
            ).to(self._device)
            input_len = inputs["input_ids"].shape[-1]
            # v0.7.2: dynamic max_new_tokens. Polish is a "minimum-edit"
            # task — output length ≈ input length. The previous static
            # 256-token cap silently truncated outputs on stress inputs
            # (~280-char zh utterances, where Qwen3 tokenizes ~1
            # token/char). Now: budget ~1.2× input tokens, floored at 64
            # for short clips and ceilinged at the configured max_tokens
            # (still acts as the hard safety cap against runaway).
            #
            # Reusing self._gen_config when the dynamic budget happens to
            # equal the configured cap preserves the v0.7.1 micro-opt of
            # not rebuilding GenerationConfig per call. Otherwise we copy
            # + override — sub-millisecond cost, negligible vs decode.
            target_max = max(64, min(int(input_len * 1.2), self._max_tokens))
            if target_max == self._gen_config.max_new_tokens:
                gen_cfg = self._gen_config
            else:
                from transformers import GenerationConfig
                gen_cfg = GenerationConfig(
                    **{**self._gen_config.to_dict(),
                       "max_new_tokens": target_max},
                )
            # v0.7.1: reuse pre-built prefix KV cache for the static
            # system block. generate() infers cached length from
            # past_key_values.get_seq_length() and only forwards the
            # new (user + assistant_prompt + generated) tokens.
            # deepcopy is required — DynamicCache mutates during decode,
            # so each call needs its own copy to keep the prefix pristine.
            pkv = copy.deepcopy(self._prefix_cache)
            with self._torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    past_key_values=pkv,
                    generation_config=gen_cfg,
                )
            new_tokens = outputs[0][input_len:]
            n_new = int(new_tokens.shape[-1])
            # v0.7.2: truncation detection. If generate() hit the budget
            # AND the last emitted token isn't the chat terminator
            # (<|im_end|> for Qwen), the polish output is cut off mid-
            # sentence. Falling back to the raw ASR text is strictly
            # better than pasting a truncated sentence.
            last_token = int(new_tokens[-1].item()) if n_new > 0 else -1
            if n_new >= target_max and last_token != self._pad_token_id:
                print(f"[stt] polish truncated at {target_max} tok "
                      f"(input {input_len} tok); using raw ASR text",
                      file=sys.stderr, flush=True)
                return text
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
        # torch is importable here — if it weren't, we'd be in the
        # ImportError branch above. Use isinstance over string-typed class
        # name compare so future torch renames/wraps don't silently break
        # OOM detection.
        try:
            import torch as _torch
            oom_cls = getattr(_torch.cuda, "OutOfMemoryError", None)
        except Exception:
            _torch = None
            oom_cls = None
        is_oom = (
            (oom_cls is not None and isinstance(e, oom_cls))
            or "out of memory" in msg.lower()
        )
        is_dll = isinstance(e, OSError) and any(
            s in msg.lower() for s in ("dll", "cudart", "cudnn", "cublas")
        )
        if is_oom:
            # Free any partial allocation before the next from_pretrained
            # (e.g. ASR backend init) tries to allocate. Without this, the
            # abandoned-but-not-yet-GC'd tensors hold CUDA memory until
            # Python's GC runs, cascading into a second OOM on the ASR
            # path even though the polish-side allocation is logically dead.
            if _torch is not None:
                try:
                    _torch.cuda.empty_cache()
                except Exception:
                    pass
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
