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

import logging
import sys
from abc import ABC, abstractmethod

logger = logging.getLogger("stt.polisher")


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


# v0.7.5: voice-edit mode prompt. Bilingual because instruction can be
# either language and selection can be any language. The load-bearing
# sentence is "輸出語言預設與選取的文字相同;若指令明確要求換語言則依
# 指令" — phrasing it as "default unless explicit override" is more
# robust than "always match selection" (loses translation) or "always
# follow instruction" (loses no-explicit-language case). Bilingual
# (zh + en) covers the case where a Chinese-only system prompt + English
# instruction leaks Chinese-flavoured English into the output.
EDIT_PROMPT = (
    "你是文字編輯助手。使用者會給你一段「選取的文字」與一段「指令」。\n"
    "依照指令修改選取的文字,只輸出修改後的結果。\n"
    "輸出語言預設與選取的文字相同;若指令明確要求換語言"
    "(例如「translate to English」「改成中文」「翻成日文」),則依指令。\n"
    "保留原本的英文技術詞彙(commit/push/function 等)與識別字"
    "(_USE_FOO 等),除非指令明確要求翻譯。\n"
    "不解釋、不加引號、不加前綴。\n"
    "\n"
    "You are a text editor. The user gives you SELECTION and INSTRUCTION.\n"
    "Apply the instruction to the selection and output ONLY the modified text.\n"
    "Default output language = selection's language. Switch only if the "
    "instruction explicitly says so.\n"
    "Preserve English tech terms and identifiers verbatim unless the "
    "instruction explicitly translates them.\n"
    "No explanations, no quotes, no prefixes."
)


def _format_edit_user_msg(selection: str, instruction: str) -> str:
    """Build the user-role message for voice-edit. Tag-wrapped
    (<selection>...<instruction>...) so the model sees clear data
    boundaries vs. asking it to compose free-form text.

    Output kept agnostic of which polish backend reads it — same helper
    serves MLX + Torch + any future cloud impl."""
    return (
        f"<selection>\n{selection}\n</selection>\n\n"
        f"<instruction>\n{instruction}\n</instruction>"
    )


def _strip_wrapping_quotes(s: str) -> str:
    """Best-effort guard: some models occasionally wrap output in quotes
    despite the prompt. Strip a single matched pair. Shared across polish +
    edit (both system prompts forbid quoting) and across the MLX + Torch
    backends."""
    if (s.startswith('"') and s.endswith('"')) or \
       (s.startswith("「") and s.endswith("」")):
        return s[1:-1].strip() or s
    return s


class TextPostProcessor(ABC):
    """Polish raw ASR text. Implementations may transform, leave unchanged,
    or fail-safely return the input. Must NEVER raise — failure modes are
    surfaced via the printed `[stt] polish failed: …` line and falling
    back to the input text."""

    @abstractmethod
    def polish(self, text: str) -> str:
        """Return polished text. Return input unchanged on any failure."""

    @abstractmethod
    def edit(self, selection: str, instruction: str) -> str | None:
        """v0.7.5 voice-edit: apply `instruction` to `selection` and
        return the modified text. Return None on ANY failure — caller
        (daemon `_transcribe_and_emit_edit`) treats None as 'abort, do not
        paste anything'. Different failure semantics from polish() — for
        polish, silently returning the input is the right UX (filler
        removal failure → paste the raw transcript). For edit, pasting
        the input back would simulate a no-op that consumed the user's
        edit gesture and the original selection would be replaced with
        itself, which is silently confusing. Explicit None lets the caller
        play a failure beep and abort cleanly."""

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

    def edit(self, selection: str, instruction: str) -> str | None:
        """Voice-edit is meaningless without an LLM (it's the LLM that
        does the editing). Return None so the daemon plays the fail beep
        instead of silently pasting the selection back unchanged. The
        daemon startup log should warn the user if POLISH_ENABLED=False
        but EDIT_TRIGGER_KEYS is set."""
        return None


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

    def _run_generation(self, system_prompt: str, user_msg: str,
                        max_tokens: int) -> str:
        """v0.7.5: shared inference path for polish() and edit(). Builds
        messages → chat template → generate → decode → quote-strip.
        Raises on backend failure — caller decides what to log/return."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        response = self._generate(
            self._model, self._tokenizer, prompt=prompt,
            max_tokens=max_tokens, verbose=False,
        )
        out = _strip_wrapping_quotes((response or "").strip())
        return out

    def polish(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text
        try:
            polished = self._run_generation(
                self._system_prompt,
                _format_polish_user_msg(text),
                self._max_tokens,
            )
            return polished or text  # empty → degrade to raw ASR
        except Exception as e:
            logger.warning("polish failed, returning raw: %s", e)
            return text

    def edit(self, selection: str, instruction: str) -> str | None:
        """v0.7.5 voice-edit on MLX. Uses EDIT_PROMPT instead of the
        polish system prompt. Budget heuristic differs from polish: edit
        can EXPAND (e.g. "expand this", "translate from Chinese to English"
        often grows length), so use 3× selection tokens floored at 256."""
        selection = selection.strip()
        instruction = instruction.strip()
        if not selection or not instruction:
            return None
        user_msg = _format_edit_user_msg(selection, instruction)
        # MLX tokenizer's encode returns a list — len() gives token count.
        selection_tokens = len(self._tokenizer.encode(selection))
        budget = max(256, min(int(selection_tokens * 3.0), self._max_tokens))
        try:
            out = self._run_generation(EDIT_PROMPT, user_msg, budget)
            return out.strip() or None
        except Exception as e:
            logger.warning("edit failed: %s", e)
            return None


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

    def _run_generation(self, system_prompt: str, user_msg: str,
                        budget_fn, past_key_values=None) -> tuple[str, int, int, int]:
        """v0.7.5: shared inference path for polish() and edit(). Returns
        (text_post_quote_strip, input_len, n_new_tokens, last_token_id).
        Caller computes the budget from input_len via `budget_fn` and
        decides truncation/empty handling. Raises on backend failure —
        caller picks the right log+return shape.

        past_key_values=None means no prefix cache (cold inference path —
        full system prompt prefills on every call). polish() passes a
        deep-copy of self._prefix_cache for the pre-warmed common system
        block; edit() passes None because its system prompt is different
        from polish's and we don't pre-warm an edit-side cache (per
        v0.7.5 plan §D — edit is called ~10× less than polish, prefix
        cache cost wouldn't amortise)."""
        # add_special_tokens=False: the chat template already inserts any
        # BOS / role-marker tokens the model expects. For Qwen tokenizers
        # add_bos_token defaults False so this is a no-op today, but
        # Llama-family tokenizers auto-prepend <s> — without this flag,
        # a future POLISH_MODEL swap to Llama-derived weights would
        # silently produce double-BOS prompts and degrade quality with
        # no error.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False,
        ).to(self._device)
        input_len = inputs["input_ids"].shape[-1]
        target_max = budget_fn(input_len)
        # Reusing self._gen_config when the budget happens to equal the
        # configured cap preserves the v0.7.1 micro-opt of not rebuilding
        # GenerationConfig per call. Otherwise copy + override —
        # sub-millisecond cost, negligible vs decode.
        if target_max == self._gen_config.max_new_tokens:
            gen_cfg = self._gen_config
        else:
            from transformers import GenerationConfig
            gen_cfg = GenerationConfig(
                **{**self._gen_config.to_dict(),
                   "max_new_tokens": target_max},
            )
        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                past_key_values=past_key_values,
                generation_config=gen_cfg,
            )
        new_tokens = outputs[0][input_len:]
        n_new = int(new_tokens.shape[-1])
        last_token = int(new_tokens[-1].item()) if n_new > 0 else -1
        text = _strip_wrapping_quotes(self._tokenizer.decode(
            new_tokens, skip_special_tokens=True,
        ).strip())
        return text, input_len, n_new, last_token

    def polish(self, text: str) -> str:
        import copy
        text = text.strip()
        if not text:
            return text
        try:
            # v0.7.2: dynamic max_new_tokens. Polish is a "minimum-edit"
            # task — output length ≈ input length. Budget ~1.2× input
            # tokens, floored at 64 for short clips, ceilinged at the
            # configured max_tokens (hard safety cap against runaway).
            #
            # v0.7.1: reuse pre-built prefix KV cache for the static
            # system block. deepcopy required — DynamicCache mutates
            # during decode, so each call needs its own copy.
            polished, input_len, n_new, last_token = self._run_generation(
                self._system_prompt,
                _format_polish_user_msg(text),
                budget_fn=lambda n: max(64, min(int(n * 1.2), self._max_tokens)),
                past_key_values=copy.deepcopy(self._prefix_cache),
            )
            # v0.7.2: truncation detection. If generate() hit the budget
            # AND the last emitted token isn't the chat terminator
            # (<|im_end|> for Qwen), the polish output is cut off mid-
            # sentence. Falling back to the raw ASR text is strictly
            # better than pasting a truncated sentence.
            target_max = max(64, min(int(input_len * 1.2), self._max_tokens))
            if n_new >= target_max and last_token != self._pad_token_id:
                logger.warning("polish truncated at %d tok (input %d tok); "
                               "using raw ASR text", target_max, input_len)
                return text
            return polished or text  # empty → degrade to raw ASR
        except Exception as e:
            logger.warning("polish failed, returning raw: %s", e)
            return text

    def edit(self, selection: str, instruction: str) -> str | None:
        """v0.7.5 voice-edit on Torch. Uses EDIT_PROMPT instead of polish's
        system prompt, NO prefix cache (different system prompt — would
        need its own cache, not worth amortising for ~10× lower call rate
        than polish per plan §D).

        Budget heuristic differs from polish: edit can EXPAND text
        (e.g. "expand this", "translate from Chinese to English" often
        grows length), so use 3× selection tokens floored at 256, capped
        at self._max_tokens. The 3× ceiling handles "expand" + safety;
        256 floor handles short selections + verbose instructions like
        "translate to Chinese with explanation"."""
        selection = selection.strip()
        instruction = instruction.strip()
        if not selection or not instruction:
            return None
        # Pre-compute selection token count for budget. Tokenize ONCE here
        # then again inside _run_generation for the full prompt — small
        # waste, but selection-only is the right unit for the heuristic
        # (full prompt is system + selection + instruction; only selection
        # bounds the output size).
        selection_tokens = len(
            self._tokenizer.encode(selection, add_special_tokens=False)
        )
        budget = max(256, min(int(selection_tokens * 3.0), self._max_tokens))
        try:
            text, _input_len, n_new, last_token = self._run_generation(
                EDIT_PROMPT,
                _format_edit_user_msg(selection, instruction),
                budget_fn=lambda _: budget,
                past_key_values=None,  # no prefix cache for edit
            )
            # Truncation handling for edit DIFFERS from polish: there's
            # no coherent fallback (raw selection is the OLD text, not the
            # edit output). Decision: log + return truncated result. User
            # can re-trigger if they want.
            if n_new >= budget and last_token != self._pad_token_id:
                logger.warning("edit truncated at %d tok (selection %d tok); "
                               "returning partial result", budget, selection_tokens)
            return text.strip() or None
        except Exception as e:
            logger.warning("edit failed: %s", e)
            return None


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
        logger.warning(
            "polish disabled — required package missing for %s: %s. "
            "Install torch+CUDA and transformers (see README → Windows "
            "安裝步驟), or set POLISH_ENABLED = False to silence this.",
            model_name, e,
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
        from stt_cuda_errors import classify_cuda_init_error
        is_oom, is_dll, _torch = classify_cuda_init_error(e)
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
            logger.warning(
                "polish disabled — CUDA OOM loading %s: %s. "
                "Try POLISH_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct' "
                "(~3 GB VRAM), or set POLISH_ENABLED = False.",
                model_name, e,
            )
        elif is_dll:
            logger.warning(
                "polish disabled — CUDA DLL load failed for %s: %s. "
                "Install nvidia-cudnn-cu12 + nvidia-cublas-cu12 (Windows), "
                "or reinstall torch with the CUDA wheel "
                "(see README -> Windows 安裝步驟).",
                model_name, e,
            )
        else:
            logger.warning(
                "polish disabled — could not initialise %s: %s",
                model_name, e,
            )
        return NoopPolisher()
