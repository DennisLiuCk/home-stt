"""Vendor wrapper around `qwen_asr.Qwen3ASRModel` exposing encoder/decoder
as separable steps for v0.8.0 press-time pipelining.

Why this exists
---------------
`qwen_asr.Qwen3ASRModel.transcribe()` is monolithic: it runs encoder +
decoder in one call. For the daemon's hold-to-talk UX, we want to run the
encoder on each ~5s slab WHILE the user is still holding the trigger,
then on release run only the decoder over the concatenated hidden states.
That cuts ~50% off the post-release wait on long utterances (verified by
the v0.8.0 spike, see `tmp/spike_torch_encoder.py`).

The encoder API exists upstream — `Qwen3ASRThinkerForConditionalGeneration
.get_audio_features()` at
`site-packages/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py:1099`
— but is NOT surfaced through the public `Qwen3ASRModel` wrapper. The
wrapper's `generate()` flow funnels through forward(), which unconditionally
re-encodes from `input_features` (modeling_qwen3_asr.py:1195).

This file works around that with a controlled monkey-patch of
`get_audio_features` during the generate call, scoped by try/finally.

Two-call design:

    handle = wrapper.start_encoder()          # cheap; just allocates dataclass
    wrapper.encode_chunk(handle, slab_5s)     # repeat per 5s slab, on worker thread
    ...
    text, lang = wrapper.finalize_with_features(handle, tail)
                                              # encodes tail, concats slabs,
                                              # runs decoder, returns parsed text

`abort(handle)` releases GPU memory if the daemon decides to fall back to
the batch path without calling finalize.

Thread safety
-------------
The daemon's `_processing` semaphore guarantees only one `finalize_with_
features()` runs at a time on a given backend instance. Within a single
recording, `encode_chunk` is called by the encoder worker thread and
`finalize_with_features` by the transcribe thread — they are NOT
concurrent (worker has exited before finalize fires).

Upgrade hazard
--------------
`qwen-asr` is pre-1.0 (no API-stability guarantees) and we touch the
internal `self.model.thinker` path. If upstream refactors `Qwen3ASRModel`
to hide `.thinker` or rename `get_audio_features`, this wrapper fails at
construct time → `_Qwen3TorchImpl.__init__` catches and falls back to the
non-streaming `Qwen3ASRModel`. Version-check via `hasattr` at import time;
streaming silently falls back if attributes are missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel, parse_asr_output

# Constants
_SAMPLE_RATE = 16000  # matches stt-daemon.py SAMPLE_RATE; daemon's contract


@dataclass
class TorchEncoderHandle:
    """Opaque (to daemon) state for one in-flight encoder pipelining session.

    raw_chunks: list of per-slab audio arrays. Needed at finalize time to
        rebuild the full-audio processor pass that gives correct
        input_ids + audio-placeholder-token count for the decoder.
    feature_slabs: list of per-slab encoder hidden states from
        `thinker.get_audio_features()`. Concatenated along dim=0 (time axis)
        at finalize time — the empirically-verified configuration that
        matched batch encoding to Levenshtein ≤2 on real zh-en speech
        (see tmp/spike_torch_encoder.py).
    """
    raw_chunks: list[np.ndarray] = field(default_factory=list)
    feature_slabs: list[torch.Tensor] = field(default_factory=list)


class StreamingQwen3ASRModel(Qwen3ASRModel):
    """Qwen3ASRModel + encoder/decoder split for press-time pipelining.

    `transcribe()` is inherited unchanged — non-streaming callers see
    identical behaviour. The streaming methods are additive.
    """

    # --- compatibility probe ----------------------------------------------
    # If the underlying model object doesn't have the .thinker.get_audio_features
    # surface we depend on, every streaming method will refuse to run and the
    # daemon's fallback path kicks in. Public so _Qwen3TorchImpl can check at
    # init time + log a one-shot "streaming disabled" warning.
    @classmethod
    def _streaming_supported(cls, model_obj: Any) -> bool:
        thinker = getattr(getattr(model_obj, "model", None), "thinker", None)
        if thinker is None:
            return False
        return callable(getattr(thinker, "get_audio_features", None))

    # --- daemon-facing API ------------------------------------------------

    def start_encoder(self) -> TorchEncoderHandle:
        """Allocate a fresh handle. Cheap — no GPU work."""
        return TorchEncoderHandle()

    def encode_chunk(self, handle: TorchEncoderHandle, samples: np.ndarray) -> None:
        """Run the encoder forward on one ~ENCODER_CHUNK_SEC slab. Stores
        the raw audio (for later full-audio scaffolding) AND the resulting
        encoder hidden states on `handle`.

        Called by the daemon's `_encoder_worker` thread. MUST NOT raise
        under normal conditions — worker wraps in try/except and falls
        back if we do raise. Slab is expected to be 1-D float32 @ 16 kHz.
        """
        text_prompt = self._build_text_prompt(context="", force_language=None)
        # processor pad=True so the input_features tensor has consistent
        # leading dim shape even though we're feeding one audio at a time.
        inputs = self.processor(
            text=[text_prompt], audio=[samples.astype(np.float32)],
            return_tensors="pt", padding=True,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        with torch.no_grad():
            slab = self.model.thinker.get_audio_features(
                inputs["input_features"],
                feature_attention_mask=inputs["feature_attention_mask"],
            )
        # Keep raw chunks AND features. The decoder pass at finalize needs
        # to re-run the processor on the FULL audio (concatenated raw) to
        # produce the right input_ids — number of audio placeholder tokens
        # depends on total audio length.
        handle.raw_chunks.append(samples.astype(np.float32))
        handle.feature_slabs.append(slab)

    def finalize_with_features(
        self,
        handle: TorchEncoderHandle,
        tail_samples: np.ndarray,
    ) -> tuple[str, str]:
        """Encode `tail_samples` as the final slab (if non-empty), concat
        all hidden states along time axis, run the decoder against the
        full sequence, return (text, language) — text post-processed by
        `parse_asr_output` to strip the "language X<asr_text>" prompt
        prefix that the model emits verbatim when force_language=None.

        Empty audio → ("", ""). Same contract as
        `Qwen3ASRModel.transcribe()`'s single-item result.
        """
        if tail_samples is not None and len(tail_samples) > 0:
            self.encode_chunk(handle, tail_samples)
        if not handle.feature_slabs:
            return "", ""

        # Rebuild full-audio inputs for text-prompt scaffolding. The
        # processor's input_ids embeds N audio-placeholder tokens where
        # N = encoded-feature-length(full_audio). For our concat'd slabs
        # to align with the placeholders, we MUST pass the same full
        # audio (raw concat) through the processor.
        full_audio = np.concatenate(handle.raw_chunks, axis=0).astype(np.float32)
        text_prompt = self._build_text_prompt(context="", force_language=None)
        inputs = self.processor(
            text=[text_prompt], audio=[full_audio],
            return_tensors="pt", padding=True,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        # Concat hidden states along time axis (dim=0). See
        # tmp/spike_torch_encoder.py:135 for why dim=0 (not dim=1 — that
        # would double hidden_dim and produce decoder garbage like
        # "假設假設假設…", the bug that initially looked like a chunked-
        # encoding failure but was a shape-axis mistake).
        concat_features = torch.cat(handle.feature_slabs, dim=0)

        # Monkey-patch the encoder to return our pre-computed features.
        # Scoped strictly to this call via try/finally. Thread-safety
        # delegated to the daemon (single _processing slot at a time).
        thinker = self.model.thinker
        orig = thinker.get_audio_features

        def _patched(input_features, feature_attention_mask=None,
                     audio_feature_lengths=None):
            return concat_features

        thinker.get_audio_features = _patched
        try:
            with torch.no_grad():
                text_ids = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens,
                )
            decoded = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            raw = decoded[0] if decoded else ""
            # parse_asr_output strips the literal "language X<asr_text>"
            # prefix that the model emits when no language is forced —
            # matches what Qwen3ASRModel.transcribe() returns externally.
            lang, txt = parse_asr_output(raw, user_language=None)
            return txt, lang
        finally:
            # Restore. Note: this leaves an instance-level attribute that
            # shadows the class method even after restoration. Both
            # reference the same bound method object, so semantically
            # identical. Could `del thinker.get_audio_features` to drop
            # the instance attr fully, but the shadow is harmless and the
            # del path has an edge case if the upstream class ever defines
            # an instance-level attribute itself.
            thinker.get_audio_features = orig

    def abort(self, handle: TorchEncoderHandle) -> None:
        """Drop references to GPU tensors on `handle`. Idempotent.
        Daemon calls this when falling back to the batch path (encoder
        worker crashed, finalize timed out, mid-utterance silence
        detected). MUST NOT raise."""
        try:
            handle.raw_chunks.clear()
            handle.feature_slabs.clear()
        except Exception:
            pass
