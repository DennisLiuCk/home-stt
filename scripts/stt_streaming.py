"""Press-time encoder pipelining framework (v0.8.0, shipped DISABLED).

Extracted from stt-daemon.py per A4 review recommendation. Encapsulates
all encoder worker state, threading, and the streaming/batch routing
logic behind an EncoderPipeline class so the daemon only sees a thin API.

Background: the goal was 50% release-to-text latency reduction by running
the ASR encoder in a background thread while the user holds the trigger.
Bench result: ~3% on RTX 5080 + Qwen3-ASR-0.6B (decoder is 95% of time).
Shipped disabled; framework preserved for future decoder-side speedups.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger("stt.streaming")

# Config constants (overridable via stt_config.py apply_to_module)
ENCODER_PIPELINING           = False
ENCODER_CHUNK_SEC            = 5.0
ENCODER_QUEUE_MAX            = 200
ENCODER_FINALIZE_TIMEOUT     = 8.0
ENCODER_FAILURE_BUDGET       = 3
ENCODER_SILENCE_FALLBACK_SEC = 2.0


class EncoderPipeline:
    """Manages press-time encoder pipelining state and worker thread.

    One instance per daemon lifetime. Methods are called from:
      - _on_press  → reset()
      - _audio_callback → on_chunk()
      - _on_release → signal_stop()
      - _transcribe_and_emit → try_streaming() / abort()
    """

    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate

        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=ENCODER_QUEUE_MAX)
        self._thread: threading.Thread | None = None
        self._handle: Any = None
        self._stop_event = threading.Event()

        self.active = False
        self.failed = False
        self.consecutive_failures = 0
        self.use_batch_fallback = False
        self.silence_run_samples = 0
        self.residual_samples: np.ndarray | None = None

    def reset(self) -> None:
        """Reset per-recording state at press time.

        consecutive_failures is intentionally NOT reset — it persists
        across utterances so N back-to-back failures suppress streaming.

        If a worker from the previous utterance is still alive (reachable
        via the busy-drop path — a release's signal_stop() set the stop
        event but _transcribe_and_emit returned early without joining),
        signal its OLD event object so it drains and exits. We then install
        FRESH event + queue objects below; the worker captured the old ones
        at spawn (see _worker), so it can neither consume the new recording's
        stop signal nor drain/poison its queue. No join here: reset() runs
        under _st.lock and a join could back-pressure the 50 ms audio
        callback. A leaked worker's stale writes are blocked by the
        `self._thread is me` guard in _worker.
        """
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.set()
        self.active = False
        self.failed = False
        self.use_batch_fallback = False
        self.silence_run_samples = 0
        self.residual_samples = None
        self._handle = None
        self._thread = None
        self._stop_event = threading.Event()
        self._queue = queue.Queue(maxsize=ENCODER_QUEUE_MAX)

    def track_silence(self, chunk: np.ndarray, is_silent: bool) -> None:
        """Update silence tracking counters. Called under _st.lock."""
        if is_silent:
            self.silence_run_samples += chunk.shape[0]
            if (self.silence_run_samples >=
                    self._sample_rate * ENCODER_SILENCE_FALLBACK_SEC
                    and not self.use_batch_fallback):
                self.use_batch_fallback = True
        else:
            self.silence_run_samples = 0

    def on_chunk(
        self, chunk: np.ndarray, is_silent: bool, backend: Any,
    ) -> None:
        """Encoder spawn + queue push. Called OUTSIDE _st.lock."""
        should_spawn = (
            not self.active
            and ENCODER_PIPELINING
            and backend is not None
            and backend.supports_streaming()
            and self.consecutive_failures < ENCODER_FAILURE_BUDGET
            and not self.use_batch_fallback
        )

        if should_spawn:
            try:
                self._handle = backend.start_encoder()
                self._stop_event.clear()
                self._thread = threading.Thread(
                    target=self._worker,
                    args=(self._handle, backend, self._queue, self._stop_event),
                    daemon=True,
                )
                self._thread.start()
                self.active = True
            except Exception as e:
                logger.warning(f"encoder spawn failed: {type(e).__name__}: {e}; "
                               f"will use batch path")
                self.failed = True
                self.consecutive_failures += 1
                return

        if self.active:
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                self.use_batch_fallback = True

    def signal_stop(self) -> None:
        """Signal the encoder worker to drain and exit."""
        if self.active:
            self._stop_event.set()

    def abort(self, backend: Any) -> None:
        """Abort the encoder — join worker, release GPU handle."""
        if not self.active:
            return
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._handle is not None and backend is not None:
            try:
                backend.abort(self._handle)
            except Exception as e:
                logger.warning(f"encoder abort raised (ignored): "
                               f"{type(e).__name__}: {e}")
        self.active = False

    def try_streaming(self, backend: Any) -> tuple[str, str, float, str] | None:
        """Attempt to finalize via the streaming path.

        Returns (raw_text, language, elapsed, "stream") on success,
        or None if streaming was not used or failed (caller should batch).
        On failure, sets self.failed and increments consecutive_failures.
        """
        if not self.active or self.failed or self.use_batch_fallback:
            return None

        # Join worker
        t0 = time.time()
        if self._thread is not None:
            self._thread.join(timeout=ENCODER_FINALIZE_TIMEOUT)
        join_elapsed = time.time() - t0

        if self._thread is not None and self._thread.is_alive():
            logger.warning(f"encoder join timed out after "
                           f"{join_elapsed:.1f}s — falling back to batch path")
            self._stop_event.set()
            if self._handle is not None and backend is not None:
                try:
                    backend.abort(self._handle)
                except Exception:
                    pass
            self.active = False
            self.failed = True
            self.consecutive_failures += 1
            return None

        # Re-check failure flags after join
        if self.failed or self.use_batch_fallback:
            return None

        tail = self.residual_samples
        if tail is None:
            tail = np.zeros(0, dtype=np.float32)

        t0 = time.time()
        try:
            raw, language = backend.finalize(self._handle, tail)
            elapsed = time.time() - t0
            self.consecutive_failures = 0
            self.active = False
            return (raw, language, elapsed, "stream")
        except Exception as e:
            logger.warning(f"encoder finalize raised "
                           f"{type(e).__name__}: {e}; falling back to batch")
            try:
                backend.abort(self._handle)
            except Exception:
                pass
            self.failed = True
            self.consecutive_failures += 1
            self.active = False
            return None

    def _worker(self, handle: Any, backend: Any,
                q: queue.Queue, stop_event: threading.Event) -> None:
        """Background thread: drain queue into chunk-sized slabs and push
        through the backend encoder.

        Captures its own queue + stop_event at spawn (instead of reading
        self._queue / self._stop_event live each iteration) so a subsequent
        reset() at the next press can install fresh objects without this
        worker racing the new recording. Shared-state writes (residual_samples
        / failed / consecutive_failures) are gated on still being the active
        worker (self._thread is me), so a worker retired by a later reset()
        can never overwrite the next recording's state."""
        me = threading.current_thread()
        chunk_size = int(self._sample_rate * ENCODER_CHUNK_SEC)
        accumulator: list[np.ndarray] = []
        accumulated_n = 0
        try:
            while not stop_event.is_set():
                try:
                    chunk = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                accumulator.append(chunk)
                accumulated_n += chunk.shape[0]
                if accumulated_n >= chunk_size:
                    concat = np.concatenate(accumulator, axis=0).flatten().astype(np.float32)
                    slab = concat[:chunk_size]
                    backend.push_chunk(handle, slab)
                    if concat.shape[0] > chunk_size:
                        accumulator = [concat[chunk_size:]]
                        accumulated_n = concat.shape[0] - chunk_size
                    else:
                        accumulator = []
                        accumulated_n = 0
            # Drain remaining queue items
            while True:
                try:
                    chunk = q.get_nowait()
                except queue.Empty:
                    break
                accumulator.append(chunk)
            if accumulator:
                residual = np.concatenate(accumulator, axis=0).flatten().astype(np.float32)
            else:
                residual = np.zeros(0, dtype=np.float32)
            # Only publish if still the active worker — a worker retired by a
            # later reset() must not clobber the new recording's residual.
            if self._thread is me:
                self.residual_samples = residual
        except Exception as e:
            logger.warning(f"encoder worker crashed: {type(e).__name__}: {e}")
            if self._thread is me:
                self.failed = True
                self.consecutive_failures += 1
