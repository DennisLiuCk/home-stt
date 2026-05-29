"""Shared CUDA init-error classification.

build_backend_with_fallback (stt_backends) and build_polisher (text_polisher)
both need to distinguish a CUDA out-of-memory failure from a CUDA DLL-load
failure from a generic init error when a from_pretrained / model load raises.
The logic was duplicated verbatim in both modules; this is the single source.
"""
from __future__ import annotations


def classify_cuda_init_error(e: BaseException) -> tuple[bool, bool, object | None]:
    """Classify a backend/polisher init exception.

    Returns (is_oom, is_dll, torch_module_or_None):
      is_oom — a ``torch.cuda.OutOfMemoryError`` instance OR ``"out of memory"``
               appears in ``str(e)``.
      is_dll — an ``OSError`` mentioning a CUDA runtime DLL token
               (dll / cudart / cudnn / cublas).
      torch  — the imported torch module (or None if unimportable), so the
               OOM caller can ``torch.cuda.empty_cache()`` without re-importing.

    isinstance beats a string class-name compare so a future torch rename/wrap
    cannot silently break OOM detection.
    """
    msg = str(e)
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
    return is_oom, is_dll, _torch
