"""Tests for build_backend_with_fallback + classify_cuda_init_error.

These paths previously had zero coverage. All mock build_backend / synthesise
exceptions — no GPU, no model load. conftest puts scripts/ on sys.path.
"""
from __future__ import annotations

import logging

import pytest

import stt_backends
from stt_cuda_errors import classify_cuda_init_error


class TestClassifyCudaInitError:
    def test_oom_by_message(self):
        is_oom, is_dll, _t = classify_cuda_init_error(
            RuntimeError("CUDA out of memory. Tried to allocate ..."))
        assert is_oom is True
        assert is_dll is False

    def test_dll_oserror(self):
        is_oom, is_dll, _t = classify_cuda_init_error(
            OSError("[WinError 126] cudnn64_9.dll not found"))
        assert is_oom is False
        assert is_dll is True

    def test_dll_token_in_non_oserror_is_not_dll(self):
        # is_dll requires an OSError; a ValueError mentioning a DLL token isn't.
        _oom, is_dll, _t = classify_cuda_init_error(ValueError("missing cublas"))
        assert is_dll is False

    def test_generic_error_is_neither(self):
        is_oom, is_dll, _t = classify_cuda_init_error(ValueError("nope"))
        assert is_oom is False
        assert is_dll is False


class TestBuildBackendFallback:
    def test_non_whisper_backend_substitutes_large_v3_turbo(self, monkeypatch):
        attempts = []

        def fake_build(name, model, sr):
            attempts.append((name, model))
            if name != "faster-whisper":
                raise RuntimeError("no cuda")
            return f"{name}:{model}"

        monkeypatch.setattr(stt_backends, "build_backend", fake_build)
        result = stt_backends.build_backend_with_fallback(
            "qwen3-asr", "Qwen/Qwen3-ASR-0.6B", 16000)
        assert result == "faster-whisper:large-v3-turbo"
        # non-faster-whisper backend → fallback is just large-v3-turbo once.
        fw = [a for a in attempts if a[0] == "faster-whisper"]
        assert fw == [("faster-whisper", "large-v3-turbo")]

    def test_faster_whisper_custom_model_substitutes_not_retried(
            self, monkeypatch, caplog):
        attempts = []

        def fake_build(name, model, sr):
            attempts.append((name, model))
            if model != "large-v3-turbo":
                raise RuntimeError("bad model")
            return f"{name}:{model}"

        monkeypatch.setattr(stt_backends, "build_backend", fake_build)
        with caplog.at_level(logging.WARNING, logger="stt"):
            result = stt_backends.build_backend_with_fallback(
                "faster-whisper", "medium", 16000)
        assert result == "faster-whisper:large-v3-turbo"
        # 'medium' attempted once (the initial try) — NOT redundantly retried.
        assert attempts.count(("faster-whisper", "medium")) == 1
        # The swap must be visible in the log (not silent).
        assert "large-v3-turbo" in caplog.text.lower()

    def test_all_fail_exits(self, monkeypatch):
        def fake_build(name, model, sr):
            raise RuntimeError("everything is broken")

        monkeypatch.setattr(stt_backends, "build_backend", fake_build)
        with pytest.raises(SystemExit):
            stt_backends.build_backend_with_fallback("qwen3-asr", "m", 16000)
