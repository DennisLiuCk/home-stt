"""Tests for stt_state — daemon state file write/read/cleanup + PID detection."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import stt_state


@pytest.fixture(autouse=True)
def _isolate_state_file(tmp_path, monkeypatch):
    """Redirect STATE_FILE to a temp dir so tests don't pollute real state."""
    fake_state = tmp_path / "stt-daemon-state.json"
    monkeypatch.setattr(stt_state, "STATE_FILE", fake_state)
    # Reset PID cache between tests
    stt_state._pid_cache_ts = 0.0
    stt_state._pid_cache_alive = False
    yield fake_state


class TestWriteState:
    def test_creates_file(self, _isolate_state_file):
        sf = _isolate_state_file
        assert not sf.exists()
        stt_state.write_state(stt_state.IDLE)
        assert sf.exists()

    def test_state_field(self, _isolate_state_file):
        stt_state.write_state(stt_state.RECORDING)
        data = json.loads(_isolate_state_file.read_text(encoding="utf-8"))
        assert data["state"] == "recording"

    def test_timestamp_recent(self, _isolate_state_file):
        before = time.time()
        stt_state.write_state(stt_state.IDLE)
        after = time.time()
        data = json.loads(_isolate_state_file.read_text(encoding="utf-8"))
        assert before <= data["ts"] <= after

    def test_edit_mode(self, _isolate_state_file):
        stt_state.write_state(stt_state.RECORDING, edit_mode=True)
        data = json.loads(_isolate_state_file.read_text(encoding="utf-8"))
        assert data["edit_mode"] is True

    def test_last_text_included(self, _isolate_state_file):
        stt_state.write_state(stt_state.IDLE, last_text="hello")
        data = json.loads(_isolate_state_file.read_text(encoding="utf-8"))
        assert data["last_text"] == "hello"

    def test_last_text_truncated(self, _isolate_state_file):
        long_text = "x" * 500
        stt_state.write_state(stt_state.IDLE, last_text=long_text)
        data = json.loads(_isolate_state_file.read_text(encoding="utf-8"))
        assert len(data["last_text"]) == 200

    def test_last_text_omitted_when_none(self, _isolate_state_file):
        stt_state.write_state(stt_state.IDLE)
        data = json.loads(_isolate_state_file.read_text(encoding="utf-8"))
        assert "last_text" not in data

    def test_last_lang(self, _isolate_state_file):
        stt_state.write_state(stt_state.IDLE, last_lang="zh")
        data = json.loads(_isolate_state_file.read_text(encoding="utf-8"))
        assert data["last_lang"] == "zh"

    def test_overwrites_previous(self, _isolate_state_file):
        stt_state.write_state(stt_state.RECORDING)
        stt_state.write_state(stt_state.PROCESSING)
        data = json.loads(_isolate_state_file.read_text(encoding="utf-8"))
        assert data["state"] == "processing"


class TestReadState:
    def test_returns_none_when_no_file(self, _isolate_state_file, monkeypatch):
        monkeypatch.setattr(stt_state, "_daemon_alive", lambda: False)
        result = stt_state.read_state()
        assert result is not None
        assert result["state"] == "stopped"

    def test_reads_written_state(self, _isolate_state_file, monkeypatch):
        stt_state.write_state(stt_state.IDLE)
        result = stt_state.read_state()
        assert result["state"] == "idle"

    def test_stale_idle_with_dead_daemon(self, _isolate_state_file, monkeypatch):
        stt_state.write_state(stt_state.IDLE)
        # Make file look old
        data = json.loads(_isolate_state_file.read_text(encoding="utf-8"))
        data["ts"] = time.time() - 30
        _isolate_state_file.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(stt_state, "_daemon_alive", lambda: False)
        result = stt_state.read_state()
        assert result["state"] == "stopped"

    def test_stale_idle_with_alive_daemon(self, _isolate_state_file, monkeypatch):
        stt_state.write_state(stt_state.IDLE)
        data = json.loads(_isolate_state_file.read_text(encoding="utf-8"))
        data["ts"] = time.time() - 30
        _isolate_state_file.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(stt_state, "_daemon_alive", lambda: True)
        result = stt_state.read_state()
        assert result["state"] == "idle"

    def test_stale_recording_with_dead_daemon(self, _isolate_state_file, monkeypatch):
        """Key regression: daemon crash mid-recording → state file stuck at recording."""
        stt_state.write_state(stt_state.RECORDING)
        data = json.loads(_isolate_state_file.read_text(encoding="utf-8"))
        data["ts"] = time.time() - 10
        _isolate_state_file.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(stt_state, "_daemon_alive", lambda: False)
        result = stt_state.read_state()
        assert result["state"] == "stopped"

    def test_fresh_recording_trusted(self, _isolate_state_file):
        """A fresh recording state (<5s) should be trusted without PID check."""
        stt_state.write_state(stt_state.RECORDING)
        result = stt_state.read_state()
        assert result["state"] == "recording"

    def test_pid_cache_avoids_repeated_checks(self, _isolate_state_file, monkeypatch):
        call_count = 0
        def counting_alive():
            nonlocal call_count
            call_count += 1
            return True
        monkeypatch.setattr(stt_state, "_daemon_alive", counting_alive)
        stt_state.write_state(stt_state.IDLE)
        data = json.loads(_isolate_state_file.read_text(encoding="utf-8"))
        data["ts"] = time.time() - 30
        _isolate_state_file.write_text(json.dumps(data), encoding="utf-8")
        stt_state.read_state()
        stt_state.read_state()
        stt_state.read_state()
        assert call_count == 1


class TestCleanup:
    def test_removes_file(self, _isolate_state_file):
        stt_state.write_state(stt_state.IDLE)
        assert _isolate_state_file.exists()
        stt_state.cleanup()
        assert not _isolate_state_file.exists()

    def test_noop_when_no_file(self, _isolate_state_file):
        stt_state.cleanup()  # should not raise
