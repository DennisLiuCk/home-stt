"""Tests for stt_config — TOML config loading, env overrides, key parsing."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import stt_config


class TestLoadDefaults:
    def test_returns_all_default_keys(self, tmp_path, monkeypatch):
        monkeypatch.setattr(stt_config, "config_path", lambda: tmp_path / "nope.toml")
        cfg = stt_config.load_config()
        for key in stt_config._DEFAULTS:
            assert key in cfg

    def test_default_polish_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(stt_config, "config_path", lambda: tmp_path / "nope.toml")
        cfg = stt_config.load_config()
        assert cfg["polish_enabled"] is True

    def test_default_stt_backend_is_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(stt_config, "config_path", lambda: tmp_path / "nope.toml")
        cfg = stt_config.load_config()
        assert cfg["stt_backend"] is None


class TestEnvOverride:
    @pytest.fixture(autouse=True)
    def _no_config_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(stt_config, "config_path", lambda: tmp_path / "nope.toml")

    def test_bool_env(self, monkeypatch):
        monkeypatch.setenv("HOME_STT_POLISH_ENABLED", "false")
        cfg = stt_config.load_config()
        assert cfg["polish_enabled"] is False

    def test_int_env(self, monkeypatch):
        monkeypatch.setenv("HOME_STT_SAMPLE_RATE", "48000")
        cfg = stt_config.load_config()
        assert cfg["sample_rate"] == 48000

    def test_float_env(self, monkeypatch):
        monkeypatch.setenv("HOME_STT_BEEP_VOLUME", "0.5")
        cfg = stt_config.load_config()
        assert cfg["beep_volume"] == 0.5

    def test_string_env(self, monkeypatch):
        monkeypatch.setenv("HOME_STT_STT_BACKEND", "faster-whisper")
        cfg = stt_config.load_config()
        assert cfg["stt_backend"] == "faster-whisper"

    def test_key_set_env(self, monkeypatch):
        monkeypatch.setenv("HOME_STT_TRIGGER_KEYS", "alt_r, ctrl_r")
        cfg = stt_config.load_config()
        assert cfg["trigger_keys"] == ["alt_r", "ctrl_r"]


class TestTomlFile:
    def test_file_overrides_defaults(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            'stt_backend = "faster-whisper"\n'
            "beep_volume = 0.8\n"
            "beeps_enabled = false\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(stt_config, "config_path", lambda: cfg_file)
        cfg = stt_config.load_config()
        assert cfg["stt_backend"] == "faster-whisper"
        assert cfg["beep_volume"] == 0.8
        assert cfg["beeps_enabled"] is False

    def test_sectioned_toml(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            "[audio]\n"
            "sample_rate = 48000\n"
            "beep_volume = 0.3\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(stt_config, "config_path", lambda: cfg_file)
        cfg = stt_config.load_config()
        assert cfg["sample_rate"] == 48000
        assert cfg["beep_volume"] == 0.3

    def test_env_beats_file(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('stt_backend = "mlx-whisper"\n', encoding="utf-8")
        monkeypatch.setattr(stt_config, "config_path", lambda: cfg_file)
        monkeypatch.setenv("HOME_STT_STT_BACKEND", "faster-whisper")
        cfg = stt_config.load_config()
        assert cfg["stt_backend"] == "faster-whisper"

    def test_trigger_keys_from_toml(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            'trigger_keys = ["alt_r", "ctrl_r"]\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(stt_config, "config_path", lambda: cfg_file)
        cfg = stt_config.load_config()
        assert cfg["trigger_keys"] == ["alt_r", "ctrl_r"]


class TestParseKey:
    def test_named_key(self):
        from pynput.keyboard import Key
        assert stt_config._parse_key("alt_r") == Key.alt_r

    def test_single_char(self):
        assert stt_config._parse_key("a") == "a"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown key"):
            stt_config._parse_key("not_a_real_key")

    def test_parse_key_set_none(self):
        assert stt_config._parse_key_set(None) is None

    def test_parse_key_set_empty(self):
        assert stt_config._parse_key_set([]) == set()

    def test_parse_key_set_multiple(self):
        from pynput.keyboard import Key
        result = stt_config._parse_key_set(["alt_r", "f13"])
        assert result == {Key.alt_r, Key.f13}


class TestInitConfig:
    def test_creates_file(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "sub" / "config.toml"
        monkeypatch.setattr(stt_config, "config_path", lambda: cfg_file)
        path = stt_config.init_config()
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "home-stt configuration" in content
        assert "stt_backend" in content

    def test_does_not_overwrite(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("existing", encoding="utf-8")
        monkeypatch.setattr(stt_config, "config_path", lambda: cfg_file)
        stt_config.init_config()
        assert cfg_file.read_text() == "existing"


class TestUpdateTriggerKeys:
    def test_creates_file_if_missing(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        monkeypatch.setattr(stt_config, "config_path", lambda: cfg_file)
        stt_config.update_trigger_keys(trigger=["alt_r"])
        assert cfg_file.exists()
        content = cfg_file.read_text(encoding="utf-8")
        assert 'trigger_keys = ["alt_r"]' in content

    def test_updates_commented_line(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            '# trigger_keys = ["alt_r"]\n'
            '# edit_trigger_keys = ["f13"]\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(stt_config, "config_path", lambda: cfg_file)
        stt_config.update_trigger_keys(trigger=["f14"], edit_trigger=["f15"])
        content = cfg_file.read_text(encoding="utf-8")
        assert 'trigger_keys = ["f14"]' in content
        assert 'edit_trigger_keys = ["f15"]' in content
        assert "# trigger_keys" not in content
        assert "# edit_trigger_keys" not in content

    def test_updates_existing_value(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            'trigger_keys = ["ctrl_r"]\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(stt_config, "config_path", lambda: cfg_file)
        stt_config.update_trigger_keys(trigger=["alt_r"])
        content = cfg_file.read_text(encoding="utf-8")
        assert 'trigger_keys = ["alt_r"]' in content
        assert "ctrl_r" not in content

    def test_none_skips_key(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            'trigger_keys = ["ctrl_r"]\n'
            '# edit_trigger_keys = ["f13"]\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(stt_config, "config_path", lambda: cfg_file)
        stt_config.update_trigger_keys(trigger=None, edit_trigger=["f15"])
        content = cfg_file.read_text(encoding="utf-8")
        assert 'trigger_keys = ["ctrl_r"]' in content
        assert 'edit_trigger_keys = ["f15"]' in content

    def test_roundtrip_produces_valid_toml(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        monkeypatch.setattr(stt_config, "config_path", lambda: cfg_file)
        stt_config.update_trigger_keys(trigger=["alt_r"], edit_trigger=["f13"])
        if stt_config.tomllib is not None:
            with open(cfg_file, "rb") as f:
                parsed = stt_config.tomllib.load(f)
            assert parsed["trigger_keys"] == ["alt_r"]
            assert parsed["edit_trigger_keys"] == ["f13"]


class TestMicDevice:
    def test_default_is_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(stt_config, "config_path", lambda: tmp_path / "nope.toml")
        cfg = stt_config.load_config()
        assert cfg["mic_device"] is None

    def test_string_from_toml(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('mic_device = "Yeti Nano"\n', encoding="utf-8")
        monkeypatch.setattr(stt_config, "config_path", lambda: cfg_file)
        cfg = stt_config.load_config()
        assert cfg["mic_device"] == "Yeti Nano"

    def test_int_from_toml(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("mic_device = 3\n", encoding="utf-8")
        monkeypatch.setattr(stt_config, "config_path", lambda: cfg_file)
        cfg = stt_config.load_config()
        assert cfg["mic_device"] == 3

    def test_env_int_coercion(self, tmp_path, monkeypatch):
        monkeypatch.setattr(stt_config, "config_path", lambda: tmp_path / "nope.toml")
        monkeypatch.setenv("HOME_STT_MIC_DEVICE", "5")
        cfg = stt_config.load_config()
        assert cfg["mic_device"] == 5

    def test_env_string(self, tmp_path, monkeypatch):
        monkeypatch.setattr(stt_config, "config_path", lambda: tmp_path / "nope.toml")
        monkeypatch.setenv("HOME_STT_MIC_DEVICE", "Yeti Nano")
        cfg = stt_config.load_config()
        assert cfg["mic_device"] == "Yeti Nano"


class TestKeyToStr:
    def test_pynput_key(self):
        from pynput.keyboard import Key
        assert stt_config._key_to_str(Key.alt_r) == "alt_r"

    def test_char(self):
        assert stt_config._key_to_str("a") == "a"


class TestGenerateDefaultConfig:
    def test_template_is_valid_toml(self, tmp_path):
        content = stt_config.generate_default_config()
        path = tmp_path / "test.toml"
        path.write_text(content, encoding="utf-8")
        if stt_config.tomllib is not None:
            with open(path, "rb") as f:
                parsed = stt_config.tomllib.load(f)
            assert isinstance(parsed, dict)
