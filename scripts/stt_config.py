"""Config file support for home-stt.

Loads user config from TOML (Python 3.11+ `tomllib`, or `tomli` fallback),
merges with code defaults, and provides a `config --init` generator.

Priority (highest wins):  environment variable > config file > code default
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("stt.config")

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


def _config_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming"
        return Path(base) / "home-stt"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "home-stt"


def config_path() -> Path:
    return _config_dir() / "config.toml"


def _parse_key(name: str):
    """Convert a string key name to a pynput Key or character.

    Accepts: 'alt_r', 'cmd_r', 'f13', 'a', etc.
    """
    from pynput.keyboard import Key
    upper = name.strip().lower()
    if hasattr(Key, upper):
        return getattr(Key, upper)
    if len(name.strip()) == 1:
        return name.strip()
    raise ValueError(f"Unknown key: {name!r}")


def _parse_key_set(value: list[str] | None) -> set | None:
    if value is None:
        return None
    if not value:
        return set()
    return {_parse_key(k) for k in value}


_DEFAULTS: dict[str, Any] = {
    "stt_backend": None,
    "stt_model": None,
    "polish_enabled": True,
    "polish_model": None,
    "polish_languages": ["zh"],
    "polish_prompt": None,
    "trigger_keys": None,
    "edit_trigger_keys": None,
    "sample_rate": 16000,
    "min_audio_sec": 0.15,
    "max_audio_sec": 120,
    "selection_capture_wait_s": 0.1,
    "encoder_pipelining": False,
    "beeps_enabled": True,
    "beep_start_hz": 880,
    "beep_end_hz": 660,
    "beep_fail_hz": 220,
    "beep_duration_ms": 80,
    "beep_volume": 0.15,
    "mic_device": None,
}

_ENV_PREFIX = "HOME_STT_"

_BOOL_KEYS = {"polish_enabled", "encoder_pipelining", "beeps_enabled"}
_INT_KEYS = {"sample_rate", "beep_start_hz", "beep_end_hz", "beep_fail_hz", "beep_duration_ms"}
_FLOAT_KEYS = {"min_audio_sec", "max_audio_sec", "selection_capture_wait_s", "beep_volume"}
_KEY_SET_KEYS = {"trigger_keys", "edit_trigger_keys"}
_LIST_STR_KEYS = {"polish_languages"}


def _coerce_env(key: str, raw: str) -> Any:
    if key in _BOOL_KEYS:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if key in _INT_KEYS:
        return int(raw)
    if key in _FLOAT_KEYS:
        return float(raw)
    if key in _KEY_SET_KEYS:
        parts = [s.strip() for s in raw.split(",") if s.strip()]
        return parts if parts else None
    if key in _LIST_STR_KEYS:
        return [s.strip() for s in raw.split(",") if s.strip()]
    if key == "mic_device":
        stripped = raw.strip()
        try:
            return int(stripped)
        except ValueError:
            return stripped
    return raw


def load_config() -> dict[str, Any]:
    """Load config with priority: env var > config file > defaults.

    Returns a dict with all keys from _DEFAULTS, with overrides applied.
    Key-set values (trigger_keys, edit_trigger_keys) are left as list[str]
    or None; the caller converts to pynput Key sets.
    """
    cfg = dict(_DEFAULTS)

    path = config_path()
    if path.exists() and tomllib is not None:
        try:
            with open(path, "rb") as f:
                file_cfg = tomllib.load(f)
            for section_vals in file_cfg.values():
                if isinstance(section_vals, dict):
                    for k, v in section_vals.items():
                        norm = k.lower()
                        if norm in cfg:
                            cfg[norm] = v
                else:
                    pass
            for k, v in file_cfg.items():
                norm = k.lower()
                if norm in cfg and not isinstance(v, dict):
                    cfg[norm] = v
        except Exception as e:
            logger.warning("failed to load %s: %s", path, e)
    elif path.exists() and tomllib is None:
        logger.warning("config file exists at %s but neither tomllib "
                       "(Python 3.11+) nor tomli is available. Config file ignored.", path)

    for key in _DEFAULTS:
        env_name = _ENV_PREFIX + key.upper()
        env_val = os.environ.get(env_name)
        if env_val is not None and env_val.strip() != "":
            try:
                cfg[key] = _coerce_env(key, env_val)
            except (ValueError, TypeError) as e:
                logger.warning("bad env %s=%r: %s", env_name, env_val, e)

    return cfg


_TEMPLATE = """\
# home-stt configuration file
# Place at: {config_path}
#
# Priority: environment variable (HOME_STT_*) > this file > code default.
# Lines starting with # are comments. Uncomment and edit to override.

# ── STT backend ──────────────────────────────────────────────────────
# Available backends: "qwen3-asr", "faster-whisper", "mlx-whisper"
# stt_backend = "qwen3-asr"
# stt_model = "Qwen/Qwen3-ASR-0.6B"

# ── Polish (LLM post-processing) ────────────────────────────────────
# polish_enabled = true
# polish_model = "Qwen/Qwen3-4B-Instruct-2507"   # Windows/Linux
# polish_model = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit"  # macOS
# polish_languages = ["zh"]

# ── Trigger keys ─────────────────────────────────────────────────────
# Dictate trigger (hold to record). Default: Right Alt/Ctrl (Win), Right Option (Mac).
# trigger_keys = ["alt_r"]
#
# Voice-edit trigger (hold to edit selection). Default: F13 (Win), Right Cmd (Mac).
# edit_trigger_keys = ["f13"]

# ── Audio ────────────────────────────────────────────────────────────
# sample_rate = 16000
# min_audio_sec = 0.15
# max_audio_sec = 120
#
# Microphone device — name (substring match) or index number.
# Run `home-stt devices` to list available devices.
# mic_device = "MacBook Pro Microphone"
# mic_device = 1

# ── Beep feedback ────────────────────────────────────────────────────
# beeps_enabled = true
# beep_start_hz = 880
# beep_end_hz = 660
# beep_fail_hz = 220
# beep_duration_ms = 80
# beep_volume = 0.15

# ── Advanced ─────────────────────────────────────────────────────────
# encoder_pipelining = false
# selection_capture_wait_s = 0.1
"""


def generate_default_config() -> str:
    return _TEMPLATE.format(config_path=config_path())


def init_config() -> Path:
    """Write the default config file if it doesn't exist. Return the path."""
    path = config_path()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_default_config(), encoding="utf-8")
    return path


def _key_to_str(key) -> str:
    """Convert a pynput Key or character back to a config-file string."""
    from pynput.keyboard import Key
    if isinstance(key, Key):
        return key.name
    return str(key)


def update_trigger_keys(
    trigger: list[str] | None = None,
    edit_trigger: list[str] | None = None,
) -> Path:
    """Write trigger key settings into the user's config.toml.

    Creates the file from the default template if it doesn't exist.
    If it exists, patches the trigger_keys / edit_trigger_keys lines
    in place; if the keys aren't present yet, appends them to the
    trigger-keys section.
    """
    import re as _re

    path = init_config()
    content = path.read_text(encoding="utf-8")

    def _format_toml_list(keys: list[str]) -> str:
        return "[" + ", ".join(f'"{k}"' for k in keys) + "]"

    for key_name, value in [("trigger_keys", trigger),
                            ("edit_trigger_keys", edit_trigger)]:
        if value is None:
            continue
        toml_val = _format_toml_list(value)
        line = f"{key_name} = {toml_val}"
        # Replace existing (commented or not)
        pattern = _re.compile(
            r"^#?\s*" + _re.escape(key_name) + r"\s*=.*$", _re.MULTILINE
        )
        if pattern.search(content):
            content = pattern.sub(line, content, count=1)
        else:
            # Append after the trigger-keys section header
            marker = "# ── Trigger keys"
            idx = content.find(marker)
            if idx != -1:
                next_section = content.find("\n# ── ", idx + len(marker))
                insert_at = next_section if next_section != -1 else len(content)
                content = content[:insert_at] + line + "\n" + content[insert_at:]
            else:
                content = content.rstrip("\n") + "\n\n" + line + "\n"

    path.write_text(content, encoding="utf-8")
    return path


def apply_to_module(cfg: dict[str, Any], module) -> None:
    """Apply loaded config values to a daemon module's globals.

    Only overrides values that differ from the code-level None sentinel
    (for stt_backend, stt_model, etc.) or from _DEFAULTS. Key-set values
    are converted from list[str] to pynput Key sets here.
    """
    _str_map = {
        "stt_backend": "STT_BACKEND",
        "stt_model": "STT_MODEL",
        "polish_model": "POLISH_MODEL",
        "polish_prompt": "POLISH_PROMPT",
    }
    for cfg_key, mod_attr in _str_map.items():
        val = cfg.get(cfg_key)
        if val is not None:
            setattr(module, mod_attr, val)

    if cfg.get("polish_enabled") is not None:
        module.POLISH_ENABLED = bool(cfg["polish_enabled"])

    if cfg.get("polish_languages") is not None:
        module.POLISH_LANGUAGES = set(cfg["polish_languages"])

    _numeric_map = {
        "sample_rate": "SAMPLE_RATE",
        "min_audio_sec": "MIN_AUDIO_SEC",
        "max_audio_sec": "MAX_AUDIO_SEC",
        "selection_capture_wait_s": "SELECTION_CAPTURE_WAIT_S",
        "beep_start_hz": "BEEP_START_HZ",
        "beep_end_hz": "BEEP_END_HZ",
        "beep_fail_hz": "BEEP_FAIL_HZ",
        "beep_duration_ms": "BEEP_DURATION_MS",
        "beep_volume": "BEEP_VOLUME",
    }
    for cfg_key, mod_attr in _numeric_map.items():
        val = cfg.get(cfg_key)
        if val is not None:
            setattr(module, mod_attr, val)

    if cfg.get("encoder_pipelining") is not None:
        val = bool(cfg["encoder_pipelining"])
        module.ENCODER_PIPELINING = val
        try:
            import stt_streaming
            stt_streaming.ENCODER_PIPELINING = val
        except ImportError:
            pass

    if cfg.get("beeps_enabled") is not None:
        module.BEEPS_ENABLED = bool(cfg["beeps_enabled"])

    for cfg_key, mod_attr in [("trigger_keys", "TRIGGER_KEYS"),
                               ("edit_trigger_keys", "EDIT_TRIGGER_KEYS")]:
        val = cfg.get(cfg_key)
        if val is not None:
            parsed = _parse_key_set(val)
            if parsed is not None:
                setattr(module, mod_attr, parsed)

    mic = cfg.get("mic_device")
    if mic is not None:
        if isinstance(mic, int):
            setattr(module, "MIC_DEVICE", mic)
        else:
            setattr(module, "MIC_DEVICE", str(mic))
