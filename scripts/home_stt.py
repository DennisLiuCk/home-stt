"""home-stt — unified controller for the hold-to-talk STT daemon.

Wraps the platform-specific start/stop scripts (`stt-start.sh` on macOS,
`stt-start.ps1` on Windows) and adds cross-platform status / log / restart
/ config subcommands that the bare scripts do not provide.

After `pip install -e .` the entry point `home-stt` is on PATH and can be
called from any directory. Standalone fallback:

    python3 scripts/home_stt.py start
    python3 scripts/home_stt.py status

Subcommands: start, stop, restart, status, log, config, tray, devices, version.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Daemon log + polish startup line contain non-ASCII (zh transcripts, ≤ in
# polish banner). Windows default console codepage (cp950 on zh-TW) cannot
# encode these and would raise UnicodeEncodeError. Force UTF-8 on stdout/err.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

SCRIPTS_DIR = Path(__file__).resolve().parent
DAEMON_SCRIPT = SCRIPTS_DIR / "stt-daemon.py"
PID_FILE = SCRIPTS_DIR / "stt-daemon.pid"

# Add scripts dir to sys.path so stt_config can be imported.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _log_paths() -> tuple[Path, Path]:
    """Match the log paths chosen by stt-start.sh / stt-start.ps1."""
    tmp = Path(tempfile.gettempdir())
    return tmp / "stt-daemon.log", tmp / "stt-daemon.err.log"


def _daemon_version() -> str:
    """Get version from importlib.metadata (pip install), falling back to
    reading __version__ from stt-daemon.py for dev/standalone runs."""
    try:
        from importlib.metadata import version
        return version("home-stt")
    except Exception:
        pass
    try:
        for line in DAEMON_SCRIPT.read_text(encoding="utf-8").splitlines()[:200]:
            m = re.match(r'^__version__\s*=\s*["\']([^"\']+)["\']', line)
            if m:
                return m.group(1)
    except OSError:
        pass
    return "?"


def _process_cmdline(pid: int) -> str | None:
    """Best-effort command line of `pid`, or None if it can't be determined.

    Used by _process_alive to confirm a PID is actually the STT daemon
    rather than an unrelated process that recycled the number. Any failure
    (timeout, tool missing, non-zero exit) returns None so the caller can
    fall back to a bare existence check instead of a false negative."""
    try:
        if sys.platform == "win32":
            r = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter "
                 f"'ProcessId={pid}').CommandLine"],
                capture_output=True, text=True, errors="replace", timeout=4,
            )
        else:
            # -ww: emit the FULL command line. BSD/macOS ps truncates the
            # args column without it, which would drop the trailing
            # stt-daemon.py token for daemons installed under a long path
            # and cause a false 'stopped'. Matches pgrep -f's full-cmdline
            # view used by stt-start/stop.sh.
            r = subprocess.run(
                ["ps", "-ww", "-p", str(pid), "-o", "command="],
                capture_output=True, text=True, errors="replace", timeout=3,
            )
    except (subprocess.SubprocessError, FileNotFoundError, OSError, ValueError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout


def _process_alive(pid: int) -> bool:
    """True iff `pid` is a live process that is (best-effort) the STT daemon.

    Existence alone is insufficient: stt-start.ps1/.sh leave the PID file
    behind on a hard crash (only the stop scripts remove it), and the OS
    eventually recycles that number for an unrelated process — `status`
    would then report a bogus 'running' with that process's RSS/uptime. So
    after confirming the PID exists we verify its command line references
    stt-daemon.py, mirroring the start/stop scripts' own command-line scan.
    If the identity query can't run or returns nothing (CIM unavailable, a
    launcher whose command line hides the script path), we FALL BACK to the
    bare existence check rather than risk a false 'stopped'."""
    if sys.platform == "win32":
        try:
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True, text=True, timeout=3,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
        if str(pid) not in r.stdout:
            return False
    else:
        try:
            os.kill(pid, 0)
        except (OSError, PermissionError):
            return False
    # PID exists. Confirm identity when we can; otherwise assume alive so a
    # query hiccup never makes a genuinely-running daemon look stopped.
    cmdline = _process_cmdline(pid)
    if not cmdline or not cmdline.strip():
        return True
    return "stt-daemon.py" in cmdline


def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    return pid if _process_alive(pid) else None


def _rss_mb(pid: int) -> float | None:
    try:
        if sys.platform == "win32":
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True, text=True, timeout=3,
            )
            # CSV: "python.exe","12345","Console","1","42,123 K"
            parts = [p.strip('"') for p in r.stdout.strip().split('","')]
            if len(parts) >= 5:
                kb = int(parts[-1].replace(",", "").replace(" K", "").strip())
                return kb / 1024.0
        else:
            r = subprocess.run(
                ["ps", "-p", str(pid), "-o", "rss="],
                capture_output=True, text=True, timeout=3,
            )
            return int(r.stdout.strip()) / 1024.0
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        return None
    return None


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}m"


def _format_size_mb(mb: float) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.0f} MB"


_STARTUP_PATTERNS = {
    "backend": re.compile(r"\[stt\]\s+backend:\s+(.+?)\s+\|\s+model:\s+(.+?)\s*$"),
    "paste":   re.compile(r"\[stt\]\s+paste path:\s+(.+?)\s*$"),
    "polish":  re.compile(r"\[stt\]\s+polish:\s+(.+?)\s*$"),
    "warmup":  re.compile(r"\[stt\]\s+warmup\s+[\d.]+s\s+.\s+(hold .+?)\.?\s*$"),
}


def _parse_startup(lines: list[str]) -> dict[str, str]:
    """Extract backend / model / polish / paste / triggers from startup log."""
    info: dict[str, str] = {}
    for line in lines:
        m = _STARTUP_PATTERNS["backend"].search(line)
        if m and "backend" not in info:
            info["backend"] = f"{m.group(1).strip()} ({m.group(2).strip()})"
            continue
        m = _STARTUP_PATTERNS["paste"].search(line)
        if m and "paste" not in info:
            info["paste"] = m.group(1).strip()
            continue
        m = _STARTUP_PATTERNS["polish"].search(line)
        if m and "polish" not in info:
            info["polish"] = m.group(1).strip()
            continue
        m = _STARTUP_PATTERNS["warmup"].search(line)
        if m and "triggers" not in info:
            info["triggers"] = m.group(1).strip()
            continue
    return info


_TRANSCRIBE_PAT = re.compile(
    r"\[stt\]\s+(zh|en|ja|ko|EDIT|voice-edit|empty|too short|silent)"
)


def _recent_transcribes(lines: list[str], n: int = 3) -> list[str]:
    out: list[str] = []
    for line in reversed(lines):
        if _TRANSCRIBE_PAT.search(line):
            out.append(line.rstrip())
            if len(out) == n:
                break
    return list(reversed(out))


def _read_lines(path: Path) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.readlines()
    except OSError:
        return []


# ---- subcommand handlers ----------------------------------------------------

def cmd_start(args) -> int:
    if sys.platform == "win32":
        script = SCRIPTS_DIR / "stt-start.ps1"
        cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    else:
        script = SCRIPTS_DIR / "stt-start.sh"
        cmd = ["bash", str(script)]
    if not script.exists():
        print(f"home-stt: missing start script {script}", file=sys.stderr)
        return 1
    rc = subprocess.call(cmd)
    if rc == 0 and getattr(args, "tray", False):
        _launch_tray_background()
    return rc


def _launch_tray_background() -> None:
    """Spawn the tray icon as a detached background process."""
    tray_script = SCRIPTS_DIR / "stt_tray.py"
    try:
        subprocess.Popen(
            [sys.executable, str(tray_script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        print("Tray icon launched.")
    except Exception as e:
        print(f"home-stt: tray launch failed: {e}", file=sys.stderr)


def cmd_stop(_args) -> int:
    if sys.platform == "win32":
        script = SCRIPTS_DIR / "stt-stop.ps1"
        cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    else:
        script = SCRIPTS_DIR / "stt-stop.sh"
        cmd = ["bash", str(script)]
    if not script.exists():
        print(f"home-stt: missing stop script {script}", file=sys.stderr)
        return 1
    return subprocess.call(cmd)


def cmd_restart(args) -> int:
    rc = cmd_stop(args)
    if rc != 0:
        print("home-stt: stop returned non-zero; continuing with start",
              file=sys.stderr)
    # Brief settle so the OS releases the audio device and the PID file is gone.
    time.sleep(0.5)
    return cmd_start(args)


def cmd_status(_args) -> int:
    log_path, err_path = _log_paths()
    pid = _read_pid()
    version = _daemon_version()
    state = "running" if pid else "stopped"
    print(f"home-stt v{version} -- {state}")

    if pid is not None:
        try:
            uptime = time.time() - PID_FILE.stat().st_mtime
            print(f"  PID:      {pid} (uptime {_format_duration(uptime)})")
        except OSError:
            print(f"  PID:      {pid}")
        rss = _rss_mb(pid)
        if rss is not None:
            print(f"  RSS:      {_format_size_mb(rss)}")
    elif PID_FILE.exists():
        print("  PID file: stale (process gone, file not cleaned up)")
    else:
        print("  PID file: (none)")

    if log_path.exists():
        try:
            age = time.time() - log_path.stat().st_mtime
            print(f"  log:      {log_path} (last write {_format_duration(age)} ago)")
        except OSError:
            print(f"  log:      {log_path}")
    else:
        print(f"  log:      (none — daemon has not run yet)")

    if err_path.exists():
        size = err_path.stat().st_size
        if size > 0:
            print(f"  err.log:  {err_path} ({size} bytes -- inspect with `home-stt log --err`)")
        else:
            print(f"  err.log:  (clean)")

    if log_path.exists():
        all_lines = _read_lines(log_path)
        startup_lines = all_lines[:80]
        info = _parse_startup(startup_lines)
        if info:
            print()
            for key in ("backend", "polish", "paste", "triggers"):
                if key in info:
                    print(f"  {key + ':':<10}{info[key]}")
        tail_lines = all_lines[-80:] if len(all_lines) > 80 else all_lines
        recent = _recent_transcribes(tail_lines, n=3)
        if recent:
            print()
            print("  recent transcribes:")
            for line in recent:
                print(f"    {line}")

    if pid is None:
        print()
        print("Use `home-stt log` to see what it was doing, "
              "or `home-stt start` to launch.")
    return 0


def cmd_log(args) -> int:
    log_path, err_path = _log_paths()
    path = err_path if args.err else log_path
    if not path.exists():
        print(f"home-stt: log not found at {path}", file=sys.stderr)
        return 1
    if args.follow:
        return _follow_log(path)
    lines = _read_lines(path)
    tail = lines[-args.tail:] if len(lines) > args.tail else lines
    sys.stdout.writelines(tail)
    if tail and not tail[-1].endswith("\n"):
        sys.stdout.write("\n")
    return 0


def _follow_log(path: Path) -> int:
    """tail -F equivalent in pure Python."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            # Show last ~10 lines first so the screen isn't empty.
            f.seek(max(0, end - 32 * 1024))
            recent = f.readlines()[-10:]
            sys.stdout.writelines(recent)
            sys.stdout.flush()
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    except OSError as e:
        print(f"home-stt: log follow failed: {e}", file=sys.stderr)
        return 1


def _capture_key(prompt_msg: str, *, timeout: float = 15.0):
    """Listen for a single key press and return it (pynput Key or char).

    Returns None if the user presses Escape or the timeout expires.
    """
    from pynput.keyboard import Key, Listener
    import threading

    result = None
    done = threading.Event()

    def on_press(key):
        nonlocal result
        if key == Key.esc:
            done.set()
            return False
        result = key
        done.set()
        return False

    print(prompt_msg, flush=True)
    listener = Listener(on_press=on_press)
    listener.start()
    done.wait(timeout=timeout)
    listener.stop()

    if result is None:
        return None

    # Normalise: KeyCode with no char → use vk lookup, plain char → return it
    if hasattr(result, "char") and result.char:
        return result.char
    return result


def _key_display(key) -> str:
    from pynput.keyboard import Key
    if isinstance(key, Key):
        return key.name
    return repr(key)


def cmd_set_trigger(_args) -> int:
    """Interactive trigger-key detection and config update."""
    from stt_config import _key_to_str, update_trigger_keys

    print("home-stt trigger key setup")
    print("=" * 40)
    print()

    # Dictate trigger
    print("Step 1: Dictate trigger (hold-to-record key)")
    print("  Default: Right Alt + Right Ctrl (Win), Right Option (Mac)")
    key = _capture_key("  >> Press the key you want to use (Esc to keep default)...")
    trigger = None
    if key is not None:
        name = _key_to_str(key)
        trigger = [name]
        print(f"  Captured: {_key_display(key)}  →  trigger_keys = [\"{name}\"]")
    else:
        print("  Skipped — keeping platform default.")
    print()

    # Edit trigger
    print("Step 2: Voice-edit trigger (hold to edit selection)")
    print("  Default: F13 (Win), Right Cmd (Mac)")
    key = _capture_key("  >> Press the key you want to use (Esc to skip)...")
    edit_trigger = None
    if key is not None:
        name = _key_to_str(key)
        edit_trigger = [name]
        print(f"  Captured: {_key_display(key)}  →  edit_trigger_keys = [\"{name}\"]")
    else:
        print("  Skipped — keeping platform default.")
    print()

    if trigger is None and edit_trigger is None:
        print("No changes made.")
        return 0

    path = update_trigger_keys(trigger=trigger, edit_trigger=edit_trigger)
    print(f"Config updated: {path}")
    print("Run `home-stt restart` to apply.")
    return 0


def cmd_config(args) -> int:
    from stt_config import config_path, init_config, load_config, generate_default_config

    if args.set_trigger:
        return cmd_set_trigger(args)

    if getattr(args, "disable_edit_trigger", False):
        from stt_config import update_trigger_keys
        path = update_trigger_keys(edit_trigger=[])
        print(f"Voice-edit trigger disabled (edit_trigger_keys = []) in {path}.")
        print("Restart the daemon to apply: home-stt restart")
        return 0

    if args.init:
        path = config_path()
        existed = path.exists()
        init_config()
        print(f"Config file: {path}")
        if existed:
            print("(already existed — not overwritten)")
        else:
            print("(created)")
        return 0

    if args.edit:
        path = config_path()
        if not path.exists():
            init_config()
            print(f"Created default config: {path}")
        editor = (os.environ.get("VISUAL") or os.environ.get("EDITOR")
                  or ("notepad" if sys.platform == "win32" else "nano"))
        return subprocess.call([editor, str(path)])

    if args.path:
        print(config_path())
        return 0

    path = config_path()
    if path.exists():
        cfg = load_config()
        print(f"Config (from {path}):")
        for key, val in sorted(cfg.items()):
            print(f"  {key} = {val!r}")
    else:
        print(f"No config file found. Using code defaults.")
        print(f"  Config path: {path}")
        print(f"  Run `home-stt config --init` to create one.")
    return 0


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return ok


def cmd_doctor(_args) -> int:
    """Run environment health checks and print a pass/fail checklist."""
    import platform as _platform

    print(f"home-stt v{_daemon_version()} doctor\n")
    all_ok = True

    # 1. Python version
    v = sys.version_info
    py_ok = v >= (3, 10)
    all_ok &= _check("Python >= 3.10", py_ok,
                      f"{v.major}.{v.minor}.{v.micro}")

    # 2. Core dependencies
    for pkg_name, import_name in [
        ("numpy", "numpy"),
        ("sounddevice", "sounddevice"),
        ("pynput", "pynput"),
        ("opencc", "opencc"),
    ]:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", getattr(mod, "VERSION", "ok"))
            all_ok &= _check(f"{pkg_name}", True, str(ver))
        except ImportError:
            all_ok &= _check(f"{pkg_name}", False, "not installed")

    # 3. TOML support (Python 3.11+ built-in, or tomli fallback)
    toml_ok = False
    toml_detail = ""
    try:
        import tomllib  # noqa: F811
        toml_ok = True
        toml_detail = "tomllib (built-in)"
    except ModuleNotFoundError:
        try:
            import tomli  # noqa: F811
            toml_ok = True
            toml_detail = f"tomli {tomli.__version__}"
        except ModuleNotFoundError:
            toml_detail = "neither tomllib nor tomli available"
    all_ok &= _check("TOML support", toml_ok, toml_detail)

    # 4. Platform-specific STT backends
    print()
    is_mac = sys.platform == "darwin" and _platform.machine() == "arm64"

    if is_mac:
        # MLX backend
        try:
            import mlx  # noqa: F401
            all_ok &= _check("mlx", True, getattr(mlx, "__version__", "ok"))
        except ImportError:
            all_ok &= _check("mlx", False, "not installed (needed for mlx-whisper / qwen3-asr on Mac)")
        try:
            import mlx_lm  # noqa: F401
            all_ok &= _check("mlx-lm (polish)", True,
                             getattr(mlx_lm, "__version__", "ok"))
        except ImportError:
            # Optional — polish degrades to NoopPolisher without mlx-lm.
            _check("mlx-lm (polish)", False, "not installed — polish will be disabled")
    else:
        # CUDA backend
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
            if cuda_ok:
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                all_ok &= _check("torch + CUDA", True,
                                 f"{torch.__version__}, {gpu_name}, "
                                 f"{vram_gb:.1f} GB VRAM")
            else:
                all_ok &= _check("torch + CUDA", False,
                                 f"torch {torch.__version__} but CUDA not available")
        except ImportError:
            all_ok &= _check("torch", False, "not installed")

        try:
            import transformers
            all_ok &= _check("transformers", True, transformers.__version__)
        except ImportError:
            all_ok &= _check("transformers", False, "not installed")

        # qwen-asr (default backend)
        try:
            import qwen_asr  # noqa: F401
            all_ok &= _check("qwen-asr", True,
                             getattr(qwen_asr, "__version__", "ok"))
        except ImportError:
            all_ok &= _check("qwen-asr", False,
                             "not installed (default STT backend — see README)")

        # faster-whisper (fallback backend) — optional if qwen-asr is present.
        try:
            import faster_whisper
            _check("faster-whisper (fallback)", True,
                   faster_whisper.__version__)
        except ImportError:
            _check("faster-whisper (fallback)", False,
                   "not installed — ok if qwen-asr is available")

    # 5. Microphone
    print()
    try:
        import sounddevice as sd
        dev = sd.query_devices(kind="input")
        mic_name = dev.get("name", "unknown")
        sr = int(dev.get("default_samplerate", 0))
        all_ok &= _check("Microphone", True, f"{mic_name} ({sr} Hz)")
    except Exception as e:
        all_ok &= _check("Microphone", False, str(e))

    # 6. macOS permissions hint
    if sys.platform == "darwin":
        print()
        print("  macOS permissions needed (grant in System Settings → "
              "Privacy & Security):")
        print("    - Input Monitoring (for global keyboard listener)")
        print("    - Accessibility (for IME-safe paste via Quartz)")
        print("    - Microphone (for audio capture)")
        py_path = sys.executable
        print(f"    Python binary: {py_path}")

    # 7. Config file
    print()
    from stt_config import config_path
    cp = config_path()
    _check("Config file", cp.exists(), str(cp))

    # Summary
    print()
    if all_ok:
        print("All checks passed. Run `home-stt start` to launch.")
    else:
        print("Some checks failed. Fix the issues above and re-run "
              "`home-stt doctor`.")
    return 0 if all_ok else 1


def cmd_tray(_args) -> int:
    """Launch the system tray icon."""
    import stt_tray
    stt_tray.main()
    return 0


def cmd_web(args) -> int:
    """Launch the Gradio web UI."""
    try:
        import stt_web
    except ImportError as e:
        if "gradio" in str(e).lower():
            print("home-stt: gradio not installed. Install with:\n"
                  "  pip install home-stt[web]", file=sys.stderr)
        else:
            print(f"home-stt: web UI import failed: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"home-stt: web UI failed to load (DLL/library error): {e}",
              file=sys.stderr)
        return 1
    try:
        stt_web.main(port=args.port, share=args.share)
    except OSError as e:
        print(f"home-stt: web server failed: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_devices(_args) -> int:
    """List available audio input devices."""
    try:
        import sounddevice as sd
    except ImportError:
        print("home-stt: sounddevice not installed", file=sys.stderr)
        return 1

    devices = sd.query_devices()
    default_input = sd.default.device[0]
    print("Available input devices:\n")
    found = False
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] <= 0:
            continue
        found = True
        marker = " *" if idx == default_input else "  "
        sr = int(dev["default_samplerate"])
        ch = dev["max_input_channels"]
        print(f"{marker} [{idx}] {dev['name']}  ({sr} Hz, {ch}ch)")
    if not found:
        print("  (no input devices found)")
    print()
    print("  * = system default")
    print()
    print("To use a specific device, add to config.toml:")
    from stt_config import config_path
    print(f"  ({config_path()})")
    print()
    print('  mic_device = "Device Name"   # substring match')
    print("  mic_device = 1               # or device index")
    return 0


def cmd_version(_args) -> int:
    print(f"home-stt v{_daemon_version()}")
    return 0


# ---- entry point ------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="home-stt",
        description="Hold-to-talk STT daemon controller (macOS + Windows).",
    )
    parser.add_argument("--version", action="store_true",
                        help="Print daemon version and exit.")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_start = sub.add_parser("start",   help="Start the daemon, or report it is already running.")
    p_start.add_argument("--tray", action="store_true",
                         help="Also launch the system tray icon.")
    sub.add_parser("stop",    help="Stop the daemon.")
    p_restart = sub.add_parser("restart", help="Stop, settle, then start. Use after config edits.")
    p_restart.add_argument("--tray", action="store_true",
                           help="Also launch the system tray icon after restart.")
    sub.add_parser("status",  help="Print PID / uptime / RSS / backend / polish / paste / recent transcribes.")

    p_log = sub.add_parser("log", help="Show the daemon log.")
    p_log.add_argument("--err", action="store_true",
                       help="Show err.log instead of stdout log.")
    p_log.add_argument("--tail", type=int, default=30,
                       help="Show last N lines (default 30).")
    p_log.add_argument("-f", "--follow", action="store_true",
                       help="Follow new output (Ctrl+C to exit).")

    p_cfg = sub.add_parser("config",
                           help="Show, create, or edit the TOML config file.")
    p_cfg.add_argument("--init", action="store_true",
                       help="Create a default config.toml if one doesn't exist.")
    p_cfg.add_argument("--edit", action="store_true",
                       help="Open config.toml in $VISUAL / $EDITOR / notepad.")
    p_cfg.add_argument("--path", action="store_true",
                       help="Print the config file path and exit.")
    p_cfg.add_argument("--set-trigger", action="store_true",
                       help="Interactive key detection — press a key to set triggers.")
    p_cfg.add_argument("--disable-edit-trigger", action="store_true",
                       help="Disable voice-edit by writing edit_trigger_keys = [] "
                            "(--set-trigger's Esc only keeps the default, can't clear).")

    sub.add_parser("doctor", help="Run environment health checks (Python, deps, mic, permissions).")
    sub.add_parser("tray", help="Launch system tray icon (Windows: pystray, macOS: rumps).")
    sub.add_parser("devices", help="List available audio input (microphone) devices.")

    p_web = sub.add_parser("web", help="Launch the web UI dashboard (Gradio).")
    p_web.add_argument("--port", type=int, default=7860,
                       help="Server port (default 7860).")
    p_web.add_argument("--share", action="store_true",
                       help="Create a public Gradio share link.")

    sub.add_parser("version", help="Print home-stt version (same as --version).")

    args = parser.parse_args(argv)
    if args.version:
        return cmd_version(args)
    if args.command is None:
        parser.print_help()
        return 0

    handlers = {
        "start":   cmd_start,
        "stop":    cmd_stop,
        "restart": cmd_restart,
        "status":  cmd_status,
        "log":     cmd_log,
        "config":  cmd_config,
        "doctor":  cmd_doctor,
        "tray":    cmd_tray,
        "devices": cmd_devices,
        "web":     cmd_web,
        "version": cmd_version,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
