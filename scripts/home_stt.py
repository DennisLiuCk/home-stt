"""home-stt — unified controller for the hold-to-talk STT daemon.

Wraps the platform-specific start/stop scripts (`stt-start.sh` on macOS,
`stt-start.ps1` on Windows) and adds cross-platform status / log / restart
/ config subcommands that the bare scripts do not provide.

After `pip install -e .` the entry point `home-stt` is on PATH and can be
called from any directory. Standalone fallback:

    python3 scripts/home_stt.py start
    python3 scripts/home_stt.py status

Subcommands: start, stop, restart, status, log, config, version.
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


def _process_alive(pid: int) -> bool:
    if sys.platform == "win32":
        try:
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True, text=True, timeout=3,
            )
            return str(pid) in r.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, PermissionError):
        return False


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

def cmd_start(_args) -> int:
    if sys.platform == "win32":
        script = SCRIPTS_DIR / "stt-start.ps1"
        cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    else:
        script = SCRIPTS_DIR / "stt-start.sh"
        cmd = ["bash", str(script)]
    if not script.exists():
        print(f"home-stt: missing start script {script}", file=sys.stderr)
        return 1
    return subprocess.call(cmd)


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


def cmd_config(args) -> int:
    from stt_config import config_path, init_config, load_config, generate_default_config

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

    sub.add_parser("start",   help="Start the daemon, or report it is already running.")
    sub.add_parser("stop",    help="Stop the daemon.")
    sub.add_parser("restart", help="Stop, settle, then start. Use after config edits.")
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
        "version": cmd_version,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
