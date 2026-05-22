"""
Hold-to-talk voice → text daemon.

Hold the trigger key (Right Alt / AltGr or Right Ctrl on Windows by default)
to record from the default microphone. Release to:
  1. Transcribe via the active STT backend (default: faster-whisper,
     model = large-v3-turbo, GPU CUDA float16 when available, CPU int8
     fallback).
  2. Convert simplified Chinese to Taiwan-traditional via OpenCC.
  3. Insert spaces at zh ↔ en/digit boundaries.
  4. Place the text on the system clipboard AND simulate Ctrl+V to paste
     it into the focused window (atomic paste — IME cannot interrupt).

Stdin/stdout is forced to UTF-8 so simplified-Chinese characters can be
logged on a zh-TW Windows locale (default cp950 cannot encode them).

First run downloads the model (~1.5 GB for large-v3-turbo) into the
HuggingFace cache.

────────────────────────────────────────────────────────────────────────
Backend abstraction
────────────────────────────────────────────────────────────────────────
  The STT engine is hidden behind the `STTBackend` interface so the rest
  of the pipeline (mic capture → post-processing → clipboard+paste) stays
  the same when swapping engines. Switch by changing `STT_BACKEND` below
  and adding a class. See `build_backend()` for the dispatch table.

  Implemented:
    - faster-whisper (Whisper large-v3-turbo via CTranslate2)
  Planned (roadmap):
    - sense-voice  (Alibaba FunASR SenseVoice-Small — fast, small, multilang)
    - paraformer   (Alibaba FunASR Paraformer-zh — Chinese SOTA, non-autoregressive)
    - mlx-whisper  (Apple Silicon Metal-native Whisper backend)

────────────────────────────────────────────────────────────────────────
Platform support
────────────────────────────────────────────────────────────────────────
  Currently:  Windows 10/11 only (clipboard + paste + DLL lookup are
              Windows-specific; everything else is portable).
  Planned:    macOS (incl. Apple Silicon) and Linux (X11 + Wayland).
              See README "Roadmap" section. The platform-specific layer
              is small — clipboard write, paste simulation, trigger key —
              and is a candidate for a Pasteboard interface refactor when
              the second platform lands.
"""
from __future__ import annotations

import ctypes
import os
import re
import site
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from ctypes import wintypes

# ---------------------------------------------------------------------------
# Stdout: force UTF-8 — Whisper may output simplified Chinese before OpenCC
# runs (during partial logs), and the default cp950 codec on zh-TW Windows
# can't encode those characters.
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------------------------------------------------------------------------
# NVIDIA DLL discovery — must run BEFORE importing faster_whisper / any
# CTranslate2-backed engine. Specific to faster-whisper backend on Windows;
# harmless no-op on platforms without site-packages/nvidia/<lib>/bin.
# ---------------------------------------------------------------------------
def _register_nvidia_dlls() -> int:
    bin_dirs: list[str] = []
    roots = [site.getusersitepackages()] + list(site.getsitepackages())
    for sp in roots:
        nv = os.path.join(sp, "nvidia")
        if not os.path.isdir(nv):
            continue
        for sub in os.listdir(nv):
            bin_dir = os.path.join(nv, sub, "bin")
            if os.path.isdir(bin_dir):
                bin_dirs.append(bin_dir)
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(bin_dir)
                    except Exception:
                        pass
    if bin_dirs:
        os.environ["PATH"] = (os.pathsep.join(bin_dirs)
                              + os.pathsep + os.environ.get("PATH", ""))
    return len(bin_dirs)


_NVIDIA_COUNT = _register_nvidia_dlls()

import numpy as np
import sounddevice as sd
from opencc import OpenCC
from pynput import keyboard
from pynput.keyboard import Key


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_RATE      = 16000
MIN_AUDIO_SEC    = 0.3                 # taps shorter than this are ignored

# Which STT backend to use. Switch this string + ensure the backend's
# dependencies are installed. See `build_backend()` below.
STT_BACKEND      = "faster-whisper"
# Model identifier passed to the backend. Interpretation is backend-specific:
#   faster-whisper:  Whisper model name ("large-v3-turbo", "medium", "small", ...)
#   sense-voice:     ModelScope ID, e.g. "iic/SenseVoiceSmall"
#   paraformer:      ModelScope ID, e.g. "iic/speech_paraformer-large_..."
STT_MODEL        = "large-v3-turbo"

# Any of these triggers recording. Right Alt collides with some Chrome
# shortcuts; Right Ctrl is the safety net. The first one pressed "wins"
# the turn — releasing it stops recording, presses of the others while
# already recording are ignored.
TRIGGER_KEYS     = {Key.alt_gr, Key.ctrl_r}


# ---------------------------------------------------------------------------
# Text post-processing (backend-agnostic)
# ---------------------------------------------------------------------------
_s2tw = OpenCC("s2tw")
_CJK  = r"[㐀-鿿]"
_AW   = r"[A-Za-z0-9]"


def post_process(text: str) -> str:
    """Simplified → Taiwan-traditional, then add spaces at CJK ↔ ASCII edges."""
    text = _s2tw.convert(text)
    text = re.sub(f"({_CJK})({_AW})", r"\1 \2", text)
    text = re.sub(f"({_AW})({_CJK})", r"\1 \2", text)
    return text


# ---------------------------------------------------------------------------
# Win32 SendInput unicode typing + clipboard (Windows-only; future macOS /
# Linux work will introduce a Pasteboard interface — see module docstring).
#
# pynput.Controller.type() falls back to virtual-key presses for ASCII
# letters, which the Bopomofo IME swallows as zhuyin keystrokes. The
# `type_text` helper forces every character through KEYEVENTF_UNICODE so
# the IME layer never sees them. (Kept as a fallback — main path uses
# clipboard + Ctrl+V because IME also interferes with type after CJK
# punctuation like 、.)
# ---------------------------------------------------------------------------
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_INPUT_KEYBOARD    = 1
_KEYEVENTF_UNICODE = 0x0004
_KEYEVENTF_KEYUP   = 0x0002
_ULONG_PTR = (ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8
              else ctypes.c_ulong)


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", _ULONG_PTR)]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", _ULONG_PTR)]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("ii", _INPUTUNION)]


def type_text(text: str) -> None:
    """Fallback typing path: SendInput + KEYEVENTF_UNICODE per character.
    Currently NOT used in the main path because the Bopomofo IME enters a
    punctuation buffer after CJK punctuation like 、 (U+3001) and swallows
    every character that follows in the same batch."""
    inputs: list[_INPUT] = []
    for ch in text:
        cp = ord(ch)
        units = ((cp,) if cp <= 0xFFFF
                 else (0xD800 + ((cp - 0x10000) >> 10),
                       0xDC00 + ((cp - 0x10000) & 0x3FF)))
        for u in units:
            for flag in (0, _KEYEVENTF_KEYUP):
                inp = _INPUT()
                inp.type = _INPUT_KEYBOARD
                inp.ii.ki.wVk = 0
                inp.ii.ki.wScan = u
                inp.ii.ki.dwFlags = _KEYEVENTF_UNICODE | flag
                inputs.append(inp)
    if not inputs:
        return
    n = len(inputs)
    arr = (_INPUT * n)(*inputs)
    _user32.SendInput(n, arr, ctypes.sizeof(_INPUT))


# Virtual-key codes for the Ctrl+V combo.
_VK_CONTROL = 0x11
_VK_V       = 0x56


def paste_clipboard() -> None:
    """Send Ctrl+V via raw SendInput. Ctrl is a system modifier and IMEs
    don't intercept Ctrl-combos, so a single Ctrl+V pastes the whole
    clipboard content atomically (no per-character IME interference)."""
    sequence = [
        (_VK_CONTROL, False),
        (_VK_V,       False),
        (_VK_V,       True),
        (_VK_CONTROL, True),
    ]
    inputs: list[_INPUT] = []
    for vk, up in sequence:
        inp = _INPUT()
        inp.type = _INPUT_KEYBOARD
        inp.ii.ki.wVk = vk
        inp.ii.ki.wScan = 0
        inp.ii.ki.dwFlags = _KEYEVENTF_KEYUP if up else 0
        inputs.append(inp)
    n = len(inputs)
    arr = (_INPUT * n)(*inputs)
    _user32.SendInput(n, arr, ctypes.sizeof(_INPUT))


def set_clipboard(text: str) -> None:
    """Place text on the Windows clipboard. PowerShell 5.1 reads stdin in
    cp950 on zh-TW locale by default — force UTF-8 or unicode comes out
    as mojibake."""
    cmd = ("[Console]::InputEncoding = [System.Text.Encoding]::UTF8; "
           "$in = [Console]::In.ReadToEnd(); Set-Clipboard -Value $in")
    proc = subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", cmd],
        stdin=subprocess.PIPE,
        creationflags=0x08000000,  # CREATE_NO_WINDOW
    )
    proc.communicate(input=text.encode("utf-8"))


# ---------------------------------------------------------------------------
# STT Backend abstraction
#
# Concrete backends own model loading, warmup, and inference. The pipeline
# only needs `transcribe(samples) -> (text, language)` and an optional
# `warmup()` to pay one-time JIT costs up front.
# ---------------------------------------------------------------------------
class STTBackend(ABC):
    """Interface for speech-to-text engines. Implementations decide their
    own model + device + precision; the daemon just hands them raw float32
    audio at 16 kHz mono and expects a (text, language) tuple back."""

    name: str = "abstract"

    @abstractmethod
    def transcribe(self, samples: np.ndarray) -> tuple[str, str]:
        """Run inference on a 1-D float32 PCM array (16 kHz, mono).
        Returns (raw_text, language_code). Backends do NOT post-process
        text (no s2tw, no spacing) — that's done downstream."""

    def warmup(self) -> None:
        """Optional: do a dummy inference so the first real request is fast.
        Default is no-op; backends override if they have a cold start."""

    @property
    def device_label(self) -> str:
        """Short human label like 'CUDA (float16)' for startup logging."""
        return "unknown"


class FasterWhisperBackend(STTBackend):
    """OpenAI Whisper via CTranslate2 (faster-whisper). Tries CUDA float16
    first, falls back to CPU int8 if CUDA initialisation fails."""

    name = "faster-whisper"

    def __init__(self, model_name: str):
        # Lazy import — keeps `faster_whisper` out of the import graph when
        # a different backend is selected.
        from faster_whisper import WhisperModel

        try:
            self._model = WhisperModel(model_name, device="cuda",
                                       compute_type="float16")
            self._device_label = "CUDA (float16)"
        except Exception as e:
            print(f"[stt] CUDA load failed ({e}); falling back to CPU int8.",
                  flush=True)
            self._model = WhisperModel(model_name, device="cpu",
                                       compute_type="int8")
            self._device_label = "CPU (int8)"

    @property
    def device_label(self) -> str:
        return self._device_label

    def transcribe(self, samples: np.ndarray) -> tuple[str, str]:
        segments, info = self._model.transcribe(
            samples,
            beam_size=1,                       # greedy — 3-5x faster, near-identical quality
            condition_on_previous_text=False,  # avoids cross-segment carryover
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        text = "".join(s.text for s in segments).strip()
        return text, info.language

    def warmup(self) -> None:
        # Cold-start CUDA JIT compile on Blackwell can take ~10s; once
        # cached it's sub-second. Pay it now so the user's first
        # hold-to-talk doesn't eat it.
        warm_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        list(self._model.transcribe(warm_audio, beam_size=1,
                                    vad_filter=False)[0])


def build_backend(name: str, model: str) -> STTBackend:
    """Factory. To add a new backend: implement STTBackend in a new class,
    add a branch here, and update STT_BACKEND in the Config section."""
    if name == "faster-whisper":
        return FasterWhisperBackend(model)
    # ── Future backends ────────────────────────────────────────────────
    # elif name == "sense-voice":
    #     from .backends import SenseVoiceBackend  # or inline class
    #     return SenseVoiceBackend(model)
    # elif name == "paraformer":
    #     return ParaformerBackend(model)
    # elif name == "mlx-whisper":
    #     return MlxWhisperBackend(model)
    raise ValueError(f"Unknown STT backend: {name!r}")


# ---------------------------------------------------------------------------
# Audio capture state
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
_buffer: list[np.ndarray] = []
_recording = False
_active_trigger = None   # which TRIGGER_KEYS member is currently held, or None
_processing = False


def _audio_callback(indata, frames, time_info, status) -> None:
    if status:
        print(f"[stt] audio status: {status}", file=sys.stderr, flush=True)
    with _state_lock:
        if _recording:
            _buffer.append(indata.copy())


# ---------------------------------------------------------------------------
# Transcription pipeline (backend-agnostic)
# ---------------------------------------------------------------------------
_backend: STTBackend  # set in main()


def _transcribe_and_emit() -> None:
    global _processing, _buffer
    with _state_lock:
        if _processing:
            return
        _processing = True
        chunks = _buffer
        _buffer = []
    try:
        if not chunks:
            return
        samples = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
        if len(samples) < SAMPLE_RATE * MIN_AUDIO_SEC:
            print(f"[stt] too short ({len(samples)/SAMPLE_RATE:.2f}s)",
                  flush=True)
            return
        t0 = time.time()
        raw, language = _backend.transcribe(samples)
        elapsed = time.time() - t0
        if not raw:
            print(f"[stt] empty ({language}, {elapsed:.2f}s)", flush=True)
            return
        text = post_process(raw)

        # Set clipboard, then Ctrl+V — atomic paste, no per-char IME drama.
        # Tiny sleep lets the clipboard write settle before the keystroke
        # (otherwise Ctrl+V can race ahead and paste empty/stale content).
        set_clipboard(text)
        time.sleep(0.15)
        paste_clipboard()

        try:
            print(f"[stt] {language} {elapsed:.2f}s -> {text}", flush=True)
        except Exception:
            print(f"[stt] inserted ({elapsed:.2f}s); log encoding failed",
                  flush=True)
    except Exception as e:
        print(f"[stt] error: {e}", file=sys.stderr, flush=True)
    finally:
        with _state_lock:
            _processing = False


# ---------------------------------------------------------------------------
# Keyboard hooks
# ---------------------------------------------------------------------------
def _on_press(key) -> None:
    global _recording, _active_trigger
    if key not in TRIGGER_KEYS:
        return
    with _state_lock:
        if _active_trigger is not None:
            return  # another trigger is already held (or OS key-repeat)
        _active_trigger = key
        _recording = True
        _buffer.clear()
    print(f"[stt] REC ({key})", flush=True)


def _on_release(key) -> None:
    global _recording, _active_trigger
    if key not in TRIGGER_KEYS:
        return
    with _state_lock:
        if _active_trigger != key:
            return  # releasing a non-active trigger (e.g. tap of the other one)
        _active_trigger = None
        _recording = False
    print("[stt] processing...", flush=True)
    threading.Thread(target=_transcribe_and_emit, daemon=True).start()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    global _backend

    print(f"[stt] NVIDIA DLL dirs registered: {_NVIDIA_COUNT}", flush=True)
    print(f"[stt] backend: {STT_BACKEND} | model: {STT_MODEL}", flush=True)

    _backend = build_backend(STT_BACKEND, STT_MODEL)

    print(f"[stt] warming up on {_backend.device_label}...", flush=True)
    t0 = time.time()
    _backend.warmup()
    trigger_labels = ", ".join(str(k) for k in TRIGGER_KEYS)
    print(f"[stt] warmup {time.time()-t0:.1f}s — hold {trigger_labels} to record.",
          flush=True)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=_audio_callback,
        blocksize=int(SAMPLE_RATE * 0.05),  # 50 ms chunks
    )
    stream.start()
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()
    try:
        listener.join()
    except KeyboardInterrupt:
        print("[stt] bye", flush=True)
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
