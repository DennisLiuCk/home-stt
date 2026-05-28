"""home-stt Web UI — single-file Gradio application.

Assembles all tab modules (Dashboard, Playground, Settings, Guide,
Diagnostics) into one gr.Blocks app with a unified header, footer,
and custom CSS.

Launch:
    python scripts/stt_web.py                   # default port 7860
    python scripts/stt_web.py --port 8080       # custom port
    home-stt web                                # via CLI entry point
    home-stt web --port 8080 --share            # with Gradio share link
"""
from __future__ import annotations

import io
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Windows DLL bootstrap — must run BEFORE `import gradio`
# ---------------------------------------------------------------------------
# Gradio's import chain invalidates os.add_dll_directory() handles on Windows,
# so torch DLLs can't be found if torch is imported AFTER gradio. Fix: register
# DLL dirs and import torch eagerly before gradio loads.

_torch = None

def _bootstrap_torch():
    global _torch
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        try:
            import torch
            _torch = torch
        except (ImportError, OSError):
            pass
        return
    try:
        import importlib.util
        import site
        spec = importlib.util.find_spec("torch")
        if spec and spec.submodule_search_locations:
            torch_dir = spec.submodule_search_locations[0]
            for sub in ("lib", "bin"):
                d = os.path.join(torch_dir, sub)
                if os.path.isdir(d):
                    os.add_dll_directory(d)
        user_sp = site.getusersitepackages()
        nvidia_base = os.path.join(user_sp, "nvidia")
        if os.path.isdir(nvidia_base):
            for pkg in os.listdir(nvidia_base):
                for sub in ("lib", "bin"):
                    d = os.path.join(nvidia_base, pkg, sub)
                    if os.path.isdir(d):
                        os.add_dll_directory(d)
        for sp in site.getsitepackages():
            nv = os.path.join(sp, "nvidia")
            if os.path.isdir(nv):
                for pkg in os.listdir(nv):
                    for sub in ("lib", "bin"):
                        d = os.path.join(nv, pkg, sub)
                        if os.path.isdir(d):
                            os.add_dll_directory(d)
        import torch
        _torch = torch
    except (ImportError, OSError):
        pass

_bootstrap_torch()

import gradio as gr

# ---------------------------------------------------------------------------
# Path bootstrap — make sibling scripts importable
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from stt_state import read_state, STATE_FILE  # noqa: E402
from stt_config import (  # noqa: E402
    load_config,
    config_path,
    init_config,
    _DEFAULTS,
    generate_default_config,
)

logger = logging.getLogger("stt.web")

# ---------------------------------------------------------------------------
# Version helper
# ---------------------------------------------------------------------------

def _daemon_version() -> str:
    try:
        from importlib.metadata import version
        return version("home-stt")
    except Exception:
        pass
    daemon_script = _SCRIPTS_DIR / "stt-daemon.py"
    try:
        for line in daemon_script.read_text(encoding="utf-8").splitlines()[:200]:
            m = re.match(r'^__version__\s*=\s*["\']([^"\']+)["\']', line)
            if m:
                return m.group(1)
    except OSError:
        pass
    return "?"


# ═══════════════════════════════════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
/* ═══════════════════════════════════════════════════════════════════════
   home-stt · "Luminous Editorial"
   A bright, airy interface — warm paper canvas, jade accent, amber highlights,
   an elegant Fraunces wordmark over Manrope, floating cards, soft motion.
   ═══════════════════════════════════════════════════════════════════════ */
/* Fonts (Fraunces / Manrope / JetBrains Mono / Noto Sans TC / Noto Serif TC —
   the CJK serif for headings) are loaded via <link> tags in the document
   <head> (see _HEAD_HTML) — more robust than an @import here, which Gradio's
   stylesheet assembly can invalidate. */

:root {
    --stt-bg:         #FBFAF7;
    --stt-surface:    #FFFFFF;
    --stt-ink:        #22302C;
    --stt-muted:      #6E7B75;
    --stt-faint:      #9AA39C;
    --stt-border:     #ECE7DD;
    --stt-jade:       #12A48E;
    --stt-jade-deep:  #0E8C7C;
    --stt-jade-soft:  #E6F4F0;
    --stt-amber:      #C8852A;
    --stt-coral:      #E0533D;
    --stt-serif: 'Fraunces','Noto Serif TC', Georgia, 'Songti TC', serif;
    --stt-sans:  'Manrope','Noto Sans TC','PingFang TC','Microsoft JhengHei', ui-sans-serif, system-ui, sans-serif;
    --stt-mono:  'JetBrains Mono', ui-monospace, SFMono-Regular, monospace;
}

/* ── Atmosphere: a fixed, full-viewport glow behind everything ─────────── */
body { background: var(--stt-bg) !important; }
gradio-app, .gradio-container, .main, .wrap { background: transparent !important; }

body::before {
    content: ''; position: fixed; inset: 0; z-index: -2; pointer-events: none;
    background:
        radial-gradient(900px 520px at 8% -10%,  rgba(18,164,142,.12), transparent 60%),
        radial-gradient(820px 480px at 104% -6%, rgba(200,133,42,.10), transparent 55%),
        radial-gradient(760px 760px at 50% 122%, rgba(18,164,142,.07), transparent 60%),
        var(--stt-bg);
}
body::after {
    content: ''; position: fixed; inset: 0; z-index: -1; pointer-events: none; opacity: .55;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
    background-size: 140px 140px;
}

.gradio-container {
    max-width: 1160px !important;
    margin: 0 auto !important;
    padding: 0 1.4rem 3rem !important;
    font-family: var(--stt-sans) !important;
    color: var(--stt-ink);
}

/* Hide Gradio's default footer for a cleaner canvas */
.gradio-container > footer,
footer.svelte-1ax1toq, footer { display: none !important; }

/* ── Header ────────────────────────────────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 2.6rem 1rem 1.4rem;
    animation: sttRise .7s cubic-bezier(.22,1,.36,1) both;
}
.app-badge {
    width: 66px; height: 66px; margin: 0 auto 1.1rem;
    border-radius: 20px;
    display: flex; align-items: center; justify-content: center; gap: 4px;
    background: linear-gradient(145deg, #15B7A0 0%, #0E8C7C 100%);
    box-shadow: 0 12px 30px -10px rgba(16,140,124,.55),
                inset 0 1px 0 rgba(255,255,255,.35);
}
.app-badge .bar {
    width: 4px; border-radius: 4px; background: rgba(255,255,255,.92);
    animation: sttWave 1.1s ease-in-out infinite;
}
.app-badge .bar:nth-child(1) { height: 14px; animation-delay: 0s;    }
.app-badge .bar:nth-child(2) { height: 26px; animation-delay: .15s;  }
.app-badge .bar:nth-child(3) { height: 34px; animation-delay: .3s;   }
.app-badge .bar:nth-child(4) { height: 22px; animation-delay: .45s;  }
.app-badge .bar:nth-child(5) { height: 12px; animation-delay: .6s;   }

.app-eyebrow {
    font-family: var(--stt-sans);
    font-size: .72rem; font-weight: 700;
    letter-spacing: .42em; text-indent: .42em;
    text-transform: uppercase;
    color: var(--stt-jade-deep);
    opacity: .85; margin-bottom: .55rem;
}
.app-header h1, .app-wordmark {
    font-family: var(--stt-serif) !important;
    font-optical-sizing: auto;
    font-size: 3.05rem; line-height: 1; font-weight: 600;
    letter-spacing: -.01em; margin: 0 0 .5rem;
    background: linear-gradient(135deg, #1B3A33 0%, #0E8C7C 70%, #15B7A0 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
}
.app-header .subtitle {
    font-size: .98rem; font-weight: 500;
    color: var(--stt-muted); letter-spacing: .01em;
}
.app-header .rule {
    width: 54px; height: 2px; margin: 1.1rem auto 0;
    border-radius: 2px;
    background: linear-gradient(90deg, transparent, var(--stt-jade), transparent);
}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
.tabs { animation: sttRise .7s cubic-bezier(.22,1,.36,1) .08s both; }
.tab-nav {
    border-bottom: 1px solid var(--stt-border) !important;
    gap: .35rem !important; margin-bottom: 1.4rem !important;
    justify-content: center;
}
.tab-nav button {
    font-family: var(--stt-sans) !important;
    font-weight: 600 !important; font-size: .96rem !important;
    color: var(--stt-muted) !important;
    background: transparent !important; border: none !important;
    padding: .7rem 1.15rem !important; margin-bottom: -1px;
    border-radius: 12px 12px 0 0 !important;
    position: relative; transition: color .2s ease, background .2s ease;
}
.tab-nav button:hover { color: var(--stt-jade-deep) !important; background: rgba(18,164,142,.06) !important; }
.tab-nav button.selected { color: var(--stt-jade-deep) !important; background: transparent !important; }
.tab-nav button.selected::after {
    content: ''; position: absolute; left: 1.05rem; right: 1.05rem; bottom: -1px; height: 3px;
    border-radius: 3px 3px 0 0;
    background: linear-gradient(90deg, var(--stt-jade), var(--stt-jade-deep));
}
.tabitem { min-height: 440px; padding-top: .4rem !important; }

/* ── Cards / blocks ────────────────────────────────────────────────────── */
.block, .gr-group, .form, .gr-box {
    border-radius: 18px !important;
}
.gr-group, .gradio-group {
    background: var(--stt-surface) !important;
    border: 1px solid var(--stt-border) !important;
    box-shadow: 0 1px 2px rgba(34,48,44,.04), 0 14px 34px -22px rgba(34,48,44,.22) !important;
}

/* ── Accordions ────────────────────────────────────────────────────────── */
.gradio-accordion {
    border: 1px solid var(--stt-border) !important;
    border-radius: 16px !important;
    background: var(--stt-surface) !important;
    box-shadow: 0 1px 2px rgba(34,48,44,.03), 0 14px 30px -24px rgba(34,48,44,.20) !important;
    overflow: hidden;
}
.label-wrap {
    font-family: var(--stt-sans) !important;
    font-weight: 700 !important; font-size: 1rem !important;
    color: var(--stt-ink) !important;
    padding: .35rem 0 !important;
}
.label-wrap > span:first-child { display: inline-flex; align-items: center; }
.label-wrap > span:first-child::before {
    content: ''; display: inline-block;
    width: 4px; height: 1.05em; margin-right: .65rem;
    border-radius: 3px;
    background: linear-gradient(180deg, var(--stt-jade), var(--stt-jade-deep));
}

/* ── Buttons ───────────────────────────────────────────────────────────── */
.gradio-container button.primary,
.gradio-container button.secondary,
.gradio-container button.stop {
    font-family: var(--stt-sans) !important;
    font-weight: 700 !important; letter-spacing: .01em;
    border-radius: 12px !important;
    transition: transform .15s ease, box-shadow .22s ease, filter .2s ease !important;
}
.gradio-container button.primary {
    background: linear-gradient(135deg, #15B7A0 0%, #0E8C7C 100%) !important;
    border: none !important; color: #fff !important;
    box-shadow: 0 8px 20px -8px rgba(16,140,124,.6) !important;
}
.gradio-container button.primary:hover { transform: translateY(-1px); filter: brightness(1.04); box-shadow: 0 12px 26px -8px rgba(16,140,124,.7) !important; }
.gradio-container button.secondary {
    background: var(--stt-surface) !important;
    border: 1px solid var(--stt-border) !important;
    color: var(--stt-ink) !important;
}
.gradio-container button.secondary:hover { transform: translateY(-1px); border-color: var(--stt-jade) !important; color: var(--stt-jade-deep) !important; background: var(--stt-jade-soft) !important; }
.gradio-container button.stop {
    background: linear-gradient(135deg, #F26B53 0%, #E0533D 100%) !important;
    border: none !important; color: #fff !important;
    box-shadow: 0 8px 20px -8px rgba(224,83,61,.55) !important;
}
.gradio-container button.stop:hover { transform: translateY(-1px); filter: brightness(1.04); }
.gradio-container button:active { transform: translateY(0) !important; }

/* ── Unified button icons (custom line set, inherits the button text colour) ─ */
.gradio-container button.btn-ico { display: inline-flex !important; align-items: center; justify-content: center; }
.gradio-container button.btn-ico::before {
    content: ''; width: 1.12em; height: 1.12em; margin-right: .55em; flex: 0 0 auto;
    background-color: currentColor;
    -webkit-mask: var(--btn-ico) center / contain no-repeat;
            mask: var(--btn-ico) center / contain no-repeat;
}
.gradio-container button.i-play     { --btn-ico: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M7 5l12 7-12 7z' fill='%23000'/%3E%3C/svg%3E"); }
.gradio-container button.i-stop     { --btn-ico: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect x='6' y='6' width='12' height='12' rx='2.5' fill='%23000'/%3E%3C/svg%3E"); }
.gradio-container button.i-restart  { --btn-ico: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 12a9 9 0 1 1-2.6-6.4'/%3E%3Cpath d='M21 4v5h-5'/%3E%3C/svg%3E"); }
.gradio-container button.i-refresh  { --btn-ico: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 12a9 9 0 0 1 14.8-6.9L21 8'/%3E%3Cpath d='M21 3v5h-5'/%3E%3Cpath d='M21 12a9 9 0 0 1-14.8 6.9L3 16'/%3E%3Cpath d='M3 21v-5h5'/%3E%3C/svg%3E"); }
.gradio-container button.i-wave     { --btn-ico: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round'%3E%3Cpath d='M4 10v4'/%3E%3Cpath d='M9 6v12'/%3E%3Cpath d='M14 8v8'/%3E%3Cpath d='M19 11v2'/%3E%3C/svg%3E"); }
.gradio-container button.i-edit     { --btn-ico: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 20h9'/%3E%3Cpath d='M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4z'/%3E%3C/svg%3E"); }
.gradio-container button.i-polish   { --btn-ico: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linejoin='round'%3E%3Cpath d='M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z'/%3E%3C/svg%3E"); }
.gradio-container button.i-save     { --btn-ico: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z'/%3E%3Cpath d='M17 21v-8H7v8'/%3E%3Cpath d='M7 3v5h7'/%3E%3C/svg%3E"); }
.gradio-container button.i-reset    { --btn-ico: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9 14L4 9l5-5'/%3E%3Cpath d='M4 9h11a5 5 0 0 1 0 10h-3'/%3E%3C/svg%3E"); }
.gradio-container button.i-activity { --btn-ico: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 12h4l3 8 4-16 3 8h4'/%3E%3C/svg%3E"); }

/* ── Fix: keep leading bold glyphs from clipping at the content edge ─────── */
.gradio-container .prose, .gradio-container .md { padding-left: 4px !important; overflow: visible !important; }
.gradio-container .prose :is(p, li, strong, h1, h2, h3, h4),
.gradio-container .md :is(p, li, strong, h1, h2, h3, h4) { overflow: visible; }

/* ── Inputs ────────────────────────────────────────────────────────────── */
textarea, input[type=text], input[type=number], .gr-input {
    font-family: var(--stt-sans) !important;
    border-radius: 12px !important;
}
textarea:focus, input[type=text]:focus, input[type=number]:focus {
    border-color: var(--stt-jade) !important;
    box-shadow: 0 0 0 3px rgba(18,164,142,.16) !important;
}
.gr-input-label, label span { font-family: var(--stt-sans) !important; }

/* ── Markdown / prose ──────────────────────────────────────────────────── */
.prose, .md { font-family: var(--stt-sans) !important; color: var(--stt-ink); }
.prose h1, .prose h2, .prose h3, .md h1, .md h2, .md h3 {
    font-family: var(--stt-serif) !important;
    font-weight: 600 !important; letter-spacing: -.01em;
    color: var(--stt-ink) !important;
}
.prose h2, .md h2 { font-size: 1.7rem !important; margin-top: .3rem; }
.prose h3, .md h3 { font-size: 1.18rem !important; }
.prose a, .md a { color: var(--stt-jade-deep) !important; text-decoration-color: rgba(18,164,142,.4); text-underline-offset: 3px; }
.prose a:hover, .md a:hover { color: var(--stt-jade) !important; }
.prose hr, .md hr {
    border: none !important; height: 1px !important; margin: 1.6rem 0 !important;
    background: linear-gradient(90deg, transparent, var(--stt-border) 18%, var(--stt-border) 82%, transparent) !important;
}
.prose code, .md code {
    font-family: var(--stt-mono) !important; font-size: .86em;
    background: #F1EDE4 !important; color: var(--stt-jade-deep) !important;
    padding: .12em .42em !important; border-radius: 6px !important;
}
.prose pre, .md pre {
    background: #1F2A28 !important; border-radius: 14px !important;
    border: 1px solid rgba(0,0,0,.06);
}
.prose pre code, .md pre code { background: transparent !important; color: #E8EEE9 !important; }
.prose blockquote, .md blockquote {
    border-left: 3px solid var(--stt-jade) !important;
    background: var(--stt-jade-soft) !important;
    border-radius: 0 10px 10px 0; padding: .55rem .95rem !important;
    color: var(--stt-ink) !important; font-style: normal;
}
.prose table, .md table { border-radius: 12px !important; overflow: hidden; border: 1px solid var(--stt-border); }
.prose thead th, .md thead th { background: var(--stt-jade-soft) !important; color: var(--stt-jade-deep) !important; font-weight: 700 !important; }

/* ── Status banner (Dashboard) ─────────────────────────────────────────── */
.stt-banner {
    display: flex; align-items: center; gap: .85rem;
    padding: 1rem 1.35rem; border-radius: 16px;
    font-family: var(--stt-sans);
    border: 1px solid var(--stt-border);
    background: var(--stt-surface);
    box-shadow: 0 1px 2px rgba(34,48,44,.04), 0 16px 34px -24px rgba(34,48,44,.28);
}
.stt-banner__dot {
    width: 12px; height: 12px; border-radius: 50%; flex: 0 0 auto;
    background: var(--stt-faint);
}
.stt-banner__caption {
    font-size: .74rem; font-weight: 700; letter-spacing: .22em; text-transform: uppercase;
    color: var(--stt-muted);
}
.stt-banner__value {
    margin-left: auto;
    font-size: 1.18rem; font-weight: 700; letter-spacing: .02em;
    color: var(--stt-ink);
}
.stt-banner--idle       { background: linear-gradient(180deg,#F3F7F4,#FFFFFF); }
.stt-banner--idle       .stt-banner__dot   { background: #5E8C80; }
.stt-banner--idle       .stt-banner__value { color: #2C3E38; }
.stt-banner--recording  { background: linear-gradient(180deg,#FDEEEB,#FFFFFF); border-color:#F4D2CB; }
.stt-banner--recording  .stt-banner__dot   { background: var(--stt-coral); animation: sttPulse 1.4s ease-out infinite; }
.stt-banner--recording  .stt-banner__value { color: #9A3422; }
.stt-banner--processing { background: linear-gradient(180deg,#FBF3E3,#FFFFFF); border-color:#F0DEB6; }
.stt-banner--processing .stt-banner__dot   { background: var(--stt-amber); animation: sttPulse 1.6s ease-out infinite; }
.stt-banner--processing .stt-banner__value { color: #7A521A; }
.stt-banner--stopped    { background: linear-gradient(180deg,#F2F0EA,#FFFFFF); }
.stt-banner--stopped    .stt-banner__dot   { background: #9A958A; }
.stt-banner--stopped    .stt-banner__value { color: #50574E; }

/* ── Footer ────────────────────────────────────────────────────────────── */
.app-footer {
    text-align: center;
    margin-top: 2.4rem; padding: 1.4rem 1rem 0;
    font-family: var(--stt-mono);
    font-size: .76rem; letter-spacing: .04em;
    color: var(--stt-faint);
    border-top: 1px solid var(--stt-border);
}
.app-footer .sep { color: var(--stt-jade); opacity: .7; margin: 0 .6rem; }

/* ── Scrollbar & selection ─────────────────────────────────────────────── */
::selection { background: rgba(18,164,142,.20); color: var(--stt-ink); }
*::-webkit-scrollbar { width: 11px; height: 11px; }
*::-webkit-scrollbar-thumb {
    background: #D8D2C6; border-radius: 9px;
    border: 3px solid transparent; background-clip: content-box;
}
*::-webkit-scrollbar-thumb:hover { background: var(--stt-jade); background-clip: content-box; }
*::-webkit-scrollbar-track { background: transparent; }

/* ── Keyframes ─────────────────────────────────────────────────────────── */
@keyframes sttRise  { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
@keyframes sttWave  { 0%,100% { transform: scaleY(.35); } 50% { transform: scaleY(1); } }
@keyframes sttPulse {
    0%   { box-shadow: 0 0 0 0 rgba(224,83,61,.5); }
    70%  { box-shadow: 0 0 0 10px rgba(224,83,61,0); }
    100% { box-shadow: 0 0 0 0 rgba(224,83,61,0); }
}

@media (prefers-reduced-motion: reduce) {
    .app-header, .tabs, .stt-banner { animation: none !important; }
    .app-badge .bar, .stt-banner__dot { animation: none !important; }
    .gradio-container button:hover { transform: none !important; }
}
"""


# ── Document <head>: load fonts up-front, then pin a bright (light) palette ──
# Fonts load via <link> (not a CSS @import) so they resolve regardless of how
# Gradio assembles the injected stylesheet. The script pins Gradio's `__theme`
# query param to "light" (one redirect) so the "明亮優雅" look holds regardless
# of the OS dark-mode setting.
_HEAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+TC:wght@400;500;700&family=Noto+Serif+TC:wght@400;600;700&display=swap">
<script>
(function () {
    try {
        var u = new URL(window.location.href);
        if (u.searchParams.get('__theme') !== 'light') {
            u.searchParams.set('__theme', 'light');
            window.location.replace(u.toString());
        }
    } catch (e) {}
})();
</script>
"""


def _build_theme() -> gr.themes.Base:
    """A warm, luminous light theme: jade primary, amber accents, paper neutrals."""
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.amber,
        neutral_hue=gr.themes.colors.stone,
        spacing_size=gr.themes.sizes.spacing_lg,
        radius_size=gr.themes.sizes.radius_lg,
        text_size=gr.themes.sizes.text_md,
        font=[
            gr.themes.GoogleFont("Manrope"),
            "Noto Sans TC", "PingFang TC", "Microsoft JhengHei",
            "ui-sans-serif", "system-ui", "sans-serif",
        ],
        font_mono=[
            gr.themes.GoogleFont("JetBrains Mono"),
            "ui-monospace", "SFMono-Regular", "monospace",
        ],
    ).set(
        # Canvas & surfaces
        body_background_fill="#FBFAF7",
        background_fill_primary="#FFFFFF",
        background_fill_secondary="#F6F3EC",
        block_background_fill="#FFFFFF",
        panel_background_fill="#FFFFFF",
        # Text
        body_text_color="#22302C",
        body_text_color_subdued="#6E7B75",
        block_title_text_color="#22302C",
        block_title_text_weight="700",
        block_label_text_color="#6E7B75",
        # Borders & radii
        border_color_primary="#ECE7DD",
        block_border_color="#ECE7DD",
        block_border_width="1px",
        input_border_color="#E4DED2",
        block_radius="18px",
        input_radius="12px",
        # Shadows
        block_shadow="0 1px 2px rgba(34,48,44,0.04), 0 14px 34px -22px rgba(34,48,44,0.22)",
        # Primary buttons (jade gradient)
        button_primary_background_fill="linear-gradient(135deg, #15B7A0 0%, #0E8C7C 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #18C6AD 0%, #109A88 100%)",
        button_primary_text_color="#FFFFFF",
        button_primary_border_color="rgba(0,0,0,0)",
        # Secondary buttons (paper)
        button_secondary_background_fill="#FFFFFF",
        button_secondary_background_fill_hover="#F4F1EA",
        button_secondary_text_color="#2B3A35",
        button_secondary_border_color="#E2DCD0",
        # Inputs
        input_background_fill="#FCFBF8",
        input_background_fill_focus="#FFFFFF",
        # Accents & links
        color_accent_soft="#E6F4F0",
        link_text_color="#0E8C7C",
        link_text_color_hover="#12A48E",
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard (儀表板)
# ═══════════════════════════════════════════════════════════════════════════

PID_FILE = _SCRIPTS_DIR / "stt-daemon.pid"
HOME_STT_PY = _SCRIPTS_DIR / "home_stt.py"

_STATE_LABELS = {
    "idle":       "閒置",
    "recording":  "錄音中",
    "processing": "處理中",
    "stopped":    "已停止",
}

def _banner_html(state: str, label: str) -> str:
    """Render the dashboard status banner as a styled pill (see .stt-banner CSS)."""
    key = state if state in _STATE_LABELS else "stopped"
    return (
        f'<div class="stt-banner stt-banner--{key}">'
        f'<span class="stt-banner__dot"></span>'
        f'<span class="stt-banner__caption">系統狀態</span>'
        f'<span class="stt-banner__value">{label}</span>'
        f'</div>'
    )

_TRANSCRIBE_PAT = re.compile(
    r"\[stt\]\s+(zh|en|ja|ko|EDIT|voice-edit|empty|too short|silent)"
)
_STARTUP_BACKEND_PAT = re.compile(
    r"\[stt\]\s+backend:\s+(.+?)\s+\|\s+model:\s+(.+?)\s*$"
)
_STARTUP_POLISH_PAT = re.compile(r"\[stt\]\s+polish:\s+(.+?)\s*$")


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} 秒"
    if seconds < 3600:
        return f"{int(seconds // 60)} 分 {int(seconds % 60)} 秒"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h} 小時 {m} 分"


def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    if sys.platform == "win32":
        try:
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True, text=True, timeout=3,
            )
            return pid if str(pid) in r.stdout else None
        except Exception:
            return None
    try:
        os.kill(pid, 0)
        return pid
    except (OSError, PermissionError):
        return None


def _rss_mb(pid: int) -> float | None:
    try:
        if sys.platform == "win32":
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True, text=True, timeout=3,
            )
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
    except Exception:
        return None


def _read_log_lines(max_bytes: int = 64 * 1024) -> list[str]:
    log_path = Path(tempfile.gettempdir()) / "stt-daemon.log"
    try:
        size = log_path.stat().st_size
        with open(log_path, encoding="utf-8", errors="replace") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()
            return f.readlines()
    except OSError:
        return []


def _recent_transcribes(lines: list[str], n: int = 10) -> list[str]:
    out: list[str] = []
    for line in reversed(lines):
        if _TRANSCRIBE_PAT.search(line):
            out.append(line.rstrip())
            if len(out) == n:
                break
    return list(reversed(out))


def _parse_startup_info(lines: list[str]) -> dict[str, str]:
    info: dict[str, str] = {}
    for line in lines[:80]:
        m = _STARTUP_BACKEND_PAT.search(line)
        if m and "backend" not in info:
            info["backend"] = f"{m.group(1).strip()} ({m.group(2).strip()})"
            continue
        m = _STARTUP_POLISH_PAT.search(line)
        if m and "polish" not in info:
            info["polish"] = m.group(1).strip()
    return info


def _get_gpu_info() -> str:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "Apple Silicon (Metal)"
    if _torch is not None:
        try:
            if _torch.cuda.is_available():
                name = _torch.cuda.get_device_name(0)
                props = _torch.cuda.get_device_properties(0)
                vram_gb = props.total_memory / (1024 ** 3)
                return f"{name}  ({vram_gb:.1f} GB VRAM)"
            return "CUDA 不可用 (CPU 模式)"
        except Exception as e:
            return f"GPU 查詢失敗：{e}"
    return "torch 未載入（DLL 或安裝問題）"


def _poll_dashboard():
    """Return (banner_html, status_md, sys_info_md, recent_md)."""
    data = read_state()
    state = data["state"] if data else "stopped"
    label = _STATE_LABELS.get(state, state)
    banner_html = _banner_html(state, label)

    # Status markdown
    pid = _read_pid()
    lines: list[str] = []
    lines.append(f"**狀態**　{label}")

    if data and data.get("edit_mode"):
        lines.append("**模式**　Voice-Edit")
    elif state not in ("stopped",):
        lines.append("**模式**　Dictate")

    if pid is not None:
        lines.append(f"**PID**　`{pid}`")
        try:
            uptime_s = time.time() - PID_FILE.stat().st_mtime
            lines.append(f"**運行時間**　{_format_duration(uptime_s)}")
        except OSError:
            pass
        rss = _rss_mb(pid)
        if rss is not None:
            rss_str = f"{rss / 1024:.2f} GB" if rss >= 1024 else f"{rss:.0f} MB"
            lines.append(f"**記憶體 (RSS)**　{rss_str}")
    else:
        lines.append("**PID**　— (未運行)")

    if data and data.get("last_text"):
        last = data["last_text"][:80] + ("..." if len(data["last_text"]) > 80 else "")
        lang = data.get("last_lang", "")
        lang_tag = f" `[{lang}]`" if lang else ""
        lines.append(f"**最後轉錄**{lang_tag}　{last}")

    status_md = "\n\n".join(lines)

    # System info
    sys_lines: list[str] = []
    py_ver = sys.version.split()[0]
    sys_lines.append(f"**Python**　`{py_ver}`")
    sys_lines.append(f"**平台**　`{sys.platform}` / {platform.machine()}")
    sys_lines.append(f"**GPU / 加速**　{_get_gpu_info()}")

    log_lines = _read_log_lines()
    startup_info = _parse_startup_info(log_lines)
    cfg = None
    if not startup_info.get("backend") or not startup_info.get("polish"):
        try:
            cfg = load_config()
        except Exception:
            pass

    if startup_info.get("backend"):
        sys_lines.append(f"**STT Backend**　`{startup_info['backend']}`")
    elif cfg:
        backend = cfg.get("stt_backend") or "qwen3-asr (預設)"
        model = cfg.get("stt_model") or "預設"
        sys_lines.append(f"**STT Backend**　`{backend}` / `{model}`")
    else:
        sys_lines.append("**STT Backend**　—")

    if startup_info.get("polish"):
        sys_lines.append(f"**Polish 模型**　`{startup_info['polish']}`")
    elif cfg:
        pm = cfg.get("polish_model") or "預設"
        pe = cfg.get("polish_enabled", True)
        sys_lines.append(f"**Polish**　{'啟用' if pe else '停用'} / `{pm}`")
    else:
        sys_lines.append("**Polish**　—")

    cfg_p = config_path()
    sys_lines.append(f"**Config 路徑**　`{cfg_p}`")

    log_path = Path(tempfile.gettempdir()) / "stt-daemon.log"
    if log_path.exists():
        try:
            age = time.time() - log_path.stat().st_mtime
            sys_lines.append(f"**Log 最後寫入**　{_format_duration(age)} 前")
        except OSError:
            pass

    sys_info_md = "\n\n".join(sys_lines)

    # Recent transcriptions
    recent = _recent_transcribes(log_lines, n=10)
    if recent:
        recent_md = "```\n" + "\n".join(recent) + "\n```"
    else:
        recent_md = "_尚無轉錄記錄_"

    return banner_html, status_md, sys_info_md, recent_md


def _run_home_stt(*args: str, timeout: int = 30) -> str:
    cmd = [sys.executable, str(HOME_STT_PY)] + list(args)
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        combined = "\n".join(filter(None, [out, err]))
        return combined or "(指令執行完畢，無輸出)"
    except subprocess.TimeoutExpired:
        return f"指令超時 ({timeout}s)"
    except Exception as e:
        return f"執行失敗：{e}"


def _do_start():
    return _run_home_stt("start", timeout=60)


def _do_stop():
    return _run_home_stt("stop", timeout=15)


def _do_restart():
    return _run_home_stt("restart", timeout=90)


def build_dashboard_tab():
    with gr.Tab("儀表板", id="dashboard"):
        banner = gr.HTML(
            value=_banner_html("stopped", "載入中…"),
            label=None,
        )

        with gr.Row():
            btn_start   = gr.Button("啟動 Daemon",  variant="primary",   scale=1, elem_classes=["btn-ico", "i-play"])
            btn_stop    = gr.Button("停止 Daemon",  variant="stop",      scale=1, elem_classes=["btn-ico", "i-stop"])
            btn_restart = gr.Button("重啟 Daemon",  variant="secondary", scale=1, elem_classes=["btn-ico", "i-restart"])
            btn_refresh = gr.Button("立即重新整理", variant="secondary", scale=1, elem_classes=["btn-ico", "i-refresh"])

        ctrl_output = gr.Textbox(
            label="指令輸出",
            lines=4,
            max_lines=8,
            interactive=False,
                        placeholder="按下按鈕後，此處顯示執行結果...",
        )

        gr.Markdown("---")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.Markdown("### 即時狀態")
                status_md = gr.Markdown("_載入中..._")
            with gr.Column(scale=1):
                gr.Markdown("### 系統資訊")
                sys_info_md = gr.Markdown("_載入中..._")

        gr.Markdown("---")

        with gr.Accordion("最近轉錄記錄（最多 10 筆）", open=True):
            recent_md = gr.Markdown("_載入中..._")

        timer = gr.Timer(2)

        timer.tick(
            fn=_poll_dashboard,
            inputs=[],
            outputs=[banner, status_md, sys_info_md, recent_md],
        )
        btn_refresh.click(
            fn=_poll_dashboard,
            inputs=[],
            outputs=[banner, status_md, sys_info_md, recent_md],
        )
        btn_start.click(fn=_do_start, inputs=[], outputs=[ctrl_output])
        btn_stop.click(fn=_do_stop, inputs=[], outputs=[ctrl_output])
        btn_restart.click(fn=_do_restart, inputs=[], outputs=[ctrl_output])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Playground (試驗場)
# ═══════════════════════════════════════════════════════════════════════════

_POLISH_PROMPT = (
    "把口語逐字稿做最小修飾。原有標點(。？！，)完整保留。\n"
    "只移除贅字(呃、嗯、就是、那個、然後、嘛、啊)、修立即重複(我我我→我)、補必要標點。\n"
    "嚴禁:翻譯英文(commit/push/function 等保留)、改動詞、替換陌生詞(看似錯字也照樣輸出)、加新詞、改句式、刪除或替換原有標點。\n"
    "中文一律繁體。只輸出修飾後文字,不解釋、不加引號、不加前綴。\n\n"
    "範例 1:\n"
    "輸入:呃我覺得這個 Python function 可以再優化\n"
    "輸出:我覺得這個 Python function 可以再優化\n\n"
    "範例 2:\n"
    "輸入:我剛剛測試了一下。發現一個問題。\n"
    "輸出:我剛剛測試了一下。發現一個問題。"
)

_GPU_WARNING = (
    "首次使用會載入模型（需要 GPU 記憶體）。建議先停止 daemon 以釋放 GPU。"
)

_EXAMPLES_EDIT = [
    (
        "這個 Python function 的邏輯很複雜，我需要重構一下。",
        "幫我加上一段簡短的說明注釋",
    ),
    (
        "The deployment pipeline failed due to missing environment variables.",
        "translate to Chinese",
    ),
    (
        "我們明天要開會討論 Q3 的 roadmap。",
        "改成更正式的書面語",
    ),
]

_EXAMPLES_POLISH = [
    "呃呃那個我覺得這個 commit 可以可以再拆小一點",
    "嗯就是然後我想說那個這個 function 需要加 unit test 啦",
    "我剛剛測試了一下。發現一個問題。這個 bug 要修掉。",
]


def _effective_config() -> dict:
    try:
        cfg = load_config()
    except Exception:
        cfg = {}
    return {
        "stt_backend": cfg.get("stt_backend") or "qwen3-asr",
        "stt_model": cfg.get("stt_model") or "Qwen/Qwen3-ASR-0.6B",
        "polish_enabled": cfg.get("polish_enabled") if cfg.get("polish_enabled") is not None else True,
        "polish_model": cfg.get("polish_model") or "Qwen/Qwen3-4B-Instruct-2507",
        "sample_rate": cfg.get("sample_rate") or 16000,
    }


_LOAD_FAILED = "LOAD_FAILED"


def _ensure_models(model_state: dict) -> tuple[dict, str]:
    if model_state.get("_loaded"):
        return model_state, ""

    from stt_backends import build_backend
    from text_polisher import build_polisher

    cfg = _effective_config()
    errors: list[str] = []
    polish_prompt = cfg.get("polish_prompt") or _POLISH_PROMPT

    backend = None
    try:
        backend = build_backend(
            cfg["stt_backend"],
            cfg["stt_model"],
            cfg["sample_rate"],
        )
        backend.warmup()
    except OSError as exc:
        logger.warning("playground: STT backend DLL load failed: %s", exc)
        errors.append(
            f"STT 後端 DLL 載入失敗：{exc}\n"
            "可能原因：torch CUDA wheel 未正確安裝，或 CUDA toolkit 版本不符。\n"
            "請嘗試：pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )
    except Exception as exc:
        logger.warning("playground: STT backend load failed: %s", exc)
        errors.append(f"STT 後端載入失敗：{exc}")

    polisher = None
    try:
        polisher = build_polisher(
            enabled=cfg["polish_enabled"],
            model_name=cfg["polish_model"],
            system_prompt=polish_prompt,
        )
    except OSError as exc:
        logger.warning("playground: polisher DLL load failed: %s", exc)
        errors.append(f"Polish 模型 DLL 載入失敗：{exc}")
    except Exception as exc:
        logger.warning("playground: polisher load failed: %s", exc)
        errors.append(f"Polish 模型載入失敗：{exc}")

    new_state = {"backend": backend, "polisher": polisher, "_loaded": True}
    err_msg = "\n".join(errors) if errors else ""
    return new_state, err_msg


def _audio_to_float32_mono(audio_tuple, target_sr: int = 16000) -> np.ndarray | None:
    if audio_tuple is None:
        return None
    try:
        sr, data = audio_tuple
        if data.dtype != np.float32:
            if np.issubdtype(data.dtype, np.integer):
                max_val = float(np.iinfo(data.dtype).max)
                data = data.astype(np.float32) / max_val
            else:
                data = data.astype(np.float32)
        if data.ndim == 2:
            data = data.mean(axis=1)
        elif data.ndim != 1:
            return None
        if len(data) == 0:
            return None
        if sr != target_sr and sr > 0:
            new_len = max(1, int(len(data) * target_sr / sr))
            indices = np.linspace(0, len(data) - 1, new_len)
            data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
        return data
    except Exception as exc:
        logger.warning("audio conversion failed: %s", exc)
        return None


def _run_transcribe(audio_input, model_state: dict):
    from stt_audio import post_process

    if audio_input is None:
        return "(請先錄音或上傳音訊)", "", "", "", "", model_state

    model_state, load_err = _ensure_models(model_state)
    backend = model_state.get("backend")
    if backend is None:
        err = load_err or "STT 後端載入失敗，請查看 log。"
        return f"錯誤：{err}", "", "", "", "", model_state

    cfg = _effective_config()
    samples = _audio_to_float32_mono(audio_input, target_sr=cfg["sample_rate"])
    if samples is None or len(samples) == 0:
        return "(音訊轉換失敗或音訊為空)", "", "", "", "", model_state

    t0 = time.perf_counter()
    try:
        raw_text, lang = backend.transcribe(samples)
    except Exception as exc:
        return f"轉錄失敗：{exc}", "", "", "", "", model_state
    t_stt = time.perf_counter() - t0

    t1 = time.perf_counter()
    try:
        postproc_text = post_process(raw_text)
    except Exception:
        postproc_text = raw_text
    t_post = time.perf_counter() - t1

    polisher = model_state.get("polisher")
    t2 = time.perf_counter()
    if polisher is not None:
        try:
            polished_text = polisher.polish(postproc_text)
        except Exception:
            polished_text = postproc_text
    else:
        polished_text = postproc_text
    t_polish = time.perf_counter() - t2

    total_ms = (t_stt + t_post + t_polish) * 1000
    timing = (
        f"STT: {t_stt*1000:.0f} ms | "
        f"後處理: {t_post*1000:.0f} ms | "
        f"Polish: {t_polish*1000:.0f} ms | "
        f"總計: {total_ms:.0f} ms"
    )
    lang_display = lang or "(未知)"
    return raw_text, postproc_text, polished_text, lang_display, timing, model_state


def _run_voice_edit(selection_text: str, instruction_audio, instruction_text: str, model_state: dict):
    from stt_audio import post_process

    if not selection_text or not selection_text.strip():
        return "", "請先輸入要編輯的選取文字。", model_state

    model_state, load_err = _ensure_models(model_state)
    if load_err:
        if model_state.get("backend") is None and model_state.get("polisher") is None:
            return "", f"模型載入錯誤：{load_err}", model_state
        if model_state.get("polisher") is None:
            return "", f"Polish 模型載入失敗（Voice-Edit 需要 LLM）：{load_err}", model_state

    instruction = ""
    if instruction_audio is not None:
        backend = model_state.get("backend")
        if backend is None:
            return "", "STT 後端未就緒，無法轉錄指令音訊。", model_state
        cfg = _effective_config()
        samples = _audio_to_float32_mono(instruction_audio, target_sr=cfg["sample_rate"])
        if samples is not None and len(samples) > 0:
            try:
                raw_instr, _ = backend.transcribe(samples)
                instruction = post_process(raw_instr)
            except Exception as exc:
                return "", f"指令音訊轉錄失敗：{exc}", model_state

    if not instruction:
        instruction = (instruction_text or "").strip()
    if not instruction:
        return "", "請提供編輯指令（錄音或文字均可）。", model_state

    polisher = model_state.get("polisher")
    if polisher is None:
        return "", "Polish 模型未就緒（請確認 polish_enabled = true 且模型已成功載入）。", model_state

    t0 = time.perf_counter()
    try:
        result = polisher.edit(selection_text.strip(), instruction)
    except Exception as exc:
        return "", f"編輯失敗：{exc}", model_state
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if result is None:
        return "", f"編輯失敗（模型回傳空結果）。耗時 {elapsed_ms:.0f} ms。", model_state

    status = f"編輯完成。耗時 {elapsed_ms:.0f} ms | 指令：{instruction}"
    return result, status, model_state


def _run_polish(input_text: str, model_state: dict):
    if not input_text or not input_text.strip():
        return "", "請先輸入要修飾的文字。", model_state

    model_state, load_err = _ensure_models(model_state)
    if load_err and model_state.get("polisher") is None:
        return "", f"模型載入錯誤：{load_err}", model_state

    polisher = model_state.get("polisher")
    if polisher is None:
        return "", "Polish 模型未就緒。", model_state

    t0 = time.perf_counter()
    try:
        result = polisher.polish(input_text.strip())
    except Exception as exc:
        return input_text, f"修飾失敗，返回原文：{exc}", model_state
    elapsed_ms = (time.perf_counter() - t0) * 1000

    status = f"修飾完成。耗時 {elapsed_ms:.0f} ms"
    return result, status, model_state


def _render_config_info() -> str:
    try:
        cfg = _effective_config()
    except Exception as exc:
        return f"**無法讀取設定：{exc}**"
    lines = [
        "| 設定項目 | 值 |",
        "|---|---|",
        f"| STT Backend | `{cfg['stt_backend']}` |",
        f"| STT Model | `{cfg['stt_model']}` |",
        f"| Polish Enabled | `{cfg['polish_enabled']}` |",
        f"| Polish Model | `{cfg['polish_model']}` |",
        f"| Sample Rate | `{cfg['sample_rate']} Hz` |",
    ]
    return "\n".join(lines)


def build_playground_tab():
    with gr.Tab("試驗場", id="playground"):
        model_state = gr.State(value={"backend": None, "polisher": None})

        gr.Markdown("## 試驗場")
        gr.Markdown(
            "在此頁面可以直接測試 STT 轉錄、Voice-Edit 語音編輯、以及文字 Polish，"
            "無需啟動 daemon。"
        )
        gr.Markdown(f"> {_GPU_WARNING}")

        # Section 1 — Voice-to-Text
        with gr.Accordion("語音轉文字 (STT)", open=True):
            gr.Markdown(
                "錄製或上傳音訊，點擊「**轉錄**」即可得到原始 ASR 結果、後處理結果與 Polish 結果。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    stt_audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="numpy",
                        label="音訊輸入（錄音 / 上傳）",
                    )
                    stt_transcribe_btn = gr.Button("轉錄", variant="primary", size="lg", elem_classes=["btn-ico", "i-wave"])

                with gr.Column(scale=2):
                    with gr.Row():
                        stt_lang_out = gr.Textbox(
                            label="偵測語言", interactive=False, scale=1,
                            max_lines=1, placeholder="—",
                        )
                        stt_timing_out = gr.Textbox(
                            label="處理時間", interactive=False, scale=3,
                            max_lines=1, placeholder="—",
                        )
                    stt_raw_out = gr.Textbox(
                        label="原始 ASR 輸出（Raw）", interactive=False, lines=3,
                        placeholder="轉錄結果將顯示於此...",
                    )
                    stt_postproc_out = gr.Textbox(
                        label="後處理結果（繁體化 + 間距）", interactive=False, lines=3,
                        placeholder="後處理結果將顯示於此...",
                    )
                    stt_polished_out = gr.Textbox(
                        label="Polish 結果（去除贅詞）", interactive=False, lines=3,
                        placeholder="Polish 結果將顯示於此...",
                    )

            stt_transcribe_btn.click(
                fn=_run_transcribe,
                inputs=[stt_audio_input, model_state],
                outputs=[stt_raw_out, stt_postproc_out, stt_polished_out,
                         stt_lang_out, stt_timing_out, model_state],
                show_progress="minimal",
            )

        # Section 2 — Voice-Edit
        with gr.Accordion("語音編輯 (Voice-Edit)", open=False):
            gr.Markdown(
                "選取要編輯的文字，並用語音（或文字）說出編輯指令。"
                "LLM 會依照指令修改選取文字。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    edit_selection_input = gr.Textbox(
                        label="選取的文字（Selection）", lines=5,
                        placeholder="貼上要編輯的文字...",
                    )
                    edit_instr_audio = gr.Audio(
                        sources=["microphone", "upload"], type="numpy",
                        label="語音指令（可選）",
                    )
                    edit_instr_text = gr.Textbox(
                        label="文字指令（備用）", lines=2,
                        placeholder="如未錄音，可在此輸入指令，例如：改成更正式的書面語",
                    )
                    edit_btn = gr.Button("編輯", variant="primary", size="lg", elem_classes=["btn-ico", "i-edit"])

                with gr.Column(scale=1):
                    edit_result_out = gr.Textbox(
                        label="編輯結果", lines=8, interactive=False,
                        placeholder="編輯結果將顯示於此...",
                    )
                    edit_status_out = gr.Textbox(
                        label="狀態", lines=2, interactive=False, placeholder="—",
                    )

            edit_btn.click(
                fn=_run_voice_edit,
                inputs=[edit_selection_input, edit_instr_audio, edit_instr_text, model_state],
                outputs=[edit_result_out, edit_status_out, model_state],
                show_progress="minimal",
            )

            gr.Markdown("**快速範例：**")
            with gr.Row():
                for _sel, _instr in _EXAMPLES_EDIT:
                    _label = _instr[:20] + ("..." if len(_instr) > 20 else "")

                    def _make_edit_filler(sel=_sel, instr=_instr):
                        def _fill():
                            return sel, instr
                        return _fill

                    gr.Button(_label, size="sm").click(
                        fn=_make_edit_filler(),
                        inputs=[],
                        outputs=[edit_selection_input, edit_instr_text],
                    )

        # Section 3 — Text Polish
        with gr.Accordion("文字修飾 (Polish)", open=False):
            gr.Markdown(
                "貼上口語逐字稿，點擊「**修飾**」去除贅詞、修正重複，同時保留原意。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    polish_input = gr.Textbox(
                        label="輸入文字（原始口語）", lines=6,
                        placeholder="貼上或輸入要修飾的文字...",
                    )
                    polish_btn = gr.Button("修飾", variant="primary", size="lg", elem_classes=["btn-ico", "i-polish"])
                    polish_status_out = gr.Textbox(
                        label="狀態", lines=1, interactive=False, placeholder="—",
                    )

                with gr.Column(scale=1):
                    polish_output = gr.Textbox(
                        label="修飾後結果", lines=6, interactive=False,
                        placeholder="修飾結果將顯示於此...",
                    )

            polish_btn.click(
                fn=_run_polish,
                inputs=[polish_input, model_state],
                outputs=[polish_output, polish_status_out, model_state],
                show_progress="minimal",
            )

            gr.Markdown("**快速範例：**")
            with gr.Row():
                for _ex in _EXAMPLES_POLISH:
                    _label = _ex[:18] + "..."

                    def _make_polish_filler(ex=_ex):
                        def _fill():
                            return ex
                        return _fill

                    gr.Button(_label, size="sm").click(
                        fn=_make_polish_filler(),
                        inputs=[],
                        outputs=[polish_input],
                    )

        # Section 4 — Model settings info
        with gr.Accordion("模型設定資訊", open=False):
            gr.Markdown("顯示目前從 config.toml 讀取的模型設定（不會啟動載入）。")
            gr.Markdown(_render_config_info())


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Settings (設定)
# ═══════════════════════════════════════════════════════════════════════════

_PRESETS: dict[str, dict[str, Any]] = {
    "Maximum（最高品質）": {
        "stt_backend": "qwen3-asr",
        "stt_model": "Qwen/Qwen3-ASR-0.6B",
        "polish_enabled": True,
        "polish_model": "Qwen/Qwen3-4B-Instruct-2507",
        "desc": "最高 STT 精度 + LLM Polish。需要 GPU（~10 GB VRAM）。",
    },
    "Balanced（均衡）": {
        "stt_backend": "qwen3-asr",
        "stt_model": "Qwen/Qwen3-ASR-0.6B",
        "polish_enabled": True,
        "polish_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "desc": "高品質 STT + 輕量 Polish 模型（~3 GB VRAM）。",
    },
    "Light（輕量）": {
        "stt_backend": "qwen3-asr",
        "stt_model": "Qwen/Qwen3-ASR-0.6B",
        "polish_enabled": False,
        "polish_model": "",
        "desc": "Qwen3-ASR（高品質 STT），停用 Polish。約 ~2 GB VRAM。",
    },
    "Mini（最小）": {
        "stt_backend": "faster-whisper",
        "stt_model": "small",
        "polish_enabled": False,
        "polish_model": "",
        "desc": "最小資源消耗，適合低配置設備。",
    },
}

_PRESET_CHOICES = ["（不套用預設）"] + list(_PRESETS.keys())

_BACKEND_MODEL_SUGGESTIONS: dict[str, str] = {
    "qwen3-asr":      "例：Qwen/Qwen3-ASR-0.6B",
    "faster-whisper":  "例：large-v3、medium、small、tiny",
    "mlx-whisper":     "例：mlx-community/whisper-large-v3-mlx",
}

_POLISH_MODEL_SUGGESTIONS = (
    "例（Windows/Linux）：Qwen/Qwen3-4B-Instruct-2507  "
    "例（macOS）：lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit"
)

_LANGUAGE_CHOICES = [
    ("中文 (zh)", "zh"),
    ("英文 (en)", "en"),
    ("日文 (ja)", "ja"),
    ("韓文 (ko)", "ko"),
    ("法文 (fr)", "fr"),
    ("德文 (de)", "de"),
    ("西班牙文 (es)", "es"),
]


def _load_cfg() -> dict[str, Any]:
    try:
        return load_config()
    except Exception:
        return dict(_DEFAULTS)


def _list_to_str(value: list[str] | None) -> str:
    if not value:
        return ""
    return ", ".join(value)


def _str_to_list(text: str) -> list[str] | None:
    parts = [s.strip() for s in text.replace(",", " ").split() if s.strip()]
    return parts if parts else None


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(v, list):
        items = ", ".join(_toml_value(i) for i in v)
        return f"[{items}]"
    if v is None:
        return '""'
    return str(v)


def _build_toml(values: dict[str, Any]) -> str:
    cpath = config_path()
    lines: list[str] = [
        "# home-stt configuration file",
        f"# 設定檔位置：{cpath}",
        "#",
        "# 優先順序：環境變數 (HOME_STT_*) > 此檔案 > 程式預設值",
        "",
    ]

    def _section(header: str, keys: list[str]) -> None:
        lines.append(f"# ── {header} {'─' * max(0, 60 - len(header))}")
        for key in keys:
            val = values.get(key)
            default = _DEFAULTS.get(key)
            toml_val = _toml_value(val)
            if val == default or val is None or val == "" or val == []:
                lines.append(f"# {key} = {toml_val}")
            else:
                lines.append(f"{key} = {toml_val}")
        lines.append("")

    _section("STT backend", ["stt_backend", "stt_model"])
    _section("Polish（LLM 後處理）", [
        "polish_enabled", "polish_model", "polish_languages", "polish_prompt",
    ])
    _section("Trigger keys（觸發鍵）", ["trigger_keys", "edit_trigger_keys"])
    _section("Audio（音訊）", [
        "mic_device", "sample_rate", "min_audio_sec", "max_audio_sec",
    ])
    _section("Beep feedback（提示音）", [
        "beeps_enabled", "beep_start_hz", "beep_end_hz",
        "beep_fail_hz", "beep_duration_ms", "beep_volume",
    ])
    _section("Advanced（進階）", [
        "encoder_pipelining", "selection_capture_wait_s",
    ])

    return "\n".join(lines)


def build_settings_tab() -> None:
    cfg = _load_cfg()
    cpath = config_path()

    with gr.Tab("設定", id="settings"):
        gr.Markdown("## 設定")
        gr.Markdown(
            f"設定檔路徑：`{cpath}`\n\n"
            "修改後按「**儲存設定**」寫入檔案，並重啟 daemon 使設定生效。"
        )

        save_notice = gr.Markdown(visible=False)

        # Hardware preset
        with gr.Accordion("硬體預設 (Hardware Preset)", open=False):
            gr.Markdown(
                "選擇一個預設後，STT 引擎與 Polish 欄位會自動填入。"
                "預設不會立即儲存，仍需按「儲存設定」。"
            )
            preset_dd = gr.Dropdown(
                choices=_PRESET_CHOICES,
                value="（不套用預設）",
                label="預設選擇",
                interactive=True,
            )
            preset_desc = gr.Markdown("")

        gr.Markdown("---")

        # STT Engine
        with gr.Accordion("STT 引擎", open=True):
            with gr.Row():
                stt_backend_dd = gr.Dropdown(
                    choices=["qwen3-asr", "faster-whisper", "mlx-whisper"],
                    value=cfg.get("stt_backend") or "qwen3-asr",
                    label="STT Backend",
                    info="選擇語音辨識引擎。qwen3-asr 為預設（推薦）。",
                    interactive=True,
                )
                stt_model_tb = gr.Textbox(
                    value=cfg.get("stt_model") or "",
                    label="STT Model（留空使用預設）",
                    placeholder=_BACKEND_MODEL_SUGGESTIONS.get(
                        cfg.get("stt_backend") or "qwen3-asr", ""
                    ),
                    interactive=True,
                )

        # Polish
        with gr.Accordion("Polish 後處理", open=True):
            polish_enabled_cb = gr.Checkbox(
                value=bool(cfg.get("polish_enabled", True)),
                label="啟用 Polish（LLM 後處理）",
                info="移除口誤、語助詞，修正重複。需要 LLM 模型與 GPU。",
                interactive=True,
            )
            with gr.Row():
                polish_model_tb = gr.Textbox(
                    value=cfg.get("polish_model") or "",
                    label="Polish Model（留空使用預設）",
                    placeholder=_POLISH_MODEL_SUGGESTIONS,
                    interactive=True,
                )
            polish_langs_cb = gr.CheckboxGroup(
                choices=_LANGUAGE_CHOICES,
                value=cfg.get("polish_languages") or ["zh"],
                label="Polish Languages（啟用 Polish 的語言）",
                info="只對勾選的語言執行 LLM 後處理；其他語言直接輸出原始 ASR 結果。",
                interactive=True,
            )

        # Trigger keys
        with gr.Accordion("觸發鍵 (Trigger Keys)", open=True):
            gr.Markdown(
                "> **注意**：Web UI 無法即時偵測按鍵。請手動輸入按鍵名稱，"
                "例如 `alt_r`、`f13`、`cmd_r`。"
                "執行 `home-stt config --set-trigger` 可進行互動式設定。"
            )
            with gr.Row():
                trigger_keys_tb = gr.Textbox(
                    value=_list_to_str(cfg.get("trigger_keys")),
                    label="Dictate Trigger（聽寫觸發鍵）",
                    placeholder="例：alt_r  或  ctrl_r  （多個鍵以逗號分隔）",
                    info="留空使用平台預設（Windows: alt_r / ctrl_r，macOS: option_r）。",
                    interactive=True,
                )
                edit_trigger_keys_tb = gr.Textbox(
                    value=_list_to_str(cfg.get("edit_trigger_keys")),
                    label="Voice-Edit Trigger（語音編輯觸發鍵）",
                    placeholder="例：f13  或  cmd_r",
                    info="留空使用平台預設（Windows: f13，macOS: cmd_r）。",
                    interactive=True,
                )

        # Audio
        with gr.Accordion("音訊 (Audio)", open=False):
            mic_device_tb = gr.Textbox(
                value=str(cfg.get("mic_device") or ""),
                label="麥克風裝置（Mic Device）",
                placeholder="裝置名稱子字串 或 裝置索引數字，留空使用系統預設",
                info="執行 `home-stt devices` 查看可用裝置列表。",
                interactive=True,
            )
            with gr.Row():
                sample_rate_nb = gr.Number(
                    value=int(cfg.get("sample_rate") or 16000),
                    label="Sample Rate（Hz）",
                    info="建議保持 16000 Hz（STT 模型訓練解析度）。",
                    minimum=8000, maximum=48000, step=1,
                    interactive=True,
                )
                min_audio_sl = gr.Slider(
                    value=float(cfg.get("min_audio_sec") or 0.15),
                    label="最短錄音時長（秒）",
                    minimum=0.05, maximum=2.0, step=0.05,
                    info="短於此時長的錄音將被忽略，避免誤觸。",
                    interactive=True,
                )
                max_audio_sl = gr.Slider(
                    value=float(cfg.get("max_audio_sec") or 120),
                    label="最長錄音時長（秒）",
                    minimum=5, maximum=300, step=5,
                    info="超過此時長強制結束錄音。",
                    interactive=True,
                )

        # Beep
        with gr.Accordion("提示音 (Beep)", open=False):
            beeps_enabled_cb = gr.Checkbox(
                value=bool(cfg.get("beeps_enabled", True)),
                label="啟用提示音",
                info="錄音開始、結束、失敗時播放音效回饋。",
                interactive=True,
            )
            with gr.Row():
                beep_volume_sl = gr.Slider(
                    value=float(cfg.get("beep_volume") or 0.15),
                    label="音量 (beep_volume)",
                    minimum=0.0, maximum=1.0, step=0.01,
                    interactive=True,
                )
            with gr.Row():
                beep_start_hz_nb = gr.Number(
                    value=int(cfg.get("beep_start_hz") or 880),
                    label="開始音頻率（Hz）",
                    minimum=100, maximum=4000, step=10,
                    interactive=True,
                )
                beep_end_hz_nb = gr.Number(
                    value=int(cfg.get("beep_end_hz") or 660),
                    label="結束音頻率（Hz）",
                    minimum=100, maximum=4000, step=10,
                    interactive=True,
                )
                beep_fail_hz_nb = gr.Number(
                    value=int(cfg.get("beep_fail_hz") or 220),
                    label="失敗音頻率（Hz）",
                    minimum=100, maximum=4000, step=10,
                    interactive=True,
                )
                beep_duration_ms_nb = gr.Number(
                    value=int(cfg.get("beep_duration_ms") or 80),
                    label="音效時長（ms）",
                    minimum=10, maximum=500, step=5,
                    interactive=True,
                )

        # Advanced
        with gr.Accordion("進階設定 (Advanced)", open=False):
            with gr.Row():
                encoder_pipeline_cb = gr.Checkbox(
                    value=bool(cfg.get("encoder_pipelining", False)),
                    label="Encoder Pipelining（實驗性）",
                    info="啟用 Qwen3-ASR 編碼器流水線，可降低延遲但增加記憶體用量。",
                    interactive=True,
                )
                selection_wait_sl = gr.Slider(
                    value=float(cfg.get("selection_capture_wait_s") or 0.1),
                    label="Selection Capture Wait（秒）",
                    minimum=0.0, maximum=1.0, step=0.01,
                    info="語音編輯模式中等待剪貼簿捕獲的時間。",
                    interactive=True,
                )
            polish_prompt_tb = gr.Textbox(
                value=cfg.get("polish_prompt") or "",
                label="Polish Prompt（自訂 System Prompt，留空使用內建提示詞）",
                placeholder="覆蓋 Polish LLM 的 system prompt。高階用途，一般使用者請留空。",
                lines=3, interactive=True,
            )

        gr.Markdown("---")

        with gr.Row():
            save_btn = gr.Button("儲存設定", variant="primary", scale=2, elem_classes=["btn-ico", "i-save"])
            reset_btn = gr.Button("重設為預設值", variant="secondary", scale=1, elem_classes=["btn-ico", "i-reset"])

        with gr.Accordion("設定檔預覽 (Config Preview)", open=False):
            config_preview = gr.Textbox(
                value=_build_toml(_load_cfg()),
                label=f"config.toml — {cpath}",
                lines=20,
                max_lines=30,
                interactive=False,
            )

        # -- Event handlers --

        def on_preset_change(preset_name: str):
            if preset_name == "（不套用預設）" or preset_name not in _PRESETS:
                return (
                    gr.update(value=""),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
            p = _PRESETS[preset_name]
            return (
                gr.update(value=f"_{p['desc']}_"),
                gr.update(value=p["stt_backend"]),
                gr.update(value=p.get("stt_model") or ""),
                gr.update(value=p["polish_enabled"]),
                gr.update(value=p.get("polish_model") or ""),
            )

        preset_dd.change(
            fn=on_preset_change,
            inputs=[preset_dd],
            outputs=[preset_desc, stt_backend_dd, stt_model_tb,
                     polish_enabled_cb, polish_model_tb],
        )

        def on_backend_change(backend: str):
            suggestion = _BACKEND_MODEL_SUGGESTIONS.get(backend, "")
            return gr.update(placeholder=suggestion)

        stt_backend_dd.change(
            fn=on_backend_change,
            inputs=[stt_backend_dd],
            outputs=[stt_model_tb],
        )

        def on_save(
            stt_backend, stt_model, polish_enabled, polish_model,
            polish_languages, polish_prompt_raw, trigger_keys_raw,
            edit_trigger_keys_raw, mic_device_raw, sample_rate,
            min_audio_sec, max_audio_sec, beeps_enabled, beep_volume,
            beep_start_hz, beep_end_hz, beep_fail_hz, beep_duration_ms,
            encoder_pipelining, selection_capture_wait_s,
        ):
            values: dict[str, Any] = {
                "stt_backend": stt_backend.strip() if stt_backend else None,
                "stt_model": stt_model.strip() if stt_model and stt_model.strip() else None,
                "polish_enabled": bool(polish_enabled),
                "polish_model": polish_model.strip() if polish_model and polish_model.strip() else None,
                "polish_languages": list(polish_languages) if polish_languages else ["zh"],
                "polish_prompt": polish_prompt_raw.strip() if polish_prompt_raw and polish_prompt_raw.strip() else None,
                "trigger_keys": _str_to_list(trigger_keys_raw),
                "edit_trigger_keys": _str_to_list(edit_trigger_keys_raw),
                "mic_device": (
                    (int(mic_device_raw.strip())
                     if mic_device_raw.strip().lstrip("-").isdigit()
                     else mic_device_raw.strip())
                    if mic_device_raw and mic_device_raw.strip()
                    else None
                ),
                "sample_rate": int(sample_rate) if sample_rate else 16000,
                "min_audio_sec": float(min_audio_sec),
                "max_audio_sec": float(max_audio_sec),
                "beeps_enabled": bool(beeps_enabled),
                "beep_volume": float(beep_volume),
                "beep_start_hz": int(beep_start_hz),
                "beep_end_hz": int(beep_end_hz),
                "beep_fail_hz": int(beep_fail_hz),
                "beep_duration_ms": int(beep_duration_ms),
                "encoder_pipelining": bool(encoder_pipelining),
                "selection_capture_wait_s": float(selection_capture_wait_s),
            }
            try:
                path = config_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                toml_text = _build_toml(values)
                path.write_text(toml_text, encoding="utf-8")
                notice = (
                    f"**設定已儲存至 `{path}`**\n\n"
                    "請執行 `home-stt restart` 重啟 daemon 使設定生效。"
                )
                return (
                    gr.update(value=notice, visible=True),
                    gr.update(value=toml_text),
                )
            except Exception as exc:
                error_msg = f"**儲存失敗：** `{exc}`"
                return (
                    gr.update(value=error_msg, visible=True),
                    gr.update(),
                )

        save_btn.click(
            fn=on_save,
            inputs=[
                stt_backend_dd, stt_model_tb, polish_enabled_cb,
                polish_model_tb, polish_langs_cb, polish_prompt_tb,
                trigger_keys_tb, edit_trigger_keys_tb, mic_device_tb,
                sample_rate_nb, min_audio_sl, max_audio_sl,
                beeps_enabled_cb, beep_volume_sl, beep_start_hz_nb,
                beep_end_hz_nb, beep_fail_hz_nb, beep_duration_ms_nb,
                encoder_pipeline_cb, selection_wait_sl,
            ],
            outputs=[save_notice, config_preview],
        )

        def on_reset():
            d = _DEFAULTS
            return (
                gr.update(value=d.get("stt_backend") or "qwen3-asr"),
                gr.update(value=""),
                gr.update(value=bool(d.get("polish_enabled", True))),
                gr.update(value=""),
                gr.update(value=d.get("polish_languages") or ["zh"]),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=int(d.get("sample_rate") or 16000)),
                gr.update(value=float(d.get("min_audio_sec") or 0.15)),
                gr.update(value=float(d.get("max_audio_sec") or 120)),
                gr.update(value=bool(d.get("beeps_enabled", True))),
                gr.update(value=float(d.get("beep_volume") or 0.15)),
                gr.update(value=int(d.get("beep_start_hz") or 880)),
                gr.update(value=int(d.get("beep_end_hz") or 660)),
                gr.update(value=int(d.get("beep_fail_hz") or 220)),
                gr.update(value=int(d.get("beep_duration_ms") or 80)),
                gr.update(value=bool(d.get("encoder_pipelining", False))),
                gr.update(value=float(d.get("selection_capture_wait_s") or 0.1)),
                gr.update(value="", visible=False),
                gr.update(value="（不套用預設）"),
            )

        reset_btn.click(
            fn=on_reset,
            inputs=[],
            outputs=[
                stt_backend_dd, stt_model_tb, polish_enabled_cb,
                polish_model_tb, polish_langs_cb, polish_prompt_tb,
                trigger_keys_tb, edit_trigger_keys_tb, mic_device_tb,
                sample_rate_nb, min_audio_sl, max_audio_sl,
                beeps_enabled_cb, beep_volume_sl, beep_start_hz_nb,
                beep_end_hz_nb, beep_fail_hz_nb, beep_duration_ms_nb,
                encoder_pipeline_cb, selection_wait_sl,
                save_notice, preset_dd,
            ],
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 & 5 — Guide (使用指南) & Diagnostics (診斷)
# ═══════════════════════════════════════════════════════════════════════════

def _log_paths() -> tuple[Path, Path]:
    tmp = Path(tempfile.gettempdir())
    return tmp / "stt-daemon.log", tmp / "stt-daemon.err.log"


def _read_last_lines(path: Path, n: int = 50) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        tail = lines[-n:] if len(lines) > n else lines
        return "\n".join(tail)
    except OSError:
        return f"(找不到 log 檔案：{path})"


def _run_doctor() -> str:
    try:
        return _run_doctor_impl()
    except Exception as exc:
        return f"診斷過程中發生未預期錯誤：\n\n{type(exc).__name__}: {exc}"


def _run_doctor_impl() -> str:
    buf = io.StringIO()
    all_ok = True

    def _check(label: str, ok: bool, detail: str = "") -> bool:
        nonlocal all_ok
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}]  {label}"
        if detail:
            line += f"  -- {detail}"
        buf.write(line + "\n")
        if not ok:
            all_ok = False
        return ok

    ver = _daemon_version()
    buf.write(f"home-stt v{ver} doctor\n")
    buf.write("=" * 50 + "\n\n")

    v = sys.version_info
    py_ok = v >= (3, 10)
    _check("Python >= 3.10", py_ok, f"{v.major}.{v.minor}.{v.micro}")

    for pkg_name, import_name in [
        ("numpy", "numpy"),
        ("sounddevice", "sounddevice"),
        ("pynput", "pynput"),
        ("opencc", "opencc"),
    ]:
        try:
            mod = __import__(import_name)
            mod_ver = getattr(mod, "__version__", getattr(mod, "VERSION", "ok"))
            _check(pkg_name, True, str(mod_ver))
        except (ImportError, OSError) as e:
            _check(pkg_name, False, f"載入失敗：{e}")

    toml_ok = False
    toml_detail = ""
    try:
        import tomllib as _tomllib  # noqa: F401
        toml_ok = True
        toml_detail = "tomllib (Python 內建)"
    except ModuleNotFoundError:
        try:
            import tomli as _tomli  # noqa: F401
            toml_ok = True
            toml_detail = f"tomli {_tomli.__version__}"
        except ModuleNotFoundError:
            toml_detail = "需安裝 tomli (Python < 3.11)"
    _check("TOML 支援", toml_ok, toml_detail)

    buf.write("\n")
    is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"

    if is_apple_silicon:
        try:
            import mlx  # noqa: F401
            _check("mlx (Apple Silicon)", True, getattr(mlx, "__version__", "ok"))
        except (ImportError, OSError) as e:
            _check("mlx (Apple Silicon)", False, f"載入失敗：{e}")
        try:
            import mlx_lm  # noqa: F401
            _check("mlx-lm (polish)", True, getattr(mlx_lm, "__version__", "ok"))
        except (ImportError, OSError) as e:
            _check("mlx-lm (polish)", False, f"載入失敗：{e}")
    else:
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
            if cuda_ok:
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                _check("torch + CUDA", True,
                       f"{torch.__version__}, {gpu_name}, {vram_gb:.1f} GB VRAM")
            else:
                _check("torch + CUDA", False,
                       f"torch {torch.__version__} 已安裝但 CUDA 不可用")
        except ImportError:
            _check("torch", False, "未安裝")
        except OSError as e:
            _check("torch", False,
                   f"DLL 載入失敗 -- {e}。"
                   "請確認 CUDA toolkit 已安裝，或重新安裝 torch CUDA wheel")

        try:
            import transformers
            _check("transformers", True, transformers.__version__)
        except (ImportError, OSError) as e:
            _check("transformers", False, f"載入失敗：{e}")

        try:
            import qwen_asr  # noqa: F401
            _check("qwen-asr (預設 STT backend)", True,
                   getattr(qwen_asr, "__version__", "ok"))
        except (ImportError, OSError) as e:
            _check("qwen-asr (預設 STT backend)", False, f"載入失敗：{e}")

        try:
            import faster_whisper
            _check("faster-whisper (fallback)", True, faster_whisper.__version__)
        except (ImportError, OSError) as e:
            _check("faster-whisper (fallback)", False, f"載入失敗：{e}")

    buf.write("\n")
    try:
        import sounddevice as sd
        dev = sd.query_devices(kind="input")
        mic_name = dev.get("name", "未知")
        sr = int(dev.get("default_samplerate", 0))
        _check("麥克風", True, f"{mic_name}（{sr} Hz）")
    except Exception as e:
        _check("麥克風", False, str(e))

    if sys.platform == "darwin":
        buf.write("\n  macOS 所需權限（系統設定 > 隱私權與安全性）：\n")
        buf.write("    - 輸入裝置監控（Input Monitoring）\n")
        buf.write("    - 輔助使用（Accessibility）\n")
        buf.write("    - 麥克風（Microphone）\n")
        buf.write(f"    Python binary: {sys.executable}\n")

    buf.write("\n")
    cp = config_path()
    _check("Config 檔", cp.exists(), str(cp))

    buf.write("\n" + "=" * 50 + "\n")
    if all_ok:
        buf.write("全部通過。執行 `home-stt start` 啟動 daemon。\n")
    else:
        buf.write("有項目未通過，請修復後重新執行診斷。\n")

    return buf.getvalue()


def _get_system_info() -> str:
    lines: list[str] = []
    lines.append(f"**Python 版本**：{sys.version}")
    lines.append(f"**平台**：{sys.platform}  ({platform.platform()})")
    lines.append(f"**架構**：{platform.machine()}")

    if sys.platform == "darwin" and platform.machine() == "arm64":
        lines.append("**加速**：Apple Metal（Apple Silicon MLX）")
    else:
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                vram = props.total_memory / (1024 ** 3)
                lines.append(f"**GPU**：{gpu}（{vram:.1f} GB VRAM）")
                lines.append(f"**CUDA 版本**：{torch.version.cuda}")
            else:
                lines.append("**GPU**：CUDA 不可用（或未安裝 torch CUDA wheel）")
        except ImportError:
            lines.append("**GPU**：torch 未安裝，無法查詢 CUDA 資訊")
        except OSError as e:
            lines.append(f"**GPU**：torch DLL 載入失敗 — {e}")

    try:
        cp = config_path()
        lines.append(f"\n**Config 檔路徑**：`{cp}`")
        if cp.exists():
            content = cp.read_text(encoding="utf-8", errors="replace")
            lines.append(f"\n```toml\n{content}\n```")
        else:
            lines.append("\n（Config 檔不存在，使用程式碼預設值）")
            lines.append(f"\n預設 Config 範本：\n```toml\n{generate_default_config()}\n```")
    except Exception as e:
        lines.append(f"\n無法讀取 Config：{e}")

    return "\n".join(lines)


_MACOS_QUICKSTART = """\
## macOS Apple Silicon -- 快速開始

### 1. Clone & 安裝套件

```bash
git clone https://github.com/DennisLiuCk/home-stt.git
cd home-stt

pip install \\
    sounddevice pynput opencc-python-reimplemented \\
    mlx-qwen3-asr mlx-lm mlx-whisper faster-whisper
```

### 2. 安裝 home-stt CLI

```bash
pip install -e .
```

### 3. 授權三個系統權限

在「**系統設定 > 隱私權與安全性**」，將 Python binary 路徑加入：

```bash
python3 -c "import sys; print(sys.executable)"
```

| 權限項目 | 原因 |
|---------|------|
| **輸入裝置監控**（Input Monitoring） | pynput 全域鍵盤 listener |
| **輔助使用**（Accessibility） | Quartz CGEvent 模擬 Cmd+V |
| **麥克風**（Microphone） | sounddevice 讀麥克風 |

### 4. 啟動 daemon

```bash
home-stt start
```

### 5. 使用

- 按住 **Right Option** > 講話 > 放開 > 文字自動貼入
- 選取文字後，按住 **Right Command** > 講編輯指令 > 放開 > 文字被改寫
"""

_WINDOWS_QUICKSTART = """\
## Windows 10/11 -- 快速開始

### 1. Clone

```powershell
git clone https://github.com/DennisLiuCk/home-stt.git
cd home-stt
```

### 2. 安裝套件（順序重要！先裝 PyTorch CUDA）

```powershell
# 步驟一：先裝 CUDA 12.x 版 PyTorch
pip install --user torch --index-url https://download.pytorch.org/whl/cu124

# 步驟二：再裝 qwen-asr
pip install --user qwen-asr

# 步驟三：其他依賴
pip install --user sounddevice pynput opencc-python-reimplemented faster-whisper
```

### 3. 安裝 home-stt CLI

```powershell
pip install -e . --user
```

### 4. 啟動 daemon

```powershell
home-stt start
```

### 5. 使用

- 按住 **Right Alt** 或 **Right Ctrl** > 講話 > 放開 > 文字自動貼入
- 選取文字後，按住 **F13** > 講編輯指令 > 放開 > 文字被改寫
"""

_FEATURES_MD = """\
## 核心功能

### 語音輸入（Dictate）

按住觸發鍵 > 對麥克風說話 > 放開 > 文字自動貼到當下焦點視窗

- 中英文混合直接說
- 自動轉繁體中文（opencc s2twp）
- CJK ↔ ASCII 自動補空格
- Polish 後處理去除口語贅字

---

### 語音編輯（Voice-Edit）

選中文字 > 按住編輯觸發鍵 > 說編輯指令 > 放開 > LLM 改寫選取並貼回

常用指令範例：
- 翻譯成英文 / 翻譯成中文
- 改成正式語氣 / 改得口語一點
- 縮短一半 / 展開成兩段
- 整理成條列式 / 合併成一段

完全離線，LLM 在本地運行。
"""

_HOTKEYS_MD_MAC = """\
## 預設快捷鍵（macOS）

| 功能 | 按鍵 | 說明 |
|------|------|------|
| 語音輸入 | **Right Option** | 按住錄音，放開貼字 |
| 語音編輯 | **Right Command** | 選取後按住，說指令，放開改寫 |

> 可在 `config.toml` 的 `trigger_keys` / `edit_trigger_keys` 自訂。
"""

_HOTKEYS_MD_WIN = """\
## 預設快捷鍵（Windows）

| 功能 | 按鍵 | 說明 |
|------|------|------|
| 語音輸入 | **Right Alt** 或 **Right Ctrl** | 按住錄音，放開貼字 |
| 語音編輯 | **F13** | 選取後按住，說指令，放開改寫 |

> 可在 `config.toml` 的 `trigger_keys` / `edit_trigger_keys` 自訂。
"""

_FAQ_MD = """\
## 常見問題 / 疑難排解

### Q：按鍵完全沒反應

**macOS**：最常見原因是系統權限未授予。
打開「系統設定 > 隱私權與安全性」，確認 Python binary 已加入
Input Monitoring、Accessibility、Microphone。

**Windows / macOS 通用**：確認 daemon 正在執行：
```bash
home-stt status
```

---

### Q：辨識出的文字是簡體字

安裝 `opencc-python-reimplemented` 後重啟 daemon。

---

### Q：辨識速度很慢

1. 確認 GPU 已啟用
2. 考慮切換到較輕量的 Preset（設定頁面可選擇）

---

### Q：CUDA OOM（記憶體不足）

降階到 Balanced 或 Light tier：
```toml
# Balanced（~5 GB VRAM）
polish_model = "Qwen/Qwen2.5-1.5B-Instruct"

# Light（~2 GB VRAM）
polish_enabled = false
```

---

### Q：Voice-Edit 沒效果

- 確認在按下觸發鍵之前有選取文字
- 某些 App 不支援 Cmd+C / Ctrl+C 抓選取
- macOS：確認 Accessibility 權限已授予

---

### Q：第一次啟動要等很久

正常。首次需從 HuggingFace 下載模型（Maximum tier ~10 GB）。
下載完後模型快取在 `~/.cache/huggingface/`，後續啟動 10-30 秒。
"""

_PRESET_MD = """\
## 硬體 Preset 比較

| Preset | STT Backend | Polish | VRAM | 品質 |
|--------|-------------|--------|------|------|
| **Maximum** | qwen3-asr | Qwen3-4B-Instruct | ~10 GB | 最高 |
| **Balanced** | qwen3-asr | Qwen2.5-1.5B | ~5 GB | 中高 |
| **Light** | qwen3-asr | 停用 | ~2 GB | 中 |
| **Mini** | faster-whisper | 停用 | <2 GB | 中 |
"""


def _quickstart_md() -> str:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return _MACOS_QUICKSTART
    return _WINDOWS_QUICKSTART


def build_guide_and_diagnostics_tabs():
    # Guide tab
    with gr.Tab("使用指南", id="guide"):
        with gr.Accordion("快速開始", open=True):
            gr.Markdown(_quickstart_md())
        with gr.Accordion("核心功能", open=False):
            gr.Markdown(_FEATURES_MD)
        with gr.Accordion("快捷鍵參考", open=False):
            if sys.platform == "darwin":
                gr.Markdown(_HOTKEYS_MD_MAC)
            else:
                gr.Markdown(_HOTKEYS_MD_WIN)
        with gr.Accordion("常見問題 / 疑難排解", open=False):
            gr.Markdown(_FAQ_MD)
        with gr.Accordion("硬體 Preset 比較表", open=False):
            gr.Markdown(_PRESET_MD)

    # Diagnostics tab
    with gr.Tab("診斷", id="diagnostics"):
        with gr.Accordion("環境健康檢查（Doctor）", open=True):
            doctor_output = gr.Textbox(
                label="診斷結果",
                value="（點擊「執行診斷」開始）",
                lines=25, max_lines=40,
                interactive=False,
            )
            run_doctor_btn = gr.Button("執行診斷", variant="primary", elem_classes=["btn-ico", "i-play"])

            def _on_run_doctor():
                return _run_doctor()

            run_doctor_btn.click(fn=_on_run_doctor, inputs=[], outputs=[doctor_output])

        with gr.Accordion("即時 Log 查看器", open=True):
            with gr.Row():
                log_type = gr.Radio(
                    choices=["stdout log（stt-daemon.log）", "err.log（stt-daemon.err.log）"],
                    value="stdout log（stt-daemon.log）",
                    label="Log 來源",
                )
                refresh_log_btn = gr.Button("重新整理", variant="secondary", elem_classes=["btn-ico", "i-refresh"])

            log_output = gr.Textbox(
                label="最後 50 行（最新在底部）",
                value="（點擊「重新整理」載入 log）",
                lines=20, max_lines=35,
                interactive=False,
            )
            log_path_display = gr.Markdown("", label="")

            def _on_refresh_log(log_type_val: str):
                stdout_log, err_log = _log_paths()
                use_err = "err.log" in log_type_val
                path = err_log if use_err else stdout_log
                content = _read_last_lines(path, n=50)
                path_md = f"**Log 路徑**：`{path}`"
                if not path.exists():
                    path_md += "  -- *檔案不存在（daemon 尚未執行過？）*"
                else:
                    try:
                        age_s = time.time() - path.stat().st_mtime
                        mins = int(age_s // 60)
                        secs = int(age_s % 60)
                        path_md += f"  -- 最後寫入：{mins}m {secs}s 前"
                    except OSError:
                        pass
                return content, path_md

            refresh_log_btn.click(
                fn=_on_refresh_log,
                inputs=[log_type],
                outputs=[log_output, log_path_display],
            )
            log_type.change(
                fn=_on_refresh_log,
                inputs=[log_type],
                outputs=[log_output, log_path_display],
            )

        with gr.Accordion("系統資訊 & Config 內容", open=False):
            sysinfo_output = gr.Markdown("（點擊「載入系統資訊」查看）")
            load_sysinfo_btn = gr.Button("載入系統資訊", variant="secondary", elem_classes=["btn-ico", "i-activity"])

            def _on_load_sysinfo():
                return _get_system_info()

            load_sysinfo_btn.click(
                fn=_on_load_sysinfo,
                inputs=[],
                outputs=[sysinfo_output],
            )


# ═══════════════════════════════════════════════════════════════════════════
# App assembly
# ═══════════════════════════════════════════════════════════════════════════

def create_app() -> gr.Blocks:
    """Create and return the complete home-stt Gradio Blocks application."""
    version = _daemon_version()

    with gr.Blocks(
        title="home-stt Web UI",
    ) as demo:

        # Header
        gr.HTML(
            '<div class="app-header">'
            '<div class="app-badge">'
            '<span class="bar"></span><span class="bar"></span>'
            '<span class="bar"></span><span class="bar"></span>'
            '<span class="bar"></span>'
            '</div>'
            '<div class="app-eyebrow">本地 · 離線 · 隱私優先</div>'
            '<h1 class="app-wordmark">home-stt</h1>'
            '<div class="subtitle">Hold-to-Talk Speech-to-Text · 按住即說，放開成字</div>'
            '<div class="rule"></div>'
            '</div>'
        )

        with gr.Tabs():
            build_dashboard_tab()
            build_playground_tab()
            build_settings_tab()
            build_guide_and_diagnostics_tabs()

        # Footer
        gr.HTML(
            f'<div class="app-footer">'
            f'home-stt v{version}'
            f'<span class="sep">/</span>'
            f'Gradio {gr.__version__}'
            f'<span class="sep">/</span>'
            f'{sys.platform} · {platform.machine()}'
            f'<span class="sep">/</span>'
            f'Python {sys.version.split()[0]}'
            f'</div>'
        )

    return demo


def main(port: int = 7860, share: bool = False) -> None:
    """Launch the web UI server."""
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=share,
        theme=_build_theme(),
        css=CUSTOM_CSS,
        head=_HEAD_HTML,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="home-stt Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()
    main(port=args.port, share=args.share)
