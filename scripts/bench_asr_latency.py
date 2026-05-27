"""ASR latency & VRAM benchmark — three-way comparison.

Measures end-to-end transcription latency and peak VRAM for each backend
across different audio lengths. Uses the same fixture audio as the quality
benchmark.

Prerequisites:
  1. Generate test audio:  python scripts/gen_asr_bench_audio.py
  2. Install backends you want to test (see bench_asr_quality.py)
  3. Run:               python scripts/bench_asr_latency.py [--backends all]

Outputs a markdown table to stdout and a JSON file to bench_asr_latency.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "asr_bench_cases.json"
REPORT_PATH = REPO_ROOT / "bench_asr_latency.json"
RUNS_PER_CASE = 5

sys.path.insert(0, str(REPO_ROOT / "scripts"))

BACKEND_CONFIGS = {
    "breeze-hf": ("breeze-asr", "MediaTek-Research/Breeze-ASR-25"),
    "breeze-fw": ("faster-whisper", "SoybeanMilk/faster-whisper-Breeze-ASR-25"),
    "qwen3-asr": ("qwen3-asr", "Qwen/Qwen3-ASR-0.6B"),
}


def _load_audio(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _audio_duration(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _try_build_backend(label: str):
    from stt_backends import build_backend
    name, model = BACKEND_CONFIGS[label]
    try:
        backend = build_backend(name, model)
        print(f"  [{label}] loaded: {backend.device_label}")
        backend.warmup()
        return backend
    except Exception as e:
        print(f"  [{label}] SKIP — {e}")
        return None


def _measure_vram() -> int | None:
    """Return peak VRAM in bytes, or None if not on CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated()
    except ImportError:
        pass
    return None


def _reset_vram():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _bench_case(backend, samples: np.ndarray, n_runs: int) -> list[float]:
    """Run n_runs transcriptions and return latencies."""
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        backend.transcribe(samples)
        latencies.append(time.perf_counter() - t0)
    return latencies


def main():
    parser = argparse.ArgumentParser(description="ASR latency benchmark")
    parser.add_argument(
        "--backends",
        default="all",
        help="Comma-separated backend labels or 'all'",
    )
    parser.add_argument("--runs", type=int, default=RUNS_PER_CASE)
    parser.add_argument("--json-out", default=str(REPORT_PATH))
    args = parser.parse_args()

    if not FIXTURE_PATH.exists():
        print(f"ERROR: fixture not found at {FIXTURE_PATH}")
        sys.exit(1)

    with open(FIXTURE_PATH, encoding="utf-8") as f:
        fixture = json.load(f)
    cases = fixture["cases"]

    labels = (
        list(BACKEND_CONFIGS.keys())
        if args.backends == "all"
        else [s.strip() for s in args.backends.split(",")]
    )

    print("=" * 70)
    print("ASR Latency & VRAM Benchmark")
    print("=" * 70)
    print(f"Runs per case: {args.runs}")
    print(f"\nLoading backends ({len(labels)})...")

    backends = {}
    for label in labels:
        if label not in BACKEND_CONFIGS:
            continue
        b = _try_build_backend(label)
        if b:
            backends[label] = b

    if not backends:
        print("\nERROR: no backends loaded.")
        sys.exit(1)

    # ── Measure warmup + VRAM per backend ──
    print("\n--- Warmup & VRAM ---")
    vram_info = {}
    for label, backend in backends.items():
        _reset_vram()
        t0 = time.perf_counter()
        backend.warmup()
        warmup_s = time.perf_counter() - t0
        vram = _measure_vram()
        vram_mb = f"{vram / 1024**2:.0f} MB" if vram else "N/A"
        vram_info[label] = {"warmup_s": round(warmup_s, 3), "vram_bytes": vram}
        print(f"  [{label}] warmup={warmup_s:.3f}s, peak_vram={vram_mb}")

    # ── Latency per case ──
    print(f"\n--- Latency ({args.runs} runs each, median reported) ---\n")

    results = {}
    for label, backend in backends.items():
        results[label] = {}
        for case in cases:
            audio_path = REPO_ROOT / case["audio_path"]
            if not audio_path.exists():
                results[label][case["id"]] = {"error": "audio not found"}
                continue

            samples = _load_audio(audio_path)
            duration = _audio_duration(audio_path)
            latencies = _bench_case(backend, samples, args.runs)
            median = sorted(latencies)[len(latencies) // 2]
            rtf = median / duration if duration > 0 else -1

            results[label][case["id"]] = {
                "audio_duration_s": round(duration, 2),
                "latencies_s": [round(t, 4) for t in latencies],
                "median_s": round(median, 4),
                "rtf": round(rtf, 3),
            }
            print(
                f"  [{label}] {case['id']:40s} "
                f"audio={duration:.1f}s  median={median:.3f}s  RTF={rtf:.2f}x"
            )

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("SUMMARY — Median latency by audio duration bucket")
    print("=" * 70)

    buckets = [
        ("short (0-3s)", 0, 3),
        ("medium (3-10s)", 3, 10),
        ("long (10-30s)", 10, 30),
    ]

    header = f"{'Bucket':20s}"
    for label in backends:
        header += f" | {label:>18s}"
    print(f"\n{header}")
    print("-" * len(header))

    for bname, lo, hi in buckets:
        row = f"{bname:20s}"
        for label in backends:
            medians = [
                r["median_s"]
                for r in results[label].values()
                if "median_s" in r
                and lo <= r.get("audio_duration_s", 0) < hi
            ]
            if medians:
                avg = sum(medians) / len(medians)
                row += f" | {avg:>15.3f}s  "
            else:
                row += f" | {'—':>18s}"
        print(row)

    print("\n--- VRAM ---")
    for label in backends:
        v = vram_info.get(label, {})
        vram = v.get("vram_bytes")
        print(f"  {label:15s}: {vram / 1024**2:.0f} MB" if vram else f"  {label:15s}: N/A")

    # ── Write JSON ──
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs_per_case": args.runs,
        "backends": {
            label: {
                "name": BACKEND_CONFIGS[label][0],
                "model": BACKEND_CONFIGS[label][1],
                **vram_info.get(label, {}),
            }
            for label in backends
        },
        "results": results,
    }
    out_path = Path(args.json_out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nJSON report: {out_path}")


if __name__ == "__main__":
    main()
