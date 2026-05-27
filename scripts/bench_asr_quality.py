"""ASR quality benchmark — three-way comparison.

Compares Breeze-ASR-25 (HF pipeline), Breeze-ASR-25 (faster-whisper), and
Qwen3-ASR-0.6B (baseline) across CER, code-switching accuracy, punctuation
recall, and must_contain/must_not_contain assertions from the fixture file.

Prerequisites:
  1. Generate test audio:  python scripts/gen_asr_bench_audio.py
  2. Install backends you want to test:
     - Breeze HF:       pip install transformers accelerate
     - Breeze FW:       pip install faster-whisper
     - Qwen3-ASR:       pip install qwen-asr
  3. Run:               python scripts/bench_asr_quality.py [--backends all]

Outputs a markdown report to stdout and a JSON file to bench_asr_quality.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "asr_bench_cases.json"
REPORT_PATH = REPO_ROOT / "bench_asr_quality.json"

sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load_audio(wav_path: Path) -> np.ndarray:
    """Load 16 kHz mono WAV as float32 numpy array."""
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()} channels"
        assert wf.getframerate() == 16000, f"Expected 16kHz, got {wf.getframerate()}"
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def _cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate via Levenshtein distance, ignoring whitespace."""
    ref = reference.replace(" ", "")
    hyp = hypothesis.replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1),
            )
        prev = curr
    return prev[m] / n


def _punct_recall(reference: str, hypothesis: str) -> float | None:
    """Recall of Chinese punctuation marks present in the reference."""
    puncts = set("。？！，、；：")
    ref_puncts = [c for c in reference if c in puncts]
    if not ref_puncts:
        return None
    hits = sum(1 for c in ref_puncts if c in hypothesis)
    return hits / len(ref_puncts)


def _is_traditional(text: str) -> bool | None:
    """Check if the Chinese characters are Traditional. Returns None if no CJK."""
    try:
        from opencc import OpenCC
    except ImportError:
        return None
    cjk_chars = [c for c in text if "一" <= c <= "鿿"]
    if not cjk_chars:
        return None
    cc = OpenCC("t2s")
    simplified = cc.convert(text)
    return simplified != text


BACKEND_CONFIGS = {
    "breeze-hf": ("breeze-asr", "MediaTek-Research/Breeze-ASR-25"),
    "breeze-fw": ("faster-whisper", "SoybeanMilk/faster-whisper-Breeze-ASR-25"),
    "qwen3-asr": ("qwen3-asr", "Qwen/Qwen3-ASR-0.6B"),
}


def _try_build_backend(label: str):
    """Try to build a backend; return None with a warning on failure."""
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


def _run_case(backend, case: dict) -> dict:
    """Run one case through a backend and compute metrics."""
    audio_path = REPO_ROOT / case["audio_path"]
    if not audio_path.exists():
        return {"error": f"audio not found: {audio_path}"}

    samples = _load_audio(audio_path)
    t0 = time.perf_counter()
    text, lang = backend.transcribe(samples)
    elapsed = time.perf_counter() - t0

    gt = case["ground_truth"]
    cer = _cer(gt, text)
    pr = _punct_recall(gt, text)
    trad = _is_traditional(text)

    failures = []
    for needle in case.get("must_contain", []):
        if needle.lower() not in text.lower():
            failures.append(f"must_contain {needle!r} missing")
    for needle in case.get("must_not_contain", []):
        if needle.lower() in text.lower():
            failures.append(f"must_not_contain {needle!r} present")

    return {
        "output": text,
        "language": lang,
        "cer": round(cer, 4),
        "punct_recall": round(pr, 4) if pr is not None else None,
        "is_traditional": trad,
        "latency_s": round(elapsed, 3),
        "pass": len(failures) == 0,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description="ASR quality benchmark")
    parser.add_argument(
        "--backends",
        default="all",
        help="Comma-separated backend labels (breeze-hf,breeze-fw,qwen3-asr) or 'all'",
    )
    parser.add_argument("--json-out", default=str(REPORT_PATH))
    args = parser.parse_args()

    if not FIXTURE_PATH.exists():
        print(f"ERROR: fixture not found at {FIXTURE_PATH}")
        print("Run: python scripts/gen_asr_bench_audio.py")
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
    print("ASR Quality Benchmark — Breeze-ASR-25 Evaluation")
    print("=" * 70)
    print(f"\nLoading backends ({len(labels)})...")

    backends = {}
    for label in labels:
        if label not in BACKEND_CONFIGS:
            print(f"  [{label}] UNKNOWN — skipping")
            continue
        b = _try_build_backend(label)
        if b:
            backends[label] = b

    if not backends:
        print("\nERROR: no backends loaded. Install required packages.")
        sys.exit(1)

    print(f"\nRunning {len(cases)} cases across {len(backends)} backend(s)...\n")

    results = {}
    for label, backend in backends.items():
        results[label] = {}
        for case in cases:
            r = _run_case(backend, case)
            results[label][case["id"]] = r
            status = "PASS" if r.get("pass") else "FAIL" if "error" not in r else "ERR"
            cer_str = f"CER={r['cer']:.1%}" if "cer" in r else ""
            print(f"  [{label}] {case['id']:40s} {status:4s} {cer_str:>10s}  {r.get('output', '')[:50]}")

    # ── Summary report ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    header = f"{'Case':40s}"
    for label in backends:
        header += f" | {label:>15s}"
    print(f"\n{header}")
    print("-" * len(header))

    for case in cases:
        row = f"{case['id']:40s}"
        for label in backends:
            r = results[label].get(case["id"], {})
            if "error" in r:
                row += f" | {'ERR':>15s}"
            else:
                cer = r.get("cer", -1)
                status = "✓" if r.get("pass") else "✗"
                row += f" | {status} CER={cer:.1%}".rjust(16)
        print(row)

    print("\n--- Averages ---")
    for label in backends:
        cers = [
            r["cer"]
            for r in results[label].values()
            if "cer" in r and "error" not in r
        ]
        passes = sum(1 for r in results[label].values() if r.get("pass"))
        total = sum(1 for r in results[label].values() if "error" not in r)
        prs = [
            r["punct_recall"]
            for r in results[label].values()
            if r.get("punct_recall") is not None
        ]
        trads = [
            r["is_traditional"]
            for r in results[label].values()
            if r.get("is_traditional") is not None
        ]
        avg_cer = sum(cers) / len(cers) if cers else -1
        avg_pr = sum(prs) / len(prs) if prs else -1
        trad_rate = sum(1 for t in trads if t) / len(trads) if trads else -1
        print(
            f"  {label:15s}: avg CER={avg_cer:.1%}, "
            f"pass={passes}/{total}, "
            f"punct_recall={avg_pr:.1%}, "
            f"traditional={trad_rate:.0%}"
        )

    # ── Write JSON ──
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "backends": {
            label: {"name": BACKEND_CONFIGS[label][0], "model": BACKEND_CONFIGS[label][1]}
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
