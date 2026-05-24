"""18-case polish-quality regression bench.

Loads the polish model (Qwen3-4B-Instruct-2507 by default) and runs each
fixture case through `_polisher.polish()`. Asserts each case's
`must_contain` / `must_not_contain` / `max_edit_ratio` rules.

**Skip-by-default in CI** — requires ~8 GB VRAM (Win) or ~4 GB RSS (Mac)
plus a downloaded model snapshot. Opt in locally with:

    pytest tests/test_polish_bench.py --run-polish-bench

or set HOME_STT_RUN_POLISH_BENCH=1.

Why this bench exists: v0.7.0 ran an 18-case bench (described in
README §"v0.7.x 效能與品質投資紀錄") to validate switching back from
Qwen2.5-1.5B to Qwen3-4B-Instruct-2507. The investigation was real, but
the cases were never committed — so any future model swap or
transformers upgrade could silently regress the fixes (commit→push,
更慢→更快, 幫我→幫你, etc.) with no automated guard. This file
captures the cases.

To add a case: edit `tests/fixtures/polish_cases.json` and re-run
the bench. Schema is documented in the JSON's _meta block.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "polish_cases.json"


def _bench_is_enabled(request) -> bool:
    return (
        request.config.getoption("--run-polish-bench")
        or os.environ.get("HOME_STT_RUN_POLISH_BENCH") == "1"
    )


@pytest.fixture(scope="session")
def polish_cases():
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


@pytest.fixture(scope="session")
def polisher(request, fresh_daemon_session):
    """Load the actual polish model. Session-scoped so one model load
    serves all 18 cases — re-loading per case would take ~30s × 18."""
    if not _bench_is_enabled(request):
        pytest.skip("polish bench disabled — pass --run-polish-bench "
                    "or set HOME_STT_RUN_POLISH_BENCH=1")
    from text_polisher import build_polisher
    p = build_polisher(
        enabled=True,
        model_name=fresh_daemon_session.POLISH_MODEL,
        system_prompt=fresh_daemon_session.POLISH_PROMPT,
    )
    # Sanity: build_polisher silently returns NoopPolisher on init failure
    # (e.g. CUDA OOM). That would make every case "pass" with the raw input.
    # Refuse to bench against a noop — abort cleanly.
    if p.__class__.__name__ == "NoopPolisher":
        pytest.skip(f"polish model {fresh_daemon_session.POLISH_MODEL} failed "
                    f"to initialise — bench would be meaningless against NoopPolisher")
    return p


@pytest.fixture(scope="session")
def fresh_daemon_session():
    """Session-scoped daemon access — bench loads model once at session
    setup, so we don't need per-test fixture reset (state isn't mutated
    by polish())."""
    import sys
    from pathlib import Path
    _SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stt_daemon", str(_SCRIPTS / "stt-daemon.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance. O(n*m) DP — fine for case
    inputs (≤100 chars each)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = curr
    return prev[-1]


def _ids(cases):
    return [c["id"] for c in cases]


# Generate one test per case. Pytest parametrize gives us per-case pass/fail
# visibility in CI output, which matters for triage ("which 3 of 18 broke?").
def _load_cases():
    if not _FIXTURE_PATH.exists():
        return []
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)["cases"]


_CASES = _load_cases()


@pytest.mark.parametrize("case", _CASES, ids=_ids(_CASES))
def test_polish_case(case, polisher, capfd):
    """Run one case through the loaded polish model, assert rules."""
    raw = case["input"]
    out = polisher.polish(raw)
    failures = []

    for needle in case.get("must_contain", []):
        if needle not in out:
            failures.append(f"must_contain {needle!r} missing")

    for needle in case.get("must_not_contain", []):
        if needle in out:
            failures.append(f"must_not_contain {needle!r} present")

    if "max_edit_ratio" in case:
        dist = _levenshtein(raw, out)
        ratio = dist / max(1, len(raw))
        if ratio > case["max_edit_ratio"]:
            failures.append(
                f"edit_ratio {ratio:.2f} > {case['max_edit_ratio']} "
                f"(dist={dist}, len={len(raw)})"
            )

    assert not failures, (
        f"\n--- Case {case['id']} ({case['category']}) ---\n"
        f"input:   {raw!r}\n"
        f"output:  {out!r}\n"
        f"note:    {case.get('note', '')}\n"
        f"failures: {failures}"
    )
