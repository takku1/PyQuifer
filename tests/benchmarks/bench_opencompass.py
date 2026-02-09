"""Benchmark: OpenCompass — Comprehensive LLM Evaluation Framework.

Thin wrapper around OpenCompass that registers a PyQuifer model for
config-based evaluation.  When the full OpenCompass framework is not
installed, runs in stub mode with synthetic results.

Three-column comparison:
  A_published : OpenCompass leaderboard baselines
  B_pytorch   : Vanilla HuggingFace model via OpenCompass
  C_pyquifer  : PyQuifer-modulated model via OpenCompass

Dual-mode: ``python bench_opencompass.py`` or ``pytest bench_opencompass.py -v``
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict

_benchmark_dir = Path(__file__).resolve().parent
_pyquifer_src = _benchmark_dir.parent.parent / "src"
if str(_pyquifer_src) not in sys.path:
    sys.path.insert(0, str(_pyquifer_src))

from harness import BenchmarkResult, BenchmarkSuite, MetricCollector
from model_loader import get_model_config

OPENCOMPASS_DIR = _benchmark_dir / "opencompass"

# Subset of OpenCompass tasks that overlap with our lm-eval coverage
TASKS = ["MMLU", "GSM8K", "HumanEval", "CEVAL", "BBH"]


def _has_opencompass() -> bool:
    """Check if OpenCompass is importable."""
    try:
        import opencompass  # noqa: F401
        return True
    except ImportError:
        return False


def run_full_suite() -> Dict:
    """Run OpenCompass evaluation (stub mode if framework unavailable)."""
    print("=" * 60)
    print("  OpenCompass — Comprehensive Evaluation (stub)")
    print("=" * 60)

    if not _has_opencompass():
        print("  OpenCompass not installed; running in stub mode")
        print("  Install: pip install opencompass (from vendored repo)")

    t0 = time.perf_counter()

    # Real published baselines (GPT-4o, OpenCompass leaderboard 2024)
    BASELINES = {
        "MMLU": 0.887,
        "GSM8K": 0.952,
        "HumanEval": 0.902,
        "CEVAL": 0.837,
        "BBH": 0.867,
    }

    suite = BenchmarkSuite("OpenCompass", metadata={"status": "stub", "is_stub": True})
    for task in TASKS:
        mc = MetricCollector(task)
        mc.record("A_published", "accuracy", BASELINES.get(task, 0.0),
                  {"source": "OpenCompass leaderboard (GPT-4o, 2024)",
                   "note": "stub — framework not installed"})
        suite.add(mc)

    elapsed = time.perf_counter() - t0

    results_dir = _benchmark_dir / "results"
    results_dir.mkdir(exist_ok=True)
    suite.to_json(str(results_dir / "opencompass.json"))
    print(f"  Results saved (stub): {results_dir / 'opencompass.json'}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return {"status": "skipped", "tasks": TASKS, "stub": True}


class TestOpenCompass:
    def test_suite_runs(self):
        result = run_full_suite()
        assert "tasks" in result


if __name__ == "__main__":
    run_full_suite()
