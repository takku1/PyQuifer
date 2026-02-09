"""Benchmark: BigCode Evaluation Harness — Code Generation.

Thin wrapper around the bigcode-evaluation-harness for code benchmarks
(HumanEval, MBPP, MultiPL-E, APPS).  Uses the same model adapter pattern
as lm-eval since bigcode-eval is a fork.

When the framework is not installed, runs in stub mode.

Three-column comparison:
  A_published : Published pass@k baselines
  B_pytorch   : Vanilla model pass@k
  C_pyquifer  : PyQuifer-modulated pass@k

Dual-mode: ``python bench_bigcode_eval.py`` or ``pytest bench_bigcode_eval.py -v``
"""
from __future__ import annotations

import os
import random
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

BIGCODE_DIR = _benchmark_dir / "bigcode-evaluation-harness"

TASKS = {
    "HumanEval": {"pass_at_1": 0.35, "source": "Chen et al. (2021)"},
    "MBPP": {"pass_at_1": 0.50, "source": "Austin et al. (2021)"},
}


def _has_bigcode() -> bool:
    try:
        sys.path.insert(0, str(BIGCODE_DIR))
        import main  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def run_full_suite() -> Dict:
    """Run BigCode evaluation (stub mode if framework unavailable)."""
    print("=" * 60)
    print("  BigCode Evaluation — Code Generation (stub)")
    print("=" * 60)

    if not _has_bigcode():
        print("  bigcode-evaluation-harness not configured; stub mode")

    t0 = time.perf_counter()
    random.seed(42)

    suite = BenchmarkSuite("BigCode Evaluation", metadata={"status": "stub", "is_stub": True})
    for task, baseline in TASKS.items():
        mc = MetricCollector(task)
        mc.record("A_published", "accuracy", baseline["pass_at_1"],
                  {"source": baseline["source"]})
        # No B/C results in stub mode — prevents synthetic score contamination
        suite.add(mc)

    elapsed = time.perf_counter() - t0

    results_dir = _benchmark_dir / "results"
    results_dir.mkdir(exist_ok=True)
    suite.to_json(str(results_dir / "bigcode_eval.json"))
    print(f"  Results saved (stub): {results_dir / 'bigcode_eval.json'}")

    return {"status": "skipped", "tasks": list(TASKS.keys()), "stub": True}


class TestBigCode:
    def test_suite_runs(self):
        result = run_full_suite()
        assert "tasks" in result


if __name__ == "__main__":
    run_full_suite()
