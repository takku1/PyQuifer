"""Benchmark: SWE-bench — Software Engineering Task Resolution.

SWE-bench requires an agent-based approach (patch generation + Docker
execution).  This is fundamentally different from text generation and
requires significant infrastructure.  Marked as skip-stub.

Dual-mode: ``python bench_swe_bench.py`` or ``pytest bench_swe_bench.py -v``
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Dict

import pytest

_benchmark_dir = Path(__file__).resolve().parent
_pyquifer_src = _benchmark_dir.parent.parent / "src"
if str(_pyquifer_src) not in sys.path:
    sys.path.insert(0, str(_pyquifer_src))

from harness import BenchmarkResult, BenchmarkSuite, MetricCollector

SWE_BENCH_DIR = _benchmark_dir / "SWE-bench"


def run_full_suite() -> Dict:
    """SWE-bench stub — requires agent infrastructure."""
    print("=" * 60)
    print("  SWE-bench — Software Engineering (stub)")
    print("=" * 60)
    print("  SKIPPED: Requires Docker + agent patch generation")
    print("  Set PYQUIFER_SWE_BENCH=1 to enable (needs Docker)")

    suite = BenchmarkSuite("SWE-bench", metadata={"status": "skipped", "is_stub": True})
    mc = MetricCollector("Patch Resolution")
    mc.record("A_published", "accuracy", 0.27,
              {"source": "SWE-bench Lite leaderboard (2024)"})
    suite.add(mc)

    results_dir = _benchmark_dir / "results"
    results_dir.mkdir(exist_ok=True)
    suite.to_json(str(results_dir / "swe_bench.json"))

    return {"status": "skipped", "reason": "requires Docker + agent"}


class TestSWEBench:
    @pytest.mark.skipif(
        not os.environ.get("PYQUIFER_SWE_BENCH"),
        reason="SWE-bench requires Docker + agent infrastructure",
    )
    def test_suite_runs(self):
        result = run_full_suite()
        assert result is not None

    def test_stub_runs(self):
        result = run_full_suite()
        assert result["status"] == "skipped"


if __name__ == "__main__":
    run_full_suite()
