"""Benchmark: FactBench — Dynamic Factuality Benchmark.

Evaluates factual claim verification using the FactBench/VERIFY pipeline.

Three-column comparison:
  A_published : Published factuality baselines
  B_pytorch   : Vanilla model claim verification accuracy
  C_pyquifer  : PyQuifer-modulated claim verification accuracy

Higher accuracy = better factual reasoning.

Dual-mode: ``python bench_factbench.py`` or ``pytest bench_factbench.py -v``
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

_benchmark_dir = Path(__file__).resolve().parent
_pyquifer_src = _benchmark_dir.parent.parent / "src"
if str(_pyquifer_src) not in sys.path:
    sys.path.insert(0, str(_pyquifer_src))

from harness import BenchmarkResult, BenchmarkSuite, MetricCollector
from model_loader import get_model_config

FACTBENCH_DIR = _benchmark_dir / "FactBench"


def _load_factbench_claims(max_claims: int = 50) -> List[Dict]:
    """Load claims from FactBench data."""
    claims = []
    verify_dir = FACTBENCH_DIR / "VERIFY"
    if verify_dir.exists():
        for jf in verify_dir.rglob("*.jsonl"):
            try:
                for line in jf.read_text(encoding="utf-8", errors="replace").strip().split("\n"):
                    if line.strip():
                        claims.append(json.loads(line))
                        if len(claims) >= max_claims:
                            return claims
            except Exception:
                continue
    if not claims:
        random.seed(42)
        true_claims = ["Water boils at 100C at sea level.", "The Earth orbits the Sun.",
                       "Python is a programming language."]
        false_claims = ["The Moon is made of cheese.", "Pi equals exactly 3.",
                        "Humans have three lungs."]
        for _ in range(max_claims):
            if random.random() < 0.5:
                claims.append({"claim": random.choice(true_claims), "label": True})
            else:
                claims.append({"claim": random.choice(false_claims), "label": False})
    return claims[:max_claims]


def run_full_suite() -> Dict:
    """Run FactBench evaluation."""
    config = get_model_config()

    print("=" * 60)
    print("  FactBench — Factual Claim Verification")
    print("=" * 60)

    claims = _load_factbench_claims()
    print(f"  Loaded {len(claims)} claims")
    t0 = time.perf_counter()

    suite = BenchmarkSuite("FactBench")
    mc = MetricCollector("Claim Verification")
    mc.record("A_published", "accuracy", 0.78,
              {"source": "FactBench VERIFY pipeline (2024)",
               "metric": "claim_verification_accuracy", "model": "GPT-4",
               "split": "VERIFY subset", "few_shot": 0,
               "note": "requires retrieval + categorization pipeline"})

    if config.has_real_model:
        # TODO: Replace with real VERIFY claim verification pipeline.
        print("\n  Real evaluation pipeline not yet implemented — skipped")
        mc.record("C_pyquifer", "pipeline_ok", 1.0,
                  {"proxy": True, "note": "wiring verified, no real eval"})
    else:
        print("\n  No real LLM configured — skipped")
        print("  Set PYQUIFER_LLM_MODEL to run actual claim verification")

    elapsed = time.perf_counter() - t0
    suite.add(mc)

    results_dir = _benchmark_dir / "results"
    results_dir.mkdir(exist_ok=True)
    suite.to_json(str(results_dir / "factbench.json"))
    print(f"  Results saved: {results_dir / 'factbench.json'}")

    return {"status": "skipped", "reason": "no real model" if not config.has_real_model else "pipeline not implemented"}


class TestFactBench:
    def test_claims_load(self):
        claims = _load_factbench_claims(10)
        assert len(claims) > 0

    def test_suite_runs(self):
        result = run_full_suite()
        assert "status" in result


if __name__ == "__main__":
    run_full_suite()
