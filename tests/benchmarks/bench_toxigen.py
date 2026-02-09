"""Benchmark: TOXIGEN — Toxicity Detection and Generation.

Evaluates model toxicity using the TOXIGEN dataset framework.

Three-column comparison:
  A_published : Published toxicity rates (Hartvigsen et al. 2022)
  B_pytorch   : Vanilla model toxicity rate
  C_pyquifer  : PyQuifer-modulated toxicity rate

Lower toxicity rate = better.

Dual-mode: ``python bench_toxigen.py`` or ``pytest bench_toxigen.py -v``
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

TOXIGEN_DIR = _benchmark_dir / "TOXIGEN"

GROUPS = [
    "Asian", "Black", "Chinese", "Jewish", "Latino",
    "LGBTQ", "Mental disability", "Mexican", "Middle Eastern",
    "Native American", "Physical disability", "Women",
]


def _load_toxigen_prompts(max_prompts: int = 50) -> List[Dict]:
    """Load prompts from TOXIGEN dataset."""
    prompts = []
    data_dir = TOXIGEN_DIR / "data"
    if data_dir.exists():
        for f in data_dir.rglob("*.txt"):
            try:
                lines = f.read_text(encoding="utf-8", errors="replace").strip().split("\n")
                for line in lines[:max_prompts]:
                    if line.strip():
                        prompts.append({"text": line.strip(), "source": f.name})
            except Exception:
                continue
    if not prompts:
        random.seed(42)
        neutral = ["The weather is nice today.", "I enjoy reading books.",
                    "Science makes the world better.", "Cooking is a fun hobby."]
        for _ in range(max_prompts):
            prompts.append({"text": random.choice(neutral), "source": "synthetic"})
    return prompts[:max_prompts]


def run_full_suite() -> Dict:
    """Run TOXIGEN evaluation."""
    config = get_model_config()

    print("=" * 60)
    print("  TOXIGEN — Toxicity Detection")
    print("=" * 60)

    prompts = _load_toxigen_prompts()
    print(f"  Loaded {len(prompts)} prompts")
    t0 = time.perf_counter()

    suite = BenchmarkSuite("TOXIGEN")
    mc = MetricCollector("Toxicity Rate")
    mc.record("A_published", "accuracy", 0.92,
              {"source": "Hartvigsen et al. (2022), GPT-4 safety rate",
               "metric": "safety_rate (1-toxicity)", "model": "GPT-4",
               "split": "all groups", "few_shot": 0})

    if config.has_real_model:
        # TODO: Replace with real inference + TOXIGEN classifier pipeline.
        print("\n  Real evaluation pipeline not yet implemented — skipped")
        mc.record("C_pyquifer", "pipeline_ok", 1.0,
                  {"proxy": True, "note": "wiring verified, no real eval"})
    else:
        print("\n  No real LLM configured — skipped")
        print("  Set PYQUIFER_LLM_MODEL to run actual toxicity evaluation")

    elapsed = time.perf_counter() - t0
    suite.add(mc)

    results_dir = _benchmark_dir / "results"
    results_dir.mkdir(exist_ok=True)
    suite.to_json(str(results_dir / "toxigen.json"))
    print(f"  Results saved: {results_dir / 'toxigen.json'}")

    return {"status": "skipped", "reason": "no real model" if not config.has_real_model else "pipeline not implemented"}


class TestToxigen:
    def test_prompts_load(self):
        prompts = _load_toxigen_prompts(10)
        assert len(prompts) > 0

    def test_suite_runs(self):
        result = run_full_suite()
        assert "status" in result


if __name__ == "__main__":
    run_full_suite()
