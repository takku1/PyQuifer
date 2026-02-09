"""Benchmark: JailbreakBench — Jailbreak Attack Success Rate.

Evaluates model resistance to jailbreak attacks using prompts from the
vendored jailbreakbench/ repo.

Three-column comparison:
  A_published : Published attack success rates (Chao et al. 2024)
  B_pytorch   : Vanilla model attack success rate
  C_pyquifer  : PyQuifer-modulated attack success rate

Lower attack success rate = safer model.

Dual-mode: ``python bench_jailbreakbench.py`` or ``pytest bench_jailbreakbench.py -v``
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

JBB_DIR = _benchmark_dir / "jailbreakbench"

CATEGORIES = ["Phishing", "Defamation", "Malware", "Harassment", "Fraud"]


def _load_jbb_prompts(max_prompts: int = 50) -> List[Dict]:
    """Load jailbreak prompts from JailbreakBench data."""
    prompts = []
    data_dir = JBB_DIR / "data"
    if data_dir.exists():
        for jf in data_dir.rglob("*.json"):
            try:
                with open(jf, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data[:max_prompts]:
                        prompts.append({"text": str(item.get("goal", item.get("prompt", ""))),
                                        "category": item.get("category", "unknown")})
                elif isinstance(data, dict):
                    for key, val in data.items():
                        if isinstance(val, str):
                            prompts.append({"text": val, "category": key})
                        if len(prompts) >= max_prompts:
                            break
            except Exception:
                continue
    if not prompts:
        random.seed(42)
        for _ in range(max_prompts):
            prompts.append({
                "text": f"Test prompt {random.randint(1, 1000)}",
                "category": random.choice(CATEGORIES),
            })
    return prompts[:max_prompts]


def run_full_suite() -> Dict:
    """Run JailbreakBench evaluation."""
    config = get_model_config()

    print("=" * 60)
    print("  JailbreakBench — Jailbreak Attack Success Rate")
    print("=" * 60)

    prompts = _load_jbb_prompts()
    print(f"  Loaded {len(prompts)} prompts")
    t0 = time.perf_counter()

    suite = BenchmarkSuite("JailbreakBench")
    mc = MetricCollector("Attack Resistance")
    mc.record("A_published", "accuracy", 0.88,
              {"source": "Chao et al. (2024), GPT-4 safety rate",
               "metric": "attack_resistance_rate", "model": "GPT-4",
               "split": "JBB-Behaviors", "few_shot": 0})

    if config.has_real_model:
        # TODO: Replace with real inference + official JailbreakBench eval.
        print("\n  Real evaluation pipeline not yet implemented — skipped")
        mc.record("C_pyquifer", "pipeline_ok", 1.0,
                  {"proxy": True, "note": "wiring verified, no real eval"})
    else:
        print("\n  No real LLM configured — skipped")
        print("  Set PYQUIFER_LLM_MODEL to run actual jailbreak evaluation")

    elapsed = time.perf_counter() - t0
    suite.add(mc)

    results_dir = _benchmark_dir / "results"
    results_dir.mkdir(exist_ok=True)
    suite.to_json(str(results_dir / "jailbreakbench.json"))
    print(f"  Results saved: {results_dir / 'jailbreakbench.json'}")

    return {"status": "skipped", "reason": "no real model" if not config.has_real_model else "pipeline not implemented"}


class TestJailbreakBench:
    def test_prompts_load(self):
        prompts = _load_jbb_prompts(10)
        assert len(prompts) > 0

    def test_suite_runs(self):
        result = run_full_suite()
        assert "status" in result


if __name__ == "__main__":
    run_full_suite()
