"""Benchmark: lm-evaluation-harness — Three-Column PyQuifer Comparison.

Runs a curated set of academic benchmarks via EleutherAI's lm-eval framework
with three conditions:

  A_published : Hardcoded baselines from published papers / leaderboards
  B_pytorch   : Vanilla HuggingFace model (--model hf)
  C_pyquifer  : PyQuifer-modulated model (--model pyquifer)

Covers 8 benchmark families through a single adapter:
  MMLU, GSM8K, TruthfulQA, HellaSwag, ARC-Challenge,
  BoolQ, WinoGrande, BIG-Bench-Hard

When no real model is configured (PYQUIFER_LLM_MODEL unset), falls back to
lm-eval's built-in ``dummy`` model for pipeline verification.

Env vars:
  PYQUIFER_LLM_MODEL   HF model name (default: "" → dummy)
  PYQUIFER_LLM_DEVICE  Device (default: "cpu")
  PYQUIFER_LM_EVAL_TASKS  Comma-separated task override (default: all)
  PYQUIFER_LM_EVAL_LIMIT  Max samples per task (default: 50 for dummy, None for real)

Dual-mode: ``python bench_lm_eval.py`` (full report) or
           ``pytest bench_lm_eval.py -v`` (smoke tests)
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

_benchmark_dir = Path(__file__).resolve().parent
_pyquifer_src = _benchmark_dir.parent.parent / "src"
if str(_pyquifer_src) not in sys.path:
    sys.path.insert(0, str(_pyquifer_src))

# Add lm-eval to path
_lm_eval_dir = _benchmark_dir / "lm-evaluation-harness"
if str(_lm_eval_dir) not in sys.path:
    sys.path.insert(0, str(_lm_eval_dir))

from harness import BenchmarkResult, BenchmarkSuite, MetricCollector


# ============================================================
# Task Configuration
# ============================================================

LM_EVAL_TASKS = {
    "MMLU": "mmlu",
    "GSM8K": "gsm8k",
    "TruthfulQA": "truthfulqa_mc2",
    "HellaSwag": "hellaswag",
    "ARC-Challenge": "arc_challenge",
    "BoolQ": "boolq",
    "WinoGrande": "winogrande",
    "BIG-Bench-Hard": "bbh",
}

# Published baselines (Phi-4-mini-instruct where available, else GPT-3.5 class)
# Each entry documents: accuracy, source, split, few_shot, metric, model
PUBLISHED_BASELINES = {
    "MMLU": {
        "accuracy": 0.70, "source": "Phi-4 technical report (2024)",
        "split": "test", "few_shot": 5, "metric": "acc",
        "model": "Phi-4-mini-instruct",
    },
    "GSM8K": {
        "accuracy": 0.85, "source": "Phi-4 technical report (2024)",
        "split": "test", "few_shot": 8, "metric": "exact_match",
        "model": "Phi-4-mini-instruct",
    },
    "TruthfulQA": {
        "accuracy": 0.55, "source": "Lin et al. (2022), GPT-3.5 class",
        "split": "validation", "few_shot": 0, "metric": "mc2",
        "model": "GPT-3.5 class", "note": "not comparable to Phi-4",
    },
    "HellaSwag": {
        "accuracy": 0.82, "source": "Phi-4 technical report (2024)",
        "split": "validation", "few_shot": 10, "metric": "acc_norm",
        "model": "Phi-4-mini-instruct",
    },
    "ARC-Challenge": {
        "accuracy": 0.62, "source": "Clark et al. (2018), GPT-3.5 class",
        "split": "test", "few_shot": 25, "metric": "acc_norm",
        "model": "GPT-3.5 class", "note": "not comparable to Phi-4",
    },
    "BoolQ": {
        "accuracy": 0.86, "source": "Clark et al. (2019)",
        "split": "validation", "few_shot": 0, "metric": "acc",
        "model": "GPT-3.5 class", "note": "not comparable to Phi-4",
    },
    "WinoGrande": {
        "accuracy": 0.75, "source": "Sakaguchi et al. (2020)",
        "split": "validation", "few_shot": 5, "metric": "acc",
        "model": "GPT-3.5 class", "note": "not comparable to Phi-4",
    },
    "BIG-Bench-Hard": {
        "accuracy": 0.60, "source": "Suzgun et al. (2022)",
        "split": "test", "few_shot": 3, "metric": "exact_match",
        "model": "GPT-3.5 class", "note": "not comparable to Phi-4",
    },
}


# ============================================================
# Runner
# ============================================================

def _get_task_list() -> Dict[str, str]:
    """Return task dict, possibly filtered by env var."""
    override = os.environ.get("PYQUIFER_LM_EVAL_TASKS", "")
    if override:
        keys = [k.strip() for k in override.split(",")]
        return {k: v for k, v in LM_EVAL_TASKS.items() if k in keys or v in keys}
    return dict(LM_EVAL_TASKS)


def _get_limit() -> Optional[int]:
    """Max samples per task.  Defaults to 50 for dummy, None for real."""
    env = os.environ.get("PYQUIFER_LM_EVAL_LIMIT", "")
    if env:
        return int(env)
    model = os.environ.get("PYQUIFER_LLM_MODEL", "")
    return None if model else 50


def _run_lm_eval(model_type: str, model_args: str, tasks: List[str],
                 limit: Optional[int]) -> Dict:
    """Run lm-eval programmatically and return results dict."""
    try:
        from lm_eval import evaluator
        results = evaluator.simple_evaluate(
            model=model_type,
            model_args=model_args,
            tasks=tasks,
            limit=limit,
            batch_size="auto",
            log_samples=False,
        )
        return results.get("results", {})
    except Exception as e:
        print(f"  lm-eval run failed ({model_type}): {e}")
        return {}


def _extract_accuracy(task_results: Dict) -> Optional[float]:
    """Extract primary accuracy metric from lm-eval task results."""
    # lm-eval uses different metric names per task
    for key in ["acc,none", "acc_norm,none", "exact_match,strict-match",
                "exact_match,none", "mc2,none", "acc,flexible-extract"]:
        if key in task_results:
            val = task_results[key]
            if isinstance(val, (int, float)):
                return float(val)
    return None


def _generate_dummy_results(tasks: Dict[str, str]) -> Dict[str, Dict]:
    """Generate synthetic results when lm-eval is not available or fails."""
    import random
    random.seed(42)
    results = {}
    for name, task_id in tasks.items():
        results[task_id] = {
            "acc,none": round(random.uniform(0.15, 0.35), 4),
            "alias": name,
            "_dummy": True,
        }
    return results


# ============================================================
# Full Suite
# ============================================================

def run_full_suite() -> Dict:
    """Run all lm-eval benchmarks and produce JSON output."""
    tasks = _get_task_list()
    limit = _get_limit()
    model_name = os.environ.get("PYQUIFER_LLM_MODEL", "")

    print("=" * 60)
    print("  lm-evaluation-harness — PyQuifer Three-Column Comparison")
    print("=" * 60)

    if model_name:
        print(f"\n  Model: {model_name}")
        print(f"  Tasks: {', '.join(tasks.keys())}")
        print(f"  Limit: {limit or 'full'}")
    else:
        print("\n  No real LLM configured (using dummy model)")
        print(f"  Tasks: {', '.join(tasks.keys())}")
        print(f"  Limit: {limit}")

    t0 = time.perf_counter()
    task_ids = list(tasks.values())

    b_results: Dict = {}
    c_results: Dict = {}

    if model_name:
        # --- Real model: run both conditions ---
        print("\n[1/2] Running Condition B (vanilla)...")
        b_model_args = f"pretrained={model_name}"
        b_results = _run_lm_eval("hf", b_model_args, task_ids, limit)

        print("\n[2/2] Running Condition C (pyquifer)...")
        c_model_args = f"pretrained={model_name}"
        c_results = _run_lm_eval("pyquifer", c_model_args, task_ids, limit)
    else:
        print("\n  No real LLM configured — only A_published baselines recorded")
        print("  Set PYQUIFER_LLM_MODEL=microsoft/Phi-4-mini-instruct to run")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")

    # --- Build harness output ---
    suite = BenchmarkSuite("lm-evaluation-harness")

    for bench_name, task_id in tasks.items():
        mc = MetricCollector(bench_name)

        # A_published — always emitted (with full provenance metadata)
        if bench_name in PUBLISHED_BASELINES:
            baseline = PUBLISHED_BASELINES[bench_name]
            meta = {k: v for k, v in baseline.items() if k != "accuracy"}
            mc.record("A_published", "accuracy", baseline["accuracy"], meta)

        # B/C only when real model results are available
        b_task = b_results.get(task_id, {})
        b_acc = _extract_accuracy(b_task)
        if b_acc is not None:
            mc.add_result(BenchmarkResult(
                name=f"{bench_name} (vanilla)", column="B_pytorch",
                metrics={"accuracy": b_acc},
                metadata={"task_id": task_id, "model_type": "hf"},
            ))

        c_task = c_results.get(task_id, {})
        c_acc = _extract_accuracy(c_task)
        if c_acc is not None:
            mc.add_result(BenchmarkResult(
                name=f"{bench_name} (pyquifer)", column="C_pyquifer",
                metrics={"accuracy": c_acc},
                metadata={"task_id": task_id, "model_type": "pyquifer"},
            ))

        suite.add(mc)

        # Print summary
        b_str = f"{b_acc:.4f}" if b_acc is not None else "--"
        c_str = f"{c_acc:.4f}" if c_acc is not None else "--"
        pub_str = f"{PUBLISHED_BASELINES.get(bench_name, {}).get('accuracy', '--')}"
        print(f"  {bench_name:20s}: A={pub_str}  B={b_str}  C={c_str}")

    results_dir = _benchmark_dir / "results"
    results_dir.mkdir(exist_ok=True)
    suite.to_json(str(results_dir / "lm_eval.json"))
    print(f"\nResults saved: {results_dir / 'lm_eval.json'}")

    return {"tasks": list(tasks.keys()), "elapsed_s": elapsed}


# ============================================================
# Pytest Smoke Tests
# ============================================================

class TestLmEvalSetup:
    def test_task_list_valid(self):
        tasks = _get_task_list()
        assert len(tasks) >= 1
        for name, task_id in tasks.items():
            assert isinstance(name, str) and len(name) > 0
            assert isinstance(task_id, str) and len(task_id) > 0

    def test_published_baselines_complete(self):
        for name in LM_EVAL_TASKS:
            assert name in PUBLISHED_BASELINES, f"Missing baseline for {name}"
            assert "accuracy" in PUBLISHED_BASELINES[name]
            assert 0 < PUBLISHED_BASELINES[name]["accuracy"] < 1

    def test_dummy_results_generation(self):
        tasks = _get_task_list()
        results = _generate_dummy_results(tasks)
        assert len(results) == len(tasks)
        for task_id in tasks.values():
            assert task_id in results
            assert "acc,none" in results[task_id]

    def test_extract_accuracy(self):
        assert _extract_accuracy({"acc,none": 0.75}) == 0.75
        assert _extract_accuracy({"mc2,none": 0.55}) == 0.55
        assert _extract_accuracy({}) is None


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_full_suite()
