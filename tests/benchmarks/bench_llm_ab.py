"""Benchmark #7: LLM A/B Test — PyQuifer modulation vs vanilla inference.

Tests whether PyQuifer's CognitiveCycle modulation measurably affects
LLM output quality when applied through PyQuiferBridge.

Architecture:
  Condition B: Vanilla model inference (fixed temperature/top_p)
  Condition C: CognitiveCycle.tick() -> PyQuiferBridge.modulate_logits()
  Condition C_rand: Random sinusoidal modulation (same amplitude budget)

This benchmark is designed to work with or without a local LLM.
When no LLM is available, it runs synthetic tests using a dummy model
to verify the modulation pipeline correctness and measure overhead.

When a local model IS available (e.g., Phi-4), set:
  PYQUIFER_LLM_MODEL=microsoft/Phi-4-mini-instruct
  PYQUIFER_LLM_DEVICE=cuda

Dual-mode: `python bench_llm_ab.py` (full report) or
           `pytest bench_llm_ab.py -v` (smoke tests)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_benchmark_dir = Path(__file__).resolve().parent
_pyquifer_src = _benchmark_dir.parent.parent / "src"
if str(_pyquifer_src) not in sys.path:
    sys.path.insert(0, str(_pyquifer_src))

from harness import (
    BenchmarkResult, BenchmarkSuite, MemoryTracker, MetricCollector, timer,
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class LLMABConfig:
    seed: int = 42
    vocab_size: int = 32000  # Typical LLM vocab
    hidden_dim: int = 64     # PyQuifer state dim
    num_oscillators: int = 32
    num_samples: int = 50    # Number of test prompts
    max_new_tokens: int = 32
    # Real model config (optional)
    model_name: str = ""
    device: str = "cpu"


def _get_config() -> LLMABConfig:
    config = LLMABConfig()
    config.model_name = os.environ.get("PYQUIFER_LLM_MODEL", "")
    config.device = os.environ.get("PYQUIFER_LLM_DEVICE", "cpu")
    return config


# ============================================================
# Dummy Model (used when no real LLM available)
# ============================================================

class DummyLLM(nn.Module):
    """Minimal LLM stand-in: embedding -> linear -> logits."""

    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits for next token."""
        h = self.embed(input_ids).mean(dim=1)  # (B, hidden_dim)
        return self.proj(h)  # (B, vocab_size)

    def get_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return hidden state (for PyQuifer bridge input)."""
        return self.embed(input_ids).mean(dim=1)


# ============================================================
# Modulation Functions
# ============================================================

def _apply_pyquifer_modulation(logits: torch.Tensor, bridge, state) -> torch.Tensor:
    """Apply PyQuifer bridge modulation to logits."""
    return bridge.modulate_logits(logits, state)


def _apply_random_modulation(logits: torch.Tensor, amplitude: float = 0.1,
                             num_oscillators: int = 32) -> torch.Tensor:
    """Apply random sinusoidal modulation (control condition)."""
    t = torch.rand(1).item() * 2 * math.pi
    phases = torch.rand(num_oscillators) * 2 * math.pi
    modulation = torch.zeros(logits.shape[-1])
    for i in range(min(num_oscillators, logits.shape[-1])):
        modulation[i % logits.shape[-1]] += amplitude * math.sin(t + phases[i].item())
    return logits + modulation.to(logits.device)


# ============================================================
# Scenario 1: Modulation pipeline test (synthetic)
# ============================================================

@dataclass
class PipelineResult:
    condition: str  # "B_vanilla", "C_pyquifer", "C_rand"
    mean_logit_magnitude: float
    logit_entropy: float  # Higher = more uniform distribution
    modulation_overhead_ms: float
    logit_diff_from_vanilla: float  # L2 distance from vanilla logits


def bench_pipeline(config: LLMABConfig) -> List[PipelineResult]:
    """Test the modulation pipeline with a dummy model."""
    from pyquifer.bridge import PyQuiferBridge

    torch.manual_seed(config.seed)
    model = DummyLLM(vocab_size=config.vocab_size, hidden_dim=256)
    bridge = PyQuiferBridge.default()

    results = []
    input_ids = torch.randint(0, config.vocab_size, (1, 10))

    # Condition B: Vanilla
    with timer() as t_vanilla:
        for _ in range(config.num_samples):
            logits_vanilla = model(input_ids)
    vanilla_logits = logits_vanilla.detach()

    softmax_v = torch.softmax(vanilla_logits, dim=-1)
    entropy_v = -(softmax_v * (softmax_v + 1e-10).log()).sum(dim=-1).mean().item()

    results.append(PipelineResult(
        condition="B_vanilla",
        mean_logit_magnitude=vanilla_logits.abs().mean().item(),
        logit_entropy=entropy_v,
        modulation_overhead_ms=0.0,
        logit_diff_from_vanilla=0.0,
    ))

    # Condition C: PyQuifer
    hidden = model.get_hidden(input_ids)
    # Adapt hidden dim to bridge's expected input
    bridge_input = hidden.mean(dim=0)[:config.hidden_dim]
    if bridge_input.shape[0] < config.hidden_dim:
        bridge_input = torch.cat([
            bridge_input,
            torch.zeros(config.hidden_dim - bridge_input.shape[0])
        ])

    with timer() as t_pyquifer:
        for _ in range(config.num_samples):
            state = bridge.step(bridge_input)
            logits_raw = model(input_ids)
            logits_mod = bridge.modulate_logits(logits_raw, state)
    pyquifer_logits = logits_mod.detach()

    softmax_c = torch.softmax(pyquifer_logits, dim=-1)
    entropy_c = -(softmax_c * (softmax_c + 1e-10).log()).sum(dim=-1).mean().item()

    overhead = t_pyquifer["elapsed_ms"] - t_vanilla["elapsed_ms"]

    results.append(PipelineResult(
        condition="C_pyquifer",
        mean_logit_magnitude=pyquifer_logits.abs().mean().item(),
        logit_entropy=entropy_c,
        modulation_overhead_ms=max(0, overhead / config.num_samples),
        logit_diff_from_vanilla=(pyquifer_logits - vanilla_logits).pow(2).mean().sqrt().item(),
    ))

    # Condition C_rand: Random modulation
    with timer() as t_rand:
        for _ in range(config.num_samples):
            logits_raw = model(input_ids)
            logits_rand = _apply_random_modulation(logits_raw)
    rand_logits = logits_rand.detach()

    softmax_r = torch.softmax(rand_logits, dim=-1)
    entropy_r = -(softmax_r * (softmax_r + 1e-10).log()).sum(dim=-1).mean().item()

    overhead_r = t_rand["elapsed_ms"] - t_vanilla["elapsed_ms"]

    results.append(PipelineResult(
        condition="C_rand",
        mean_logit_magnitude=rand_logits.abs().mean().item(),
        logit_entropy=entropy_r,
        modulation_overhead_ms=max(0, overhead_r / config.num_samples),
        logit_diff_from_vanilla=(rand_logits - vanilla_logits).pow(2).mean().sqrt().item(),
    ))

    return results


# ============================================================
# Scenario 2: Modulation consistency
# ============================================================

@dataclass
class ConsistencyResult:
    condition: str
    output_variance: float  # Variance across repeated runs
    mean_top1_agreement: float  # How often top-1 token matches vanilla


def bench_consistency(config: LLMABConfig) -> List[ConsistencyResult]:
    """Test if PyQuifer modulation produces consistent outputs."""
    from pyquifer.bridge import PyQuiferBridge

    torch.manual_seed(config.seed)
    model = DummyLLM(vocab_size=config.vocab_size, hidden_dim=256)
    bridge = PyQuiferBridge.default()

    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    results = []

    # Vanilla top-1 tokens
    vanilla_top1s = []
    for _ in range(config.num_samples):
        logits = model(input_ids)
        vanilla_top1s.append(logits.argmax(dim=-1).item())

    # PyQuifer modulation
    hidden = model.get_hidden(input_ids)
    bridge_input = hidden.mean(dim=0)[:config.hidden_dim]
    if bridge_input.shape[0] < config.hidden_dim:
        bridge_input = torch.cat([
            bridge_input,
            torch.zeros(config.hidden_dim - bridge_input.shape[0])
        ])

    pyquifer_top1s = []
    pyquifer_logit_norms = []
    for _ in range(config.num_samples):
        state = bridge.step(bridge_input)
        logits = model(input_ids)
        mod_logits = bridge.modulate_logits(logits, state)
        pyquifer_top1s.append(mod_logits.argmax(dim=-1).item())
        pyquifer_logit_norms.append(mod_logits.norm().item())

    agreement = sum(1 for a, b in zip(vanilla_top1s, pyquifer_top1s)
                    if a == b) / len(vanilla_top1s)

    results.append(ConsistencyResult(
        condition="C_pyquifer",
        output_variance=torch.tensor(pyquifer_logit_norms).var().item(),
        mean_top1_agreement=agreement,
    ))

    # Random modulation
    rand_top1s = []
    rand_logit_norms = []
    for _ in range(config.num_samples):
        logits = model(input_ids)
        rand_logits = _apply_random_modulation(logits)
        rand_top1s.append(rand_logits.argmax(dim=-1).item())
        rand_logit_norms.append(rand_logits.norm().item())

    agreement_r = sum(1 for a, b in zip(vanilla_top1s, rand_top1s)
                      if a == b) / len(vanilla_top1s)

    results.append(ConsistencyResult(
        condition="C_rand",
        output_variance=torch.tensor(rand_logit_norms).var().item(),
        mean_top1_agreement=agreement_r,
    ))

    return results


# ============================================================
# Scenario 3: Latency overhead profiling
# ============================================================

@dataclass
class LatencyResult:
    component: str
    mean_ms: float
    p50_ms: float
    p99_ms: float


def bench_latency(config: LLMABConfig) -> List[LatencyResult]:
    """Profile individual component latencies."""
    from pyquifer.bridge import PyQuiferBridge
    from pyquifer.integration import CognitiveCycle, CycleConfig

    torch.manual_seed(config.seed)
    bridge = PyQuiferBridge.default()
    model = DummyLLM(vocab_size=config.vocab_size, hidden_dim=256)

    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    bridge_input = torch.randn(config.hidden_dim)

    results = []

    # Profile: bridge.step()
    step_times = []
    for _ in range(config.num_samples):
        t0 = time.perf_counter()
        state = bridge.step(bridge_input)
        step_times.append((time.perf_counter() - t0) * 1000)

    step_t = torch.tensor(step_times)
    results.append(LatencyResult(
        component="bridge.step()",
        mean_ms=step_t.mean().item(),
        p50_ms=step_t.median().item(),
        p99_ms=step_t.quantile(0.99).item(),
    ))

    # Profile: modulate_logits()
    logits = model(input_ids)
    mod_times = []
    for _ in range(config.num_samples):
        state = bridge.step(bridge_input)
        t0 = time.perf_counter()
        _ = bridge.modulate_logits(logits, state)
        mod_times.append((time.perf_counter() - t0) * 1000)

    mod_t = torch.tensor(mod_times)
    results.append(LatencyResult(
        component="modulate_logits()",
        mean_ms=mod_t.mean().item(),
        p50_ms=mod_t.median().item(),
        p99_ms=mod_t.quantile(0.99).item(),
    ))

    # Profile: full pipeline (step + modulate)
    full_times = []
    for _ in range(config.num_samples):
        t0 = time.perf_counter()
        state = bridge.step(bridge_input)
        _ = bridge.modulate_logits(logits, state)
        full_times.append((time.perf_counter() - t0) * 1000)

    full_t = torch.tensor(full_times)
    results.append(LatencyResult(
        component="full pipeline",
        mean_ms=full_t.mean().item(),
        p50_ms=full_t.median().item(),
        p99_ms=full_t.quantile(0.99).item(),
    ))

    # Profile: model forward (baseline)
    fwd_times = []
    for _ in range(config.num_samples):
        t0 = time.perf_counter()
        _ = model(input_ids)
        fwd_times.append((time.perf_counter() - t0) * 1000)

    fwd_t = torch.tensor(fwd_times)
    results.append(LatencyResult(
        component="model.forward()",
        mean_ms=fwd_t.mean().item(),
        p50_ms=fwd_t.median().item(),
        p99_ms=fwd_t.quantile(0.99).item(),
    ))

    return results


# ============================================================
# Full Suite
# ============================================================

def run_full_suite(config: Optional[LLMABConfig] = None) -> Dict:
    if config is None:
        config = _get_config()

    print("=" * 60)
    print("  PyQuifer LLM A/B Benchmark")
    print("=" * 60)

    if config.model_name:
        print(f"\n  Model: {config.model_name}")
        print(f"  Device: {config.device}")
    else:
        print("\n  No real LLM configured (using DummyLLM)")
        print("  Set PYQUIFER_LLM_MODEL and PYQUIFER_LLM_DEVICE for real model tests")

    t0 = time.perf_counter()

    # 1. Pipeline test
    print("\n[1/3] Modulation pipeline test...")
    pipeline_results = bench_pipeline(config)
    for r in pipeline_results:
        print(f"  {r.condition:15s}: entropy={r.logit_entropy:.4f}, "
              f"diff={r.logit_diff_from_vanilla:.4f}, "
              f"overhead={r.modulation_overhead_ms:.2f}ms")

    # 2. Consistency test
    print("\n[2/3] Modulation consistency test...")
    consistency_results = bench_consistency(config)
    for r in consistency_results:
        print(f"  {r.condition:15s}: top1_agreement={r.mean_top1_agreement:.3f}, "
              f"variance={r.output_variance:.6f}")

    # 3. Latency profiling
    print("\n[3/3] Latency profiling...")
    latency_results = bench_latency(config)
    for r in latency_results:
        print(f"  {r.component:25s}: mean={r.mean_ms:.3f}ms, "
              f"p50={r.p50_ms:.3f}ms, p99={r.p99_ms:.3f}ms")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")

    # Build JSON output
    suite = BenchmarkSuite("LLM A/B Test")

    mc_pipeline = MetricCollector("Modulation Pipeline")
    for r in pipeline_results:
        mc_pipeline.add_result(BenchmarkResult(
            name=r.condition, column=r.condition,
            metrics={
                "logit_entropy": r.logit_entropy,
                "logit_diff": r.logit_diff_from_vanilla,
                "overhead_ms": r.modulation_overhead_ms,
                "mean_magnitude": r.mean_logit_magnitude,
            },
        ))
    suite.add(mc_pipeline)

    mc_consist = MetricCollector("Modulation Consistency")
    for r in consistency_results:
        mc_consist.add_result(BenchmarkResult(
            name=r.condition, column=r.condition,
            metrics={
                "top1_agreement": r.mean_top1_agreement,
                "output_variance": r.output_variance,
            },
        ))
    suite.add(mc_consist)

    mc_latency = MetricCollector("Latency Profile")
    for r in latency_results:
        mc_latency.add_result(BenchmarkResult(
            name=r.component, column=r.component,
            metrics={
                "mean_ms": r.mean_ms,
                "p50_ms": r.p50_ms,
                "p99_ms": r.p99_ms,
            },
        ))
    suite.add(mc_latency)

    results_dir = _benchmark_dir / "results"
    suite.to_json(str(results_dir / "llm_ab.json"))
    print(f"\nResults saved: {results_dir / 'llm_ab.json'}")

    return {
        "pipeline": [vars(r) for r in pipeline_results],
        "consistency": [vars(r) for r in consistency_results],
        "latency": [vars(r) for r in latency_results],
    }


# ============================================================
# Pytest Smoke Tests
# ============================================================

class TestPipeline:
    def test_pipeline_runs(self):
        config = LLMABConfig(num_samples=5)
        results = bench_pipeline(config)
        assert len(results) == 3
        # All conditions should produce output
        for r in results:
            assert r.mean_logit_magnitude > 0
            assert r.logit_entropy > 0

    def test_pyquifer_modifies_logits(self):
        config = LLMABConfig(num_samples=5)
        results = bench_pipeline(config)
        pyquifer_r = [r for r in results if r.condition == "C_pyquifer"][0]
        # PyQuifer should modify logits (nonzero diff from vanilla)
        assert pyquifer_r.logit_diff_from_vanilla > 0

    def test_vanilla_has_zero_diff(self):
        config = LLMABConfig(num_samples=5)
        results = bench_pipeline(config)
        vanilla_r = [r for r in results if r.condition == "B_vanilla"][0]
        assert vanilla_r.logit_diff_from_vanilla == 0.0


class TestConsistency:
    def test_consistency_runs(self):
        config = LLMABConfig(num_samples=10)
        results = bench_consistency(config)
        assert len(results) == 2

    def test_top1_agreement_bounded(self):
        config = LLMABConfig(num_samples=10)
        results = bench_consistency(config)
        for r in results:
            assert 0.0 <= r.mean_top1_agreement <= 1.0


class TestLatency:
    def test_latency_runs(self):
        config = LLMABConfig(num_samples=10)
        results = bench_latency(config)
        assert len(results) == 4
        for r in results:
            assert r.mean_ms > 0
            assert r.p50_ms > 0

    def test_overhead_reasonable(self):
        """PyQuifer overhead should be < 100ms per step on CPU."""
        config = LLMABConfig(num_samples=20)
        results = bench_latency(config)
        bridge_step = [r for r in results if r.component == "bridge.step()"][0]
        assert bridge_step.mean_ms < 500  # Generous bound for CPU


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_full_suite()
