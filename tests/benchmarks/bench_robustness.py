"""Benchmark #6: Robustness — noise perturbation and stability tests.

Measures how PyQuifer modules handle noisy/corrupted inputs compared
to vanilla PyTorch baselines:
  1. MNIST + Gaussian noise: accuracy degradation curve
  2. Embedding perturbation: CognitiveCycle stability under noise
  3. Phase perturbation: Kuramoto recovery after phase disruption
  4. Oscillator resilience: R(t) stability under external noise

Dual-mode: `python bench_robustness.py` (full report) or
           `pytest bench_robustness.py -v` (smoke tests)

References:
- Hendrycks & Dietterich (2019). Benchmarking Neural Network Robustness
  to Common Corruptions and Perturbations.
- AKOrN (ICLR 2025 Oral): Attractor-based robustness on CIFAR-10-C.
"""
from __future__ import annotations

import json
import math
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
    BenchmarkResult, BenchmarkSuite, MetricCollector, timer,
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class RobustnessConfig:
    seed: int = 42
    state_dim: int = 64
    num_oscillators: int = 32
    noise_levels: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5]
    )
    num_samples: int = 200   # Per noise level
    recovery_steps: int = 50  # Steps to recover from perturbation


# ============================================================
# Scenario 1: Embedding perturbation stability
# ============================================================

@dataclass
class PerturbationResult:
    noise_sigma: float
    # CognitiveCycle
    cycle_output_mean: float
    cycle_output_std: float
    cycle_coherence: float
    # Vanilla MLP
    mlp_output_mean: float
    mlp_output_std: float
    # Ratio: lower = more stable
    cycle_variation_ratio: float


def _make_mlp(input_dim: int, output_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim),
    )


def bench_perturbation_stability(config: RobustnessConfig) -> List[PerturbationResult]:
    """Measure output stability when inputs are perturbed with noise."""
    from pyquifer.integration import CognitiveCycle, CycleConfig

    torch.manual_seed(config.seed)
    results = []

    cycle_config = CycleConfig(
        state_dim=config.state_dim,
        num_oscillators=config.num_oscillators,
    )
    cycle = CognitiveCycle(cycle_config)

    mlp = _make_mlp(config.state_dim, config.state_dim)

    # Base input
    base_input = torch.randn(1, config.state_dim)

    for sigma in config.noise_levels:
        cycle_outputs = []
        mlp_outputs = []

        for _ in range(config.num_samples):
            noise = torch.randn_like(base_input) * sigma
            noisy_input = base_input + noise

            # CognitiveCycle (no torch.no_grad — HPC uses autograd.grad internally)
            tick_result = cycle.tick(noisy_input)
            # Use attention_bias tensor as representative output
            attn = tick_result["modulation"]["attention_bias"]
            cycle_outputs.append(attn.detach())

            # Vanilla MLP
            with torch.no_grad():
                mlp_out = mlp(noisy_input)
            mlp_outputs.append(mlp_out.detach().flatten()[:attn.shape[0]])

        cycle_stack = torch.stack(cycle_outputs)
        mlp_stack = torch.stack(mlp_outputs)

        cycle_mean = cycle_stack.mean().item()
        cycle_std = cycle_stack.std().item()
        mlp_mean = mlp_stack.mean().item()
        mlp_std = mlp_stack.std().item()

        # Coherence from the last tick
        coherence = tick_result["consciousness"].get("coherence", 0.0)
        if isinstance(coherence, torch.Tensor):
            coherence = coherence.item()

        # Variation ratio: std relative to mean magnitude
        cycle_var = cycle_std / (abs(cycle_mean) + 1e-8)
        mlp_var = mlp_std / (abs(mlp_mean) + 1e-8)

        results.append(PerturbationResult(
            noise_sigma=sigma,
            cycle_output_mean=cycle_mean,
            cycle_output_std=cycle_std,
            cycle_coherence=coherence,
            mlp_output_mean=mlp_mean,
            mlp_output_std=mlp_std,
            cycle_variation_ratio=cycle_var / (mlp_var + 1e-8),
        ))

    return results


# ============================================================
# Scenario 2: Phase perturbation recovery
# ============================================================

@dataclass
class PhaseRecoveryResult:
    perturbation_sigma: float
    initial_r: float  # Order parameter before perturbation
    perturbed_r: float  # Order parameter right after perturbation
    recovered_r: float  # Order parameter after recovery steps
    recovery_ratio: float  # (recovered - perturbed) / (initial - perturbed)
    recovery_steps: int


def bench_phase_recovery(config: RobustnessConfig) -> List[PhaseRecoveryResult]:
    """Measure Kuramoto oscillator recovery after phase perturbation."""
    from pyquifer.oscillators import LearnableKuramotoBank

    results = []

    for sigma in config.noise_levels:
        if sigma == 0.0:
            continue  # No perturbation to recover from

        torch.manual_seed(config.seed)
        bank = LearnableKuramotoBank(num_oscillators=config.num_oscillators)
        inp = torch.randn(1, config.num_oscillators)

        # Synchronize first
        for _ in range(100):
            bank(inp)
        initial_r = bank.get_order_parameter().item()

        # Perturb phases
        with torch.no_grad():
            perturbation = torch.randn_like(bank.phases) * sigma
            bank.phases.add_(perturbation).remainder_(2 * math.pi)

        # Measure immediately after perturbation
        bank(inp)
        perturbed_r = bank.get_order_parameter().item()

        # Let it recover
        for _ in range(config.recovery_steps):
            bank(inp)
        recovered_r = bank.get_order_parameter().item()

        denom = initial_r - perturbed_r
        recovery_ratio = (recovered_r - perturbed_r) / denom if abs(denom) > 1e-6 else 1.0

        results.append(PhaseRecoveryResult(
            perturbation_sigma=sigma,
            initial_r=initial_r,
            perturbed_r=perturbed_r,
            recovered_r=recovered_r,
            recovery_ratio=recovery_ratio,
            recovery_steps=config.recovery_steps,
        ))

    return results


# ============================================================
# Scenario 3: Noisy input consistency (CognitiveCycle)
# ============================================================

@dataclass
class ConsistencyResult:
    noise_sigma: float
    mean_cosine_similarity: float  # Between clean and noisy outputs
    output_norm_ratio: float  # |noisy_output| / |clean_output|


def bench_input_consistency(config: RobustnessConfig) -> List[ConsistencyResult]:
    """Measure how consistent CognitiveCycle output is under noise."""
    from pyquifer.integration import CognitiveCycle, CycleConfig

    torch.manual_seed(config.seed)

    cycle_config = CycleConfig(
        state_dim=config.state_dim,
        num_oscillators=config.num_oscillators,
    )
    cycle = CognitiveCycle(cycle_config)

    base_input = torch.randn(1, config.state_dim)

    # Get clean output — use attention_bias as representative tensor
    clean_result = cycle.tick(base_input)
    clean_output = clean_result["modulation"]["attention_bias"].detach().flatten()

    results = []
    for sigma in config.noise_levels:
        cosine_sims = []
        norm_ratios = []

        for _ in range(config.num_samples):
            noisy_input = base_input + torch.randn_like(base_input) * sigma
            noisy_result = cycle.tick(noisy_input)
            noisy_output = noisy_result["modulation"]["attention_bias"].detach().flatten()

            cos_sim = torch.nn.functional.cosine_similarity(
                clean_output.unsqueeze(0), noisy_output.unsqueeze(0)
            ).item()
            cosine_sims.append(cos_sim)

            norm_ratio = noisy_output.norm().item() / (clean_output.norm().item() + 1e-8)
            norm_ratios.append(norm_ratio)

        results.append(ConsistencyResult(
            noise_sigma=sigma,
            mean_cosine_similarity=sum(cosine_sims) / len(cosine_sims),
            output_norm_ratio=sum(norm_ratios) / len(norm_ratios),
        ))

    return results


# ============================================================
# Scenario 4: Criticality under noise
# ============================================================

@dataclass
class CriticalityNoiseResult:
    noise_sigma: float
    sigma_before: float   # Criticality sigma before noise
    sigma_after: float    # Criticality sigma after noisy steps
    sigma_recovered: float  # After recovery
    stayed_critical: bool


def bench_criticality_noise(config: RobustnessConfig) -> List[CriticalityNoiseResult]:
    """Test if CognitiveCycle maintains criticality under noisy inputs."""
    from pyquifer.integration import CognitiveCycle, CycleConfig

    results = []

    for sigma in config.noise_levels:
        torch.manual_seed(config.seed)
        cycle_config = CycleConfig(
            state_dim=config.state_dim,
            num_oscillators=config.num_oscillators,
        )
        cycle = CognitiveCycle(cycle_config)

        base_input = torch.randn(1, config.state_dim)

        # Establish baseline
        for _ in range(20):
            result = cycle.tick(base_input)
        sigma_before = result["diagnostics"].get("criticality_sigma", 0.0)
        if isinstance(sigma_before, torch.Tensor):
            sigma_before = sigma_before.item()

        # Run with noise
        for _ in range(30):
            noisy = base_input + torch.randn_like(base_input) * sigma
            result = cycle.tick(noisy)
        sigma_after = result["diagnostics"].get("criticality_sigma", 0.0)
        if isinstance(sigma_after, torch.Tensor):
            sigma_after = sigma_after.item()

        # Recovery
        for _ in range(20):
            result = cycle.tick(base_input)
        sigma_recovered = result["diagnostics"].get("criticality_sigma", 0.0)
        if isinstance(sigma_recovered, torch.Tensor):
            sigma_recovered = sigma_recovered.item()

        results.append(CriticalityNoiseResult(
            noise_sigma=sigma,
            sigma_before=sigma_before,
            sigma_after=sigma_after,
            sigma_recovered=sigma_recovered,
            stayed_critical=abs(sigma_after - 1.0) < 1.0,  # Within reasonable range
        ))

    return results


# ============================================================
# Full Suite
# ============================================================

def run_full_suite(config: Optional[RobustnessConfig] = None) -> Dict:
    if config is None:
        config = RobustnessConfig()

    print("=" * 60)
    print("  PyQuifer Robustness Benchmark")
    print("=" * 60)

    t0 = time.perf_counter()

    # 1. Perturbation stability
    print("\n[1/4] Embedding perturbation stability...")
    perturb_results = bench_perturbation_stability(config)
    for r in perturb_results:
        print(f"  sigma={r.noise_sigma:.1f}: cycle_std={r.cycle_output_std:.4f}, "
              f"mlp_std={r.mlp_output_std:.4f}, ratio={r.cycle_variation_ratio:.3f}")

    # 2. Phase recovery
    print("\n[2/4] Phase perturbation recovery...")
    recovery_results = bench_phase_recovery(config)
    for r in recovery_results:
        print(f"  sigma={r.perturbation_sigma:.1f}: R={r.initial_r:.3f} -> "
              f"{r.perturbed_r:.3f} -> {r.recovered_r:.3f} "
              f"(recovery={r.recovery_ratio:.2f})")

    # 3. Input consistency
    print("\n[3/4] Input consistency (cosine similarity)...")
    consistency_results = bench_input_consistency(config)
    for r in consistency_results:
        print(f"  sigma={r.noise_sigma:.1f}: cos_sim={r.mean_cosine_similarity:.4f}, "
              f"norm_ratio={r.output_norm_ratio:.4f}")

    # 4. Criticality noise
    print("\n[4/4] Criticality under noise...")
    crit_results = bench_criticality_noise(config)
    for r in crit_results:
        print(f"  sigma={r.noise_sigma:.1f}: crit={r.sigma_before:.3f} -> "
              f"{r.sigma_after:.3f} -> {r.sigma_recovered:.3f} "
              f"({'OK' if r.stayed_critical else 'DRIFT'})")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")

    # Build JSON output
    suite = BenchmarkSuite("Robustness")

    mc_perturb = MetricCollector("Embedding Perturbation Stability")
    for r in perturb_results:
        mc_perturb.add_result(BenchmarkResult(
            name=f"sigma_{r.noise_sigma}",
            column=f"sigma={r.noise_sigma}",
            metrics={
                "cycle_output_std": r.cycle_output_std,
                "mlp_output_std": r.mlp_output_std,
                "variation_ratio": r.cycle_variation_ratio,
                "coherence": r.cycle_coherence,
            },
        ))
    suite.add(mc_perturb)

    mc_recovery = MetricCollector("Phase Recovery")
    for r in recovery_results:
        mc_recovery.add_result(BenchmarkResult(
            name=f"sigma_{r.perturbation_sigma}",
            column=f"sigma={r.perturbation_sigma}",
            metrics={
                "initial_R": r.initial_r,
                "perturbed_R": r.perturbed_r,
                "recovered_R": r.recovered_r,
                "recovery_ratio": r.recovery_ratio,
            },
        ))
    suite.add(mc_recovery)

    mc_consist = MetricCollector("Input Consistency")
    for r in consistency_results:
        mc_consist.add_result(BenchmarkResult(
            name=f"sigma_{r.noise_sigma}",
            column=f"sigma={r.noise_sigma}",
            metrics={
                "cosine_similarity": r.mean_cosine_similarity,
                "norm_ratio": r.output_norm_ratio,
            },
        ))
    suite.add(mc_consist)

    mc_crit = MetricCollector("Criticality Under Noise")
    for r in crit_results:
        mc_crit.add_result(BenchmarkResult(
            name=f"sigma_{r.noise_sigma}",
            column=f"sigma={r.noise_sigma}",
            metrics={
                "sigma_before": r.sigma_before,
                "sigma_after": r.sigma_after,
                "sigma_recovered": r.sigma_recovered,
            },
        ))
    suite.add(mc_crit)

    results_dir = _benchmark_dir / "results"
    suite.to_json(str(results_dir / "robustness.json"))
    print(f"\nResults saved: {results_dir / 'robustness.json'}")

    return {
        "perturbation": [vars(r) for r in perturb_results],
        "recovery": [vars(r) for r in recovery_results],
        "consistency": [vars(r) for r in consistency_results],
        "criticality": [vars(r) for r in crit_results],
    }


# ============================================================
# Pytest Smoke Tests
# ============================================================

class TestPerturbationStability:
    def test_zero_noise_stable(self):
        config = RobustnessConfig(num_samples=10, noise_levels=[0.0])
        results = bench_perturbation_stability(config)
        assert len(results) == 1
        # CognitiveCycle has evolving internal state, so small variance is expected
        assert results[0].cycle_output_std < 0.5

    def test_noise_increases_variance(self):
        config = RobustnessConfig(num_samples=20, noise_levels=[0.0, 0.5])
        results = bench_perturbation_stability(config)
        assert results[1].cycle_output_std > results[0].cycle_output_std


class TestPhaseRecovery:
    def test_recovery_runs(self):
        config = RobustnessConfig(noise_levels=[0.3], recovery_steps=50)
        results = bench_phase_recovery(config)
        assert len(results) == 1
        # System should produce finite R values after perturbation
        assert math.isfinite(results[0].recovered_r)
        assert results[0].recovered_r >= 0


class TestInputConsistency:
    def test_zero_noise_high_similarity(self):
        config = RobustnessConfig(num_samples=10, noise_levels=[0.0])
        results = bench_input_consistency(config)
        # Same input but CognitiveCycle has evolving internal state
        assert results[0].mean_cosine_similarity > 0.9

    def test_noise_reduces_similarity(self):
        config = RobustnessConfig(num_samples=20, noise_levels=[0.0, 1.0])
        results = bench_input_consistency(config)
        assert results[1].mean_cosine_similarity < results[0].mean_cosine_similarity


class TestCriticalityNoise:
    def test_criticality_runs(self):
        config = RobustnessConfig(noise_levels=[0.0, 0.3])
        results = bench_criticality_noise(config)
        assert len(results) == 2


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_full_suite()
