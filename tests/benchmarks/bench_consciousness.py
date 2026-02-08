"""Consciousness & Dynamics Metrics Benchmark.

Dual-mode:
  pytest:  python -m pytest tests/benchmarks/bench_consciousness.py -v --timeout=60
  CLI:     python tests/benchmarks/bench_consciousness.py

Scenarios:
  1. PCI measurement — perturb CognitiveCycle, measure Lempel-Ziv complexity
  2. Metastability tracking — run CognitiveCycle for N ticks, track var(R)
  3. Coherence-complexity-performance — vary coupling strength, correlate R/LZ with accuracy
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from harness import (
    BenchmarkSuite, MetricCollector, get_device, set_seed, timer,
)

# ---------------------------------------------------------------------------
# Lempel-Ziv Complexity
# ---------------------------------------------------------------------------

def lempel_ziv_complexity(binary_sequence: List[int]) -> float:
    """Compute Lempel-Ziv complexity of a binary sequence.
    Returns normalized complexity c(n) / (n / log2(n))."""
    n = len(binary_sequence)
    if n <= 1:
        return 0.0
    s = "".join(str(b) for b in binary_sequence)
    words = set()
    w = ""
    complexity = 0
    for c in s:
        w += c
        if w not in words:
            words.add(w)
            complexity += 1
            w = ""
    if w:
        complexity += 1
    # Normalize by n / log2(n)
    norm = n / max(math.log2(n), 1.0)
    return complexity / norm


def phase_to_binary(phases: torch.Tensor, threshold: float = math.pi) -> List[int]:
    """Convert phase time series to binary: 1 if phase > threshold, else 0."""
    return [1 if p > threshold else 0 for p in phases.view(-1).tolist()]


# ---------------------------------------------------------------------------
# Scenario 1: PCI Measurement
# ---------------------------------------------------------------------------

def bench_pci(num_ticks: int = 200, num_perturbations: int = 5,
              perturbation_strength: float = 0.5) -> MetricCollector:
    """Measure PCI using CognitiveCycle dynamics."""
    device = get_device()
    set_seed(42)
    mc = MetricCollector("PCI Measurement")

    from pyquifer.integration import CognitiveCycle, CycleConfig

    cfg = CycleConfig.small()
    cycle = CognitiveCycle(cfg).to(device)
    sensory = torch.randn(cfg.state_dim, device=device) * 0.5

    # Run baseline to settle dynamics
    for _ in range(50):
        cycle.tick(sensory, reward=0.0)
    R_baseline = cycle.oscillators.get_order_parameter().item()

    # Measure PCI via perturbation + Lempel-Ziv complexity
    pci_values = []
    for p_idx in range(num_perturbations):
        set_seed(42 + p_idx)
        # Save state
        saved_phases = cycle.oscillators.phases.clone()

        # Perturb
        perturbation = torch.randn_like(cycle.oscillators.phases) * perturbation_strength
        with torch.no_grad():
            cycle.oscillators.phases.add_(perturbation)

        # Record phase response
        phase_response = []
        for _ in range(num_ticks):
            cycle.tick(sensory, reward=0.0)
            phase_response.append(cycle.oscillators.phases.clone())

        # Compute LZ complexity of phase response
        all_phases = torch.stack(phase_response)  # (ticks, n_osc)
        binary = phase_to_binary(all_phases)
        lz = lempel_ziv_complexity(binary)
        pci_values.append(lz)

        # Restore phases
        with torch.no_grad():
            cycle.oscillators.phases.copy_(saved_phases)

    mean_pci = sum(pci_values) / len(pci_values) if pci_values else 0.0
    std_pci = (sum((v - mean_pci) ** 2 for v in pci_values) / max(len(pci_values) - 1, 1)) ** 0.5

    mc.record("C_pyquifer", "pci_mean", round(mean_pci, 4))
    mc.record("C_pyquifer", "pci_std", round(std_pci, 4))
    mc.record("C_pyquifer", "pci_target_min", 0.31,
              {"note": "PCI > 0.31 indicates conscious processing"})
    mc.record("C_pyquifer", "R_baseline", round(R_baseline, 4))

    # Also measure using the built-in PerturbationalComplexity module
    try:
        from pyquifer.consciousness import PerturbationalComplexity
        pci_mod = PerturbationalComplexity(
            state_dim=cfg.num_oscillators,
            num_perturbations=num_perturbations,
            perturbation_strength=perturbation_strength,
            response_window=min(num_ticks, 20),
        ).to(device)

        def dynamics_fn(state):
            with torch.no_grad():
                cycle.oscillators.phases.copy_(state)
                cycle.oscillators(steps=1)
                return cycle.oscillators.phases.clone()

        initial = cycle.oscillators.phases.clone()
        pci_result = pci_mod(dynamics_fn, initial)
        mc.record("C_pyquifer", "pci_builtin",
                  round(pci_result['pci'].item(), 4))
    except Exception:
        pass

    return mc


# ---------------------------------------------------------------------------
# Scenario 2: Metastability Tracking
# ---------------------------------------------------------------------------

def bench_metastability(num_ticks: int = 500) -> MetricCollector:
    """Track var(R) over time as metastability index."""
    device = get_device()
    set_seed(42)
    mc = MetricCollector("Metastability Tracking")

    from pyquifer.integration import CognitiveCycle, CycleConfig

    cfg = CycleConfig.small()
    cycle = CognitiveCycle(cfg).to(device)
    sensory = torch.randn(cfg.state_dim, device=device) * 0.5

    R_values = []
    for t in range(num_ticks):
        cycle.tick(sensory, reward=0.0)
        R = cycle.oscillators.get_order_parameter().item()
        R_values.append(R)

    R_arr = np.array(R_values)
    mc.record("C_pyquifer", "R_mean", round(float(R_arr.mean()), 4))
    mc.record("C_pyquifer", "R_std", round(float(R_arr.std()), 4))
    mc.record("C_pyquifer", "R_min", round(float(R_arr.min()), 4))
    mc.record("C_pyquifer", "R_max", round(float(R_arr.max()), 4))
    mc.record("C_pyquifer", "metastability_index", round(float(R_arr.var()), 6))

    # Phase segments: compute LZ complexity of R time series
    binary_R = [1 if r > R_arr.mean() else 0 for r in R_values]
    lz_R = lempel_ziv_complexity(binary_R)
    mc.record("C_pyquifer", "R_lz_complexity", round(lz_R, 4))

    # Target: medium coherence (0.3 < R_mean < 0.7) + high complexity
    mc.record("C_pyquifer", "target_R_range", 0.5,
              {"note": "Ideal: 0.3 < R_mean < 0.7"})

    return mc


# ---------------------------------------------------------------------------
# Scenario 3: Coherence-Complexity-Performance Correlation
# ---------------------------------------------------------------------------

def bench_coherence_complexity(coupling_values: List[float] = None,
                               num_ticks: int = 200) -> MetricCollector:
    """Vary coupling strength, measure R and LZ complexity."""
    if coupling_values is None:
        coupling_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    device = get_device()
    mc = MetricCollector("Coherence-Complexity Sweep")

    from pyquifer.oscillators import LearnableKuramotoBank

    for K in coupling_values:
        set_seed(42)
        bank = LearnableKuramotoBank(
            num_oscillators=32, dt=0.01, topology='global',
        ).to(device)

        with torch.no_grad():
            bank.coupling_strength.fill_(K)

        ext = torch.randn(32, device=device) * 0.1

        R_values = []
        all_phases = []
        for _ in range(num_ticks):
            bank(external_input=ext, steps=1)
            R = bank.get_order_parameter().item()
            R_values.append(R)
            all_phases.append(bank.phases.clone())

        R_arr = np.array(R_values)
        phase_tensor = torch.stack(all_phases)
        binary = phase_to_binary(phase_tensor)
        lz = lempel_ziv_complexity(binary)

        mc.record("C_pyquifer", f"K={K:.1f}_R_mean", round(float(R_arr.mean()), 4))
        mc.record("C_pyquifer", f"K={K:.1f}_R_std", round(float(R_arr.std()), 4))
        mc.record("C_pyquifer", f"K={K:.1f}_LZ", round(lz, 4))
        mc.record("C_pyquifer", f"K={K:.1f}_metastability",
                  round(float(R_arr.var()), 6))

    return mc


# ---------------------------------------------------------------------------
# Full Suite
# ---------------------------------------------------------------------------

def run_full_suite():
    print("=" * 60)
    print("Consciousness & Dynamics Benchmark — Full Suite")
    print("=" * 60)

    suite = BenchmarkSuite("Consciousness & Dynamics Benchmarks")

    print("\n--- PCI Measurement ---")
    mc_pci = bench_pci(num_ticks=200, num_perturbations=10)
    suite.add(mc_pci)
    print(mc_pci.to_markdown_table())

    print("\n--- Metastability ---")
    mc_meta = bench_metastability(num_ticks=1000)
    suite.add(mc_meta)
    print(mc_meta.to_markdown_table())

    print("\n--- Coherence-Complexity Sweep ---")
    mc_cc = bench_coherence_complexity(num_ticks=500)
    suite.add(mc_cc)
    print(mc_cc.to_markdown_table())

    json_path = str(Path(__file__).parent / "results" / "consciousness.json")
    suite.to_json(json_path)
    print(f"\nResults saved to {json_path}")


# ---------------------------------------------------------------------------
# Pytest Classes
# ---------------------------------------------------------------------------

class TestPCI:
    def test_pci_runs(self):
        mc = bench_pci(num_ticks=20, num_perturbations=2,
                       perturbation_strength=0.3)
        assert len(mc.results) >= 1
        # Check pci_mean was recorded
        for r in mc.results:
            assert "pci_mean" in r.metrics

    def test_lempel_ziv(self):
        # Simple test: all same = low complexity
        low = lempel_ziv_complexity([0] * 100)
        # Alternating = higher complexity
        high = lempel_ziv_complexity([0, 1] * 50)
        assert high > low


class TestMetastability:
    def test_metastability_runs(self):
        mc = bench_metastability(num_ticks=20)
        assert len(mc.results) >= 1
        for r in mc.results:
            assert "R_mean" in r.metrics
            assert "metastability_index" in r.metrics


class TestCoherenceComplexity:
    def test_sweep_runs(self):
        mc = bench_coherence_complexity(coupling_values=[0.5, 2.0],
                                        num_ticks=20)
        assert len(mc.results) >= 1


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_full_suite()
