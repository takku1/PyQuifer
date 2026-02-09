"""CognitiveCycle System Benchmark — Dedicated Cycle-Level Evaluation.

Dual-mode:
  pytest:  python -m pytest tests/benchmarks/bench_cycle.py -v --timeout=120
  CLI:     python tests/benchmarks/bench_cycle.py

Scenarios:
  1. Steady-state throughput — ticks/sec, latency percentiles, R(t) mean
  2. Config flag sweep — per-flag overhead measurement
  3. Scaling sweep — oscillator count vs tick latency
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from harness import (
    BenchmarkSuite, MemoryTracker, MetricCollector, get_device, set_seed,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def measure_tick_throughput(cycle, sensory, warmup: int = 10,
                           steps: int = 100) -> Dict[str, float]:
    """Measure tick throughput and latency percentiles."""
    for _ in range(warmup):
        cycle.tick(sensory, reward=0.0)

    latencies = []
    for _ in range(steps):
        t0 = time.perf_counter()
        cycle.tick(sensory, reward=0.0)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    total_s = sum(latencies) / 1000
    return {
        "ticks_per_sec": round(steps / total_s, 1),
        "latency_p50_ms": round(latencies[len(latencies) // 2], 3),
        "latency_p95_ms": round(latencies[int(len(latencies) * 0.95)], 3),
        "latency_p99_ms": round(latencies[int(len(latencies) * 0.99)], 3),
    }


# ---------------------------------------------------------------------------
# All 16 integration flags
# ---------------------------------------------------------------------------

ALL_FLAGS = [
    "use_stp", "use_mean_field", "use_stuart_landau", "use_koopman",
    "use_neural_mass", "use_attractor_stability", "use_hypothesis_arena",
    "use_evidence_aggregator", "use_no_progress", "use_phase_cache",
    "use_ep_training", "use_oscillation_gated_plasticity", "use_three_factor",
    "use_oscillatory_predictive", "use_sleep_consolidation", "use_dendritic_credit",
]


# ---------------------------------------------------------------------------
# Scenario 1: Steady-State Throughput
# ---------------------------------------------------------------------------

def bench_steady_state(ticks: int = 200, warmup: int = 20) -> MetricCollector:
    """Measure steady-state CognitiveCycle performance."""
    device = get_device()
    mc = MetricCollector("CognitiveCycle Steady-State")

    from pyquifer.integration import CognitiveCycle, CycleConfig

    # Column A: Published baselines
    mc.record("A_published", "ticks_per_sec", 10.0,
              {"source": "Cognitive cycle 100-300ms (3-10 Hz), Baars (2005)"})
    mc.record("A_published", "latency_p50_ms", 200.0,
              {"source": "200ms typical cognitive cycle, Dehaene & Naccache (2001)"})

    # Column B: Vanilla GRU baseline (same state_dim, same tick count)
    set_seed(42)
    cfg = CycleConfig.small()
    gru = torch.nn.GRUCell(cfg.state_dim, cfg.state_dim).to(device)
    sensory = torch.randn(cfg.state_dim, device=device)
    h_b = torch.zeros(1, cfg.state_dim, device=device)

    for _ in range(warmup):
        h_b = gru(sensory.unsqueeze(0), h_b)
    lat_b = []
    for _ in range(ticks):
        t0 = time.perf_counter()
        h_b = gru(sensory.unsqueeze(0), h_b)
        lat_b.append((time.perf_counter() - t0) * 1000)
    lat_b.sort()
    total_b = sum(lat_b) / 1000
    mc.record("B_pytorch", "ticks_per_sec", round(ticks / total_b, 1))
    mc.record("B_pytorch", "latency_p50_ms", round(lat_b[len(lat_b) // 2], 3))

    # Column C: PyQuifer CognitiveCycle
    set_seed(42)
    cycle = CognitiveCycle(cfg).to(device)
    sensory = torch.randn(cfg.state_dim, device=device)

    with MemoryTracker() as mem:
        perf = measure_tick_throughput(cycle, sensory, warmup=warmup, steps=ticks)

    # Get R(t) mean over final portion
    R_values = []
    for _ in range(min(ticks, 50)):
        cycle.tick(sensory, reward=0.0)
        R = cycle.oscillators.get_order_parameter().item()
        R_values.append(R)

    R_mean = sum(R_values) / len(R_values) if R_values else 0.0

    mc.record("C_pyquifer", "ticks_per_sec", perf["ticks_per_sec"])
    mc.record("C_pyquifer", "latency_p50_ms", perf["latency_p50_ms"])
    mc.record("C_pyquifer", "latency_p95_ms", perf["latency_p95_ms"])
    mc.record("C_pyquifer", "latency_p99_ms", perf["latency_p99_ms"])
    mc.record("C_pyquifer", "peak_memory_mb", mem.peak_mb)
    mc.record("C_pyquifer", "R_mean", round(R_mean, 4))

    return mc


# ---------------------------------------------------------------------------
# Scenario 2: Config Flag Sweep
# ---------------------------------------------------------------------------

def bench_flag_sweep(ticks: int = 50, warmup: int = 10) -> MetricCollector:
    """Measure per-flag overhead: baseline + each flag in isolation + all combined."""
    device = get_device()
    mc = MetricCollector("CognitiveCycle Flag Sweep")

    from pyquifer.integration import CognitiveCycle, CycleConfig

    # Column A: Published baselines for flag overhead
    mc.record("A_published", "per_flag_overhead_ms", 5.0,
              {"source": "Per-module overhead 1-10ms typical, modular architectures"})
    mc.record("A_published", "all_flags_overhead_ms", 50.0,
              {"source": "16 modules × ~3ms avg = ~50ms cumulative overhead"})

    # Column B: GRU baseline (no flags, just a recurrent step)
    set_seed(42)
    cfg_base = CycleConfig.small()
    gru_f = torch.nn.GRUCell(cfg_base.state_dim, cfg_base.state_dim).to(device)
    sensory_gru = torch.randn(cfg_base.state_dim, device=device)
    h_gru = torch.zeros(1, cfg_base.state_dim, device=device)
    for _ in range(warmup):
        h_gru = gru_f(sensory_gru.unsqueeze(0), h_gru)
    lat_gru = []
    for _ in range(ticks):
        t0 = time.perf_counter()
        h_gru = gru_f(sensory_gru.unsqueeze(0), h_gru)
        lat_gru.append((time.perf_counter() - t0) * 1000)
    lat_gru.sort()
    total_gru = sum(lat_gru) / 1000
    mc.record("B_pytorch", "baseline_ticks_per_sec", round(ticks / total_gru, 1))
    mc.record("B_pytorch", "baseline_p50_ms", round(lat_gru[len(lat_gru) // 2], 3))

    # Baseline (no flags)
    set_seed(42)
    cycle_base = CognitiveCycle(cfg_base).to(device)
    sensory = torch.randn(cfg_base.state_dim, device=device)
    perf_base = measure_tick_throughput(cycle_base, sensory, warmup=warmup, steps=ticks)
    baseline_p50 = perf_base["latency_p50_ms"]

    mc.record("C_pyquifer", "baseline_ticks_per_sec", perf_base["ticks_per_sec"])
    mc.record("C_pyquifer", "baseline_p50_ms", baseline_p50)

    # Each flag in isolation
    for flag_name in ALL_FLAGS:
        set_seed(42)
        cfg = CycleConfig.small()
        setattr(cfg, flag_name, True)
        try:
            cycle = CognitiveCycle(cfg).to(device)
            sensory_f = torch.randn(cfg.state_dim, device=device)
            perf = measure_tick_throughput(cycle, sensory_f, warmup=warmup, steps=ticks)
            overhead = perf["latency_p50_ms"] - baseline_p50
            mc.record("C_pyquifer", f"{flag_name}_ticks_per_sec", perf["ticks_per_sec"])
            mc.record("C_pyquifer", f"{flag_name}_p50_ms", perf["latency_p50_ms"])
            mc.record("C_pyquifer", f"{flag_name}_overhead_ms", round(overhead, 3))
        except Exception as e:
            mc.record("C_pyquifer", f"{flag_name}_ticks_per_sec", 0.0,
                      {"error": str(e)[:200]})

    # All flags combined
    set_seed(42)
    cfg_all = CycleConfig.small()
    for flag_name in ALL_FLAGS:
        setattr(cfg_all, flag_name, True)
    cycle_all = CognitiveCycle(cfg_all).to(device)
    sensory_all = torch.randn(cfg_all.state_dim, device=device)
    perf_all = measure_tick_throughput(cycle_all, sensory_all, warmup=warmup, steps=ticks)
    mc.record("C_pyquifer", "all_flags_ticks_per_sec", perf_all["ticks_per_sec"])
    mc.record("C_pyquifer", "all_flags_p50_ms", perf_all["latency_p50_ms"])
    mc.record("C_pyquifer", "all_flags_overhead_ms",
              round(perf_all["latency_p50_ms"] - baseline_p50, 3))

    return mc


# ---------------------------------------------------------------------------
# Scenario 3: Scaling Sweep
# ---------------------------------------------------------------------------

def bench_scaling(osc_counts: List[int] = None, ticks: int = 50,
                  warmup: int = 10) -> MetricCollector:
    """Sweep oscillator counts vs tick latency."""
    if osc_counts is None:
        osc_counts = [16, 32, 64, 128, 256, 512, 1024]

    device = get_device()
    mc = MetricCollector("CognitiveCycle Scaling Sweep")

    from pyquifer.integration import CognitiveCycle, CycleConfig

    # Column A: Published baselines
    mc.record("A_published", "scaling_law", 2.0,
              {"source": "O(N²) coupling complexity, Strogatz (2000) Physica D"})

    for n_osc in osc_counts:
        # --- Column B: GRU at matching hidden dim ---
        set_seed(42)
        cfg = CycleConfig.small()
        gru_s = torch.nn.GRUCell(cfg.state_dim, cfg.state_dim).to(device)
        s_in = torch.randn(cfg.state_dim, device=device)
        h_s = torch.zeros(1, cfg.state_dim, device=device)
        for _ in range(warmup):
            h_s = gru_s(s_in.unsqueeze(0), h_s)
        lat_s = []
        for _ in range(ticks):
            t0 = time.perf_counter()
            h_s = gru_s(s_in.unsqueeze(0), h_s)
            lat_s.append((time.perf_counter() - t0) * 1000)
        lat_s.sort()
        total_s = sum(lat_s) / 1000
        mc.record("B_pytorch", f"N={n_osc}_ticks_per_sec", round(ticks / total_s, 1))
        mc.record("B_pytorch", f"N={n_osc}_p50_ms", round(lat_s[len(lat_s) // 2], 3))

        # --- Column C: PyQuifer CognitiveCycle ---
        set_seed(42)
        cfg.num_oscillators = n_osc
        cfg.hierarchy_dims = [cfg.state_dim, cfg.state_dim // 2, cfg.state_dim // 4]

        try:
            cycle = CognitiveCycle(cfg).to(device)
            sensory = torch.randn(cfg.state_dim, device=device)

            with MemoryTracker() as mem:
                perf = measure_tick_throughput(cycle, sensory, warmup=warmup, steps=ticks)

            mc.record("C_pyquifer", f"N={n_osc}_ticks_per_sec", perf["ticks_per_sec"])
            mc.record("C_pyquifer", f"N={n_osc}_p50_ms", perf["latency_p50_ms"])
            mc.record("C_pyquifer", f"N={n_osc}_p95_ms", perf["latency_p95_ms"])
            mc.record("C_pyquifer", f"N={n_osc}_peak_mb", mem.peak_mb)
        except Exception as e:
            mc.record("C_pyquifer", f"N={n_osc}_ticks_per_sec", 0.0,
                      {"error": str(e)[:200]})

    return mc


# ---------------------------------------------------------------------------
# Full Suite
# ---------------------------------------------------------------------------

def run_full_suite():
    print("=" * 60)
    print("CognitiveCycle System Benchmark — Full Suite")
    print("=" * 60)

    suite = BenchmarkSuite("CognitiveCycle Benchmarks")

    for name, fn in [
        ("Steady-State", bench_steady_state),
        ("Flag Sweep", bench_flag_sweep),
        ("Scaling", bench_scaling),
    ]:
        print(f"\n--- {name} ---")
        mc = fn()
        suite.add(mc)
        print(mc.to_markdown_table())

    json_path = str(Path(__file__).parent / "results" / "cycle.json")
    suite.to_json(json_path)
    print(f"\nResults saved to {json_path}")


# ---------------------------------------------------------------------------
# Pytest Classes (smoke tests)
# ---------------------------------------------------------------------------

class TestSteadyState:
    def test_steady_state_runs(self):
        mc = bench_steady_state(ticks=10, warmup=3)
        assert len(mc.results) >= 1
        metric_keys = set()
        for r in mc.results:
            metric_keys.update(r.metrics.keys())
        assert "ticks_per_sec" in metric_keys
        assert "R_mean" in metric_keys

    def test_throughput_positive(self):
        mc = bench_steady_state(ticks=5, warmup=2)
        for r in mc.results:
            if "ticks_per_sec" in r.metrics:
                assert r.metrics["ticks_per_sec"] > 0


class TestFlagSweep:
    def test_flag_sweep_runs(self):
        mc = bench_flag_sweep(ticks=5, warmup=2)
        assert len(mc.results) >= 1
        metric_keys = set()
        for r in mc.results:
            metric_keys.update(r.metrics.keys())
        assert "baseline_ticks_per_sec" in metric_keys
        assert "all_flags_ticks_per_sec" in metric_keys

    def test_flag_sweep_has_individual_flags(self):
        mc = bench_flag_sweep(ticks=3, warmup=1)
        metric_keys = set()
        for r in mc.results:
            metric_keys.update(r.metrics.keys())
        # At least some individual flag measurements
        assert any("use_stp" in k for k in metric_keys)


class TestScaling:
    def test_scaling_sweep_runs(self):
        mc = bench_scaling(osc_counts=[8, 16, 32], ticks=5, warmup=2)
        assert len(mc.results) >= 1
        metric_keys = set()
        for r in mc.results:
            metric_keys.update(r.metrics.keys())
        assert any("N=8" in k for k in metric_keys)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_full_suite()
