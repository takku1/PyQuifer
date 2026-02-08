"""Benchmark #5: Efficiency — dtype, compile, scaling sweeps.

Measures throughput, memory, and numerical accuracy of PyQuifer modules
across different configurations:
  1. Dtype sweep: fp32 vs bf16 vs fp16
  2. torch.compile: eager vs compiled
  3. Oscillator count scaling: 16 → 512
  4. Batch size scaling: 1 → 64

Dual-mode: `python bench_efficiency.py` (full report) or
           `pytest bench_efficiency.py -v` (smoke tests)

References:
- PyTorch AMP documentation
- torch.compile / inductor backend
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
class EfficiencyConfig:
    seed: int = 42
    warmup_steps: int = 10
    bench_steps: int = 100
    state_dim: int = 64
    # Sweep ranges
    dtypes: List[str] = field(default_factory=lambda: ["fp32", "bf16", "fp16"])
    oscillator_counts: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 16, 64])


DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


# ============================================================
# Utilities
# ============================================================

def _make_kuramoto(num_osc: int, dtype: torch.dtype = torch.float32):
    from pyquifer.oscillators import LearnableKuramotoBank
    bank = LearnableKuramotoBank(num_oscillators=num_osc)
    return bank.to(dtype=dtype)


def _make_cycle(num_osc: int = 32, state_dim: int = 64, dtype: torch.dtype = torch.float32):
    from pyquifer.integration import CognitiveCycle, CycleConfig
    config = CycleConfig(
        state_dim=state_dim,
        num_oscillators=num_osc,
    )
    cycle = CognitiveCycle(config)
    return cycle.to(dtype=dtype)


def _measure_throughput(fn, steps: int, warmup: int = 10) -> Dict[str, float]:
    """Run fn() for warmup + steps, return steps/sec and total ms."""
    for _ in range(warmup):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    return {
        "steps_per_sec": steps / elapsed if elapsed > 0 else 0,
        "total_ms": elapsed * 1000,
        "ms_per_step": (elapsed / steps) * 1000 if steps > 0 else 0,
    }


# ============================================================
# Scenario 1: Dtype sweep on Kuramoto
# ============================================================

@dataclass
class DtypeResult:
    dtype: str
    steps_per_sec: float
    ms_per_step: float
    peak_memory_mb: float
    r_mean: float  # Mean order parameter
    max_phase_error: float  # vs fp32 reference


def bench_dtype_sweep(config: EfficiencyConfig) -> List[DtypeResult]:
    """Sweep fp32/bf16/fp16 on LearnableKuramotoBank."""
    torch.manual_seed(config.seed)
    results = []

    # First get fp32 reference phases
    num_osc = 64
    ref_bank = _make_kuramoto(num_osc, torch.float32)
    ref_input = torch.randn(1, num_osc)
    _ = ref_bank(ref_input)
    ref_phases = ref_bank.phases.clone()

    for dtype_name in config.dtypes:
        dtype = DTYPE_MAP[dtype_name]

        torch.manual_seed(config.seed)
        bank = _make_kuramoto(num_osc, dtype)
        inp = torch.randn(1, num_osc, dtype=dtype)

        try:
            with MemoryTracker() as mem:
                perf = _measure_throughput(
                    lambda: bank(inp),
                    steps=config.bench_steps,
                    warmup=config.warmup_steps,
                )

            # Measure accuracy vs fp32
            _ = bank(inp)
            current_phases = bank.phases.float()
            r_val = bank.get_order_parameter().item()
            max_error = (current_phases - ref_phases).abs().max().item() if ref_phases.shape == current_phases.shape else float('nan')

            results.append(DtypeResult(
                dtype=dtype_name,
                steps_per_sec=perf["steps_per_sec"],
                ms_per_step=perf["ms_per_step"],
                peak_memory_mb=mem.peak_mb,
                r_mean=r_val,
                max_phase_error=max_error,
            ))
        except RuntimeError as e:
            # fp16 ComplexHalf not supported on CPU
            results.append(DtypeResult(
                dtype=f"{dtype_name} (unsupported)",
                steps_per_sec=0.0,
                ms_per_step=0.0,
                peak_memory_mb=0.0,
                r_mean=0.0,
                max_phase_error=float('nan'),
            ))

    return results


# ============================================================
# Scenario 2: torch.compile sweep
# ============================================================

@dataclass
class CompileResult:
    mode: str  # "eager" or "compiled"
    steps_per_sec: float
    ms_per_step: float
    compile_time_ms: float
    output_matches: bool


def bench_compile_sweep(config: EfficiencyConfig) -> List[CompileResult]:
    """Compare eager vs torch.compile on Kuramoto forward."""
    results = []

    # Eager
    torch.manual_seed(config.seed)
    num_osc = 64
    bank = _make_kuramoto(num_osc)
    inp = torch.randn(1, num_osc)

    perf_eager = _measure_throughput(
        lambda: bank(inp),
        steps=config.bench_steps,
        warmup=config.warmup_steps,
    )
    _ = bank(inp)
    eager_r = bank.get_order_parameter().item()

    results.append(CompileResult(
        mode="eager",
        steps_per_sec=perf_eager["steps_per_sec"],
        ms_per_step=perf_eager["ms_per_step"],
        compile_time_ms=0.0,
        output_matches=True,
    ))

    # Compiled (if available)
    try:
        torch.manual_seed(config.seed)
        bank2 = _make_kuramoto(num_osc)
        t0 = time.perf_counter()
        compiled_forward = torch.compile(bank2)
        # Warm up the compiled version
        _ = compiled_forward(inp)
        compile_time = (time.perf_counter() - t0) * 1000

        perf_compiled = _measure_throughput(
            lambda: compiled_forward(inp),
            steps=config.bench_steps,
            warmup=config.warmup_steps,
        )
        _ = compiled_forward(inp)
        compiled_r = bank2.get_order_parameter().item()

        # Check output matches
        matches = abs(eager_r - compiled_r) < 0.1

        results.append(CompileResult(
            mode="compiled",
            steps_per_sec=perf_compiled["steps_per_sec"],
            ms_per_step=perf_compiled["ms_per_step"],
            compile_time_ms=compile_time,
            output_matches=matches,
        ))
    except Exception as e:
        results.append(CompileResult(
            mode="compiled (failed)",
            steps_per_sec=0.0,
            ms_per_step=0.0,
            compile_time_ms=0.0,
            output_matches=False,
        ))

    return results


# ============================================================
# Scenario 3: Oscillator count scaling
# ============================================================

@dataclass
class ScalingResult:
    num_oscillators: int
    steps_per_sec: float
    ms_per_step: float
    peak_memory_mb: float
    r_mean: float


def bench_oscillator_scaling(config: EfficiencyConfig) -> List[ScalingResult]:
    """Measure throughput as oscillator count increases."""
    results = []

    for n_osc in config.oscillator_counts:
        torch.manual_seed(config.seed)
        bank = _make_kuramoto(n_osc)
        inp = torch.randn(1, n_osc)  # Input must match num_oscillators

        with MemoryTracker() as mem:
            perf = _measure_throughput(
                lambda: bank(inp),
                steps=config.bench_steps,
                warmup=config.warmup_steps,
            )

        _ = bank(inp)
        r_val = bank.get_order_parameter().item()
        results.append(ScalingResult(
            num_oscillators=n_osc,
            steps_per_sec=perf["steps_per_sec"],
            ms_per_step=perf["ms_per_step"],
            peak_memory_mb=mem.peak_mb,
            r_mean=r_val,
        ))

    return results


# ============================================================
# Scenario 4: Batch size scaling on CognitiveCycle
# ============================================================

@dataclass
class BatchResult:
    batch_size: int
    steps_per_sec: float
    ms_per_step: float
    samples_per_sec: float
    peak_memory_mb: float


def bench_batch_scaling(config: EfficiencyConfig) -> List[BatchResult]:
    """Measure CognitiveCycle throughput vs batch size."""
    results = []

    for bs in config.batch_sizes:
        torch.manual_seed(config.seed)
        cycle = _make_cycle(num_osc=32, state_dim=config.state_dim)
        inp = torch.randn(bs, config.state_dim)

        with MemoryTracker() as mem:
            perf = _measure_throughput(
                lambda: cycle.tick(inp),
                steps=min(config.bench_steps, 50),  # CognitiveCycle is slower
                warmup=min(config.warmup_steps, 5),
            )

        results.append(BatchResult(
            batch_size=bs,
            steps_per_sec=perf["steps_per_sec"],
            ms_per_step=perf["ms_per_step"],
            samples_per_sec=perf["steps_per_sec"] * bs,
            peak_memory_mb=mem.peak_mb,
        ))

    return results


# ============================================================
# Full Suite
# ============================================================

def run_full_suite(config: Optional[EfficiencyConfig] = None) -> Dict:
    """Run all efficiency benchmarks, return results dict."""
    if config is None:
        config = EfficiencyConfig()

    print("=" * 60)
    print("  PyQuifer Efficiency Benchmark")
    print("=" * 60)

    t0 = time.perf_counter()

    # 1. Dtype sweep
    print("\n[1/4] Dtype sweep...")
    dtype_results = bench_dtype_sweep(config)
    for r in dtype_results:
        print(f"  {r.dtype:6s}: {r.steps_per_sec:8.1f} steps/s, "
              f"{r.ms_per_step:6.2f} ms/step, "
              f"phase_err={r.max_phase_error:.6f}")

    # 2. Compile sweep
    print("\n[2/4] torch.compile sweep...")
    compile_results = bench_compile_sweep(config)
    for r in compile_results:
        print(f"  {r.mode:20s}: {r.steps_per_sec:8.1f} steps/s, "
              f"{r.ms_per_step:6.2f} ms/step")

    # 3. Oscillator scaling
    print("\n[3/4] Oscillator count scaling...")
    scaling_results = bench_oscillator_scaling(config)
    for r in scaling_results:
        print(f"  N={r.num_oscillators:4d}: {r.steps_per_sec:8.1f} steps/s, "
              f"{r.ms_per_step:6.2f} ms/step, "
              f"R={r.r_mean:.4f}")

    # 4. Batch scaling
    print("\n[4/4] Batch size scaling...")
    batch_results = bench_batch_scaling(config)
    for r in batch_results:
        print(f"  BS={r.batch_size:4d}: {r.steps_per_sec:8.1f} ticks/s, "
              f"{r.samples_per_sec:8.1f} samples/s, "
              f"mem={r.peak_memory_mb:.1f} MB")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")

    # Build JSON output
    suite = BenchmarkSuite("Efficiency")

    # Dtype scenario
    mc_dtype = MetricCollector("Dtype Sweep (Kuramoto N=64)")
    for r in dtype_results:
        mc_dtype.add_result(BenchmarkResult(
            name=f"dtype_{r.dtype}",
            column=f"dtype_{r.dtype}",
            metrics={
                "steps_per_sec": r.steps_per_sec,
                "ms_per_step": r.ms_per_step,
                "peak_memory_mb": r.peak_memory_mb,
                "max_phase_error": r.max_phase_error,
            },
        ))
    suite.add(mc_dtype)

    # Compile scenario
    mc_compile = MetricCollector("torch.compile Comparison")
    for r in compile_results:
        mc_compile.add_result(BenchmarkResult(
            name=f"compile_{r.mode}",
            column=r.mode,
            metrics={
                "steps_per_sec": r.steps_per_sec,
                "ms_per_step": r.ms_per_step,
                "compile_time_ms": r.compile_time_ms,
            },
        ))
    suite.add(mc_compile)

    # Scaling scenario
    mc_scale = MetricCollector("Oscillator Count Scaling")
    for r in scaling_results:
        mc_scale.add_result(BenchmarkResult(
            name=f"osc_N{r.num_oscillators}",
            column=f"N={r.num_oscillators}",
            metrics={
                "steps_per_sec": r.steps_per_sec,
                "ms_per_step": r.ms_per_step,
                "peak_memory_mb": r.peak_memory_mb,
            },
        ))
    suite.add(mc_scale)

    # Batch scenario
    mc_batch = MetricCollector("Batch Size Scaling (CognitiveCycle)")
    for r in batch_results:
        mc_batch.add_result(BenchmarkResult(
            name=f"batch_{r.batch_size}",
            column=f"BS={r.batch_size}",
            metrics={
                "steps_per_sec": r.steps_per_sec,
                "samples_per_sec": r.samples_per_sec,
                "ms_per_step": r.ms_per_step,
                "peak_memory_mb": r.peak_memory_mb,
            },
        ))
    suite.add(mc_batch)

    results_dir = _benchmark_dir / "results"
    suite.to_json(str(results_dir / "efficiency.json"))
    print(f"\nResults saved: {results_dir / 'efficiency.json'}")

    return {
        "dtype": [vars(r) for r in dtype_results],
        "compile": [vars(r) for r in compile_results],
        "scaling": [vars(r) for r in scaling_results],
        "batch": [vars(r) for r in batch_results],
    }


# ============================================================
# Pytest Smoke Tests
# ============================================================

class TestDtypeSweep:
    def test_fp32_runs(self):
        config = EfficiencyConfig(bench_steps=5, warmup_steps=2)
        config.dtypes = ["fp32"]
        results = bench_dtype_sweep(config)
        assert len(results) == 1
        assert results[0].steps_per_sec > 0

    def test_all_dtypes_produce_output(self):
        config = EfficiencyConfig(bench_steps=5, warmup_steps=2)
        results = bench_dtype_sweep(config)
        # fp16 may fail on CPU (ComplexHalf not implemented)
        assert len(results) >= 2
        supported = [r for r in results if "unsupported" not in r.dtype]
        assert len(supported) >= 2  # At least fp32 + bf16
        for r in supported:
            assert r.steps_per_sec > 0
            assert r.ms_per_step > 0


class TestCompileSweep:
    def test_eager_runs(self):
        config = EfficiencyConfig(bench_steps=5, warmup_steps=2)
        results = bench_compile_sweep(config)
        assert len(results) >= 1
        assert results[0].mode == "eager"
        assert results[0].steps_per_sec > 0


class TestOscillatorScaling:
    def test_scaling_runs(self):
        config = EfficiencyConfig(
            bench_steps=5, warmup_steps=2,
            oscillator_counts=[16, 32],
        )
        results = bench_oscillator_scaling(config)
        assert len(results) == 2
        # Smaller should be faster or at least run
        for r in results:
            assert r.steps_per_sec > 0


class TestBatchScaling:
    def test_batch_scaling_runs(self):
        config = EfficiencyConfig(
            bench_steps=5, warmup_steps=2,
            batch_sizes=[1, 4],
        )
        results = bench_batch_scaling(config)
        assert len(results) == 2
        for r in results:
            assert r.steps_per_sec > 0
            assert r.samples_per_sec >= r.steps_per_sec


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_full_suite()
