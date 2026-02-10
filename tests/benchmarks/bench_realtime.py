"""Benchmark: Real-Time Product Targets — Bridge Latency & Overhead.

Measures whether PyQuifer modulation fits within real-time generation
budgets for interactive "Neuro-sama"/"Her"-style agents.

Product targets (from BENCHMARKING_INTEGRATION_AUDIT.md):
  bridge.step()          p50 <=2 ms (CPU), <=1 ms (GPU)
  modulate_logits()      p50 <=0.1 ms
  modulate_hidden()      should not dominate ITL
  overhead ratio         C_ITL / B_ITL close to 1.0

Three-column comparison:
  A_published : Human turn-taking / cognitive cycle baselines
  B_pytorch   : Bare model operations (no PyQuifer)
  C_pyquifer  : Same operations with PyQuifer modulation

Dual-mode: ``python bench_realtime.py`` or ``pytest bench_realtime.py -v``
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

_benchmark_dir = Path(__file__).resolve().parent
_pyquifer_src = _benchmark_dir.parent.parent / "src"
if str(_pyquifer_src) not in sys.path:
    sys.path.insert(0, str(_pyquifer_src))

from harness import BenchmarkSuite, MetricCollector, get_device, set_seed


# ---------------------------------------------------------------------------
# Latency helpers
# ---------------------------------------------------------------------------

def _percentiles(values: List[float]) -> Dict[str, float]:
    """Compute p50/p95/p99 from a sorted list of latencies (ms)."""
    values = sorted(values)
    n = len(values)
    return {
        "p50_ms": round(values[n // 2], 4),
        "p95_ms": round(values[int(n * 0.95)], 4),
        "p99_ms": round(values[int(n * 0.99)], 4),
        "mean_ms": round(sum(values) / n, 4),
    }


# ---------------------------------------------------------------------------
# Scenario 1: bridge.step() latency distribution
# ---------------------------------------------------------------------------

def _measure_bridge_latency(bridge, sensory, device, warmup, steps):
    """Core timing loop for bridge.step() on a given device."""
    for _ in range(warmup):
        bridge.step(sensory)
    lats = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        bridge.step(sensory)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000)
    return lats


def bench_bridge_step(warmup: int = 20, steps: int = 500) -> MetricCollector:
    """Measure bridge.step() latency distribution.

    Target: p50 <=2 ms (CPU), <=1 ms (GPU).

    Always measures on CPU (primary product target). Also measures on
    CUDA if available. This prevents the Cat 1 vs Cat 11 discrepancy
    where different devices give wildly different numbers due to CUDA
    kernel launch overhead on many small ops.
    """
    gpu_device = get_device()
    cpu_device = torch.device("cpu")
    mc = MetricCollector("bridge.step() Latency")

    from pyquifer.bridge import PyQuiferBridge

    # A_published: cognitive cycle timing
    mc.record("A_published", "p50_ms", 200.0,
              {"source": "Dehaene & Naccache (2001), 200ms cognitive cycle",
               "note": "biological reference, not a target for PyQuifer"})

    # --- B_pytorch: bare GRU step on CPU (canonical baseline) ---
    set_seed(42)
    bridge = PyQuiferBridge.small()  # CPU
    gru = torch.nn.GRUCell(bridge.config.state_dim, bridge.config.state_dim)
    sensory = torch.randn(bridge.config.state_dim)
    h = torch.zeros(1, bridge.config.state_dim)

    for _ in range(warmup):
        h = gru(sensory.unsqueeze(0), h)
    lat_b = []
    for _ in range(steps):
        t0 = time.perf_counter()
        h = gru(sensory.unsqueeze(0), h)
        lat_b.append((time.perf_counter() - t0) * 1000)

    bp = _percentiles(lat_b)
    mc.record("B_pytorch", "p50_ms", bp["p50_ms"])
    mc.record("B_pytorch", "p95_ms", bp["p95_ms"])
    mc.record("B_pytorch", "p99_ms", bp["p99_ms"])
    mc.record("B_pytorch", "mean_ms", bp["mean_ms"])

    # --- C_pyquifer: bridge.step() on CPU (primary product target) ---
    set_seed(42)
    bridge = PyQuiferBridge.small()  # CPU
    sensory = torch.randn(bridge.config.state_dim)
    lat_c = _measure_bridge_latency(bridge, sensory, cpu_device, warmup, steps)

    cp = _percentiles(lat_c)
    mc.record("C_pyquifer", "cpu_p50_ms", cp["p50_ms"])
    mc.record("C_pyquifer", "cpu_p95_ms", cp["p95_ms"])
    mc.record("C_pyquifer", "cpu_mean_ms", cp["mean_ms"])
    # Keep the p50_ms key for backwards-compat with scoring
    mc.record("C_pyquifer", "p50_ms", cp["p50_ms"])
    mc.record("C_pyquifer", "p95_ms", cp["p95_ms"])
    mc.record("C_pyquifer", "p99_ms", cp["p99_ms"])
    mc.record("C_pyquifer", "mean_ms", cp["mean_ms"])

    # Target check (CPU target = 2ms)
    mc.record("C_pyquifer", "target_ms", 2.0,
              {"meets_target": cp["p50_ms"] <= 2.0})

    # --- CUDA measurements (secondary, if available) ---
    if gpu_device.type == "cuda":
        set_seed(42)
        bridge_gpu = PyQuiferBridge.small().to(gpu_device)
        sensory_gpu = torch.randn(bridge_gpu.config.state_dim, device=gpu_device)
        lat_gpu = _measure_bridge_latency(bridge_gpu, sensory_gpu, gpu_device, warmup, steps)
        gp = _percentiles(lat_gpu)
        mc.record("C_pyquifer", "cuda_p50_ms", gp["p50_ms"])
        mc.record("C_pyquifer", "cuda_p95_ms", gp["p95_ms"])
        mc.record("C_pyquifer", "cuda_mean_ms", gp["mean_ms"])

    # --- Interactive mode on CPU (primary target path) ---
    set_seed(42)
    bridge_int = PyQuiferBridge.interactive()  # CPU
    sensory_int = torch.randn(bridge_int.config.state_dim)
    lat_d = _measure_bridge_latency(bridge_int, sensory_int, cpu_device, warmup, steps)

    dp = _percentiles(lat_d)
    mc.record("C_pyquifer", "interactive_p50_ms", dp["p50_ms"])
    mc.record("C_pyquifer", "interactive_p95_ms", dp["p95_ms"])
    mc.record("C_pyquifer", "interactive_mean_ms", dp["mean_ms"])
    mc.record("C_pyquifer", "interactive_meets_target",
              1.0 if round(dp["p50_ms"], 2) <= 2.0 else 0.0)

    # --- Realtime mode on CPU (minimum latency) ---
    set_seed(42)
    bridge_rt = PyQuiferBridge.realtime()  # CPU
    sensory_rt = torch.randn(bridge_rt.config.state_dim)
    lat_rt = _measure_bridge_latency(bridge_rt, sensory_rt, cpu_device, warmup, steps)

    rp = _percentiles(lat_rt)
    mc.record("C_pyquifer", "realtime_p50_ms", rp["p50_ms"])
    mc.record("C_pyquifer", "realtime_p95_ms", rp["p95_ms"])
    mc.record("C_pyquifer", "realtime_mean_ms", rp["mean_ms"])
    mc.record("C_pyquifer", "realtime_meets_target",
              1.0 if round(rp["p50_ms"], 2) <= 2.0 else 0.0)

    # --- torch.compile CUDA (if inductor/triton available) ---
    if gpu_device.type == "cuda":
        try:
            set_seed(42)
            bridge_comp = PyQuiferBridge.default().to(gpu_device)
            bridge_comp.compile(mode="default")
            sensory_comp = torch.randn(bridge_comp.config.state_dim, device=gpu_device)
            lat_comp = _measure_bridge_latency(bridge_comp, sensory_comp, gpu_device,
                                               warmup=40, steps=steps)
            compp = _percentiles(lat_comp)
            mc.record("C_pyquifer", "compiled_cuda_p50_ms", compp["p50_ms"])
            mc.record("C_pyquifer", "compiled_cuda_p95_ms", compp["p95_ms"])
            mc.record("C_pyquifer", "compiled_cuda_mean_ms", compp["mean_ms"])
        except Exception:
            pass  # inductor/triton not available — skip

    return mc


# ---------------------------------------------------------------------------
# Scenario 2: modulate_logits() latency
# ---------------------------------------------------------------------------

def bench_modulate_logits(warmup: int = 20, steps: int = 500) -> MetricCollector:
    """Measure modulate_logits() latency.

    Target: p50 <=0.1 ms.
    """
    device = get_device()
    mc = MetricCollector("modulate_logits() Latency")

    from pyquifer.bridge import PyQuiferBridge

    mc.record("A_published", "p50_ms", 0.1,
              {"source": "Product target: <=0.1 ms for interactive feel"})

    set_seed(42)
    bridge = PyQuiferBridge.small().to(device)
    sensory = torch.randn(bridge.config.state_dim, device=device)
    state = bridge.step(sensory)

    # Test with realistic vocab sizes
    logits_2d = torch.randn(1, 32000, device=device)  # (batch, vocab)
    logits_3d = torch.randn(1, 64, 32000, device=device)  # (batch, seq, vocab)

    # Warmup
    for _ in range(warmup):
        bridge.modulate_logits(logits_2d, state)

    # B_pytorch: no-op baseline (just a tensor copy for fair comparison)
    lat_b = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = logits_2d / 1.0  # minimal op
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat_b.append((time.perf_counter() - t0) * 1000)

    bp = _percentiles(lat_b)
    mc.record("B_pytorch", "p50_ms", bp["p50_ms"])

    # C_pyquifer: modulate_logits (2D)
    lat_c_2d = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        bridge.modulate_logits(logits_2d, state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat_c_2d.append((time.perf_counter() - t0) * 1000)

    cp2d = _percentiles(lat_c_2d)
    mc.record("C_pyquifer", "p50_ms_2d", cp2d["p50_ms"])
    mc.record("C_pyquifer", "p95_ms_2d", cp2d["p95_ms"])

    # C_pyquifer: modulate_logits (3D)
    lat_c_3d = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        bridge.modulate_logits(logits_3d, state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat_c_3d.append((time.perf_counter() - t0) * 1000)

    cp3d = _percentiles(lat_c_3d)
    mc.record("C_pyquifer", "p50_ms_3d", cp3d["p50_ms"])
    mc.record("C_pyquifer", "p95_ms_3d", cp3d["p95_ms"])

    mc.record("C_pyquifer", "meets_target_2d", 1.0 if cp2d["p50_ms"] <= 0.1 else 0.0)

    return mc


# ---------------------------------------------------------------------------
# Scenario 3: modulate_hidden() latency
# ---------------------------------------------------------------------------

def bench_modulate_hidden(warmup: int = 20, steps: int = 500) -> MetricCollector:
    """Measure modulate_hidden() latency for typical hidden state sizes."""
    device = get_device()
    mc = MetricCollector("modulate_hidden() Latency")

    from pyquifer.bridge import PyQuiferBridge

    mc.record("A_published", "p50_ms", 0.5,
              {"source": "Product target: should not dominate ITL (~40ms)",
               "note": "<=0.5ms keeps hidden hook overhead <2% of ITL"})

    set_seed(42)
    bridge = PyQuiferBridge.small().to(device)
    sensory = torch.randn(bridge.config.state_dim, device=device)
    state = bridge.step(sensory)

    # Typical transformer hidden state shapes
    hidden_sizes = [
        ("small_32d", (1, 64, 32)),
        ("medium_768d", (1, 128, 768)),
        ("large_4096d", (1, 256, 4096)),
    ]

    for label, shape in hidden_sizes:
        hidden = torch.randn(*shape, device=device)

        # Warmup
        for _ in range(warmup):
            bridge.modulate_hidden(hidden, state)

        # B_pytorch: identity operation
        lat_b = []
        for _ in range(steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = hidden + 0.0
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat_b.append((time.perf_counter() - t0) * 1000)

        bp = _percentiles(lat_b)
        mc.record("B_pytorch", f"{label}_p50_ms", bp["p50_ms"])

        # C_pyquifer
        lat_c = []
        for _ in range(steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            bridge.modulate_hidden(hidden, state)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat_c.append((time.perf_counter() - t0) * 1000)

        cp = _percentiles(lat_c)
        mc.record("C_pyquifer", f"{label}_p50_ms", cp["p50_ms"])
        mc.record("C_pyquifer", f"{label}_p95_ms", cp["p95_ms"])

        # Overhead ratio
        if bp["p50_ms"] > 0:
            ratio = cp["p50_ms"] / bp["p50_ms"]
            mc.record("C_pyquifer", f"{label}_overhead_ratio", round(ratio, 2))

    return mc


# ---------------------------------------------------------------------------
# Scenario 4: End-to-end per-token overhead
# ---------------------------------------------------------------------------

def bench_per_token_overhead(warmup: int = 10, tokens: int = 200) -> MetricCollector:
    """Measure per-token overhead: step + modulate_logits + modulate_hidden.

    Simulates a generation loop where each token triggers the full
    PyQuifer pipeline vs a bare forward pass.
    """
    device = get_device()
    mc = MetricCollector("Per-Token Overhead")

    from pyquifer.bridge import PyQuiferBridge

    mc.record("A_published", "overhead_ratio", 1.0,
              {"source": "Target: C_ITL / B_ITL close to 1.0",
               "note": "<=1.05 means <5% overhead"})

    set_seed(42)
    bridge = PyQuiferBridge.small().to(device)
    dim = bridge.config.state_dim

    # Simulate a transformer layer output
    hidden = torch.randn(1, 1, dim, device=device)
    logits = torch.randn(1, 32000, device=device)
    sensory = torch.randn(dim, device=device)

    # Warmup
    for _ in range(warmup):
        bridge.step(sensory)

    # B_pytorch: bare "forward" (GRU + logit scale)
    gru = torch.nn.GRUCell(dim, dim).to(device)
    h = torch.zeros(1, dim, device=device)

    lat_b = []
    for _ in range(tokens):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        h = gru(sensory.unsqueeze(0), h)
        _ = logits / 1.0
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat_b.append((time.perf_counter() - t0) * 1000)

    bp = _percentiles(lat_b)
    mc.record("B_pytorch", "p50_ms", bp["p50_ms"])
    mc.record("B_pytorch", "p95_ms", bp["p95_ms"])

    # C_pyquifer: full pipeline per token
    lat_c = []
    for _ in range(tokens):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        state = bridge.step(sensory)
        _ = bridge.modulate_hidden(hidden, state)
        _ = bridge.modulate_logits(logits, state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat_c.append((time.perf_counter() - t0) * 1000)

    cp = _percentiles(lat_c)
    mc.record("C_pyquifer", "p50_ms", cp["p50_ms"])
    mc.record("C_pyquifer", "p95_ms", cp["p95_ms"])
    mc.record("C_pyquifer", "p99_ms", cp["p99_ms"])

    # Overhead ratio
    if bp["p50_ms"] > 0:
        ratio = cp["p50_ms"] / bp["p50_ms"]
        mc.record("C_pyquifer", "overhead_ratio", round(ratio, 2))
        mc.record("C_pyquifer", "meets_5pct_target",
                  1.0 if ratio <= 1.05 else 0.0)

    # Stepped mode: bridge.step() every N tokens (interpolated)
    for step_every in [4, 8, 16]:
        lat_stepped = []
        state = bridge.step(sensory)
        for i in range(tokens):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if i % step_every == 0:
                state = bridge.step(sensory)
            _ = bridge.modulate_hidden(hidden, state)
            _ = bridge.modulate_logits(logits, state)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat_stepped.append((time.perf_counter() - t0) * 1000)

        sp = _percentiles(lat_stepped)
        mc.record("C_pyquifer", f"step_every_{step_every}_p50_ms", sp["p50_ms"])
        if bp["p50_ms"] > 0:
            mc.record("C_pyquifer", f"step_every_{step_every}_ratio",
                      round(sp["p50_ms"] / bp["p50_ms"], 2))

    return mc


# ---------------------------------------------------------------------------
# Full Suite
# ---------------------------------------------------------------------------

def run_full_suite() -> Dict:
    """Run all real-time latency benchmarks."""
    print("=" * 60)
    print("  Real-Time Product Targets — Bridge Latency & Overhead")
    print("=" * 60)

    suite = BenchmarkSuite("Real-Time Latency")

    for name, fn in [
        ("bridge.step()", bench_bridge_step),
        ("modulate_logits()", bench_modulate_logits),
        ("modulate_hidden()", bench_modulate_hidden),
        ("Per-Token Overhead", bench_per_token_overhead),
    ]:
        print(f"\n--- {name} ---")
        mc = fn()
        suite.add(mc)
        print(mc.to_markdown_table())

    results_dir = _benchmark_dir / "results"
    results_dir.mkdir(exist_ok=True)
    suite.to_json(str(results_dir / "realtime.json"))
    print(f"\nResults saved: {results_dir / 'realtime.json'}")

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Pytest
# ---------------------------------------------------------------------------

class TestBridgeStepLatency:
    def test_step_latency_measured(self):
        mc = bench_bridge_step(warmup=5, steps=20)
        keys = set()
        for r in mc.results:
            keys.update(r.metrics.keys())
        assert "p50_ms" in keys
        assert "target_ms" in keys

    def test_step_under_50ms(self):
        """Sanity: bridge.step() should be well under 50ms."""
        mc = bench_bridge_step(warmup=5, steps=50)
        for r in mc.results:
            if r.column == "C_pyquifer" and "p50_ms" in r.metrics:
                assert r.metrics["p50_ms"] < 50, "bridge.step() >50ms"


class TestModulateLogitsLatency:
    def test_logits_latency_measured(self):
        mc = bench_modulate_logits(warmup=5, steps=20)
        keys = set()
        for r in mc.results:
            keys.update(r.metrics.keys())
        assert "p50_ms_2d" in keys


class TestModulateHiddenLatency:
    def test_hidden_latency_measured(self):
        mc = bench_modulate_hidden(warmup=5, steps=20)
        keys = set()
        for r in mc.results:
            keys.update(r.metrics.keys())
        assert "small_32d_p50_ms" in keys


class TestPerTokenOverhead:
    def test_overhead_measured(self):
        mc = bench_per_token_overhead(warmup=3, tokens=20)
        keys = set()
        for r in mc.results:
            keys.update(r.metrics.keys())
        assert "overhead_ratio" in keys
        assert "step_every_8_p50_ms" in keys


if __name__ == "__main__":
    run_full_suite()
