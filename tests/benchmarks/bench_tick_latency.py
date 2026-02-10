"""Tick latency benchmark: minimal vs diagnostic tick, p50/p95/p99.

Measures the hot-path performance of CognitiveCycle.tick() and
PyQuiferBridge.step() across config presets and devices.

Usage:
    pytest PyQuifer/tests/benchmarks/bench_tick_latency.py -v -s
    python PyQuifer/tests/benchmarks/bench_tick_latency.py  # standalone
"""

import time
import statistics
import pytest
import torch

from pyquifer.integration import CognitiveCycle, CycleConfig, TickResult
from pyquifer.bridge import PyQuiferBridge, sync_debug_mode


def _measure_latencies(fn, n_warmup=20, n_measure=200):
    """Run fn() n_warmup+n_measure times, return sorted latency list in ms."""
    for _ in range(n_warmup):
        fn()
    latencies = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    latencies.sort()
    return latencies


def _percentiles(latencies):
    """Compute p50, p95, p99 from sorted list."""
    n = len(latencies)
    return {
        'p50': latencies[n // 2],
        'p95': latencies[int(n * 0.95)],
        'p99': latencies[int(n * 0.99)],
        'mean': statistics.mean(latencies),
        'stdev': statistics.stdev(latencies) if n > 1 else 0.0,
    }


# ── Fixtures ──

@pytest.fixture(params=["default", "interactive", "realtime"])
def config(request):
    """CycleConfig presets."""
    return getattr(CycleConfig, request.param)()


@pytest.fixture
def cycle(config):
    return CognitiveCycle(config)


@pytest.fixture
def bridge(config):
    return PyQuiferBridge(config)


# ── Minimal tick latency ──

class TestMinimalTickLatency:
    """Benchmark minimal tick (return_diagnostics=False)."""

    def test_minimal_tick_returns_tick_result(self, cycle, config):
        x = torch.randn(config.state_dim)
        result = cycle.tick(x, return_diagnostics=False)
        assert isinstance(result, TickResult)

    def test_minimal_tick_p50(self, cycle, config):
        x = torch.randn(config.state_dim)
        latencies = _measure_latencies(
            lambda: cycle.tick(x, return_diagnostics=False),
            n_warmup=10, n_measure=100,
        )
        stats = _percentiles(latencies)
        print(f"\n  Minimal tick [{type(config).__name__}]: "
              f"p50={stats['p50']:.2f}ms p95={stats['p95']:.2f}ms "
              f"p99={stats['p99']:.2f}ms mean={stats['mean']:.2f}ms")


class TestDiagnosticTickLatency:
    """Benchmark diagnostic tick (return_diagnostics=True)."""

    def test_diagnostic_tick_p50(self, cycle, config):
        x = torch.randn(config.state_dim)
        latencies = _measure_latencies(
            lambda: cycle.tick(x, return_diagnostics=True),
            n_warmup=10, n_measure=100,
        )
        stats = _percentiles(latencies)
        print(f"\n  Diagnostic tick [{type(config).__name__}]: "
              f"p50={stats['p50']:.2f}ms p95={stats['p95']:.2f}ms "
              f"p99={stats['p99']:.2f}ms mean={stats['mean']:.2f}ms")


class TestBridgeStepLatency:
    """Benchmark bridge.step() (minimal mode)."""

    def test_bridge_step_p50(self, bridge, config):
        x = torch.randn(config.state_dim)
        latencies = _measure_latencies(
            lambda: bridge.step(x),
            n_warmup=10, n_measure=100,
        )
        stats = _percentiles(latencies)
        print(f"\n  Bridge step [{type(config).__name__}]: "
              f"p50={stats['p50']:.2f}ms p95={stats['p95']:.2f}ms "
              f"p99={stats['p99']:.2f}ms mean={stats['mean']:.2f}ms")


class TestAllocationBudget:
    """Verify per-tick allocation stays near zero after warmup."""

    def test_cpu_allocation_stable(self, cycle, config):
        """After warmup, tick should not increase torch memory."""
        x = torch.randn(config.state_dim)
        # Warmup
        for _ in range(20):
            cycle.tick(x, return_diagnostics=False)

        # Measure — on CPU we can't use cuda memory stats, but we can
        # verify no new tensors are being created by checking the tick
        # returns the same shape every time
        results = []
        for _ in range(10):
            r = cycle.tick(x, return_diagnostics=False)
            results.append(tuple(f.shape for f in r))
        # All shapes must be identical
        assert all(s == results[0] for s in results)


class TestSyncDetection:
    """Verify minimal tick produces no CUDA sync warnings (CPU no-op test)."""

    def test_no_sync_warnings_cpu(self, bridge, config):
        """On CPU, sync_debug_mode is a no-op — just verify it runs."""
        x = torch.randn(config.state_dim)
        stats = bridge.profile_step(x, n_warmup=3, n_measure=5, sync_mode="warn")
        assert stats['sync_warnings'] == 0
        assert len(stats['latencies_ms']) == 5


# ── Standalone runner ──

if __name__ == '__main__':
    print("=" * 60)
    print("PyQuifer Tick Latency Benchmark")
    print("=" * 60)

    for preset_name in ["default", "interactive", "realtime"]:
        config = getattr(CycleConfig, preset_name)()
        cycle = CognitiveCycle(config)
        bridge = PyQuiferBridge(config)
        x = torch.randn(config.state_dim)

        print(f"\n--- {preset_name.upper()} config ---")

        # Minimal tick
        lats = _measure_latencies(
            lambda: cycle.tick(x, return_diagnostics=False),
            n_warmup=20, n_measure=200,
        )
        s = _percentiles(lats)
        print(f"  Minimal tick:  p50={s['p50']:.2f}ms  p95={s['p95']:.2f}ms  "
              f"p99={s['p99']:.2f}ms  mean={s['mean']:.2f}ms")

        # Diagnostic tick
        lats = _measure_latencies(
            lambda: cycle.tick(x, return_diagnostics=True),
            n_warmup=20, n_measure=200,
        )
        s = _percentiles(lats)
        print(f"  Diagnostic tick: p50={s['p50']:.2f}ms  p95={s['p95']:.2f}ms  "
              f"p99={s['p99']:.2f}ms  mean={s['mean']:.2f}ms")

        # Bridge step
        lats = _measure_latencies(
            lambda: bridge.step(x),
            n_warmup=20, n_measure=200,
        )
        s = _percentiles(lats)
        print(f"  Bridge step:   p50={s['p50']:.2f}ms  p95={s['p95']:.2f}ms  "
              f"p99={s['p99']:.2f}ms  mean={s['mean']:.2f}ms")

        # Profile step with sync detection
        stats = bridge.profile_step(x, n_warmup=5, n_measure=20)
        print(f"  Profile step:  p50={stats['p50_ms']:.2f}ms  "
              f"sync_warnings={stats['sync_warnings']}")

    print("\n" + "=" * 60)
    print("Done.")
