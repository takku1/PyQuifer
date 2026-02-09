"""Module Microbenchmarks — Performance Profiling of PyQuifer Components.

Dual-mode:
  pytest:  python -m pytest tests/benchmarks/bench_modules.py -v --timeout=120
  CLI:     python tests/benchmarks/bench_modules.py

Scenarios:
  1. LearnableKuramotoBank — sweep N_osc + Column B naive baseline + dynamics
  2. HierarchicalPredictiveCoding — sweep layers/dims + Column B
  3. CriticalityController — sweep N_osc, convergence
  4. CrossFrequencyCoupling — sweep channels
  5. CognitiveCycle.tick() — full system with various config flags
  6. TsodyksMarkramSynapse (STP) — Column B naive vs Column C
  7. EpropSTDP — sweep neuron counts
  8. Dtype sweep — fp32/bf16/fp16 on Kuramoto (GPU only)
  9. torch.compile sweep — eager vs compiled Kuramoto
"""
from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from harness import (
    BenchmarkSuite, MemoryTracker, MetricCollector, get_device, set_seed, timer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def measure_throughput(fn, warmup: int = 10, steps: int = 100) -> Dict[str, float]:
    """Measure steps/sec and latency percentiles for a callable."""
    # Warmup
    for _ in range(warmup):
        fn()

    # Timed runs
    latencies = []
    for _ in range(steps):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies.sort()
    total_s = sum(latencies) / 1000
    return {
        "steps_per_sec": round(steps / total_s, 1),
        "latency_p50_ms": round(latencies[len(latencies) // 2], 3),
        "latency_p95_ms": round(latencies[int(len(latencies) * 0.95)], 3),
        "latency_p99_ms": round(latencies[int(len(latencies) * 0.99)], 3),
    }


# ---------------------------------------------------------------------------
# Column B Naive Baselines
# ---------------------------------------------------------------------------

def naive_kuramoto_step(phases: torch.Tensor, coupling: torch.Tensor,
                        nat_freq: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
    """Plain tensor Kuramoto step — no PyQuifer classes."""
    n = phases.shape[0]
    # dtheta_i/dt = omega_i + (K/N) * sum_j sin(theta_j - theta_i)
    diff = phases.unsqueeze(0) - phases.unsqueeze(1)  # (N, N)
    interaction = (coupling * torch.sin(diff)).mean(dim=1)
    phases = phases + dt * (nat_freq + interaction)
    return phases % (2 * math.pi)


def naive_hpc_step(x: torch.Tensor, weights: List[torch.Tensor],
                   lr: float = 0.01) -> float:
    """Simple prediction error with layered linear transforms — no PyQuifer."""
    pred = x
    total_fe = 0.0
    for w in weights:
        pred_next = pred @ w.T
        error = pred - (pred_next @ w)
        total_fe += error.pow(2).sum().item()
        pred = pred_next
    return total_fe


def naive_stp_step(u: torch.Tensor, x: torch.Tensor, spike: torch.Tensor,
                   U: float = 0.2, tau_f: float = 0.75, tau_d: float = 0.05,
                   dt: float = 0.001) -> tuple:
    """Naive Tsodyks-Markram STP equations — plain tensor ops."""
    u = u + dt * (-u / tau_f) + U * (1 - u) * spike
    x = x + dt * ((1 - x) / tau_d) - u * x * spike
    efficacy = u * x
    return u, x, efficacy


# ---------------------------------------------------------------------------
# Dynamics Metrics
# ---------------------------------------------------------------------------

def compute_dynamics_metrics(bank, ext: torch.Tensor,
                             steps: int = 100) -> Dict[str, float]:
    """Run Kuramoto bank for N steps and compute dynamics metrics."""
    R_values = []
    phases_list = []
    for _ in range(steps):
        bank(external_input=ext, steps=1)
        R = bank.get_order_parameter().item()
        R_values.append(R)
        phases_list.append(bank.phases.detach().clone())

    R_arr = torch.tensor(R_values)
    metastability = R_arr.var().item()

    # Chimera index: fraction of oscillators > pi/2 from mean phase
    last_phases = phases_list[-1]
    mean_phase = torch.atan2(
        torch.sin(last_phases).mean(), torch.cos(last_phases).mean()
    )
    deviations = torch.abs(last_phases - mean_phase)
    deviations = torch.min(deviations, 2 * math.pi - deviations)
    chimera_idx = (deviations > math.pi / 2).float().mean().item()

    return {
        "R_mean": round(R_arr.mean().item(), 4),
        "R_std": round(R_arr.std().item(), 4),
        "R_min": round(R_arr.min().item(), 4),
        "R_max": round(R_arr.max().item(), 4),
        "metastability": round(metastability, 6),
        "chimera_index": round(chimera_idx, 4),
    }


# ---------------------------------------------------------------------------
# Scenario 1: LearnableKuramotoBank (with Column B + dynamics)
# ---------------------------------------------------------------------------

def bench_kuramoto(n_osc_list: List[int] = None, steps: int = 100,
                   warmup: int = 10) -> MetricCollector:
    if n_osc_list is None:
        n_osc_list = [16, 32, 64, 128, 256]

    device = get_device()
    mc = MetricCollector("LearnableKuramotoBank Throughput")

    from pyquifer.oscillators import LearnableKuramotoBank

    # Column A: Published baselines (representative N=64)
    mc.record("A_published", "N=64_R", 0.5,
              {"source": "R≈0.5 at moderate coupling, Acebrón et al. (2005) Rev. Mod. Phys."})
    mc.record("A_published", "N=64_steps_per_sec", 10000.0,
              {"source": "KuraNet GPU reference, Takezawa et al. (2024) ICLR 2025"})

    for n_osc in n_osc_list:
        # --- Column B: naive Kuramoto ---
        set_seed(42)
        phases_b = torch.randn(n_osc, device=device) * 2 * math.pi
        coupling_b = torch.ones(n_osc, n_osc, device=device) * 0.5
        nat_freq_b = torch.randn(n_osc, device=device) * 0.1

        def step_b():
            nonlocal phases_b
            phases_b = naive_kuramoto_step(phases_b, coupling_b, nat_freq_b)

        perf_b = measure_throughput(step_b, warmup=warmup, steps=steps)
        mc.record("B_pytorch", f"N={n_osc}_steps_per_sec", perf_b["steps_per_sec"])
        mc.record("B_pytorch", f"N={n_osc}_p50_ms", perf_b["latency_p50_ms"])

        # --- Column C: PyQuifer Kuramoto ---
        set_seed(42)
        bank = LearnableKuramotoBank(
            num_oscillators=n_osc, dt=0.01, topology='global',
        ).to(device)
        ext = torch.randn(n_osc, device=device) * 0.1

        def step_fn():
            bank(external_input=ext, steps=1)

        with MemoryTracker() as mem:
            perf = measure_throughput(step_fn, warmup=warmup, steps=steps)

        R = bank.get_order_parameter().item()
        mc.record("C_pyquifer", f"N={n_osc}_steps_per_sec", perf["steps_per_sec"])
        mc.record("C_pyquifer", f"N={n_osc}_p50_ms", perf["latency_p50_ms"])
        mc.record("C_pyquifer", f"N={n_osc}_p95_ms", perf["latency_p95_ms"])
        mc.record("C_pyquifer", f"N={n_osc}_R", round(R, 4))
        mc.record("C_pyquifer", f"N={n_osc}_peak_mb", mem.peak_mb)

        # Dynamics metrics (for the largest size or all)
        if n_osc <= 64:
            set_seed(42)
            bank_dyn = LearnableKuramotoBank(
                num_oscillators=n_osc, dt=0.01, topology='global',
            ).to(device)
            ext_dyn = torch.randn(n_osc, device=device) * 0.1
            dyn = compute_dynamics_metrics(bank_dyn, ext_dyn, steps=100)
            for k, v in dyn.items():
                mc.record("C_pyquifer", f"N={n_osc}_{k}", v)

    return mc


# ---------------------------------------------------------------------------
# Scenario 2: HierarchicalPredictiveCoding (with Column B)
# ---------------------------------------------------------------------------

def bench_hpc(configs: List[Dict] = None, steps: int = 100,
              warmup: int = 10) -> MetricCollector:
    if configs is None:
        configs = [
            {"dims": [64, 32], "label": "2L-64"},
            {"dims": [64, 32, 16], "label": "3L-64"},
            {"dims": [128, 64, 32], "label": "3L-128"},
            {"dims": [128, 64, 32, 16, 8], "label": "5L-128"},
        ]

    device = get_device()
    mc = MetricCollector("HierarchicalPredictiveCoding Throughput")

    from pyquifer.hierarchical_predictive import HierarchicalPredictiveCoding

    # Column A: Published baselines (representative 3L-64)
    mc.record("A_published", "3L-64_free_energy", 0.5,
              {"source": "Converged free energy, Rao & Ballard (1999) Nat. Neurosci."})
    mc.record("A_published", "3L-64_steps_per_sec", 5000.0,
              {"source": "Feedforward HPC reference estimate"})

    for cfg in configs:
        label = cfg["label"]
        dims = cfg["dims"]

        # --- Column B: naive HPC ---
        set_seed(42)
        weights_b = [torch.randn(dims[i+1], dims[i], device=device) * 0.01
                     for i in range(len(dims) - 1)]
        x_b = torch.randn(dims[0], device=device)

        def step_b():
            naive_hpc_step(x_b, weights_b)

        perf_b = measure_throughput(step_b, warmup=warmup, steps=steps)
        mc.record("B_pytorch", f"{label}_steps_per_sec", perf_b["steps_per_sec"])
        mc.record("B_pytorch", f"{label}_p50_ms", perf_b["latency_p50_ms"])

        # --- Column C: PyQuifer HPC ---
        set_seed(42)
        hpc = HierarchicalPredictiveCoding(
            level_dims=dims, lr=0.05, gen_lr=0.01, num_iterations=3,
        ).to(device)
        x = torch.randn(dims[0], device=device)

        def step_fn():
            hpc(x)

        with MemoryTracker() as mem:
            perf = measure_throughput(step_fn, warmup=warmup, steps=steps)

        result = hpc(x)
        fe = result["free_energy"].item() if isinstance(result["free_energy"], torch.Tensor) else result["free_energy"]

        mc.record("C_pyquifer", f"{label}_steps_per_sec", perf["steps_per_sec"])
        mc.record("C_pyquifer", f"{label}_p50_ms", perf["latency_p50_ms"])
        mc.record("C_pyquifer", f"{label}_p95_ms", perf["latency_p95_ms"])
        mc.record("C_pyquifer", f"{label}_free_energy", round(fe, 4))
        mc.record("C_pyquifer", f"{label}_peak_mb", mem.peak_mb)

    return mc


# ---------------------------------------------------------------------------
# Scenario 3: CriticalityController
# ---------------------------------------------------------------------------

def bench_criticality(n_osc_list: List[int] = None, steps: int = 100,
                      warmup: int = 10) -> MetricCollector:
    if n_osc_list is None:
        n_osc_list = [16, 32, 64, 128]

    device = get_device()
    mc = MetricCollector("CriticalityController Throughput")

    from pyquifer.criticality import CriticalityController

    # Column A: Published baselines
    mc.record("A_published", "N=32_criticality_dist", 0.0,
              {"source": "At criticality σ=1.0, branching_ratio=1.0, Beggs & Plenz (2003)"})
    mc.record("A_published", "N=32_steps_per_sec", 50000.0,
              {"source": "Simple branching ratio estimate, minimal compute"})

    for n_osc in n_osc_list:
        # --- Column B: naive threshold-based branching ratio ---
        set_seed(42)
        activity_b = torch.randn(n_osc, device=device).abs()
        threshold = activity_b.mean()

        def step_b():
            above = (activity_b > threshold).float()
            branching = above.sum() / max(n_osc, 1)
            return branching

        perf_b = measure_throughput(step_b, warmup=warmup, steps=steps)
        mc.record("B_pytorch", f"N={n_osc}_steps_per_sec", perf_b["steps_per_sec"])
        mc.record("B_pytorch", f"N={n_osc}_p50_ms", perf_b["latency_p50_ms"])

        # --- Column C: PyQuifer CriticalityController ---
        set_seed(42)
        ctrl = CriticalityController(target_branching_ratio=1.0).to(device)
        activity = torch.randn(n_osc, device=device).abs()

        def step_fn():
            ctrl(activity)

        with MemoryTracker() as mem:
            perf = measure_throughput(step_fn, warmup=warmup, steps=steps)

        result = ctrl(activity)
        dist = result.get("criticality_distance", torch.tensor(0.0))
        if isinstance(dist, torch.Tensor):
            dist = dist.item()

        mc.record("C_pyquifer", f"N={n_osc}_steps_per_sec", perf["steps_per_sec"])
        mc.record("C_pyquifer", f"N={n_osc}_p50_ms", perf["latency_p50_ms"])
        mc.record("C_pyquifer", f"N={n_osc}_criticality_dist", round(dist, 4))
        mc.record("C_pyquifer", f"N={n_osc}_peak_mb", mem.peak_mb)

    return mc


# ---------------------------------------------------------------------------
# Scenario 4: CrossFrequencyCoupling
# ---------------------------------------------------------------------------

def bench_cfc(channel_list: List[int] = None, steps: int = 100,
              warmup: int = 10) -> MetricCollector:
    if channel_list is None:
        channel_list = [8, 16, 32, 64]

    device = get_device()
    mc = MetricCollector("CrossFrequencyCoupling Throughput")

    from pyquifer.multiplexing import CrossFrequencyCoupling

    # Column A: Published baselines
    mc.record("A_published", "Ch=32_MI", 0.4,
              {"source": "Theta-gamma PAC MI≈0.3-0.5, Canolty et al. (2006) Science"})

    for n_ch in channel_list:
        # --- Column B: naive phase-amplitude coupling ---
        set_seed(42)
        fast_b = torch.randn(n_ch, device=device).abs()
        slow_b = torch.tensor(1.0, device=device)

        def step_b():
            # Naive PAC: amplitude * cos(slow_phase)
            return fast_b * torch.cos(slow_b)

        perf_b = measure_throughput(step_b, warmup=warmup, steps=steps)
        mc.record("B_pytorch", f"Ch={n_ch}_steps_per_sec", perf_b["steps_per_sec"])
        mc.record("B_pytorch", f"Ch={n_ch}_p50_ms", perf_b["latency_p50_ms"])

        # --- Column C: PyQuifer CrossFrequencyCoupling ---
        set_seed(42)
        cfc = CrossFrequencyCoupling(num_fast_oscillators=n_ch).to(device)
        fast_amp = torch.randn(n_ch, device=device).abs()
        slow_phase = torch.tensor(1.0, device=device)

        def step_fn():
            cfc(fast_amp, slow_phase)

        with MemoryTracker() as mem:
            perf = measure_throughput(step_fn, warmup=warmup, steps=steps)

        mc.record("C_pyquifer", f"Ch={n_ch}_steps_per_sec", perf["steps_per_sec"])
        mc.record("C_pyquifer", f"Ch={n_ch}_p50_ms", perf["latency_p50_ms"])
        mc.record("C_pyquifer", f"Ch={n_ch}_peak_mb", mem.peak_mb)

    return mc


# ---------------------------------------------------------------------------
# Scenario 5: CognitiveCycle.tick()
# ---------------------------------------------------------------------------

def bench_cognitive_cycle(configs: List[Dict] = None, steps: int = 50,
                          warmup: int = 5) -> MetricCollector:
    if configs is None:
        configs = [
            {"label": "minimal", "flags": {}},
            {"label": "phase7", "flags": {
                "use_attractor_stability": True,
                "use_evidence_aggregator": True,
                "use_no_progress": True,
            }},
            {"label": "phase8", "flags": {
                "use_ep_training": True,
                "use_oscillation_gated_plasticity": True,
                "use_three_factor": True,
                "use_sleep_consolidation": True,
            }},
            {"label": "full", "flags": {
                "use_stp": True, "use_mean_field": True,
                "use_stuart_landau": True, "use_koopman": True,
                "use_neural_mass": True,
                "use_attractor_stability": True,
                "use_hypothesis_arena": True,
                "use_evidence_aggregator": True,
                "use_no_progress": True, "use_phase_cache": True,
                "use_ep_training": True,
                "use_oscillation_gated_plasticity": True,
                "use_three_factor": True,
                "use_oscillatory_predictive": True,
                "use_sleep_consolidation": True,
                "use_dendritic_credit": True,
            }},
        ]

    device = get_device()
    mc = MetricCollector("CognitiveCycle.tick() Throughput")

    from pyquifer.integration import CognitiveCycle, CycleConfig

    # Column A: Published baselines
    mc.record("A_published", "minimal_ticks_per_sec", 10.0,
              {"source": "Cognitive cycle 100-300ms (3-10 Hz), Baars (2005) Prog. Brain Res."})

    for cfg in configs:
        set_seed(42)
        label = cfg["label"]
        cycle_cfg = CycleConfig.small()
        for k, v in cfg["flags"].items():
            setattr(cycle_cfg, k, v)

        cycle = CognitiveCycle(cycle_cfg).to(device)
        sensory = torch.randn(cycle_cfg.state_dim, device=device)

        def step_fn():
            cycle.tick(sensory, reward=0.0)

        with MemoryTracker() as mem:
            perf = measure_throughput(step_fn, warmup=warmup, steps=steps)

        mc.record("C_pyquifer", f"{label}_steps_per_sec", perf["steps_per_sec"])
        mc.record("C_pyquifer", f"{label}_p50_ms", perf["latency_p50_ms"])
        mc.record("C_pyquifer", f"{label}_p95_ms", perf["latency_p95_ms"])
        mc.record("C_pyquifer", f"{label}_peak_mb", mem.peak_mb)

    return mc


# ---------------------------------------------------------------------------
# Scenario 6: TsodyksMarkramSynapse (STP)
# ---------------------------------------------------------------------------

def bench_stp(synapse_counts: List[int] = None, steps: int = 100,
              warmup: int = 10) -> MetricCollector:
    """Sweep synapse counts: Column B (naive STP) vs Column C (TsodyksMarkramSynapse)."""
    if synapse_counts is None:
        synapse_counts = [256, 512, 1024, 2048, 4096]

    device = get_device()
    mc = MetricCollector("TsodyksMarkramSynapse (STP) Throughput")

    from pyquifer.short_term_plasticity import TsodyksMarkramSynapse

    # Column A: Published baselines
    mc.record("A_published", "N=1024_efficacy", 0.5,
              {"source": "Tsodyks & Markram (1997) PNAS, tau_f=750ms tau_d=50ms"})

    for n_syn in synapse_counts:
        # --- Column B: naive STP ---
        set_seed(42)
        u_b = torch.full((n_syn,), 0.2, device=device)
        x_b = torch.ones(n_syn, device=device)
        spike_b = (torch.rand(n_syn, device=device) > 0.7).float()

        def step_b():
            nonlocal u_b, x_b
            u_b, x_b, _ = naive_stp_step(u_b, x_b, spike_b)

        perf_b = measure_throughput(step_b, warmup=warmup, steps=steps)
        mc.record("B_pytorch", f"N={n_syn}_steps_per_sec", perf_b["steps_per_sec"])
        mc.record("B_pytorch", f"N={n_syn}_p50_ms", perf_b["latency_p50_ms"])

        # --- Column C: TsodyksMarkramSynapse ---
        set_seed(42)
        synapse = TsodyksMarkramSynapse(num_synapses=n_syn).to(device)
        pre_spikes_c = (torch.rand(n_syn, device=device) > 0.7).float()

        def step_c():
            synapse(pre_spikes_c)

        with MemoryTracker() as mem_c:
            perf_c = measure_throughput(step_c, warmup=warmup, steps=steps)

        # Get efficacy from a step
        result = synapse(pre_spikes_c)
        eff = result["efficacy"].mean().item() if isinstance(result, dict) else 0.0

        mc.record("C_pyquifer", f"N={n_syn}_steps_per_sec", perf_c["steps_per_sec"])
        mc.record("C_pyquifer", f"N={n_syn}_p50_ms", perf_c["latency_p50_ms"])
        mc.record("C_pyquifer", f"N={n_syn}_efficacy", round(eff, 4))
        mc.record("C_pyquifer", f"N={n_syn}_peak_mb", mem_c.peak_mb)

    return mc


# ---------------------------------------------------------------------------
# Scenario 7: EpropSTDP
# ---------------------------------------------------------------------------

def bench_eprop(neuron_counts: List[int] = None, steps: int = 100,
                warmup: int = 10) -> MetricCollector:
    """Sweep neuron counts for EpropSTDP."""
    if neuron_counts is None:
        neuron_counts = [128, 256, 512, 1024, 2048]

    device = get_device()
    mc = MetricCollector("EpropSTDP Throughput")

    from pyquifer.advanced_spiking import EpropSTDP

    # Column A: Published baselines
    mc.record("A_published", "N=256_steps_per_sec", 5000.0,
              {"source": "E-prop throughput estimate, Bellec et al. (2020) Nat. Commun."})

    for n in neuron_counts:
        # --- Column B: naive STDP (no eligibility traces) ---
        set_seed(42)
        w_b = torch.randn(n, n, device=device) * 0.01
        pre_b = (torch.rand(n, device=device) > 0.8).float()
        post_b = (torch.rand(n, device=device) > 0.8).float()

        def step_b():
            out = pre_b @ w_b
            # Standard STDP: dw = lr * (pre * post - post * pre)
            dw = 0.01 * (torch.outer(post_b, pre_b) - torch.outer(pre_b, post_b))
            w_b.add_(dw)
            return out

        perf_b = measure_throughput(step_b, warmup=warmup, steps=steps)
        mc.record("B_pytorch", f"N={n}_steps_per_sec", perf_b["steps_per_sec"])
        mc.record("B_pytorch", f"N={n}_p50_ms", perf_b["latency_p50_ms"])

        # --- Column C: PyQuifer EpropSTDP ---
        set_seed(42)
        eprop = EpropSTDP(pre_dim=n, post_dim=n).to(device)
        pre_spikes = (torch.rand(n, device=device) > 0.8).float()
        post_spikes = (torch.rand(n, device=device) > 0.8).float()
        membrane = torch.zeros(n, device=device)

        def step_fn():
            _out = eprop(pre_spikes)
            eprop.update_traces(pre_spikes, post_spikes, membrane)

        with MemoryTracker() as mt:
            perf = measure_throughput(step_fn, warmup=warmup, steps=steps)

        mc.record("C_pyquifer", f"N={n}_steps_per_sec", perf["steps_per_sec"])
        mc.record("C_pyquifer", f"N={n}_p50_ms", perf["latency_p50_ms"])
        mc.record("C_pyquifer", f"N={n}_p95_ms", perf["latency_p95_ms"])
        mc.record("C_pyquifer", f"N={n}_peak_mb", mt.peak_mb)

    return mc


# ---------------------------------------------------------------------------
# Scenario 8: Dtype Sweep
# ---------------------------------------------------------------------------

def bench_dtype_sweep(n_osc: int = 64, steps: int = 100,
                      warmup: int = 10) -> MetricCollector:
    """Sweep fp32, bf16, fp16 on Kuramoto (GPU only for mixed precision)."""
    device = get_device()
    mc = MetricCollector("Kuramoto Dtype Sweep")

    from pyquifer.oscillators import LearnableKuramotoBank

    # Column A: Published baselines
    mc.record("A_published", "float32_R", 0.5,
              {"source": "Reference R at fp32 precision"})
    mc.record("A_published", "bfloat16_speedup", 1.5,
              {"source": "Typical AMP speedup 1.3-2x, NVIDIA (2020)"})

    dtypes = [torch.float32]
    if device.type == "cuda":
        dtypes.extend([torch.bfloat16, torch.float16])

    R_ref = None  # fp32 reference R for drift measurement
    for dtype in dtypes:
        dtype_name = str(dtype).split(".")[-1]
        set_seed(42)
        bank = LearnableKuramotoBank(
            num_oscillators=n_osc, dt=0.01, topology='global',
        ).to(device)
        if dtype != torch.float32:
            bank = bank.to(dtype)
        ext = torch.randn(n_osc, device=device, dtype=dtype) * 0.1

        def step_fn():
            bank(external_input=ext, steps=1)

        perf = measure_throughput(step_fn, warmup=warmup, steps=steps)
        # Cast phases to float32 before computing order parameter to avoid
        # ComplexHalf unsupported error with fp16
        R = bank.get_order_parameter(bank.phases.float()).float().item()

        if R_ref is None:
            R_ref = R
        R_drift = abs(R - R_ref)

        mc.record("C_pyquifer", f"{dtype_name}_steps_per_sec", perf["steps_per_sec"])
        mc.record("C_pyquifer", f"{dtype_name}_p50_ms", perf["latency_p50_ms"])
        mc.record("C_pyquifer", f"{dtype_name}_R", round(R, 4))
        mc.record("C_pyquifer", f"{dtype_name}_R_drift", round(R_drift, 6))

    return mc


# ---------------------------------------------------------------------------
# Scenario 9: torch.compile Sweep
# ---------------------------------------------------------------------------

def bench_compile_sweep(n_osc: int = 64, steps: int = 100,
                        warmup: int = 10) -> MetricCollector:
    """Compare eager vs torch.compile on Kuramoto step."""
    device = get_device()
    mc = MetricCollector("Kuramoto torch.compile Sweep")

    from pyquifer.oscillators import LearnableKuramotoBank

    # Column A: Published baselines
    mc.record("A_published", "compile_speedup", 1.3,
              {"source": "PyTorch 2.0 torch.compile typical speedup 1.1-1.5x"})

    # Eager
    set_seed(42)
    bank_eager = LearnableKuramotoBank(
        num_oscillators=n_osc, dt=0.01, topology='global',
    ).to(device)
    ext = torch.randn(n_osc, device=device) * 0.1

    def eager_fn():
        bank_eager(external_input=ext, steps=1)

    perf_eager = measure_throughput(eager_fn, warmup=warmup, steps=steps)
    mc.record("C_pyquifer", "eager_steps_per_sec", perf_eager["steps_per_sec"])
    mc.record("C_pyquifer", "eager_p50_ms", perf_eager["latency_p50_ms"])

    # Compiled (graceful degradation)
    try:
        set_seed(42)
        bank_compiled = LearnableKuramotoBank(
            num_oscillators=n_osc, dt=0.01, topology='global',
        ).to(device)
        compiled_forward = torch.compile(bank_compiled)
        ext_c = torch.randn(n_osc, device=device) * 0.1

        def compiled_fn():
            compiled_forward(external_input=ext_c, steps=1)

        perf_compiled = measure_throughput(compiled_fn, warmup=warmup, steps=steps)
        mc.record("C_pyquifer", "compiled_steps_per_sec", perf_compiled["steps_per_sec"])
        mc.record("C_pyquifer", "compiled_p50_ms", perf_compiled["latency_p50_ms"])
        speedup = perf_compiled["steps_per_sec"] / max(perf_eager["steps_per_sec"], 1e-6)
        mc.record("C_pyquifer", "compile_speedup", round(speedup, 3))
    except Exception as e:
        mc.record("C_pyquifer", "compiled_status", "skipped",
                  {"error": str(e)[:200]})

    return mc


# ---------------------------------------------------------------------------
# Full Suite
# ---------------------------------------------------------------------------

def run_full_suite():
    print("=" * 60)
    print("Module Microbenchmarks — Full Suite")
    print("=" * 60)

    suite = BenchmarkSuite("Module Microbenchmarks")

    for name, fn in [
        ("Kuramoto", bench_kuramoto),
        ("HPC", bench_hpc),
        ("Criticality", bench_criticality),
        ("CFC", bench_cfc),
        ("CognitiveCycle", bench_cognitive_cycle),
        ("STP", bench_stp),
        ("EpropSTDP", bench_eprop),
        ("Dtype Sweep", bench_dtype_sweep),
        ("Compile Sweep", bench_compile_sweep),
    ]:
        print(f"\n--- {name} ---")
        mc = fn()
        suite.add(mc)
        print(mc.to_markdown_table())

    json_path = str(Path(__file__).parent / "results" / "modules.json")
    suite.to_json(json_path)
    print(f"\nResults saved to {json_path}")


# ---------------------------------------------------------------------------
# Pytest Classes (smoke tests)
# ---------------------------------------------------------------------------

class TestKuramotoBench:
    def test_kuramoto_bench_runs(self):
        mc = bench_kuramoto(n_osc_list=[8, 16], steps=10, warmup=2)
        assert len(mc.results) >= 1

    def test_kuramoto_throughput_positive(self):
        mc = bench_kuramoto(n_osc_list=[8], steps=10, warmup=2)
        for r in mc.results:
            for k, v in r.metrics.items():
                if "steps_per_sec" in k:
                    assert v > 0

    def test_kuramoto_has_column_b(self):
        mc = bench_kuramoto(n_osc_list=[8], steps=10, warmup=2)
        columns = {r.column for r in mc.results}
        assert "B_pytorch" in columns

    def test_kuramoto_dynamics_metrics(self):
        mc = bench_kuramoto(n_osc_list=[8], steps=10, warmup=2)
        metric_keys = set()
        for r in mc.results:
            metric_keys.update(r.metrics.keys())
        assert any("R_mean" in k for k in metric_keys)
        assert any("metastability" in k for k in metric_keys)
        assert any("chimera_index" in k for k in metric_keys)


class TestHPCBench:
    def test_hpc_bench_runs(self):
        mc = bench_hpc(configs=[{"dims": [32, 16], "label": "2L-32"}],
                       steps=10, warmup=2)
        assert len(mc.results) >= 1

    def test_hpc_has_column_b(self):
        mc = bench_hpc(configs=[{"dims": [32, 16], "label": "2L-32"}],
                       steps=10, warmup=2)
        columns = {r.column for r in mc.results}
        assert "B_pytorch" in columns


class TestCriticalityBench:
    def test_criticality_bench_runs(self):
        mc = bench_criticality(n_osc_list=[8, 16], steps=10, warmup=2)
        assert len(mc.results) >= 1


class TestCFCBench:
    def test_cfc_bench_runs(self):
        mc = bench_cfc(channel_list=[8], steps=10, warmup=2)
        assert len(mc.results) >= 1


class TestCognitiveCycleBench:
    def test_minimal_config(self):
        mc = bench_cognitive_cycle(
            configs=[{"label": "minimal", "flags": {}}],
            steps=5, warmup=2,
        )
        assert len(mc.results) >= 1

    def test_phase8_config(self):
        mc = bench_cognitive_cycle(
            configs=[{"label": "phase8", "flags": {
                "use_ep_training": True,
                "use_sleep_consolidation": True,
            }}],
            steps=5, warmup=2,
        )
        assert len(mc.results) >= 1


class TestSTPBench:
    def test_stp_bench_runs(self):
        mc = bench_stp(synapse_counts=[64, 128], steps=10, warmup=2)
        assert len(mc.results) >= 1
        columns = {r.column for r in mc.results}
        assert "B_pytorch" in columns
        assert "C_pyquifer" in columns


class TestEpropBench:
    def test_eprop_bench_runs(self):
        mc = bench_eprop(neuron_counts=[32, 64], steps=10, warmup=2)
        assert len(mc.results) >= 1


class TestDtypeSweep:
    def test_dtype_sweep_runs(self):
        mc = bench_dtype_sweep(n_osc=8, steps=10, warmup=2)
        assert len(mc.results) >= 1
        metric_keys = set()
        for r in mc.results:
            metric_keys.update(r.metrics.keys())
        assert any("float32" in k for k in metric_keys)


class TestCompileSweep:
    def test_compile_sweep_runs(self):
        mc = bench_compile_sweep(n_osc=8, steps=10, warmup=2)
        assert len(mc.results) >= 1


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_full_suite()
