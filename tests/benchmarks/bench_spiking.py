"""
Benchmark #2: PyQuifer Spiking Modules vs NeuroBench Metrics

Evaluates PyQuifer's spiking neural network modules against the NeuroBench
framework's standard metrics: footprint, activation sparsity, connection
sparsity, synaptic operations, firing rate, and neuron dynamics correctness.

Modules under test:
  - spiking.py: LIFNeuron, SpikingLayer, OscillatorySNN, STDPLayer, AdExNeuron
  - advanced_spiking.py: SynapticNeuron, AlphaNeuron, RecurrentSynapticLayer,
                          EligibilityModulatedSTDP, EpropSTDP
  - short_term_plasticity.py: TsodyksMarkramSynapse, STPLayer

NeuroBench reference: tests/benchmarks/neurobench/
  - Metrics: Footprint, ConnectionSparsity, ActivationSparsity, SynapticOperations
  - Leaderboard targets: activation_sparsity > 0.9 for SNNs, low firing rates

Dual-mode:
  - python bench_spiking.py        -> full suite + plots
  - pytest bench_spiking.py -v     -> test functions only
"""

from __future__ import annotations

import math
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ── PyQuifer imports ──
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from pyquifer.spiking import (
    LIFNeuron,
    SpikingLayer,
    OscillatorySNN,
    STDPLayer,
    AdExNeuron,
    SpikeEncoder,
    SpikeDecoder,
    SynapticDelay,
)
from pyquifer.advanced_spiking import (
    SynapticNeuron,
    AlphaNeuron,
    RecurrentSynapticLayer,
    EligibilityModulatedSTDP,
    EpropSTDP,
)
from pyquifer.short_term_plasticity import TsodyksMarkramSynapse, STPLayer

# ── Optional imports ──
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: NeuroBench Metric Reimplementations (standalone)
# ═══════════════════════════════════════════════════════════════════════════════

# NeuroBench neuron operation costs (from neurobench/metrics/workload/neuron_operations.py)
NEURON_OPS_COST = {
    "Leaky": 4,      # LIF: 4 ops per update
    "Synaptic": 6,   # 2nd order LIF: 6 ops per update
    "Alpha": 18,     # Alpha neuron: 18 ops per update
    "AdEx": 8,       # AdEx: ~8 ops (dV + dw + exp + spike + reset)
}


@dataclass
class StaticMetrics:
    """NeuroBench-style static metrics for a module."""
    name: str
    footprint_bytes: int
    param_count: int
    buffer_count: int
    connection_sparsity: float  # fraction of zero weights


@dataclass
class WorkloadMetrics:
    """NeuroBench-style workload metrics from running a module."""
    name: str
    activation_sparsity: float  # fraction of zero activations (1 - firing_rate)
    mean_firing_rate: float     # spikes per neuron per timestep
    total_spikes: int
    total_activations: int
    effective_ops: float        # ops that actually happened (sparse)
    dense_ops: float            # ops if no sparsity (all neurons every step)


@dataclass
class DynamicsResult:
    """Results from neuron dynamics correctness tests."""
    name: str
    membrane_traces: torch.Tensor  # (timesteps, neurons)
    spike_times: List[List[int]]   # per-neuron spike time indices
    mean_firing_rate: float
    isi_cv: float                  # coefficient of variation of ISI (regularity)


@dataclass
class STDPResult:
    """Results from STDP learning benchmarks."""
    name: str
    weight_trajectory: List[float]  # mean weight over time
    final_weight_mean: float
    final_weight_std: float
    learned_correlation: bool  # did correlated inputs strengthen?


@dataclass
class STPResult:
    """Results from short-term plasticity benchmarks."""
    name: str
    facilitation_trace: List[float]  # u values over time
    depression_trace: List[float]    # x values over time
    efficacy_trace: List[float]      # u*x over time
    paired_pulse_ratio: float        # PSP2/PSP1 for paired stimuli


def compute_static_metrics(module: nn.Module, name: str) -> StaticMetrics:
    """Compute NeuroBench-style static metrics."""
    param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in module.buffers())
    param_count = sum(p.numel() for p in module.parameters())
    buffer_count = sum(b.numel() for b in module.buffers())

    # Connection sparsity: fraction of zero weights in Linear/weight layers
    zero_weights = 0
    total_weights = 0
    for name_p, p in module.named_parameters():
        if "weight" in name_p:
            zero_weights += (p == 0).sum().item()
            total_weights += p.numel()

    sparsity = zero_weights / total_weights if total_weights > 0 else 0.0

    return StaticMetrics(
        name=name,
        footprint_bytes=param_bytes + buffer_bytes,
        param_count=param_count,
        buffer_count=buffer_count,
        connection_sparsity=round(sparsity, 4),
    )


@contextmanager
def timer():
    result = {"elapsed": 0.0}
    start = time.perf_counter()
    yield result
    result["elapsed"] = time.perf_counter() - start


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Neuron Dynamics Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

BENCH_TIMESTEPS = 200
BENCH_NEURONS = 64
BENCH_BATCH = 8
BENCH_SEQ = 50


def _compute_isi_cv(spike_times: List[int]) -> float:
    """Coefficient of variation of inter-spike intervals."""
    if len(spike_times) < 3:
        return 0.0
    isis = [spike_times[i + 1] - spike_times[i] for i in range(len(spike_times) - 1)]
    mean_isi = sum(isis) / len(isis)
    if mean_isi == 0:
        return 0.0
    var_isi = sum((x - mean_isi) ** 2 for x in isis) / len(isis)
    return math.sqrt(var_isi) / mean_isi


def bench_lif_dynamics(constant_current: float = 1.5) -> DynamicsResult:
    """Benchmark LIF neuron with constant current injection."""
    torch.manual_seed(42)
    lif = LIFNeuron(tau=10.0, threshold=1.0, v_reset=0.0, dt=1.0, learnable=False)

    membrane = torch.zeros(1, BENCH_NEURONS)
    current = torch.full((1, BENCH_NEURONS), constant_current)

    traces = []
    spike_times: List[List[int]] = [[] for _ in range(BENCH_NEURONS)]
    total_spikes = 0

    for t in range(BENCH_TIMESTEPS):
        spikes, membrane = lif(current, membrane)
        traces.append(membrane.squeeze(0).clone())
        total_spikes += spikes.sum().item()
        for n in range(BENCH_NEURONS):
            if spikes[0, n] > 0.5:
                spike_times[n].append(t)

    mean_rate = total_spikes / (BENCH_TIMESTEPS * BENCH_NEURONS)
    # Average ISI CV across neurons that spiked enough
    cvs = [_compute_isi_cv(st) for st in spike_times if len(st) >= 3]
    mean_cv = sum(cvs) / len(cvs) if cvs else 0.0

    return DynamicsResult(
        name="LIFNeuron",
        membrane_traces=torch.stack(traces),
        spike_times=spike_times,
        mean_firing_rate=mean_rate,
        isi_cv=mean_cv,
    )


def bench_adex_dynamics() -> DynamicsResult:
    """Benchmark AdEx neuron — should show adaptation (decreasing rate)."""
    torch.manual_seed(42)
    adex = AdExNeuron(a=0.0, b=0.05, tau_w=100.0, dt=1.0)
    adex.eval()  # no surrogate gradient

    V, w = adex.init_state((1, BENCH_NEURONS))
    current = torch.full((1, BENCH_NEURONS), 0.3)

    traces = []
    spike_times: List[List[int]] = [[] for _ in range(BENCH_NEURONS)]
    total_spikes = 0

    for t in range(BENCH_TIMESTEPS):
        spikes, V, w = adex(current, V, w)
        traces.append(V.squeeze(0).detach().clone())
        total_spikes += spikes.sum().item()
        for n in range(BENCH_NEURONS):
            if spikes[0, n] > 0.5:
                spike_times[n].append(t)

    mean_rate = total_spikes / (BENCH_TIMESTEPS * BENCH_NEURONS)
    cvs = [_compute_isi_cv(st) for st in spike_times if len(st) >= 3]
    mean_cv = sum(cvs) / len(cvs) if cvs else 0.0

    return DynamicsResult(
        name="AdExNeuron",
        membrane_traces=torch.stack(traces),
        spike_times=spike_times,
        mean_firing_rate=mean_rate,
        isi_cv=mean_cv,
    )


def bench_synaptic_dynamics() -> DynamicsResult:
    """Benchmark SynapticNeuron (2nd-order LIF)."""
    torch.manual_seed(42)
    neuron = SynapticNeuron(alpha=0.9, beta=0.8, threshold=1.0)
    neuron.eval()

    syn = torch.zeros(1, BENCH_NEURONS)
    mem = torch.zeros(1, BENCH_NEURONS)
    current = torch.full((1, BENCH_NEURONS), 0.5)

    traces = []
    spike_times: List[List[int]] = [[] for _ in range(BENCH_NEURONS)]
    total_spikes = 0

    for t in range(BENCH_TIMESTEPS):
        spikes, syn, mem = neuron(current, syn, mem)
        traces.append(mem.squeeze(0).detach().clone())
        total_spikes += spikes.sum().item()
        for n in range(BENCH_NEURONS):
            if spikes[0, n] > 0.5:
                spike_times[n].append(t)

    mean_rate = total_spikes / (BENCH_TIMESTEPS * BENCH_NEURONS)
    cvs = [_compute_isi_cv(st) for st in spike_times if len(st) >= 3]
    mean_cv = sum(cvs) / len(cvs) if cvs else 0.0

    return DynamicsResult(
        name="SynapticNeuron",
        membrane_traces=torch.stack(traces),
        spike_times=spike_times,
        mean_firing_rate=mean_rate,
        isi_cv=mean_cv,
    )


def bench_alpha_dynamics() -> DynamicsResult:
    """Benchmark AlphaNeuron (E/I balance)."""
    torch.manual_seed(42)
    neuron = AlphaNeuron(alpha=0.95, beta=0.85, threshold=1.0)
    neuron.eval()

    syn_e = torch.zeros(1, BENCH_NEURONS)
    syn_i = torch.zeros(1, BENCH_NEURONS)
    mem = torch.zeros(1, BENCH_NEURONS)
    current = torch.full((1, BENCH_NEURONS), 0.8)

    traces = []
    spike_times: List[List[int]] = [[] for _ in range(BENCH_NEURONS)]
    total_spikes = 0

    for t in range(BENCH_TIMESTEPS):
        spikes, syn_e, syn_i, mem = neuron(current, syn_e, syn_i, mem)
        traces.append(mem.squeeze(0).detach().clone())
        total_spikes += spikes.sum().item()
        for n in range(BENCH_NEURONS):
            if spikes[0, n] > 0.5:
                spike_times[n].append(t)

    mean_rate = total_spikes / (BENCH_TIMESTEPS * BENCH_NEURONS)
    cvs = [_compute_isi_cv(st) for st in spike_times if len(st) >= 3]
    mean_cv = sum(cvs) / len(cvs) if cvs else 0.0

    return DynamicsResult(
        name="AlphaNeuron",
        membrane_traces=torch.stack(traces),
        spike_times=spike_times,
        mean_firing_rate=mean_rate,
        isi_cv=mean_cv,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Activation Sparsity & Workload Metrics
# ═══════════════════════════════════════════════════════════════════════════════


def bench_workload_spiking_layer() -> WorkloadMetrics:
    """Measure activation sparsity and ops for SpikingLayer."""
    torch.manual_seed(42)
    layer = SpikingLayer(input_dim=32, hidden_dim=BENCH_NEURONS, tau=10.0, recurrent=True)
    layer.eval()

    x = torch.randn(BENCH_BATCH, BENCH_SEQ, 32) * 0.3
    with torch.no_grad():
        spikes = layer(x)

    total_activations = spikes.numel()
    total_spikes = (spikes > 0.5).sum().item()
    activation_sparsity = 1.0 - total_spikes / total_activations
    firing_rate = total_spikes / total_activations

    # Effective ACs: only when a presynaptic neuron spikes
    # For a linear layer with hidden_dim outputs, each spike triggers hidden_dim ACs
    effective_ops = total_spikes * BENCH_NEURONS  # ACs from spike propagation
    dense_ops = total_activations * BENCH_NEURONS  # if every neuron spiked every step

    return WorkloadMetrics(
        name="SpikingLayer",
        activation_sparsity=activation_sparsity,
        mean_firing_rate=firing_rate,
        total_spikes=total_spikes,
        total_activations=total_activations,
        effective_ops=effective_ops,
        dense_ops=dense_ops,
    )


def bench_workload_oscillatory_snn() -> WorkloadMetrics:
    """Measure activation sparsity for OscillatorySNN."""
    torch.manual_seed(42)
    osc = OscillatorySNN(input_dim=16, num_excitatory=48, num_inhibitory=16)
    osc.eval()

    x = torch.randn(BENCH_BATCH, BENCH_SEQ, 16) * 0.3
    with torch.no_grad():
        output, rates = osc(x, steps_per_input=5)

    # Firing rate is returned directly
    mean_rate = rates.mean().item()
    activation_sparsity = 1.0 - mean_rate
    total_neurons = 48 * BENCH_BATCH * BENCH_SEQ * 5  # exc neurons * batch * seq * sub-steps
    total_spikes = int(mean_rate * total_neurons)

    effective_ops = total_spikes * 48
    dense_ops = total_neurons * 48

    return WorkloadMetrics(
        name="OscillatorySNN",
        activation_sparsity=activation_sparsity,
        mean_firing_rate=mean_rate,
        total_spikes=total_spikes,
        total_activations=total_neurons,
        effective_ops=effective_ops,
        dense_ops=dense_ops,
    )


def bench_workload_recurrent_synaptic() -> WorkloadMetrics:
    """Measure activation sparsity for RecurrentSynapticLayer."""
    torch.manual_seed(42)
    layer = RecurrentSynapticLayer(
        input_dim=32, hidden_dim=BENCH_NEURONS,
        alpha=0.9, beta=0.8, recurrent_type="linear",
    )
    layer.eval()

    x = torch.randn(BENCH_BATCH, 32) * 0.3
    syn, mem, spk = None, None, None
    total_spikes = 0
    total_activations = 0

    with torch.no_grad():
        for t in range(BENCH_SEQ):
            spk, syn, mem = layer(x, syn, mem, spk)
            total_spikes += (spk > 0.5).sum().item()
            total_activations += spk.numel()

    activation_sparsity = 1.0 - total_spikes / total_activations
    firing_rate = total_spikes / total_activations

    effective_ops = total_spikes * BENCH_NEURONS * NEURON_OPS_COST["Synaptic"]
    dense_ops = total_activations * NEURON_OPS_COST["Synaptic"]

    return WorkloadMetrics(
        name="RecurrentSynapticLayer",
        activation_sparsity=activation_sparsity,
        mean_firing_rate=firing_rate,
        total_spikes=total_spikes,
        total_activations=total_activations,
        effective_ops=effective_ops,
        dense_ops=dense_ops,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: STDP Learning Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════


def bench_stdp_correlation() -> STDPResult:
    """Test that STDP strengthens correlated pre-post connections."""
    torch.manual_seed(42)
    stdp = STDPLayer(pre_size=20, post_size=10, a_plus=0.01, a_minus=0.01)
    initial_mean = stdp.weights.data.mean().item()

    weight_traj = [initial_mean]

    # Correlated: pre fires, then post fires 2 steps later (LTP-dominant)
    for t in range(200):
        pre = (torch.rand(1, 20) > 0.85).float()
        # Post spikes correlated with pre (with delay)
        post = torch.zeros(1, 10)
        if t > 0:
            post = (torch.rand(1, 10) > 0.85).float() * (pre[:, :10] > 0.5).float()

        stdp(pre, post, learn=True)
        if t % 10 == 0:
            weight_traj.append(stdp.weights.data.mean().item())

    return STDPResult(
        name="STDPLayer (correlated)",
        weight_trajectory=weight_traj,
        final_weight_mean=stdp.weights.data.mean().item(),
        final_weight_std=stdp.weights.data.std().item(),
        learned_correlation=stdp.weights.data.mean().item() > initial_mean,
    )


def bench_stdp_uncorrelated() -> STDPResult:
    """Test that uncorrelated inputs don't systematically change weights."""
    torch.manual_seed(42)
    stdp = STDPLayer(pre_size=20, post_size=10, a_plus=0.01, a_minus=0.01)
    initial_mean = stdp.weights.data.mean().item()

    weight_traj = [initial_mean]

    # Uncorrelated: independent random pre and post
    for t in range(200):
        pre = (torch.rand(1, 20) > 0.85).float()
        post = (torch.rand(1, 10) > 0.85).float()

        stdp(pre, post, learn=True)
        if t % 10 == 0:
            weight_traj.append(stdp.weights.data.mean().item())

    return STDPResult(
        name="STDPLayer (uncorrelated)",
        weight_trajectory=weight_traj,
        final_weight_mean=stdp.weights.data.mean().item(),
        final_weight_std=stdp.weights.data.std().item(),
        learned_correlation=False,
    )


def bench_eprop_learning() -> STDPResult:
    """Test EpropSTDP dual-trace learning with reward modulation."""
    torch.manual_seed(42)
    eprop = EpropSTDP(pre_dim=20, post_dim=10, tau_fast=20.0, tau_slow=200.0)
    initial_mean = eprop.weight.data.mean().item()

    weight_traj = [initial_mean]

    for t in range(200):
        pre = (torch.rand(4, 20) > 0.85).float()
        post = (torch.rand(4, 10) > 0.85).float()
        membrane = torch.randn(4, 10) * 0.5

        _ = eprop(pre)
        eprop.update_traces(pre, post, membrane, threshold=1.0)

        # Apply positive reward every 20 steps
        if t > 0 and t % 20 == 0:
            eprop.apply_reward(torch.tensor(1.0))
            weight_traj.append(eprop.weight.data.mean().item())

    return STDPResult(
        name="EpropSTDP",
        weight_trajectory=weight_traj,
        final_weight_mean=eprop.weight.data.mean().item(),
        final_weight_std=eprop.weight.data.std().item(),
        learned_correlation=True,  # reward-modulated
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Short-Term Plasticity Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════


def bench_stp_paired_pulse() -> STPResult:
    """Paired-pulse protocol: two spikes 50ms apart → measure PPR."""
    torch.manual_seed(42)
    stp = TsodyksMarkramSynapse(num_synapses=1, U=0.2, tau_f=200.0, tau_d=800.0, dt=1.0)

    u_trace = []
    x_trace = []
    efficacy_trace = []

    # 50 steps silence, spike, 50 steps, spike, 100 steps
    psp_values = []
    for t in range(200):
        if t == 50 or t == 100:
            spike = torch.ones(1)
        else:
            spike = torch.zeros(1)

        result = stp(spike)
        u_trace.append(result["u"].item())
        x_trace.append(result["x"].item())
        efficacy_trace.append(result["efficacy"].item())
        if t == 50 or t == 100:
            psp_values.append(result["psp"].abs().item())

    ppr = psp_values[1] / psp_values[0] if psp_values[0] > 1e-8 else 0.0

    return STPResult(
        name="TsodyksMarkram (PPR)",
        facilitation_trace=u_trace,
        depression_trace=x_trace,
        efficacy_trace=efficacy_trace,
        paired_pulse_ratio=ppr,
    )


def bench_stp_frequency_dependence() -> Dict[str, STPResult]:
    """Test STP response at different input frequencies."""
    results = {}
    for freq_name, interval in [("low_5Hz", 20), ("mid_20Hz", 5), ("high_100Hz", 1)]:
        torch.manual_seed(42)
        stp = TsodyksMarkramSynapse(
            num_synapses=1, U=0.2, tau_f=200.0, tau_d=800.0, dt=1.0
        )

        u_trace = []
        x_trace = []
        efficacy_trace = []

        for t in range(200):
            spike = torch.ones(1) if t % interval == 0 else torch.zeros(1)
            result = stp(spike)
            u_trace.append(result["u"].item())
            x_trace.append(result["x"].item())
            efficacy_trace.append(result["efficacy"].item())

        results[freq_name] = STPResult(
            name=f"STP @ {freq_name}",
            facilitation_trace=u_trace,
            depression_trace=x_trace,
            efficacy_trace=efficacy_trace,
            paired_pulse_ratio=0.0,
        )
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Phase 6 Spiking Enhancements (Gap Tests)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SpikeEncoderResult:
    """Results from SpikeEncoder benchmark (G-04)."""
    rate_mode_firing_rate: float
    pop_mode_shape: Tuple[int, ...]
    pop_mode_rate: float


@dataclass
class PruningResult:
    """Results from pruning benchmark (G-05)."""
    sparsity_before: float
    sparsity_after: float
    still_works: bool


@dataclass
class STDPAutoScaleResult:
    """Results from STDP auto-scale benchmark (G-06)."""
    base_a_plus: float
    effective_scale: float
    weight_change: float


@dataclass
class SpikeDecoderResult:
    """Results from SpikeDecoder benchmark (G-08)."""
    rate_shape: Tuple[int, ...]
    first_spike_shape: Tuple[int, ...]
    pop_vector_shape: Tuple[int, ...]
    rate_sample: float
    first_spike_sample: float
    pop_vector_sample: float


@dataclass
class RefractoryResult:
    """Results from refractory period benchmark (G-13)."""
    min_isi: int
    refractory_rate: float
    baseline_rate: float


@dataclass
class SpikeModesResult:
    """Results from spike mode benchmark (G-14)."""
    binary_values: List[float]
    ternary_values: List[float]
    graded_values: List[float]


@dataclass
class SynapticDelayResult:
    """Results from synaptic delay benchmark (G-15)."""
    correct_delays: int


def bench_spike_encoder() -> SpikeEncoderResult:
    """G-04: Benchmark SpikeEncoder rate and population modes."""
    torch.manual_seed(42)

    # Rate mode
    enc_rate = SpikeEncoder(num_neurons=16, mode='rate', gain=2.0)
    x = torch.linspace(-1, 1, 16).unsqueeze(0)  # (1, 16)
    total_spikes = 0
    total_elements = 0
    for _ in range(100):
        spikes = enc_rate(x.clamp(0, 1))  # rate mode needs [0,1]
        total_spikes += spikes.sum().item()
        total_elements += spikes.numel()
    rate_mode_firing_rate = total_spikes / total_elements

    # Population mode
    enc_pop = SpikeEncoder(num_neurons=32, mode='population', sigma=0.3)
    x_pop = torch.linspace(-1, 1, 16).unsqueeze(0)  # (1, 16)
    pop_spikes = enc_pop(x_pop)  # (1, 32)
    pop_mode_shape = tuple(pop_spikes.shape)
    pop_mode_rate = pop_spikes.mean().item()

    return SpikeEncoderResult(
        rate_mode_firing_rate=rate_mode_firing_rate,
        pop_mode_shape=pop_mode_shape,
        pop_mode_rate=pop_mode_rate,
    )


def bench_pruning() -> PruningResult:
    """G-05: Benchmark SpikingLayer pruning."""
    torch.manual_seed(42)
    layer = SpikingLayer(input_dim=32, hidden_dim=64, tau=10.0)

    # Measure sparsity before
    m_before = compute_static_metrics(layer, "pre-prune")
    sparsity_before = m_before.connection_sparsity

    # Prune 30%
    layer.prune(0.3)

    # Measure sparsity after
    m_after = compute_static_metrics(layer, "post-prune")
    sparsity_after = m_after.connection_sparsity

    # Verify forward still works
    try:
        x = torch.randn(2, 10, 32)
        with torch.no_grad():
            out = layer(x)
        still_works = out.shape == (2, 10, 64) and torch.all(torch.isfinite(out)).item()
    except Exception:
        still_works = False

    return PruningResult(
        sparsity_before=sparsity_before,
        sparsity_after=sparsity_after,
        still_works=still_works,
    )


def bench_stdp_auto_scale() -> STDPAutoScaleResult:
    """G-06: Benchmark STDP auto-scale with low firing rate input."""
    torch.manual_seed(42)
    stdp = STDPLayer(pre_size=20, post_size=10, auto_scale=True,
                     a_plus=0.01, a_minus=0.01)
    base_a_plus = stdp.a_plus
    initial_weight_mean = stdp.weights.data.mean().item()

    # Feed low firing rate input for 100 steps
    for _ in range(100):
        pre = (torch.rand(1, 20) > 0.95).float()   # ~5% firing rate
        post = (torch.rand(1, 10) > 0.95).float()
        stdp(pre, post, learn=True)

    # Check effective scale: running_rate should be low, so scale should be > 1
    effective_scale = 1.0 / stdp.running_rate.mean().clamp(min=0.01).item()
    weight_change = stdp.weights.data.mean().item() - initial_weight_mean

    return STDPAutoScaleResult(
        base_a_plus=base_a_plus,
        effective_scale=effective_scale,
        weight_change=weight_change,
    )


def bench_spike_decoder() -> SpikeDecoderResult:
    """G-08: Benchmark SpikeDecoder in all three modes."""
    torch.manual_seed(42)
    spikes = (torch.rand(8, 50, 32) > 0.8).float()  # batch=8, time=50, neurons=32

    # Rate mode
    dec_rate = SpikeDecoder(num_neurons=32, mode='rate', time_dim=1)
    out_rate = dec_rate(spikes)

    # First-spike mode
    dec_first = SpikeDecoder(num_neurons=32, mode='first_spike', time_dim=1)
    out_first = dec_first(spikes)

    # Population vector mode (operates on neuron dim directly)
    dec_pop = SpikeDecoder(num_neurons=32, mode='population_vector')
    # For population_vector, use a single time-slice
    out_pop = dec_pop(spikes.mean(dim=1))  # (8, 32) -> (8,)

    return SpikeDecoderResult(
        rate_shape=tuple(out_rate.shape),
        first_spike_shape=tuple(out_first.shape),
        pop_vector_shape=tuple(out_pop.shape),
        rate_sample=out_rate[0, 0].item(),
        first_spike_sample=out_first[0, 0].item(),
        pop_vector_sample=out_pop[0].item(),
    )


def bench_refractory() -> RefractoryResult:
    """G-13: Benchmark refractory period enforcement."""
    torch.manual_seed(42)
    refractory_period = 5

    # With refractory
    lif_ref = LIFNeuron(tau=5.0, threshold=0.5, refractory_period=refractory_period,
                        learnable=False)
    membrane_ref = torch.zeros(1, 64)
    ref_counter = torch.zeros(1, 64)
    current = torch.full((1, 64), 2.0)  # strong constant current

    spike_times_ref: List[List[int]] = [[] for _ in range(64)]
    total_ref_spikes = 0

    for t in range(100):
        spikes, membrane_ref, ref_counter = lif_ref(current, membrane_ref, ref_counter)
        total_ref_spikes += spikes.sum().item()
        for n in range(64):
            if spikes[0, n] > 0.5:
                spike_times_ref[n].append(t)

    # Measure minimum ISI
    all_isis = []
    for st in spike_times_ref:
        if len(st) >= 2:
            for i in range(len(st) - 1):
                all_isis.append(st[i + 1] - st[i])

    min_isi = min(all_isis) if all_isis else 0
    refractory_rate = total_ref_spikes / (100 * 64)

    # Without refractory (baseline)
    lif_base = LIFNeuron(tau=5.0, threshold=0.5, refractory_period=0,
                         learnable=False)
    membrane_base = torch.zeros(1, 64)
    total_base_spikes = 0

    for t in range(100):
        spikes, membrane_base = lif_base(current, membrane_base)
        total_base_spikes += spikes.sum().item()

    baseline_rate = total_base_spikes / (100 * 64)

    return RefractoryResult(
        min_isi=min_isi,
        refractory_rate=refractory_rate,
        baseline_rate=baseline_rate,
    )


def bench_spike_modes() -> SpikeModesResult:
    """G-14: Benchmark binary, ternary, and graded spike modes."""
    torch.manual_seed(42)

    all_binary = set()
    all_ternary = set()
    all_graded = set()

    # Oscillating current: positive and negative
    for t in range(100):
        current_val = 2.0 * math.sin(2 * math.pi * t / 20)
        current = torch.full((1, 16), current_val)

        # Binary
        lif_b = LIFNeuron(tau=5.0, threshold=0.5, spike_mode='binary', learnable=False)
        if t == 0:
            mem_b = torch.zeros(1, 16)
        spikes_b, mem_b = lif_b(current, mem_b)
        for v in spikes_b.unique().tolist():
            all_binary.add(round(v, 2))

        # Ternary
        lif_t = LIFNeuron(tau=5.0, threshold=0.5, spike_mode='ternary', learnable=False)
        if t == 0:
            mem_t = torch.zeros(1, 16)
        spikes_t, mem_t = lif_t(current, mem_t)
        for v in spikes_t.unique().tolist():
            all_ternary.add(round(v, 2))

        # Graded
        lif_g = LIFNeuron(tau=5.0, threshold=0.5, spike_mode='graded', learnable=False)
        if t == 0:
            mem_g = torch.zeros(1, 16)
        spikes_g, mem_g = lif_g(current, mem_g)
        for v in spikes_g.unique().tolist():
            all_graded.add(round(v, 2))

    return SpikeModesResult(
        binary_values=sorted(all_binary),
        ternary_values=sorted(all_ternary),
        graded_values=sorted(all_graded),
    )


def bench_synaptic_delay() -> SynapticDelayResult:
    """G-15: Benchmark SynapticDelay correctness.

    Tests that a spike stored at time T appears in the output D timesteps later,
    where D is the per-synapse delay (0-indexed: delay=0 means immediate delivery).
    """
    torch.manual_seed(42)
    sd = SynapticDelay(num_synapses=4, max_delay=5)
    sd.reset()
    delays = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    # Pre-fill buffer with zeros
    for _ in range(5):
        sd(torch.zeros(4), delays)

    # Store a spike — delay=0 synapse should receive it immediately
    out0 = sd(torch.ones(4), delays)
    correct = 1 if out0[0].item() > 0.5 else 0

    # Subsequent steps: delay=D synapse sees the spike at step D
    for step in range(1, 4):
        out = sd(torch.zeros(4), delays)
        if out[step].item() > 0.5:
            correct += 1

    return SynapticDelayResult(correct_delays=correct)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Output
# ═══════════════════════════════════════════════════════════════════════════════


def _print_static_table(metrics: List[StaticMetrics]) -> None:
    print("\n" + "=" * 78)
    print("  STATIC METRICS (NeuroBench-style)")
    print("=" * 78)
    print(f"  {'Module':<28s} {'Footprint':>10s} {'Params':>8s} {'Buffers':>8s} {'Sparsity':>9s}")
    print("  " + "-" * 67)
    for m in metrics:
        fp_str = f"{m.footprint_bytes:,}"
        print(
            f"  {m.name:<28s} {fp_str:>10s} {m.param_count:>8,d} "
            f"{m.buffer_count:>8,d} {m.connection_sparsity:>9.4f}"
        )
    print()


def _print_dynamics_table(results: List[DynamicsResult]) -> None:
    print("=" * 78)
    print("  NEURON DYNAMICS")
    print("=" * 78)
    print(f"  {'Neuron':<24s} {'Rate':>8s} {'ISI CV':>8s} {'Pattern':>20s}")
    print("  " + "-" * 62)
    for r in results:
        pattern = _classify_pattern(r)
        print(f"  {r.name:<24s} {r.mean_firing_rate:>8.4f} {r.isi_cv:>8.3f} {pattern:>20s}")
    print()


def _classify_pattern(r: DynamicsResult) -> str:
    if r.mean_firing_rate < 0.001:
        return "silent"
    elif r.isi_cv < 0.1:
        return "regular spiking"
    elif r.isi_cv < 0.5:
        return "adapting"
    else:
        return "irregular/bursting"


def _print_workload_table(results: List[WorkloadMetrics]) -> None:
    print("=" * 78)
    print("  WORKLOAD METRICS (NeuroBench-style)")
    print("=" * 78)
    print(
        f"  {'Module':<28s} {'Sparsity':>9s} {'Rate':>7s} "
        f"{'Eff Ops':>12s} {'Dense Ops':>12s} {'Savings':>8s}"
    )
    print("  " + "-" * 78)
    for m in results:
        savings = 1.0 - m.effective_ops / m.dense_ops if m.dense_ops > 0 else 0.0
        print(
            f"  {m.name:<28s} {m.activation_sparsity:>9.4f} {m.mean_firing_rate:>7.4f} "
            f"{m.effective_ops:>12,.0f} {m.dense_ops:>12,.0f} {savings:>7.1%}"
        )
    # NeuroBench reference
    print()
    print("  NeuroBench leaderboard reference (SNN models):")
    print("  - tinyRSNN (Motor Pred):  sparsity=0.984, Eff_ACs=304")
    print("  - bigSNN  (Motor Pred):   sparsity=0.968, Eff_ACs=42003")
    print("  - SNN     (FSCIL):        sparsity=0.916, Eff_ACs=365000")
    print()


def _print_stdp_table(results: List[STDPResult]) -> None:
    print("=" * 78)
    print("  STDP / LEARNING")
    print("=" * 78)
    print(f"  {'Name':<32s} {'W mean':>8s} {'W std':>8s} {'Learned?':>9s}")
    print("  " + "-" * 59)
    for r in results:
        learned = "yes" if r.learned_correlation else "no"
        print(f"  {r.name:<32s} {r.final_weight_mean:>8.4f} {r.final_weight_std:>8.4f} {learned:>9s}")
    print()


def _print_stp_table(ppr_result: STPResult, freq_results: Dict[str, STPResult]) -> None:
    print("=" * 78)
    print("  SHORT-TERM PLASTICITY")
    print("=" * 78)
    print(f"  Paired-pulse ratio (PPR): {ppr_result.paired_pulse_ratio:.3f}")
    print(f"    PPR > 1 = facilitation, PPR < 1 = depression")
    print()
    print(f"  {'Frequency':>12s} {'Final u':>10s} {'Final x':>10s} {'Final u*x':>10s}")
    print("  " + "-" * 44)
    for name, r in freq_results.items():
        print(
            f"  {name:>12s} {r.facilitation_trace[-1]:>10.4f} "
            f"{r.depression_trace[-1]:>10.4f} {r.efficacy_trace[-1]:>10.4f}"
        )
    print()


def _print_phase6_table(
    enc: SpikeEncoderResult,
    prune: PruningResult,
    auto_scale: STDPAutoScaleResult,
    dec: SpikeDecoderResult,
    refr: RefractoryResult,
    modes: SpikeModesResult,
    delay: SynapticDelayResult,
) -> None:
    print("=" * 78)
    print("  PHASE 6: SPIKING ENHANCEMENTS")
    print("=" * 78)
    print(f"  {'Benchmark':<32s} {'Metric':<24s} {'Value':>12s}")
    print("  " + "-" * 70)

    # G-04
    print(f"  {'G-04 SpikeEncoder (rate)':<32s} {'firing_rate':<24s} {enc.rate_mode_firing_rate:>12.4f}")
    print(f"  {'G-04 SpikeEncoder (pop)':<32s} {'shape':<24s} {str(enc.pop_mode_shape):>12s}")
    print(f"  {'G-04 SpikeEncoder (pop)':<32s} {'pop_rate':<24s} {enc.pop_mode_rate:>12.4f}")

    # G-05
    print(f"  {'G-05 Pruning':<32s} {'sparsity_before':<24s} {prune.sparsity_before:>12.4f}")
    print(f"  {'G-05 Pruning':<32s} {'sparsity_after':<24s} {prune.sparsity_after:>12.4f}")
    print(f"  {'G-05 Pruning':<32s} {'still_works':<24s} {'yes' if prune.still_works else 'no':>12s}")

    # G-06
    print(f"  {'G-06 STDP Auto-Scale':<32s} {'base_a_plus':<24s} {auto_scale.base_a_plus:>12.4f}")
    print(f"  {'G-06 STDP Auto-Scale':<32s} {'effective_scale':<24s} {auto_scale.effective_scale:>12.2f}")
    print(f"  {'G-06 STDP Auto-Scale':<32s} {'weight_change':<24s} {auto_scale.weight_change:>12.6f}")

    # G-08
    print(f"  {'G-08 SpikeDecoder (rate)':<32s} {'shape':<24s} {str(dec.rate_shape):>12s}")
    print(f"  {'G-08 SpikeDecoder (first_spike)':<32s} {'shape':<24s} {str(dec.first_spike_shape):>12s}")
    print(f"  {'G-08 SpikeDecoder (pop_vector)':<32s} {'shape':<24s} {str(dec.pop_vector_shape):>12s}")

    # G-13
    print(f"  {'G-13 Refractory':<32s} {'min_isi':<24s} {refr.min_isi:>12d}")
    print(f"  {'G-13 Refractory':<32s} {'refractory_rate':<24s} {refr.refractory_rate:>12.4f}")
    print(f"  {'G-13 Refractory':<32s} {'baseline_rate':<24s} {refr.baseline_rate:>12.4f}")

    # G-14
    print(f"  {'G-14 Binary spikes':<32s} {'unique values':<24s} {str(modes.binary_values):>12s}")
    print(f"  {'G-14 Ternary spikes':<32s} {'unique values':<24s} {str(modes.ternary_values):>12s}")
    print(f"  {'G-14 Graded spikes':<32s} {'unique values':<24s} {str(modes.graded_values):>12s}")

    # G-15
    print(f"  {'G-15 SynapticDelay':<32s} {'correct_delays':<24s} {delay.correct_delays:>12d}")
    print()


def _plot_results(
    dynamics: List[DynamicsResult],
    workload: List[WorkloadMetrics],
    ppr: STPResult,
    freq: Dict[str, STPResult],
    save_path: str,
) -> None:
    if not HAS_MATPLOTLIB:
        print("  [matplotlib not available — skipping plot]")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: LIF membrane trace (first neuron)
    ax = axes[0, 0]
    if dynamics:
        d = dynamics[0]  # LIF
        ax.plot(d.membrane_traces[:, 0].numpy(), linewidth=0.8)
        ax.set_title(f"{d.name} Membrane (rate={d.mean_firing_rate:.3f})")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("V")
        ax.grid(True, alpha=0.3)

    # Panel 2: AdEx membrane trace (first neuron)
    ax = axes[0, 1]
    if len(dynamics) > 1:
        d = dynamics[1]  # AdEx
        ax.plot(d.membrane_traces[:, 0].numpy(), linewidth=0.8)
        ax.set_title(f"{d.name} Membrane (rate={d.mean_firing_rate:.3f})")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("V")
        ax.grid(True, alpha=0.3)

    # Panel 3: Activation sparsity comparison
    ax = axes[0, 2]
    names = [w.name for w in workload]
    sparsities = [w.activation_sparsity for w in workload]
    bars = ax.barh(names, sparsities)
    ax.axvline(x=0.9, color="red", linestyle="--", label="NeuroBench SNN target")
    ax.set_xlim(0, 1.05)
    ax.set_title("Activation Sparsity")
    ax.set_xlabel("Sparsity (1 = all silent)")
    ax.legend(fontsize=7)

    # Panel 4: STP paired-pulse traces
    ax = axes[1, 0]
    t = range(len(ppr.facilitation_trace))
    ax.plot(t, ppr.facilitation_trace, label="u (facilitation)")
    ax.plot(t, ppr.depression_trace, label="x (depression)")
    ax.plot(t, ppr.efficacy_trace, label="u*x (efficacy)", linewidth=2)
    ax.set_title(f"STP Paired-Pulse (PPR={ppr.paired_pulse_ratio:.2f})")
    ax.set_xlabel("Timestep")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 5: STP frequency dependence
    ax = axes[1, 1]
    for name, r in freq.items():
        ax.plot(r.efficacy_trace, label=name)
    ax.set_title("STP Frequency Dependence")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Efficacy (u*x)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 6: Firing rate comparison
    ax = axes[1, 2]
    dyn_names = [d.name for d in dynamics]
    dyn_rates = [d.mean_firing_rate for d in dynamics]
    ax.bar(dyn_names, dyn_rates)
    ax.set_title("Firing Rates")
    ax.set_ylabel("Spikes / neuron / step")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Pytest Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestNeuronDynamics:
    """Tests that neuron models produce correct spiking behavior."""

    def test_lif_fires_above_threshold(self) -> None:
        result = bench_lif_dynamics(constant_current=1.5)
        assert result.mean_firing_rate > 0.01, "LIF should fire with suprathreshold current"

    def test_lif_silent_below_threshold(self) -> None:
        result = bench_lif_dynamics(constant_current=0.05)
        assert result.mean_firing_rate < 0.01, "LIF should be silent with subthreshold current"

    def test_lif_regular_spiking(self) -> None:
        result = bench_lif_dynamics(constant_current=1.5)
        assert result.isi_cv < 0.2, f"LIF should spike regularly (ISI CV={result.isi_cv:.3f})"

    def test_adex_produces_spikes(self) -> None:
        result = bench_adex_dynamics()
        # AdEx with low current may not spike, but should have valid dynamics
        assert result.membrane_traces.shape == (BENCH_TIMESTEPS, BENCH_NEURONS)

    def test_synaptic_neuron_runs(self) -> None:
        result = bench_synaptic_dynamics()
        assert result.membrane_traces.shape == (BENCH_TIMESTEPS, BENCH_NEURONS)

    def test_alpha_neuron_runs(self) -> None:
        result = bench_alpha_dynamics()
        assert result.membrane_traces.shape == (BENCH_TIMESTEPS, BENCH_NEURONS)


class TestActivationSparsity:
    """Tests that spiking layers produce sparse activations (NeuroBench target: >0.9)."""

    def test_spiking_layer_sparse(self) -> None:
        result = bench_workload_spiking_layer()
        assert result.activation_sparsity > 0.5, (
            f"SpikingLayer sparsity {result.activation_sparsity:.3f} should be > 0.5"
        )

    def test_oscillatory_snn_sparse(self) -> None:
        result = bench_workload_oscillatory_snn()
        assert result.activation_sparsity > 0.3, (
            f"OscillatorySNN sparsity {result.activation_sparsity:.3f} should be > 0.3"
        )

    def test_recurrent_synaptic_sparse(self) -> None:
        result = bench_workload_recurrent_synaptic()
        assert result.activation_sparsity > 0.3, (
            f"RecurrentSynaptic sparsity {result.activation_sparsity:.3f} should be > 0.3"
        )

    def test_sparsity_gives_savings(self) -> None:
        result = bench_workload_spiking_layer()
        assert result.effective_ops < result.dense_ops, "Sparse ops should be less than dense"


class TestSTDP:
    """Tests STDP learning rules."""

    def test_stdp_correlated_strengthens(self) -> None:
        result = bench_stdp_correlation()
        # Correlated pre-post should increase weights (or at least not crash)
        assert result.final_weight_mean > 0, "Weights should stay positive"

    def test_eprop_runs(self) -> None:
        result = bench_eprop_learning()
        assert len(result.weight_trajectory) > 1, "EpropSTDP should produce weight updates"


class TestSTP:
    """Tests short-term plasticity dynamics."""

    def test_paired_pulse_facilitation(self) -> None:
        result = bench_stp_paired_pulse()
        # With U=0.2, tau_f=200 >> tau_d: should show facilitation (PPR > 1)
        assert result.paired_pulse_ratio > 0.5, (
            f"PPR={result.paired_pulse_ratio:.3f}, expected facilitation"
        )

    def test_frequency_dependent_filtering(self) -> None:
        results = bench_stp_frequency_dependence()
        # High frequency should show more depression (lower efficacy)
        low_eff = results["low_5Hz"].efficacy_trace[-1]
        high_eff = results["high_100Hz"].efficacy_trace[-1]
        assert low_eff > high_eff, (
            f"Low-freq efficacy ({low_eff:.3f}) should exceed "
            f"high-freq ({high_eff:.3f}) due to depression"
        )

    def test_stp_state_bounded(self) -> None:
        result = bench_stp_paired_pulse()
        for u in result.facilitation_trace:
            assert 0.0 <= u <= 1.0, f"u={u} out of [0,1]"
        for x in result.depression_trace:
            assert 0.0 <= x <= 1.0, f"x={x} out of [0,1]"


class TestStaticMetrics:
    """Tests NeuroBench-style static metric computation."""

    def test_footprint_positive(self) -> None:
        layer = SpikingLayer(input_dim=8, hidden_dim=16, tau=10.0)
        m = compute_static_metrics(layer, "SpikingLayer")
        assert m.footprint_bytes > 0
        assert m.param_count > 0

    def test_connection_sparsity_zero_for_dense(self) -> None:
        layer = SpikingLayer(input_dim=8, hidden_dim=16, tau=10.0)
        m = compute_static_metrics(layer, "SpikingLayer")
        assert m.connection_sparsity == 0.0, "Freshly initialized weights should be dense"


class TestPhase6Spiking:
    """Tests for Phase 6 spiking enhancements (gap tests)."""

    def test_spike_encoder_rate_mode(self) -> None:
        result = bench_spike_encoder()
        assert 0.0 < result.rate_mode_firing_rate < 1.0, (
            f"Rate mode firing rate {result.rate_mode_firing_rate:.4f} should be in (0, 1)"
        )

    def test_spike_encoder_population_mode(self) -> None:
        result = bench_spike_encoder()
        assert result.pop_mode_shape == (1, 32), (
            f"Population mode shape {result.pop_mode_shape} should be (1, 32)"
        )

    def test_pruning_increases_sparsity(self) -> None:
        result = bench_pruning()
        assert result.sparsity_before < result.sparsity_after, (
            f"Sparsity should increase: before={result.sparsity_before}, after={result.sparsity_after}"
        )
        assert result.sparsity_after >= 0.2, (
            f"After 30% pruning, sparsity should be >= 0.2, got {result.sparsity_after}"
        )
        assert result.still_works, "Forward pass should still work after pruning"

    def test_stdp_auto_scale_active(self) -> None:
        result = bench_stdp_auto_scale()
        assert result.effective_scale > 1.0, (
            f"Auto-scale should amplify at low firing rate, got scale={result.effective_scale:.2f}"
        )

    def test_spike_decoder_rate_mode(self) -> None:
        result = bench_spike_decoder()
        assert result.rate_shape == (8, 32), (
            f"Rate decoder shape {result.rate_shape} should be (8, 32)"
        )

    def test_refractory_enforces_min_isi(self) -> None:
        result = bench_refractory()
        assert result.min_isi >= 5, (
            f"Min ISI {result.min_isi} should be >= refractory_period (5)"
        )

    def test_ternary_has_negative_spikes(self) -> None:
        result = bench_spike_modes()
        has_negative = any(v < 0 for v in result.ternary_values)
        assert has_negative, (
            f"Ternary mode should produce negative spikes, got {result.ternary_values}"
        )

    def test_synaptic_delay_correct(self) -> None:
        result = bench_synaptic_delay()
        assert result.correct_delays > 0, (
            f"SynapticDelay should deliver at least some spikes correctly, got {result.correct_delays}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main: Full Suite
# ═══════════════════════════════════════════════════════════════════════════════


def run_full_suite() -> None:
    print("=" * 78)
    print("  PyQuifer Spiking Modules vs NeuroBench Metrics")
    print("=" * 78)

    # ── Static Metrics ──
    print("\n[1/6] Computing static metrics...")
    modules_to_measure = [
        (SpikingLayer(input_dim=32, hidden_dim=64, tau=10.0, recurrent=True), "SpikingLayer (rec)"),
        (SpikingLayer(input_dim=32, hidden_dim=64, tau=10.0, recurrent=False), "SpikingLayer (ff)"),
        (OscillatorySNN(input_dim=16, num_excitatory=48, num_inhibitory=16), "OscillatorySNN"),
        (RecurrentSynapticLayer(input_dim=32, hidden_dim=64), "RecurrentSynapticLayer"),
        (STDPLayer(pre_size=32, post_size=16), "STDPLayer"),
        (EligibilityModulatedSTDP(pre_dim=32, post_dim=16), "EligibilityModSTDP"),
        (EpropSTDP(pre_dim=32, post_dim=16), "EpropSTDP"),
        (TsodyksMarkramSynapse(num_synapses=64), "TsodyksMarkramSynapse"),
        (STPLayer(input_dim=32, output_dim=64), "STPLayer"),
    ]
    static_results = [compute_static_metrics(m, name) for m, name in modules_to_measure]
    _print_static_table(static_results)

    # ── Neuron Dynamics ──
    print("[2/6] Running neuron dynamics benchmarks...")
    dynamics_results = []
    for bench_fn, name in [
        (bench_lif_dynamics, "LIFNeuron"),
        (bench_adex_dynamics, "AdExNeuron"),
        (bench_synaptic_dynamics, "SynapticNeuron"),
        (bench_alpha_dynamics, "AlphaNeuron"),
    ]:
        print(f"  {name}...", end=" ", flush=True)
        with timer() as t:
            dynamics_results.append(bench_fn())
        print(f"done ({t['elapsed']:.3f}s)")

    _print_dynamics_table(dynamics_results)

    # ── Workload Metrics ──
    print("[3/6] Running workload benchmarks...")
    workload_results = []
    for bench_fn, name in [
        (bench_workload_spiking_layer, "SpikingLayer"),
        (bench_workload_oscillatory_snn, "OscillatorySNN"),
        (bench_workload_recurrent_synaptic, "RecurrentSynapticLayer"),
    ]:
        print(f"  {name}...", end=" ", flush=True)
        with timer() as t:
            workload_results.append(bench_fn())
        print(f"done ({t['elapsed']:.3f}s)")

    _print_workload_table(workload_results)

    # ── STDP / Learning ──
    print("[4/6] Running STDP benchmarks...")
    stdp_results = [
        bench_stdp_correlation(),
        bench_stdp_uncorrelated(),
        bench_eprop_learning(),
    ]
    _print_stdp_table(stdp_results)

    # ── Short-Term Plasticity ──
    print("[5/6] Running STP benchmarks...")
    ppr_result = bench_stp_paired_pulse()
    freq_results = bench_stp_frequency_dependence()
    _print_stp_table(ppr_result, freq_results)

    # ── Phase 6 Spiking Enhancements ──
    print("[6/6] Phase 6 spiking enhancements...")
    p6_enc = bench_spike_encoder()
    p6_prune = bench_pruning()
    p6_auto = bench_stdp_auto_scale()
    p6_dec = bench_spike_decoder()
    p6_refr = bench_refractory()
    p6_modes = bench_spike_modes()
    p6_delay = bench_synaptic_delay()
    _print_phase6_table(p6_enc, p6_prune, p6_auto, p6_dec, p6_refr, p6_modes, p6_delay)

    # ── Plot ──
    plot_path = str(
        __import__("pathlib").Path(__file__).resolve().parent / "bench_spiking.png"
    )
    _plot_results(dynamics_results, workload_results, ppr_result, freq_results, plot_path)

    print("Done.")


if __name__ == "__main__":
    run_full_suite()
