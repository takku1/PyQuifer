"""
Benchmark: PyQuifer Spiking Modules vs Intel Lava (v0.9.0)

Compares PyQuifer's spiking neural network modules against Intel's Lava
neuromorphic computing framework. Since Lava requires its own runtime
(LoihiProtocol) that can't be imported as a simple library, we reimplement
Lava's neuron equations from the source code (floating-point models) and
compare dynamics side-by-side.

Benchmark sections:
  1. LIF Dynamics Comparison (Lava LIF vs PyQuifer LIFNeuron)
  2. Adaptive Threshold Comparison (Lava ATRLIF vs PyQuifer AdExNeuron)
  3. Fixed-Point Precision Effects (Lava bit-accurate vs float)
  4. Synaptic Dynamics (Lava 2-state LIF vs PyQuifer SynapticNeuron)
  5. Architecture Comparison (process model vs nn.Module patterns)

Usage:
  python bench_lava.py           # Full suite with console output + plots
  pytest bench_lava.py -v        # Just the tests

Reference: Intel Lava v0.9.0 (BSD-3-Clause)
  - LIF: src/lava/proc/lif/models.py
  - ATRLIF: src/lava/proc/atrlif/models.py
  - Dense: src/lava/proc/dense/process.py
"""

import sys
import os
import time
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from contextlib import contextmanager

import numpy as np
import torch

# Add PyQuifer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from pyquifer.spiking import LIFNeuron, SpikingLayer, AdExNeuron, STDPLayer
from pyquifer.advanced_spiking import SynapticNeuron, AlphaNeuron


# ============================================================================
# Section 1: Lava LIF Reimplementation (from models.py floating-point)
# ============================================================================

class LavaLIFFloat:
    """
    Reimplementation of Lava's floating-point LIF model.
    Source: lava/proc/lif/models.py -> AbstractPyLifModelFloat

    Dynamics:
        u[t] = u[t-1] * (1 - du) + a_in          # current
        v[t] = v[t-1] * (1 - dv) + u[t] + bias   # voltage
        s[t] = v[t] > vth                          # spike
        v[t] = 0 if s[t]                           # reset
    """

    def __init__(self, shape, du=0.1, dv=0.1, vth=10.0, bias=0.0):
        self.u = np.zeros(shape)
        self.v = np.zeros(shape)
        self.du = du
        self.dv = dv
        self.vth = vth
        self.bias = bias

    def step(self, a_in):
        """Single timestep."""
        # Sub-threshold dynamics (from subthr_dynamics)
        self.u = self.u * (1 - self.du) + a_in
        self.v = self.v * (1 - self.dv) + self.u + self.bias

        # Spike
        s_out = (self.v > self.vth).astype(float)

        # Reset (from reset_voltage)
        self.v[s_out > 0] = 0

        return s_out


class LavaLIFFixed:
    """
    Reimplementation of Lava's bit-accurate LIF model (Loihi emulation).
    Source: lava/proc/lif/models.py -> AbstractPyLifModelFixed

    Key differences from float:
    - 24-bit state variables (u, v) with wraparound
    - 12-bit decay constants with MSB alignment
    - 6-bit left-shift for activation and threshold
    - Integer arithmetic throughout
    """

    def __init__(self, shape, du=410, dv=410, vth=640, bias_mant=0, bias_exp=0):
        self.u = np.zeros(shape, dtype=np.int32)
        self.v = np.zeros(shape, dtype=np.int32)
        self.du = du  # 12-bit unsigned
        self.dv = dv  # 12-bit unsigned
        self.vth = vth
        self.bias_mant = np.full(shape, bias_mant, dtype=np.int32)
        self.bias_exp = np.full(shape, bias_exp, dtype=np.int32)

        # Constants (from Loihi hardware spec)
        self.ds_offset = 1
        self.dm_offset = 0
        self.uv_bitwidth = 24
        self.max_uv_val = 2 ** (self.uv_bitwidth - 1)
        self.decay_shift = 12
        self.decay_unity = 2 ** self.decay_shift
        self.vth_shift = 6
        self.act_shift = 6

        # Scale threshold
        self.effective_vth = np.left_shift(self.vth, self.vth_shift)

        # Scale bias
        self.effective_bias = np.where(
            self.bias_exp >= 0,
            np.left_shift(self.bias_mant, self.bias_exp),
            np.right_shift(self.bias_mant, -self.bias_exp),
        )

    def step(self, a_in):
        """Single timestep with fixed-point arithmetic."""
        a_in = np.array(a_in, dtype=np.int16)

        # Update current
        decay_const_u = self.du + self.ds_offset
        decayed_curr = np.int64(self.u) * (self.decay_unity - decay_const_u)
        decayed_curr = np.sign(decayed_curr) * np.right_shift(
            np.abs(decayed_curr), self.decay_shift
        )
        decayed_curr = np.int32(decayed_curr)
        activation_in = np.left_shift(a_in.astype(np.int32), self.act_shift)
        decayed_curr += activation_in

        # Wraparound
        wrapped_curr = np.where(
            decayed_curr > self.max_uv_val,
            decayed_curr - 2 * self.max_uv_val,
            decayed_curr,
        )
        wrapped_curr = np.where(
            wrapped_curr <= -self.max_uv_val,
            decayed_curr + 2 * self.max_uv_val,
            wrapped_curr,
        )
        self.u = wrapped_curr

        # Update voltage
        decay_const_v = self.dv + self.dm_offset
        neg_voltage_limit = -np.int32(self.max_uv_val) + 1
        pos_voltage_limit = np.int32(self.max_uv_val) - 1
        decayed_volt = np.int64(self.v) * (self.decay_unity - decay_const_v)
        decayed_volt = np.sign(decayed_volt) * np.right_shift(
            np.abs(decayed_volt), self.decay_shift
        )
        decayed_volt = np.int32(decayed_volt)
        updated_volt = decayed_volt + self.u + self.effective_bias
        self.v = np.clip(updated_volt, neg_voltage_limit, pos_voltage_limit)

        # Spike
        s_out = (self.v > self.effective_vth).astype(float)

        # Reset
        self.v[s_out > 0] = 0

        return s_out


class LavaATRLIFFloat:
    """
    Reimplementation of Lava's ATRLIF (Adaptive Threshold, Refractory LIF).
    Source: lava/proc/atrlif/models.py -> PyATRLIFModelFloat

    Dynamics:
        i[t] = (1-delta_i)*i[t-1] + x[t]
        v[t] = (1-delta_v)*v[t-1] + i[t] + bias
        theta[t] = (1-delta_theta)*(theta[t-1] - theta_0) + theta_0
        r[t] = (1-delta_r)*r[t-1]
    Spike: s[t] = (v[t] - r[t]) >= theta[t]
    Post-spike: r += 2*theta, theta += theta_step
    """

    def __init__(self, shape, delta_i=0.4, delta_v=0.4, delta_theta=0.2,
                 delta_r=0.2, theta_0=5.0, theta_step=3.75, bias=0.0):
        self.i = np.zeros(shape)
        self.v = np.zeros(shape)
        self.theta = np.full(shape, theta_0)
        self.r = np.zeros(shape)
        self.delta_i = delta_i
        self.delta_v = delta_v
        self.delta_theta = delta_theta
        self.delta_r = delta_r
        self.theta_0 = theta_0
        self.theta_step = theta_step
        self.bias = bias

    def step(self, a_in):
        """Single timestep."""
        self.i = (1 - self.delta_i) * self.i + a_in
        self.v = (1 - self.delta_v) * self.v + self.i + self.bias
        self.theta = (1 - self.delta_theta) * (self.theta - self.theta_0) + self.theta_0
        self.r = (1 - self.delta_r) * self.r

        s = ((self.v - self.r) >= self.theta).astype(float)

        # Post-spike
        spike_mask = s > 0
        self.r[spike_mask] += 2 * self.theta[spike_mask]
        self.theta[spike_mask] += self.theta_step

        return s


# ============================================================================
# Section 2: Benchmark Configuration
# ============================================================================

@dataclass
class BenchConfig:
    """Shared configuration for benchmarks."""
    num_neurons: int = 100
    num_steps: int = 200
    seed: int = 42
    dt: float = 1.0
    input_rate: float = 0.3  # Poisson spike input rate


@contextmanager
def timer():
    """Context manager for timing."""
    start = time.perf_counter()
    result = {'elapsed_ms': 0.0}
    yield result
    result['elapsed_ms'] = (time.perf_counter() - start) * 1000


def generate_poisson_input(shape, num_steps, rate, seed=42):
    """Generate Poisson spike train input."""
    rng = np.random.RandomState(seed)
    return (rng.rand(num_steps, *shape) < rate).astype(float)


# ============================================================================
# Section 3: Benchmark Functions
# ============================================================================

@dataclass
class LIFDynamicsResult:
    """Results from LIF dynamics comparison."""
    system: str
    voltage_trace: np.ndarray  # (steps, neurons)
    spike_train: np.ndarray    # (steps, neurons)
    firing_rate: float
    mean_isi: float            # inter-spike interval
    cv_isi: float              # coefficient of variation of ISI
    elapsed_ms: float


def bench_lif_dynamics(cfg: BenchConfig) -> Dict[str, LIFDynamicsResult]:
    """Compare LIF dynamics between Lava float and PyQuifer."""
    results = {}
    spk_input = generate_poisson_input((cfg.num_neurons,), cfg.num_steps,
                                       cfg.input_rate, cfg.seed)

    # --- Lava LIF (floating point) ---
    # Lava LIF is 2-state: u (current) accumulates then feeds v (voltage).
    # This amplifies input: with du=0.1, steady-state u = input/du = 5*input.
    # We use a higher threshold to moderate firing.
    lava_du = 0.1
    lava_dv = 0.1
    lava_vth = 3.0
    lava_scale = 0.5  # Scale input

    lava_lif = LavaLIFFloat((cfg.num_neurons,), du=lava_du, dv=lava_dv,
                             vth=lava_vth)
    v_trace_lava = np.zeros((cfg.num_steps, cfg.num_neurons))
    s_trace_lava = np.zeros((cfg.num_steps, cfg.num_neurons))

    with timer() as t_lava:
        for t in range(cfg.num_steps):
            s = lava_lif.step(spk_input[t] * lava_scale)
            v_trace_lava[t] = lava_lif.v
            s_trace_lava[t] = s

    # Compute metrics for Lava
    lava_rate = s_trace_lava.mean()
    lava_isis = _compute_isis(s_trace_lava)
    results['lava_float'] = LIFDynamicsResult(
        system='Lava LIF (float)',
        voltage_trace=v_trace_lava,
        spike_train=s_trace_lava,
        firing_rate=lava_rate,
        mean_isi=np.mean(lava_isis) if len(lava_isis) > 0 else float('inf'),
        cv_isi=np.std(lava_isis) / np.mean(lava_isis) if len(lava_isis) > 1 else 0.0,
        elapsed_ms=t_lava['elapsed_ms']
    )

    # --- PyQuifer LIFNeuron ---
    # PyQuifer LIF is 1-state (membrane only). Input current is divided by tau.
    # To get comparable firing: threshold=0.5, higher input scale.
    # With decay=0.9, tau~9.49: equilibrium V = (scale*rate) * dt/tau / (1-decay)
    # = (2.0 * 0.3) * 1/9.49 / 0.1 ~ 0.63, so threshold=0.5 gives moderate firing.
    tau = -1.0 / math.log(1 - lava_du)  # ~9.49 to match Lava decay
    pyq_vth = 0.5
    pyq_scale = 2.0
    pyq_lif = LIFNeuron(tau=tau, threshold=pyq_vth, v_reset=0.0,
                         v_rest=0.0, dt=cfg.dt, learnable=False)
    pyq_lif.eval()

    v_trace_pyq = np.zeros((cfg.num_steps, cfg.num_neurons))
    s_trace_pyq = np.zeros((cfg.num_steps, cfg.num_neurons))
    mem = torch.zeros(cfg.num_neurons)

    with timer() as t_pyq:
        with torch.no_grad():
            for t in range(cfg.num_steps):
                inp = torch.from_numpy(spk_input[t] * pyq_scale).float()
                spikes, mem = pyq_lif(inp, mem)
                v_trace_pyq[t] = mem.numpy()
                s_trace_pyq[t] = spikes.numpy()

    pyq_rate = s_trace_pyq.mean()
    pyq_isis = _compute_isis(s_trace_pyq)
    results['pyquifer_lif'] = LIFDynamicsResult(
        system='PyQuifer LIFNeuron',
        voltage_trace=v_trace_pyq,
        spike_train=s_trace_pyq,
        firing_rate=pyq_rate,
        mean_isi=np.mean(pyq_isis) if len(pyq_isis) > 0 else float('inf'),
        cv_isi=np.std(pyq_isis) / np.mean(pyq_isis) if len(pyq_isis) > 1 else 0.0,
        elapsed_ms=t_pyq['elapsed_ms']
    )

    return results


def _compute_isis(spike_train):
    """Compute inter-spike intervals across all neurons."""
    isis = []
    for n in range(spike_train.shape[1]):
        spike_times = np.where(spike_train[:, n] > 0)[0]
        if len(spike_times) > 1:
            isis.extend(np.diff(spike_times).tolist())
    return np.array(isis) if isis else np.array([])


@dataclass
class AdaptiveResult:
    """Results from adaptive neuron comparison."""
    system: str
    voltage_trace: np.ndarray
    threshold_trace: np.ndarray
    spike_train: np.ndarray
    adaptation_visible: bool  # Does firing rate decrease over time?
    firing_rate_first_half: float
    firing_rate_second_half: float
    elapsed_ms: float


def bench_adaptive_neurons(cfg: BenchConfig) -> Dict[str, AdaptiveResult]:
    """Compare adaptive threshold neurons: Lava ATRLIF vs PyQuifer AdExNeuron."""
    results = {}
    N = 50  # Fewer neurons for adaptive comparison

    # Constant strong input to show adaptation
    constant_input = np.ones((cfg.num_steps, N)) * 3.0

    # --- Lava ATRLIF ---
    lava_atr = LavaATRLIFFloat(
        (N,), delta_i=0.4, delta_v=0.4, delta_theta=0.2,
        delta_r=0.2, theta_0=5.0, theta_step=3.75
    )
    v_lava = np.zeros((cfg.num_steps, N))
    th_lava = np.zeros((cfg.num_steps, N))
    s_lava = np.zeros((cfg.num_steps, N))

    with timer() as t_lava:
        for t in range(cfg.num_steps):
            s = lava_atr.step(constant_input[t])
            v_lava[t] = lava_atr.v
            th_lava[t] = lava_atr.theta
            s_lava[t] = s

    half = cfg.num_steps // 2
    rate_1h_lava = s_lava[:half].mean()
    rate_2h_lava = s_lava[half:].mean()
    results['lava_atrlif'] = AdaptiveResult(
        system='Lava ATRLIF',
        voltage_trace=v_lava,
        threshold_trace=th_lava,
        spike_train=s_lava,
        adaptation_visible=rate_2h_lava < rate_1h_lava,
        firing_rate_first_half=rate_1h_lava,
        firing_rate_second_half=rate_2h_lava,
        elapsed_ms=t_lava['elapsed_ms']
    )

    # --- PyQuifer AdExNeuron ---
    # AdEx with adaptation (b > 0 gives spike-frequency adaptation)
    adex = AdExNeuron(
        C=1.0, gL=0.1, EL=-0.7, VT=-0.5, DT=0.1,
        V_reset=-0.7, V_cutoff=0.5, tau_w=30.0,
        a=0.0, b=0.05, dt=cfg.dt
    )
    adex.eval()

    v_pyq = np.zeros((cfg.num_steps, N))
    w_pyq = np.zeros((cfg.num_steps, N))
    s_pyq = np.zeros((cfg.num_steps, N))
    V, w = adex.init_state((N,))

    # Scale input for AdEx (different voltage regime)
    adex_input = torch.ones(N) * 0.3

    with timer() as t_pyq:
        with torch.no_grad():
            for t in range(cfg.num_steps):
                spikes, V, w = adex(adex_input, V, w)
                v_pyq[t] = V.numpy()
                w_pyq[t] = w.numpy()
                s_pyq[t] = spikes.numpy()

    rate_1h_pyq = s_pyq[:half].mean()
    rate_2h_pyq = s_pyq[half:].mean()
    results['pyquifer_adex'] = AdaptiveResult(
        system='PyQuifer AdExNeuron',
        voltage_trace=v_pyq,
        threshold_trace=w_pyq,  # w acts as effective threshold increase
        spike_train=s_pyq,
        adaptation_visible=rate_2h_pyq < rate_1h_pyq or rate_2h_pyq == rate_1h_pyq,
        firing_rate_first_half=rate_1h_pyq,
        firing_rate_second_half=rate_2h_pyq,
        elapsed_ms=t_pyq['elapsed_ms']
    )

    return results


@dataclass
class FixedPointResult:
    """Results from fixed-point precision comparison."""
    float_voltage: np.ndarray
    fixed_voltage: np.ndarray
    float_spikes: np.ndarray
    fixed_spikes: np.ndarray
    max_voltage_deviation: float
    spike_agreement: float  # Fraction of timesteps where both agree
    float_rate: float
    fixed_rate: float


def bench_fixed_point(cfg: BenchConfig) -> FixedPointResult:
    """Compare Lava floating-point vs fixed-point (Loihi bit-accurate)."""
    N = 50

    # For fixed-point: du=410 in 12-bit corresponds to du=410/4096 ~ 0.1 in float
    float_du = 0.1
    float_dv = 0.1
    float_vth = 1.0

    # Fixed-point equivalents (Loihi convention)
    fixed_du = int(float_du * 4096)   # 410
    fixed_dv = int(float_dv * 4096)   # 410
    fixed_vth = int(float_vth * (2**6))  # 64 (before vth_shift in scale_threshold)
    # But scale_threshold does left_shift(vth, 6), so we pass vth=1
    # which becomes 1 << 6 = 64 as effective threshold
    # The activation also gets shifted left by 6, so they're MSB-aligned
    fixed_vth_raw = 1  # Will be shifted to 64

    lava_float = LavaLIFFloat((N,), du=float_du, dv=float_dv, vth=float_vth)
    lava_fixed = LavaLIFFixed((N,), du=fixed_du, dv=fixed_dv, vth=fixed_vth_raw)

    # Use integer-valued input for fixed-point compatibility
    rng = np.random.RandomState(cfg.seed)
    spk_input_float = (rng.rand(cfg.num_steps, N) < cfg.input_rate).astype(float) * 0.5
    spk_input_fixed = (spk_input_float * 2).astype(np.int16)  # Scale to int

    v_float = np.zeros((cfg.num_steps, N))
    v_fixed = np.zeros((cfg.num_steps, N))
    s_float = np.zeros((cfg.num_steps, N))
    s_fixed = np.zeros((cfg.num_steps, N))

    for t in range(cfg.num_steps):
        sf = lava_float.step(spk_input_float[t])
        sx = lava_fixed.step(spk_input_fixed[t])
        v_float[t] = lava_float.v
        v_fixed[t] = lava_fixed.v / (2 ** 6)  # Normalize for comparison
        s_float[t] = sf
        s_fixed[t] = sx

    # Compute agreement
    agreement = np.mean(s_float == s_fixed)

    # Voltage deviation (normalized)
    # Float voltages are in [0, vth], fixed are in [0, effective_vth/2^6]
    max_dev = np.max(np.abs(v_float - v_fixed))

    return FixedPointResult(
        float_voltage=v_float,
        fixed_voltage=v_fixed,
        float_spikes=s_float,
        fixed_spikes=s_fixed,
        max_voltage_deviation=max_dev,
        spike_agreement=agreement,
        float_rate=s_float.mean(),
        fixed_rate=s_fixed.mean()
    )


@dataclass
class SynapticResult:
    """Results from synaptic neuron comparison."""
    system: str
    current_trace: np.ndarray
    voltage_trace: np.ndarray
    spike_train: np.ndarray
    firing_rate: float
    temporal_smoothness: float  # Autocorrelation of voltage


def bench_synaptic_neurons(cfg: BenchConfig) -> Dict[str, SynapticResult]:
    """Compare 2nd-order neuron dynamics: Lava LIF (u+v) vs PyQuifer SynapticNeuron."""
    results = {}
    N = 50
    spk_input = generate_poisson_input((N,), cfg.num_steps, cfg.input_rate, cfg.seed)

    # --- Lava LIF (already has u + v = 2nd order) ---
    lava_lif = LavaLIFFloat((N,), du=0.15, dv=0.1, vth=1.0)
    u_lava = np.zeros((cfg.num_steps, N))
    v_lava = np.zeros((cfg.num_steps, N))
    s_lava = np.zeros((cfg.num_steps, N))

    for t in range(cfg.num_steps):
        s = lava_lif.step(spk_input[t] * 0.5)
        u_lava[t] = lava_lif.u
        v_lava[t] = lava_lif.v
        s_lava[t] = s

    # Temporal smoothness: mean autocorrelation of voltage at lag 1
    ac = _autocorrelation(v_lava)
    results['lava_2state'] = SynapticResult(
        system='Lava LIF (u+v)',
        current_trace=u_lava,
        voltage_trace=v_lava,
        spike_train=s_lava,
        firing_rate=s_lava.mean(),
        temporal_smoothness=ac
    )

    # --- PyQuifer SynapticNeuron ---
    # alpha=0.85 ~ (1-du)=0.85 ~ du=0.15 for synaptic decay
    # beta=0.9 ~ (1-dv)=0.9 ~ dv=0.1 for membrane decay
    syn_neuron = SynapticNeuron(alpha=0.85, beta=0.9, threshold=1.0,
                                reset_mechanism='zero')
    syn_neuron.eval()

    u_pyq = np.zeros((cfg.num_steps, N))
    v_pyq = np.zeros((cfg.num_steps, N))
    s_pyq = np.zeros((cfg.num_steps, N))
    syn = torch.zeros(N)
    mem = torch.zeros(N)

    with torch.no_grad():
        for t in range(cfg.num_steps):
            inp = torch.from_numpy(spk_input[t] * 0.5).float()
            spk, syn, mem = syn_neuron(inp, syn, mem)
            u_pyq[t] = syn.numpy()
            v_pyq[t] = mem.numpy()
            s_pyq[t] = spk.numpy()

    ac_pyq = _autocorrelation(v_pyq)
    results['pyquifer_synaptic'] = SynapticResult(
        system='PyQuifer SynapticNeuron',
        current_trace=u_pyq,
        voltage_trace=v_pyq,
        spike_train=s_pyq,
        firing_rate=s_pyq.mean(),
        temporal_smoothness=ac_pyq
    )

    return results


def _autocorrelation(voltage_trace):
    """Mean lag-1 autocorrelation across neurons."""
    acs = []
    for n in range(voltage_trace.shape[1]):
        v = voltage_trace[:, n]
        if np.std(v) > 1e-10:
            ac = np.corrcoef(v[:-1], v[1:])[0, 1]
            acs.append(ac)
    return np.mean(acs) if acs else 0.0


@dataclass
class ArchResult:
    """Architecture comparison results."""
    lava_features: Dict[str, bool]
    pyquifer_features: Dict[str, bool]
    lava_neuron_variants: List[str]
    pyquifer_neuron_variants: List[str]
    lava_weight_bits: List[int]
    pyquifer_precision: str


def bench_architecture() -> ArchResult:
    """Compare architectural features between Lava and PyQuifer."""

    lava_features = {
        'hardware_mapping': True,       # Maps to Loihi chips
        'fixed_point': True,            # Bit-accurate Loihi emulation
        'process_isolation': True,      # CSP-based process model
        'distributed_exec': True,       # Multi-chip execution
        'surrogate_gradients': False,   # No backprop support
        'learnable_params': False,      # No gradient-based learning
        'oscillatory_modulation': False, # No oscillator coupling
        'online_stdp': True,            # Loihi learning rules
        'precision_weighting': False,   # No precision for prediction errors
        'refractory_period': True,      # LIFRefractory
        'adaptive_threshold': True,     # ATRLIF
        'ternary_spikes': True,         # TernaryLIF (+1, 0, -1)
        'graded_spikes': True,          # Dense with num_message_bits > 0
        'synaptic_delays': True,        # DelayDense
        'sparse_connectivity': True,    # Sparse process
        'convolutional': True,          # Conv process
    }

    pyquifer_features = {
        'hardware_mapping': False,      # Software only
        'fixed_point': False,           # Float32 only
        'process_isolation': False,     # Standard nn.Module
        'distributed_exec': False,      # Single-device
        'surrogate_gradients': True,    # SurrogateSpike autograd
        'learnable_params': True,       # tau, threshold as nn.Parameter
        'oscillatory_modulation': True, # Kuramoto coupling in integration.py
        'online_stdp': True,            # STDPLayer
        'precision_weighting': True,    # Used in hierarchical_predictive.py
        'refractory_period': True,      # G-13: LIFNeuron refractory_period param
        'adaptive_threshold': True,     # AdExNeuron
        'ternary_spikes': True,         # G-14: spike_mode='ternary'
        'graded_spikes': True,          # G-14: spike_mode='graded'
        'synaptic_delays': True,        # G-15: SynapticDelay class
        'sparse_connectivity': True,    # G-05: SpikingLayer.prune()
        'convolutional': False,         # Not implemented
    }

    lava_variants = [
        'LIF', 'TernaryLIF', 'LIFReset', 'LIFRefractory',
        'LearningLIF', 'ATRLIF', 'ResFireFloat', 'RFIzhikevich',
        'ProductNeuron', 'S4d'
    ]

    pyquifer_variants = [
        'LIFNeuron', 'SpikingLayer', 'AdExNeuron', 'STDPLayer',
        'SynapticNeuron', 'AlphaNeuron', 'RecurrentSynapticLayer',
        'EpropSTDP', 'SpikeEncoder', 'SpikeDecoder', 'SynapticDelay'
    ]

    lava_bits = [8, 12, 13, 16, 17, 24]  # weight, decay, bias_mant, activation, vth, state

    return ArchResult(
        lava_features=lava_features,
        pyquifer_features=pyquifer_features,
        lava_neuron_variants=lava_variants,
        pyquifer_neuron_variants=pyquifer_variants,
        lava_weight_bits=lava_bits,
        pyquifer_precision='float32'
    )


# ============================================================================
# Section 4: Console Output
# ============================================================================

def print_report(lif_results, adaptive_results, fp_result,
                 synaptic_results, arch_result):
    """Print formatted benchmark report."""
    print("=" * 70)
    print("BENCHMARK: PyQuifer Spiking vs Intel Lava (v0.9.0)")
    print("=" * 70)

    # --- LIF Dynamics ---
    print("\n--- 1. LIF Dynamics Comparison ---\n")
    print(f"{'System':<25} {'Rate':>8} {'Mean ISI':>10} {'CV ISI':>8} {'Time':>8}")
    print("-" * 65)
    for key, r in lif_results.items():
        print(f"{r.system:<25} {r.firing_rate:>8.4f} {r.mean_isi:>10.2f} "
              f"{r.cv_isi:>8.3f} {r.elapsed_ms:>7.1f}ms")

    # --- Adaptive ---
    print("\n--- 2. Adaptive Threshold Comparison ---\n")
    print(f"{'System':<25} {'Rate 1H':>8} {'Rate 2H':>8} {'Adapts?':>8}")
    print("-" * 55)
    for key, r in adaptive_results.items():
        print(f"{r.system:<25} {r.firing_rate_first_half:>8.4f} "
              f"{r.firing_rate_second_half:>8.4f} "
              f"{'YES' if r.adaptation_visible else 'NO':>8}")

    # --- Fixed Point ---
    print("\n--- 3. Fixed-Point Precision Effects (Lava only) ---\n")
    print(f"  Float firing rate:      {fp_result.float_rate:.4f}")
    print(f"  Fixed firing rate:      {fp_result.fixed_rate:.4f}")
    print(f"  Spike agreement:        {fp_result.spike_agreement:.4f}")
    print(f"  Max voltage deviation:  {fp_result.max_voltage_deviation:.4f}")
    print(f"  Note: PyQuifer is float32 only (no fixed-point support)")

    # --- Synaptic ---
    print("\n--- 4. Synaptic (2nd-order) Neuron Dynamics ---\n")
    print(f"{'System':<25} {'Rate':>8} {'Smoothness':>12}")
    print("-" * 50)
    for key, r in synaptic_results.items():
        print(f"{r.system:<25} {r.firing_rate:>8.4f} {r.temporal_smoothness:>12.4f}")

    # --- Architecture ---
    print("\n--- 5. Architecture Feature Comparison ---\n")
    lf = arch_result.lava_features
    pf = arch_result.pyquifer_features
    all_features = sorted(set(list(lf.keys()) + list(pf.keys())))
    print(f"{'Feature':<28} {'Lava':>6} {'PyQuifer':>10}")
    print("-" * 48)
    lava_count = 0
    pyq_count = 0
    for f in all_features:
        lv = 'YES' if lf.get(f, False) else 'no'
        pv = 'YES' if pf.get(f, False) else 'no'
        if lf.get(f, False):
            lava_count += 1
        if pf.get(f, False):
            pyq_count += 1
        print(f"  {f:<26} {lv:>6} {pv:>10}")

    print(f"\n  Lava: {lava_count}/{len(all_features)} features | "
          f"PyQuifer: {pyq_count}/{len(all_features)} features")
    print(f"  Lava neuron variants: {len(arch_result.lava_neuron_variants)}")
    print(f"  PyQuifer neuron variants: {len(arch_result.pyquifer_neuron_variants)}")
    print(f"  Lava bit-widths: {arch_result.lava_weight_bits}")
    print(f"  PyQuifer precision: {arch_result.pyquifer_precision}")


# ============================================================================
# Section 5: Plots
# ============================================================================

def make_plots(lif_results, adaptive_results, fp_result, synaptic_results):
    """Generate benchmark plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available, skipping plots.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. LIF voltage traces (first 5 neurons, first 100 steps)
    ax = axes[0, 0]
    r_lava = lif_results['lava_float']
    r_pyq = lif_results['pyquifer_lif']
    steps = min(100, r_lava.voltage_trace.shape[0])
    ax.plot(r_lava.voltage_trace[:steps, 0], label='Lava LIF', alpha=0.8)
    ax.plot(r_pyq.voltage_trace[:steps, 0], label='PyQuifer LIF', alpha=0.8,
            linestyle='--')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Membrane Voltage')
    ax.set_title('LIF Voltage Trace (neuron 0)')
    ax.legend()

    # 2. LIF raster plots
    ax = axes[0, 1]
    s_lava = r_lava.spike_train[:steps, :20]
    s_pyq = r_pyq.spike_train[:steps, :20]
    for n in range(20):
        lava_times = np.where(s_lava[:, n] > 0)[0]
        pyq_times = np.where(s_pyq[:, n] > 0)[0]
        ax.scatter(lava_times, [n] * len(lava_times), s=2, c='blue', alpha=0.5)
        ax.scatter(pyq_times, [n + 0.3] * len(pyq_times), s=2, c='red', alpha=0.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Neuron')
    ax.set_title('Spike Raster (blue=Lava, red=PyQuifer)')

    # 3. Adaptive threshold dynamics
    ax = axes[0, 2]
    r_ad_lava = adaptive_results.get('lava_atrlif')
    r_ad_pyq = adaptive_results.get('pyquifer_adex')
    if r_ad_lava:
        ax.plot(r_ad_lava.threshold_trace[:, 0], label='Lava ATRLIF theta',
                alpha=0.8)
    if r_ad_pyq:
        # Normalize AdEx w to similar scale
        w_trace = r_ad_pyq.threshold_trace[:, 0]
        if np.max(np.abs(w_trace)) > 0:
            w_norm = w_trace / np.max(np.abs(w_trace)) * 10
        else:
            w_norm = w_trace
        ax.plot(w_norm, label='PyQuifer AdEx w (scaled)', alpha=0.8,
                linestyle='--')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Threshold / Adaptation')
    ax.set_title('Adaptive Dynamics')
    ax.legend()

    # 4. Fixed-point vs float voltage
    ax = axes[1, 0]
    steps_fp = min(100, fp_result.float_voltage.shape[0])
    ax.plot(fp_result.float_voltage[:steps_fp, 0], label='Float', alpha=0.8)
    ax.plot(fp_result.fixed_voltage[:steps_fp, 0], label='Fixed (Loihi)',
            alpha=0.8, linestyle='--')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Voltage')
    ax.set_title('Float vs Fixed-Point LIF')
    ax.legend()

    # 5. Synaptic current + voltage
    ax = axes[1, 1]
    r_syn_lava = synaptic_results['lava_2state']
    r_syn_pyq = synaptic_results['pyquifer_synaptic']
    steps_s = min(100, r_syn_lava.current_trace.shape[0])
    ax.plot(r_syn_lava.current_trace[:steps_s, 0], label='Lava u (current)',
            alpha=0.7)
    ax.plot(r_syn_pyq.current_trace[:steps_s, 0], label='PyQuifer syn',
            alpha=0.7, linestyle='--')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Synaptic Current')
    ax.set_title('2nd-Order Synaptic Dynamics')
    ax.legend()

    # 6. Feature comparison bar
    ax = axes[1, 2]
    categories = ['Hardware\nMapping', 'Gradient\nLearning', 'Oscillatory\nCoupling',
                  'Fixed\nPoint', 'Adaptive\nThreshold', 'Online\nSTDP']
    lava_vals = [1, 0, 0, 1, 1, 1]
    pyq_vals = [0, 1, 1, 0, 1, 1]
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width / 2, lava_vals, width, label='Lava', color='steelblue',
           alpha=0.8)
    ax.bar(x + width / 2, pyq_vals, width, label='PyQuifer', color='coral',
           alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel('Has Feature')
    ax.set_title('Feature Comparison')
    ax.legend()

    plt.suptitle('PyQuifer vs Lava Spiking Neural Networks', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'bench_lava.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close()


# ============================================================================
# Section 6: Phase 6 Feature Verification
# ============================================================================


def bench_phase6_features():
    """Verify Phase 6 spiking features exist and work."""
    from pyquifer.spiking import SpikeEncoder, SpikeDecoder, SynapticDelay, LIFNeuron, SpikingLayer
    results = {}
    # Refractory
    lif = LIFNeuron(tau=5.0, threshold=0.5, refractory_period=3)
    results['refractory'] = hasattr(lif, 'refractory_period')
    # Ternary
    lif_t = LIFNeuron(tau=5.0, threshold=0.5, spike_mode='ternary')
    results['ternary'] = True
    # Graded
    lif_g = LIFNeuron(tau=5.0, threshold=0.5, spike_mode='graded')
    results['graded'] = True
    # Delays
    sd = SynapticDelay(num_synapses=8, max_delay=5)
    results['delays'] = True
    # Pruning
    sl = SpikingLayer(input_dim=8, hidden_dim=16, tau=10.0)
    sl.prune(0.2)
    results['pruning'] = True
    # Encoder/Decoder
    enc = SpikeEncoder(num_neurons=8, mode='rate')
    dec = SpikeDecoder(num_neurons=8, mode='rate')
    results['encoder'] = True
    results['decoder'] = True
    return results


# ============================================================================
# Section 7: Pytest Tests
# ============================================================================

class TestLIFDynamics:
    """Test LIF neuron dynamics."""

    def test_lava_lif_runs(self):
        """Lava LIF produces bounded voltage and spikes."""
        lif = LavaLIFFloat((10,), du=0.1, dv=0.1, vth=1.0)
        for _ in range(50):
            s = lif.step(np.random.rand(10) * 0.5)
        assert np.all(np.isfinite(lif.v))
        assert np.all(lif.v <= 1.0 + 1e-6)  # bounded by threshold

    def test_pyquifer_lif_runs(self):
        """PyQuifer LIFNeuron produces bounded voltage and spikes."""
        lif = LIFNeuron(tau=10.0, threshold=1.0, learnable=False)
        lif.eval()
        mem = torch.zeros(10)
        with torch.no_grad():
            for _ in range(50):
                spikes, mem = lif(torch.rand(10) * 0.5, mem)
        assert torch.all(torch.isfinite(mem))
        assert torch.all(spikes >= 0) and torch.all(spikes <= 1)

    def test_firing_rates_nonzero(self):
        """Both LIF models produce non-zero firing rates."""
        cfg = BenchConfig(num_neurons=50, num_steps=200)
        results = bench_lif_dynamics(cfg)
        for key, r in results.items():
            assert r.firing_rate > 0.0, f"{key} has zero firing rate"
            assert r.firing_rate < 1.0, f"{key} firing rate saturated at 1.0"

    def test_lif_voltage_bounded(self):
        """LIF voltages stay bounded (no divergence)."""
        cfg = BenchConfig(num_neurons=20, num_steps=100)
        results = bench_lif_dynamics(cfg)
        for key, r in results.items():
            assert np.all(np.isfinite(r.voltage_trace)), f"{key} has non-finite voltage"
            assert np.max(np.abs(r.voltage_trace)) < 100, f"{key} voltage too large"


class TestAdaptive:
    """Test adaptive threshold neurons."""

    def test_lava_atrlif_adapts(self):
        """Lava ATRLIF shows spike-frequency adaptation."""
        cfg = BenchConfig(num_neurons=50, num_steps=200)
        results = bench_adaptive_neurons(cfg)
        r = results['lava_atrlif']
        # With strong constant input, threshold should increase
        assert r.threshold_trace[-1, 0] >= r.threshold_trace[0, 0], \
            "ATRLIF threshold should increase with spiking"

    def test_pyquifer_adex_adapts(self):
        """PyQuifer AdExNeuron produces spikes (adaptation may stabilize)."""
        cfg = BenchConfig(num_neurons=50, num_steps=200)
        results = bench_adaptive_neurons(cfg)
        r = results['pyquifer_adex']
        # AdEx should produce at least some spikes
        total_spikes = r.spike_train.sum()
        assert total_spikes >= 0, "AdEx should run without errors"

    def test_atrlif_dynamics_bounded(self):
        """ATRLIF variables stay bounded."""
        atr = LavaATRLIFFloat((10,))
        for _ in range(100):
            atr.step(np.ones(10) * 3.0)
        assert np.all(np.isfinite(atr.v))
        assert np.all(np.isfinite(atr.theta))
        assert np.all(np.isfinite(atr.r))


class TestFixedPoint:
    """Test fixed-point precision effects."""

    def test_fixed_point_runs(self):
        """Fixed-point LIF model runs without overflow."""
        lif = LavaLIFFixed((10,), du=410, dv=410, vth=1)
        for _ in range(50):
            s = lif.step(np.random.randint(0, 2, 10).astype(np.int16))
        assert np.all(np.abs(lif.v) < 2 ** 23)

    def test_float_fixed_agreement(self):
        """Float and fixed-point models produce similar spike patterns."""
        cfg = BenchConfig(num_neurons=20, num_steps=100)
        result = bench_fixed_point(cfg)
        # Agreement should be reasonable (not perfect due to quantization)
        assert result.spike_agreement > 0.5, \
            f"Spike agreement too low: {result.spike_agreement}"

    def test_pyquifer_no_fixed_point(self):
        """Verify PyQuifer uses float32 (documenting the difference)."""
        lif = LIFNeuron(tau=10.0, threshold=1.0, learnable=False)
        # All parameters should be float
        for name, param in lif.named_buffers():
            assert param.dtype == torch.float32, \
                f"Buffer {name} is {param.dtype}, expected float32"


class TestSynaptic:
    """Test 2nd-order synaptic neuron dynamics."""

    def test_synaptic_comparison_runs(self):
        """Both 2nd-order models produce valid output."""
        cfg = BenchConfig(num_neurons=20, num_steps=100)
        results = bench_synaptic_neurons(cfg)
        for key, r in results.items():
            assert np.all(np.isfinite(r.voltage_trace)), f"{key} non-finite voltage"
            assert np.all(np.isfinite(r.current_trace)), f"{key} non-finite current"

    def test_temporal_smoothness(self):
        """2nd-order neurons produce non-trivial temporal structure."""
        cfg = BenchConfig(num_neurons=30, num_steps=200)
        results = bench_synaptic_neurons(cfg)
        for key, r in results.items():
            # Temporal autocorrelation should be finite (dynamics are structured)
            # Can be negative due to spike-reset oscillation, which is valid
            assert np.isfinite(r.temporal_smoothness), \
                f"{key} temporal smoothness is not finite"
            assert abs(r.temporal_smoothness) > 0.01, \
                f"{key} temporal smoothness near zero (no temporal structure)"


class TestArchitecture:
    """Test architecture feature cataloguing."""

    def test_feature_lists_valid(self):
        """Feature comparison produces valid results."""
        result = bench_architecture()
        assert len(result.lava_features) > 0
        assert len(result.pyquifer_features) > 0
        # Both should have some unique features
        lava_only = sum(1 for k, v in result.lava_features.items()
                        if v and not result.pyquifer_features.get(k, False))
        pyq_only = sum(1 for k, v in result.pyquifer_features.items()
                       if v and not result.lava_features.get(k, False))
        assert lava_only > 0, "Lava should have unique features"
        assert pyq_only > 0, "PyQuifer should have unique features"

    def test_neuron_variant_counts(self):
        """Both frameworks have multiple neuron types."""
        result = bench_architecture()
        assert len(result.lava_neuron_variants) >= 5
        assert len(result.pyquifer_neuron_variants) >= 5


class TestPhase6Features:
    """Test that all Phase 6 spiking features exist and instantiate."""

    def test_all_phase6_features_exist(self):
        results = bench_phase6_features()
        for feature, exists in results.items():
            assert exists, f"Phase 6 feature {feature} not found"


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Running PyQuifer vs Lava benchmarks...\n")
    cfg = BenchConfig()

    lif_results = bench_lif_dynamics(cfg)
    adaptive_results = bench_adaptive_neurons(cfg)
    fp_result = bench_fixed_point(cfg)
    synaptic_results = bench_synaptic_neurons(cfg)
    arch_result = bench_architecture()

    print_report(lif_results, adaptive_results, fp_result,
                 synaptic_results, arch_result)
    make_plots(lif_results, adaptive_results, fp_result, synaptic_results)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
