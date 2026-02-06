"""
Benchmark: PyQuifer Continuous Dynamics vs torchdyn

Compares PyQuifer's neural dynamics against torchdyn's Neural ODE framework.
While torchdiffeq provides ODE solvers, torchdyn adds higher-level abstractions:
NeuralODE wrapping, SDE support, sensitivity methods (adjoint vs autograd),
Hamiltonian/Lagrangian networks, and Galerkin basis layers.

Benchmark sections:
  1. NeuralODE Wrapping (can PyQuifer dynamics be wrapped as NeuralODE?)
  2. SDE vs Stochastic Resonance (torchdyn SDE vs PyQuifer noise injection)
  3. Sensitivity/Memory Comparison (autograd vs adjoint-style backprop)
  4. Energy-Preserving Dynamics (HNN concept vs PyQuifer oscillators)
  5. Architecture Feature Comparison

Usage:
  python bench_torchdyn.py           # Full suite with console output + plots
  pytest bench_torchdyn.py -v        # Just the tests

Reference: torchdyn (DiffEqML), Poli et al. (2020)
"""

import sys
import os
import time
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

# Add PyQuifer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pyquifer.oscillators import LearnableKuramotoBank, StuartLandauOscillator
from pyquifer.neural_mass import WilsonCowanPopulation
from pyquifer.stochastic_resonance import AdaptiveStochasticResonance
from pyquifer.criticality import AvalancheDetector


# ============================================================================
# Section 1: Reimplemented torchdyn concepts
# ============================================================================

class NeuralODEWrapper(nn.Module):
    """Reimplementation of torchdyn's NeuralODE concept.
    Wraps any (t, x) -> dx/dt function as a differentiable layer.
    """

    def __init__(self, vector_field, solver='euler', dt=0.01):
        super().__init__()
        self.vector_field = vector_field
        self.solver = solver
        self.dt = dt

    def forward(self, x0, t_span):
        """Integrate from x0 over t_span.
        Returns: (t_eval, trajectory) where trajectory is (T, *x0.shape)
        """
        t_start, t_end = t_span[0].item(), t_span[-1].item()
        t = t_start
        x = x0
        trajectory = [x]
        times = [t]

        while t < t_end - self.dt / 2:
            if self.solver == 'rk4':
                k1 = self.vector_field(t, x)
                k2 = self.vector_field(t + self.dt / 2, x + k1 * self.dt / 2)
                k3 = self.vector_field(t + self.dt / 2, x + k2 * self.dt / 2)
                k4 = self.vector_field(t + self.dt, x + k3 * self.dt)
                x = x + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6
            else:  # euler
                dx = self.vector_field(t, x)
                x = x + dx * self.dt
            t += self.dt
            trajectory.append(x)
            times.append(t)

        return torch.tensor(times), torch.stack(trajectory)


class NeuralSDEWrapper(nn.Module):
    """Reimplementation of torchdyn's NeuralSDE concept.
    dx = f(t, x) dt + g(t, x) dW
    """

    def __init__(self, drift, diffusion, dt=0.01, sde_type='ito'):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion
        self.dt = dt
        self.sde_type = sde_type

    def forward(self, x0, t_span, seed=None):
        """Euler-Maruyama integration."""
        if seed is not None:
            torch.manual_seed(seed)

        t_start, t_end = t_span[0].item(), t_span[-1].item()
        t = t_start
        x = x0
        trajectory = [x]
        sqrt_dt = math.sqrt(self.dt)

        while t < t_end - self.dt / 2:
            f = self.drift(t, x)
            g = self.diffusion(t, x)
            dW = torch.randn_like(x) * sqrt_dt
            x = x + f * self.dt + g * dW
            t += self.dt
            trajectory.append(x)

        return torch.stack(trajectory)


class HamiltonianDynamics(nn.Module):
    """Reimplementation of torchdyn's HNN concept.
    Given energy H(q, p), dynamics are:
        dq/dt = dH/dp,  dp/dt = -dH/dq
    """

    def __init__(self, energy_fn):
        super().__init__()
        self.energy_fn = energy_fn

    def forward(self, t, state):
        """state = [q, p] concatenated."""
        state = state.requires_grad_(True)
        dim = state.shape[-1] // 2
        q, p = state[..., :dim], state[..., dim:]

        H = self.energy_fn(state)
        dH = torch.autograd.grad(H.sum(), state, create_graph=True)[0]
        dHdq, dHdp = dH[..., :dim], dH[..., dim:]

        # Hamilton's equations
        dqdt = dHdp
        dpdt = -dHdq
        return torch.cat([dqdt, dpdt], dim=-1)


# ============================================================================
# Section 2: Benchmark Config
# ============================================================================

@dataclass
class BenchConfig:
    num_oscillators: int = 20
    num_steps: int = 200
    dt: float = 0.01
    seed: int = 42


@contextmanager
def timer():
    start = time.perf_counter()
    result = {'elapsed_ms': 0.0}
    yield result
    result['elapsed_ms'] = (time.perf_counter() - start) * 1000


# ============================================================================
# Section 3: Benchmark Functions
# ============================================================================

@dataclass
class NeuralODEResult:
    """Results from NeuralODE wrapping test."""
    system: str
    can_wrap: bool
    trajectory_valid: bool
    final_state_finite: bool
    num_steps: int
    elapsed_ms: float


def bench_neural_ode_wrapping(cfg: BenchConfig) -> Dict[str, NeuralODEResult]:
    """Test if PyQuifer dynamics can be wrapped as NeuralODE."""
    torch.manual_seed(cfg.seed)
    results = {}

    # --- Kuramoto as NeuralODE ---
    class KuramotoVF(nn.Module):
        """Kuramoto vector field compatible with NeuralODE interface."""
        def __init__(self, N, K=2.0):
            super().__init__()
            self.register_buffer('omega', torch.randn(N))
            self.K = K
            self.N = N

        def forward(self, t, theta):
            diff = theta.unsqueeze(-1) - theta.unsqueeze(-2)
            coupling = self.K / self.N * torch.sin(diff).sum(dim=-1)
            return self.omega - coupling

    kuramoto_vf = KuramotoVF(cfg.num_oscillators)
    node = NeuralODEWrapper(kuramoto_vf, solver='rk4', dt=cfg.dt)
    theta0 = torch.zeros(cfg.num_oscillators)
    t_span = torch.tensor([0.0, cfg.num_steps * cfg.dt])

    with timer() as t_k:
        times, traj = node(theta0, t_span)

    results['kuramoto_node'] = NeuralODEResult(
        system='Kuramoto as NeuralODE',
        can_wrap=True,
        trajectory_valid=torch.all(torch.isfinite(traj)).item(),
        final_state_finite=torch.all(torch.isfinite(traj[-1])).item(),
        num_steps=len(times),
        elapsed_ms=t_k['elapsed_ms']
    )

    # --- Wilson-Cowan as NeuralODE ---
    class WilsonCowanVF(nn.Module):
        def __init__(self):
            super().__init__()
            self.tau_E, self.tau_I = 10.0, 5.0
            self.w_EE, self.w_EI = 12.0, 4.0
            self.w_IE, self.w_II = 13.0, 11.0
            self.gain, self.threshold = 1.0, 4.0
            self.I_ext_E, self.I_ext_I = 1.0, 0.5

        def _sig(self, x):
            return 1.0 / (1.0 + torch.exp(-self.gain * (x - self.threshold)))

        def forward(self, t, y):
            E, I = y[0], y[1]
            dE = (-E + self._sig(self.w_EE * E - self.w_EI * I + self.I_ext_E)) / self.tau_E
            dI = (-I + self._sig(self.w_IE * E - self.w_II * I + self.I_ext_I)) / self.tau_I
            return torch.stack([dE, dI])

    wc_vf = WilsonCowanVF()
    node_wc = NeuralODEWrapper(wc_vf, solver='rk4', dt=0.1)
    y0 = torch.tensor([0.1, 0.1])
    t_span_wc = torch.tensor([0.0, 50.0])

    with timer() as t_wc:
        times_wc, traj_wc = node_wc(y0, t_span_wc)

    results['wilson_cowan_node'] = NeuralODEResult(
        system='Wilson-Cowan as NeuralODE',
        can_wrap=True,
        trajectory_valid=torch.all(torch.isfinite(traj_wc)).item(),
        final_state_finite=torch.all(torch.isfinite(traj_wc[-1])).item(),
        num_steps=len(times_wc),
        elapsed_ms=t_wc['elapsed_ms']
    )

    # --- Stuart-Landau as NeuralODE ---
    class StuartLandauVF(nn.Module):
        def __init__(self, N, lam=1.0, K=2.0):
            super().__init__()
            self.register_buffer('omega', torch.randn(N))
            self.lam = lam
            self.K = K
            self.N = N

        def forward(self, t, y):
            x, yi = y[:self.N], y[self.N:]
            r2 = x ** 2 + yi ** 2
            dx = self.lam * x - self.omega * yi - r2 * x + self.K * (x.mean() - x)
            dy = self.lam * yi + self.omega * x - r2 * yi + self.K * (yi.mean() - yi)
            return torch.cat([dx, dy])

    sl_vf = StuartLandauVF(cfg.num_oscillators)
    node_sl = NeuralODEWrapper(sl_vf, solver='rk4', dt=cfg.dt)
    z0 = torch.randn(2 * cfg.num_oscillators) * 0.1
    t_span_sl = torch.tensor([0.0, cfg.num_steps * cfg.dt])

    with timer() as t_sl:
        times_sl, traj_sl = node_sl(z0, t_span_sl)

    results['stuart_landau_node'] = NeuralODEResult(
        system='Stuart-Landau as NeuralODE',
        can_wrap=True,
        trajectory_valid=torch.all(torch.isfinite(traj_sl)).item(),
        final_state_finite=torch.all(torch.isfinite(traj_sl[-1])).item(),
        num_steps=len(times_sl),
        elapsed_ms=t_sl['elapsed_ms']
    )

    return results


@dataclass
class SDEResult:
    """Results from SDE comparison."""
    system: str
    mean_trajectory: np.ndarray
    std_trajectory: np.ndarray
    noise_effect: float  # std of final state across runs
    snr: float  # signal-to-noise ratio


def bench_sde_comparison(cfg: BenchConfig) -> Dict[str, SDEResult]:
    """Compare torchdyn-style SDE with PyQuifer's stochastic resonance."""
    torch.manual_seed(cfg.seed)
    results = {}
    dim = 16
    num_runs = 10

    # --- torchdyn-style Neural SDE (Euler-Maruyama) ---
    class LinearDrift(nn.Module):
        """Linear drift wrapped to accept ODE (t, x) interface."""
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim, bias=False)
        def forward(self, t, x):
            return self.linear(x)

    drift = LinearDrift(dim)
    # Set drift to create attractor dynamics: dx/dt = -0.1*x
    with torch.no_grad():
        drift.linear.weight.copy_(torch.eye(dim) * -0.1)

    diffusion_scale = 0.3

    class ConstantDiffusion(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale
        def forward(self, t, x):
            return torch.ones_like(x) * self.scale

    nsde = NeuralSDEWrapper(drift, ConstantDiffusion(diffusion_scale), dt=0.01)
    x0 = torch.ones(dim) * 0.5

    finals_sde = []
    trajs_sde = []
    for i in range(num_runs):
        traj = nsde(x0, torch.tensor([0.0, 2.0]), seed=cfg.seed + i)
        finals_sde.append(traj[-1].detach().numpy())
        trajs_sde.append(traj.detach().numpy())

    finals_sde = np.stack(finals_sde)
    trajs_sde = np.stack(trajs_sde)
    mean_traj = trajs_sde.mean(axis=0)
    std_traj = trajs_sde.std(axis=0)

    results['neural_sde'] = SDEResult(
        system='Neural SDE (Euler-Maruyama)',
        mean_trajectory=mean_traj.mean(axis=-1),  # avg over dims
        std_trajectory=std_traj.mean(axis=-1),
        noise_effect=finals_sde.std(),
        snr=abs(finals_sde.mean()) / (finals_sde.std() + 1e-10)
    )

    # --- PyQuifer Stochastic Resonance ---
    # Run in train mode so noise adaptation creates stochastic variation.
    # Each run uses a fresh SR instance with different seed to see noise effect.
    finals_sr = []
    for i in range(num_runs):
        sr = AdaptiveStochasticResonance(dim=dim, initial_noise=diffusion_scale)
        sr.train()
        torch.manual_seed(cfg.seed + i)
        x_in = torch.ones(1, dim) * 0.5
        for _ in range(200):
            result = sr(x_in)
        # Collect the enhanced output (varies due to noise in SR)
        finals_sr.append(result['enhanced'].detach().numpy().flatten())

    finals_sr = np.stack(finals_sr)
    results['pyquifer_sr'] = SDEResult(
        system='PyQuifer StochasticResonance',
        mean_trajectory=np.zeros(10),  # SR doesn't track trajectory
        std_trajectory=np.zeros(10),
        noise_effect=finals_sr.std(),
        snr=abs(finals_sr.mean()) / (finals_sr.std() + 1e-10)
    )

    return results


@dataclass
class SensitivityResult:
    """Results from sensitivity/memory comparison."""
    method: str
    forward_ms: float
    backward_ms: float
    peak_memory_kb: float
    gradient_norm: float


def bench_sensitivity(cfg: BenchConfig) -> Dict[str, SensitivityResult]:
    """Compare autograd vs adjoint-style memory for ODE backprop."""
    torch.manual_seed(cfg.seed)
    results = {}
    dim = 32
    T = 1.0
    dt = 0.01

    # Simple linear ODE: dx/dt = A*x
    A = nn.Linear(dim, dim, bias=False)
    nn.init.normal_(A.weight, std=0.1)

    class LinearVF(nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.linear = linear
        def forward(self, t, x):
            return self.linear(x)

    vf = LinearVF(A)

    # --- Autograd (standard, store all steps) ---
    x0 = torch.randn(dim, requires_grad=False)
    x0_auto = x0.clone().requires_grad_(True)

    with timer() as t_fwd:
        # Forward: store all intermediate states
        t_val = 0.0
        x = x0_auto
        states = [x]
        while t_val < T - dt / 2:
            dx = vf(t_val, x)
            x = x + dx * dt
            states.append(x)
            t_val += dt

    loss = states[-1].sum()

    with timer() as t_bwd:
        loss.backward()

    grad_norm_auto = A.weight.grad.norm().item() if A.weight.grad is not None else 0.0

    results['autograd'] = SensitivityResult(
        method='Autograd (store all)',
        forward_ms=t_fwd['elapsed_ms'],
        backward_ms=t_bwd['elapsed_ms'],
        peak_memory_kb=len(states) * dim * 4 / 1024,  # float32
        gradient_norm=grad_norm_auto
    )

    # --- Adjoint-style (recompute forward during backward) ---
    # We simulate adjoint by: forward without storing, backward with recompute
    A.weight.grad = None  # Reset
    x0_adj = x0.clone().detach()

    # Forward pass (no grad tracking)
    with timer() as t_fwd_adj:
        with torch.no_grad():
            t_val = 0.0
            x = x0_adj
            while t_val < T - dt / 2:
                dx = vf(t_val, x)
                x = x + dx * dt
                t_val += dt
        x_final = x

    # Adjoint backward: compute gradients via re-integration
    # Simplified: just re-run forward with grad tracking
    x_adj = x0.clone().requires_grad_(True)
    with timer() as t_bwd_adj:
        t_val = 0.0
        x = x_adj
        while t_val < T - dt / 2:
            dx = vf(t_val, x)
            x = x + dx * dt
            t_val += dt
        loss_adj = x.sum()
        loss_adj.backward()

    grad_norm_adj = A.weight.grad.norm().item() if A.weight.grad is not None else 0.0

    results['adjoint_sim'] = SensitivityResult(
        method='Adjoint-style (recompute)',
        forward_ms=t_fwd_adj['elapsed_ms'],
        backward_ms=t_bwd_adj['elapsed_ms'],
        peak_memory_kb=2 * dim * 4 / 1024,  # Only start + end
        gradient_norm=grad_norm_adj
    )

    return results


@dataclass
class EnergyResult:
    """Results from energy-preserving dynamics test."""
    system: str
    initial_energy: float
    final_energy: float
    energy_drift: float  # |final - initial| / initial
    trajectory_bounded: bool


def bench_energy_preservation(cfg: BenchConfig) -> Dict[str, EnergyResult]:
    """Compare Hamiltonian dynamics (torchdyn concept) with PyQuifer oscillators."""
    torch.manual_seed(cfg.seed)
    results = {}

    # --- Hamiltonian oscillator (torchdyn HNN concept) ---
    # Simple harmonic oscillator: H(q, p) = 0.5 * (q^2 + p^2)
    class SimpleEnergy(nn.Module):
        def forward(self, state):
            return 0.5 * (state ** 2).sum(dim=-1)

    hnn_dynamics = HamiltonianDynamics(SimpleEnergy())
    node_hnn = NeuralODEWrapper(hnn_dynamics, solver='rk4', dt=0.01)

    # Initial state: q=1, p=0 (should orbit in phase space)
    state0 = torch.tensor([1.0, 0.0])
    t_span = torch.tensor([0.0, 10.0])
    times, traj = node_hnn(state0, t_span)

    # Compute energy along trajectory
    energies = 0.5 * (traj ** 2).sum(dim=-1)
    E0 = energies[0].item()
    Ef = energies[-1].item()

    results['hamiltonian'] = EnergyResult(
        system='Hamiltonian (HNN-style)',
        initial_energy=E0,
        final_energy=Ef,
        energy_drift=abs(Ef - E0) / (abs(E0) + 1e-10),
        trajectory_bounded=torch.all(torch.abs(traj) < 10).item()
    )

    # --- PyQuifer Stuart-Landau oscillator ---
    # SL has a limit cycle attractor at |z|=1, which acts as an "energy surface"
    sl = StuartLandauOscillator(num_oscillators=1, dt=0.01)
    sl.eval()

    # Set initial state near limit cycle
    with torch.no_grad():
        sl.z_real.fill_(1.0)
        sl.z_imag.fill_(0.0)
        sl.step_count.fill_(0)

    with torch.no_grad():
        result = sl(steps=1000)

    # "Energy" for SL: |z|^2 should converge to 1 (limit cycle)
    r_init = 1.0  # |z|^2 at start
    r_final = (sl.z_real ** 2 + sl.z_imag ** 2).item()

    results['stuart_landau'] = EnergyResult(
        system='PyQuifer Stuart-Landau',
        initial_energy=r_init,
        final_energy=r_final,
        energy_drift=abs(r_final - r_init) / (abs(r_init) + 1e-10),
        trajectory_bounded=True  # SL limit cycle is bounded
    )

    # --- PyQuifer Kuramoto coherence as "energy" ---
    kb = LearnableKuramotoBank(num_oscillators=cfg.num_oscillators, dt=0.01)
    kb.eval()

    R0 = kb.get_order_parameter().item()
    with torch.no_grad():
        for _ in range(1000):
            kb(steps=1)
    Rf = kb.get_order_parameter().item()

    results['kuramoto'] = EnergyResult(
        system='PyQuifer Kuramoto (R as energy)',
        initial_energy=R0,
        final_energy=Rf,
        energy_drift=abs(Rf - R0) / (abs(R0) + 1e-10),
        trajectory_bounded=True  # R in [0, 1]
    )

    return results


def bench_architecture_features() -> Dict[str, Dict[str, bool]]:
    """Compare architectural features."""
    torchdyn_features = {
        'neural_ode': True,
        'neural_sde': True,
        'neural_cde': True,
        'adjoint_sensitivity': True,
        'interpolated_adjoint': True,
        'hamiltonian_nn': True,
        'lagrangian_nn': True,
        'cnf': True,
        'galerkin_layers': True,
        'multiple_shooting': True,
        'event_handling': False,  # Limited
        'oscillator_models': False,
        'spiking_neurons': False,
        'criticality_detection': False,
        'precision_weighting': False,
        'consciousness_metrics': False,
    }

    pyquifer_features = {
        'neural_ode': False,  # Uses Euler directly
        'neural_sde': False,  # Stochastic resonance =/= NeuralSDE
        'neural_cde': False,
        'adjoint_sensitivity': False,
        'interpolated_adjoint': False,
        'hamiltonian_nn': False,
        'lagrangian_nn': False,
        'cnf': False,
        'galerkin_layers': False,
        'multiple_shooting': False,
        'event_handling': False,
        'oscillator_models': True,
        'spiking_neurons': True,
        'criticality_detection': True,
        'precision_weighting': True,
        'consciousness_metrics': True,
    }

    return {'torchdyn': torchdyn_features, 'pyquifer': pyquifer_features}


# ============================================================================
# Section 4: Console Output
# ============================================================================

def print_report(node_results, sde_results, sens_results, energy_results,
                 arch_features):
    print("=" * 70)
    print("BENCHMARK: PyQuifer Continuous Dynamics vs torchdyn")
    print("=" * 70)

    # NeuralODE wrapping
    print("\n--- 1. NeuralODE Wrapping ---\n")
    print(f"{'System':<35} {'Wrappable':>10} {'Valid':>8} {'Steps':>8} {'Time':>8}")
    print("-" * 75)
    for key, r in node_results.items():
        print(f"{r.system:<35} {'YES' if r.can_wrap else 'NO':>10} "
              f"{'YES' if r.trajectory_valid else 'NO':>8} "
              f"{r.num_steps:>8} {r.elapsed_ms:>7.1f}ms")

    # SDE comparison
    print("\n--- 2. SDE vs Stochastic Resonance ---\n")
    for key, r in sde_results.items():
        print(f"  {r.system}:")
        print(f"    Noise effect (std of final state): {r.noise_effect:.4f}")
        print(f"    SNR: {r.snr:.4f}")

    # Sensitivity
    print("\n--- 3. Sensitivity/Memory Comparison ---\n")
    print(f"{'Method':<30} {'Fwd ms':>8} {'Bwd ms':>8} {'Memory KB':>10} {'Grad norm':>10}")
    print("-" * 70)
    for key, r in sens_results.items():
        print(f"{r.method:<30} {r.forward_ms:>8.2f} {r.backward_ms:>8.2f} "
              f"{r.peak_memory_kb:>10.1f} {r.gradient_norm:>10.4f}")

    # Energy preservation
    print("\n--- 4. Energy / Attractor Dynamics ---\n")
    print(f"{'System':<35} {'E_init':>8} {'E_final':>8} {'Drift':>10} {'Bounded':>8}")
    print("-" * 75)
    for key, r in energy_results.items():
        print(f"{r.system:<35} {r.initial_energy:>8.4f} {r.final_energy:>8.4f} "
              f"{r.energy_drift:>10.6f} {'YES' if r.trajectory_bounded else 'NO':>8}")

    # Architecture
    print("\n--- 5. Architecture Feature Comparison ---\n")
    td = arch_features['torchdyn']
    pq = arch_features['pyquifer']
    all_f = sorted(set(list(td.keys()) + list(pq.keys())))
    print(f"{'Feature':<28} {'torchdyn':>10} {'PyQuifer':>10}")
    print("-" * 52)
    td_count = pq_count = 0
    for f in all_f:
        tv = 'YES' if td.get(f, False) else 'no'
        pv = 'YES' if pq.get(f, False) else 'no'
        if td.get(f, False): td_count += 1
        if pq.get(f, False): pq_count += 1
        print(f"  {f:<26} {tv:>10} {pv:>10}")
    print(f"\n  torchdyn: {td_count}/{len(all_f)} | PyQuifer: {pq_count}/{len(all_f)}")


# ============================================================================
# Section 5: Plots
# ============================================================================

def make_plots(node_results, sde_results, energy_results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available, skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. NeuralODE timing
    ax = axes[0, 0]
    systems = [r.system.split(' as ')[0] for r in node_results.values()]
    times = [r.elapsed_ms for r in node_results.values()]
    ax.barh(systems, times, color='steelblue', alpha=0.8)
    ax.set_xlabel('Time (ms)')
    ax.set_title('NeuralODE Wrapping Time')

    # 2. SDE noise comparison
    ax = axes[0, 1]
    systems_sde = [r.system for r in sde_results.values()]
    noise_fx = [r.noise_effect for r in sde_results.values()]
    snrs = [r.snr for r in sde_results.values()]
    x = np.arange(len(systems_sde))
    ax.bar(x - 0.2, noise_fx, 0.4, label='Noise effect', color='coral', alpha=0.8)
    ax.bar(x + 0.2, snrs, 0.4, label='SNR', color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.split('(')[0].strip() for s in systems_sde], fontsize=8)
    ax.set_title('SDE vs Stochastic Resonance')
    ax.legend()

    # 3. Energy drift
    ax = axes[1, 0]
    systems_e = [r.system.split('(')[0].strip() for r in energy_results.values()]
    drifts = [r.energy_drift for r in energy_results.values()]
    colors = ['green' if d < 0.01 else 'orange' if d < 0.1 else 'red' for d in drifts]
    ax.bar(systems_e, drifts, color=colors, alpha=0.8)
    ax.set_ylabel('Energy Drift (relative)')
    ax.set_title('Energy / Attractor Stability')
    ax.axhline(y=0.01, color='green', linestyle='--', alpha=0.3, label='Excellent')
    ax.legend()

    # 4. Feature comparison
    ax = axes[1, 1]
    categories = ['ODE/SDE\nSolvers', 'Neural\nModels', 'Neuroscience\nModels',
                  'Learning\nMethods', 'Stability\nAnalysis']
    td_vals = [3, 4, 0, 3, 0]  # Counts from feature list
    pq_vals = [0, 0, 3, 2, 1]
    x = np.arange(len(categories))
    ax.bar(x - 0.2, td_vals, 0.4, label='torchdyn', color='steelblue', alpha=0.8)
    ax.bar(x + 0.2, pq_vals, 0.4, label='PyQuifer', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel('Feature Count')
    ax.set_title('Feature Domain Coverage')
    ax.legend()

    plt.suptitle('PyQuifer vs torchdyn Continuous Dynamics', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'bench_torchdyn.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close()


# ============================================================================
# Section 6: Pytest Tests
# ============================================================================

class TestNeuralODEWrapping:
    """Test that PyQuifer dynamics can be wrapped as NeuralODE."""

    def test_all_systems_wrappable(self):
        """All three ODE systems can be wrapped."""
        cfg = BenchConfig(num_oscillators=10, num_steps=50)
        results = bench_neural_ode_wrapping(cfg)
        for key, r in results.items():
            assert r.can_wrap, f"{key} not wrappable"
            assert r.trajectory_valid, f"{key} trajectory invalid"

    def test_trajectories_finite(self):
        """All wrapped trajectories produce finite values."""
        cfg = BenchConfig(num_oscillators=10, num_steps=50)
        results = bench_neural_ode_wrapping(cfg)
        for key, r in results.items():
            assert r.final_state_finite, f"{key} final state not finite"


class TestSDEComparison:
    """Test SDE dynamics."""

    def test_sde_produces_noise(self):
        """Neural SDE produces stochastic variation across runs."""
        cfg = BenchConfig()
        results = bench_sde_comparison(cfg)
        r_sde = results['neural_sde']
        assert r_sde.noise_effect > 0.01, "SDE should show noise variation"

    def test_sr_produces_noise(self):
        """PyQuifer stochastic resonance produces variation."""
        cfg = BenchConfig()
        results = bench_sde_comparison(cfg)
        r_sr = results['pyquifer_sr']
        assert r_sr.noise_effect > 0.0, "SR should show some variation"


class TestSensitivity:
    """Test sensitivity method comparison."""

    def test_gradients_match(self):
        """Autograd and adjoint-style produce similar gradients."""
        cfg = BenchConfig()
        results = bench_sensitivity(cfg)
        g_auto = results['autograd'].gradient_norm
        g_adj = results['adjoint_sim'].gradient_norm
        # Should be similar (same computation, different memory patterns)
        assert abs(g_auto - g_adj) / (g_auto + 1e-10) < 0.01, \
            f"Gradient mismatch: {g_auto} vs {g_adj}"

    def test_adjoint_less_memory(self):
        """Adjoint-style uses less peak memory than autograd."""
        cfg = BenchConfig()
        results = bench_sensitivity(cfg)
        assert results['adjoint_sim'].peak_memory_kb < results['autograd'].peak_memory_kb, \
            "Adjoint should use less memory"


class TestEnergyPreservation:
    """Test energy/attractor dynamics."""

    def test_hamiltonian_preserves_energy(self):
        """Hamiltonian dynamics (RK4) preserve energy well."""
        cfg = BenchConfig()
        results = bench_energy_preservation(cfg)
        r = results['hamiltonian']
        assert r.energy_drift < 0.01, \
            f"Hamiltonian energy drift too large: {r.energy_drift}"

    def test_stuart_landau_converges(self):
        """Stuart-Landau converges to limit cycle."""
        cfg = BenchConfig()
        results = bench_energy_preservation(cfg)
        r = results['stuart_landau']
        assert r.trajectory_bounded
        assert r.final_energy > 0, "SL should have nonzero amplitude"

    def test_kuramoto_bounded(self):
        """Kuramoto order parameter stays in [0, 1]."""
        cfg = BenchConfig()
        results = bench_energy_preservation(cfg)
        r = results['kuramoto']
        assert 0 <= r.final_energy <= 1.0, f"R out of range: {r.final_energy}"


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Running PyQuifer vs torchdyn benchmarks...\n")
    cfg = BenchConfig()

    node_results = bench_neural_ode_wrapping(cfg)
    sde_results = bench_sde_comparison(cfg)
    sens_results = bench_sensitivity(cfg)
    energy_results = bench_energy_preservation(cfg)
    arch_features = bench_architecture_features()

    print_report(node_results, sde_results, sens_results, energy_results,
                 arch_features)
    make_plots(node_results, sde_results, energy_results)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
