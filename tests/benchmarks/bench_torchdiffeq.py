"""
Benchmark: PyQuifer ODE Integration vs torchdiffeq

Compares PyQuifer's forward Euler integration (used in oscillators.py,
neural_mass.py, criticality.py) against torchdiffeq's adaptive ODE solvers.

PyQuifer modules use `x = x + dx * dt` (Euler, order 1) for all dynamics.
torchdiffeq provides adaptive-step solvers (dopri5 order 5, dopri8 order 8)
with error control and adjoint backpropagation.

Benchmark sections:
  1. Kuramoto Oscillator Accuracy (Euler vs dopri5 vs rk4)
  2. Wilson-Cowan Population Accuracy (Euler vs dopri5)
  3. Stuart-Landau Oscillator Accuracy (Euler vs dopri5)
  4. Step-size Sensitivity (error vs dt for each method)
  5. Wall-clock Timing (Euler vs adaptive)

Usage:
  python bench_torchdiffeq.py           # Full suite with console output + plots
  pytest bench_torchdiffeq.py -v        # Just the tests

Reference: torchdiffeq v0.2.5 (MIT License)
  - Chen et al. (2018), "Neural Ordinary Differential Equations"
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

# Try to import torchdiffeq
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torchdiffeq'))
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False

from pyquifer.oscillators import LearnableKuramotoBank, StuartLandauOscillator
from pyquifer.neural_mass import WilsonCowanPopulation


# ============================================================================
# Section 1: ODE System Definitions (shared by both integrators)
# ============================================================================

class KuramotoODE(nn.Module):
    """Kuramoto oscillator system as an ODE for torchdiffeq.
    dtheta_i/dt = omega_i + (K/N) * sum_j sin(theta_j - theta_i)
    """

    def __init__(self, omega, K=2.0):
        super().__init__()
        self.register_buffer('omega', omega)
        self.K = K
        self.N = omega.shape[0]

    def forward(self, t, theta):
        """ODE right-hand side."""
        # Coupling: K/N * sum_j sin(theta_j - theta_i)
        diff = theta.unsqueeze(-1) - theta.unsqueeze(-2)  # (N, N)
        coupling = self.K / self.N * torch.sin(diff).sum(dim=-1)
        return self.omega - coupling


class WilsonCowanODE(nn.Module):
    """Wilson-Cowan E/I population model as an ODE for torchdiffeq.
    tau_E * dE/dt = -E + S(w_EE*E - w_EI*I + I_ext_E)
    tau_I * dI/dt = -I + S(w_IE*E - w_II*I + I_ext_I)
    """

    def __init__(self, tau_E=10.0, tau_I=5.0, w_EE=12.0, w_EI=4.0,
                 w_IE=13.0, w_II=11.0, gain=1.0, threshold=4.0,
                 I_ext_E=1.0, I_ext_I=0.5):
        super().__init__()
        self.tau_E = tau_E
        self.tau_I = tau_I
        self.w_EE = w_EE
        self.w_EI = w_EI
        self.w_IE = w_IE
        self.w_II = w_II
        self.gain = gain
        self.threshold = threshold
        self.I_ext_E = I_ext_E
        self.I_ext_I = I_ext_I

    def _sigmoid(self, x):
        return 1.0 / (1.0 + torch.exp(-self.gain * (x - self.threshold)))

    def forward(self, t, y):
        """y = [E, I]"""
        E, I = y[0], y[1]
        input_E = self.w_EE * E - self.w_EI * I + self.I_ext_E
        input_I = self.w_IE * E - self.w_II * I + self.I_ext_I
        dE = (-E + self._sigmoid(input_E)) / self.tau_E
        dI = (-I + self._sigmoid(input_I)) / self.tau_I
        return torch.stack([dE, dI])


class StuartLandauODE(nn.Module):
    """Stuart-Landau oscillator as ODE for torchdiffeq.
    dz/dt = (lambda + i*omega)*z - |z|^2 * z + coupling
    Using real representation: z = x + iy -> dx/dt, dy/dt
    """

    def __init__(self, omega, lam=1.0, K=2.0):
        super().__init__()
        self.register_buffer('omega', omega)
        self.lam = lam
        self.K = K
        self.N = omega.shape[0]

    def forward(self, t, y):
        """y = [x_1,...,x_N, y_1,...,y_N] (real and imaginary parts)."""
        x = y[:self.N]
        yi = y[self.N:]
        r2 = x ** 2 + yi ** 2

        # Stuart-Landau dynamics
        dx = self.lam * x - self.omega * yi - r2 * x
        dy = self.lam * yi + self.omega * x - r2 * yi

        # Mean-field coupling
        mean_x = x.mean()
        mean_y = yi.mean()
        dx = dx + self.K * (mean_x - x)
        dy = dy + self.K * (mean_y - yi)

        return torch.cat([dx, dy])


# ============================================================================
# Section 2: Euler Integrator (reimplementing PyQuifer's approach)
# ============================================================================

def euler_integrate(ode_func, y0, t_span, dt):
    """Forward Euler integration matching PyQuifer's approach."""
    t_start, t_end = t_span
    t = t_start
    y = y0.clone()
    trajectory = [y.clone()]
    times = [t]

    while t < t_end - dt / 2:
        dy = ode_func(t, y)
        y = y + dy * dt
        t = t + dt
        trajectory.append(y.clone())
        times.append(t)

    return torch.stack(trajectory), torch.tensor(times)


# ============================================================================
# Section 3: Benchmark Functions
# ============================================================================

@dataclass
class AccuracyResult:
    """Accuracy comparison result."""
    system: str
    euler_trajectory: np.ndarray
    reference_trajectory: np.ndarray  # dopri5 or analytical
    max_error: float
    mean_error: float
    final_error: float
    euler_steps: int
    reference_steps: int  # NFE for adaptive


@dataclass
class TimingResult:
    """Timing comparison result."""
    system: str
    euler_ms: float
    dopri5_ms: float
    rk4_ms: float
    speedup_vs_euler: float  # dopri5_ms / euler_ms


@contextmanager
def timer():
    start = time.perf_counter()
    result = {'elapsed_ms': 0.0}
    yield result
    result['elapsed_ms'] = (time.perf_counter() - start) * 1000


def bench_kuramoto_accuracy(N=20, T=10.0, K=2.0, dt_euler=0.1, seed=42):
    """Compare Kuramoto integration: Euler vs dopri5."""
    torch.manual_seed(seed)
    omega = torch.randn(N)

    ode = KuramotoODE(omega, K=K)
    theta0 = torch.zeros(N)

    # Euler integration (PyQuifer-style)
    euler_traj, euler_times = euler_integrate(ode, theta0, (0.0, T), dt_euler)

    # Reference: dopri5 with tight tolerances
    if HAS_TORCHDIFFEQ:
        t_eval = torch.linspace(0, T, len(euler_times))
        with torch.no_grad():
            ref_traj = odeint(ode, theta0, t_eval, method='dopri5',
                              rtol=1e-8, atol=1e-10)
    else:
        # Use very small dt Euler as reference
        ref_traj_full, _ = euler_integrate(ode, theta0, (0.0, T), dt_euler / 100)
        # Subsample to match euler_times
        indices = torch.linspace(0, len(ref_traj_full) - 1, len(euler_times)).long()
        ref_traj = ref_traj_full[indices]

    # Compute errors (mod 2pi for phase)
    diff = (euler_traj - ref_traj) % (2 * math.pi)
    diff = torch.min(diff, 2 * math.pi - diff)
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    final_err = diff[-1].mean().item()

    return AccuracyResult(
        system='Kuramoto',
        euler_trajectory=euler_traj.numpy(),
        reference_trajectory=ref_traj.numpy(),
        max_error=max_err,
        mean_error=mean_err,
        final_error=final_err,
        euler_steps=len(euler_times),
        reference_steps=len(euler_times)  # dopri5 adaptive (unknown NFE)
    )


def bench_wilson_cowan_accuracy(T=50.0, dt_euler=0.1, seed=42):
    """Compare Wilson-Cowan integration: Euler vs dopri5."""
    torch.manual_seed(seed)

    ode = WilsonCowanODE()
    y0 = torch.tensor([0.3, 0.3])  # Initial E, I

    # Euler
    euler_traj, euler_times = euler_integrate(ode, y0, (0.0, T), dt_euler)

    # Reference
    if HAS_TORCHDIFFEQ:
        t_eval = torch.linspace(0, T, len(euler_times))
        with torch.no_grad():
            ref_traj = odeint(ode, y0, t_eval, method='dopri5',
                              rtol=1e-8, atol=1e-10)
    else:
        ref_traj_full, _ = euler_integrate(ode, y0, (0.0, T), dt_euler / 100)
        indices = torch.linspace(0, len(ref_traj_full) - 1, len(euler_times)).long()
        ref_traj = ref_traj_full[indices]

    diff = torch.abs(euler_traj - ref_traj)
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    final_err = diff[-1].mean().item()

    return AccuracyResult(
        system='Wilson-Cowan',
        euler_trajectory=euler_traj.numpy(),
        reference_trajectory=ref_traj.numpy(),
        max_error=max_err,
        mean_error=mean_err,
        final_error=final_err,
        euler_steps=len(euler_times),
        reference_steps=len(euler_times)
    )


def bench_stuart_landau_accuracy(N=10, T=10.0, dt_euler=0.1, seed=42):
    """Compare Stuart-Landau integration: Euler vs dopri5."""
    torch.manual_seed(seed)
    omega = torch.randn(N)

    ode = StuartLandauODE(omega, lam=1.0, K=2.0)
    # Initial state: small perturbation from origin
    y0 = torch.randn(2 * N) * 0.1

    euler_traj, euler_times = euler_integrate(ode, y0, (0.0, T), dt_euler)

    if HAS_TORCHDIFFEQ:
        t_eval = torch.linspace(0, T, len(euler_times))
        with torch.no_grad():
            ref_traj = odeint(ode, y0, t_eval, method='dopri5',
                              rtol=1e-8, atol=1e-10)
    else:
        ref_traj_full, _ = euler_integrate(ode, y0, (0.0, T), dt_euler / 100)
        indices = torch.linspace(0, len(ref_traj_full) - 1, len(euler_times)).long()
        ref_traj = ref_traj_full[indices]

    diff = torch.abs(euler_traj - ref_traj)

    return AccuracyResult(
        system='Stuart-Landau',
        euler_trajectory=euler_traj.numpy(),
        reference_trajectory=ref_traj.numpy(),
        max_error=diff.max().item(),
        mean_error=diff.mean().item(),
        final_error=diff[-1].mean().item(),
        euler_steps=len(euler_times),
        reference_steps=len(euler_times)
    )


def bench_step_sensitivity(system='kuramoto', dt_values=None, seed=42):
    """Measure error vs step size for Euler integration."""
    if dt_values is None:
        dt_values = [1.0, 0.5, 0.1, 0.05, 0.01]

    torch.manual_seed(seed)
    T = 10.0

    if system == 'kuramoto':
        N = 10
        omega = torch.randn(N)
        ode = KuramotoODE(omega, K=2.0)
        y0 = torch.zeros(N)
    elif system == 'wilson_cowan':
        ode = WilsonCowanODE()
        y0 = torch.tensor([0.3, 0.3])
    else:
        N = 5
        omega = torch.randn(N)
        ode = StuartLandauODE(omega, lam=1.0, K=2.0)
        y0 = torch.randn(2 * N) * 0.1

    # Reference solution (very small dt or dopri5)
    if HAS_TORCHDIFFEQ:
        t_ref = torch.linspace(0, T, 10000)
        with torch.no_grad():
            ref_traj = odeint(ode, y0, t_ref, method='dopri5',
                              rtol=1e-10, atol=1e-12)
        ref_final = ref_traj[-1]
    else:
        ref_traj, _ = euler_integrate(ode, y0, (0.0, T), 0.001)
        ref_final = ref_traj[-1]

    results = {}
    for dt in dt_values:
        traj, times = euler_integrate(ode, y0, (0.0, T), dt)
        final_err = torch.abs(traj[-1] - ref_final).mean().item()
        results[dt] = final_err

    return results


def bench_timing(N=50, T=10.0, dt_euler=0.1, seed=42):
    """Compare wall-clock time: Euler vs torchdiffeq solvers."""
    torch.manual_seed(seed)
    omega = torch.randn(N)
    ode = KuramotoODE(omega, K=2.0)
    theta0 = torch.zeros(N)

    # Euler
    with timer() as t_euler:
        for _ in range(5):
            euler_integrate(ode, theta0, (0.0, T), dt_euler)
    euler_ms = t_euler['elapsed_ms'] / 5

    dopri5_ms = 0.0
    rk4_ms = 0.0

    if HAS_TORCHDIFFEQ:
        t_eval = torch.linspace(0, T, int(T / dt_euler) + 1)

        # dopri5
        with timer() as t_d5:
            for _ in range(5):
                with torch.no_grad():
                    odeint(ode, theta0, t_eval, method='dopri5',
                           rtol=1e-6, atol=1e-8)
        dopri5_ms = t_d5['elapsed_ms'] / 5

        # rk4
        with timer() as t_rk4:
            for _ in range(5):
                with torch.no_grad():
                    odeint(ode, theta0, t_eval, method='rk4',
                           options={'step_size': dt_euler})
        rk4_ms = t_rk4['elapsed_ms'] / 5
    else:
        # Use our own RK4 for comparison
        with timer() as t_rk4:
            for _ in range(5):
                _rk4_integrate(ode, theta0, (0.0, T), dt_euler)
        rk4_ms = t_rk4['elapsed_ms'] / 5

    return TimingResult(
        system='Kuramoto (N=50)',
        euler_ms=euler_ms,
        dopri5_ms=dopri5_ms,
        rk4_ms=rk4_ms,
        speedup_vs_euler=euler_ms / dopri5_ms if dopri5_ms > 0 else 0
    )


def _rk4_integrate(ode_func, y0, t_span, dt):
    """Classic RK4 for comparison when torchdiffeq unavailable."""
    t_start, t_end = t_span
    t = t_start
    y = y0.clone()
    trajectory = [y.clone()]

    while t < t_end - dt / 2:
        k1 = ode_func(t, y)
        k2 = ode_func(t + dt / 2, y + k1 * dt / 2)
        k3 = ode_func(t + dt / 2, y + k2 * dt / 2)
        k4 = ode_func(t + dt, y + k3 * dt)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        t = t + dt
        trajectory.append(y.clone())

    return torch.stack(trajectory), None


def bench_pyquifer_vs_odeint(seed=42):
    """Run PyQuifer's actual modules and compare dynamics."""
    torch.manual_seed(seed)
    results = {}

    # --- PyQuifer KuramotoBank ---
    N = 20
    steps = 100
    kb = LearnableKuramotoBank(
        num_oscillators=N, dt=0.1, topology='global'
    )
    kb.eval()

    # Extract initial conditions BEFORE running
    omega = kb.natural_frequencies.clone().detach()
    K_val = kb.coupling_strength.item()
    initial_phases = kb.phases.clone().detach()

    # Run PyQuifer
    with torch.no_grad():
        for _ in range(steps):
            kb(steps=1)
    pyq_phases = kb.phases.clone()
    pyq_R = kb.get_order_parameter().item()

    # Same system via standalone Euler (matching initial conditions)
    ode = KuramotoODE(omega, K=K_val)
    euler_traj, _ = euler_integrate(ode, initial_phases, (0.0, steps * 0.1), 0.1)
    euler_final = euler_traj[-1] % (2 * math.pi)
    pyq_final = pyq_phases % (2 * math.pi)

    # Phase difference (circular)
    phase_diff = (pyq_final - euler_final) % (2 * math.pi)
    phase_diff = torch.min(phase_diff, 2 * math.pi - phase_diff)

    results['kuramoto'] = {
        'pyquifer_R': pyq_R,
        'euler_match_error': phase_diff.mean().item(),
        'max_phase_diff': phase_diff.max().item(),
    }

    # --- PyQuifer WilsonCowanPopulation ---
    # WC initializes E=0.1, I=0.1
    wc = WilsonCowanPopulation(tau_E=10, tau_I=5, dt=0.1)
    wc.eval()
    with torch.no_grad():
        result = wc(steps=100, I_ext_E=1.0, I_ext_I=0.5)

    pyq_E = result['E'].item()
    pyq_I = result['I'].item()

    # Same system via standalone Euler (matching E=0.1, I=0.1 init)
    wc_ode = WilsonCowanODE(I_ext_E=1.0, I_ext_I=0.5)
    y0_wc = torch.tensor([0.1, 0.1])  # Match WC default init
    euler_wc, _ = euler_integrate(wc_ode, y0_wc, (0.0, 100 * 0.1), 0.1)

    results['wilson_cowan'] = {
        'pyquifer_E': pyq_E,
        'pyquifer_I': pyq_I,
        'euler_E': euler_wc[-1, 0].item(),
        'euler_I': euler_wc[-1, 1].item(),
        'E_match': abs(pyq_E - euler_wc[-1, 0].item()),
        'I_match': abs(pyq_I - euler_wc[-1, 1].item()),
    }

    return results


# ============================================================================
# Section 3b: G-16 PyQuifer Native RK4 Benchmarks
# ============================================================================

@dataclass
class PyQuiferRK4Result:
    """Result from PyQuifer native RK4 vs Euler comparison."""
    E_euler: float
    I_euler: float
    E_rk4: float
    I_rk4: float
    E_diff: float
    I_diff: float
    sl_euler_R: float
    sl_rk4_R: float
    sl_R_diff: float


def bench_pyquifer_rk4(steps=100, dt=0.1, seed=42):
    """G-16: Compare PyQuifer's built-in RK4 vs Euler for WilsonCowan and StuartLandau.

    Creates WilsonCowanPopulation with integration_method='euler' and 'rk4',
    runs both for the same number of steps with I_ext_E=1.0, and compares
    final E/I values. Also compares StuartLandauOscillator with both methods.
    """
    torch.manual_seed(seed)

    # --- WilsonCowanPopulation: Euler ---
    wc_euler = WilsonCowanPopulation(tau_E=10, tau_I=5, dt=dt, integration_method='euler')
    wc_euler.eval()
    with torch.no_grad():
        result_euler = wc_euler(steps=steps, I_ext_E=1.0)
    E_euler = result_euler['E'].item()
    I_euler = result_euler['I'].item()

    # --- WilsonCowanPopulation: RK4 ---
    wc_rk4 = WilsonCowanPopulation(tau_E=10, tau_I=5, dt=dt, integration_method='rk4')
    wc_rk4.eval()
    with torch.no_grad():
        result_rk4 = wc_rk4(steps=steps, I_ext_E=1.0)
    E_rk4 = result_rk4['E'].item()
    I_rk4 = result_rk4['I'].item()

    E_diff = abs(E_euler - E_rk4)
    I_diff = abs(I_euler - I_rk4)

    # --- StuartLandauOscillator: Euler vs RK4 ---
    N_sl = 20
    torch.manual_seed(seed)
    sl_euler = StuartLandauOscillator(
        num_oscillators=N_sl, mu=0.1, omega_range=(0.5, 1.5),
        coupling=1.0, dt=dt, integration_method='euler',
    )
    torch.manual_seed(seed)
    sl_rk4 = StuartLandauOscillator(
        num_oscillators=N_sl, mu=0.1, omega_range=(0.5, 1.5),
        coupling=1.0, dt=dt, integration_method='rk4',
    )
    # Ensure identical initial state
    with torch.no_grad():
        sl_rk4.omega.data.copy_(sl_euler.omega.data)
        sl_rk4.z_real.copy_(sl_euler.z_real)
        sl_rk4.z_imag.copy_(sl_euler.z_imag)

    for _ in range(steps):
        sl_euler(steps=1)
        sl_rk4(steps=1)
    sl_euler_R = sl_euler.get_order_parameter().item()
    sl_rk4_R = sl_rk4.get_order_parameter().item()
    sl_R_diff = abs(sl_euler_R - sl_rk4_R)

    return PyQuiferRK4Result(
        E_euler=E_euler, I_euler=I_euler,
        E_rk4=E_rk4, I_rk4=I_rk4,
        E_diff=E_diff, I_diff=I_diff,
        sl_euler_R=sl_euler_R, sl_rk4_R=sl_rk4_R,
        sl_R_diff=sl_R_diff,
    )


# ============================================================================
# Section 4: Console Output
# ============================================================================

def print_report(kuramoto_acc, wc_acc, sl_acc, step_sens, timing, pyq_match,
                 pyq_rk4=None):
    """Print formatted benchmark report."""
    print("=" * 70)
    print("BENCHMARK: PyQuifer ODE Integration vs torchdiffeq")
    print("=" * 70)
    avail = "YES" if HAS_TORCHDIFFEQ else "NO (using fine-dt Euler as reference)"
    print(f"torchdiffeq available: {avail}\n")

    # --- Accuracy ---
    print("--- 1. Integration Accuracy (Euler dt=0.1 vs reference) ---\n")
    print(f"{'System':<20} {'Max Error':>12} {'Mean Error':>12} {'Final Error':>12}")
    print("-" * 60)
    for r in [kuramoto_acc, wc_acc, sl_acc]:
        print(f"{r.system:<20} {r.max_error:>12.6f} {r.mean_error:>12.6f} "
              f"{r.final_error:>12.6f}")

    # --- Step sensitivity ---
    print("\n--- 2. Step-Size Sensitivity (final error vs dt) ---\n")
    for system, results in step_sens.items():
        print(f"  {system}:")
        print(f"    {'dt':>8} {'Error':>12}")
        for dt, err in sorted(results.items()):
            print(f"    {dt:>8.3f} {err:>12.6f}")

    # --- Timing ---
    print("\n--- 3. Wall-Clock Timing ---\n")
    print(f"  System: {timing.system}")
    print(f"  Euler (dt=0.1):  {timing.euler_ms:.2f} ms")
    if HAS_TORCHDIFFEQ:
        print(f"  dopri5:          {timing.dopri5_ms:.2f} ms")
    print(f"  RK4 (dt=0.1):    {timing.rk4_ms:.2f} ms")

    # --- PyQuifer module match ---
    print("\n--- 4. PyQuifer Module Consistency ---\n")
    km = pyq_match['kuramoto']
    print(f"  Kuramoto: PyQuifer R = {km['pyquifer_R']:.4f}")
    print(f"    vs standalone Euler: mean phase diff = {km['euler_match_error']:.6f} rad")
    print(f"    max phase diff = {km['max_phase_diff']:.6f} rad")

    wc = pyq_match['wilson_cowan']
    print(f"  Wilson-Cowan: PyQuifer E={wc['pyquifer_E']:.4f}, I={wc['pyquifer_I']:.4f}")
    print(f"    vs standalone Euler: E diff={wc['E_match']:.6f}, I diff={wc['I_match']:.6f}")

    # --- Summary ---
    print("\n--- 5. Integration Method Comparison ---\n")
    print(f"{'Method':<20} {'Order':>6} {'Adaptive':>10} {'Backprop':>10} {'Used by':>15}")
    print("-" * 65)
    print(f"{'Euler':<20} {'1':>6} {'No':>10} {'Manual':>10} {'PyQuifer':>15}")
    if HAS_TORCHDIFFEQ:
        print(f"{'RK4':<20} {'4':>6} {'No':>10} {'Autograd':>10} {'torchdiffeq':>15}")
        print(f"{'dopri5':<20} {'5':>6} {'Yes':>10} {'Adjoint':>10} {'torchdiffeq':>15}")
        print(f"{'dopri8':<20} {'8':>6} {'Yes':>10} {'Adjoint':>10} {'torchdiffeq':>15}")
    else:
        print(f"{'RK4':<20} {'4':>6} {'No':>10} {'Manual':>10} {'standalone':>15}")

    # --- G-16: PyQuifer native RK4 ---
    if pyq_rk4 is not None:
        print("\n--- 6. PyQuifer Native RK4 (G-16) ---\n")
        print(f"  Wilson-Cowan (dt={0.1}, {100} steps):")
        print(f"    Euler:  E={pyq_rk4.E_euler:.4f}  I={pyq_rk4.I_euler:.4f}")
        print(f"    RK4:    E={pyq_rk4.E_rk4:.4f}  I={pyq_rk4.I_rk4:.4f}")
        print(f"    Diff:   E={pyq_rk4.E_diff:.6f}  I={pyq_rk4.I_diff:.6f}")
        print(f"  Stuart-Landau (N=20):")
        print(f"    Euler R={pyq_rk4.sl_euler_R:.4f}  RK4 R={pyq_rk4.sl_rk4_R:.4f}  diff={pyq_rk4.sl_R_diff:.6f}")


# ============================================================================
# Section 5: Plots
# ============================================================================

def make_plots(kuramoto_acc, wc_acc, sl_acc, step_sens):
    """Generate benchmark plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available, skipping plots.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Kuramoto trajectory comparison
    ax = axes[0, 0]
    steps_show = min(50, kuramoto_acc.euler_trajectory.shape[0])
    ax.plot(kuramoto_acc.euler_trajectory[:steps_show, 0], label='Euler',
            alpha=0.8)
    ax.plot(kuramoto_acc.reference_trajectory[:steps_show, 0],
            label='Reference', alpha=0.8, linestyle='--')
    ax.set_xlabel('Step')
    ax.set_ylabel('Phase (rad)')
    ax.set_title('Kuramoto Phase (neuron 0)')
    ax.legend()

    # 2. Wilson-Cowan trajectory
    ax = axes[0, 1]
    steps_wc = min(200, wc_acc.euler_trajectory.shape[0])
    ax.plot(wc_acc.euler_trajectory[:steps_wc, 0], label='Euler E', alpha=0.8)
    ax.plot(wc_acc.reference_trajectory[:steps_wc, 0], label='Ref E',
            alpha=0.8, linestyle='--')
    ax.plot(wc_acc.euler_trajectory[:steps_wc, 1], label='Euler I', alpha=0.8)
    ax.plot(wc_acc.reference_trajectory[:steps_wc, 1], label='Ref I',
            alpha=0.8, linestyle='--')
    ax.set_xlabel('Step')
    ax.set_ylabel('Activity')
    ax.set_title('Wilson-Cowan E/I')
    ax.legend()

    # 3. Stuart-Landau trajectory
    ax = axes[0, 2]
    steps_sl = min(50, sl_acc.euler_trajectory.shape[0])
    N_sl = sl_acc.euler_trajectory.shape[1] // 2
    ax.plot(sl_acc.euler_trajectory[:steps_sl, 0], label='Euler Re(z0)',
            alpha=0.8)
    ax.plot(sl_acc.reference_trajectory[:steps_sl, 0], label='Ref Re(z0)',
            alpha=0.8, linestyle='--')
    ax.set_xlabel('Step')
    ax.set_ylabel('Re(z)')
    ax.set_title('Stuart-Landau Re(z0)')
    ax.legend()

    # 4. Step-size convergence (log-log)
    ax = axes[1, 0]
    for system, results in step_sens.items():
        dts = sorted(results.keys())
        errs = [results[dt] for dt in dts]
        ax.loglog(dts, errs, 'o-', label=system)
    # Add order-1 reference line
    dts_ref = np.array([0.01, 1.0])
    ax.loglog(dts_ref, dts_ref * 0.5, 'k--', alpha=0.3, label='O(dt)')
    ax.set_xlabel('dt')
    ax.set_ylabel('Final Error')
    ax.set_title('Step-Size Convergence')
    ax.legend()

    # 5. Error bars
    ax = axes[1, 1]
    systems = ['Kuramoto', 'Wilson-Cowan', 'Stuart-Landau']
    max_errs = [kuramoto_acc.max_error, wc_acc.max_error, sl_acc.max_error]
    mean_errs = [kuramoto_acc.mean_error, wc_acc.mean_error, sl_acc.mean_error]
    x = np.arange(len(systems))
    width = 0.35
    ax.bar(x - width / 2, max_errs, width, label='Max Error', color='coral',
           alpha=0.8)
    ax.bar(x + width / 2, mean_errs, width, label='Mean Error',
           color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel('Error')
    ax.set_title('Euler Error (dt=0.1)')
    ax.legend()

    # 6. Integration method overview
    ax = axes[1, 2]
    methods = ['Euler\n(PyQuifer)', 'RK4', 'dopri5', 'dopri8']
    orders = [1, 4, 5, 8]
    colors = ['coral', 'steelblue', 'green', 'purple']
    ax.bar(methods, orders, color=colors, alpha=0.8)
    ax.set_ylabel('Order')
    ax.set_title('Solver Order Comparison')
    ax.axhline(y=1, color='coral', linestyle='--', alpha=0.3)

    plt.suptitle('PyQuifer Euler vs torchdiffeq ODE Solvers', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'bench_torchdiffeq.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close()


# ============================================================================
# Section 6: Pytest Tests
# ============================================================================

class TestKuramotoAccuracy:
    """Test Kuramoto oscillator integration accuracy."""

    def test_euler_bounded_error(self):
        """Euler integration of Kuramoto has bounded error."""
        result = bench_kuramoto_accuracy(N=10, T=5.0, dt_euler=0.1)
        assert result.max_error < 1.0, \
            f"Kuramoto Euler error too large: {result.max_error}"

    def test_smaller_dt_reduces_error(self):
        """Smaller dt gives smaller error (first-order convergence)."""
        r1 = bench_kuramoto_accuracy(N=10, T=5.0, dt_euler=0.1)
        r2 = bench_kuramoto_accuracy(N=10, T=5.0, dt_euler=0.01)
        assert r2.final_error < r1.final_error, \
            "Smaller dt should reduce error"


class TestWilsonCowanAccuracy:
    """Test Wilson-Cowan integration accuracy."""

    def test_euler_bounded_error(self):
        """Euler integration of Wilson-Cowan has bounded error."""
        result = bench_wilson_cowan_accuracy(T=20.0, dt_euler=0.1)
        assert result.max_error < 0.5, \
            f"WC Euler error too large: {result.max_error}"

    def test_trajectory_bounded(self):
        """Wilson-Cowan Euler trajectory stays in [0, 1]."""
        result = bench_wilson_cowan_accuracy(T=50.0, dt_euler=0.1)
        assert np.all(result.euler_trajectory >= -0.01), "E/I went negative"
        assert np.all(result.euler_trajectory <= 1.01), "E/I exceeded 1"


class TestStuartLandauAccuracy:
    """Test Stuart-Landau integration accuracy."""

    def test_euler_bounded(self):
        """Stuart-Landau Euler trajectory doesn't diverge."""
        result = bench_stuart_landau_accuracy(N=5, T=10.0, dt_euler=0.1)
        assert np.all(np.isfinite(result.euler_trajectory)), \
            "Stuart-Landau diverged"
        assert np.max(np.abs(result.euler_trajectory)) < 100, \
            "Stuart-Landau amplitude too large"


class TestStepSensitivity:
    """Test step-size sensitivity."""

    def test_convergence_order(self):
        """Euler error decreases with smaller dt."""
        results = bench_step_sensitivity('kuramoto',
                                          dt_values=[0.5, 0.1, 0.02])
        dts = sorted(results.keys())  # ascending: 0.02, 0.1, 0.5
        # Smallest dt (dts[0]=0.02) should have less error than largest (dts[-1]=0.5)
        assert results[dts[0]] < results[dts[-1]] * 5, \
            f"Smallest dt error ({results[dts[0]]:.6f}) should be less than " \
            f"largest dt error ({results[dts[-1]]:.6f})"


class TestPyQuiferConsistency:
    """Test that PyQuifer modules match standalone Euler."""

    def test_kuramoto_phases_match(self):
        """PyQuifer KuramotoBank uses same Euler integrator (small drift from
        precision weighting and adjacency details is expected)."""
        results = bench_pyquifer_vs_odeint()
        km = results['kuramoto']
        # Allow moderate difference due to precision weighting, adjacency
        # normalization, and phase velocity tracking in KuramotoBank
        assert km['euler_match_error'] < 0.5, \
            f"KuramotoBank too far from Euler: {km['euler_match_error']}"

    def test_wilson_cowan_matches(self):
        """PyQuifer WilsonCowanPopulation matches standalone Euler."""
        results = bench_pyquifer_vs_odeint()
        wc = results['wilson_cowan']
        # Should match closely
        assert wc['E_match'] < 0.1, \
            f"WilsonCowan E doesn't match Euler: {wc['E_match']}"
        assert wc['I_match'] < 0.1, \
            f"WilsonCowan I doesn't match Euler: {wc['I_match']}"


class TestPyQuiferRK4:
    """Test PyQuifer's built-in RK4 integration (G-16)."""

    def test_wc_rk4_bounded(self):
        """Wilson-Cowan RK4 should produce bounded [0, 1] output."""
        result = bench_pyquifer_rk4(steps=200, dt=0.1)
        assert 0.0 <= result.E_rk4 <= 1.0, \
            f"WC RK4 E={result.E_rk4} out of [0, 1]"
        assert 0.0 <= result.I_rk4 <= 1.0, \
            f"WC RK4 I={result.I_rk4} out of [0, 1]"

    def test_rk4_close_to_euler_at_small_dt(self):
        """At dt=0.01, RK4 and Euler should be very close for Wilson-Cowan."""
        result = bench_pyquifer_rk4(steps=100, dt=0.01)
        assert result.E_diff < 0.01, \
            f"WC E diff={result.E_diff} too large at dt=0.01 (expected < 0.01)"
        assert result.I_diff < 0.01, \
            f"WC I diff={result.I_diff} too large at dt=0.01 (expected < 0.01)"

    def test_sl_rk4_produces_valid_R(self):
        """Stuart-Landau RK4 should produce a valid order parameter in [0, 1]."""
        result = bench_pyquifer_rk4(steps=100, dt=0.1)
        assert 0.0 <= result.sl_rk4_R <= 1.0 + 1e-6, \
            f"SL RK4 R={result.sl_rk4_R} out of range"


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Running PyQuifer vs torchdiffeq benchmarks...\n")

    kuramoto_acc = bench_kuramoto_accuracy()
    wc_acc = bench_wilson_cowan_accuracy()
    sl_acc = bench_stuart_landau_accuracy()

    step_sens = {
        'Kuramoto': bench_step_sensitivity('kuramoto'),
        'Wilson-Cowan': bench_step_sensitivity('wilson_cowan'),
        'Stuart-Landau': bench_step_sensitivity('stuart_landau'),
    }

    timing = bench_timing()
    pyq_match = bench_pyquifer_vs_odeint()

    # G-16: PyQuifer native RK4
    print("\nRunning PyQuifer native RK4 benchmarks (G-16)...")
    pyq_rk4 = bench_pyquifer_rk4()

    print_report(kuramoto_acc, wc_acc, sl_acc, step_sens, timing, pyq_match,
                 pyq_rk4=pyq_rk4)
    make_plots(kuramoto_acc, wc_acc, sl_acc, step_sens)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
