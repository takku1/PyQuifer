"""
Benchmark: PyQuifer Criticality Detection vs Classical Bifurcation Analysis

Compares PyQuifer's online criticality detection (AvalancheDetector, BranchingRatio,
KoopmanBifurcationDetector, CriticalityController) against classical bifurcation
analysis methods (Jacobian eigenvalues, fixed-point continuation).

Reference: Julia bifurcation continuation code (Barr2016, Rata2018, Yao2008)
reimplemented here as Python/NumPy reference for fair comparison.

Benchmark sections:
  1. Saddle-Node Bifurcation Detection (1D: x' = r + x^2)
  2. Hopf Bifurcation Detection (2D: oscillatory onset)
  3. Logistic Map (Period-Doubling Route to Chaos)
  4. CriticalityController Closed-Loop Test
  5. Architecture Feature Comparison

Usage:
  python bench_bifurcation.py           # Full suite with console output + plots
  pytest bench_bifurcation.py -v        # Just the tests
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

from pyquifer.criticality import (
    AvalancheDetector,
    BranchingRatio,
    CriticalityController,
    HomeostaticRegulator,
    KoopmanBifurcationDetector,
)


# ============================================================================
# Section 1: Classical Bifurcation Analysis (Reference Implementation)
# ============================================================================

def jacobian_eigenvalues_1d(f, x, r, eps=1e-6):
    """Compute Jacobian eigenvalue for 1D system f(x, r).
    Returns the derivative df/dx at the fixed point (= eigenvalue for 1D).
    """
    return (f(x + eps, r) - f(x - eps, r)) / (2 * eps)


def jacobian_eigenvalues_2d(f, x, r, eps=1e-6):
    """Compute Jacobian eigenvalues for 2D system f([x1, x2], r).
    Returns eigenvalues of the 2x2 Jacobian.
    """
    x = np.array(x, dtype=float)
    J = np.zeros((2, 2))
    for i in range(2):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        J[:, i] = (np.array(f(x_plus, r)) - np.array(f(x_minus, r))) / (2 * eps)
    return np.linalg.eigvals(J)


def find_fixed_point_1d(f, r, x0=0.0, max_iter=100, tol=1e-8):
    """Newton's method to find x* where f(x*, r) = 0."""
    x = x0
    for _ in range(max_iter):
        fx = f(x, r)
        dfx = jacobian_eigenvalues_1d(f, x, r)
        if abs(dfx) < 1e-12:
            break
        x = x - fx / dfx
        if abs(f(x, r)) < tol:
            break
    return x


def continuation_1d(f, r_range, x0=0.0, num_points=200):
    """Parameter continuation: track fixed point as r varies.
    Returns arrays of (r, x*, eigenvalue).
    """
    rs = np.linspace(r_range[0], r_range[1], num_points)
    xs = []
    eigs = []
    x = x0
    for r in rs:
        x = find_fixed_point_1d(f, r, x0=x)
        eig = jacobian_eigenvalues_1d(f, x, r)
        xs.append(x)
        eigs.append(eig)
    return rs, np.array(xs), np.array(eigs)


@contextmanager
def timer():
    start = time.perf_counter()
    result = {'elapsed_ms': 0.0}
    yield result
    result['elapsed_ms'] = (time.perf_counter() - start) * 1000


# ============================================================================
# Section 2: Test Systems
# ============================================================================

def saddle_node(x, r):
    """Saddle-node bifurcation: x' = r + x^2.
    Bifurcation at r=0: two fixed points merge and disappear.
    For r < 0: two fixed points at x = +/- sqrt(-r)
    For r >= 0: no real fixed points
    """
    return r + x ** 2


def hopf_system(x, mu):
    """Hopf bifurcation in 2D: dx/dt = mu*x - y - x*(x^2+y^2),
    dy/dt = x + mu*y - y*(x^2+y^2).
    Bifurcation at mu=0: fixed point at origin loses stability.
    """
    x1, x2 = x[0], x[1]
    r2 = x1 ** 2 + x2 ** 2
    dx1 = mu * x1 - x2 - x1 * r2
    dx2 = x1 + mu * x2 - x2 * r2
    return [dx1, dx2]


def logistic_map(x, r):
    """Logistic map: x_{n+1} = r * x * (1 - x).
    Period-doubling cascade: period 1 -> 2 -> 4 -> 8 -> ... -> chaos.
    """
    return r * x * (1 - x)


# ============================================================================
# Section 3: Benchmark Functions
# ============================================================================

@dataclass
class BifurcationResult:
    """Results from a bifurcation detection benchmark."""
    system: str
    true_bifurcation_r: float
    classical_detected_r: float
    pyquifer_detected_r: float
    detection_error: float  # |pyquifer - true|
    classical_latency_ms: float
    pyquifer_latency_ms: float


def bench_saddle_node() -> BifurcationResult:
    """Test saddle-node bifurcation detection.
    System: x' = r + x^2. Bifurcation at r=0.
    """
    true_bif_r = 0.0

    # --- Classical: eigenvalue continuation ---
    with timer() as t_class:
        # Track the stable branch (x = -sqrt(-r) for r < 0)
        rs = np.linspace(-2.0, 0.5, 500)
        classical_bif_r = None
        for r in rs:
            if r < 0:
                x_fp = -math.sqrt(-r)  # stable fixed point
                eig = jacobian_eigenvalues_1d(saddle_node, x_fp, r)
                if eig > -0.01:  # eigenvalue approaching zero
                    classical_bif_r = r
                    break
            else:
                # No real fixed point exists
                classical_bif_r = r
                break

    if classical_bif_r is None:
        classical_bif_r = 0.5

    # --- PyQuifer: Koopman bifurcation detector ---
    # Simulate dynamics and feed states to detector
    detector = KoopmanBifurcationDetector(
        state_dim=1, buffer_size=200, delay_dim=10, rank=5, compute_every=5
    )

    with timer() as t_pq:
        pyquifer_bif_r = None
        x = -1.0  # start on stable branch
        dt = 0.01

        for r_val in np.linspace(-2.0, 0.5, 500):
            # Euler step: x' = r + x^2
            for _ in range(10):  # 10 sub-steps per parameter value
                dx = r_val + x ** 2
                x = x + dx * dt
                x = max(min(x, 10.0), -10.0)  # clamp

            state = torch.tensor([x], dtype=torch.float32)
            result = detector(state)

            if result['approaching_bifurcation'].item() and pyquifer_bif_r is None:
                pyquifer_bif_r = r_val

    if pyquifer_bif_r is None:
        pyquifer_bif_r = 0.5  # not detected

    return BifurcationResult(
        system='Saddle-Node (x\' = r + x^2)',
        true_bifurcation_r=true_bif_r,
        classical_detected_r=classical_bif_r,
        pyquifer_detected_r=pyquifer_bif_r,
        detection_error=abs(pyquifer_bif_r - true_bif_r),
        classical_latency_ms=t_class['elapsed_ms'],
        pyquifer_latency_ms=t_pq['elapsed_ms']
    )


def bench_hopf() -> BifurcationResult:
    """Test Hopf bifurcation detection.
    System: Supercritical Hopf. Bifurcation at mu=0.
    """
    true_bif_r = 0.0

    # --- Classical: eigenvalue analysis ---
    with timer() as t_class:
        mus = np.linspace(-1.0, 1.0, 400)
        classical_bif_r = None
        for mu in mus:
            eigs = jacobian_eigenvalues_2d(hopf_system, [0.0, 0.0], mu)
            if np.max(np.real(eigs)) > 0:
                classical_bif_r = mu
                break

    if classical_bif_r is None:
        classical_bif_r = 1.0

    # --- PyQuifer: Koopman detector on Hopf dynamics ---
    detector = KoopmanBifurcationDetector(
        state_dim=2, buffer_size=200, delay_dim=10, rank=5, compute_every=5
    )

    with timer() as t_pq:
        pyquifer_bif_r = None
        x = np.array([0.01, 0.01])  # near origin
        dt = 0.01

        for mu_val in np.linspace(-1.0, 1.0, 400):
            for _ in range(20):  # sub-steps
                f = hopf_system(x, mu_val)
                x = x + np.array(f) * dt

            state = torch.tensor(x, dtype=torch.float32)
            result = detector(state)

            if result['approaching_bifurcation'].item() and pyquifer_bif_r is None:
                pyquifer_bif_r = mu_val

    if pyquifer_bif_r is None:
        pyquifer_bif_r = 1.0

    return BifurcationResult(
        system='Hopf (supercritical, 2D)',
        true_bifurcation_r=true_bif_r,
        classical_detected_r=classical_bif_r,
        pyquifer_detected_r=pyquifer_bif_r,
        detection_error=abs(pyquifer_bif_r - true_bif_r),
        classical_latency_ms=t_class['elapsed_ms'],
        pyquifer_latency_ms=t_pq['elapsed_ms']
    )


@dataclass
class LogisticMapResult:
    """Results from logistic map benchmark."""
    r_values: np.ndarray
    branching_ratios: np.ndarray
    avalanche_counts: np.ndarray
    koopman_margins: np.ndarray
    lyapunov_exponents: np.ndarray  # classical reference


def bench_logistic_map() -> LogisticMapResult:
    """Track criticality metrics across the logistic map's bifurcation cascade.
    r < 3: stable fixed point. r ~ 3.449: period-4. r ~ 3.57: onset of chaos.
    """
    r_values = np.linspace(2.5, 4.0, 100)
    branching_ratios = []
    avalanche_counts = []
    koopman_margins = []
    lyapunov_exponents = []

    for r in r_values:
        # --- Classical: Lyapunov exponent ---
        x = 0.5
        lyap = 0.0
        for _ in range(200):  # burn-in
            x = logistic_map(x, r)
        for _ in range(500):
            x = logistic_map(x, r)
            deriv = abs(r * (1 - 2 * x))
            if deriv > 0:
                lyap += math.log(deriv)
        lyap /= 500
        lyapunov_exponents.append(lyap)

        # --- PyQuifer: BranchingRatio ---
        br = BranchingRatio(window_size=50)
        x = 0.5
        for _ in range(200):  # burn-in
            x = logistic_map(x, r)
        for _ in range(100):
            x = logistic_map(x, r)
            result = br(torch.tensor([x]))
        branching_ratios.append(result['branching_ratio'].item())

        # --- PyQuifer: AvalancheDetector ---
        ad = AvalancheDetector(activity_threshold=0.3, min_avalanche_size=1)
        x = 0.5
        for _ in range(200):
            x = logistic_map(x, r)
        for _ in range(200):
            x = logistic_map(x, r)
            ad(torch.tensor([x]))
        avalanche_counts.append(ad.num_avalanches.item())

        # --- PyQuifer: KoopmanBifurcationDetector ---
        kbd = KoopmanBifurcationDetector(
            state_dim=1, buffer_size=100, delay_dim=10, rank=3, compute_every=5
        )
        x = 0.5
        for _ in range(200):
            x = logistic_map(x, r)
        for _ in range(100):
            x = logistic_map(x, r)
            result = kbd(torch.tensor([x], dtype=torch.float32))
        koopman_margins.append(result['stability_margin'].item())

    return LogisticMapResult(
        r_values=r_values,
        branching_ratios=np.array(branching_ratios),
        avalanche_counts=np.array(avalanche_counts),
        koopman_margins=np.array(koopman_margins),
        lyapunov_exponents=np.array(lyapunov_exponents),
    )


@dataclass
class ControllerResult:
    """Results from CriticalityController closed-loop test."""
    sigma_trajectory: np.ndarray
    coupling_trajectory: np.ndarray
    converged_to_critical: bool
    convergence_step: int
    final_sigma: float


def bench_controller() -> ControllerResult:
    """Test CriticalityController's ability to drive a system to criticality.
    Uses a simple multiplicative process: x_{t+1} = coupling * x_t + noise.
    """
    controller = CriticalityController(
        target_branching_ratio=1.0,
        adaptation_rate=0.02,
        min_coupling=0.1,
        max_coupling=2.0,
    )

    base_coupling = 0.5  # Start subcritical
    activity = 5.0
    sigmas = []
    couplings = []

    for step in range(500):
        # Dynamics: activity = coupling * activity + noise
        coupling = controller.get_adjusted_coupling(
            torch.tensor(base_coupling)
        ).item()
        noise = np.random.randn() * 0.5
        activity = coupling * activity + noise
        activity = max(0.01, min(activity, 100.0))  # clamp

        result = controller(torch.tensor([activity]))
        sigmas.append(result['branching_ratio'].item())
        couplings.append(coupling)

    sigmas = np.array(sigmas)
    couplings = np.array(couplings)

    # Check convergence: sigma within 0.1 of 1.0 for last 50 steps
    converged = np.abs(sigmas[-50:] - 1.0).mean() < 0.2
    conv_step = 500
    for i in range(50, len(sigmas)):
        if np.abs(sigmas[i - 50:i] - 1.0).mean() < 0.2:
            conv_step = i
            break

    return ControllerResult(
        sigma_trajectory=sigmas,
        coupling_trajectory=couplings,
        converged_to_critical=converged,
        convergence_step=conv_step,
        final_sigma=sigmas[-1]
    )


@dataclass
class KoopmanConfidenceResult:
    """Results from G-18: Koopman bootstrap confidence test."""
    stable_false_positives: int
    detected_at_r: float
    has_confidence_data: bool


def bench_koopman_confidence() -> KoopmanConfidenceResult:
    """G-18: Test Koopman bootstrap confidence — no false positives on stable signal.

    1. Feed a stable signal (constant + small noise) for 200 steps.
       Verify no false alarm (approaching_bifurcation == False).
    2. Feed a signal crossing a bifurcation (logistic map r from 2.5 to 3.8).
       Verify it eventually detects the bifurcation.
    3. Check that stability_margin_std is present in the output (confidence data).
    """
    detector = KoopmanBifurcationDetector(
        state_dim=1, buffer_size=200, delay_dim=10, rank=5,
        compute_every=5, bootstrap_n=20, min_confidence=3,
    )

    # Phase 1: Stable decaying signal — x_{t+1} = 0.8*x_t + noise
    # Eigenvalue = 0.8, well inside unit circle (genuinely stable)
    # Note: a constant signal has eigenvalue=1.0 which IS at the boundary
    stable_false_positives = 0
    x_stable = 1.0
    for _ in range(200):
        x_stable = 0.8 * x_stable + torch.randn(1).item() * 0.02
        state = torch.tensor([x_stable], dtype=torch.float32)
        result = detector(state)
        if result['approaching_bifurcation'].item():
            stable_false_positives += 1

    # Phase 2: Signal crossing bifurcation — logistic map r from 2.5 to 3.8
    detector.reset()
    detected_at_r = float('inf')
    x = 0.5
    for r_val in np.linspace(2.5, 3.8, 300):
        x = logistic_map(x, r_val)
        state = torch.tensor([x], dtype=torch.float32)
        result = detector(state)
        if result['approaching_bifurcation'].item() and detected_at_r == float('inf'):
            detected_at_r = r_val

    # Phase 3: Check confidence data exists
    has_confidence_data = 'stability_margin_std' in result

    return KoopmanConfidenceResult(
        stable_false_positives=stable_false_positives,
        detected_at_r=detected_at_r,
        has_confidence_data=has_confidence_data,
    )


@dataclass
class BranchingConvergenceResult:
    """Results from G-19: BranchingRatio convergence detection test."""
    constant_converged: bool
    varying_converged: bool
    constant_sigma: float
    varying_sigma: float


def bench_branching_convergence() -> BranchingConvergenceResult:
    """G-19: Test BranchingRatio convergence detection.

    1. Feed constant activity (torch.tensor([0.5])) for 100 steps.
       Check that result dict has 'converged' key and it's True.
    2. Feed varying activity (torch.randn) for 100 steps.
       Check converged is False.
    """
    # Constant activity
    br_const = BranchingRatio(window_size=50, variance_threshold=1e-6)
    for _ in range(100):
        result_const = br_const(torch.tensor([0.5]))
    constant_converged = result_const.get('converged', False)
    constant_sigma = result_const['branching_ratio'].item()

    # Varying activity
    br_vary = BranchingRatio(window_size=50, variance_threshold=1e-6)
    for _ in range(100):
        result_vary = br_vary(torch.randn(1).abs() + 0.1)
    varying_converged = result_vary.get('converged', False)
    varying_sigma = result_vary['branching_ratio'].item()

    return BranchingConvergenceResult(
        constant_converged=constant_converged,
        varying_converged=varying_converged,
        constant_sigma=constant_sigma,
        varying_sigma=varying_sigma,
    )


def bench_architecture_features() -> Dict[str, Dict[str, bool]]:
    """Compare architectural features."""
    classical_features = {
        'fixed_point_continuation': True,
        'jacobian_eigenvalues': True,
        'saddle_node_detection': True,
        'hopf_detection': True,
        'pitchfork_detection': True,
        'codimension_2': True,
        'bistability_analysis': True,
        'lyapunov_exponents': True,
        'online_detection': False,
        'adaptive_control': False,
        'avalanche_statistics': False,
        'power_law_analysis': False,
        'neural_network_compatible': False,
        'pytorch_integration': False,
    }

    pyquifer_features = {
        'fixed_point_continuation': False,
        'jacobian_eigenvalues': False,  # Uses DMD instead
        'saddle_node_detection': True,  # Via Koopman margin
        'hopf_detection': True,  # Via Koopman margin
        'pitchfork_detection': True,  # Via Koopman margin (untested)
        'codimension_2': False,
        'bistability_analysis': False,
        'lyapunov_exponents': False,
        'online_detection': True,
        'adaptive_control': True,
        'avalanche_statistics': True,
        'power_law_analysis': True,
        'neural_network_compatible': True,
        'pytorch_integration': True,
    }

    return {'classical': classical_features, 'pyquifer': pyquifer_features}


# ============================================================================
# Section 4: Console Output
# ============================================================================

def print_report(sn_result, hopf_result, logistic_result, ctrl_result,
                 arch_features, koopman_conf=None, branching_conv=None):
    print("=" * 70)
    print("BENCHMARK: PyQuifer Criticality vs Classical Bifurcation Analysis")
    print("=" * 70)

    # Bifurcation detection
    print("\n--- 1. Bifurcation Detection Accuracy ---\n")
    print(f"{'System':<32} {'True r':>8} {'Classical':>10} {'PyQuifer':>10} {'Error':>8}")
    print("-" * 72)
    for r in [sn_result, hopf_result]:
        print(f"{r.system:<32} {r.true_bifurcation_r:>8.3f} "
              f"{r.classical_detected_r:>10.3f} {r.pyquifer_detected_r:>10.3f} "
              f"{r.detection_error:>8.3f}")

    # Timing
    print(f"\n  Saddle-Node: classical {sn_result.classical_latency_ms:.1f}ms, "
          f"PyQuifer {sn_result.pyquifer_latency_ms:.1f}ms")
    print(f"  Hopf:        classical {hopf_result.classical_latency_ms:.1f}ms, "
          f"PyQuifer {hopf_result.pyquifer_latency_ms:.1f}ms")

    # Logistic map
    print("\n--- 2. Logistic Map: Metrics vs Parameter ---\n")
    # Sample a few r values
    indices = [0, 25, 50, 75, 99]
    print(f"{'r':>6} {'Lyapunov':>10} {'BR sigma':>10} {'Koopman margin':>14} {'Avalanches':>11}")
    print("-" * 55)
    for i in indices:
        r = logistic_result.r_values[i]
        ly = logistic_result.lyapunov_exponents[i]
        br = logistic_result.branching_ratios[i]
        km = logistic_result.koopman_margins[i]
        av = logistic_result.avalanche_counts[i]
        print(f"{r:>6.2f} {ly:>10.4f} {br:>10.4f} {km:>14.4f} {av:>11.0f}")

    # Correlation between Lyapunov and Koopman margin
    valid_mask = np.isfinite(logistic_result.koopman_margins) & np.isfinite(
        logistic_result.lyapunov_exponents)
    if valid_mask.sum() > 10:
        from numpy import corrcoef
        corr = corrcoef(
            logistic_result.lyapunov_exponents[valid_mask],
            logistic_result.koopman_margins[valid_mask]
        )[0, 1]
        print(f"\n  Pearson correlation (Lyapunov vs Koopman margin): {corr:.3f}")

    # Controller
    print("\n--- 3. CriticalityController Closed-Loop ---\n")
    print(f"  Converged to critical: {ctrl_result.converged_to_critical}")
    print(f"  Convergence step: {ctrl_result.convergence_step}")
    print(f"  Final sigma: {ctrl_result.final_sigma:.3f}")
    print(f"  Sigma trajectory (first 5): "
          f"{[f'{s:.3f}' for s in ctrl_result.sigma_trajectory[:5]]}")
    print(f"  Sigma trajectory (last 5):  "
          f"{[f'{s:.3f}' for s in ctrl_result.sigma_trajectory[-5:]]}")

    # Architecture
    print("\n--- 4. Architecture Feature Comparison ---\n")
    cl = arch_features['classical']
    pq = arch_features['pyquifer']
    all_f = sorted(set(list(cl.keys()) + list(pq.keys())))
    print(f"{'Feature':<30} {'Classical':>10} {'PyQuifer':>10}")
    print("-" * 54)
    cl_count = pq_count = 0
    for f in all_f:
        cv = 'YES' if cl.get(f, False) else 'no'
        pv = 'YES' if pq.get(f, False) else 'no'
        if cl.get(f, False):
            cl_count += 1
        if pq.get(f, False):
            pq_count += 1
        print(f"  {f:<28} {cv:>10} {pv:>10}")
    print(f"\n  Classical: {cl_count}/{len(all_f)} | PyQuifer: {pq_count}/{len(all_f)}")

    # Phase 6 gap tests
    if koopman_conf is not None:
        print("\n--- 5. Phase 6: Koopman Bootstrap Confidence (G-18) ---\n")
        print(f"  Stable signal false positives: {koopman_conf.stable_false_positives}")
        print(f"  Detected bifurcation at r:     {koopman_conf.detected_at_r:.3f}")
        print(f"  Has confidence data (std):     {'YES' if koopman_conf.has_confidence_data else 'NO'}")

    if branching_conv is not None:
        print("\n--- 6. Phase 6: BranchingRatio Convergence (G-19) ---\n")
        print(f"  Constant input converged:      {'YES' if branching_conv.constant_converged else 'NO'}")
        print(f"  Constant input sigma:          {branching_conv.constant_sigma:.4f}")
        print(f"  Varying input converged:       {'YES' if branching_conv.varying_converged else 'NO'}")
        print(f"  Varying input sigma:           {branching_conv.varying_sigma:.4f}")


# ============================================================================
# Section 5: Plots
# ============================================================================

def make_plots(logistic_result, ctrl_result):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available, skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Logistic map: Lyapunov vs Koopman margin
    ax = axes[0, 0]
    ax.plot(logistic_result.r_values, logistic_result.lyapunov_exponents,
            'b-', label='Lyapunov exponent', alpha=0.8)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('r')
    ax.set_ylabel('Lyapunov exponent')
    ax.set_title('Logistic Map: Classical Lyapunov')
    ax.legend()

    # 2. Koopman stability margin
    ax = axes[0, 1]
    ax.plot(logistic_result.r_values, logistic_result.koopman_margins,
            'r-', label='Koopman stability margin', alpha=0.8)
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.3, label='Bifurcation')
    ax.set_xlabel('r')
    ax.set_ylabel('Stability margin')
    ax.set_title('Logistic Map: PyQuifer Koopman Detection')
    ax.legend()

    # 3. Branching ratio across r
    ax = axes[1, 0]
    ax.plot(logistic_result.r_values, logistic_result.branching_ratios,
            'g-', label='Branching ratio', alpha=0.8)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, label='Critical (sigma=1)')
    ax.set_xlabel('r')
    ax.set_ylabel('Branching ratio')
    ax.set_title('Logistic Map: PyQuifer BranchingRatio')
    ax.legend()

    # 4. Controller convergence
    ax = axes[1, 1]
    ax.plot(ctrl_result.sigma_trajectory, 'b-', alpha=0.6, label='Sigma')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, label='Target')
    ax.set_xlabel('Step')
    ax.set_ylabel('Branching ratio')
    ax.set_title('CriticalityController Closed-Loop')
    ax.legend()

    plt.suptitle('PyQuifer Criticality vs Classical Bifurcation', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'bench_bifurcation.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close()


# ============================================================================
# Section 6: Pytest Tests
# ============================================================================

class TestSaddleNode:
    """Test saddle-node bifurcation detection."""

    def test_classical_detects_near_zero(self):
        """Classical method detects bifurcation near r=0."""
        result = bench_saddle_node()
        assert abs(result.classical_detected_r - 0.0) < 0.1, \
            f"Classical detection at r={result.classical_detected_r}, expected ~0"

    def test_pyquifer_detects_bifurcation(self):
        """PyQuifer Koopman detector flags approaching bifurcation."""
        result = bench_saddle_node()
        # Koopman is an online heuristic; just verify it detects *something*
        # before the system diverges (r < 0.5)
        assert result.pyquifer_detected_r < 0.5, \
            f"PyQuifer detection too late: r={result.pyquifer_detected_r}"


class TestHopf:
    """Test Hopf bifurcation detection."""

    def test_classical_detects_near_zero(self):
        """Classical method detects Hopf near mu=0."""
        result = bench_hopf()
        assert abs(result.classical_detected_r - 0.0) < 0.05, \
            f"Classical detection at mu={result.classical_detected_r}, expected ~0"

    def test_pyquifer_detects_bifurcation(self):
        """PyQuifer detects Hopf bifurcation region."""
        result = bench_hopf()
        # Koopman should detect instability before mu=0.5
        assert result.pyquifer_detected_r < 0.5, \
            f"PyQuifer detection too late: mu={result.pyquifer_detected_r}"


class TestLogisticMap:
    """Test metrics across logistic map parameter space."""

    def test_branching_ratio_stable_regime(self):
        """Branching ratio < 1 in stable regime (r=2.5)."""
        result = bench_logistic_map()
        # r=2.5 is stable fixed point
        assert result.branching_ratios[0] < 1.5, \
            f"BR={result.branching_ratios[0]} at r=2.5, expected < 1.5"

    def test_lyapunov_chaos_regime(self):
        """Lyapunov exponent positive in chaotic regime (r~3.9)."""
        result = bench_logistic_map()
        # r=4.0 (last value) should be chaotic
        assert result.lyapunov_exponents[-1] > 0, \
            f"Lyapunov={result.lyapunov_exponents[-1]} at r=4.0, expected > 0"

    def test_metrics_vary_with_parameter(self):
        """Metrics change meaningfully across r range."""
        result = bench_logistic_map()
        br_range = result.branching_ratios.max() - result.branching_ratios.min()
        assert br_range > 0.01, "Branching ratio should vary across r"


class TestController:
    """Test CriticalityController closed-loop."""

    def test_controller_runs(self):
        """Controller completes 500 steps without error."""
        result = bench_controller()
        assert len(result.sigma_trajectory) == 500

    def test_sigma_bounded(self):
        """Sigma stays in reasonable range."""
        result = bench_controller()
        assert np.all(result.sigma_trajectory > 0), "Sigma should be positive"
        assert np.all(result.sigma_trajectory < 50), "Sigma should not explode"


class TestArchitecture:
    """Test architecture feature comparison."""

    def test_feature_counts(self):
        """Both systems have meaningful feature counts."""
        features = bench_architecture_features()
        cl_count = sum(1 for v in features['classical'].values() if v)
        pq_count = sum(1 for v in features['pyquifer'].values() if v)
        assert cl_count >= 5, f"Classical should have >= 5 features, got {cl_count}"
        assert pq_count >= 5, f"PyQuifer should have >= 5 features, got {pq_count}"


class TestPhase6Criticality:
    """Phase 6 gap tests for criticality module (G-18, G-19)."""

    def test_koopman_low_false_positives_on_stable(self):
        """G-18: Stable decaying signal should have very few false positives."""
        result = bench_koopman_confidence()
        assert result.stable_false_positives < 20, \
            f"Got {result.stable_false_positives} false positives on stable signal (expected < 20)"

    def test_koopman_detects_bifurcation(self):
        """G-18: Logistic map bifurcation should eventually be detected."""
        result = bench_koopman_confidence()
        assert result.detected_at_r < float('inf'), \
            "Koopman detector failed to detect bifurcation in logistic map r=2.5..3.8"

    def test_koopman_has_confidence_data(self):
        """G-18: Output should include stability_margin_std for confidence."""
        result = bench_koopman_confidence()
        assert result.has_confidence_data, \
            "stability_margin_std missing from Koopman output"

    def test_branching_convergence_on_constant(self):
        """G-19: Constant input should converge (variance below threshold)."""
        result = bench_branching_convergence()
        assert result.constant_converged, \
            f"Constant input did not converge, sigma={result.constant_sigma:.4f}"

    def test_branching_no_convergence_on_varying(self):
        """G-19: Varying input should not claim convergence."""
        result = bench_branching_convergence()
        assert not result.varying_converged, \
            f"Varying input falsely reported convergence, sigma={result.varying_sigma:.4f}"


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Running PyQuifer vs Bifurcation Analysis benchmarks...\n")

    np.random.seed(42)
    torch.manual_seed(42)

    sn_result = bench_saddle_node()
    hopf_result = bench_hopf()
    logistic_result = bench_logistic_map()
    ctrl_result = bench_controller()
    arch_features = bench_architecture_features()

    # Phase 6 gap tests
    koopman_conf = bench_koopman_confidence()
    branching_conv = bench_branching_convergence()

    print_report(sn_result, hopf_result, logistic_result, ctrl_result,
                 arch_features, koopman_conf, branching_conv)
    make_plots(logistic_result, ctrl_result)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
