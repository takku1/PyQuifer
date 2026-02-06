"""
Benchmark #1: PyQuifer Oscillators vs KuraNet

Compares PyQuifer's three oscillator modules (LearnableKuramotoBank,
KuramotoDaidoMeanField, StuartLandauOscillator) against KuraNet's methodology
and metrics, measuring sync convergence, wall-clock scaling, and mean-field
approximation accuracy.

Dual-mode:
  - python bench_oscillators.py        → full suite + plots
  - pytest bench_oscillators.py -v     → test functions only

KuraNet reference: PyQuifer/tests/benchmarks/KuraNet/
  - circular_variance from utils.py:99-118
  - Default config: N=100, dt=0.1 (alpha), initial_phase=zero, burn_in=100
"""

from __future__ import annotations

import math
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# ── PyQuifer imports ──
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from pyquifer.oscillators import (
    KuramotoDaidoMeanField,
    LearnableKuramotoBank,
    StuartLandauOscillator,
)

# ── Optional imports ──
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import torchdiffeq  # noqa: F401

    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Shared Setup
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BenchmarkConfig:
    """Config matching KuraNet DEFAULT: experiments.cfg."""

    N: int = 100
    dt: float = 0.1  # KuraNet alpha
    burn_in: int = 100
    total_steps: int = 1000
    record_every: int = 10
    seed: int = 42
    coupling: float = 1.0


@dataclass
class ConvergenceResult:
    model_name: str
    step_indices: List[int]
    order_params: List[float]
    circular_variances: List[float]
    final_R: float


@dataclass
class TimingResult:
    model_name: str
    N: int
    ms_per_step: float


@dataclass
class MeanFieldResult:
    pearson_correlation: float
    max_deviation: float
    mean_deviation: float
    pairwise_R: List[float]
    meanfield_R: List[float]
    step_indices: List[int]


# ── Metrics (reimplemented from KuraNet utils.py:99-118) ──


def circular_variance_standalone(phases: torch.Tensor) -> float:
    """Circular variance for a single snapshot.

    Reimplements KuraNet's circular_variance:
        CV = 1 - sqrt(sum_cos^2 + sum_sin^2) / N

    Args:
        phases: (N,) tensor of oscillator phases.

    Returns:
        float in [0, 1]. 0 = perfectly synchronized, 1 = uniform.
    """
    N = phases.shape[-1]
    xx = torch.cos(phases)
    yy = torch.sin(phases)
    return (1 - torch.sqrt(xx.sum(-1) ** 2 + yy.sum(-1) ** 2) / N).item()


def circular_variance_trajectory(phase_history: torch.Tensor) -> float:
    """Time-averaged circular variance.

    Args:
        phase_history: (T, N) tensor of phase snapshots.

    Returns:
        float: mean CV over all time steps.
    """
    N = phase_history.shape[-1]
    xx = torch.cos(phase_history)
    yy = torch.sin(phase_history)
    cv_per_step = 1 - torch.sqrt(xx.sum(-1) ** 2 + yy.sum(-1) ** 2) / N
    return cv_per_step.mean().item()


# ── Timer ──


@contextmanager
def timer():
    """Context manager returning elapsed wall-clock time in seconds."""
    result = {"elapsed": 0.0}
    start = time.perf_counter()
    yield result
    result["elapsed"] = time.perf_counter() - start


# ── Initial condition generator ──


def generate_shared_initial_conditions(
    config: BenchmarkConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate omega ~ U(-1, 1) and theta_init = zeros, matching KuraNet DEFAULT."""
    torch.manual_seed(config.seed)
    omega = torch.rand(config.N) * 2 - 1  # Uniform(-1, 1)
    theta_init = torch.zeros(config.N)
    return omega, theta_init


# ── Factory functions ──


def _make_kuramoto_bank(
    omega: torch.Tensor, theta_init: torch.Tensor, config: BenchmarkConfig
) -> LearnableKuramotoBank:
    N = omega.shape[0]
    bank = LearnableKuramotoBank(
        num_oscillators=N,
        dt=config.dt,
        initial_frequency_range=(0.0, 1.0),  # will be overwritten
        initial_phase_range=(0.0, 0.01),  # will be overwritten
        topology="global",
    )
    # Override natural frequencies and phases with shared initial conditions
    with torch.no_grad():
        bank.natural_frequencies.data.copy_(omega)
        bank.phases.copy_(theta_init)
        bank.prev_phases.copy_(theta_init)
        bank.coupling_strength.data.fill_(config.coupling)
    return bank


def _make_meanfield(
    omega: torch.Tensor, config: BenchmarkConfig
) -> KuramotoDaidoMeanField:
    omega_mean = omega.mean().item()
    spread = (omega.max() - omega.min()).item()
    # Cauchy half-width approximation from uniform distribution:
    # For U(-a, a), a reasonable Cauchy approximation uses delta = spread * pi / 4
    # This maps the flat distribution width to Cauchy HWHM.
    delta = spread * math.pi / 4

    mf = KuramotoDaidoMeanField(
        omega_mean=omega_mean,
        delta=delta,
        coupling=config.coupling,
        dt=config.dt,
    )
    # Initialize Z to match zero-phase initial condition (R=1, Psi=0)
    with torch.no_grad():
        mf.Z_real.fill_(1.0)
        mf.Z_imag.fill_(0.0)
    return mf


def _make_stuart_landau(
    omega: torch.Tensor, theta_init: torch.Tensor, config: BenchmarkConfig
) -> StuartLandauOscillator:
    N = omega.shape[0]
    sl = StuartLandauOscillator(
        num_oscillators=N,
        mu=0.1,
        omega_range=(0.0, 1.0),  # will be overwritten
        coupling=0.5,
        dt=config.dt,
    )
    with torch.no_grad():
        sl.omega.data.copy_(omega)
        sl.z_real.copy_(torch.cos(theta_init))
        sl.z_imag.copy_(torch.sin(theta_init))
    return sl


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Sync Convergence (R over time)
# ═══════════════════════════════════════════════════════════════════════════════


def bench_convergence_kuramoto_bank(
    config: BenchmarkConfig,
) -> ConvergenceResult:
    omega, theta_init = generate_shared_initial_conditions(config)
    bank = _make_kuramoto_bank(omega, theta_init, config)

    step_indices = []
    order_params = []
    circular_vars = []

    for step in range(config.total_steps):
        bank(steps=1, use_precision=False)
        if step % config.record_every == 0:
            R = bank.get_order_parameter().item()
            cv = circular_variance_standalone(bank.phases)
            step_indices.append(step)
            order_params.append(R)
            circular_vars.append(cv)

    return ConvergenceResult(
        model_name="LearnableKuramotoBank",
        step_indices=step_indices,
        order_params=order_params,
        circular_variances=circular_vars,
        final_R=order_params[-1],
    )


def bench_convergence_meanfield(
    config: BenchmarkConfig,
) -> ConvergenceResult:
    omega, _ = generate_shared_initial_conditions(config)
    mf = _make_meanfield(omega, config)

    step_indices = []
    order_params = []
    circular_vars = []

    for step in range(config.total_steps):
        result = mf(steps=1)
        if step % config.record_every == 0:
            R = result["R"].item()
            # Mean-field has no individual phases; CV = 1 - R
            cv = 1.0 - R
            step_indices.append(step)
            order_params.append(R)
            circular_vars.append(cv)

    return ConvergenceResult(
        model_name="KuramotoDaidoMeanField",
        step_indices=step_indices,
        order_params=order_params,
        circular_variances=circular_vars,
        final_R=order_params[-1],
    )


def bench_convergence_stuart_landau(
    config: BenchmarkConfig,
) -> ConvergenceResult:
    omega, theta_init = generate_shared_initial_conditions(config)
    sl = _make_stuart_landau(omega, theta_init, config)

    step_indices = []
    order_params = []
    circular_vars = []

    for step in range(config.total_steps):
        result = sl(steps=1)
        if step % config.record_every == 0:
            R = result["order_parameter"].item()
            cv = circular_variance_standalone(result["phases"])
            step_indices.append(step)
            order_params.append(R)
            circular_vars.append(cv)

    return ConvergenceResult(
        model_name="StuartLandauOscillator",
        step_indices=step_indices,
        order_params=order_params,
        circular_variances=circular_vars,
        final_R=order_params[-1],
    )


def bench_convergence_kuranet(
    config: BenchmarkConfig,
) -> Optional[ConvergenceResult]:
    """Optional KuraNet comparison. Requires torchdiffeq."""
    if not HAS_TORCHDIFFEQ:
        return None

    try:
        kuranet_dir = __import__("pathlib").Path(__file__).resolve().parent / "KuraNet"
        sys.path.insert(0, str(kuranet_dir))
        from models import KuraNet_xy  # type: ignore

        omega, theta_init = generate_shared_initial_conditions(config)
        N = config.N

        # KuraNet_xy expects feature_dim (we use 1 dummy feature)
        model = KuraNet_xy(
            feature_dim=1,
            num_hid_units=128,
            normalize="node",
            avg_deg=1.0,
            symmetric=True,
            alpha=config.dt,
            gd_steps=config.total_steps,
            burn_in_steps=0,
            initial_phase="zero",
            solver_method="euler",
        )

        # Build input: (N, feature_dim + 1) where last col is phase
        x_features = omega.unsqueeze(1)  # (N, 1)
        y = torch.cat([x_features, theta_init.unsqueeze(1)], dim=1)  # (N, 2)

        step_indices = []
        order_params = []
        circular_vars = []

        t = torch.tensor([0.0])
        for step in range(config.total_steps):
            dy = model(t, y)
            y = y + dy * config.dt
            t = t + config.dt

            if step % config.record_every == 0:
                phases = y[:, -1]
                complex_phases = torch.exp(1j * phases)
                R = torch.abs(complex_phases.mean()).item()
                cv = circular_variance_standalone(phases)
                step_indices.append(step)
                order_params.append(R)
                circular_vars.append(cv)

        return ConvergenceResult(
            model_name="KuraNet_xy",
            step_indices=step_indices,
            order_params=order_params,
            circular_variances=circular_vars,
            final_R=order_params[-1],
        )
    except Exception as e:
        print(f"  [KuraNet skipped: {e}]")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2b: Phase 6 Gap Tests (G-01, G-02, G-16)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FromFrequenciesResult:
    naive_R: float
    fitted_R: float
    improvement: float


@dataclass
class CriticalCouplingResult:
    Kc: float
    super_R: float
    sub_R: float


@dataclass
class RK4vsEulerResult:
    euler_R: float
    rk4_R: float
    euler_R_sl: float
    rk4_R_sl: float


def bench_from_frequencies(config: BenchmarkConfig) -> FromFrequenciesResult:
    """G-01: Compare naive mean-field construction vs from_frequencies classmethod.

    from_frequencies uses IQR-based Cauchy fitting which should yield better
    Ott-Antonsen tracking than the naive mean/spread mapping.
    """
    torch.manual_seed(config.seed)
    omega = torch.rand(config.N) * 2 - 1  # Uniform [-1, 1]

    # Naive construction (existing factory)
    mf_naive = _make_meanfield(omega, config)

    # Fitted construction via from_frequencies classmethod
    mf_fitted = KuramotoDaidoMeanField.from_frequencies(
        omega, coupling=config.coupling, dt=config.dt
    )
    # Initialize Z to match zero-phase initial condition (R=1, Psi=0)
    with torch.no_grad():
        mf_fitted.Z_real.fill_(1.0)
        mf_fitted.Z_imag.fill_(0.0)

    # Run both for total_steps
    for _ in range(config.total_steps):
        naive_result = mf_naive(steps=1)
        fitted_result = mf_fitted(steps=1)

    naive_R = naive_result["R"].item()
    fitted_R = fitted_result["R"].item()
    improvement = fitted_R - naive_R

    print(f"\n  G-01 from_frequencies comparison:")
    print(f"    {'Method':<24s} {'Final R':>8s} {'delta':>10s}")
    print(f"    {'-' * 44}")
    print(f"    {'Naive (mean/spread)':<24s} {naive_R:>8.4f} {mf_naive.delta.item():>10.4f}")
    print(f"    {'from_frequencies (IQR)':<24s} {fitted_R:>8.4f} {mf_fitted.delta.item():>10.4f}")
    print(f"    {'Improvement':<24s} {improvement:>+8.4f}")

    return FromFrequenciesResult(
        naive_R=naive_R, fitted_R=fitted_R, improvement=improvement
    )


def bench_critical_coupling(config: BenchmarkConfig) -> CriticalCouplingResult:
    """G-02: Test get_critical_coupling on StuartLandauOscillator.

    Creates SL oscillators, estimates Kc, then runs at Kc*1.5 (supercritical)
    and Kc*0.5 (subcritical) to verify that supercritical coupling synchronizes
    more than subcritical.
    """
    torch.manual_seed(config.seed)
    N = config.N

    # Create baseline SL to get Kc
    sl_probe = StuartLandauOscillator(
        num_oscillators=N,
        mu=0.1,
        omega_range=(0.0, 2.0),
        coupling=1.0,
        dt=config.dt,
    )
    Kc = sl_probe.get_critical_coupling().item()

    # Supercritical run (coupling = 1.5 * Kc)
    torch.manual_seed(config.seed)
    sl_super = StuartLandauOscillator(
        num_oscillators=N,
        mu=0.1,
        omega_range=(0.0, 2.0),
        coupling=Kc * 1.5,
        dt=config.dt,
    )
    # Copy omega from probe so Kc is consistent
    with torch.no_grad():
        sl_super.omega.data.copy_(sl_probe.omega.data)
    for _ in range(config.total_steps):
        sl_super(steps=1)
    super_R = sl_super.get_order_parameter().item()

    # Subcritical run (coupling = 0.5 * Kc)
    torch.manual_seed(config.seed)
    sl_sub = StuartLandauOscillator(
        num_oscillators=N,
        mu=0.1,
        omega_range=(0.0, 2.0),
        coupling=Kc * 0.5,
        dt=config.dt,
    )
    with torch.no_grad():
        sl_sub.omega.data.copy_(sl_probe.omega.data)
    for _ in range(config.total_steps):
        sl_sub(steps=1)
    sub_R = sl_sub.get_order_parameter().item()

    print(f"\n  G-02 critical coupling (StuartLandau, N={N}):")
    print(f"    Kc (estimated):         {Kc:.4f}")
    print(f"    Supercritical (1.5*Kc):  R = {super_R:.4f}")
    print(f"    Subcritical  (0.5*Kc):  R = {sub_R:.4f}")

    return CriticalCouplingResult(Kc=Kc, super_R=super_R, sub_R=sub_R)


def bench_rk4_vs_euler(config: BenchmarkConfig) -> RK4vsEulerResult:
    """G-16: Compare RK4 vs Euler integration for LearnableKuramotoBank and StuartLandau.

    Uses a large dt (0.5 by default from caller, or config.dt) so the
    difference between integrators is visible.
    """
    torch.manual_seed(config.seed)
    omega = torch.rand(config.N) * 2 - 1
    theta_init = torch.zeros(config.N)

    # --- LearnableKuramotoBank: Euler ---
    bank_euler = LearnableKuramotoBank(
        num_oscillators=config.N,
        dt=config.dt,
        initial_frequency_range=(0.0, 1.0),
        initial_phase_range=(0.0, 0.01),
        topology="global",
        integration_method="euler",
    )
    with torch.no_grad():
        bank_euler.natural_frequencies.data.copy_(omega)
        bank_euler.phases.copy_(theta_init)
        bank_euler.prev_phases.copy_(theta_init)
        bank_euler.coupling_strength.data.fill_(config.coupling)
    for _ in range(config.total_steps):
        bank_euler(steps=1, use_precision=False)
    euler_R = bank_euler.get_order_parameter().item()

    # --- LearnableKuramotoBank: RK4 ---
    bank_rk4 = LearnableKuramotoBank(
        num_oscillators=config.N,
        dt=config.dt,
        initial_frequency_range=(0.0, 1.0),
        initial_phase_range=(0.0, 0.01),
        topology="global",
        integration_method="rk4",
    )
    with torch.no_grad():
        bank_rk4.natural_frequencies.data.copy_(omega)
        bank_rk4.phases.copy_(theta_init)
        bank_rk4.prev_phases.copy_(theta_init)
        bank_rk4.coupling_strength.data.fill_(config.coupling)
    for _ in range(config.total_steps):
        bank_rk4(steps=1, use_precision=False)
    rk4_R = bank_rk4.get_order_parameter().item()

    # --- StuartLandauOscillator: Euler vs RK4 ---
    torch.manual_seed(config.seed)
    sl_euler = StuartLandauOscillator(
        num_oscillators=config.N,
        mu=0.1,
        omega_range=(0.5, 1.5),
        coupling=1.0,
        dt=config.dt,
        integration_method="euler",
    )
    torch.manual_seed(config.seed)
    sl_rk4 = StuartLandauOscillator(
        num_oscillators=config.N,
        mu=0.1,
        omega_range=(0.5, 1.5),
        coupling=1.0,
        dt=config.dt,
        integration_method="rk4",
    )
    # Copy state so both start identically
    with torch.no_grad():
        sl_rk4.omega.data.copy_(sl_euler.omega.data)
        sl_rk4.z_real.copy_(sl_euler.z_real)
        sl_rk4.z_imag.copy_(sl_euler.z_imag)

    for _ in range(config.total_steps):
        sl_euler(steps=1)
        sl_rk4(steps=1)
    euler_R_sl = sl_euler.get_order_parameter().item()
    rk4_R_sl = sl_rk4.get_order_parameter().item()

    print(f"\n  G-16 RK4 vs Euler (dt={config.dt}, steps={config.total_steps}):")
    print(f"    {'Model':<28s} {'Euler R':>8s} {'RK4 R':>8s} {'|diff|':>8s}")
    print(f"    {'-' * 54}")
    print(
        f"    {'LearnableKuramotoBank':<28s} {euler_R:>8.4f} {rk4_R:>8.4f} "
        f"{abs(euler_R - rk4_R):>8.4f}"
    )
    print(
        f"    {'StuartLandauOscillator':<28s} {euler_R_sl:>8.4f} {rk4_R_sl:>8.4f} "
        f"{abs(euler_R_sl - rk4_R_sl):>8.4f}"
    )

    return RK4vsEulerResult(
        euler_R=euler_R, rk4_R=rk4_R,
        euler_R_sl=euler_R_sl, rk4_R_sl=rk4_R_sl,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Wall-Clock Scaling (ms/step vs N)
# ═══════════════════════════════════════════════════════════════════════════════

SCALING_SIZES = [10, 50, 100, 500, 1000]
WARMUP_STEPS = 10
TIMED_STEPS = 100


def _scaling_config(N: int) -> BenchmarkConfig:
    return BenchmarkConfig(N=N, dt=0.1, seed=42)


def bench_scaling_kuramoto_bank(N: int) -> TimingResult:
    cfg = _scaling_config(N)
    omega, theta_init = generate_shared_initial_conditions(cfg)
    bank = _make_kuramoto_bank(omega, theta_init, cfg)

    # Warmup
    for _ in range(WARMUP_STEPS):
        bank(steps=1, use_precision=False)

    # Timed
    with timer() as t:
        for _ in range(TIMED_STEPS):
            bank(steps=1, use_precision=False)

    return TimingResult("LearnableKuramotoBank", N, t["elapsed"] * 1000 / TIMED_STEPS)


def bench_scaling_meanfield(N: int) -> TimingResult:
    """Mean-field is O(1) — N is nominal only."""
    cfg = _scaling_config(N)
    omega, _ = generate_shared_initial_conditions(cfg)
    mf = _make_meanfield(omega, cfg)

    for _ in range(WARMUP_STEPS):
        mf(steps=1)

    with timer() as t:
        for _ in range(TIMED_STEPS):
            mf(steps=1)

    return TimingResult("KuramotoDaidoMeanField", N, t["elapsed"] * 1000 / TIMED_STEPS)


def bench_scaling_stuart_landau(N: int) -> TimingResult:
    cfg = _scaling_config(N)
    omega, theta_init = generate_shared_initial_conditions(cfg)
    sl = _make_stuart_landau(omega, theta_init, cfg)

    for _ in range(WARMUP_STEPS):
        sl(steps=1)

    with timer() as t:
        for _ in range(TIMED_STEPS):
            sl(steps=1)

    return TimingResult("StuartLandauOscillator", N, t["elapsed"] * 1000 / TIMED_STEPS)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Mean-Field vs Pairwise Accuracy
# ═══════════════════════════════════════════════════════════════════════════════


def bench_meanfield_accuracy(config: BenchmarkConfig) -> MeanFieldResult:
    """Compare pairwise Kuramoto bank R-trajectory against mean-field R."""
    omega, theta_init = generate_shared_initial_conditions(config)

    bank = _make_kuramoto_bank(omega, theta_init, config)
    mf = _make_meanfield(omega, config)

    pairwise_R = []
    meanfield_R = []
    step_indices = []

    for step in range(config.total_steps):
        bank(steps=1, use_precision=False)
        mf_result = mf(steps=1)

        if step % config.record_every == 0:
            pR = bank.get_order_parameter().item()
            mR = mf_result["R"].item()
            pairwise_R.append(pR)
            meanfield_R.append(mR)
            step_indices.append(step)

    # Pearson correlation
    pR_t = torch.tensor(pairwise_R)
    mR_t = torch.tensor(meanfield_R)

    pR_centered = pR_t - pR_t.mean()
    mR_centered = mR_t - mR_t.mean()

    num = (pR_centered * mR_centered).sum()
    den = torch.sqrt((pR_centered**2).sum() * (mR_centered**2).sum())
    pearson = (num / den.clamp(min=1e-12)).item()

    deviations = (pR_t - mR_t).abs()
    max_dev = deviations.max().item()
    mean_dev = deviations.mean().item()

    return MeanFieldResult(
        pearson_correlation=pearson,
        max_deviation=max_dev,
        mean_deviation=mean_dev,
        pairwise_R=pairwise_R,
        meanfield_R=meanfield_R,
        step_indices=step_indices,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Output
# ═══════════════════════════════════════════════════════════════════════════════


def _print_convergence_table(results: List[ConvergenceResult]) -> None:
    print("\n" + "=" * 68)
    print("  CONVERGENCE: Order Parameter R over Time")
    print("=" * 68)
    print(f"  {'Model':<28s} {'Final R':>8s} {'Final CV':>9s}")
    print("  " + "-" * 47)
    for r in results:
        print(f"  {r.model_name:<28s} {r.final_R:>8.4f} {r.circular_variances[-1]:>9.4f}")
    print()


def _print_scaling_table(
    results: Dict[str, List[TimingResult]], sizes: List[int]
) -> None:
    print("=" * 78)
    print("  WALL-CLOCK SCALING: ms/step vs N")
    print("=" * 78)

    header = f"  {'Model':<28s}"
    for N in sizes:
        header += f" {'N=' + str(N):>8s}"
    header += " {'ratio':>8s}"
    # Build proper header
    header = f"  {'Model':<28s}"
    for N in sizes:
        header += f"  N={N:<5d}"
    header += "  ratio"
    print(header)
    print("  " + "-" * (28 + 8 * len(sizes) + 8))

    for model_name, timings in results.items():
        row = f"  {model_name:<28s}"
        ms_values = {t.N: t.ms_per_step for t in timings}
        base = ms_values.get(sizes[0], 1e-9)
        for N in sizes:
            ms = ms_values.get(N, float("nan"))
            row += f"  {ms:<7.3f}"
        last = ms_values.get(sizes[-1], base)
        ratio = last / base if base > 0 else float("inf")
        row += f"  {ratio:.1f}x"
        print(row)
    print()


def _print_meanfield_table(mf: MeanFieldResult) -> None:
    print("=" * 68)
    print("  MEAN-FIELD vs PAIRWISE ACCURACY")
    print("=" * 68)
    print(f"  Pearson correlation:  {mf.pearson_correlation:>+.4f}")
    print(f"  Max |R| deviation:    {mf.max_deviation:>.4f}")
    print(f"  Mean |R| deviation:   {mf.mean_deviation:>.4f}")
    print()


def _print_phase6_table(
    ff: FromFrequenciesResult,
    cc: CriticalCouplingResult,
    rk4: RK4vsEulerResult,
) -> None:
    print("\n" + "=" * 68)
    print("  PHASE 6 GAP FIXES")
    print("=" * 68)
    print(f"  G-01 from_frequencies:  naive R={ff.naive_R:.4f}  fitted R={ff.fitted_R:.4f}  improvement={ff.improvement:+.4f}")
    print(f"  G-02 critical coupling: Kc={cc.Kc:.4f}  super R={cc.super_R:.4f}  sub R={cc.sub_R:.4f}")
    print(f"  G-16 RK4 vs Euler:      KB euler={rk4.euler_R:.4f} rk4={rk4.rk4_R:.4f}  SL euler={rk4.euler_R_sl:.4f} rk4={rk4.rk4_R_sl:.4f}")
    print()


def _plot_results(
    convergence: List[ConvergenceResult],
    scaling: Dict[str, List[TimingResult]],
    meanfield: MeanFieldResult,
    sizes: List[int],
    save_path: str,
) -> None:
    if not HAS_MATPLOTLIB:
        print("  [matplotlib not available — skipping plot]")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Convergence curves
    ax = axes[0]
    for r in convergence:
        ax.plot(r.step_indices, r.order_params, label=r.model_name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Order Parameter R")
    ax.set_title("Sync Convergence")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: Log-log scaling
    ax = axes[1]
    for model_name, timings in scaling.items():
        ns = [t.N for t in timings]
        ms = [t.ms_per_step for t in timings]
        ax.loglog(ns, ms, "o-", label=model_name)
    ax.set_xlabel("N (oscillators)")
    ax.set_ylabel("ms / step")
    ax.set_title("Wall-Clock Scaling")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    # Panel 3: Mean-field vs pairwise overlay
    ax = axes[2]
    ax.plot(meanfield.step_indices, meanfield.pairwise_R, label="Pairwise (KuramotoBank)")
    ax.plot(
        meanfield.step_indices,
        meanfield.meanfield_R,
        "--",
        label="Mean-Field (KuramotoDaido)",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Order Parameter R")
    ax.set_title(
        f"Mean-Field Accuracy (r={meanfield.pearson_correlation:.3f})"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Pytest Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvergence:
    """Tests that each oscillator model runs and produces valid R values."""

    def test_kuramoto_bank_runs(self) -> None:
        cfg = BenchmarkConfig(N=20, total_steps=50, record_every=10, seed=42)
        result = bench_convergence_kuramoto_bank(cfg)
        assert len(result.order_params) > 0
        for R in result.order_params:
            assert 0.0 <= R <= 1.0 + 1e-6, f"R={R} out of range"

    def test_meanfield_runs(self) -> None:
        cfg = BenchmarkConfig(N=20, total_steps=50, record_every=10, seed=42)
        result = bench_convergence_meanfield(cfg)
        assert len(result.order_params) > 0
        for R in result.order_params:
            assert 0.0 <= R <= 1.0 + 1e-6, f"R={R} out of range"

    def test_stuart_landau_runs(self) -> None:
        cfg = BenchmarkConfig(N=20, total_steps=50, record_every=10, seed=42)
        result = bench_convergence_stuart_landau(cfg)
        assert len(result.order_params) > 0
        for R in result.order_params:
            assert 0.0 <= R <= 1.0 + 1e-6, f"R={R} out of range"

    def test_circular_variance_metric(self) -> None:
        # Perfectly synchronized: all phases = 0 → CV = 0
        sync_phases = torch.zeros(50)
        cv_sync = circular_variance_standalone(sync_phases)
        assert abs(cv_sync) < 1e-6, f"Synced CV should be ~0, got {cv_sync}"

        # Uniformly distributed phases → CV ≈ 1 (for large N)
        torch.manual_seed(123)
        uniform_phases = torch.rand(1000) * 2 * math.pi
        cv_uniform = circular_variance_standalone(uniform_phases)
        assert cv_uniform > 0.8, f"Uniform CV should be ~1, got {cv_uniform}"


class TestScaling:
    """Tests wall-clock scaling behavior."""

    def test_scaling_runs(self) -> None:
        for N in [10, 20]:
            r = bench_scaling_kuramoto_bank(N)
            assert r.ms_per_step > 0
            r = bench_scaling_meanfield(N)
            assert r.ms_per_step > 0
            r = bench_scaling_stuart_landau(N)
            assert r.ms_per_step > 0

    def test_meanfield_constant_scaling(self) -> None:
        """Mean-field should be roughly O(1) — ratio N=500/N=10 < 3x."""
        r10 = bench_scaling_meanfield(10)
        r500 = bench_scaling_meanfield(500)
        ratio = r500.ms_per_step / r10.ms_per_step
        assert ratio < 3.0, (
            f"Mean-field scaling ratio N=500/N=10 = {ratio:.2f}x "
            f"(expected < 3x for O(1) complexity)"
        )


class TestMeanFieldAccuracy:
    """Tests mean-field approximation quality."""

    def test_meanfield_correlation(self) -> None:
        cfg = BenchmarkConfig(N=50, total_steps=200, record_every=10, seed=42)
        result = bench_meanfield_accuracy(cfg)
        assert result.pearson_correlation > 0, (
            f"Mean-field correlation {result.pearson_correlation:.4f} should be > 0"
        )


class TestPhase6Features:
    """Tests for Phase 6 gap-fix features (G-01, G-02, G-16)."""

    def test_from_frequencies_improves_r(self) -> None:
        """from_frequencies should give higher R than naive Cauchy mapping."""
        # Use supercritical coupling (K=3.0) so both methods can sync,
        # but from_frequencies (IQR-based delta) should track better.
        cfg = BenchmarkConfig(N=100, total_steps=500, seed=42, coupling=3.0)
        result = bench_from_frequencies(cfg)
        # from_frequencies should improve over naive mapping
        assert result.improvement > 0.0, (
            f"from_frequencies should improve over naive: "
            f"fitted R={result.fitted_R:.4f}, naive R={result.naive_R:.4f}"
        )
        # With supercritical coupling, fitted should reach meaningful sync
        assert result.fitted_R > 0.3, (
            f"from_frequencies fitted R={result.fitted_R:.4f} expected > 0.3 "
            f"at coupling=3.0"
        )

    def test_critical_coupling_exists(self) -> None:
        """get_critical_coupling should return a positive float."""
        cfg = BenchmarkConfig(N=50, seed=42)
        result = bench_critical_coupling(cfg)
        assert result.Kc > 0.0, f"Kc should be > 0, got {result.Kc}"
        # Supercritical should sync more than subcritical
        assert result.super_R > result.sub_R, (
            f"Supercritical R={result.super_R:.4f} should exceed "
            f"subcritical R={result.sub_R:.4f}"
        )

    def test_rk4_matches_euler(self) -> None:
        """RK4 and Euler should give similar R at small dt."""
        cfg = BenchmarkConfig(N=30, total_steps=200, dt=0.01, seed=42)
        result = bench_rk4_vs_euler(cfg)
        # At small dt both integrators should converge to similar R
        diff = abs(result.euler_R - result.rk4_R)
        assert diff < 0.3, (
            f"RK4 R={result.rk4_R:.4f} vs Euler R={result.euler_R:.4f} "
            f"differ by {diff:.4f} (expected < 0.3 at dt={cfg.dt})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main: Full Suite
# ═══════════════════════════════════════════════════════════════════════════════


def run_full_suite() -> None:
    config = BenchmarkConfig()
    print("=" * 68)
    print("  PyQuifer Oscillators vs KuraNet — Benchmark Suite")
    print(f"  N={config.N}  dt={config.dt}  steps={config.total_steps}  seed={config.seed}")
    print("=" * 68)

    # ── Convergence ──
    print("\n[1/4] Running convergence benchmarks...")
    convergence_results: List[ConvergenceResult] = []

    print("  LearnableKuramotoBank...", end=" ", flush=True)
    with timer() as t:
        convergence_results.append(bench_convergence_kuramoto_bank(config))
    print(f"done ({t['elapsed']:.2f}s)")

    print("  KuramotoDaidoMeanField...", end=" ", flush=True)
    with timer() as t:
        convergence_results.append(bench_convergence_meanfield(config))
    print(f"done ({t['elapsed']:.2f}s)")

    print("  StuartLandauOscillator...", end=" ", flush=True)
    with timer() as t:
        convergence_results.append(bench_convergence_stuart_landau(config))
    print(f"done ({t['elapsed']:.2f}s)")

    print("  KuraNet_xy (optional)...", end=" ", flush=True)
    kuranet_result = bench_convergence_kuranet(config)
    if kuranet_result is not None:
        convergence_results.append(kuranet_result)
        print("done")
    else:
        print("skipped (torchdiffeq not installed)")

    _print_convergence_table(convergence_results)

    # ── Scaling ──
    print("[2/4] Running wall-clock scaling benchmarks...")
    scaling_results: Dict[str, List[TimingResult]] = {
        "LearnableKuramotoBank": [],
        "KuramotoDaidoMeanField": [],
        "StuartLandauOscillator": [],
    }

    for N in SCALING_SIZES:
        print(f"  N={N}...", end=" ", flush=True)
        scaling_results["LearnableKuramotoBank"].append(bench_scaling_kuramoto_bank(N))
        scaling_results["KuramotoDaidoMeanField"].append(bench_scaling_meanfield(N))
        scaling_results["StuartLandauOscillator"].append(bench_scaling_stuart_landau(N))
        print("done")

    _print_scaling_table(scaling_results, SCALING_SIZES)

    # ── Mean-field accuracy ──
    print("[3/4] Running mean-field accuracy benchmark...")
    mf_result = bench_meanfield_accuracy(config)
    _print_meanfield_table(mf_result)

    # ── Phase 6 gap fixes ──
    print("[4/4] Phase 6 gap fixes...")

    print("  G-01: from_frequencies classmethod...", end=" ", flush=True)
    with timer() as t:
        ff_result = bench_from_frequencies(config)
    print(f"done ({t['elapsed']:.2f}s)")

    print("  G-02: get_critical_coupling...", end=" ", flush=True)
    with timer() as t:
        cc_result = bench_critical_coupling(config)
    print(f"done ({t['elapsed']:.2f}s)")

    print("  G-16: RK4 vs Euler (large dt=0.5)...", end=" ", flush=True)
    rk4_config = BenchmarkConfig(
        N=config.N, dt=0.5, total_steps=200,
        seed=config.seed, coupling=config.coupling,
    )
    with timer() as t:
        rk4_result = bench_rk4_vs_euler(rk4_config)
    print(f"done ({t['elapsed']:.2f}s)")

    _print_phase6_table(ff_result, cc_result, rk4_result)

    # ── Plot ──
    plot_path = str(
        __import__("pathlib").Path(__file__).resolve().parent / "bench_oscillators.png"
    )
    _plot_results(convergence_results, scaling_results, mf_result, SCALING_SIZES, plot_path)

    print("Done.")


if __name__ == "__main__":
    run_full_suite()
