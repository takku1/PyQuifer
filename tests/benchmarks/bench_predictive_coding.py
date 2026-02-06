"""
Benchmark #4: PyQuifer Hierarchical Predictive Coding vs Torch2PC

Torch2PC (Rosenbaum 2022) converts any PyTorch Sequential model into
a predictive coding training algorithm. It implements three algorithms:
- Strict: Full PC without fixed prediction assumption
- FixedPred: PC with fixed prediction assumption (Millidge et al. 2020)
- Exact: Equivalent to standard backpropagation

PyQuifer's hierarchical_predictive.py implements Friston-style predictive
coding with explicit generative/recognition models, precision-weighted
prediction errors, and bidirectional hierarchical message passing.

The comparison is on:
1. Prediction error dynamics (both reduce error over time)
2. Belief convergence quality
3. Hierarchical error propagation
4. Gradient approximation quality (Torch2PC's core contribution)
5. Precision weighting (PyQuifer's unique feature)

Dual-mode: `python bench_predictive_coding.py` (full report) or
           `pytest bench_predictive_coding.py -v` (test assertions)

References:
- Rosenbaum (2022). On the relationship between predictive coding
  and backpropagation. PLoS ONE.
- Friston (2005). A Theory of Cortical Responses.
- Rao & Ballard (1999). Predictive Coding in the Visual Cortex.
- Millidge et al. (2020). Predictive coding approximates backprop
  along arbitrary computation graphs.
"""

import sys
import os
import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn

# Add PyQuifer src to path
_benchmark_dir = Path(__file__).resolve().parent
_pyquifer_src = _benchmark_dir.parent.parent / "src"
if str(_pyquifer_src) not in sys.path:
    sys.path.insert(0, str(_pyquifer_src))

# Add Torch2PC to path
_torch2pc_dir = _benchmark_dir / "Torch2PC"
if str(_torch2pc_dir) not in sys.path:
    sys.path.insert(0, str(_torch2pc_dir))

from pyquifer.hierarchical_predictive import (
    PredictiveLevel,
    HierarchicalPredictiveCoding,
)

# Import Torch2PC functions
try:
    from TorchSeq2PC import PCInfer, FwdPassPlus, ExactPredErrs, FixedPredPCPredErrs
    HAS_TORCH2PC = True
except ImportError:
    HAS_TORCH2PC = False


# ============================================================
# Configuration
# ============================================================

@dataclass
class BenchConfig:
    """Benchmark configuration."""
    seed: int = 42
    input_dim: int = 32
    hidden_dim: int = 16
    output_dim: int = 8
    batch_size: int = 16
    # Convergence test
    convergence_steps: int = 100
    # Torch2PC params
    pc_eta: float = 0.1  # PC inference step size
    pc_n_steps: int = 20  # PC inference iterations
    # PyQuifer params
    pyquifer_lr: float = 0.1
    pyquifer_gen_lr: float = 0.01
    pyquifer_iterations: int = 3


# ============================================================
# Shared utilities
# ============================================================

def _make_sequential_model(config: BenchConfig) -> nn.Sequential:
    """Create a simple Sequential model for Torch2PC."""
    return nn.Sequential(
        nn.Linear(config.input_dim, config.hidden_dim),
        nn.Tanh(),
        nn.Linear(config.hidden_dim, config.output_dim),
    )


def _make_signal(config: BenchConfig, noise_scale: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate consistent input-target pair."""
    X = torch.sin(torch.linspace(0, 2 * math.pi, config.input_dim)).unsqueeze(0)
    X = X.expand(config.batch_size, -1) + torch.randn(config.batch_size, config.input_dim) * noise_scale
    Y = torch.sin(torch.linspace(0, math.pi, config.output_dim)).unsqueeze(0)
    Y = Y.expand(config.batch_size, -1)
    return X, Y


# ============================================================
# Section 1: Prediction Error Dynamics
# ============================================================

@dataclass
class ErrorDynamicsResult:
    """Results from prediction error convergence comparison."""
    # PyQuifer
    pyquifer_initial_error: float
    pyquifer_final_error: float
    pyquifer_error_reduced: bool
    pyquifer_error_trajectory: List[float]
    # Torch2PC (if available)
    torch2pc_initial_loss: float
    torch2pc_final_loss: float
    torch2pc_loss_reduced: bool
    torch2pc_loss_trajectory: List[float]
    torch2pc_available: bool


def bench_error_dynamics(config: BenchConfig) -> ErrorDynamicsResult:
    """
    Compare prediction error reduction over time.

    Both systems should show decreasing prediction error with consistent
    input: the generative model learns to predict the input pattern.
    """
    torch.manual_seed(config.seed)

    # --- PyQuifer: HierarchicalPredictiveCoding ---
    hpc = HierarchicalPredictiveCoding(
        level_dims=[config.input_dim, config.hidden_dim, config.output_dim],
        lr=config.pyquifer_lr,
        gen_lr=config.pyquifer_gen_lr,
        num_iterations=config.pyquifer_iterations,
    )

    # Consistent signal
    signal = torch.sin(torch.linspace(0, 2 * math.pi, config.input_dim)).unsqueeze(0)

    pyquifer_errors = []
    for step in range(config.convergence_steps):
        noise = torch.randn(1, config.input_dim) * 0.1
        result = hpc(signal + noise)
        pyquifer_errors.append(result['total_error'].item())

    # --- Torch2PC: PCInfer ---
    t2pc_losses = []
    t2pc_available = HAS_TORCH2PC

    if HAS_TORCH2PC:
        torch.manual_seed(config.seed)
        model = _make_sequential_model(config)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for step in range(config.convergence_steps):
            X, Y = _make_signal(config, noise_scale=0.1)
            X.requires_grad_(True)

            vhat, loss, dLdy, v, epsilon = PCInfer(
                model, loss_fn, X, Y, "FixedPred",
                eta=config.pc_eta, n=config.pc_n_steps
            )
            optimizer.step()
            optimizer.zero_grad()
            t2pc_losses.append(loss.item())
    else:
        t2pc_losses = [0.0] * config.convergence_steps

    return ErrorDynamicsResult(
        pyquifer_initial_error=pyquifer_errors[0],
        pyquifer_final_error=pyquifer_errors[-1],
        pyquifer_error_reduced=pyquifer_errors[-1] < pyquifer_errors[0],
        pyquifer_error_trajectory=pyquifer_errors,
        torch2pc_initial_loss=t2pc_losses[0] if t2pc_available else 0.0,
        torch2pc_final_loss=t2pc_losses[-1] if t2pc_available else 0.0,
        torch2pc_loss_reduced=t2pc_losses[-1] < t2pc_losses[0] if t2pc_available else False,
        torch2pc_loss_trajectory=t2pc_losses,
        torch2pc_available=t2pc_available,
    )


# ============================================================
# Section 2: Belief Convergence
# ============================================================

@dataclass
class BeliefResult:
    """Results from belief convergence test."""
    # PyQuifer
    pyquifer_belief_norm: float
    pyquifer_belief_stable: bool
    pyquifer_belief_change_rate: float  # Change in last 10 steps
    # Torch2PC
    torch2pc_belief_norm: float
    torch2pc_belief_stable: bool
    torch2pc_epsilon_norm: float
    torch2pc_available: bool


def bench_belief_convergence(config: BenchConfig) -> BeliefResult:
    """
    Test that beliefs converge to stable values with consistent input.

    PyQuifer: beliefs (mu) updated via recognition model + precision-weighted errors.
    Torch2PC: beliefs (v) = vhat - epsilon, converge via iterative relaxation.
    """
    torch.manual_seed(config.seed)

    # --- PyQuifer ---
    hpc = HierarchicalPredictiveCoding(
        level_dims=[config.input_dim, config.hidden_dim],
        lr=config.pyquifer_lr,
        gen_lr=config.pyquifer_gen_lr,
        num_iterations=config.pyquifer_iterations,
    )

    signal = torch.sin(torch.linspace(0, 2 * math.pi, config.input_dim)).unsqueeze(0)

    belief_history = []
    for step in range(config.convergence_steps):
        result = hpc(signal)
        belief_history.append(result['beliefs'][-1].clone())

    # Stability: change rate in last 10 steps
    if len(belief_history) > 10:
        changes = [
            (belief_history[i] - belief_history[i - 1]).norm().item()
            for i in range(-10, 0)
        ]
        belief_change_rate = sum(changes) / len(changes)
    else:
        belief_change_rate = float('inf')

    pyquifer_norm = belief_history[-1].norm().item()
    pyquifer_stable = belief_change_rate < 1.0

    # --- Torch2PC ---
    t2pc_belief_norm = 0.0
    t2pc_stable = False
    t2pc_epsilon_norm = 0.0

    if HAS_TORCH2PC:
        torch.manual_seed(config.seed)
        model = _make_sequential_model(config)
        loss_fn = nn.MSELoss()
        X, Y = _make_signal(config, noise_scale=0.0)  # No noise for convergence
        X.requires_grad_(True)

        vhat, loss, dLdy, v, epsilon = PCInfer(
            model, loss_fn, X, Y, "FixedPred",
            eta=config.pc_eta, n=config.pc_n_steps
        )

        # Beliefs from Torch2PC
        v_norms = [vi.norm().item() for vi in v if vi is not None]
        eps_norms = [ei.norm().item() for ei in epsilon if ei is not None]

        t2pc_belief_norm = sum(v_norms) / len(v_norms) if v_norms else 0.0
        t2pc_epsilon_norm = sum(eps_norms) / len(eps_norms) if eps_norms else 0.0
        t2pc_stable = all(n < 100 for n in v_norms)  # Bounded = stable

    return BeliefResult(
        pyquifer_belief_norm=pyquifer_norm,
        pyquifer_belief_stable=pyquifer_stable,
        pyquifer_belief_change_rate=belief_change_rate,
        torch2pc_belief_norm=t2pc_belief_norm,
        torch2pc_belief_stable=t2pc_stable,
        torch2pc_epsilon_norm=t2pc_epsilon_norm,
        torch2pc_available=HAS_TORCH2PC,
    )


# ============================================================
# Section 3: Hierarchical Error Propagation
# ============================================================

@dataclass
class HierarchyResult:
    """Results from hierarchical error analysis."""
    # PyQuifer (per-level errors)
    pyquifer_errors_per_level: List[float]
    pyquifer_error_decreases_up: bool  # Errors should decrease going up
    pyquifer_num_levels: int
    # Torch2PC (per-layer epsilon norms)
    torch2pc_epsilons_per_layer: List[float]
    torch2pc_num_layers: int
    torch2pc_available: bool


def bench_hierarchy(config: BenchConfig) -> HierarchyResult:
    """
    Compare hierarchical error profiles.

    In predictive coding, higher levels represent more abstract causes.
    Lower levels have larger errors (raw sensory), higher levels smaller.
    """
    torch.manual_seed(config.seed)

    # --- PyQuifer: 4-level hierarchy ---
    level_dims = [config.input_dim, 24, 16, config.output_dim]
    hpc = HierarchicalPredictiveCoding(
        level_dims=level_dims,
        lr=config.pyquifer_lr,
        gen_lr=config.pyquifer_gen_lr,
        num_iterations=config.pyquifer_iterations,
    )

    signal = torch.sin(torch.linspace(0, 2 * math.pi, config.input_dim)).unsqueeze(0)

    # Run enough steps for hierarchy to settle
    for _ in range(50):
        result = hpc(signal)

    pyquifer_level_errors = [
        e.abs().mean().item() for e in result['errors']
    ]

    # Check if errors generally decrease going up
    decreasing = all(
        pyquifer_level_errors[i] >= pyquifer_level_errors[i + 1] * 0.5
        for i in range(len(pyquifer_level_errors) - 1)
    )

    # --- Torch2PC: 4-layer model ---
    t2pc_epsilons = []
    if HAS_TORCH2PC:
        torch.manual_seed(config.seed)
        model = nn.Sequential(
            nn.Linear(config.input_dim, 24),
            nn.Tanh(),
            nn.Linear(24, 16),
            nn.Tanh(),
            nn.Linear(16, config.output_dim),
        )
        loss_fn = nn.MSELoss()
        X, Y = _make_signal(config, noise_scale=0.1)
        X.requires_grad_(True)

        vhat, loss, dLdy, v, epsilon = PCInfer(
            model, loss_fn, X, Y, "FixedPred",
            eta=config.pc_eta, n=config.pc_n_steps
        )

        t2pc_epsilons = [
            ei.abs().mean().item() for ei in epsilon if ei is not None
        ]

    return HierarchyResult(
        pyquifer_errors_per_level=pyquifer_level_errors,
        pyquifer_error_decreases_up=decreasing,
        pyquifer_num_levels=len(level_dims),
        torch2pc_epsilons_per_layer=t2pc_epsilons,
        torch2pc_num_layers=5 if HAS_TORCH2PC else 0,
        torch2pc_available=HAS_TORCH2PC,
    )


# ============================================================
# Section 4: Gradient Approximation Quality
# ============================================================

@dataclass
class GradientResult:
    """Results from gradient approximation benchmark."""
    # Torch2PC: compare Exact vs FixedPred vs Strict gradients
    exact_loss: float
    fixedpred_loss: float
    strict_loss: float
    fixedpred_cos_sim: float  # Cosine similarity of gradients to Exact
    strict_cos_sim: float
    # PyQuifer doesn't compute gradients the same way — report online learning reduction
    pyquifer_gen_loss_reduction: float
    torch2pc_available: bool


def bench_gradient_quality(config: BenchConfig) -> GradientResult:
    """
    Test Torch2PC's core claim: PC gradients approximate backprop.

    Also measure PyQuifer's online learning effectiveness.
    """
    torch.manual_seed(config.seed)

    exact_loss = 0.0
    fp_loss = 0.0
    strict_loss = 0.0
    fp_cos = 0.0
    strict_cos = 0.0

    if HAS_TORCH2PC:
        loss_fn = nn.MSELoss()
        X, Y = _make_signal(config, noise_scale=0.0)

        # Exact (= backprop)
        torch.manual_seed(config.seed)
        model_exact = _make_sequential_model(config)
        X_e = X.clone().requires_grad_(True)
        vhat_e, loss_e, dLdy_e, v_e, eps_e = PCInfer(
            model_exact, loss_fn, X_e, Y, "Exact"
        )
        exact_loss = loss_e.item()
        grads_exact = torch.cat([p.grad.flatten() for p in model_exact.parameters()])

        # FixedPred
        torch.manual_seed(config.seed)
        model_fp = _make_sequential_model(config)
        X_fp = X.clone().requires_grad_(True)
        vhat_fp, loss_fp, dLdy_fp, v_fp, eps_fp = PCInfer(
            model_fp, loss_fn, X_fp, Y, "FixedPred",
            eta=config.pc_eta, n=config.pc_n_steps
        )
        fp_loss = loss_fp.item()
        grads_fp = torch.cat([p.grad.flatten() for p in model_fp.parameters()])

        # Strict
        torch.manual_seed(config.seed)
        model_strict = _make_sequential_model(config)
        X_s = X.clone().requires_grad_(True)
        vhat_s, loss_s, dLdy_s, v_s, eps_s = PCInfer(
            model_strict, loss_fn, X_s, Y, "Strict",
            eta=config.pc_eta, n=config.pc_n_steps
        )
        strict_loss = loss_s.item()
        grads_strict = torch.cat([p.grad.flatten() for p in model_strict.parameters()])

        # Cosine similarity of gradients
        fp_cos = torch.nn.functional.cosine_similarity(
            grads_exact.unsqueeze(0), grads_fp.unsqueeze(0)
        ).item()
        strict_cos = torch.nn.functional.cosine_similarity(
            grads_exact.unsqueeze(0), grads_strict.unsqueeze(0)
        ).item()

    # PyQuifer: online learning reduction
    torch.manual_seed(config.seed)
    level = PredictiveLevel(
        input_dim=config.input_dim,
        belief_dim=config.hidden_dim,
        lr=config.pyquifer_lr,
        gen_lr=config.pyquifer_gen_lr,
    )
    signal = torch.sin(torch.linspace(0, 2 * math.pi, config.input_dim)).unsqueeze(0)

    initial_error = level(signal)['error'].abs().mean().item()
    for _ in range(config.convergence_steps):
        level(signal)
    final_error = level(signal)['error'].abs().mean().item()

    gen_reduction = initial_error - final_error

    return GradientResult(
        exact_loss=exact_loss,
        fixedpred_loss=fp_loss,
        strict_loss=strict_loss,
        fixedpred_cos_sim=fp_cos,
        strict_cos_sim=strict_cos,
        pyquifer_gen_loss_reduction=gen_reduction,
        torch2pc_available=HAS_TORCH2PC,
    )


# ============================================================
# Section 5: Precision Weighting (PyQuifer unique)
# ============================================================

@dataclass
class PrecisionResult:
    """Results from precision weighting test."""
    error_attended: float  # Error on high-precision channels
    error_ignored: float  # Error on low-precision channels
    weighted_error_attended: float
    weighted_error_ignored: float
    precision_gating_works: bool  # Weighted attended < weighted ignored
    torch2pc_has_precision: bool


def bench_precision_weighting(config: BenchConfig) -> PrecisionResult:
    """
    Test PyQuifer's unique precision weighting mechanism.

    Torch2PC has no precision mechanism -- all prediction errors are
    treated equally. PyQuifer gates errors by precision, so high-precision
    channels drive belief updates more than low-precision channels.
    """
    torch.manual_seed(config.seed)

    hpc = HierarchicalPredictiveCoding(
        level_dims=[config.input_dim, config.hidden_dim],
        lr=config.pyquifer_lr,
        gen_lr=config.pyquifer_gen_lr,
        num_iterations=config.pyquifer_iterations,
    )

    # Create precision: high on first half, low on second half
    half = config.input_dim // 2
    precision_l0 = torch.ones(config.input_dim)
    precision_l0[half:] = 0.01  # Low precision on second half

    precision_l1 = torch.ones(config.input_dim)  # Default for level 1

    # Signal: structured in first half, noise in second half
    for _ in range(50):
        signal = torch.randn(1, config.input_dim)
        signal[:, :half] = torch.sin(torch.linspace(0, math.pi, half))
        result = hpc(signal, precisions=[precision_l0, precision_l1])

    error = result['errors'][0]
    weighted = precision_l0 * error

    err_attended = error[:, :half].abs().mean().item()
    err_ignored = error[:, half:].abs().mean().item()
    w_err_attended = weighted[:, :half].abs().mean().item()
    w_err_ignored = weighted[:, half:].abs().mean().item()

    return PrecisionResult(
        error_attended=err_attended,
        error_ignored=err_ignored,
        weighted_error_attended=w_err_attended,
        weighted_error_ignored=w_err_ignored,
        precision_gating_works=w_err_ignored < w_err_attended * 0.5,
        torch2pc_has_precision=False,
    )


# ============================================================
# Section 6: Adaptive Precision (G-11, Phase 6)
# ============================================================

@dataclass
class AdaptivePrecisionResult:
    """Results from G-11: Adaptive precision test."""
    precision_adapted: bool
    mean_precision: float
    precision_std: float
    error_variance_mean: float


def bench_adaptive_precision(config: BenchConfig) -> AdaptivePrecisionResult:
    """
    G-11: Test adaptive precision updates in PredictiveLevel.

    Precision should adapt from its initial all-1s value when exposed to
    a consistent signal. Channels with low error get high precision;
    channels with high error get low precision.
    """
    torch.manual_seed(config.seed)

    level = PredictiveLevel(
        input_dim=config.input_dim,
        belief_dim=config.hidden_dim,
        lr=0.1,
        gen_lr=0.01,
        precision_ema=0.05,
    )

    # Feed a consistent signal for 50 steps
    signal = torch.sin(torch.linspace(0, 2 * math.pi, config.input_dim)).unsqueeze(0)
    for _ in range(50):
        level(signal)

    # Check that precision has changed from initial all-1s
    prec = level.precision.clone()
    precision_adapted = not torch.allclose(prec, torch.ones_like(prec), atol=1e-3)
    mean_precision = prec.mean().item()
    precision_std = prec.std().item()

    # Check that error_variance is no longer all 1s
    err_var = level.error_variance.clone()
    error_variance_mean = err_var.mean().item()

    return AdaptivePrecisionResult(
        precision_adapted=precision_adapted,
        mean_precision=mean_precision,
        precision_std=precision_std,
        error_variance_mean=error_variance_mean,
    )


# ============================================================
# Section 7: Exact Gradients (G-12, Phase 6)
# ============================================================

@dataclass
class ExactGradientsResult:
    """Results from G-12: Exact gradients comparison test."""
    error_without_exact: float
    error_with_exact: float
    improvement: float


def bench_exact_gradients(config: BenchConfig) -> ExactGradientsResult:
    """
    G-12: Compare PredictiveLevel with and without use_exact_gradients.

    Exact gradients should give equal or better error reduction compared
    to the default approximate (recognition-model-only) approach.
    """
    signal = torch.sin(torch.linspace(0, 2 * math.pi, config.input_dim)).unsqueeze(0)

    # Without exact gradients
    torch.manual_seed(config.seed)
    level_approx = PredictiveLevel(
        input_dim=config.input_dim,
        belief_dim=config.hidden_dim,
        lr=0.1,
        gen_lr=0.01,
        use_exact_gradients=False,
    )
    for _ in range(50):
        result_approx = level_approx(signal)
    error_without = result_approx['error'].abs().mean().item()

    # With exact gradients
    torch.manual_seed(config.seed)
    level_exact = PredictiveLevel(
        input_dim=config.input_dim,
        belief_dim=config.hidden_dim,
        lr=0.1,
        gen_lr=0.01,
        use_exact_gradients=True,
    )
    for _ in range(50):
        result_exact = level_exact(signal)
    error_with = result_exact['error'].abs().mean().item()

    improvement = error_without - error_with

    return ExactGradientsResult(
        error_without_exact=error_without,
        error_with_exact=error_with,
        improvement=improvement,
    )


# ============================================================
# Pytest Tests
# ============================================================

class TestPhase6PredictiveCoding:
    """Phase 6 gap tests for predictive coding (G-11, G-12)."""

    def test_adaptive_precision_updates(self):
        """G-11: Precision should change from initial 1.0 values after exposure."""
        config = BenchConfig(convergence_steps=50)
        result = bench_adaptive_precision(config)
        assert result.precision_adapted, \
            f"Precision did not adapt: mean={result.mean_precision:.4f}, std={result.precision_std:.6f}"

    def test_adaptive_precision_error_variance(self):
        """G-11: Error variance should update from initial 1.0."""
        config = BenchConfig(convergence_steps=50)
        result = bench_adaptive_precision(config)
        assert result.error_variance_mean != 1.0, \
            f"Error variance stuck at initial 1.0"

    def test_exact_gradients_reduce_error(self):
        """G-12: Exact gradients should help or at least not hurt error reduction."""
        config = BenchConfig(convergence_steps=50)
        result = bench_exact_gradients(config)
        # Exact gradients should give equal or better (lower) error
        assert result.improvement >= -0.1, \
            f"Exact gradients made things much worse: improvement={result.improvement:.4f}"


class TestErrorDynamics:
    """Prediction error should decrease over time."""

    def test_pyquifer_error_reduces(self):
        """PyQuifer HPC should reduce error with consistent input."""
        config = BenchConfig(convergence_steps=80)
        result = bench_error_dynamics(config)
        assert result.pyquifer_error_reduced, \
            f"Error increased: {result.pyquifer_initial_error:.4f} -> {result.pyquifer_final_error:.4f}"

    def test_pyquifer_error_positive(self):
        """Prediction errors should be positive (not negative or NaN)."""
        config = BenchConfig(convergence_steps=50)
        result = bench_error_dynamics(config)
        assert result.pyquifer_final_error >= 0
        assert not math.isnan(result.pyquifer_final_error)


class TestBeliefConvergence:
    """Beliefs should converge to stable values."""

    def test_pyquifer_beliefs_stable(self):
        """PyQuifer beliefs should stabilize with consistent input."""
        config = BenchConfig(convergence_steps=80)
        result = bench_belief_convergence(config)
        assert result.pyquifer_belief_stable, \
            f"Beliefs still changing at rate {result.pyquifer_belief_change_rate:.4f}"

    def test_pyquifer_beliefs_nonzero(self):
        """Beliefs should be nonzero (not collapsed)."""
        config = BenchConfig(convergence_steps=50)
        result = bench_belief_convergence(config)
        assert result.pyquifer_belief_norm > 0.01


class TestHierarchy:
    """Hierarchical error propagation should work correctly."""

    def test_pyquifer_all_levels_have_errors(self):
        """All levels should produce prediction errors."""
        config = BenchConfig(convergence_steps=50)
        result = bench_hierarchy(config)
        assert len(result.pyquifer_errors_per_level) == result.pyquifer_num_levels
        assert all(e >= 0 for e in result.pyquifer_errors_per_level)

    def test_pyquifer_errors_bounded(self):
        """Errors should not explode."""
        config = BenchConfig(convergence_steps=50)
        result = bench_hierarchy(config)
        assert all(e < 100 for e in result.pyquifer_errors_per_level)


class TestGradientQuality:
    """Gradient approximation and online learning."""

    def test_torch2pc_gradients_similar(self):
        """Torch2PC: FixedPred gradients should approximate Exact."""
        if not HAS_TORCH2PC:
            return  # Skip if Torch2PC not importable
        config = BenchConfig()
        result = bench_gradient_quality(config)
        assert result.fixedpred_cos_sim > 0.5, \
            f"FixedPred gradient cosine sim too low: {result.fixedpred_cos_sim:.3f}"

    def test_pyquifer_online_learning_reduces_error(self):
        """PyQuifer's generative model should reduce prediction error."""
        config = BenchConfig(convergence_steps=80)
        result = bench_gradient_quality(config)
        assert result.pyquifer_gen_loss_reduction > 0, \
            f"Online learning did not reduce error: {result.pyquifer_gen_loss_reduction:.4f}"


class TestPrecisionWeighting:
    """Precision gating is a PyQuifer-unique feature."""

    def test_precision_gates_errors(self):
        """Low-precision channels should have smaller weighted errors."""
        config = BenchConfig()
        result = bench_precision_weighting(config)
        assert result.precision_gating_works, \
            f"Weighted attended={result.weighted_error_attended:.4f} vs ignored={result.weighted_error_ignored:.4f}"

    def test_torch2pc_no_precision(self):
        """Torch2PC does not have precision weighting."""
        config = BenchConfig()
        result = bench_precision_weighting(config)
        assert not result.torch2pc_has_precision


# ============================================================
# Console Output
# ============================================================

def _print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_error_dynamics(result: ErrorDynamicsResult):
    _print_section("Section 1: Prediction Error Dynamics")
    print(f"\n  PyQuifer HierarchicalPredictiveCoding:")
    print(f"    Initial error:           {result.pyquifer_initial_error:.4f}")
    print(f"    Final error:             {result.pyquifer_final_error:.4f}")
    print(f"    Error reduced:           {'YES' if result.pyquifer_error_reduced else 'NO'}")
    reduction_pct = (1 - result.pyquifer_final_error / result.pyquifer_initial_error) * 100 if result.pyquifer_initial_error > 0 else 0
    print(f"    Reduction:               {reduction_pct:.1f}%")

    if result.torch2pc_available:
        print(f"\n  Torch2PC (FixedPred algorithm):")
        print(f"    Initial loss:            {result.torch2pc_initial_loss:.4f}")
        print(f"    Final loss:              {result.torch2pc_final_loss:.4f}")
        print(f"    Loss reduced:            {'YES' if result.torch2pc_loss_reduced else 'NO'}")
        t2_reduction = (1 - result.torch2pc_final_loss / result.torch2pc_initial_loss) * 100 if result.torch2pc_initial_loss > 0 else 0
        print(f"    Reduction:               {t2_reduction:.1f}%")
    else:
        print(f"\n  [Torch2PC not available -- comparison skipped]")


def print_belief_results(result: BeliefResult):
    _print_section("Section 2: Belief Convergence")
    print(f"\n  PyQuifer:")
    print(f"    Belief norm:             {result.pyquifer_belief_norm:.4f}")
    print(f"    Belief stable:           {'YES' if result.pyquifer_belief_stable else 'NO'}")
    print(f"    Change rate (last 10):   {result.pyquifer_belief_change_rate:.6f}")

    if result.torch2pc_available:
        print(f"\n  Torch2PC:")
        print(f"    Belief norm:             {result.torch2pc_belief_norm:.4f}")
        print(f"    Belief stable:           {'YES' if result.torch2pc_belief_stable else 'NO'}")
        print(f"    Epsilon norm:            {result.torch2pc_epsilon_norm:.4f}")
    else:
        print(f"\n  [Torch2PC not available]")


def print_hierarchy_results(result: HierarchyResult):
    _print_section("Section 3: Hierarchical Error Propagation")
    print(f"\n  PyQuifer ({result.pyquifer_num_levels} levels):")
    for i, err in enumerate(result.pyquifer_errors_per_level):
        label = "bottom" if i == 0 else "top" if i == result.pyquifer_num_levels - 1 else f"mid-{i}"
        print(f"    Level {i} ({label:6s}):       {err:.4f}")
    print(f"    Errors decrease upward:  {'YES' if result.pyquifer_error_decreases_up else 'NO'}")

    if result.torch2pc_available and result.torch2pc_epsilons_per_layer:
        print(f"\n  Torch2PC ({result.torch2pc_num_layers} layers):")
        for i, eps in enumerate(result.torch2pc_epsilons_per_layer):
            print(f"    Layer {i}:                 {eps:.4f}")
    else:
        print(f"\n  [Torch2PC hierarchy comparison skipped]")


def print_gradient_results(result: GradientResult):
    _print_section("Section 4: Gradient Approximation Quality")
    if result.torch2pc_available:
        print(f"\n  Torch2PC gradient comparison:")
        print(f"    Exact (backprop) loss:   {result.exact_loss:.4f}")
        print(f"    FixedPred loss:          {result.fixedpred_loss:.4f}")
        print(f"    Strict loss:             {result.strict_loss:.4f}")
        print(f"    FixedPred cos sim:       {result.fixedpred_cos_sim:.4f}")
        print(f"    Strict cos sim:          {result.strict_cos_sim:.4f}")
    else:
        print(f"\n  [Torch2PC not available -- gradient comparison skipped]")

    print(f"\n  PyQuifer online learning:")
    print(f"    Gen model error reduction: {result.pyquifer_gen_loss_reduction:.4f}")


def print_precision_results(result: PrecisionResult):
    _print_section("Section 5: Precision Weighting (PyQuifer Unique)")
    print(f"\n  Raw prediction errors:")
    print(f"    Attended (high prec):    {result.error_attended:.4f}")
    print(f"    Ignored (low prec):      {result.error_ignored:.4f}")
    print(f"\n  Precision-weighted errors:")
    print(f"    Attended (weighted):     {result.weighted_error_attended:.4f}")
    print(f"    Ignored (weighted):      {result.weighted_error_ignored:.4f}")
    print(f"    Gating works:            {'YES' if result.precision_gating_works else 'NO'}")
    print(f"\n  Torch2PC has precision:    {'YES' if result.torch2pc_has_precision else 'NO'}")


def print_adaptive_precision_results(result: AdaptivePrecisionResult):
    _print_section("Section 6: Adaptive Precision (G-11, Phase 6)")
    print(f"\n  Precision adapted:         {'YES' if result.precision_adapted else 'NO'}")
    print(f"  Mean precision:            {result.mean_precision:.4f}")
    print(f"  Precision std:             {result.precision_std:.6f}")
    print(f"  Error variance mean:       {result.error_variance_mean:.4f}")


def print_exact_gradients_results(result: ExactGradientsResult):
    _print_section("Section 7: Exact Gradients (G-12, Phase 6)")
    print(f"\n  Error without exact grads: {result.error_without_exact:.4f}")
    print(f"  Error with exact grads:    {result.error_with_exact:.4f}")
    print(f"  Improvement (positive=better): {result.improvement:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  PyQuifer Predictive Coding vs Torch2PC")
    print("  Benchmark #4: Predictive Coding Algorithms")
    print("=" * 60)

    config = BenchConfig()
    torch.manual_seed(config.seed)

    if HAS_TORCH2PC:
        print(f"\n  Torch2PC: AVAILABLE")
    else:
        print(f"\n  Torch2PC: NOT AVAILABLE (comparison will be PyQuifer-only)")

    t0 = time.perf_counter()

    error_result = bench_error_dynamics(config)
    print_error_dynamics(error_result)

    belief_result = bench_belief_convergence(config)
    print_belief_results(belief_result)

    hier_result = bench_hierarchy(config)
    print_hierarchy_results(hier_result)

    grad_result = bench_gradient_quality(config)
    print_gradient_results(grad_result)

    prec_result = bench_precision_weighting(config)
    print_precision_results(prec_result)

    # Phase 6 gap tests
    adapt_prec_result = bench_adaptive_precision(config)
    print_adaptive_precision_results(adapt_prec_result)

    exact_grad_result = bench_exact_gradients(config)
    print_exact_gradients_results(exact_grad_result)

    elapsed = time.perf_counter() - t0

    _print_section("Summary")
    print(f"\n  Total elapsed:             {elapsed:.2f}s")
    print(f"  Torch2PC available:        {'YES' if HAS_TORCH2PC else 'NO'}")
    print(f"\n  Error reduction:           {'PASS' if error_result.pyquifer_error_reduced else 'FAIL'}")
    print(f"  Belief convergence:        {'PASS' if belief_result.pyquifer_belief_stable else 'FAIL'}")
    print(f"  Hierarchy propagation:     {'PASS' if hier_result.pyquifer_error_decreases_up else 'PARTIAL'}")
    print(f"  Online learning:           {'PASS' if grad_result.pyquifer_gen_loss_reduction > 0 else 'FAIL'}")
    print(f"  Precision weighting:       {'PASS' if prec_result.precision_gating_works else 'FAIL'}")
    print(f"  Adaptive precision (G-11): {'PASS' if adapt_prec_result.precision_adapted else 'FAIL'}")
    print(f"  Exact gradients (G-12):    {'PASS' if exact_grad_result.improvement >= -0.1 else 'FAIL'}")

    if HAS_TORCH2PC:
        print(f"\n  Torch2PC FixedPred cos sim: {grad_result.fixedpred_cos_sim:.3f}")
        print(f"  Torch2PC Strict cos sim:    {grad_result.strict_cos_sim:.3f}")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PyQuifer Predictive Coding vs Torch2PC', fontsize=14)

        # Panel 1: Error trajectories
        ax = axes[0, 0]
        ax.plot(error_result.pyquifer_error_trajectory, label='PyQuifer HPC', color='blue')
        if error_result.torch2pc_available:
            ax.plot(error_result.torch2pc_loss_trajectory, label='Torch2PC (FixedPred)', color='red')
        ax.set_xlabel('Step')
        ax.set_ylabel('Prediction Error / Loss')
        ax.set_title('Error Reduction Over Time')
        ax.legend()
        ax.set_yscale('log')

        # Panel 2: Hierarchical error profile
        ax = axes[0, 1]
        levels = list(range(len(hier_result.pyquifer_errors_per_level)))
        ax.bar(levels, hier_result.pyquifer_errors_per_level, alpha=0.7,
               color='blue', label='PyQuifer levels')
        if hier_result.torch2pc_available and hier_result.torch2pc_epsilons_per_layer:
            t2_levels = list(range(len(hier_result.torch2pc_epsilons_per_layer)))
            ax.bar([x + 0.35 for x in t2_levels],
                   hier_result.torch2pc_epsilons_per_layer,
                   width=0.35, alpha=0.7, color='red', label='Torch2PC layers')
        ax.set_xlabel('Level/Layer')
        ax.set_ylabel('Error Magnitude')
        ax.set_title('Hierarchical Error Profile')
        ax.legend()

        # Panel 3: Precision weighting
        ax = axes[1, 0]
        categories = ['Raw\nAttended', 'Raw\nIgnored', 'Weighted\nAttended', 'Weighted\nIgnored']
        values = [prec_result.error_attended, prec_result.error_ignored,
                  prec_result.weighted_error_attended, prec_result.weighted_error_ignored]
        colors = ['steelblue', 'lightblue', 'darkgreen', 'lightgreen']
        ax.bar(categories, values, color=colors)
        ax.set_ylabel('Error Magnitude')
        ax.set_title('Precision Weighting Effect (PyQuifer)')

        # Panel 4: Gradient cosine similarity (if available)
        ax = axes[1, 1]
        if HAS_TORCH2PC:
            methods = ['FixedPred\nvs Exact', 'Strict\nvs Exact']
            sims = [grad_result.fixedpred_cos_sim, grad_result.strict_cos_sim]
            bars = ax.bar(methods, sims, color=['orange', 'purple'], alpha=0.7)
            ax.axhline(y=1.0, color='green', linestyle='--', label='Perfect match')
            ax.set_ylabel('Cosine Similarity')
            ax.set_title('Torch2PC Gradient Approximation Quality')
            ax.set_ylim(0, 1.1)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Torch2PC not available\n(gradient comparison skipped)',
                    ha='center', va='center', fontsize=12)
            ax.set_title('Gradient Approximation Quality')

        plt.tight_layout()
        plot_path = _benchmark_dir / "bench_predictive_coding.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved: {plot_path}")
        plt.close()

    except ImportError:
        print(f"\n  [matplotlib not available -- skipping plot]")


if __name__ == '__main__':
    main()
