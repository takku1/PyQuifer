"""
Adaptive ODE Solvers for PyQuifer

Configurable integration backends for all continuous dynamics: Kuramoto oscillators,
Liquid Time-Constant cells, diffusion processes, and Wilson-Cowan neural mass models.

Provides three solver tiers:
1. EulerSolver — fixed-step forward Euler (lowest cost, O(dt) local error)
2. RK4Solver — classical 4th-order Runge-Kutta (fixed-step, O(dt^5) local error)
3. DopriSolver — Dormand-Prince 4(5) adaptive step-size with embedded error estimate

All solvers operate on batched tensors and accept RHS functions of the form
    f(t: float, y: Tensor) -> Tensor
following the torchdyn / torchode convention where t is a scalar and y has
arbitrary batch + state dimensions.

The functional API ``solve_ivp()`` dispatches to the configured solver and
returns the full trajectory (or just the final state if ``dense=False``).

References:
- Dormand & Prince (1980). A family of embedded Runge-Kutta formulae.
- Hairer, Norsett & Wanner (1993). Solving Ordinary Differential Equations I.
- Chen et al. (2018). Neural Ordinary Differential Equations.
- Lienen & Gunnemann (2022). torchode: A Parallel ODE Solver for PyTorch.
- Kidger (2022). On Neural Differential Equations. PhD thesis.
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Type alias for the RHS function: f(t, y) -> dy/dt
# t is a Python float (or scalar tensor), y is a batched state tensor.
# ---------------------------------------------------------------------------
RHSFunc = Callable[[float, torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SolverConfig:
    """Configuration for ODE solvers.

    Attributes:
        method: Integration method. One of 'euler', 'rk4', 'dopri'.
        atol: Absolute tolerance for adaptive step-size control (DopriSolver).
        rtol: Relative tolerance for adaptive step-size control (DopriSolver).
        min_step: Minimum allowed step size (prevents infinite refinement).
        max_step: Maximum allowed step size (prevents overshoot).
        max_num_steps: Safety limit on total steps to prevent infinite loops.
        first_step: Initial step size for adaptive solvers. If None, estimated
            automatically from the initial slope.
        safety: Safety factor for step-size controller (typical 0.8-0.95).
        dtype: Torch dtype for internal computations. If None, inferred from y0.
    """
    method: Literal['euler', 'rk4', 'dopri'] = 'euler'
    atol: float = 1e-6
    rtol: float = 1e-3
    min_step: float = 1e-10
    max_step: float = 1.0
    max_num_steps: int = 10_000
    first_step: Optional[float] = None
    safety: float = 0.9
    dtype: Optional[torch.dtype] = None

    # ── Presets ──

    @classmethod
    def fast(cls) -> 'SolverConfig':
        """Euler with large fixed step — cheapest option for real-time tick."""
        return cls(method='euler')

    @classmethod
    def balanced(cls) -> 'SolverConfig':
        """RK4 with moderate tolerance — good accuracy/cost tradeoff."""
        return cls(method='rk4')

    @classmethod
    def accurate(cls) -> 'SolverConfig':
        """Dormand-Prince adaptive — highest accuracy, variable cost."""
        return cls(method='dopri', atol=1e-8, rtol=1e-6)

    @classmethod
    def realtime(cls) -> 'SolverConfig':
        """Dormand-Prince with loose tolerance — adaptive but fast."""
        return cls(method='dopri', atol=1e-3, rtol=1e-2, max_num_steps=500)


# ---------------------------------------------------------------------------
# Solver Result
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    """Output of an ODE solve.

    Attributes:
        t: Tensor of evaluation times, shape ``(num_points,)``.
        y: Tensor of states at each time, shape ``(num_points, *state_shape)``.
        n_steps: Total number of RHS evaluations performed.
        n_accepted: Number of accepted steps (equals n_steps for fixed-step).
        n_rejected: Number of rejected steps (0 for fixed-step solvers).
    """
    t: torch.Tensor
    y: torch.Tensor
    n_steps: int = 0
    n_accepted: int = 0
    n_rejected: int = 0


# ---------------------------------------------------------------------------
# Base Solver
# ---------------------------------------------------------------------------

class BaseSolver(nn.Module):
    """Abstract base class for ODE solvers.

    Subclasses must implement ``_step()`` which advances the state by one
    solver step.  The ``solve()`` method handles the outer integration loop,
    trajectory storage, and step-size bookkeeping.

    Inherits ``nn.Module`` so that any learnable RHS parameters are properly
    tracked and the solver participates in ``torch.compile`` graphs.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        super().__init__()
        self.config = config or SolverConfig()

    # ── Interface ──

    def _step(
        self,
        func: RHSFunc,
        t: float,
        y: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Advance state by one step.

        Args:
            func: RHS function f(t, y) -> dy/dt.
            t: Current time (scalar).
            y: Current state tensor (arbitrary batch dims).
            dt: Step size.

        Returns:
            Tuple of (y_next, error_estimate).
            ``error_estimate`` is None for fixed-step solvers.
        """
        raise NotImplementedError

    # ── Outer loop ──

    def solve(
        self,
        func: RHSFunc,
        y0: torch.Tensor,
        t_span: Tuple[float, float],
        t_eval: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
    ) -> SolverResult:
        """Integrate the ODE y' = f(t, y) over ``t_span``.

        Args:
            func: RHS function f(t, y) -> dy/dt.
            y0: Initial state, shape ``(*batch, state_dim)``.
            t_span: ``(t0, t1)`` integration interval.
            t_eval: Optional specific times at which to record the solution.
                If None, the solver records at every internal step.
            dt: Fixed step size. Required for Euler and RK4. Ignored by Dopri
                (which determines its own step size).

        Returns:
            SolverResult containing the trajectory.
        """
        t0, t1 = t_span
        if t1 <= t0:
            raise ValueError(f"t_span must satisfy t1 > t0, got ({t0}, {t1})")

        # Default step size for fixed-step solvers
        if dt is None:
            dt = (t1 - t0) / 100.0

        return self._integrate(func, y0, t0, t1, dt, t_eval)

    def _integrate(
        self,
        func: RHSFunc,
        y0: torch.Tensor,
        t0: float,
        t1: float,
        dt: float,
        t_eval: Optional[torch.Tensor],
    ) -> SolverResult:
        """Fixed-step integration loop (overridden by adaptive solvers)."""
        t_list: List[float] = [t0]
        y_list: List[torch.Tensor] = [y0]
        y = y0
        t = t0
        n_steps = 0

        while t < t1 - 1e-12:
            # Clamp last step to hit t1 exactly
            step = min(dt, t1 - t)
            y, _ = self._step(func, t, y, step)
            t = t + step
            n_steps += 1
            t_list.append(t)
            y_list.append(y)

            if n_steps > self.config.max_num_steps:
                break

        t_tensor = torch.tensor(t_list, dtype=y0.dtype, device=y0.device)
        y_tensor = torch.stack(y_list, dim=0)

        # Interpolate to t_eval if requested
        if t_eval is not None:
            y_tensor = self._interpolate_to_eval(
                t_tensor, y_tensor, t_eval
            )
            t_tensor = t_eval

        return SolverResult(
            t=t_tensor,
            y=y_tensor,
            n_steps=n_steps,
            n_accepted=n_steps,
            n_rejected=0,
        )

    @staticmethod
    def _interpolate_to_eval(
        t_all: torch.Tensor,
        y_all: torch.Tensor,
        t_eval: torch.Tensor,
    ) -> torch.Tensor:
        """Linear interpolation of trajectory to requested evaluation times.

        Args:
            t_all: ``(N,)`` monotonically increasing times from the solver.
            y_all: ``(N, *state_shape)`` states at those times.
            t_eval: ``(M,)`` desired evaluation times.

        Returns:
            ``(M, *state_shape)`` interpolated states.
        """
        out = []
        idx = 0
        n = t_all.shape[0]
        for te in t_eval:
            te_val = te.item()
            # Advance index to bracket te
            while idx < n - 2 and t_all[idx + 1].item() < te_val:
                idx += 1
            t_lo = t_all[idx].item()
            t_hi = t_all[idx + 1].item() if idx < n - 1 else t_lo
            if abs(t_hi - t_lo) < 1e-15:
                out.append(y_all[idx])
            else:
                alpha = (te_val - t_lo) / (t_hi - t_lo)
                alpha = max(0.0, min(1.0, alpha))
                y_interp = y_all[idx] * (1.0 - alpha) + y_all[idx + 1] * alpha
                out.append(y_interp)
        return torch.stack(out, dim=0)

    def forward(
        self,
        func: RHSFunc,
        y0: torch.Tensor,
        t_span: Tuple[float, float],
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """nn.Module forward — returns final state only (no trajectory).

        This is the primary entry point for use inside larger ``nn.Module``
        pipelines (Kuramoto, LTC, etc.) where only the endpoint matters.

        Args:
            func: RHS function f(t, y) -> dy/dt.
            y0: Initial state.
            t_span: ``(t0, t1)`` integration interval.
            dt: Step size (fixed-step solvers only).

        Returns:
            Final state y(t1).
        """
        result = self.solve(func, y0, t_span, dt=dt)
        return result.y[-1]


# ---------------------------------------------------------------------------
# Euler Solver
# ---------------------------------------------------------------------------

class EulerSolver(BaseSolver):
    """Forward Euler integrator (first-order, fixed-step).

    Wraps the fixed-step Euler method currently used throughout PyQuifer
    into the unified solver interface.

    Local truncation error: O(dt^2)
    Global error: O(dt)
    Cost per step: 1 RHS evaluation

    This is the cheapest solver and should be preferred for real-time
    tick loops where latency matters more than accuracy.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        cfg = config or SolverConfig(method='euler')
        super().__init__(cfg)

    def _step(
        self,
        func: RHSFunc,
        t: float,
        y: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, None]:
        """Single forward Euler step: y_{n+1} = y_n + dt * f(t_n, y_n)."""
        dy = func(t, y)
        y_next = y + dt * dy
        return y_next, None


# ---------------------------------------------------------------------------
# RK4 Solver
# ---------------------------------------------------------------------------

class RK4Solver(BaseSolver):
    """Classical 4th-order Runge-Kutta integrator (fixed-step).

    The workhorse method — excellent accuracy for smooth dynamics with
    moderate step sizes.  Matches the existing ``_rk4_step`` in
    ``oscillators.py`` but generalized to the ``f(t, y)`` convention.

    Local truncation error: O(dt^5)
    Global error: O(dt^4)
    Cost per step: 4 RHS evaluations

    References:
    - Kutta (1901). Beitrag zur naherungsweisen Integration totaler
      Differentialgleichungen.
    - Pre-allocated buffer style per torchode (Lienen 2022).
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        cfg = config or SolverConfig(method='rk4')
        super().__init__(cfg)

    def _step(
        self,
        func: RHSFunc,
        t: float,
        y: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, None]:
        """Classical RK4 step with 4 stage evaluations."""
        k1 = func(t, y)
        k2 = func(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = func(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = func(t + dt, y + dt * k3)
        y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return y_next, None


# ---------------------------------------------------------------------------
# Dormand-Prince 4(5) Adaptive Solver
# ---------------------------------------------------------------------------

# Butcher tableau for Dormand-Prince 4(5)
# 7 stages, FSAL (First Same As Last) property — effectively 6 evals per step
_DOPRI_A = [
    [],
    [1.0 / 5.0],
    [3.0 / 40.0, 9.0 / 40.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
    [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0],
    [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0],
    [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0],
]

# 5th-order weights (for the solution)
_DOPRI_B5 = [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0,
             -2187.0 / 6784.0, 11.0 / 84.0, 0.0]

# 4th-order weights (for the error estimate)
_DOPRI_B4 = [5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0,
             -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0]

# Node positions (c_i)
_DOPRI_C = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0]

# Error coefficients: e_i = b5_i - b4_i
_DOPRI_E = [b5 - b4 for b5, b4 in zip(_DOPRI_B5, _DOPRI_B4)]


class DopriSolver(BaseSolver):
    """Dormand-Prince 4(5) adaptive step-size solver.

    An embedded Runge-Kutta pair that provides both a 4th-order and 5th-order
    solution simultaneously.  The difference is used as an error estimate to
    control step size automatically.

    Features:
    - FSAL (First Same As Last): k7 of the current step = k1 of the next step,
      so effectively only 6 function evaluations per accepted step.
    - PI step-size controller with safety factor for smooth step adaptation.
    - Configurable absolute and relative tolerances (atol, rtol).
    - Hard limits on minimum/maximum step size to prevent pathological behavior.

    Local truncation error: O(dt^6) (5th-order solution used for propagation)
    Cost per step: 6 RHS evaluations (FSAL), plus rejected step overhead.

    References:
    - Dormand & Prince (1980). A family of embedded Runge-Kutta formulae.
      J. Comput. Appl. Math. 6(1), 19-26.
    - Hairer, Norsett & Wanner (1993). Solving Ordinary Differential
      Equations I: Nonstiff Problems. Springer, Ch. II.4.
    - Shampine (1986). Some practical Runge-Kutta formulas.
      Mathematics of Computation 46(173), 135-150.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        cfg = config or SolverConfig(method='dopri')
        super().__init__(cfg)

        # Register Butcher tableau as buffers for device/dtype consistency
        # These are registered once and moved with .to(device)
        self.register_buffer(
            '_b5', torch.tensor(_DOPRI_B5, dtype=torch.float64), persistent=False
        )
        self.register_buffer(
            '_e', torch.tensor(_DOPRI_E, dtype=torch.float64), persistent=False
        )

        # FSAL: cache k1 from previous step (set during integration)
        self._fsal_k: Optional[torch.Tensor] = None

    def _step(
        self,
        func: RHSFunc,
        t: float,
        y: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single Dormand-Prince step with embedded error estimate.

        Returns:
            (y_next_5th, error_estimate) where error_estimate has the same
            shape as y.
        """
        # Stage evaluations
        k = [None] * 7

        # FSAL: reuse k1 from previous step if available
        if self._fsal_k is not None:
            k[0] = self._fsal_k
        else:
            k[0] = func(t, y)

        # Stages 2-6
        for i in range(1, 6):
            t_i = t + _DOPRI_C[i] * dt
            y_i = y.clone()
            for j in range(i):
                y_i = y_i + (dt * _DOPRI_A[i][j]) * k[j]
            k[i] = func(t_i, y_i)

        # 5th-order solution (propagated)
        y5 = y.clone()
        for i in range(6):
            if _DOPRI_B5[i] != 0.0:
                y5 = y5 + (dt * _DOPRI_B5[i]) * k[i]

        # Stage 7 (FSAL — this becomes k1 of the next step if accepted)
        k[6] = func(t + dt, y5)

        # Error estimate: sum of e_i * k_i * dt
        err = torch.zeros_like(y)
        for i in range(7):
            if _DOPRI_E[i] != 0.0:
                err = err + (_DOPRI_E[i] * dt) * k[i]

        # Cache k7 for FSAL
        self._fsal_k = k[6]

        return y5, err

    def _compute_error_norm(
        self,
        err: torch.Tensor,
        y: torch.Tensor,
        y_next: torch.Tensor,
    ) -> float:
        """Compute the scaled error norm for step-size control.

        Uses the mixed absolute/relative tolerance:
            sc_i = atol + rtol * max(|y_i|, |y_next_i|)
            error_norm = sqrt(mean((err_i / sc_i)^2))

        This is the standard Hairer-Norsett-Wanner error norm.
        """
        atol = self.config.atol
        rtol = self.config.rtol
        scale = atol + rtol * torch.max(y.abs(), y_next.abs())
        # Flatten for RMS computation across all dims
        ratio = (err / scale).flatten()
        return math.sqrt((ratio * ratio).mean().item())

    def _initial_step_size(
        self,
        func: RHSFunc,
        t0: float,
        y0: torch.Tensor,
    ) -> float:
        """Estimate a good initial step size (Hairer algorithm).

        Uses the norm of the initial slope and second derivative estimate
        to pick a step that keeps the local error near tolerance.
        """
        f0 = func(t0, y0)
        d0 = torch.max(y0.abs().max(), torch.tensor(1e-5, device=y0.device)).item()
        d1 = torch.max(f0.abs().max(), torch.tensor(1e-5, device=y0.device)).item()

        h0 = 0.01 * d0 / d1
        h0 = min(h0, self.config.max_step)
        h0 = max(h0, self.config.min_step)

        # Estimate second derivative
        y1 = y0 + h0 * f0
        f1 = func(t0 + h0, y1)
        d2 = ((f1 - f0).abs().max().item()) / h0

        if max(d1, d2) <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            # h1 = (0.01 / max(d1, d2)) ^ (1/5)
            h1 = (0.01 / max(d1, d2)) ** 0.2

        return min(100.0 * h0, h1, self.config.max_step)

    def _integrate(
        self,
        func: RHSFunc,
        y0: torch.Tensor,
        t0: float,
        t1: float,
        dt: float,
        t_eval: Optional[torch.Tensor],
    ) -> SolverResult:
        """Adaptive integration loop with PI step-size control."""
        # Reset FSAL cache
        self._fsal_k = None

        # Determine initial step size
        if self.config.first_step is not None:
            h = self.config.first_step
        else:
            h = self._initial_step_size(func, t0, y0)

        h = min(h, t1 - t0, self.config.max_step)
        h = max(h, self.config.min_step)

        t_list: List[float] = [t0]
        y_list: List[torch.Tensor] = [y0]
        y = y0
        t = t0
        n_steps = 0
        n_accepted = 0
        n_rejected = 0

        safety = self.config.safety
        min_step = self.config.min_step
        max_step = self.config.max_step
        max_num_steps = self.config.max_num_steps

        while t < t1 - 1e-12 and n_steps < max_num_steps:
            # Clamp step to not overshoot t1
            h = min(h, t1 - t)
            h = max(h, min_step)

            y_next, err = self._step(func, t, y, h)
            n_steps += 1

            # Compute error norm
            err_norm = self._compute_error_norm(err, y, y_next)

            if err_norm <= 1.0:
                # ── Accept step ──
                t = t + h
                y = y_next
                n_accepted += 1
                t_list.append(t)
                y_list.append(y)

                # PI step-size controller (Hairer, Ch. II.4)
                # h_new = h * safety * err_norm^(-1/5)
                if err_norm < 1e-15:
                    factor = 5.0  # Maximum growth
                else:
                    factor = safety * err_norm ** (-0.2)
                # Clamp growth/shrink to [0.2, 5.0]
                factor = max(0.2, min(5.0, factor))
                h = h * factor
                h = min(h, max_step)
                h = max(h, min_step)
            else:
                # ── Reject step ──
                n_rejected += 1
                # Invalidate FSAL cache
                self._fsal_k = None

                # Shrink step size
                if err_norm < 1e-15:
                    factor = 0.5
                else:
                    factor = safety * err_norm ** (-0.2)
                factor = max(0.2, min(1.0, factor))
                h = h * factor
                h = max(h, min_step)

        t_tensor = torch.tensor(t_list, dtype=y0.dtype, device=y0.device)
        y_tensor = torch.stack(y_list, dim=0)

        # Interpolate to t_eval if requested
        if t_eval is not None:
            y_tensor = self._interpolate_to_eval(t_tensor, y_tensor, t_eval)
            t_tensor = t_eval

        return SolverResult(
            t=t_tensor,
            y=y_tensor,
            n_steps=n_steps,
            n_accepted=n_accepted,
            n_rejected=n_rejected,
        )


# ---------------------------------------------------------------------------
# Solver Factory
# ---------------------------------------------------------------------------

def create_solver(config: Optional[SolverConfig] = None) -> BaseSolver:
    """Create a solver instance from a SolverConfig.

    Args:
        config: Solver configuration. Defaults to EulerSolver if None.

    Returns:
        Instantiated solver matching ``config.method``.

    Raises:
        ValueError: If ``config.method`` is not recognized.
    """
    if config is None:
        config = SolverConfig()

    if config.method == 'euler':
        return EulerSolver(config)
    elif config.method == 'rk4':
        return RK4Solver(config)
    elif config.method == 'dopri':
        return DopriSolver(config)
    else:
        raise ValueError(
            f"Unknown solver method '{config.method}'. "
            f"Choose from: 'euler', 'rk4', 'dopri'."
        )


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def solve_ivp(
    func: RHSFunc,
    y0: torch.Tensor,
    t_span: Tuple[float, float],
    config: Optional[SolverConfig] = None,
    t_eval: Optional[torch.Tensor] = None,
    dt: Optional[float] = None,
) -> SolverResult:
    """Solve an initial value problem y' = f(t, y), y(t0) = y0.

    Functional API that creates the appropriate solver from ``config`` and
    integrates over ``t_span``.  This is the recommended entry point for
    one-shot integration (as opposed to embedding a solver in an nn.Module).

    Args:
        func: Right-hand side function ``f(t, y) -> dy/dt``.
            * ``t``: scalar float (current time).
            * ``y``: tensor of shape ``(*batch, state_dim)``.
            * Returns: tensor of same shape as ``y``.
        y0: Initial state tensor, shape ``(*batch, state_dim)``.
        t_span: ``(t0, t1)`` integration interval.
        config: Solver configuration. Defaults to Euler if None.
        t_eval: Optional tensor of times at which to record the solution.
            Shape ``(num_eval_points,)``.  If None, the solver records at
            every internal step.
        dt: Fixed step size for Euler / RK4.  Ignored by Dopri.
            If None, defaults to ``(t1 - t0) / 100``.

    Returns:
        SolverResult with fields ``t``, ``y``, ``n_steps``, ``n_accepted``,
        ``n_rejected``.

    Example::

        >>> import torch
        >>> from pyquifer.ode_solvers import solve_ivp, SolverConfig
        >>>
        >>> # Simple exponential decay: dy/dt = -y
        >>> def decay(t, y):
        ...     return -y
        >>>
        >>> y0 = torch.tensor([1.0, 2.0, 3.0])
        >>> result = solve_ivp(decay, y0, (0.0, 1.0),
        ...                    config=SolverConfig.balanced(), dt=0.01)
        >>> print(result.y[-1])  # Should be ~ [0.368, 0.736, 1.104]
    """
    solver = create_solver(config)
    # Move solver buffers to match y0 device
    solver = solver.to(y0.device)
    return solver.solve(func, y0, t_span, t_eval=t_eval, dt=dt)


# ---------------------------------------------------------------------------
# Convenience: wrap existing _rk4_step signature
# ---------------------------------------------------------------------------

def rk4_step(f: Callable[[torch.Tensor], torch.Tensor],
             y: torch.Tensor,
             dt: float) -> torch.Tensor:
    """Drop-in replacement for ``oscillators._rk4_step``.

    Accepts the legacy ``f(y) -> dy`` signature (no time argument) used by
    LearnableKuramotoBank and WilsonCowanPopulation.

    Args:
        f: RHS function ``f(y) -> dy/dt`` (no time parameter).
        y: Current state tensor.
        dt: Timestep.

    Returns:
        Updated state ``y_{n+1}``.

    This is a thin wrapper — for new code, prefer ``solve_ivp`` or the
    solver classes directly.
    """
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def euler_step(f: Callable[[torch.Tensor], torch.Tensor],
               y: torch.Tensor,
               dt: float) -> torch.Tensor:
    """Single Euler step with legacy ``f(y)`` signature.

    Args:
        f: RHS function ``f(y) -> dy/dt`` (no time parameter).
        y: Current state tensor.
        dt: Timestep.

    Returns:
        Updated state ``y_{n+1} = y + dt * f(y)``.
    """
    return y + dt * f(y)
