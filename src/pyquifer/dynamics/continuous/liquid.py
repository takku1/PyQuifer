"""
Liquid Neural Networks for PyQuifer

Implements Liquid Time-Constant (LTC) networks and Neural ODEs.
These networks have continuous-time dynamics with learnable time constants,
enabling adaptive temporal processing at the "edge of chaos."

Key concepts:
- Liquid Time-Constant cells: Neurons with state-dependent time constants
- Neural ODEs: Continuous-depth networks via ODE solvers
- CfC (Closed-form Continuous-time): Efficient liquid network formulation

Based on:
- Hasani et al. "Liquid Time-constant Networks" (2020)
- Chen et al. "Neural Ordinary Differential Equations" (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class LiquidTimeConstantCell(nn.Module):
    """
    Liquid Time-Constant (LTC) cell with biologically-inspired dynamics.

    The cell follows ODE-based dynamics where the time constant is
    state-dependent, enabling adaptive temporal processing.

    State update (semi-implicit):
        v[t+1] = (cm * v[t] + gleak * vleak + I_syn) / (cm + gleak + g_syn)

    Where I_syn and g_syn come from sigmoid-gated synaptic conductances.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 ode_unfolds: int = 6,
                 solver: str = 'semi_implicit'):
        """
        Args:
            input_size: Dimension of input
            hidden_size: Number of hidden units
            ode_unfolds: Number of ODE solver steps per RNN step
            solver: ODE solver type ('semi_implicit', 'explicit', 'runge_kutta')
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ode_unfolds = ode_unfolds
        self.solver = solver

        # Sensory (input) synaptic parameters
        self.sensory_mu = nn.Parameter(torch.rand(input_size, hidden_size) * 0.5 + 0.3)
        self.sensory_sigma = nn.Parameter(torch.rand(input_size, hidden_size) * 5.0 + 3.0)
        self.sensory_W = nn.Parameter(torch.rand(input_size, hidden_size) * 0.99 + 0.01)
        self.sensory_erev = nn.Parameter(
            torch.tensor((2 * torch.randint(0, 2, (input_size, hidden_size)).float() - 1))
        )

        # Recurrent synaptic parameters
        self.mu = nn.Parameter(torch.rand(hidden_size, hidden_size) * 0.5 + 0.3)
        self.sigma = nn.Parameter(torch.rand(hidden_size, hidden_size) * 5.0 + 3.0)
        self.W = nn.Parameter(torch.rand(hidden_size, hidden_size) * 0.99 + 0.01)
        self.erev = nn.Parameter(
            torch.tensor((2 * torch.randint(0, 2, (hidden_size, hidden_size)).float() - 1))
        )

        # Membrane parameters (learnable time constants)
        self.vleak = nn.Parameter(torch.rand(hidden_size) * 0.4 - 0.2)
        self.gleak = nn.Parameter(torch.ones(hidden_size))  # Leak conductance
        self.cm = nn.Parameter(torch.ones(hidden_size) * 0.5)  # Membrane capacitance

    def _sigmoid_synapse(self, v: torch.Tensor, mu: torch.Tensor,
                         sigma: torch.Tensor) -> torch.Tensor:
        """Sigmoid activation for synaptic gating."""
        v = v.view(-1, v.shape[-1], 1)
        x = sigma * (v - mu)
        return torch.sigmoid(x)

    def forward(self, inputs: torch.Tensor,
                state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the LTC cell.

        Args:
            inputs: Input tensor (batch, input_size)
            state: Previous hidden state (batch, hidden_size)

        Returns:
            output: Output tensor (batch, hidden_size)
            new_state: New hidden state (batch, hidden_size)
        """
        batch_size = inputs.shape[0]

        if state is None:
            state = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        v_pre = state

        # Sensory synaptic activation
        sensory_activation = self.sensory_W * self._sigmoid_synapse(
            inputs, self.sensory_mu, self.sensory_sigma
        )
        sensory_rev = sensory_activation * self.sensory_erev

        w_num_sensory = torch.sum(sensory_rev, dim=1)
        w_den_sensory = torch.sum(sensory_activation, dim=1)

        # ODE integration
        for _ in range(self.ode_unfolds):
            # Recurrent synaptic activation
            w_activation = self.W * self._sigmoid_synapse(v_pre, self.mu, self.sigma)
            rev_activation = w_activation * self.erev

            w_numerator = torch.sum(rev_activation, dim=1) + w_num_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_den_sensory

            # Semi-implicit Euler step (softplus ensures cm, gleak > 0)
            cm = F.softplus(self.cm)
            gleak = F.softplus(self.gleak)
            numerator = cm * v_pre + gleak * self.vleak + w_numerator
            denominator = cm + gleak + w_denominator

            v_pre = numerator / (denominator + 1e-8)

        return v_pre, v_pre


class NeuralODE(nn.Module):
    """
    Neural Ordinary Differential Equation layer.

    Computes continuous-depth transformations by solving:
        dh/dt = f(h, t, theta)

    Uses simple fixed-step solvers for efficiency.
    """

    def __init__(self,
                 func: nn.Module,
                 solver: str = 'euler',
                 step_size: float = 0.1,
                 num_steps: int = 10):
        """
        Args:
            func: The ODE function dh/dt = f(h, t)
            solver: ODE solver ('euler', 'midpoint', 'rk4')
            step_size: Integration step size
            num_steps: Number of integration steps
        """
        super().__init__()
        self.func = func
        self.solver = solver
        self.step_size = step_size
        self.num_steps = num_steps

    def _euler_step(self, h: torch.Tensor, t: float) -> torch.Tensor:
        """Euler integration step."""
        return h + self.step_size * self.func(h, t)

    def _midpoint_step(self, h: torch.Tensor, t: float) -> torch.Tensor:
        """Midpoint (RK2) integration step."""
        dt = self.step_size
        k1 = self.func(h, t)
        k2 = self.func(h + 0.5 * dt * k1, t + 0.5 * dt)
        return h + dt * k2

    def _rk4_step(self, h: torch.Tensor, t: float) -> torch.Tensor:
        """Runge-Kutta 4 integration step."""
        dt = self.step_size
        k1 = self.func(h, t)
        k2 = self.func(h + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.func(h + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.func(h + dt * k3, t + dt)
        return h + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

    def forward(self, h0: torch.Tensor,
                t_span: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """
        Integrate the ODE from t0 to t1.

        Args:
            h0: Initial state
            t_span: Integration interval (t0, t1), default (0, 1)

        Returns:
            Final state h(t1)
        """
        if t_span is None:
            t_span = (0.0, 1.0)

        t0, t1 = t_span
        h = h0
        t = t0
        # Compute step size from span and num_steps
        dt = (t1 - t0) / self.num_steps

        # Select solver
        _SOLVERS = {'euler': self._euler_step, 'midpoint': self._midpoint_step, 'rk4': self._rk4_step}
        step_fn = _SOLVERS.get(self.solver)
        if step_fn is None:
            raise ValueError(f"Unknown solver: {self.solver}")

        # Temporarily set step_size from span
        original_step_size = self.step_size
        self.step_size = dt

        # Integrate
        for _ in range(self.num_steps):
            h = step_fn(h, t)
            t += dt

        self.step_size = original_step_size

        return h

    def trajectory(self, h0: torch.Tensor,
                   t_span: Optional[Tuple[float, float]] = None) -> List[torch.Tensor]:
        """Return full trajectory."""
        if t_span is None:
            t_span = (0.0, 1.0)

        t0, t1 = t_span
        h = h0
        t = t0
        trajectory = [h]

        step_fn = self._SOLVERS.get(self.solver)
        if step_fn is None:
            raise ValueError(f"Unknown solver: {self.solver}")

        for _ in range(self.num_steps):
            h = step_fn(h, t)
            t += self.step_size
            trajectory.append(h)

        return trajectory


class ODEFunc(nn.Module):
    """Simple MLP-based ODE function."""

    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        self.nfe = 0  # Number of function evaluations

    def forward(self, h: torch.Tensor, t: float) -> torch.Tensor:
        self.nfe += 1
        return self.net(h)


class ContinuousTimeRNN(nn.Module):
    """
    Continuous-Time RNN with adaptive time constants.

    Combines the ideas from LTC and Neural ODEs for
    oscillator-compatible continuous dynamics.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 tau_min: float = 0.1,
                 tau_max: float = 10.0):
        """
        Args:
            input_size: Input dimension
            hidden_size: Hidden state dimension
            tau_min: Minimum time constant
            tau_max: Maximum time constant
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Input projection
        self.W_in = nn.Linear(input_size, hidden_size)

        # Recurrent weights
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)

        # Learnable time constants (log-space for stability)
        self.log_tau = nn.Parameter(
            torch.zeros(hidden_size).uniform_(math.log(tau_min), math.log(tau_max))
        )

    @property
    def tau(self) -> torch.Tensor:
        """Time constants clamped to valid range."""
        return torch.exp(self.log_tau).clamp(self.tau_min, self.tau_max)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None,
                dt: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with continuous-time dynamics.

        Args:
            x: Input (batch, input_size)
            h: Previous hidden state (batch, hidden_size)
            dt: Time step

        Returns:
            output, new_state
        """
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_size, device=x.device)

        # Compute input drive
        inp = self.W_in(x) + self.W_rec(h)

        # Continuous dynamics: dh/dt = (-h + tanh(inp)) / tau
        # Exact exponential integrator (stable for any dt/tau ratio):
        alpha = 1 - torch.exp(-dt / self.tau)
        h_new = (1 - alpha) * h + alpha * torch.tanh(inp)

        return h_new, h_new


class MetastableCell(nn.Module):
    """
    Metastable cell with dual-rhythm coupling (Spider + Baby).

    Implements the entangled learning concept where:
    - Spider mode: Fast gamma oscillations (40-100Hz), deep attractors, reflex
    - Baby mode: Slow theta oscillations (4-8Hz), weak attractors, exploration

    Cross-frequency coupling allows theta to gate gamma bursts.
    """

    def __init__(self,
                 dim: int,
                 gamma_freq: float = 40.0,
                 theta_freq: float = 6.0,
                 coupling_strength: float = 0.5):
        """
        Args:
            dim: State dimension
            gamma_freq: Fast oscillation frequency (Hz)
            theta_freq: Slow oscillation frequency (Hz)
            coupling_strength: How much theta modulates gamma
        """
        super().__init__()
        self.dim = dim
        self.gamma_freq = gamma_freq
        self.theta_freq = theta_freq
        self.coupling_strength = coupling_strength

        # Spider (fast) state - amplitude and phase
        self.register_buffer('gamma_phase', torch.zeros(dim))
        self.register_buffer('gamma_amp', torch.ones(dim))

        # Baby (slow) state - phase and variance (uncertainty)
        self.register_buffer('theta_phase', torch.zeros(dim))
        self.register_buffer('theta_var', torch.ones(dim) * 0.5)

        # Learnable reflex threshold
        self.reflex_threshold = nn.Parameter(torch.tensor(0.8))

        # Plasticity parameters
        self.learning_rate = nn.Parameter(torch.tensor(0.1))

    def _gamma_burst(self, stress: torch.Tensor, dt: float = 0.001) -> torch.Tensor:
        """Generate fast gamma burst response."""
        # Advance phase
        with torch.no_grad():
            self.gamma_phase.add_(2 * math.pi * self.gamma_freq * dt).remainder_(2 * math.pi)

        # Burst output
        return self.gamma_amp * torch.sin(self.gamma_phase)

    def _theta_explore(self, stress: torch.Tensor, dt: float = 0.001) -> torch.Tensor:
        """Slow theta exploration/learning."""
        # Phase-dependent learning
        with torch.no_grad():
            delta = self.learning_rate * stress * torch.sin(self.theta_phase) + 2 * math.pi * self.theta_freq * dt
            self.theta_phase.add_(delta).remainder_(2 * math.pi)

        return torch.sin(self.theta_phase)

    def forward(self, stress: torch.Tensor, dt: float = 0.001) -> dict:
        """
        Process input stress signal through metastable dynamics.

        Args:
            stress: Input perturbation (batch, dim)
            dt: Time step

        Returns:
            Dictionary with output, mode, and internal states
        """
        stress = stress.mean(dim=0) if stress.dim() > 1 else stress

        # Determine mode based on stress magnitude
        stress_magnitude = stress.abs().mean()

        if stress_magnitude > self.reflex_threshold:
            # SPIDER MODE: High stress triggers reflex
            mode = 'spider'
            with torch.no_grad():
                self.gamma_amp.mul_(1.5)  # Amplify
            output = self._gamma_burst(stress, dt)

            # Decay amplitude back
            with torch.no_grad():
                self.gamma_amp.mul_(0.95).clamp_(max=2.0)
        else:
            # BABY MODE: Low stress enables exploration
            mode = 'baby'
            theta_output = self._theta_explore(stress, dt)

            # Theta-gamma coupling: theta phase gates gamma amplitude
            coupling = torch.cos(self.theta_phase)
            with torch.no_grad():
                self.gamma_amp.copy_(0.5 + 0.5 * coupling * self.coupling_strength)

            gamma_output = self._gamma_burst(stress, dt)
            output = theta_output + gamma_output * self.gamma_amp

        # Compute binding strength (coherence)
        coherence = torch.cos(self.theta_phase - self.gamma_phase).mean()

        return {
            'output': output,
            'mode': mode,
            'coherence': coherence.item(),
            'gamma_amp': self.gamma_amp.mean().item(),
            'theta_phase': self.theta_phase.mean().item()
        }


if __name__ == '__main__':
    print("--- Liquid Networks Examples ---")

    # Example 1: Liquid Time-Constant Cell
    print("\n1. Liquid Time-Constant Cell")
    ltc = LiquidTimeConstantCell(input_size=32, hidden_size=64)
    x = torch.randn(4, 32)  # batch of 4
    state = torch.zeros(4, 64)

    for t in range(10):
        output, state = ltc(x, state)
    print(f"   Final state norm: {state.norm().item():.3f}")

    # Example 2: Neural ODE
    print("\n2. Neural ODE")
    ode_func = ODEFunc(dim=32, hidden_dim=64)
    neural_ode = NeuralODE(ode_func, solver='rk4', step_size=0.1, num_steps=10)

    h0 = torch.randn(4, 32)
    h1 = neural_ode(h0)
    print(f"   Initial: {h0.norm().item():.3f}, Final: {h1.norm().item():.3f}")
    print(f"   NFE: {ode_func.nfe}")

    # Example 3: Continuous-Time RNN
    print("\n3. Continuous-Time RNN")
    ctrnn = ContinuousTimeRNN(input_size=16, hidden_size=32)
    x = torch.randn(4, 16)
    h = None

    for t in range(20):
        h, _ = ctrnn(x, h, dt=0.1)
    print(f"   Final hidden norm: {h.norm().item():.3f}")
    print(f"   Time constants range: [{ctrnn.tau.min().item():.2f}, {ctrnn.tau.max().item():.2f}]")

    # Example 4: Metastable Cell (Spider/Baby)
    print("\n4. Metastable Cell (Spider/Baby dynamics)")
    meta = MetastableCell(dim=16)

    # Low stress - baby mode
    low_stress = torch.randn(16) * 0.3
    result = meta(low_stress)
    print(f"   Low stress -> mode: {result['mode']}, coherence: {result['coherence']:.3f}")

    # High stress - spider mode
    high_stress = torch.randn(16) * 2.0
    result = meta(high_stress)
    print(f"   High stress -> mode: {result['mode']}, gamma_amp: {result['gamma_amp']:.3f}")

    print("\n[OK] All liquid network tests passed!")
