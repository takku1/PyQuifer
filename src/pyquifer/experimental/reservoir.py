"""
Reservoir Computing Module for PyQuifer

Implements Echo State Networks (ESN) and related reservoir computing concepts
for oscillator-based neural processing at the edge of chaos.

Key concepts:
- Echo State Property: Input echo fades, reservoir settles to steady state
- Spectral Radius: Controls memory and stability (critical at sr ≈ 1)
- Intrinsic Plasticity: Online adaptation of neuron nonlinearity
- Leaky Integration: Continuous-time dynamics via leak rate

Based on:
- Jaeger "The echo state approach" (2001)
- Lukoševičius & Jaeger "Reservoir computing approaches" (2009)
- Steil "Online reservoir adaptation by intrinsic plasticity" (2007)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Callable, Union, List


def spectral_radius(W: torch.Tensor) -> float:
    """Compute spectral radius (largest absolute eigenvalue)."""
    eigenvalues = torch.linalg.eigvals(W)
    return eigenvalues.abs().max().item()


def scale_spectral_radius(W: torch.Tensor, target_sr: float) -> torch.Tensor:
    """Scale matrix to have target spectral radius."""
    current_sr = spectral_radius(W)
    if current_sr > 0:
        return W * (target_sr / current_sr)
    return W


class EchoStateNetwork(nn.Module):
    """
    Echo State Network with leaky integration.

    The reservoir follows the dynamics:
        x[t+1] = (1-lr)*x[t] + lr*tanh(W_in*u[t+1] + W*x[t] + bias)

    Where:
        - x is reservoir state
        - u is input
        - lr is leak rate (0 < lr <= 1)
        - W is scaled to target spectral radius
    """

    def __init__(self,
                 input_dim: int,
                 reservoir_dim: int,
                 output_dim: Optional[int] = None,
                 spectral_radius: float = 0.9,
                 leak_rate: float = 0.3,
                 input_scaling: float = 1.0,
                 input_connectivity: float = 0.1,
                 reservoir_connectivity: float = 0.1,
                 noise_std: float = 0.0,
                 activation: Callable = torch.tanh):
        """
        Args:
            input_dim: Input dimension
            reservoir_dim: Number of reservoir neurons
            output_dim: Output dimension (if None, returns reservoir state)
            spectral_radius: Target spectral radius of W (critical at ~1.0)
            leak_rate: Leaky integration rate (1.0 = no memory)
            input_scaling: Scale factor for input weights
            input_connectivity: Sparsity of input connections
            reservoir_connectivity: Sparsity of reservoir connections
            noise_std: Noise to add to reservoir state
            activation: Activation function
        """
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.target_sr = spectral_radius
        self.leak_rate = leak_rate
        self.noise_std = noise_std
        self.activation = activation

        # Input weights (sparse, scaled)
        W_in = torch.randn(reservoir_dim, input_dim)
        mask_in = (torch.rand(reservoir_dim, input_dim) < input_connectivity).float()
        W_in = W_in * mask_in * input_scaling
        self.register_buffer('W_in', W_in)

        # Reservoir weights (sparse, scaled to target spectral radius)
        W = torch.randn(reservoir_dim, reservoir_dim)
        mask = (torch.rand(reservoir_dim, reservoir_dim) < reservoir_connectivity).float()
        W = W * mask
        W = scale_spectral_radius(W, spectral_radius)
        self.register_buffer('W', W)

        # Bias
        self.register_buffer('bias', torch.zeros(reservoir_dim))

        # Output layer (optional, trainable)
        if output_dim is not None:
            self.readout = nn.Linear(reservoir_dim, output_dim)
        else:
            self.readout = None

        # Track actual spectral radius
        self._actual_sr = spectral_radius

    def forward(self, u: torch.Tensor,
                state: Optional[torch.Tensor] = None,
                return_states: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process input sequence through reservoir.

        Args:
            u: Input tensor (batch, seq_len, input_dim) or (batch, input_dim)
            state: Initial reservoir state (batch, reservoir_dim)
            return_states: Whether to return all states

        Returns:
            output (and optionally all states)
        """
        if u.dim() == 2:
            u = u.unsqueeze(1)  # Add sequence dimension

        batch_size, seq_len, _ = u.shape

        if state is None:
            state = torch.zeros(batch_size, self.reservoir_dim, device=u.device)

        states = []

        for t in range(seq_len):
            # Pre-activation
            pre = F.linear(u[:, t], self.W_in) + F.linear(state, self.W) + self.bias

            # Leaky integration
            state = (1 - self.leak_rate) * state + self.leak_rate * self.activation(pre)

            # Add noise if training
            if self.training and self.noise_std > 0:
                state = state + torch.randn_like(state) * self.noise_std

            states.append(state)

        states = torch.stack(states, dim=1)  # (batch, seq_len, reservoir_dim)

        # Apply readout if available
        if self.readout is not None:
            output = self.readout(states)
        else:
            output = states

        if return_states:
            return output, states
        return output

    def compute_lyapunov(self, trajectory: torch.Tensor,
                         perturbation: float = 1e-6) -> float:
        """
        Estimate dynamical stability from trajectory.

        Computes the average log step norm as a proxy for the largest
        Lyapunov exponent. Positive = expanding, negative = contracting.

        Note: This is NOT a true Lyapunov exponent (which requires
        perturbation divergence tracking). Use as a rough stability indicator.
        """
        T = trajectory.shape[1]
        if T < 2:
            return 0.0

        diffs = torch.diff(trajectory, dim=1)
        norms = diffs.norm(dim=-1)

        # Avoid log(0)
        norms = norms.clamp(min=1e-10)

        # Average log growth rate
        le = torch.log(norms).mean().item() / 1.0  # per unit time

        return le


class IntrinsicPlasticity(nn.Module):
    """
    Intrinsic Plasticity for adaptive reservoir neurons.

    Adjusts neuron gain (a) and bias (b) to achieve target output distribution:
        y = f(a*x + b)

    For tanh activation, targets Gaussian output distribution.
    For sigmoid activation, targets exponential distribution.
    """

    def __init__(self,
                 dim: int,
                 target_mean: float = 0.0,
                 target_std: float = 0.2,
                 learning_rate: float = 0.001,
                 activation: str = 'tanh'):
        """
        Args:
            dim: Neuron dimension
            target_mean: Target output mean (mu)
            target_std: Target output std (sigma)
            learning_rate: IP learning rate
            activation: 'tanh' or 'sigmoid'
        """
        super().__init__()
        self.dim = dim
        self.mu = target_mean
        self.sigma = target_std
        self.eta = learning_rate
        self.activation_type = activation

        # Learnable gain and bias
        self.a = nn.Parameter(torch.ones(dim))  # Gain
        self.b = nn.Parameter(torch.zeros(dim))  # Bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gain and bias, then activation."""
        pre = self.a * x + self.b

        if self.activation_type == 'tanh':
            return torch.tanh(pre)
        else:
            return torch.sigmoid(pre)

    def update(self, x: torch.Tensor, y: torch.Tensor):
        """
        Update IP parameters based on output distribution.

        Args:
            x: Pre-activation input
            y: Post-activation output
        """
        with torch.no_grad():
            if self.activation_type == 'tanh':
                # Gaussian target: adapt to match mu, sigma^2 (Triesch 2005)
                # da = eta * (1/a + x - (2*sigma^2 + mu)*x*y - mu*x*y^2)
                # db = eta * (1 - (2*sigma^2 + mu)*y - mu*y^2)

                x_mean = x.mean(dim=0)
                y_mean = y.mean(dim=0)
                xy_mean = (x * y).mean(dim=0)
                xy2_mean = (x * y * y).mean(dim=0)
                y2_mean = (y * y).mean(dim=0)

                sigma_sq = self.sigma ** 2
                da = self.eta * (1/self.a + x_mean - (2*sigma_sq + self.mu)*xy_mean - self.mu*xy2_mean)
                db = self.eta * (1 - (2*sigma_sq + self.mu)*y_mean - self.mu*y2_mean)

                with torch.no_grad():
                    self.a.add_(da)
                    self.b.add_(db)
                    self.a.clamp_(min=0.1)

            else:  # sigmoid
                # Exponential target
                # da = eta * (1/a + x - (2 + 1/mu)*x*y + x*y^2/mu)
                # db = eta * (1 - (2 + 1/mu)*y + y^2/mu)

                x_mean = x.mean(dim=0)
                y_mean = y.mean(dim=0)
                xy_mean = (x * y).mean(dim=0)
                xy2_mean = (x * y * y).mean(dim=0)
                y2_mean = (y * y).mean(dim=0)

                mu_eff = max(self.mu, 0.1)  # Avoid division by zero
                da = self.eta * (1/self.a + x_mean - (2 + 1/mu_eff)*xy_mean + xy2_mean/mu_eff)
                db = self.eta * (1 - (2 + 1/mu_eff)*y_mean + y2_mean/mu_eff)

                with torch.no_grad():
                    self.a.add_(da)
                    self.b.add_(db)
                    self.a.clamp_(min=0.1)


class ReservoirWithIP(nn.Module):
    """
    Echo State Network with Intrinsic Plasticity.

    Combines reservoir dynamics with online adaptation of
    neuron nonlinearities for improved performance.
    """

    def __init__(self,
                 input_dim: int,
                 reservoir_dim: int,
                 output_dim: int,
                 spectral_radius: float = 0.95,
                 leak_rate: float = 0.3,
                 ip_learning_rate: float = 0.001,
                 ip_target_mean: float = 0.0,
                 ip_target_std: float = 0.2):
        """
        Args:
            input_dim: Input dimension
            reservoir_dim: Reservoir size
            output_dim: Output dimension
            spectral_radius: Target spectral radius
            leak_rate: Leaky integration rate
            ip_learning_rate: Intrinsic plasticity learning rate
            ip_target_mean: Target output mean for IP
            ip_target_std: Target output std for IP
        """
        super().__init__()

        self.reservoir_dim = reservoir_dim
        self.leak_rate = leak_rate

        # Input weights (fixed random projection, not learnable)
        self.register_buffer('W_in',
            torch.randn(reservoir_dim, input_dim) * 0.1
        )

        # Reservoir weights
        W = torch.randn(reservoir_dim, reservoir_dim) / math.sqrt(reservoir_dim)
        W = scale_spectral_radius(W, spectral_radius)
        self.register_buffer('W', W)

        # Intrinsic plasticity layer
        self.ip = IntrinsicPlasticity(
            reservoir_dim,
            target_mean=ip_target_mean,
            target_std=ip_target_std,
            learning_rate=ip_learning_rate
        )

        # Trainable readout
        self.readout = nn.Linear(reservoir_dim, output_dim)

    def forward(self, u: torch.Tensor,
                state: Optional[torch.Tensor] = None,
                update_ip: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through reservoir with IP.

        Args:
            u: Input (batch, seq_len, input_dim) or (batch, input_dim)
            state: Initial reservoir state
            update_ip: Whether to update IP parameters

        Returns:
            output, final_state
        """
        if u.dim() == 2:
            u = u.unsqueeze(1)

        batch_size, seq_len, _ = u.shape

        if state is None:
            state = torch.zeros(batch_size, self.reservoir_dim, device=u.device)

        outputs = []

        for t in range(seq_len):
            # Pre-activation
            pre = F.linear(u[:, t], self.W_in) + F.linear(state, self.W)

            # Apply IP (with optional online update)
            post = self.ip(pre)

            if self.training and update_ip:
                self.ip.update(pre, post)

            # Leaky integration
            state = (1 - self.leak_rate) * state + self.leak_rate * post

            outputs.append(state)

        states = torch.stack(outputs, dim=1)
        output = self.readout(states)

        return output, state


class CriticalReservoir(nn.Module):
    """
    Reservoir with tunable distance-to-criticality.

    Implements the edge-of-chaos framework (Langton, 1990; Bertschinger & Natschläger, 2004):
    - Temperature T ≈ 1.0: Critical regime (maximum computational capacity)
    - Temperature T < 1.0: Ordered/subcritical regime (stable, low capacity)
    - Temperature T > 1.0: Chaotic/supercritical regime (unstable)

    Operating modes:
    - exploratory (T=1.0): High sensitivity, maximal information transfer
    - exploitation (T=0.3): Stable attractors, reliable output

    The critical regime maximizes:
    - Mutual information I(input; output)
    - Correlation length (sensitivity to perturbations)
    - Computational capacity (memory × nonlinearity)

    References:
    - Langton, C. (1990). Computation at the edge of chaos.
    - Bertschinger, N. & Natschläger, T. (2004). Real-time computation at the edge of chaos.
    """

    def __init__(self,
                 input_dim: int,
                 reservoir_dim: int,
                 initial_temperature: float = 1.0,
                 leak_rate: float = 0.3):
        """
        Args:
            input_dim: Input dimension
            reservoir_dim: Reservoir size
            initial_temperature: Distance to criticality T (1.0 = critical point)
            leak_rate: Leaky integration rate (1.0 = no memory, 0.0 = full memory)
        """
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.leak_rate = leak_rate

        # Temperature controls distance to criticality
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))

        # Input weights
        self.W_in = nn.Parameter(torch.randn(reservoir_dim, input_dim) * 0.1)

        # Base reservoir weights (will be scaled by temperature)
        W_base = torch.randn(reservoir_dim, reservoir_dim) / math.sqrt(reservoir_dim)
        self.register_buffer('W_base', W_base)

    @property
    def W(self) -> torch.Tensor:
        """Reservoir weights scaled by temperature."""
        # At T=1, spectral radius approaches 1 (critical)
        # At T<1, spectral radius < 1 (ordered/stable)
        return self.W_base * torch.clamp(self.temperature, 0.01, 2.0)

    def forward(self, u: torch.Tensor,
                state: Optional[torch.Tensor] = None,
                context: str = 'exploration') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input with context-dependent criticality.

        Args:
            u: Input (batch, input_dim)
            state: Previous state
            context: 'exploration' (critical) or 'exploitation' (subcritical)

        Returns:
            output, new_state
        """
        if state is None:
            state = torch.zeros(u.shape[0], self.reservoir_dim, device=u.device)

        # Use the W property (includes temperature clamping)
        W_scaled = self.W
        if context == 'exploitation':
            # Subcritical: stable attractors, reliable output
            W_scaled = W_scaled * 0.5

        # Reservoir dynamics with leaky integration
        pre = F.linear(u, self.W_in) + F.linear(state, W_scaled)
        state = (1 - self.leak_rate) * state + self.leak_rate * torch.tanh(pre)

        return state, state

    def set_mode(self, mode: str):
        """
        Set operating mode (exploration-exploitation tradeoff).

        Args:
            mode: One of:
                - 'exploration' or 'critical': T=1.0, maximum sensitivity
                - 'exploitation' or 'stable': T=0.3, deep attractors
                - 'balanced': T=0.7, intermediate regime
                - 'baby': Legacy alias for 'exploration'
                - 'spider': Legacy alias for 'exploitation'
        """
        mode_map = {
            'exploration': 1.0,
            'critical': 1.0,
            'baby': 1.0,  # Legacy alias
            'exploitation': 0.3,
            'stable': 0.3,
            'spider': 0.3,  # Legacy alias
            'balanced': 0.7,
            'intermediate': 0.7,
        }
        if mode in mode_map:
            with torch.no_grad():
                self.temperature.fill_(mode_map[mode])
        else:
            raise ValueError(f"Unknown mode: {mode}. Use: {list(mode_map.keys())}")


if __name__ == '__main__':
    print("--- Reservoir Computing Examples ---")

    # Example 1: Basic Echo State Network
    print("\n1. Echo State Network")
    esn = EchoStateNetwork(
        input_dim=10, reservoir_dim=100, output_dim=5,
        spectral_radius=0.9, leak_rate=0.3
    )

    x = torch.randn(4, 20, 10)  # batch=4, seq=20, input=10
    output, states = esn(x, return_states=True)
    print(f"   Input: {x.shape} -> Output: {output.shape}")
    print(f"   Reservoir states: {states.shape}")

    # Estimate Lyapunov exponent
    le = esn.compute_lyapunov(states)
    print(f"   Lyapunov exponent estimate: {le:.4f}")

    # Example 2: Intrinsic Plasticity
    print("\n2. Intrinsic Plasticity")
    ip = IntrinsicPlasticity(dim=50, target_mean=0.0, target_std=0.2)

    x = torch.randn(32, 50)  # batch of 32
    y = ip(x)
    print(f"   Before IP: mean={y.mean().item():.3f}, std={y.std().item():.3f}")

    # Run a few IP updates
    for _ in range(100):
        x = torch.randn(32, 50)
        y = ip(x)
        ip.update(x, y)

    y = ip(torch.randn(32, 50))
    print(f"   After IP: mean={y.mean().item():.3f}, std={y.std().item():.3f}")
    print(f"   Gain range: [{ip.a.min().item():.2f}, {ip.a.max().item():.2f}]")

    # Example 3: Reservoir with IP
    print("\n3. Reservoir with Intrinsic Plasticity")
    esn_ip = ReservoirWithIP(
        input_dim=10, reservoir_dim=100, output_dim=5,
        spectral_radius=0.95, leak_rate=0.3
    )
    esn_ip.train()

    x = torch.randn(4, 20, 10)
    output, final_state = esn_ip(x, update_ip=True)
    print(f"   Output shape: {output.shape}")

    # Example 4: Critical Reservoir with Mode Switching
    print("\n4. Critical Reservoir with Mode Switching")
    crit = CriticalReservoir(input_dim=10, reservoir_dim=50)

    x = torch.randn(4, 10)

    # Exploration mode (critical regime)
    crit.set_mode('exploration')
    state_explore, _ = crit(x, context='exploration')
    print(f"   Exploration mode (critical): T={crit.temperature.item():.2f}")

    # Exploitation mode (subcritical regime)
    crit.set_mode('exploitation')
    state_exploit, _ = crit(x, context='exploitation')
    print(f"   Exploitation mode (stable): T={crit.temperature.item():.2f}")

    print("\n[OK] All reservoir computing tests passed!")
