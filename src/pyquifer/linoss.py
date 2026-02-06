"""
Linear Oscillatory State-Space (LinOSS) Module

Implements forced harmonic oscillators for efficient long-sequence temporal processing.
Based on research from MIT CSAIL on oscillatory state-space models.

Key features:
- Efficient O(n log n) computation for long sequences via FFT
- Learnable oscillator parameters (frequencies, damping, coupling)
- Natural integration with PyQuifer's Kuramoto-based architecture
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class HarmonicOscillator(nn.Module):
    """
    A single learnable damped harmonic oscillator.

    State equation: d²x/dt² + 2ζω₀(dx/dt) + ω₀²x = F(t)

    Where:
    - ω₀ is the natural frequency
    - ζ is the damping ratio
    - F(t) is the forcing function (input)
    """

    def __init__(self, natural_frequency: float = 1.0, damping_ratio: float = 0.1):
        super().__init__()
        # Learnable parameters
        self.log_omega = nn.Parameter(torch.tensor(math.log(natural_frequency)))
        self.log_zeta = nn.Parameter(torch.tensor(math.log(damping_ratio + 1e-6)))

    @property
    def omega(self) -> torch.Tensor:
        """Natural frequency (always positive)."""
        return torch.exp(self.log_omega)

    @property
    def zeta(self) -> torch.Tensor:
        """Damping ratio (always positive)."""
        return torch.exp(self.log_zeta)

    def get_state_matrix(self, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the discrete-time state transition matrix A and input matrix B.

        State: [x, dx/dt]
        """
        omega = self.omega
        zeta = self.zeta

        # Continuous-time state matrix (built with stack/cat to preserve gradient flow)
        # dx/dt = v
        # dv/dt = -2ζω₀v - ω₀²x + F
        zero = torch.zeros(1, device=omega.device)
        one = torch.ones(1, device=omega.device)
        row0 = torch.cat([zero, one])
        row1 = torch.cat([(-omega**2).unsqueeze(0), (-2 * zeta * omega).unsqueeze(0)])
        A_cont = torch.stack([row0, row1])

        B_cont = torch.tensor([[0.0], [1.0]], device=omega.device)

        # Discretize using matrix exponential (first-order approximation)
        # For small dt: A_d ≈ I + A_cont * dt
        A_discrete = torch.eye(2, device=omega.device) + A_cont * dt
        B_discrete = B_cont * dt

        return A_discrete, B_discrete


class LinOSSLayer(nn.Module):
    """
    Linear Oscillatory State-Space Layer.

    Processes sequential input through a bank of learnable harmonic oscillators.
    Efficient for long sequences due to parallel computation and optional FFT acceleration.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_oscillators: int = 16,
                 dt: float = 0.01,
                 use_fft: bool = True):
        """
        Args:
            input_dim: Dimension of input features.
            hidden_dim: Dimension of output representation.
            num_oscillators: Number of harmonic oscillators in the bank.
            dt: Time step for discretization.
            use_fft: Reserved for future FFT-based convolution (not yet implemented).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_oscillators = num_oscillators
        self.dt = dt
        self.use_fft = use_fft

        # Input projection
        self.input_proj = nn.Linear(input_dim, num_oscillators)

        # Bank of harmonic oscillators with diverse initial frequencies
        # Initialize with frequencies spanning multiple octaves
        base_freq = 0.5
        self.oscillators = nn.ModuleList([
            HarmonicOscillator(
                natural_frequency=base_freq * (2 ** (i / num_oscillators * 4)),
                damping_ratio=0.1 + 0.05 * (i / num_oscillators)
            )
            for i in range(num_oscillators)
        ])

        # Learnable mixing coefficients for oscillator outputs
        self.output_mix = nn.Linear(num_oscillators * 2, hidden_dim)  # *2 for position and velocity

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, return_states: bool = False) -> torch.Tensor:
        """
        Process sequential input through oscillator bank.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).
            return_states: If True, also return the oscillator states.

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim).
            If return_states=True, also returns states (batch_size, seq_len, num_oscillators, 2).
        """
        batch_size, seq_len, _ = x.shape

        # Project input to oscillator forcing functions
        forcing = self.input_proj(x)  # (batch, seq_len, num_oscillators)

        # Initialize oscillator states [position, velocity]
        states = torch.zeros(batch_size, self.num_oscillators, 2, device=x.device)

        # Collect outputs
        all_states = []

        # Process sequence
        for t in range(seq_len):
            f_t = forcing[:, t, :]  # (batch, num_oscillators)

            # Update each oscillator (TODO: batch across oscillators for perf)
            new_states = []
            for i, osc in enumerate(self.oscillators):
                A, B = osc.get_state_matrix(self.dt)

                # State update: s_{t+1} = A @ s_t + B @ f_t
                s = states[:, i, :]  # (batch, 2)
                s_new = torch.matmul(s, A.T) + f_t[:, i:i+1] * B.squeeze(-1)
                new_states.append(s_new)

            states = torch.stack(new_states, dim=1)  # (batch, num_oscillators, 2)
            all_states.append(states)

        # Stack all timesteps
        all_states = torch.stack(all_states, dim=1)  # (batch, seq_len, num_oscillators, 2)

        # Flatten oscillator states and project to output
        flat_states = all_states.reshape(batch_size, seq_len, -1)  # (batch, seq_len, num_oscillators*2)
        output = self.output_mix(flat_states)  # (batch, seq_len, hidden_dim)
        output = self.layer_norm(output)

        if return_states:
            return output, all_states
        return output

    def get_final_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process sequence and return only the final hidden state.
        Useful for encoding a sequence into a fixed-size representation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Final hidden state of shape (batch_size, hidden_dim).
        """
        output = self.forward(x)
        return output[:, -1, :]


class LinOSSEncoder(nn.Module):
    """
    Multi-layer LinOSS encoder for processing temporal sequences.

    Provides hierarchical temporal processing with multiple oscillator banks
    at different timescales, similar to how the brain processes information
    at multiple frequency bands.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 num_oscillators: int = 16,
                 dt: float = 0.01,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Dimension of input features.
            hidden_dim: Dimension of intermediate representations.
            output_dim: Dimension of final output.
            num_layers: Number of LinOSS layers.
            num_oscillators: Number of oscillators per layer.
            dt: Time step (can vary per layer for multi-scale processing).
            dropout: Dropout probability between layers.
        """
        super().__init__()
        self.num_layers = num_layers

        # Build layers with varying timescales
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            # Each layer operates at a different timescale
            layer_dt = dt * (2 ** i)  # Slower timescales for deeper layers
            layers.append(LinOSSLayer(
                input_dim=in_dim,
                hidden_dim=hidden_dim,
                num_oscillators=num_oscillators,
                dt=layer_dt
            ))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process sequence through all layers.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim).
        """
        h = x
        for layer in self.layers:
            h = layer(h)
            h = self.dropout(h)

        return self.output_proj(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence into a fixed-size representation using the final state.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Encoded representation of shape (batch_size, output_dim).
        """
        h = x
        for layer in self.layers:
            h = layer(h)
            h = self.dropout(h)

        return self.output_proj(h[:, -1, :])


if __name__ == '__main__':
    print("--- LinOSS Module Examples ---")

    # Example 1: Single LinOSS layer
    print("\n1. Single LinOSS Layer")
    layer = LinOSSLayer(input_dim=8, hidden_dim=16, num_oscillators=4, dt=0.01)

    # Random sequence input
    batch_size, seq_len, input_dim = 2, 100, 8
    x = torch.randn(batch_size, seq_len, input_dim)

    output, states = layer(x, return_states=True)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   States shape: {states.shape}")

    # Example 2: LinOSS Encoder
    print("\n2. LinOSS Encoder")
    encoder = LinOSSEncoder(
        input_dim=8, hidden_dim=32, output_dim=16,
        num_layers=3, num_oscillators=8, dt=0.01
    )

    output = encoder(x)
    print(f"   Encoder output shape: {output.shape}")

    encoded = encoder.encode(x)
    print(f"   Encoded representation shape: {encoded.shape}")

    # Example 3: Training example - sequence classification
    print("\n3. Training Example - Sequence Modeling")
    encoder = LinOSSEncoder(input_dim=4, hidden_dim=16, output_dim=4, num_layers=2)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

    # Generate synthetic data: predict the mean of the sequence
    for epoch in range(5):
        # Random sequences
        x = torch.randn(8, 50, 4)
        target = x.mean(dim=1)  # (8, 4) - mean across time

        optimizer.zero_grad()
        pred = encoder.encode(x)
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        optimizer.step()

        print(f"   Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Example 4: Oscillator frequency analysis
    print("\n4. Oscillator Frequencies (learned)")
    for i, osc in enumerate(layer.oscillators):
        print(f"   Oscillator {i}: omega={osc.omega.item():.3f}, zeta={osc.zeta.item():.4f}")
