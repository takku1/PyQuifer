"""
Hypernetwork Module for PyQuifer

Implements hypernetworks that generate weights for other networks dynamically.
This enables context-dependent parameter adaptation for oscillator systems.

Key concepts:
- Weight generation: One network generates weights for another
- Context conditioning: Parameters adapt to input/state
- Oscillator modulation: Dynamically adjust Kuramoto coupling

Based on Ha et al. "HyperNetworks" (2016) and patterns from torchhyper/hyper-nn.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum
from typing import Optional, Dict, List, Literal, Tuple, Union


class InputEncoding(Enum):
    """Input encoding mode for hypernetwork context (from hyperlight patterns)."""
    IDENTITY = 'identity'       # Pass through unchanged
    COS_SIN = 'cos_sin'         # Concatenate [cos(x), sin(x)] — doubles dim, adds frequency info
    NORMALIZED = 'normalized'   # L2-normalize input


def encode_input(x: torch.Tensor, mode: InputEncoding) -> torch.Tensor:
    """Apply input encoding to context vector before hypernetwork processing."""
    # Dispatch table for input encoding modes
    encoders = {
        InputEncoding.IDENTITY: lambda x: x,
        InputEncoding.COS_SIN: lambda x: torch.cat([torch.cos(x), torch.sin(x)], dim=-1),
        InputEncoding.NORMALIZED: lambda x: F.normalize(x, dim=-1),
    }
    return encoders.get(mode, lambda x: x)(x)


class HyperNetwork(nn.Module):
    """
    Basic hypernetwork that generates weights for a target network.

    Given a conditioning input (context), generates all parameters
    for the target network dynamically.
    """

    def __init__(self,
                 context_dim: int,
                 target_shapes: Dict[str, Tuple[int, ...]],
                 hidden_dims: List[int] = [256, 256]):
        """
        Args:
            context_dim: Dimension of conditioning input
            target_shapes: Dict of parameter names to shapes
            hidden_dims: Hidden layer dimensions for generator
        """
        super().__init__()

        self.context_dim = context_dim
        self.target_shapes = target_shapes

        # Calculate total parameter count
        self.param_counts = {
            name: math.prod(shape)
            for name, shape in target_shapes.items()
        }
        self.total_params = sum(self.param_counts.values())

        # Build generator network
        layers = []
        in_dim = context_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Separate heads for each target parameter
        # Replace dots with underscores for ModuleDict compatibility
        self._name_mapping = {name.replace('.', '_'): name for name in self.param_counts}
        self.param_heads = nn.ModuleDict({
            name.replace('.', '_'): nn.Linear(hidden_dims[-1], count)
            for name, count in self.param_counts.items()
        })

        self._init_weights()

    def _init_weights(self):
        """Initialize heads with parameter-type-aware init (from hyperlight patterns).

        Weight heads get Xavier-scaled init, bias heads get near-zero init.
        This prevents magnitude mismatch between weight and bias parameters.
        """
        for safe_name, head in self.param_heads.items():
            original_name = self._name_mapping.get(safe_name, safe_name)
            if 'bias' in original_name:
                # Bias heads: near-zero output
                nn.init.normal_(head.weight, std=0.001)
                nn.init.zeros_(head.bias)
            else:
                # Weight heads: Xavier-scaled for target layer fan-in/fan-out
                target_shape = self.target_shapes[original_name]
                if len(target_shape) >= 2:
                    fan_in, fan_out = target_shape[-1], target_shape[-2]
                    std = math.sqrt(2.0 / (fan_in + fan_out))
                else:
                    std = 0.01
                nn.init.normal_(head.weight, std=std)
                nn.init.zeros_(head.bias)

    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate target network parameters from context.

        Args:
            context: Conditioning input (batch, context_dim) or (context_dim,)

        Returns:
            Dict of parameter tensors with target shapes
        """
        if context.dim() == 1:
            context = context.unsqueeze(0)

        # Get features
        features = self.backbone(context)

        # Generate each parameter
        params = {}
        for name, shape in self.target_shapes.items():
            safe_name = name.replace('.', '_')
            flat_param = self.param_heads[safe_name](features)
            # Reshape to target shape (add batch dim if present)
            if context.shape[0] == 1:
                params[name] = flat_param.squeeze(0).view(shape)
            else:
                params[name] = flat_param.view(-1, *shape)

        return params


class OscillatorHyperNetwork(nn.Module):
    """
    Hypernetwork specialized for generating oscillator parameters.

    Generates Kuramoto coupling matrix, natural frequencies, and
    phase offsets based on input context.
    """

    def __init__(self,
                 context_dim: int,
                 num_oscillators: int,
                 hidden_dim: int = 128,
                 generate_coupling: bool = True,
                 generate_frequencies: bool = True,
                 generate_phases: bool = False,
                 input_encoding: Literal['identity', 'cos_sin', 'normalized'] = 'identity'):
        """
        Args:
            context_dim: Dimension of conditioning input
            num_oscillators: Number of oscillators to generate params for
            hidden_dim: Hidden layer dimension
            generate_coupling: Generate coupling matrix K
            generate_frequencies: Generate natural frequencies ω
            generate_phases: Generate initial phases θ
            input_encoding: Input encoding mode (from hyperlight patterns):
                - 'identity': Pass context unchanged (default)
                - 'cos_sin': Concatenate [cos(ctx), sin(ctx)] — doubles dim, adds
                  frequency features. Best for oscillator-related contexts.
                - 'normalized': L2-normalize context before encoding.
        """
        super().__init__()

        self.num_osc = num_oscillators
        self.generate_coupling = generate_coupling
        self.generate_frequencies = generate_frequencies
        self.generate_phases = generate_phases
        self.input_encoding = InputEncoding(input_encoding)

        # Input encoding may change effective dimension
        effective_dim = context_dim * 2 if input_encoding == 'cos_sin' else context_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(effective_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Parameter generators
        if generate_coupling:
            # Coupling matrix (N x N)
            self.coupling_head = nn.Linear(hidden_dim, num_oscillators * num_oscillators)

        if generate_frequencies:
            # Natural frequencies (N,)
            self.freq_head = nn.Linear(hidden_dim, num_oscillators)

        if generate_phases:
            # Initial phases (N,)
            self.phase_head = nn.Linear(hidden_dim, num_oscillators)

        # Base parameters (when not generated)
        if not generate_coupling:
            self.base_coupling = nn.Parameter(
                torch.randn(num_oscillators, num_oscillators) / num_oscillators
            )
        if not generate_frequencies:
            self.base_freq = nn.Parameter(torch.randn(num_oscillators))

    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate oscillator parameters from context.

        Args:
            context: (batch, context_dim) or (context_dim,)

        Returns:
            Dict with 'coupling', 'frequencies', 'phases'
        """
        if context.dim() == 1:
            context = context.unsqueeze(0)

        # Apply input encoding before encoder
        context = encode_input(context, self.input_encoding)

        batch_size = context.shape[0]
        features = self.encoder(context)

        result = {}

        if self.generate_coupling:
            # Generate and reshape coupling matrix
            K_flat = self.coupling_head(features)
            K = K_flat.view(batch_size, self.num_osc, self.num_osc)
            # Make symmetric for undirected coupling
            K = (K + K.transpose(-1, -2)) / 2
            result['coupling'] = K.squeeze(0) if batch_size == 1 else K
        else:
            K = self.base_coupling.unsqueeze(0).expand(batch_size, -1, -1)
            result['coupling'] = self.base_coupling if batch_size == 1 else K

        if self.generate_frequencies:
            omega = self.freq_head(features)
            result['frequencies'] = omega.squeeze(0) if batch_size == 1 else omega
        else:
            result['frequencies'] = self.base_freq

        if self.generate_phases:
            # Generate phases in [-π, π]
            theta = torch.tanh(self.phase_head(features)) * math.pi
            result['phases'] = theta.squeeze(0) if batch_size == 1 else theta

        return result


class DynamicLinear(nn.Module):
    """
    Linear layer with dynamically generated weights.

    The weights are generated by a hypernetwork rather than
    being stored as parameters.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 context_dim: int,
                 bias: bool = True):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            context_dim: Dimension of conditioning context
            bias: Whether to include bias
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Weight generator
        weight_size = in_features * out_features
        self.weight_gen = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, weight_size)
        )

        if bias:
            self.bias_gen = nn.Sequential(
                nn.Linear(context_dim, 64),
                nn.ReLU(),
                nn.Linear(64, out_features)
            )

        # Scale initialization
        self._init_scale = 1.0 / math.sqrt(in_features)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with context-dependent weights.

        Args:
            x: Input tensor (batch, in_features)
            context: Conditioning context (batch, context_dim) or (context_dim,)

        Returns:
            Output tensor (batch, out_features)
        """
        if context.dim() == 1:
            context = context.unsqueeze(0)

        # Generate weights
        weight_flat = self.weight_gen(context) * self._init_scale
        weight = weight_flat.view(-1, self.out_features, self.in_features)

        # Handle batched matmul
        if weight.shape[0] == 1:
            # Same weights for all inputs
            output = F.linear(x, weight.squeeze(0))
        else:
            # Different weights per sample (batch matrix multiply)
            output = torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1)

        if self.use_bias:
            bias = self.bias_gen(context)
            if bias.shape[0] == 1:
                output = output + bias
            else:
                output = output + bias

        return output


class ContextualReservoir(nn.Module):
    """
    Reservoir computing with hypernetwork-generated weights.

    The reservoir weights adapt based on context, enabling
    different dynamics for different inputs.
    """

    def __init__(self,
                 input_dim: int,
                 reservoir_dim: int,
                 context_dim: int,
                 spectral_radius: float = 0.9,
                 leak_rate: float = 0.3):
        """
        Args:
            input_dim: Input dimension
            reservoir_dim: Reservoir size
            context_dim: Context dimension for adaptation
            spectral_radius: Target spectral radius
            leak_rate: Leaky integration rate
        """
        super().__init__()

        self.reservoir_dim = reservoir_dim
        self.leak_rate = leak_rate
        self.target_sr = spectral_radius

        # Fixed input weights
        self.W_in = nn.Parameter(
            torch.randn(reservoir_dim, input_dim) * 0.1,
            requires_grad=False
        )

        # Hypernetwork for reservoir weights
        self.W_hypernet = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, reservoir_dim * reservoir_dim)
        )

        # Learnable scaling
        self.weight_scale = nn.Parameter(torch.tensor(spectral_radius))

    def forward(self, x: torch.Tensor,
                context: torch.Tensor,
                state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with context-dependent reservoir.

        Args:
            x: Input (batch, input_dim)
            context: Context for weight generation
            state: Previous state (batch, reservoir_dim)

        Returns:
            output, new_state
        """
        batch_size = x.shape[0]

        if state is None:
            state = torch.zeros(batch_size, self.reservoir_dim, device=x.device)

        # Generate reservoir weights from context
        if context.dim() == 1:
            context = context.unsqueeze(0)

        W_flat = self.W_hypernet(context)
        W = W_flat.view(-1, self.reservoir_dim, self.reservoir_dim)
        W = W * self.weight_scale / self.reservoir_dim  # Scale for stability

        # Reservoir dynamics
        if W.shape[0] == 1:
            # Same weights for all
            W = W.squeeze(0)
            pre = F.linear(x, self.W_in) + F.linear(state, W)
        else:
            # Per-sample weights
            pre = F.linear(x, self.W_in) + torch.bmm(W, state.unsqueeze(-1)).squeeze(-1)

        # Leaky integration
        new_state = (1 - self.leak_rate) * state + self.leak_rate * torch.tanh(pre)

        return new_state, new_state


if __name__ == '__main__':
    print("--- Hypernetwork Examples ---")

    # Example 1: Basic HyperNetwork
    print("\n1. Basic HyperNetwork")
    target_shapes = {
        'layer1.weight': (64, 32),
        'layer1.bias': (64,),
        'layer2.weight': (10, 64),
        'layer2.bias': (10,)
    }
    hypernet = HyperNetwork(
        context_dim=16,
        target_shapes=target_shapes,
        hidden_dims=[128, 128]
    )

    context = torch.randn(16)
    params = hypernet(context)
    print(f"   Generated params:")
    for name, tensor in params.items():
        print(f"     {name}: {tensor.shape}")

    # Example 2: Oscillator HyperNetwork
    print("\n2. OscillatorHyperNetwork")
    osc_hypernet = OscillatorHyperNetwork(
        context_dim=32,
        num_oscillators=8,
        hidden_dim=64,
        generate_coupling=True,
        generate_frequencies=True,
        generate_phases=True
    )

    context = torch.randn(32)
    osc_params = osc_hypernet(context)
    print(f"   Coupling: {osc_params['coupling'].shape}")
    print(f"   Frequencies: {osc_params['frequencies'].shape}")
    print(f"   Phases: {osc_params['phases'].shape}")

    # Example 3: DynamicLinear
    print("\n3. DynamicLinear layer")
    dyn_linear = DynamicLinear(
        in_features=32,
        out_features=64,
        context_dim=16
    )

    x = torch.randn(4, 32)
    context = torch.randn(16)
    output = dyn_linear(x, context)
    print(f"   Input: {x.shape} -> Output: {output.shape}")

    # Example 4: ContextualReservoir
    print("\n4. ContextualReservoir")
    ctx_reservoir = ContextualReservoir(
        input_dim=10,
        reservoir_dim=50,
        context_dim=8
    )

    x = torch.randn(4, 10)
    context = torch.randn(8)
    state = None

    for t in range(5):
        output, state = ctx_reservoir(x, context, state)

    print(f"   Final state norm: {state.norm().item():.3f}")

    # Gradient check
    loss = output.sum()
    loss.backward()
    print("\n5. Gradient check: OK")

    print("\n[OK] All hypernetwork tests passed!")
