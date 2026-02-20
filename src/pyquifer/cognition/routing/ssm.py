"""
Selective State Space Model — input-dependent dynamics for O(n) sequences.

Implements the Mamba-style S6 layer where the state space matrices A, B, C
are functions of the input, enabling content-aware selection. Includes an
oscillator-gated variant where selection is modulated by oscillator phase.

Key classes:
- SelectiveStateSpace: Core S6 layer with input-dependent parameters
- SelectiveScan: Hardware-aware parallel scan (associative scan)
- SSMBlock: SSM + gated MLP, stackable like Transformer blocks
- MambaLayer: Full Mamba block with conv1d + selective SSM
- OscillatorySSM: SSM with oscillator-phase-modulated selectivity

References:
- Gu & Dao (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
- Gu, Goel & Re (2022). Efficiently Modeling Long Sequences with Structured State Spaces.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveScan(nn.Module):
    """Parallel associative scan for state space models.

    Implements the selective scan operator that processes sequences
    in O(n) time with input-dependent transition matrices.

    The recurrence is: h_t = A_t * h_{t-1} + B_t * x_t

    Args:
        state_dim: Hidden state dimension (N).
    """

    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        x: torch.Tensor,
        D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run selective scan.

        Args:
            A: (B, L, D, N) discretized state transition
            B: (B, L, D, N) input projection
            C: (B, L, N) output projection
            x: (B, L, D) input sequence
            D: Optional (D,) skip connection

        Returns:
            (B, L, D) output sequence
        """
        batch, seq_len, d_inner = x.shape
        N = self.state_dim

        # Sequential scan (parallel scan is CUDA-only optimization)
        h = torch.zeros(batch, d_inner, N, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            # h_t = A_t * h_{t-1} + B_t * x_t
            h = A[:, t] * h + B[:, t] * x[:, t].unsqueeze(-1)
            # y_t = C_t * h_t
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, D)
            outputs.append(y)

        y = torch.stack(outputs, dim=1)  # (B, L, D)

        if D is not None:
            y = y + x * D.unsqueeze(0).unsqueeze(0)

        return y


class SelectiveStateSpace(nn.Module):
    """Core S6 selective state space layer.

    Input-dependent A, B, C matrices enable content-aware selection:
    the model learns WHAT to remember and WHAT to forget based on
    the current input.

    Args:
        d_model: Input/output dimension.
        d_state: State dimension (N).
        d_inner: Inner expansion dimension.
        dt_rank: Rank of dt projection.
        dt_min: Minimum discretization step.
        dt_max: Maximum discretization step.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_inner: int = 0,
        dt_rank: int = 0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner if d_inner > 0 else d_model * 2
        self.dt_rank = dt_rank if dt_rank > 0 else max(1, d_model // 16)

        # A parameter: initialized as negative real (stable dynamics)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A.repeat(self.d_inner, 1)))

        # D: skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Input-dependent projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

        # Initialize dt_proj to produce values in [dt_min, dt_max]
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) +
                math.log(dt_min)
            )
            inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_dt)

        self.scan = SelectiveScan(d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply selective state space transformation.

        Args:
            x: (B, L, d_inner) input features

        Returns:
            (B, L, d_inner) transformed features
        """
        B, L, D = x.shape
        N = self.d_state

        # Input-dependent parameters
        x_proj = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        dt, B_proj, C = torch.split(
            x_proj, [self.dt_rank, N, N], dim=-1
        )

        # Discretize
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        # A: continuous → discrete via ZOH
        A = -torch.exp(self.A_log)  # (d_inner, N)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)

        # B: input-dependent
        dB = dt.unsqueeze(-1) * B_proj.unsqueeze(2)  # (B, L, D, N)

        # Run scan
        y = self.scan(dA, dB, C, x, self.D)

        return y


class SSMBlock(nn.Module):
    """SSM + gated MLP block, stackable like Transformer blocks.

    Architecture: LayerNorm → SSM → Residual → LayerNorm → Gated MLP → Residual

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        mlp_ratio: MLP expansion ratio.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        d_inner = int(d_model * mlp_ratio)

        # SSM branch
        self.norm1 = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner)
        self.ssm = SelectiveStateSpace(d_model, d_state, d_inner=d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)

        # Gated MLP branch
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp_gate = nn.Linear(d_model, d_inner)
        self.mlp_up = nn.Linear(d_model, d_inner)
        self.mlp_down = nn.Linear(d_inner, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SSM block.

        Args:
            x: (B, L, d_model) input

        Returns:
            (B, L, d_model) output
        """
        # SSM branch
        h = self.norm1(x)
        h = self.in_proj(h)
        h = self.ssm(h)
        h = self.out_proj(h)
        x = x + self.dropout(h)

        # Gated MLP branch
        h = self.norm2(x)
        gate = torch.sigmoid(self.mlp_gate(h))
        h = self.mlp_up(h) * gate
        h = self.mlp_down(h)
        x = x + self.dropout(h)

        return x


class MambaLayer(nn.Module):
    """Full Mamba block with conv1d + selective SSM.

    Architecture:
    1. Linear projection to expanded dim
    2. Conv1d for local context
    3. SiLU activation
    4. Selective SSM
    5. Gated output with parallel branch

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        expand: Expansion factor.
        d_conv: Conv1d kernel size.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        d_inner = d_model * expand

        self.norm = nn.LayerNorm(d_model)

        # Input projections (split into SSM path and gate path)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Conv1d for local context
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, d_conv,
            padding=d_conv - 1, groups=d_inner,
        )

        # Selective SSM
        self.ssm = SelectiveStateSpace(d_model, d_state, d_inner=d_inner)

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Mamba layer.

        Args:
            x: (B, L, d_model) input

        Returns:
            (B, L, d_model) output
        """
        residual = x
        x = self.norm(x)

        # Split into SSM path and gate
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Conv1d (causal: trim right padding)
        x_conv = x_ssm.transpose(1, 2)  # (B, D, L)
        x_conv = self.conv1d(x_conv)[:, :, :x.shape[1]]  # Trim to L
        x_conv = x_conv.transpose(1, 2)  # (B, L, D)
        x_conv = F.silu(x_conv)

        # Selective SSM
        y = self.ssm(x_conv)

        # Gated output
        y = y * F.silu(z)
        y = self.out_proj(y)

        return residual + y


class OscillatorySSM(nn.Module):
    """SSM with oscillator-phase-modulated selectivity.

    The selection mechanism (what to remember/forget) is modulated
    by oscillator phase. High coherence → selective (remember important),
    low coherence → permissive (let everything through).

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        num_oscillators: Number of modulating oscillators.
        expand: Expansion factor.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        num_oscillators: int = 8,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_oscillators = num_oscillators

        self.mamba = MambaLayer(d_model, d_state, expand)

        # Phase-to-selectivity mapping
        self.phase_gate = nn.Sequential(
            nn.Linear(num_oscillators, d_model),
            nn.Sigmoid(),
        )

        # Oscillator natural frequencies
        self.register_buffer(
            'omega',
            torch.randn(num_oscillators) * 0.5 + 2.0,
        )
        self.register_buffer('_phase', torch.zeros(num_oscillators))

    def step_oscillators(self, dt: float = 0.01) -> torch.Tensor:
        """Advance oscillator phases."""
        with torch.no_grad():
            self._phase.add_(2 * math.pi * self.omega * dt)
            self._phase.remainder_(2 * math.pi)
        return self._phase.clone()

    def forward(
        self,
        x: torch.Tensor,
        phases: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply oscillator-modulated SSM.

        Args:
            x: (B, L, d_model) input sequence
            phases: Optional (num_oscillators,) external oscillator phases.
                    If None, uses internal oscillators.

        Returns:
            (B, L, d_model) output
        """
        if phases is None:
            phases = self.step_oscillators()

        # Compute selectivity gate from oscillator phases
        phase_features = torch.cos(phases)  # (num_osc,)
        gate = self.phase_gate(phase_features)  # (d_model,)

        # Modulate input selectivity
        x_modulated = x * gate.unsqueeze(0).unsqueeze(0)

        # Apply Mamba
        return self.mamba(x_modulated)
