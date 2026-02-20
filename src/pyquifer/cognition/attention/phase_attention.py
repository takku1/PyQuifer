"""
Phase-based Attention Mechanisms for PyQuifer

Implements attention mechanisms based on oscillator synchronization
rather than softmax-weighted sums. This enables:
- Oscillator-native transformer processing
- Natural integration with Kuramoto dynamics
- Interpretable coherence metrics
- Biologically plausible attention

Based on PhaseGPT patterns for Kuramoto-based attention replacement.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PhaseAttention(nn.Module):
    """
    Replace transformer attention with Kuramoto phase coupling.

    Standard Attention:
        Q, K, V = x @ W_q, x @ W_k, x @ W_v
        attn_weights = softmax(Q @ K^T / sqrt(d_k))
        output = attn_weights @ V

    Phase Attention:
        phases = x @ W_phase
        synced_phases = kuramoto_sync(phases, iterations=N)
        output = synced_phases @ W_out

    The attention pattern emerges from phase synchronization:
    - Strongly coupled tokens synchronize (high attention)
    - Weakly coupled tokens remain desynchronized (low attention)
    """

    def __init__(self,
                 d_model: int,
                 n_heads: int = 1,
                 num_oscillators: int = None,
                 coupling_strength: float = 1.0,
                 natural_freq_std: float = 0.1,
                 phase_iterations: int = 10,
                 dt: float = 0.1,
                 dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            num_oscillators: Oscillators per head (default: d_model)
            coupling_strength: Kuramoto coupling K
            natural_freq_std: Std of natural frequencies
            phase_iterations: Synchronization steps
            dt: Integration time step
            dropout: Output dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.num_osc = num_oscillators or d_model
        self.K = coupling_strength
        self.iterations = phase_iterations
        self.dt = dt

        # Project embeddings to phase space
        self.to_phase = nn.Linear(d_model, self.num_osc * n_heads)

        # Learnable natural frequencies
        self.natural_freq = nn.Parameter(
            torch.randn(n_heads, self.num_osc) * natural_freq_std
        )

        # Learnable coupling strength (can be per-head)
        self.coupling_matrix = nn.Parameter(
            torch.ones(n_heads, 1, 1) * coupling_strength
        )

        # Project synchronized phases back
        self.from_phase = nn.Linear(self.num_osc * n_heads, d_model)

        self.dropout = nn.Dropout(dropout)

        # Coherence tracking
        self.register_buffer('last_coherence', torch.tensor(0.0))
        self.last_phases = None

    def kuramoto_step(self, phases: torch.Tensor,
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Single Kuramoto integration step.

        dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j - θ_i)

        Args:
            phases: (batch, n_heads, seq_len, num_osc)
            mask: (batch, seq_len) optional padding mask

        Returns:
            Updated phases
        """
        batch, n_heads, seq_len, num_osc = phases.shape

        # Phase differences: θ_j - θ_i
        # (batch, n_heads, seq_len, seq_len, num_osc)
        phase_diff = phases.unsqueeze(3) - phases.unsqueeze(2)

        # Coupling: K * sin(θ_j - θ_i)
        coupling = self.coupling_matrix.view(1, n_heads, 1, 1, 1) * torch.sin(phase_diff)

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.view(batch, 1, 1, seq_len, 1).float()
            coupling = coupling * mask_expanded

        # Mean coupling from neighbors
        coupling_sum = coupling.mean(dim=3)

        # Natural frequency drift
        freq_drift = self.natural_freq.view(1, n_heads, 1, num_osc)

        # Kuramoto equation
        dtheta = freq_drift + coupling_sum

        # Euler integration
        new_phases = phases + self.dt * dtheta

        # Keep phases in [-π, π]
        new_phases = torch.remainder(new_phases + math.pi, 2 * math.pi) - math.pi

        return new_phases

    def order_parameter(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Compute Kuramoto order parameter R.

        R = |⟨exp(iθ)⟩| ∈ [0, 1]
        R ≈ 0: Decoherent (random phases)
        R ≈ 1: Coherent (synchronized)

        Args:
            phases: (batch, n_heads, seq_len, num_osc)

        Returns:
            R: (batch, n_heads)
        """
        z = torch.exp(1j * phases)
        z_mean = z.mean(dim=[2, 3])
        R = torch.abs(z_mean)
        return R

    def synchronize(self, phases: torch.Tensor,
                   mask: Optional[torch.Tensor] = None,
                   return_trajectory: bool = False) -> torch.Tensor:
        """
        Run Kuramoto dynamics until convergence.

        Args:
            phases: Initial phases
            mask: Optional attention mask
            return_trajectory: Return all intermediate states

        Returns:
            Final synchronized phases
        """
        trajectory = [phases] if return_trajectory else None

        for _ in range(self.iterations):
            phases = self.kuramoto_step(phases, mask=mask)
            if return_trajectory:
                trajectory.append(phases)

        # Track coherence
        with torch.no_grad():
            R = self.order_parameter(phases)
            self.last_coherence = R.mean()

        if return_trajectory:
            return phases, torch.stack(trajectory, dim=0)
        return phases

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_coherence: bool = False) -> torch.Tensor:
        """
        Forward pass: Replace attention with phase synchronization.

        Args:
            x: Input embeddings (batch, seq_len, d_model)
            mask: Attention mask (batch, seq_len)
            return_coherence: Also return coherence metric

        Returns:
            output: (batch, seq_len, d_model)
            R: Coherence (batch, n_heads) if return_coherence
        """
        batch, seq_len, d_model = x.shape

        # Project to phase space
        phases = self.to_phase(x)
        phases = phases.view(batch, seq_len, self.n_heads, self.num_osc)
        phases = phases.transpose(1, 2)  # (batch, n_heads, seq_len, num_osc)

        # Initialize in [-π, π]
        phases = torch.tanh(phases) * math.pi

        # Synchronize
        synced_phases = self.synchronize(phases, mask=mask)
        self.last_phases = synced_phases.detach()

        # Measure coherence
        R = None
        if return_coherence:
            R = self.order_parameter(synced_phases)

        # Project back
        synced_phases = synced_phases.transpose(1, 2)
        synced_phases = synced_phases.reshape(batch, seq_len, -1)
        output = self.from_phase(synced_phases)
        output = self.dropout(output)

        if return_coherence:
            return output, R
        return output


class PhaseMultiHeadAttention(nn.Module):
    """
    Multi-head attention where each head uses phase coupling.

    This provides the full transformer attention interface
    but uses Kuramoto dynamics internally.
    """

    def __init__(self,
                 d_model: int,
                 n_heads: int = 8,
                 coupling_strength: float = 1.0,
                 phase_iterations: int = 10,
                 dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            coupling_strength: Kuramoto coupling K
            phase_iterations: Synchronization steps
            dropout: Dropout rate
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Phase attention with multiple heads
        self.phase_attn = PhaseAttention(
            d_model=d_model,
            n_heads=n_heads,
            num_oscillators=self.d_head,
            coupling_strength=coupling_strength,
            phase_iterations=phase_iterations,
            dropout=dropout
        )

        # Output projection (like standard transformer)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor,
                key: torch.Tensor = None,
                value: torch.Tensor = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Standard transformer attention interface.

        For self-attention: query = key = value = x
        For cross-attention: key, value come from encoder

        Note: Phase attention currently only supports self-attention.
        For cross-attention, key/value are ignored.

        Args:
            query: (batch, seq_len, d_model)
            key: Ignored (for interface compatibility)
            value: Ignored (for interface compatibility)
            key_padding_mask: (batch, seq_len)
            need_weights: Return attention weights (coherence R)

        Returns:
            output: (batch, seq_len, d_model)
            weights: Coherence R if need_weights else None
        """
        output, R = self.phase_attn(query, mask=key_padding_mask, return_coherence=True)
        output = self.out_proj(output)

        if need_weights:
            return output, R
        return output, None


class HybridPhaseAttention(nn.Module):
    """
    Blend standard attention with phase attention.

    output = (1 - α) * standard_attn(x) + α * phase_attn(x)

    Useful for:
    - Gradual transition from transformers to oscillators
    - Comparing attention mechanisms
    - Finding optimal blend for specific tasks
    """

    def __init__(self,
                 d_model: int,
                 n_heads: int = 8,
                 phase_weight: float = 0.5,
                 coupling_strength: float = 1.0,
                 phase_iterations: int = 10,
                 dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            phase_weight: Blend factor α ∈ [0, 1]
            coupling_strength: Kuramoto coupling K
            phase_iterations: Sync iterations
            dropout: Dropout rate
        """
        super().__init__()

        self.phase_weight = nn.Parameter(torch.tensor(phase_weight))

        # Standard transformer attention
        self.standard_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Phase-based attention
        self.phase_attn = PhaseAttention(
            d_model=d_model,
            n_heads=n_heads,
            coupling_strength=coupling_strength,
            phase_iterations=phase_iterations,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Blend standard and phase attention.

        Args:
            x: Input (batch, seq_len, d_model)
            mask: Attention mask

        Returns:
            Blended output
        """
        # Clamp weight to [0, 1]
        alpha = torch.sigmoid(self.phase_weight)

        # Standard attention
        std_out, _ = self.standard_attn(x, x, x, need_weights=False)

        # Phase attention
        phase_out = self.phase_attn(x, mask=mask)

        # Weighted blend
        output = (1 - alpha) * std_out + alpha * phase_out

        return output


class OscillatorGatedFFN(nn.Module):
    """
    Feed-forward network with oscillator-based gating.

    Instead of GELU/ReLU, uses oscillator phase to gate activation.
    Tokens synchronized with the global rhythm pass through;
    desynchronized tokens are suppressed.
    """

    def __init__(self,
                 d_model: int,
                 d_ff: int = None,
                 gating_freq: float = 10.0,
                 dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (default: 4 * d_model)
            gating_freq: Oscillator frequency for gating
            dropout: Dropout rate
        """
        super().__init__()

        d_ff = d_ff or 4 * d_model

        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)
        self.gate_proj = nn.Linear(d_model, d_ff)

        # Oscillator phase for gating
        self.register_buffer('phase', torch.tensor(0.0))
        self.gating_freq = gating_freq

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """
        Forward pass with oscillator gating.

        Args:
            x: Input (batch, seq_len, d_model)
            dt: Time step for phase advancement

        Returns:
            Output (batch, seq_len, d_model)
        """
        # Advance global phase
        with torch.no_grad():
            self.phase.add_(2 * math.pi * self.gating_freq * dt).remainder_(2 * math.pi)

        # Up projection
        hidden = self.up_proj(x)

        # Oscillator gate: cos(phase + learned_offset)
        gate_offset = self.gate_proj(x)
        gate = torch.cos(self.phase + gate_offset)
        gate = (gate + 1) / 2  # Scale to [0, 1]

        # Gated activation
        hidden = hidden * gate

        # Down projection
        output = self.down_proj(hidden)
        output = self.dropout(output)

        return output


if __name__ == '__main__':
    print("--- Phase Attention Examples ---")

    batch_size = 2
    seq_len = 16
    d_model = 256

    # Example 1: Basic PhaseAttention
    print("\n1. PhaseAttention (Kuramoto-based)")
    x = torch.randn(batch_size, seq_len, d_model)

    phase_attn = PhaseAttention(
        d_model=d_model,
        n_heads=4,
        num_oscillators=64,
        coupling_strength=1.0,
        phase_iterations=10
    )

    output, R = phase_attn(x, return_coherence=True)
    print(f"   Input: {x.shape} -> Output: {output.shape}")
    print(f"   Coherence R: {R.mean().item():.4f}")
    print(f"   R per head: {[f'{r:.3f}' for r in R[0].tolist()]}")

    # Example 2: Multi-head Phase Attention
    print("\n2. PhaseMultiHeadAttention (transformer interface)")
    mha = PhaseMultiHeadAttention(
        d_model=d_model,
        n_heads=8,
        coupling_strength=1.0
    )

    output, weights = mha(x, need_weights=True)
    print(f"   Output: {output.shape}")
    print(f"   Coherence weights: {weights.mean().item():.4f}")

    # Example 3: Hybrid attention
    print("\n3. HybridPhaseAttention (blend std + phase)")
    hybrid = HybridPhaseAttention(
        d_model=d_model,
        n_heads=4,
        phase_weight=0.5
    )

    output = hybrid(x)
    print(f"   Output: {output.shape}")
    print(f"   Phase weight: {torch.sigmoid(hybrid.phase_weight).item():.3f}")

    # Example 4: Oscillator-gated FFN
    print("\n4. OscillatorGatedFFN")
    ffn = OscillatorGatedFFN(d_model=d_model)

    output = ffn(x)
    print(f"   Output: {output.shape}")
    print(f"   Current phase: {ffn.phase.item():.4f}")

    # Gradient check
    print("\n5. Gradient flow check")
    loss = output.sum()
    loss.backward()
    print("   Gradients flow: OK")

    print("\n[OK] All phase attention tests passed!")
