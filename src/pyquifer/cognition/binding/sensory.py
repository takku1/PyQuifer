"""
Cross-Modal Sensory Binding via Inter-Bank Phase Locking.

Implements the oscillatory solution to the binding problem: features from
different modalities (vision, audio, text) synchronize their phases when
they refer to the same entity or event.

Key classes:
- MultimodalBinder: Cross-modal binding via inter-bank phase locking
- BindingStrength: Measure phase coherence between modality banks
- CrossModalAttention: Phase-gated cross-attention
- ModalityEncoder: Encode raw modality input for oscillator binding

References:
- von der Malsburg (1981). The correlation theory of brain function.
- Singer & Gray (1995). Visual feature integration and the temporal
  correlation hypothesis. Annual Review of Neuroscience.
- Engel et al. (2001). Dynamic predictions: oscillations and synchrony
  in top-down processing. Nature Reviews Neuroscience.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple


class ModalityEncoder(nn.Module):
    """Encode raw modality input into oscillator-compatible representation.

    Projects input features to oscillator space and derives initial
    phases from the input structure.

    Args:
        input_dim: Dimension of raw modality input.
        oscillator_dim: Dimension of oscillator representation.
        num_oscillators: Number of oscillators for this modality.
    """

    def __init__(self, input_dim: int, oscillator_dim: int, num_oscillators: int):
        super().__init__()
        self.input_dim = input_dim
        self.oscillator_dim = oscillator_dim
        self.num_oscillators = num_oscillators

        self.feature_proj = nn.Linear(input_dim, oscillator_dim)
        self.phase_proj = nn.Linear(input_dim, num_oscillators)
        self.amplitude_proj = nn.Linear(input_dim, num_oscillators)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode modality input.

        Args:
            x: (..., input_dim) raw features

        Returns:
            Dict with 'encoded' (..., oscillator_dim), 'phases' (..., N),
            'amplitudes' (..., N)
        """
        encoded = self.feature_proj(x)
        phases = self.phase_proj(x) % (2 * math.pi)
        amplitudes = torch.sigmoid(self.amplitude_proj(x))
        return {
            'encoded': encoded,
            'phases': phases,
            'amplitudes': amplitudes,
        }


class BindingStrength(nn.Module):
    """Measure how strongly two modalities are bound via phase coherence.

    Uses inter-bank phase coherence: the mean of cos(phi_a_i - phi_b_j)
    over coupled oscillator pairs. High coherence means the modalities
    are "about the same thing."

    Phase-locking value (PLV) is a standard neuroscience metric for
    assessing functional connectivity between neural populations.
    """

    def forward(
        self,
        phases_a: torch.Tensor,
        phases_b: torch.Tensor,
        coupling_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute binding strength between two modality banks.

        Args:
            phases_a: (..., N_a) phases from modality A
            phases_b: (..., N_b) phases from modality B
            coupling_mask: Optional (N_a, N_b) mask for coupled pairs

        Returns:
            (...,) binding strength in [0, 1]
        """
        # Phase differences: (..., N_a, N_b)
        diff = phases_a.unsqueeze(-1) - phases_b.unsqueeze(-2)

        # Phase locking value: |mean(exp(i * diff))|
        z = torch.exp(1j * diff.to(torch.complex64))

        if coupling_mask is not None:
            z = z * coupling_mask.unsqueeze(0) if z.dim() > 2 else z * coupling_mask
            n_pairs = coupling_mask.sum().clamp(min=1)
            plv = torch.abs(z.sum(dim=(-2, -1)) / n_pairs)
        else:
            plv = torch.abs(z.mean(dim=(-2, -1)))

        return plv.real


class CrossModalAttention(nn.Module):
    """Phase-gated cross-attention between modalities.

    Standard cross-attention where attention weights are modulated by
    oscillator phase differences: in-phase tokens attend more strongly.

    gate(i,j) = (1 + cos(phi_a_i - phi_b_j)) / 2

    This implements the "communication through coherence" (CTC) hypothesis
    (Fries, 2005): neural populations communicate more effectively when
    their oscillations are phase-aligned.

    Args:
        dim_a: Feature dimension for modality A.
        dim_b: Feature dimension for modality B.
        num_heads: Number of attention heads.
    """

    def __init__(self, dim_a: int, dim_b: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_a // num_heads

        assert dim_a % num_heads == 0

        # Cross attention: A attends to B
        self.q_a = nn.Linear(dim_a, dim_a, bias=False)
        self.k_b = nn.Linear(dim_b, dim_a, bias=False)
        self.v_b = nn.Linear(dim_b, dim_a, bias=False)
        self.out_a = nn.Linear(dim_a, dim_a, bias=False)

        # Cross attention: B attends to A
        self.q_b = nn.Linear(dim_b, dim_b, bias=False)
        self.k_a = nn.Linear(dim_a, dim_b, bias=False)
        self.v_a = nn.Linear(dim_a, dim_b, bias=False)
        self.out_b = nn.Linear(dim_b, dim_b, bias=False)

    def _cross_attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        phases_q: torch.Tensor,
        phases_k: torch.Tensor,
        out_proj: nn.Linear,
    ) -> torch.Tensor:
        """Single-direction phase-gated cross-attention."""
        B = q.shape[0]
        T_q = q.shape[1] if q.dim() > 2 else 1
        T_k = k.shape[1] if k.dim() > 2 else 1

        if q.dim() == 2:
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)

        H = self.num_heads
        d = self.head_dim

        q_h = q.view(B, T_q, H, d).transpose(1, 2)
        k_h = k.view(B, T_k, H, d).transpose(1, 2)
        v_h = v.view(B, T_k, H, d).transpose(1, 2)

        # Standard attention scores
        attn = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(d)

        # Phase gating: (1 + cos(phi_q - phi_k)) / 2
        if phases_q.dim() == 1:
            phases_q = phases_q.unsqueeze(0)
        if phases_k.dim() == 1:
            phases_k = phases_k.unsqueeze(0)

        # Average phases across oscillators for per-position gate
        # 3D (B, T, N_osc) → mean over osc → (B, T, 1)
        # 2D (B, N_osc) → mean over osc → (B, 1) → (B, 1, 1) for broadcasting
        if phases_q.dim() > 2:
            pq = phases_q.mean(dim=-1, keepdim=True)
        else:
            pq = phases_q.mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
        if phases_k.dim() > 2:
            pk = phases_k.mean(dim=-1, keepdim=True)
        else:
            pk = phases_k.mean(dim=-1).unsqueeze(-1).unsqueeze(-1)

        # Ensure shapes: (B, T_q, 1) and (B, T_k, 1) for broadcasting
        if pq.shape[1] != T_q:
            pq = pq.expand(B, T_q, 1)
        if pk.shape[1] != T_k:
            pk = pk.expand(B, T_k, 1)

        phase_diff = pq - pk.transpose(1, 2)  # (B, T_q, T_k)
        phase_gate = (1 + torch.cos(phase_diff)) / 2  # [0, 1]
        phase_gate = phase_gate.unsqueeze(1)  # (B, 1, T_q, T_k) broadcast over heads

        # Gated attention
        attn = torch.softmax(attn, dim=-1) * phase_gate
        # Renormalize
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        out = torch.matmul(attn, v_h)
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1)
        return out_proj(out).squeeze(1) if T_q == 1 else out_proj(out)

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        phases_a: torch.Tensor,
        phases_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bidirectional phase-gated cross-attention.

        Args:
            x_a: (B, [T_a,] D_a) features from modality A
            x_b: (B, [T_b,] D_b) features from modality B
            phases_a: (B, N_a) phases from modality A oscillators
            phases_b: (B, N_b) phases from modality B oscillators

        Returns:
            (attended_a, attended_b): Phase-gated cross-attended features
        """
        attended_a = self._cross_attend(
            self.q_a(x_a), self.k_b(x_b), self.v_b(x_b),
            phases_a, phases_b, self.out_a,
        )
        attended_b = self._cross_attend(
            self.q_b(x_b), self.k_a(x_a), self.v_a(x_a),
            phases_b, phases_a, self.out_b,
        )
        return attended_a, attended_b


class MultimodalBinder(nn.Module):
    """Cross-modal binding via inter-bank phase locking.

    Each modality gets its own oscillator bank. Inter-bank coupling
    drives oscillators from different modalities to synchronize when
    they refer to the same entity. The coupling matrix is learned.

    This implements the oscillatory solution to the binding problem:
    "what I see" and "what I hear" are bound when their oscillator
    phases are locked.

    Args:
        modality_dims: Dict mapping modality name to feature dimension.
        binding_dim: Common binding representation dimension.
        num_oscillators_per_modality: Oscillators per modality bank.
        dt: Integration timestep.
        coupling_strength: Base inter-modality coupling strength.
        num_steps: Number of Kuramoto synchronization steps.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        binding_dim: int = 64,
        num_oscillators_per_modality: int = 16,
        dt: float = 0.01,
        coupling_strength: float = 1.0,
        num_steps: int = 10,
    ):
        super().__init__()
        self.modality_names = sorted(modality_dims.keys())
        self.binding_dim = binding_dim
        self.num_osc = num_oscillators_per_modality
        self.dt = dt
        self.num_steps = num_steps
        self.num_modalities = len(modality_dims)

        # Per-modality encoders
        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(dim, binding_dim, num_oscillators_per_modality)
            for name, dim in modality_dims.items()
        })

        # Per-modality natural frequencies (different base frequencies)
        self.omegas = nn.ParameterDict({
            name: nn.Parameter(
                torch.randn(num_oscillators_per_modality) * 0.1 + 1.0 + i * 0.5
            )
            for i, name in enumerate(self.modality_names)
        })

        # Inter-modality coupling (learned)
        # Shape: (total_osc, total_osc) where total_osc = num_modalities * num_osc
        total_osc = self.num_modalities * num_oscillators_per_modality
        self.inter_coupling = nn.Parameter(
            torch.randn(total_osc, total_osc) * 0.1 * coupling_strength
        )

        # Intra-modality coupling
        self.intra_coupling = nn.Parameter(torch.tensor(coupling_strength))

        # Fusion layer
        self.fusion = nn.Linear(binding_dim * self.num_modalities, binding_dim)

        # Binding strength meter
        self.binding_meter = BindingStrength()

    def forward(
        self, modality_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Bind multi-modal inputs via phase synchronization.

        Args:
            modality_inputs: Dict mapping modality name to (B, D_m) features.
                Must contain at least the modalities specified at init.

        Returns:
            Dict with:
            - bound_representation: (B, binding_dim) fused representation
            - binding_matrix: (M, M) pairwise binding strengths
            - per_modality_phases: Dict[str, (B, N)] final phases
            - per_modality_encoded: Dict[str, (B, binding_dim)]
            - total_binding: (B,) scalar binding strength
        """
        B = next(iter(modality_inputs.values())).shape[0]
        device = next(iter(modality_inputs.values())).device

        # Encode each modality
        encoded = {}
        all_phases = []
        for name in self.modality_names:
            if name not in modality_inputs:
                # Use zeros for missing modalities
                enc = {
                    'encoded': torch.zeros(B, self.binding_dim, device=device),
                    'phases': torch.rand(B, self.num_osc, device=device) * 2 * math.pi,
                    'amplitudes': torch.ones(B, self.num_osc, device=device),
                }
            else:
                enc = self.encoders[name](modality_inputs[name])
            encoded[name] = enc
            all_phases.append(enc['phases'])

        # Concatenate all phases: (B, total_osc)
        phases = torch.cat(all_phases, dim=-1)

        # Build omega vector: (total_osc,)
        omega = torch.cat([self.omegas[name] for name in self.modality_names])

        # Symmetrize coupling matrix
        coupling = (self.inter_coupling + self.inter_coupling.T) / 2

        # Run Kuramoto dynamics with inter-modality coupling
        N = phases.shape[-1]
        for _ in range(self.num_steps):
            # Phase differences: (B, N, N)
            diff = phases.unsqueeze(-1) - phases.unsqueeze(-2)
            # Coupling forces
            forces = (coupling.unsqueeze(0) * torch.sin(diff)).sum(dim=-1) / N
            # Euler step
            phases = (phases + self.dt * (omega.unsqueeze(0) + forces)) % (2 * math.pi)

        # Split phases back per modality
        per_modality_phases = {}
        idx = 0
        for name in self.modality_names:
            per_modality_phases[name] = phases[:, idx:idx + self.num_osc]
            idx += self.num_osc

        # Compute binding matrix (M x M pairwise strengths)
        M = self.num_modalities
        binding_matrix = torch.zeros(M, M, device=device)
        for i, name_i in enumerate(self.modality_names):
            for j, name_j in enumerate(self.modality_names):
                if i != j:
                    bs = self.binding_meter(
                        per_modality_phases[name_i],
                        per_modality_phases[name_j],
                    )
                    binding_matrix[i, j] = bs.mean()
                else:
                    binding_matrix[i, j] = 1.0

        # Phase-modulated fusion
        fused_parts = []
        for name in self.modality_names:
            # Modulate encoded features by phase coherence
            phase_mod = torch.cos(per_modality_phases[name]).mean(dim=-1, keepdim=True)
            fused_parts.append(encoded[name]['encoded'] * (0.5 + 0.5 * phase_mod))

        fused = torch.cat(fused_parts, dim=-1)
        bound = self.fusion(fused)

        # Total binding strength
        total_binding = binding_matrix[
            torch.triu(torch.ones(M, M, dtype=torch.bool), diagonal=1)
        ].mean().expand(B)

        return {
            'bound_representation': bound,
            'binding_matrix': binding_matrix,
            'per_modality_phases': per_modality_phases,
            'per_modality_encoded': {n: encoded[n]['encoded'] for n in self.modality_names},
            'total_binding': total_binding,
        }
