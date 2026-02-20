"""
Multimodal Phase-Locking Bus.

Coordinates per-modality oscillator banks via Kuramoto dynamics,
computes cross-modal PLV coherence, and exposes coherence signals
for OscillatoryWriteGate and CrossBleedGate modulation.

Wraps MultimodalBinder + BindingStrength from sensory_binding.py into
a tick-level coordinator that produces a fused representation and
coherence signals for downstream gate modulation.

References:
- Fries (2005). A mechanism for cognitive dynamics: neuronal
  communication through neuronal coherence. Trends in Cognitive Sciences.
- von der Malsburg (1981). The correlation theory of brain function.
"""

from dataclasses import dataclass, field
from typing import Dict, NamedTuple

import torch
import torch.nn as nn

from pyquifer.cognition.binding.sensory import BindingStrength, MultimodalBinder


class BusOutput(NamedTuple):
    """Output from PhaseLockBus.forward()."""
    fused_representation: torch.Tensor   # (B, binding_dim) cross-modal fused signal
    binding_matrix: torch.Tensor         # (M, M) pairwise PLV
    mean_coherence: torch.Tensor         # scalar, mean off-diagonal PLV
    per_modality_phases: Dict[str, torch.Tensor]  # final phases per modality
    total_binding: torch.Tensor          # (B,) scalar total binding
    modality_count: int                  # how many non-zero modalities this tick


@dataclass
class BusConfig:
    """Configuration for PhaseLockBus."""
    modality_dims: Dict[str, int] = field(default_factory=lambda: {"text": 64})
    binding_dim: int = 64
    num_oscillators_per_modality: int = 16
    coupling_strength: float = 1.0
    num_sync_steps: int = 5
    coherence_ema_alpha: float = 0.1
    gate_coherence_weight: float = 1.0
    bleed_coherence_weight: float = 0.5


class PhaseLockBus(nn.Module):
    """Multimodal phase-locking bus.

    Coordinates per-modality oscillator banks via Kuramoto dynamics,
    computes cross-modal PLV coherence, and exposes coherence signals
    for OscillatoryWriteGate and CrossBleedGate modulation.

    Single-modality graceful degradation: with one modality, returns
    coherence=0.5 (neutral) so downstream gates are unaffected.

    Args:
        config: BusConfig with modality dimensions and coupling parameters.
    """

    def __init__(self, config: BusConfig):
        super().__init__()
        self.config = config

        self.binder = MultimodalBinder(
            modality_dims=config.modality_dims,
            binding_dim=config.binding_dim,
            num_oscillators_per_modality=config.num_oscillators_per_modality,
            coupling_strength=config.coupling_strength,
            num_steps=config.num_sync_steps,
        )

        self.binding_meter = BindingStrength()

        # EMA-smoothed coherence (persistent across ticks)
        self.register_buffer('_coherence_ema', torch.tensor(0.5))

        self._gate_weight = config.gate_coherence_weight
        self._bleed_weight = config.bleed_coherence_weight
        self._ema_alpha = config.coherence_ema_alpha

    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> BusOutput:
        """Run phase-locking bus on multimodal inputs.

        Args:
            modality_inputs: Dict mapping modality name to (B, D_m) features.
                Can pass a subset of the registered modalities.

        Returns:
            BusOutput with fused representation, binding matrix, and coherence.
        """
        # Count non-zero modalities actually provided
        modality_count = sum(
            1 for v in modality_inputs.values()
            if v is not None and v.abs().sum() > 0
        )

        # Delegate to MultimodalBinder (runs Kuramoto dynamics)
        result = self.binder(modality_inputs)

        binding_matrix = result['binding_matrix']
        per_modality_phases = result['per_modality_phases']
        fused = result['bound_representation']
        total_binding = result['total_binding']

        # Compute mean cross-coherence from off-diagonal binding matrix
        M = binding_matrix.shape[0]
        if modality_count <= 1:
            # Single modality: neutral coherence so gates are unaffected
            mean_coherence = torch.tensor(0.5, device=binding_matrix.device)
        else:
            # Mean of upper-triangular off-diagonal entries
            mask = torch.triu(torch.ones(M, M, dtype=torch.bool, device=binding_matrix.device), diagonal=1)
            off_diag = binding_matrix[mask]
            mean_coherence = off_diag.mean() if off_diag.numel() > 0 else torch.tensor(0.5, device=binding_matrix.device)

        # Clamp to [0, 1]
        mean_coherence = mean_coherence.clamp(0.0, 1.0)

        # Update EMA
        with torch.no_grad():
            self._coherence_ema.lerp_(mean_coherence, self._ema_alpha)

        return BusOutput(
            fused_representation=fused,
            binding_matrix=binding_matrix,
            mean_coherence=mean_coherence,
            per_modality_phases=per_modality_phases,
            total_binding=total_binding,
            modality_count=modality_count,
        )

    def get_write_gate_modulation(self) -> float:
        """Return coherence-based scaling for OscillatoryWriteGate.

        Returns value in [0, gate_coherence_weight], typically [0, 1].
        """
        return float(self._coherence_ema.clamp(0.0, 1.0) * self._gate_weight)

    def get_bleed_modulation(self) -> float:
        """Return coherence-based scaling for cross-bleed strength.

        Returns value in [0, bleed_coherence_weight], typically [0, 0.5].
        """
        return float(self._coherence_ema.clamp(0.0, 1.0) * self._bleed_weight)
