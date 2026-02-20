"""Attention mechanisms: phase-based attention and precision weighting."""
from pyquifer.cognition.attention.phase_attention import (
    OscillatorGatedFFN,
    PhaseAttention,
    PhaseMultiHeadAttention,
)
from pyquifer.cognition.attention.precision_weighting import AttentionAsPrecision

__all__ = [
    "PhaseAttention",
    "PhaseMultiHeadAttention",
    "OscillatorGatedFFN",
    "AttentionAsPrecision",
]
