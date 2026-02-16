"""Tensor-only tick result type."""
import torch
from typing import NamedTuple


# Processing mode constants (tensor-friendly, no strings in hot path)
PROCESSING_MODE_PERCEPTION = 0
PROCESSING_MODE_IMAGINATION = 1
PROCESSING_MODE_BALANCED = 2
PROCESSING_MODE_NAMES = {
    PROCESSING_MODE_PERCEPTION: "perception",
    PROCESSING_MODE_IMAGINATION: "imagination",
    PROCESSING_MODE_BALANCED: "balanced",
}
_PROCESSING_MODE_FROM_STR = {v: k for k, v in PROCESSING_MODE_NAMES.items()}


class TickResult(NamedTuple):
    """Tensor-only return type for minimal tick (return_diagnostics=False).

    ALL fields are tensors — no Python strings, ints, or dicts.
    This makes TickResult compatible with torch.compile, CUDA graphs,
    and allocation-free replay.

    Use ``PROCESSING_MODE_NAMES[int(result.processing_mode)]`` to recover
    the string name when needed (diagnostics/logging only).
    """
    temperature: torch.Tensor        # scalar (0-dim)
    personality_blend: torch.Tensor   # (num_populations,) normalized weights
    attention_bias: torch.Tensor      # (state_dim,) or (hierarchy_dims[0],)
    processing_mode: torch.Tensor     # scalar int tensor (0=perception, 1=imagination, 2=balanced)
    coherence: torch.Tensor           # scalar (0-dim)
    dominant_state: torch.Tensor      # scalar int tensor
    motivation: torch.Tensor          # scalar (0-dim)
    sleep_signal: torch.Tensor        # scalar (0-dim)
