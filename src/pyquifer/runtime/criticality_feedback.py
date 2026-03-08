"""Criticality feedback loop for torch.compile."""
import math
from typing import Optional

import torch
import torch.nn.functional as F


def _criticality_feedback(
    osc_phases: torch.Tensor,
    osc_coupling_data: torch.Tensor,
    crit_sigma: torch.Tensor,
    R_current: torch.Tensor,
    dephasing_gain: Optional[torch.Tensor] = None,
    R_target: Optional[torch.Tensor] = None,
    colored_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure-tensor criticality feedback loop (slow + fast paths).

    Extracted from tick() so torch.compile can fuse the ~15 element-wise
    ops into 1-2 kernels.  No .item() calls, no Python branching on
    tensor values, no dict construction — fully traceable.

    Args:
        osc_phases: Oscillator phase buffer (num_oscillators,)
        osc_coupling_data: coupling_strength parameter (scalar tensor, modified in-place)
        crit_sigma: Branching ratio σ from KuramotoCriticalityMonitor
        R_current: Instantaneous order parameter R
        dephasing_gain: Metastability-modulated gain (scalar tensor). When
            provided, replaces the fixed 1.5 multiplier with a value that
            targets SD(R) ≈ target_metastability. Novel formula from
            Michel & Koenig 2018 / Cabral et al.
        R_target: Target R for dephasing threshold (default 0.42 if None)
        colored_noise: Pre-computed OU cascade noise (num_oscillators,).
            When provided, replaces white randn noise for 1/f^β spectrum.

    Returns:
        Updated phases tensor (with dephasing applied)
    """
    # ── SLOW PATH: Homeostatic coupling adjustment ──
    # Smooth dead zone via tanh ramp:
    #   |sigma_error| < 0.05 → scale ≈ 0 (rest near criticality, tiny nudge only)
    #   |sigma_error| > 0.20 → scale ≈ 1 (full proportional correction)
    # Replaces hard torch.where(abs_err < 0.15, fixed_tiny, larger_gain) with
    # a continuous function — no kink at the dead-zone boundary.
    sigma_error = crit_sigma - 1.0
    abs_err = sigma_error.abs()
    _dead_scale = torch.tanh(((abs_err - 0.05) / 0.05).clamp(min=0.0))
    gain = 0.0005 + _dead_scale * torch.where(
        sigma_error >= 0,
        0.030 * abs_err,   # super-critical: gentle reduction
        0.050 * abs_err,   # sub-critical: faster recovery
    )
    coupling_delta = -sigma_error * gain
    new_K = (osc_coupling_data + coupling_delta).clamp(min=0.1, max=5.0)
    osc_coupling_data.copy_(new_K)

    # ── FAST PATH: Metastability-targeting dephasing ──
    # When dephasing_gain is provided (from R_history SD), use it to
    # modulate noise injection. This targets SD(R) as a homeostatic
    # variable rather than a static R threshold.
    _R_tgt = R_target if R_target is not None else torch.tensor(0.42, device=R_current.device)
    _gain = dephasing_gain if dephasing_gain is not None else torch.tensor(1.5, device=R_current.device)
    R_distance = (R_current - _R_tgt).abs()
    noise_scale = (R_distance * _gain).clamp(max=1.2)

    # ── Chimera band gating [0.35, 0.65] ──
    # Smooth softplus replaces relu (clamp min=0) at each band edge.
    # softplus(x, beta=20) ≈ relu(x) for |x| > 0.15, smooth for |x| < 0.15 —
    # removes the kink at exactly R=0.35 and R=0.65 for cleaner gradient flow.
    # beta=20 → transition width ≈ 0.15/20 = 0.075 (tight but differentiable).
    #
    # in-band indicator uses the same soft formula as before, now fed smooth edges.
    # out_scale grows naturally with softplus (no hard .clamp(max=3.0) needed since
    # R ∈ [0,1] and softplus(0.35, beta=20) ≈ 0.35 → out_scale_max ≈ 2.05).
    _excess_hi  = F.softplus(R_current - 0.65, beta=20.0)  # smooth relu above 0.65
    _deficit_lo = F.softplus(0.35 - R_current, beta=20.0)  # smooth relu below 0.35
    _band_deviation = _excess_hi + _deficit_lo
    _in_band    = 1.0 - (_band_deviation / (_band_deviation + 0.05))
    _out_scale  = 1.0 + 3.0 * _excess_hi   # no hard clip: bounded by R ∈ [0,1]
    chimera_mod = _in_band * 0.25 + (1.0 - _in_band) * _out_scale
    noise_scale = (noise_scale * chimera_mod).clamp(max=1.5)

    if colored_noise is not None:
        # Floor ensures colored noise is always present (preserves 1/f
        # temporal structure). R-dependent scaling adds homeostatic amplitude
        # modulation on top — slow enough not to whiten the spectrum.
        noise_scale = noise_scale.clamp(min=0.3)
        dephasing = colored_noise * noise_scale
    else:
        dephasing = torch.randn_like(osc_phases) * noise_scale
    osc_phases.add_(dephasing).remainder_(2 * math.pi)
    return osc_phases


# Compiled version — created lazily by compile_modules()
_compiled_criticality_feedback = None
