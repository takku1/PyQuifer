"""Criticality feedback loop for torch.compile."""
import math
import torch


def _criticality_feedback(
    osc_phases: torch.Tensor,
    osc_coupling_data: torch.Tensor,
    crit_sigma: torch.Tensor,
    R_current: torch.Tensor,
    dephasing_gain: torch.Tensor = None,
    R_target: torch.Tensor = None,
    colored_noise: torch.Tensor = None,
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
    sigma_error = crit_sigma - 1.0
    abs_err = sigma_error.abs()
    gain_super = 0.005 + 0.03 * sigma_error
    gain_sub = 0.008 + 0.05 * abs_err
    gain = torch.where(
        abs_err < 0.15,
        torch.zeros_like(sigma_error).fill_(0.0005),
        torch.where(sigma_error > 0, gain_super, gain_sub),
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
