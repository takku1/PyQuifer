"""
Cognitive Cycle Integration Module for PyQuifer

Wires all 49 modules into a single coordinated cognitive loop.
One tick of this loop = one "moment of consciousness."

The loop:
  1. Oscillators step → phases update
  2. Criticality check → coupling/noise adjustment
  3. Sensory input → hierarchical predictive coding
  4. Precision weighting gates prediction errors
  5. Global workspace competition → conscious broadcast
  6. Metastability → stream of consciousness flow
  7. Causal flow → who's driving whom
  8. Neural darwinism → group selection
  9. Self-model update → narrative identity
  10. Memory consolidation (if sleeping)
  11. Neuromodulation → DA/5-HT/NE/ACh
  12. Output → modulation parameters for LLM

All submodules are imported from the pyquifer package (no external optional deps).

References:
- Friston (2010). The free-energy principle: a unified brain theory?
- Edelman (1987). Neural Darwinism: The Theory of Neuronal Group Selection.
- Rabinovich et al. (2001). Dynamical principles in neuroscience.
- Tononi (2004). An information integration theory of consciousness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Dict, Any, List, NamedTuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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


@dataclass
class CycleConfig:
    """Configuration for CognitiveCycle dimensions.

    All dimension parameters must be consistent across modules.
    Use CycleConfig.default() for a reasonable starting point.
    """
    # Core dimensions
    state_dim: int = 64          # Primary representation dimension
    belief_dim: int = 32         # Abstract belief dimension
    semantic_dim: int = 16       # Compressed semantic dimension

    # Oscillator settings
    num_oscillators: int = 32
    oscillator_dt: float = 0.01

    # Hierarchy levels (bottom to top)
    hierarchy_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    hpc_lr: float = 0.05
    hpc_gen_lr: float = 0.01
    hpc_iterations: int = 3          # Message-passing iterations in HPC (1=fast, 3=accurate)

    # Precision
    precision_tau: float = 20.0
    precision_max: float = 10.0

    # Metastability
    num_populations: int = 6

    # Neural darwinism
    num_groups: int = 6
    group_dim: int = 16
    total_budget: float = 10.0

    # Self-model
    internal_dim: int = 32
    sensory_dim: int = 16
    active_dim: int = 16
    tonic_drift: Optional[torch.Tensor] = None  # Narrative identity drift vector

    # Memory
    episodic_capacity: int = 500
    semantic_slots: int = 100

    # Criticality
    target_branching_ratio: float = 1.0

    # Volatility filter
    volatility_base_lr: float = 0.01
    volatility_min_lr: float = 0.001
    volatility_max_lr: float = 0.1
    volatility_tonic: float = -4.0

    # Phase 5 optional modules (set True to enable)
    use_stp: bool = False           # Tsodyks-Markram short-term plasticity
    use_mean_field: bool = False    # Kuramoto-Daido mean-field reduction
    use_stuart_landau: bool = False # Stuart-Landau oscillator (amplitude+phase)
    use_koopman: bool = False       # Koopman/DMD bifurcation detection
    use_neural_mass: bool = False   # Wilson-Cowan E/I population dynamics

    # Phase 7: Benchmark gap-closure features
    use_attractor_stability: bool = False   # Kuramoto ASI perturbation analysis
    use_hypothesis_arena: bool = False      # Evidence-driven hypothesis competition
    use_evidence_aggregator: bool = False   # Calibrated confidence from evidence
    use_no_progress: bool = False           # Stagnation detection
    use_phase_cache: bool = False           # Phase topology Bayesian prior

    # Phase 9: Global Workspace / Organ protocol
    use_global_workspace: bool = False      # Enable GWT workspace competition
    workspace_dim: int = 256                # Common workspace dimension
    gw_n_winners: int = 1                   # Number of competition winners
    gw_cycle_consistency: bool = False       # Cycle-consistency loss for adapters
    gw_diversity_pressure: float = 0.1      # Anti-collapse pressure

    # Phase 8: Training core
    use_ep_training: bool = False           # Equilibrium Propagation for Kuramoto
    use_oscillation_gated_plasticity: bool = False  # Theta-phase gated learning
    use_three_factor: bool = False          # Three-factor learning rule
    use_oscillatory_predictive: bool = False # Frequency-specific predictive coding
    use_sleep_consolidation: bool = False   # SRC Hebbian sleep update
    use_dendritic_credit: bool = False      # Two-compartment dendritic neurons

    # Phase 10: Multi-workspace ensemble
    use_workspace_ensemble: bool = False   # Enable parallel workspaces with cross-bleed
    n_workspaces: int = 2                  # Number of parallel workspaces
    ensemble_workspace_dim: int = 256      # Workspace dim for ensemble (uses workspace_dim if 0)
    cross_bleed_strength: float = 0.3      # How much background workspaces bleed into active
    standing_momentum: float = 0.9         # EMA momentum for standing broadcasts

    # Phase 9b: MCP-as-Organ protocol
    use_mcp_organs: bool = False           # Enable MCP resource endpoints as workspace organs
    mcp_organ_latent_dim: int = 64         # Default latent dim for MCP organs

    # Phase 9b: Cross-module integration wiring
    use_circadian: bool = False            # ChronobiologicalSystem → consolidation timing
    use_personality_attractor: bool = False # PersonalityAttractor → metastability/personality
    use_somatic: bool = False              # SomaticManifold → self-model/narrative
    use_phase_dominance: bool = False      # Oscillator phases → causal flow dominance
    use_motivation_priority: bool = False  # Motivation → memory consolidation priority

    # Phase 11-18: New module feature flags
    use_visual_binding: bool = False       # AKOrN visual segmentation
    use_temporal_binding: bool = False     # SequenceAKOrN token binding
    use_sensory_binding: bool = False      # Cross-modal phase-locked binding
    use_deep_aif: bool = False             # Multi-step deep active inference
    use_jepa_world_model: bool = False     # JEPA latent predictor
    use_deliberation: bool = False         # Test-time compute / deliberation
    use_cls_memory: bool = False           # Complementary learning systems
    use_causal_reasoning: bool = False     # Causal graph / do-calculus
    use_appraisal: bool = False            # Cognitive appraisal model
    use_selective_ssm: bool = False        # Mamba-style selective SSM
    use_oscillatory_moe: bool = False      # Oscillator-driven MoE routing
    use_prospective_learning: bool = False # Prospective configuration learning
    solver: str = "euler"                  # "euler", "rk4", "dopri5"
    use_complex_oscillators: bool = False  # Complex z=A*exp(i*phi) backend
    use_cuda_kernels: bool = False         # CUDA Kuramoto acceleration

    # Basal ganglia gating loop
    use_gating_loop: bool = False          # Enable BG-thalamic gating loop
    gating_da_go_weight: float = 2.0      # D1 pathway DA sensitivity
    gating_da_nogo_weight: float = 1.5    # D2 pathway DA sensitivity
    gating_stn_surprise: float = 0.8      # Hyperdirect surprise threshold
    gating_switching_penalty: float = 0.1  # Hysteresis for channel switching

    # Multimodal Phase-Locking Bus
    use_phase_lock_bus: bool = False
    phase_lock_modality_dims: Optional[Dict[str, int]] = None  # None = {"text": state_dim}
    phase_lock_sync_steps: int = 5
    phase_lock_coherence_ema: float = 0.1

    # Enhancement C: Energy-Metabolic Budget
    use_metabolic_budget: bool = False     # Enable energy/metabolic constraints per tick
    metabolic_capacity: float = 1.0       # Max energy budget (0-1 scale)
    metabolic_recovery_rate: float = 0.02  # Energy recovered per idle tick
    metabolic_oscillation_cost: float = 0.005  # Cost per tick of high-coherence oscillation
    metabolic_ignition_cost: float = 0.05  # Cost of workspace ignition event
    metabolic_broadcast_cost: float = 0.02 # Cost of workspace broadcast write

    # v7: Neuroscience dynamics alignment
    use_ou_dephasing: bool = False         # Multi-timescale OU noise cascade (1/f spectrum)
    use_spike_activity: bool = False       # Spike-based activity buffer (avalanche scaling)
    diagnostics_buffer_len: int = 200      # Length of R/activity history buffers

    # Phase 6: Integration method (G-17)
    integration_method: str = 'euler'  # 'euler' or 'rk4' — passed to oscillators/neural_mass

    # Metastability-targeting dephasing (Fix 6 — neuroscience alignment v3)
    # Target SD(R) rather than static R threshold. Consciousness =
    # medium R with high SD(R) (metastability), not high R.
    # Based on: Michel & Koenig 2018 microstates, Cabral et al. connectome.
    target_metastability: float = 0.0   # 0 = auto-scale via 1/sqrt(N); >0 = manual override
    target_R: float = 0.42              # optimal R for consciousness (0.3-0.5 range)

    # Oscillator topology (Fix 13: modular → richer metastability)
    oscillator_topology: str = 'global'       # 'global', 'small_world', 'scale_free', 'ring', 'modular'
    oscillator_topology_params: Optional[Dict[str, Any]] = None  # Topology-specific params
    oscillator_frustration: float = 0.0  # Kuramoto-Sakaguchi phase-lag α (radians). π/4 promotes chimera states.

    # True theta-gamma PAC (Fix 11: Tort MI)
    use_theta_gamma_pac: bool = False     # Partition oscillators into theta/gamma banks
    theta_oscillators: int = 8            # Number of oscillators in theta band (4-8 Hz)
    theta_freq_range: tuple = (4.0, 8.0)  # Theta frequency in Hz
    gamma_freq_range: tuple = (30.0, 80.0) # Gamma frequency in Hz

    # R-oscillation meta-coupling (renamed from use_cross_freq_coupling for clarity)
    # This is NOT true PAC — it couples the R order-parameter oscillation (~1-2 Hz)
    # as a "theta" proxy. For real theta-gamma PAC, use use_theta_gamma_pac instead.
    use_cross_freq_coupling: bool = False  # Backward compat alias
    use_meta_r_coupling: bool = False      # Preferred name (same effect)

    # Hierarchical timestepping: run slow modules less often than fast ones.
    # Value = run every N ticks. 1 = every tick (default/legacy behavior).
    step_every_hpc: int = 1           # Hierarchical predictive coding
    step_every_sr: int = 1            # Stochastic resonance
    step_every_motivation: int = 1    # IntrinsicMotivationSystem
    step_every_metastability: int = 1 # WinnerlessCompetition
    step_every_precision: int = 1     # Precision weighting
    step_every_arena: int = 1         # Neural darwinism arena
    step_every_selfmodel: int = 1     # Self-model + narrative

    @staticmethod
    def default():
        return CycleConfig()

    @staticmethod
    def small():
        """Smaller config for testing or low-resource environments."""
        return CycleConfig(
            state_dim=32, belief_dim=16, semantic_dim=8,
            num_oscillators=16, hierarchy_dims=[32, 16, 8],
            num_populations=4, num_groups=4, group_dim=8,
            internal_dim=16, sensory_dim=8, active_dim=8,
            episodic_capacity=200, semantic_slots=50,
            volatility_base_lr=0.01, volatility_min_lr=0.001,
            volatility_max_lr=0.1,
        )

    @staticmethod
    def interactive():
        """Config tuned for real-time streaming (<=2ms target).

        Uses aggressive hierarchical timestepping: slow modules run
        less often, with cached results interpolated between updates.
        HPC runs with 1 iteration (bottom-up only) for speed.
        """
        return CycleConfig(
            hpc_iterations=1,          # Bottom-up only (3x faster HPC)
            step_every_hpc=2,          # HPC every 2 ticks
            step_every_sr=3,           # SR every 3 ticks
            step_every_motivation=5,   # Motivation every 5 ticks
            step_every_metastability=3,# Metastability every 3 ticks
            step_every_precision=2,    # Precision every 2 ticks
            step_every_arena=3,        # Arena every 3 ticks
            step_every_selfmodel=5,    # Self-model every 5 ticks
        )

    @staticmethod
    def realtime():
        """Config for absolute minimum latency (<1.5ms target).

        Maximum hierarchical timestepping: HPC runs 1 iteration every
        3 ticks. Motivation/self-model run very infrequently. Arena
        and metastability heavily cached. The recognition model
        provides amortized inference; iterative refinement only fires
        when prediction error is high (early stopping in HPC).
        """
        return CycleConfig(
            hpc_iterations=1,           # Single bottom-up sweep
            step_every_hpc=3,           # HPC every 3 ticks
            step_every_sr=5,            # SR every 5 ticks
            step_every_motivation=8,    # Motivation every 8 ticks
            step_every_metastability=5, # Metastability every 5 ticks
            step_every_precision=3,     # Precision every 3 ticks
            step_every_arena=5,         # Arena every 5 ticks
            step_every_selfmodel=10,    # Self-model every 10 ticks
        )

    @staticmethod
    def neuroscience():
        """Config tuned for maximum neuroscience alignment (10/10).

        Uses modular topology (Deco et al. 2017), true theta-gamma PAC
        with Tort MI (Tort et al. 2010), size-normalized metastability
        target, and avalanche-consciousness linking (Neuron 2025).
        """
        return CycleConfig(
            oscillator_topology='modular',
            oscillator_topology_params={
                'n_modules': 4,
                'intra_density': 0.8,
                'inter_density': 0.1,
            },
            oscillator_frustration=math.pi / 4,  # Kuramoto-Sakaguchi α — promotes chimera states
            use_theta_gamma_pac=True,
            theta_oscillators=8,
            target_metastability=0.0,  # auto-scale via 1/sqrt(N)
            # v7: Dynamics fixes for PhD-defensible neuroscience metrics
            use_ou_dephasing=True,        # OU cascade → 1/f^β spectral slope
            use_spike_activity=True,      # Spike-based activity → power-law avalanches
            diagnostics_buffer_len=1000,  # Longer buffer → proper LZ/DFA scaling
        )


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


class CognitiveCycle(nn.Module):
    """
    One tick of the full cognitive loop.

    Wires together all PyQuifer modules into a coordinated cycle.
    All submodules are internal to pyquifer and imported at init time.
    Use CycleConfig flags to enable/disable optional subsystems.

    Usage:
        config = CycleConfig.default()
        cycle = CognitiveCycle(config)

        # Each tick takes sensory input and returns modulation state
        for sensory in stream:
            result = cycle.tick(sensory)
            # result['modulation'] has temperature, personality, attention params
    """

    def __init__(self, config: CycleConfig = None):
        super().__init__()
        self.config = config or CycleConfig.default()
        c = self.config

        # Track tick count (buffer for serialisation, Python int for
        # fast modular checks without GPU sync)
        self.register_buffer('tick_count', torch.tensor(0))
        self._tick_py: int = 0

        # Preallocated scratch buffers for the fast return path (TickResult).
        # These avoid per-tick torch.tensor() allocations for scalar conversions.
        self.register_buffer('_scratch_temperature', torch.tensor(0.5))
        self.register_buffer('_scratch_processing_mode', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_scratch_dominant_state', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_scratch_motivation', torch.tensor(0.5))
        self.register_buffer('_scratch_sleep_signal', torch.tensor(0.0))
        self.register_buffer('_scratch_personality', torch.zeros(c.num_populations))

        # Enhancement C: Metabolic budget state
        self.register_buffer('_energy_budget', torch.tensor(c.metabolic_capacity))
        self.register_buffer('_metabolic_debt', torch.tensor(0.0))

        # Cached consciousness metrics (updated every tick, readable from bridge)
        self.register_buffer('_cached_criticality_distance', torch.tensor(0.5))
        self.register_buffer('_cached_free_energy', torch.tensor(0.0))
        self.register_buffer('_cached_identity_strength', torch.tensor(0.0))

        # Cached workspace state (updated when workspace competition runs)
        self._cached_gw_winner_id: str = ""
        self._cached_gw_broadcast: Optional[torch.Tensor] = None

        # ── Fix 6: R history circular buffer for metastability targeting ──
        # Track recent R values to compute SD(R) = observed metastability.
        # Michel & Koenig 2018: optimal consciousness at SD(R) ≈ 0.1-0.2.
        self.register_buffer('_R_history', torch.zeros(50))
        self._R_history_ptr: int = 0
        self.register_buffer('_cached_metastability', torch.tensor(0.0))
        self.register_buffer('_cached_dephasing_gain', torch.tensor(1.5))
        self.register_buffer('_R_target', torch.tensor(c.target_R))

        # ── Long R history for neuroscience diagnostics (DFA, spectral, etc.) ──
        # Separate from the 50-sample buffer used for SD(R) — that one is tuned.
        _buf_len = c.diagnostics_buffer_len
        self.register_buffer('_R_long_history', torch.zeros(_buf_len))
        self._R_long_ptr: int = 0

        # ── Oscillator activity buffer for avalanche detection ──
        # Sum of |sin(phase_i)| per tick — analogous to summed LFP/MEA activity.
        # R(t) is too smooth for proper avalanche scaling; this captures bursts.
        self.register_buffer('_activity_long_history', torch.zeros(_buf_len))
        self._activity_long_ptr: int = 0

        # ── v7: Multi-timescale OU dephasing cascade ──
        # Superposition of 4 OU processes with log-spaced tau produces
        # approximate 1/f^β noise (Kaulakys & Meskauskas 1998).
        self._ou_cascade = None
        if c.use_ou_dephasing:
            from pyquifer.stochastic_resonance import OrnsteinUhlenbeckNoise
            # σ²τ = const for each component → 1/f superposition
            # (Kaulakys & Meskauskas 1998). C=2.0 chosen empirically to
            # dominate Kuramoto mean-field smoothing on R(t).
            _C = 2.0
            self._ou_cascade = nn.ModuleList([
                OrnsteinUhlenbeckNoise(dim=c.num_oscillators, tau=2.0, sigma=_C / 2.0**0.5),
                OrnsteinUhlenbeckNoise(dim=c.num_oscillators, tau=10.0, sigma=_C / 10.0**0.5),
                OrnsteinUhlenbeckNoise(dim=c.num_oscillators, tau=50.0, sigma=_C / 50.0**0.5),
                OrnsteinUhlenbeckNoise(dim=c.num_oscillators, tau=200.0, sigma=_C / 200.0**0.5),
            ])

        # ── Fix 12: Size-normalized metastability target ──
        # SD(R) ~ 1/sqrt(N) for finite Kuramoto (analytical result).
        # Base 0.15 calibrated for N=64 (Cabral et al. connectome).
        if c.target_metastability > 0:
            effective_meta = c.target_metastability
        else:
            effective_meta = 0.15 * math.sqrt(64.0 / c.num_oscillators)
        self._effective_target_meta = effective_meta

        # ── Fix 9/11: Cross-frequency PAC ──
        self._cfc = None
        self._theta_gamma_enabled = c.use_theta_gamma_pac
        self.register_buffer('_cached_pac_strength', torch.tensor(0.0))
        self.register_buffer('_cached_pac_mi', torch.tensor(0.0))  # Tort MI
        self.register_buffer('_R_ema', torch.tensor(0.5))

        # Tort MI circular buffers (18 bins × 20° each = 360°)
        self._tort_n_bins = 18
        self.register_buffer('_tort_theta_buf', torch.zeros(200))  # circular buffer
        self.register_buffer('_tort_gamma_buf', torch.zeros(200))
        self._tort_buf_ptr: int = 0
        self._tort_buf_filled: int = 0

        # ── Fix 16: Working memory capacity prediction (Lisman & Jensen 2013) ──
        self.register_buffer('_cached_wm_capacity', torch.tensor(0.0))

        # ── Fix 10: Consciousness quality metric Φ_m ──
        self.register_buffer('_cached_phi_m', torch.tensor(0.0))

        # === Layer 3: Dynamical Core ===
        from pyquifer.oscillators import LearnableKuramotoBank, SensoryCoupling
        self.oscillators = LearnableKuramotoBank(
            num_oscillators=c.num_oscillators,
            dt=c.oscillator_dt,
            integration_method=c.integration_method,
            topology=c.oscillator_topology,
            topology_params=c.oscillator_topology_params,
            frustration=c.oscillator_frustration,
        )

        # ── Fix 11: Theta-gamma frequency band assignment ──
        # Partition oscillators into theta (4-8 Hz) and gamma (30-80 Hz) banks.
        # Uses set_frequency_bands() to convert Hz → rad/step.
        if c.use_theta_gamma_pac:
            n_theta = min(c.theta_oscillators, c.num_oscillators - 1)
            n_gamma = c.num_oscillators - n_theta
            self.oscillators.set_frequency_bands([
                (n_theta, c.theta_freq_range[0], c.theta_freq_range[1]),
                (n_gamma, c.gamma_freq_range[0], c.gamma_freq_range[1]),
            ], dt=c.oscillator_dt)

        # Sensory-oscillator coupling: frequency entrainment, phase resets,
        # coupling modulation from input (Lakatos et al. 2008; Fries 2005)
        self._sensory_coupling = SensoryCoupling(
            input_dim=c.state_dim,
            num_oscillators=c.num_oscillators,
        )

        from pyquifer.criticality import (
            CriticalityController, KuramotoCriticalityMonitor,
            phase_activity_to_spikes,
        )
        self._phase_to_spikes = phase_activity_to_spikes
        self.criticality = CriticalityController(
            target_branching_ratio=c.target_branching_ratio,
        )
        # Kuramoto-specific criticality: uses order parameter R directly
        self._kuramoto_criticality = KuramotoCriticalityMonitor(
            window_size=50,
        )

        from pyquifer.metastability import MetastabilityIndex
        self.metastability = MetastabilityIndex(
            num_populations=c.num_populations,
        )

        # === Layer 5: Consciousness ===
        from pyquifer.hierarchical_predictive import HierarchicalPredictiveCoding
        self.hpc = HierarchicalPredictiveCoding(
            level_dims=c.hierarchy_dims,
            lr=c.hpc_lr,
            gen_lr=c.hpc_gen_lr,
            learn_every=3,  # amortize autograd cost: learn every 3rd call
            num_iterations=c.hpc_iterations,
        )

        from pyquifer.precision_weighting import AttentionAsPrecision
        self.precision = AttentionAsPrecision(
            num_channels=c.hierarchy_dims[0],
            tau=c.precision_tau,
            max_precision=c.precision_max,
        )

        from pyquifer.neuromodulation import NeuromodulatorDynamics
        self.neuromodulation = NeuromodulatorDynamics()

        # === Layer 4: Information Flow ===
        from pyquifer.causal_flow import DominanceDetector
        self.dominance = DominanceDetector(
            num_levels=len(c.hierarchy_dims),
            buffer_size=200,
        )

        from pyquifer.stochastic_resonance import AdaptiveStochasticResonance
        self.stochastic_resonance = AdaptiveStochasticResonance(
            dim=c.hierarchy_dims[0],
            threshold=0.5,
        )

        # === Layer 1: Learning ===
        from pyquifer.neural_darwinism import SelectionArena, SymbiogenesisDetector
        self.arena = SelectionArena(
            num_groups=c.num_groups,
            group_dim=c.group_dim,
            total_budget=c.total_budget,
        )
        self.symbiogenesis = SymbiogenesisDetector(
            num_groups=c.num_groups,
            group_dim=c.group_dim,
        )

        from pyquifer.memory_consolidation import (
            EpisodicBuffer, SharpWaveRipple, ConsolidationEngine
        )
        self.episodic_buffer = EpisodicBuffer(
            state_dim=c.state_dim,
            capacity=c.episodic_capacity,
        )
        self.sharp_wave_ripple = SharpWaveRipple(state_dim=c.state_dim)
        self.consolidation = ConsolidationEngine(
            state_dim=c.state_dim,
            semantic_dim=c.semantic_dim,
            num_semantic_slots=c.semantic_slots,
        )

        # === Layer 6: Metacognition ===
        from pyquifer.motivation import IntrinsicMotivationSystem
        self.motivation = IntrinsicMotivationSystem(state_dim=c.state_dim)

        # === Layer 7: Self-Model ===
        from pyquifer.self_model import MarkovBlanket, SelfModel, NarrativeIdentity
        self.markov_blanket = MarkovBlanket(
            internal_dim=c.internal_dim,
            sensory_dim=c.sensory_dim,
            active_dim=c.active_dim,
        )
        self.self_model = SelfModel(self_dim=c.internal_dim)
        self.narrative = NarrativeIdentity(
            dim=c.internal_dim,
            tonic_drift=c.tonic_drift,
        )

        # === Layer 8: Volatility Filter ===
        from pyquifer.volatility_filter import VolatilityGatedLearning
        self.volatility_gate = VolatilityGatedLearning(
            dim=c.hierarchy_dims[0],
            base_lr=c.volatility_base_lr,
            min_lr=c.volatility_min_lr,
            max_lr=c.volatility_max_lr,
            tonic_volatility=c.volatility_tonic,
        )

        # === Phase 5 Optional Modules ===
        self._stp = None
        self._mean_field = None
        self._stuart_landau = None
        self._koopman = None
        self._neural_mass = None

        if c.use_stp:
            from pyquifer.short_term_plasticity import TsodyksMarkramSynapse
            self._stp = TsodyksMarkramSynapse(num_synapses=c.hierarchy_dims[0])

        if c.use_mean_field:
            from pyquifer.oscillators import KuramotoDaidoMeanField
            self._mean_field = KuramotoDaidoMeanField(
                omega_mean=1.0,
                delta=0.5,
                coupling=1.0,
                dt=c.oscillator_dt,
            )

        if c.use_stuart_landau:
            from pyquifer.oscillators import StuartLandauOscillator
            self._stuart_landau = StuartLandauOscillator(
                num_oscillators=c.num_oscillators,
                mu=0.1,  # Near criticality
                coupling=1.0,
                dt=c.oscillator_dt,
                integration_method=c.integration_method,
            )

        if c.use_koopman:
            from pyquifer.criticality import KoopmanBifurcationDetector
            self._koopman = KoopmanBifurcationDetector(
                state_dim=c.state_dim,
                delay_dim=10,
                rank=5,
                compute_every=10,
            )

        if c.use_neural_mass:
            from pyquifer.neural_mass import WilsonCowanNetwork
            self._neural_mass = WilsonCowanNetwork(
                num_populations=c.num_populations,
                coupling_strength=0.5,
                dt=0.1,
                integration_method=c.integration_method,
            )

        # === Phase 8 Training Core ===
        self._ep_trainer = None
        self._oscillation_gated = None
        self._three_factor = None
        self._oscillatory_predictive = None
        self._sleep_consolidation = None
        self._dendritic_stack = None

        if c.use_ep_training:
            from pyquifer.equilibrium_propagation import EquilibriumPropagationTrainer
            self._ep_trainer = EquilibriumPropagationTrainer(
                self.oscillators, lr=0.01, beta=0.1,
                free_steps=100, nudge_steps=100,
            )

        if c.use_oscillation_gated_plasticity:
            from pyquifer.learning import OscillationGatedPlasticity
            self._oscillation_gated = OscillationGatedPlasticity(
                shape=(c.hierarchy_dims[0],),
                preferred_phase=math.pi,
                decay_rate=0.95,
            )

        if c.use_three_factor:
            from pyquifer.learning import ThreeFactorRule
            self._three_factor = ThreeFactorRule(
                input_dim=c.hierarchy_dims[0],
                output_dim=c.hierarchy_dims[0],
                trace_decay=0.95,
                homeostatic_target=0.1,
            )

        if c.use_oscillatory_predictive:
            from pyquifer.hierarchical_predictive import OscillatoryPredictiveCoding
            self._oscillatory_predictive = OscillatoryPredictiveCoding(
                dims=c.hierarchy_dims,
                lr=c.hpc_lr,
                inference_steps=5,
            )

        if c.use_sleep_consolidation:
            from pyquifer.memory_consolidation import SleepReplayConsolidation
            self._sleep_consolidation = SleepReplayConsolidation(
                layer_dims=c.hierarchy_dims,
                sleep_lr=0.001,
                noise_scale=1.0,
                num_replay_steps=50,
            )

        if c.use_dendritic_credit:
            from pyquifer.dendritic import DendriticStack
            self._dendritic_stack = DendriticStack(
                dims=c.hierarchy_dims,
                lr=0.01,
            )

        # === Phase 7 Optional Modules ===
        self._no_progress = None
        self._evidence_aggregator = None
        self._phase_cache = None

        if c.use_no_progress:
            from pyquifer.criticality import NoProgressDetector
            self._no_progress = NoProgressDetector(window_size=30)

        if c.use_evidence_aggregator:
            from pyquifer.metacognitive import EvidenceAggregator
            self._evidence_aggregator = EvidenceAggregator()

        if c.use_phase_cache:
            from pyquifer.oscillators import PhaseTopologyCache
            self._phase_cache = PhaseTopologyCache(capacity=1000)

        # === Phase 9: Global Workspace / Organ Protocol ===
        self._organs = []  # List of (organ, adapter) tuples
        self._gw = None
        self._write_gate = None
        self._diversity_tracker = None

        if c.use_global_workspace:
            from pyquifer.global_workspace import GlobalWorkspace, DiversityTracker
            from pyquifer.organ import OscillatoryWriteGate
            self._gw = GlobalWorkspace(
                content_dim=c.workspace_dim,
                workspace_dim=c.workspace_dim,
                n_slots=8,
                n_winners=c.gw_n_winners,
                context_dim=c.workspace_dim,
            )
            self._write_gate = OscillatoryWriteGate()
            self._diversity_tracker = DiversityTracker(
                pressure=c.gw_diversity_pressure,
            )

        # === Phase 10: Multi-Workspace Ensemble ===
        self._workspace_ensemble = None
        if c.use_workspace_ensemble:
            from pyquifer.global_workspace import WorkspaceEnsemble
            ws_dim = c.ensemble_workspace_dim if c.ensemble_workspace_dim > 0 else c.workspace_dim
            self._workspace_ensemble = WorkspaceEnsemble(
                n_workspaces=c.n_workspaces,
                content_dim=c.workspace_dim,
                workspace_dim=ws_dim,
                n_slots=8,
                n_winners=c.gw_n_winners,
                bleed_strength=c.cross_bleed_strength,
                standing_momentum=c.standing_momentum,
            )

        # === Phase 9b: Cross-module wiring modules ===
        self._circadian = None
        self._personality = None
        self._somatic = None

        if c.use_circadian:
            from pyquifer.ecology import ChronobiologicalSystem
            self._circadian = ChronobiologicalSystem(dim=c.state_dim)

        if c.use_personality_attractor:
            from pyquifer.strange_attractor import PersonalityAttractor
            self._personality = PersonalityAttractor(
                dim=min(c.state_dim, 16),
                num_attractors=c.num_populations,
            )

        if c.use_somatic:
            from pyquifer.somatic import SomaticManifold
            self._somatic = SomaticManifold(
                num_oscillators=c.num_oscillators,
                manifold_dim=min(c.internal_dim, 8),
            )

        # === Basal Ganglia Gating Loop ===
        self._gating_loop = None
        if c.use_gating_loop:
            from pyquifer.basal_ganglia import BasalGangliaLoop
            self._gating_loop = BasalGangliaLoop(
                max_channels=8,
                da_go_weight=c.gating_da_go_weight,
                da_nogo_weight=c.gating_da_nogo_weight,
                stn_surprise_threshold=c.gating_stn_surprise,
                switching_penalty=c.gating_switching_penalty,
            )

        # === Phase 11-18: New Module Wiring ===
        # Each flag gates a lazy import + instantiation.
        # Modules are called in tick() at the appropriate pipeline stage.

        # Phase 11: ODE solver (used by oscillators if not euler)
        self._ode_solver = None
        if c.solver != "euler":
            from pyquifer.ode_solvers import SolverConfig
            solver_map = {"rk4": "rk4", "dopri5": "dopri", "dopri": "dopri"}
            method = solver_map.get(c.solver, "rk4")
            self._ode_solver = SolverConfig(method=method)

        # Phase 11: Complex oscillator backend
        self._complex_oscillators = None
        if c.use_complex_oscillators:
            from pyquifer.complex_oscillators import ComplexKuramotoBank
            self._complex_oscillators = ComplexKuramotoBank(
                num_oscillators=c.num_oscillators,
                dt=c.oscillator_dt,
                integration_method=c.integration_method,
            )

        # Phase 11: CUDA kernel acceleration
        self._cuda_kernel = None
        if c.use_cuda_kernels:
            try:
                from pyquifer._cuda.kuramoto_kernel import KuramotoCUDAKernel
                self._cuda_kernel = KuramotoCUDAKernel(use_triton=True)
            except Exception as e:
                logger.warning("CUDA kernel init failed: %s — falling back to PyTorch", e)

        # Phase 12: Visual binding (AKOrN)
        self._visual_binding = None
        if c.use_visual_binding:
            from pyquifer.visual_binding import AKOrNEncoder
            self._visual_binding = AKOrNEncoder(
                dim=c.state_dim, depth=2, num_heads=4,
            )

        # Phase 12: Temporal binding (sequence AKOrN)
        self._temporal_binding = None
        if c.use_temporal_binding:
            from pyquifer.temporal_binding import SequenceAKOrN
            self._temporal_binding = SequenceAKOrN(
                dim=c.state_dim, num_heads=4,
            )

        # Phase 12: Sensory binding (cross-modal)
        self._sensory_binding = None
        if c.use_sensory_binding:
            from pyquifer.sensory_binding import MultimodalBinder
            self._sensory_binding = MultimodalBinder(
                modality_dims={"default": c.state_dim},
                binding_dim=c.state_dim,
                num_oscillators_per_modality=min(c.num_oscillators, 16),
            )

        # Phase-Lock Bus (multimodal coordinator)
        self._phase_lock_bus = None
        if c.use_phase_lock_bus:
            from pyquifer.phase_lock_bus import PhaseLockBus, BusConfig
            _mod_dims = c.phase_lock_modality_dims or {"text": c.state_dim}
            self._phase_lock_bus = PhaseLockBus(BusConfig(
                modality_dims=_mod_dims,
                binding_dim=c.state_dim,
                num_oscillators_per_modality=min(c.num_oscillators, 16),
                num_sync_steps=c.phase_lock_sync_steps,
                coherence_ema_alpha=c.phase_lock_coherence_ema,
            ))

        # Phase 13: Deep active inference
        self._deep_aif = None
        if c.use_deep_aif:
            from pyquifer.deep_active_inference import DeepAIF
            self._deep_aif = DeepAIF(
                obs_dim=c.state_dim,
                latent_dim=c.belief_dim,
                action_dim=4,  # Abstract action space
                horizon=5,
            )

        # Phase 13: JEPA world model
        self._jepa = None
        if c.use_jepa_world_model:
            from pyquifer.jepa import ActionJEPA
            self._jepa = ActionJEPA(
                obs_dim=c.state_dim,
                action_dim=4,
                latent_dim=c.belief_dim,
            )

        # Phase 13: Deliberation (test-time compute)
        self._deliberator = None
        if c.use_deliberation:
            from pyquifer.deliberation import Deliberator
            self._deliberator = Deliberator(
                dim=c.state_dim,
                min_steps=4,
                max_steps=16,
                beam_width=2,
            )

        # Phase 14: Complementary learning systems
        self._hippocampal = None
        self._neocortical = None
        if c.use_cls_memory:
            from pyquifer.cls_memory import HippocampalModule, NeocorticalModule
            self._hippocampal = HippocampalModule(
                dim=c.state_dim,
                capacity=c.episodic_capacity,
            )
            self._neocortical = NeocorticalModule(
                dim=c.state_dim,
                schema_dim=c.semantic_dim,
                num_schemas=c.semantic_slots,
            )

        # Phase 15: Causal reasoning
        self._causal_graph = None
        if c.use_causal_reasoning:
            from pyquifer.causal_reasoning import CausalGraph, DoOperator
            self._causal_graph = CausalGraph()
            self._do_operator = DoOperator(
                dim=c.state_dim,
                num_variables=len(c.hierarchy_dims),
            )

        # Phase 15: Cognitive appraisal
        self._appraisal = None
        if c.use_appraisal:
            from pyquifer.appraisal import AppraisalChain, OCC_Model
            self._appraisal = AppraisalChain(dim=c.state_dim)
            self._occ_model = OCC_Model()

        # Phase 16: Selective SSM (oscillatory)
        self._selective_ssm = None
        if c.use_selective_ssm:
            from pyquifer.selective_ssm import OscillatorySSM
            self._selective_ssm = OscillatorySSM(
                d_model=c.state_dim,
                d_state=16,
                num_oscillators=min(c.num_oscillators, 8),
            )

        # Phase 16: Oscillatory MoE
        self._oscillatory_moe = None
        if c.use_oscillatory_moe:
            from pyquifer.oscillatory_moe import SparseMoE
            self._oscillatory_moe = SparseMoE(
                d_model=c.state_dim,
                num_experts=4,
                top_k=2,
                num_oscillator_features=min(c.num_oscillators, 8),
            )

        # Phase 17: Prospective configuration learning
        self._prospective = None
        if c.use_prospective_learning:
            from pyquifer.prospective_config import InferThenModify
            self._prospective = InferThenModify(
                dims=c.hierarchy_dims,
                num_inference_steps=10,
                modification_lr=0.01,
            )

        # Fix 9/11: Cross-frequency coupling wiring
        # use_theta_gamma_pac also instantiates CFC for the gamma modulation path
        _want_cfc = c.use_cross_freq_coupling or c.use_meta_r_coupling or c.use_theta_gamma_pac
        if _want_cfc:
            from pyquifer.multiplexing import CrossFrequencyCoupling
            n_fast = c.num_oscillators - c.theta_oscillators if c.use_theta_gamma_pac else c.num_oscillators
            self._cfc = CrossFrequencyCoupling(
                num_fast_oscillators=n_fast,
                coupling_strength=0.5,
            )

        # === Projection layers (bridge mismatched dimensions) ===
        # Project oscillator phases to state_dim for various modules
        self.phase_to_state = nn.Linear(c.num_oscillators, c.state_dim, bias=False)
        # Project hierarchy errors to level activations for dominance detection
        self.error_to_levels = nn.Linear(c.hierarchy_dims[0], len(c.hierarchy_dims), bias=False)
        # Project state_dim to sensory_dim for Markov blanket
        if c.state_dim != c.sensory_dim:
            self.state_to_sensory = nn.Linear(c.state_dim, c.sensory_dim, bias=False)
        else:
            self.state_to_sensory = nn.Identity()
        # Project state to group_dim for neural darwinism
        if c.state_dim != c.group_dim:
            self.state_to_group = nn.Linear(c.state_dim, c.group_dim, bias=False)
        else:
            self.state_to_group = nn.Identity()

        # === Hierarchical Timestepping Caches ===
        # These store the last result from modules that don't run every tick.
        # Initialized to None; populated on first run of each module.
        self._cached_hpc: Optional[Dict] = None
        self._cached_sr: Optional[Dict] = None
        self._cached_motiv: Optional[Dict] = None
        self._cached_meta: Optional[Dict] = None
        self._cached_prec: Optional[Dict] = None
        self._cached_arena: Optional[Dict] = None
        self._cached_self: Optional[Dict] = None
        self._cached_narr: Optional[Dict] = None
        self._cached_blanket: Optional[Dict] = None

    def register_organ(self, organ, adapter=None):
        """
        Register a specialist organ for workspace competition.

        Args:
            organ: Organ instance (must implement observe/propose/accept)
            adapter: Optional PreGWAdapter. If None and workspace is enabled,
                    one will be created automatically.
        """
        if adapter is None and self.config.use_global_workspace:
            from pyquifer.organ import PreGWAdapter
            adapter = PreGWAdapter(
                organ_dim=organ.latent_dim,
                workspace_dim=self.config.workspace_dim,
            )
        self._organs.append((organ, adapter))

    def _run_workspace_competition(self, sensory_input: torch.Tensor,
                                    phases: torch.Tensor,
                                    neuro_levels: Optional[torch.Tensor] = None,
                                    coherence_val: float = 0.5,
                                    input_novelty: float = 0.0,
                                    ) -> Dict[str, Any]:
        """
        Run global workspace competition among registered organs.

        Returns dict with broadcast tensor and competition metadata.
        """
        if not self._organs or self._gw is None:
            return {}

        device = sensory_input.device
        global_phase = phases.mean()  # Use mean phase as global rhythm

        # 1. Each organ observes + proposes
        proposals = []
        for organ, adapter in self._organs:
            organ.step_oscillator(dt=self.config.oscillator_dt,
                                  global_phase=global_phase)
            organ.observe(sensory_input)
            proposal = organ.propose()

            # Apply oscillatory write gate
            gate_val = self._write_gate(
                organ.phase, global_phase,
                novelty=min(proposal.salience, 1.0),
            ).item()

            # Scale write gate by bus coherence (if bus enabled)
            if self._phase_lock_bus is not None:
                _bus_mod = 0.5 + 0.5 * self._phase_lock_bus.get_write_gate_modulation()
                gate_val = gate_val * _bus_mod

            # Apply diversity pressure
            boost = self._diversity_tracker.get_boost(organ.organ_id)
            proposal.salience = proposal.salience * gate_val + boost

            proposals.append((organ, adapter, proposal))

        # 1b. Basal ganglia gating (if enabled) — modulates salience priors
        bg_info = {}
        if self._gating_loop is not None and proposals:
            sal_t = torch.tensor(
                [p.salience for _o, _a, p in proposals],
                device=device, dtype=torch.float32,
            )
            cost_t = torch.tensor(
                [p.cost for _o, _a, p in proposals],
                device=device, dtype=torch.float32,
            )
            tags_list = [p.tags for _o, _a, p in proposals]
            _nl = neuro_levels if neuro_levels is not None else torch.tensor(
                [0.5, 0.5, 0.5, 0.5, 0.0], device=device,
            )
            gating_out = self._gating_loop.step(
                saliences=sal_t,
                costs=cost_t,
                tags_list=tags_list,
                neuro_levels=_nl,
                coherence=coherence_val,
                novelty=input_novelty,
            )
            # Apply salience priors from BG gating
            for i, (_organ, _adapter, proposal) in enumerate(proposals):
                if i < len(gating_out.salience_prior):
                    proposal.salience *= (1.0 + float(gating_out.salience_prior[i]))
            bg_info = {
                'bg_selected_channel': gating_out.selected_channel,
                'bg_thalamic_gate': gating_out.thalamic_gate,
                'bg_stn_active': gating_out.stn_active,
                'bg_da_bias': gating_out.da_bias,
                'bg_switching_cost': gating_out.switching_cost,
                'bg_mode_bias': gating_out.processing_mode_bias,
                'bg_channel_activations': gating_out.channel_activations,
            }

        # 2. Project to workspace dim and run competition
        ws_dim = self.config.workspace_dim
        n_items = len(proposals)
        contents = torch.zeros(1, n_items, ws_dim, device=device)
        saliences = torch.zeros(1, n_items, device=device)

        for i, (organ, adapter, proposal) in enumerate(proposals):
            if adapter is not None:
                projected = adapter.encode(proposal.content.unsqueeze(0))
                contents[0, i] = projected.squeeze(0)
            else:
                # Pad/trim to workspace_dim
                src = proposal.content
                dim = min(src.shape[-1], ws_dim)
                contents[0, i, :dim] = src[:dim]
            saliences[0, i] = proposal.salience

        # Use contents as both content and context
        gw_result = self._gw(contents, contents)

        # 3. Broadcast winner back to all organs
        broadcast = gw_result['workspace'].squeeze(0)  # (workspace_dim,)
        winner_idx = gw_result['winners'][0].argmax().item()
        winner_id = proposals[winner_idx][2].organ_id
        self._diversity_tracker.record_win(winner_id)

        for organ, adapter, proposal in proposals:
            if adapter is not None:
                organ_broadcast = adapter.decode(broadcast)
            else:
                organ_broadcast = broadcast[:organ.latent_dim]
            organ.accept(organ_broadcast)

        # 4. Cycle consistency loss
        cc_loss = torch.tensor(0.0, device=device)
        if self.config.gw_cycle_consistency:
            for organ, adapter, proposal in proposals:
                if adapter is not None:
                    cc_loss = cc_loss + adapter.cycle_consistency_loss(proposal.content)

        # 5. Collect organ standings for bridge exposure
        organ_standings = {}
        for organ, _adapter in self._organs:
            organ_standings[organ.organ_id] = organ.standing_latent.clone().detach()

        return {
            'gw_broadcast': broadcast.detach(),
            'gw_winner': winner_id,
            'gw_saliences': saliences.detach().squeeze(0),
            'gw_did_ignite': gw_result['did_ignite'].any().item(),
            'gw_cycle_consistency_loss': cc_loss,
            'organ_standings': organ_standings,
            **bg_info,
        }

    def tick(self,
             sensory_input: torch.Tensor,
             reward: float = 0.0,
             sleep_signal: float = 0.0,
             return_diagnostics: bool = True,
             ) -> Dict[str, Any]:
        """
        Run one cognitive tick.

        Args:
            sensory_input: Raw sensory input, shape ``(state_dim,)`` or
                           ``(1, state_dim)``.  Batch dim >1 is rejected.
                           A ``(1, state_dim)`` input is squeezed automatically.
            reward: External reward signal (for motivation + memory priority)
            sleep_signal: 0.0 = awake, 1.0 = deep sleep (gates consolidation)
            return_diagnostics: If False, skip building the heavy diagnostics
                dict and most .item() conversions.  Returns only the modulation
                parameters needed by the LLM bridge.  ~2x faster on GPU.

        Returns:
            Dict with:
            - modulation: Dict of LLM modulation parameters
            - consciousness: Dict of consciousness metrics (if return_diagnostics)
            - self_state: Dict of self-model state (if return_diagnostics)
            - diagnostics: Dict of internal metrics (if return_diagnostics)
        """
        c = self.config
        # Enforce single-sample input — batched tick is not supported.
        if sensory_input.dim() == 2:
            if sensory_input.shape[0] != 1:
                raise ValueError(
                    f"tick() does not support batched input. "
                    f"Expected (state_dim,) or (1, state_dim), got {sensory_input.shape}"
                )
            sensory_input = sensory_input.squeeze(0)
        elif sensory_input.dim() != 1:
            raise ValueError(
                f"tick() expects 1-D input (state_dim,), got shape {sensory_input.shape}"
            )

        device = sensory_input.device

        # ── Step 0: Sensory-oscillator coupling ──
        # Input drives oscillators via frequency entrainment, phase resets,
        # and coupling modulation (Lakatos et al. 2008; Fries 2005 CTC).
        # All modifications are within torch.no_grad — gradient flow from
        # the LLM to oscillators is severed by design.
        sc = self._sensory_coupling(sensory_input, self.oscillators.phases)

        with torch.no_grad():
            # Phase reset: novel stimuli partially reset phases toward
            # input-derived targets (event-related desynchronization).
            # Uses clamped interpolation weight instead of branching on
            # a tensor value — avoids GPU sync and is torch.compile safe.
            reset_s = sc['reset_strength']
            # Threshold: below 0.05 → weight=0, above → actual reset_s
            reset_w = (reset_s - 0.05).clamp(min=0.0) / max(1.0 - 0.05, 1e-8) * (1.0 - 0.05) + 0.05
            reset_w = reset_s * (reset_s > 0.05).float()  # simpler: zero below threshold
            blended = (
                (1.0 - reset_w) * self.oscillators.phases
                + reset_w * sc['phase_targets']
            ) % (2 * math.pi)
            self.oscillators.phases.copy_(blended)

            # Coupling modulation: input salience scales K
            self.oscillators.coupling_strength.mul_(
                sc['coupling_scale']
            ).clamp_(min=0.1, max=5.0)

        # ── Step 1: Oscillator dynamics ──
        # Frequency entrainment: input energy shifts natural frequencies
        phases = self.oscillators(
            external_input=sc['freq_modulation'],
            steps=1,
            use_precision=True,
        )
        order_param = self.oscillators.get_order_parameter()
        # Keep as tensor — .item() deferred to batch extraction block at end
        # get_order_parameter() always returns a tensor
        coherence = order_param

        # ── Step 1a2: Complex oscillator backend (parallel to real-valued) ──
        complex_osc_info = {}
        if self._complex_oscillators is not None:
            complex_phases = self._complex_oscillators(
                external_input=sc['freq_modulation'][:self._complex_oscillators.num_oscillators]
                if sc['freq_modulation'].shape[0] >= self._complex_oscillators.num_oscillators
                else None,
                steps=1,
            )
            complex_R = self._complex_oscillators.get_order_parameter()
            complex_osc_info = {
                'complex_R': complex_R.detach(),
                'complex_phases': self._complex_oscillators.get_phases().detach(),
            }

        # ── Step 1b: Optional Stuart-Landau oscillator ──
        stuart_landau_info = {}
        _sl_result = None
        if self._stuart_landau is not None:
            _sl_result = self._stuart_landau(steps=1)

        # ── Step 1c: Optional Kuramoto-Daido mean-field ──
        mean_field_info = {}
        _mf_result = None
        if self._mean_field is not None:
            _mf_result = self._mean_field(steps=1)

        # ── Step 1d: Circadian rhythm (modulates sleep/plasticity) ──
        circadian_info = {}
        circadian_plasticity = 1.0
        if self._circadian is not None:
            circ_result = self._circadian.step()
            circadian_plasticity = circ_result['plasticity']
            circadian_mode = circ_result.get('mode', 'wake')
            # Circadian rhythm modulates effective sleep signal
            if circadian_mode == 'sleep' and sleep_signal < 0.3:
                sleep_signal = max(sleep_signal, 0.4)  # Gentle circadian sleep push
            circadian_info = {
                'circadian_phase': circ_result['circadian_phase'],
                'circadian_plasticity': circadian_plasticity,
                'circadian_mode': circadian_mode,
                'circadian_temperature': circ_result['temperature'],
                'circadian_hour': circ_result['hour'],
            }

        # ── Step 2: Criticality check ──
        # Use Kuramoto-specific criticality monitor based on order parameter R
        # This is the correct metric for continuous oscillator networks:
        # susceptibility χ = N·var(R) peaks at the critical coupling K_c
        # Reuse cached R from forward() — avoids redundant sin/cos computation
        R_current = coherence
        crit = self._kuramoto_criticality(
            R_current, num_oscillators=self.oscillators.num_oscillators,
        )
        # Keep as tensor — .item() deferred to batch extraction block
        criticality_distance = crit['criticality_distance']

        # Also feed spike-based activity to the PI controller (for avalanche tracking)
        spike_count = self._phase_to_spikes(phases)
        crit_pi = self.criticality(spike_count)

        # Close the criticality feedback loop using dual-timescale control,
        # mirroring how real cortex maintains criticality:
        #
        # SLOW PATH (homeostatic plasticity): Adjust coupling K based on
        # the windowed sigma (~50 ticks). Like synaptic homeostasis that
        # adapts excitatory/inhibitory balance over minutes/hours.
        #
        # FAST PATH (GABAergic inhibition): Inject dephasing noise based
        # on INSTANTANEOUS R, not the lagged sigma. Like fast inhibitory
        # interneurons that prevent epileptic hypersynchrony within ms.
        #
        # This dual-timescale approach prevents the limit-cycle oscillation
        # that occurs when both paths use the lagged sigma: by the time
        # sigma reports "supercritical", R is already at 0.9 and deeply
        # phase-locked. The fast path catches this before it happens.
        # (Poil et al. 2012 J Neurosci — homeostatic criticality;
        #  Brunel & Wang 2003 J Comp Neurosci — E/I balance)
        # Keep as tensor — .item() deferred to batch extraction block
        crit_sigma = crit['branching_ratio']

        with torch.no_grad():
            # ── Fix 6: Update R history and compute metastability ──
            # Push R_current into circular buffer
            idx = self._R_history_ptr % 50
            self._R_history[idx] = R_current.detach()
            self._R_history_ptr += 1

            # Update long R history for neuroscience diagnostics
            _buf_len = self.config.diagnostics_buffer_len
            long_idx = self._R_long_ptr % _buf_len
            self._R_long_history[long_idx] = R_current.detach()
            self._R_long_ptr += 1

            # Update oscillator activity buffer for avalanche detection
            # v7: spike-based activity (Beggs & Plenz 2003) when enabled,
            # else sum of |sin(phase_i)| as LFP analog.
            if self.config.use_spike_activity:
                from pyquifer.criticality import phase_activity_to_spikes
                activity = phase_activity_to_spikes(self.oscillators.phases.detach())
            else:
                activity = self.oscillators.phases.detach().sin().abs().sum()
            act_idx = self._activity_long_ptr % _buf_len
            self._activity_long_history[act_idx] = activity
            self._activity_long_ptr += 1

            # Update R EMA for Fix 9 (CFC theta derivation)
            self._R_ema.mul_(0.95).add_(0.05 * R_current.detach())

            # Compute observed metastability = SD(R) over filled portion
            n_filled = min(self._R_history_ptr, 50)
            if n_filled >= 5:
                R_slice = self._R_history[:n_filled]
                observed_meta = R_slice.std()
                self._cached_metastability.copy_(observed_meta)

                # Metastability-targeting dephasing gain:
                # When SD(R) < target → increase dephasing (more variability needed)
                # When SD(R) > target → decrease dephasing (too chaotic)
                meta_error = self._effective_target_meta - observed_meta
                dephasing_gain = (1.0 + meta_error * 8.0).clamp(0.5, 3.0)
                self._cached_dephasing_gain.copy_(dephasing_gain)

            # Criticality feedback: dual-timescale control extracted into a
            # standalone function so torch.compile can fuse all ~15 element-wise
            # ops into 1-2 kernels.  See _criticality_feedback() docstring.
            # v7: Generate colored noise from OU cascade if enabled
            _colored = None
            if self._ou_cascade is not None:
                _colored = sum(ou.forward() for ou in self._ou_cascade)
                _colored = _colored.to(self.oscillators.phases.device)

            _crit_fn = _compiled_criticality_feedback or _criticality_feedback
            _crit_fn(
                self.oscillators.phases,
                self.oscillators.coupling_strength,
                crit_sigma,
                R_current,
                dephasing_gain=self._cached_dephasing_gain,
                R_target=self._R_target,
                colored_noise=_colored,
            )

        # ── Step 2b: Optional Koopman bifurcation detection ──
        koopman_info = {}
        _koopman_result = None
        if self._koopman is not None:
            _koopman_result = self._koopman(sensory_input.detach())

        # ── Step 3: Neuromodulation ──
        # Update neuromodulator levels based on current signals
        # reward/novelty/threat/success drive DA/5-HT/NE/ACh/Cortisol
        neuro_state = self.neuromodulation.step(
            reward_signal=min(1.0, max(-1.0, reward)),
            novelty_signal=min(1.0, max(0.0, abs(reward) * 0.5)),
            threat_signal=0.0,
            success_signal=min(1.0, max(0.0, reward)),
        )

        # Extract neuromodulator levels: [DA, 5HT, NE, ACh, Cortisol]
        # Keep as 0-dim tensors — .item() deferred to batch extraction block
        levels = self.neuromodulation.levels
        ach = levels[3]  # acetylcholine
        ne = levels[2]   # norepinephrine

        # ── Step 4: Stochastic resonance ──
        if self._tick_py % c.step_every_sr == 0 or self._cached_sr is None:
            sr_result = self.stochastic_resonance(
                sensory_input,
                criticality_distance=criticality_distance,
            )
            self._cached_sr = sr_result
        else:
            sr_result = self._cached_sr
        enhanced_input = sr_result['enhanced'].unsqueeze(0)

        # ── Step 4b: Optional short-term plasticity ──
        stp_info = {}
        if self._stp is not None:
            # Use enhanced_input magnitude as spike proxy
            spike_proxy = (enhanced_input.squeeze(0).abs() > 0.5).float()
            stp_result = self._stp(spike_proxy)
            stp_info = {
                'stp_psp': stp_result['psp'].detach(),
                'mean_facilitation': stp_result['u'].mean(),
                'mean_depression': stp_result['x'].mean(),
            }
            # Modulate enhanced input by STP output
            enhanced_input = enhanced_input * stp_result['psp'].unsqueeze(0)

        # ── Step 4c: Optional oscillation-gated plasticity ──
        theta_gate_value = 0.0
        if self._oscillation_gated is not None:
            # Use first oscillator phase as theta proxy
            theta_phase = phases[0]
            activity = (enhanced_input.squeeze(0) if enhanced_input.dim() > 1
                        else enhanced_input)
            # Only use first hierarchy_dims[0] elements
            activity_for_gate = activity[:c.hierarchy_dims[0]]
            self._oscillation_gated(activity_for_gate, theta_phase)
            theta_gate_value = self._oscillation_gated.gate_value

        # ── Step 4d: Phase 12 — Perceptual binding (if enabled) ──
        binding_info = {}
        if self._visual_binding is not None:
            # AKOrN needs (B, N, D) — treat input as single-token sequence
            vb_input = enhanced_input.view(1, 1, -1)
            vb_out = self._visual_binding(vb_input)
            # Squeeze back to 1D (D,) for tick() pipeline
            enhanced_input = vb_out.view(-1)
            binding_info['visual_binding_applied'] = True

        if self._temporal_binding is not None:
            tb_input = enhanced_input.view(1, 1, -1)
            tb_out = self._temporal_binding(tb_input)
            enhanced_input = tb_out.view(-1)
            binding_info['temporal_binding_applied'] = True

        if self._sensory_binding is not None:
            sb_input = enhanced_input.squeeze(0) if enhanced_input.dim() > 1 else enhanced_input
            sb_result = self._sensory_binding({'default': sb_input.unsqueeze(0)})
            binding_info['total_binding'] = sb_result['total_binding'].detach()

        # ── Step 4d2: Phase-Lock Bus (multimodal coordinator) ──
        bus_info = {}
        if self._phase_lock_bus is not None:
            bus_input = enhanced_input.squeeze(0) if enhanced_input.dim() > 1 else enhanced_input
            bus_modalities = {"text": bus_input.unsqueeze(0)}
            bus_out = self._phase_lock_bus(bus_modalities)
            enhanced_input = bus_out.fused_representation.squeeze(0)
            bus_info = {
                'bus_binding_matrix': bus_out.binding_matrix.detach(),
                'bus_mean_coherence': bus_out.mean_coherence.detach(),
                'bus_modality_count': bus_out.modality_count,
            }

        # ── Step 4e: Phase 16 — Selective SSM / MoE (if enabled) ──
        ssm_moe_info = {}
        if self._selective_ssm is not None:
            ssm_input = enhanced_input.view(1, 1, -1)
            # Trim phases to SSM's num_oscillators
            ssm_phases = phases[:self._selective_ssm.num_oscillators]
            ssm_out = self._selective_ssm(ssm_input, phases=ssm_phases)
            enhanced_input = ssm_out.view(-1)
            ssm_moe_info['ssm_applied'] = True

        if self._oscillatory_moe is not None:
            moe_input = enhanced_input.view(1, 1, -1)
            n_osc_feat = self._oscillatory_moe.router.osc_gate[0].in_features
            osc_state = torch.sin(phases[:n_osc_feat])
            moe_result = self._oscillatory_moe(moe_input, oscillator_state=osc_state)
            enhanced_input = moe_result['output'].view(-1)
            ssm_moe_info['moe_balance_loss'] = moe_result['balance_loss'].detach()

        # ── Step 5: Hierarchical Predictive Coding ──
        # Trim or pad input to match hierarchy bottom dimension
        bottom_dim = c.hierarchy_dims[0]
        if enhanced_input.shape[-1] != bottom_dim:
            if enhanced_input.shape[-1] > bottom_dim:
                hpc_input = enhanced_input[..., :bottom_dim]
            else:
                pad = torch.zeros(*enhanced_input.shape[:-1], bottom_dim - enhanced_input.shape[-1],
                                  device=device)
                hpc_input = torch.cat([enhanced_input, pad], dim=-1)
        else:
            hpc_input = enhanced_input

        # Use oscillatory predictive coding if enabled, otherwise standard HPC
        opc_info = {}
        if self._oscillatory_predictive is not None:
            opc_result = self._oscillatory_predictive.learn(hpc_input, slow_phase=phases[0:1])
            opc_info = {
                'gamma_power': opc_result['gamma_power'],
                'alpha_beta_power': opc_result['alpha_beta_power'],
            }

        if self._tick_py % c.step_every_hpc == 0 or self._cached_hpc is None:
            hpc_result = self.hpc(hpc_input)
            self._cached_hpc = hpc_result
        else:
            hpc_result = self._cached_hpc
        prediction_error = hpc_result['errors'][0]  # Bottom-level errors
        free_energy = hpc_result['free_energy']
        top_beliefs = hpc_result['top_level_beliefs']

        # ── Step 5a: Surprise-driven phase perturbation ──
        # When free energy exceeds the system's running mean + 1σ (surprise),
        # inject phase noise proportional to the z-score. This creates the
        # "perturbation → recovery" pattern observed in TMS studies (Casali 2013).
        # Threshold is fully adaptive: derived from the system's own FE statistics,
        # so "surprise" is relative to recent experience.
        # NOTE: Applied AFTER criticality feedback (step 2) so the homeostatic
        # controller doesn't immediately undo the desynchronization.
        fe_val = float(free_energy) if isinstance(free_energy, torch.Tensor) else free_energy
        if not hasattr(self, '_fe_ema'):
            self._fe_ema = fe_val
            self._fe_var_ema = 0.001
        else:
            fe_alpha = 0.05
            self._fe_var_ema = (1 - fe_alpha) * self._fe_var_ema + fe_alpha * (fe_val - self._fe_ema) ** 2
            self._fe_ema = (1 - fe_alpha) * self._fe_ema + fe_alpha * fe_val

        fe_sigma = math.sqrt(max(1e-6, self._fe_var_ema))
        surprise_threshold = self._fe_ema + fe_sigma
        if fe_val > surprise_threshold and self._tick_py > 10:
            z_score = (fe_val - self._fe_ema) / fe_sigma
            # Noise scales with z-score: 1σ→0.2rad, 2σ→0.4rad, 3σ+→0.6rad cap
            noise_magnitude = min(0.6, z_score * 0.2)
            with torch.no_grad():
                phase_noise = torch.randn_like(self.oscillators.phases) * noise_magnitude
                self.oscillators.phases.add_(phase_noise)
                self.oscillators.phases.remainder_(2 * math.pi)
                # Transiently reduce coupling so desynchronization persists
                # for a few ticks before natural re-entrainment.
                coupling_damping = max(0.7, 1.0 - z_score * 0.1)
                self.oscillators.coupling_strength.mul_(coupling_damping)

        # ── Step 5b: Volatility filter (adaptive learning rate) ──
        # Track prediction errors to determine environmental volatility
        pe_for_vol = prediction_error.detach()
        if pe_for_vol.dim() > 1:
            pe_for_vol = pe_for_vol.squeeze(0)
        vol_result = self.volatility_gate(pe_for_vol)
        adaptive_lr = vol_result['effective_lr']  # Per-dimension adaptive LR

        # ── Step 6: Precision weighting ──
        if self._tick_py % c.step_every_precision == 0 or self._cached_prec is None:
            prec_result = self.precision(
                hpc_input,
                prediction_error,
                acetylcholine=ach,
                norepinephrine=ne,
            )
            self._cached_prec = prec_result
        else:
            prec_result = self._cached_prec
        attention_map = prec_result['attention_map']

        # ── Step 7: Causal flow (dominance detection) ──
        # Use HPC error norms at each level as level activations
        level_activations = torch.stack(
            [err.norm() for err in hpc_result['errors']]
        ).to(device)

        # Phase → Causal Flow wiring: modulate level activations with oscillator
        # phase coherence per frequency band (subgroups of oscillators)
        if c.use_phase_dominance:
            n_osc = c.num_oscillators
            n_levels = len(c.hierarchy_dims)
            band_size = max(1, n_osc // n_levels)
            for lvl in range(n_levels):
                start = lvl * band_size
                end = min(start + band_size, n_osc)
                if end > start:
                    band_phases = phases[start:end]
                    # Phase coherence in this band (order parameter)
                    band_r = torch.abs(torch.exp(1j * band_phases.to(torch.complex64)).mean())
                    # Modulate: high band coherence amplifies that level's activation
                    level_activations[lvl] = level_activations[lvl] * (0.5 + band_r)

        # Fix 7: pass criticality_distance for critical slowing down
        _crit_dist_f = float(criticality_distance) if isinstance(criticality_distance, torch.Tensor) else criticality_distance
        dom_result = self.dominance(level_activations, compute_every=5, criticality_distance=_crit_dist_f)
        dominance_ratio = dom_result['dominance_ratio']  # keep tensor
        processing_mode = dom_result['mode']

        # Coherence-gated override: extreme coherence states override dominance mode.
        # Very high R → locked system processing input → perception
        # Very low R → chaotic system generating internally → imagination
        # Thresholds are functional: derived from the homeostatic target (σ=1.0
        # targets R≈0.5), so extremes are genuinely outside normal dynamics.
        R = coherence  # already a tensor from oscillators
        R_val = float(R) if isinstance(R, torch.Tensor) else R
        if R_val > 0.85:
            processing_mode = 'perception'
        elif R_val < 0.2:
            processing_mode = 'imagination'

        # ── Step 8: Metastability (stream of consciousness) ──
        if self._tick_py % c.step_every_metastability == 0 or self._cached_meta is None:
            meta_result = self.metastability()
            self._cached_meta = meta_result
        else:
            meta_result = self._cached_meta
        # Keep as tensors — extract to int/float in batch block
        dominant_state_t = meta_result['dominant']
        coalition_entropy = meta_result['coalition_entropy']

        # ── Step 8a: Personality attractor (if enabled) ──
        personality_info = {}
        if self._personality is not None:
            # Drive attractor with oscillator phase state (projected to attractor dim)
            phase_state = torch.sin(phases[:min(len(phases), self._personality.dim)])
            pers_state = self._personality.step(external_input=phase_state)
            personality_info = {
                'personality_state': pers_state.detach(),
                'lyapunov_exponent': self._personality.get_lyapunov_exponent(),
            }

        # ── Step 8b: Global Workspace competition (if enabled) ──
        gw_info = {}
        if self.config.use_global_workspace and self._organs:
            _nov_f = float(sc['input_novelty']) if isinstance(sc['input_novelty'], torch.Tensor) else sc['input_novelty']
            R_val_for_gw = float(coherence) if isinstance(coherence, torch.Tensor) else coherence
            gw_info = self._run_workspace_competition(
                sensory_input, phases,
                neuro_levels=levels,
                coherence_val=R_val_for_gw,
                input_novelty=_nov_f,
            )
            # Cache for bridge minimal path
            self._cached_gw_winner_id = gw_info.get('gw_winner', '')
            self._cached_gw_broadcast = gw_info.get('gw_broadcast')

            # BG mode bias can override DominanceDetector
            if gw_info.get('bg_stn_active', False):
                processing_mode = 'balanced'  # Emergency: neutral mode
            elif 'bg_mode_bias' in gw_info:
                _bg_mode = gw_info['bg_mode_bias']
                if _bg_mode == 0:
                    processing_mode = 'perception'
                elif _bg_mode == 1:
                    processing_mode = 'imagination'
                # 2 = balanced, leave as-is (DominanceDetector's choice)

        # ── Step 8c: Multi-Workspace Ensemble (if enabled) ──
        # Scale cross-bleed by bus coherence
        if self._phase_lock_bus is not None and self._workspace_ensemble is not None:
            _bleed_mod = self._phase_lock_bus.get_bleed_modulation()
            self._workspace_ensemble.bleed_strength = (
                self.config.cross_bleed_strength * (0.3 + 0.7 * _bleed_mod)
            )

        ensemble_info = {}
        if self._workspace_ensemble is not None and self.config.use_global_workspace:
            n_ws = self.config.n_workspaces
            ws_dim = self.config.workspace_dim
            # Build per-workspace contents from organ proposals + standing latents
            # Active workspace (idx 0) gets fresh proposals, background gets standings
            contents_per_ws = []
            contexts_per_ws = []
            for wi in range(n_ws):
                if wi == int(self._workspace_ensemble.active_idx) and gw_info:
                    # Active workspace: use the same proposals from GW competition
                    # Repackage gw_info broadcast as content
                    bcast = gw_info.get('gw_broadcast', torch.zeros(ws_dim, device=device))
                    content = bcast.unsqueeze(0).unsqueeze(0).expand(1, 1, -1)
                    # Use as its own context too
                    ctx = content.clone()
                else:
                    # Background workspace: use organ standing latents
                    standings = []
                    for organ, adapter in self._organs:
                        sl = organ.standing_latent
                        if adapter is not None:
                            sl = adapter.encode(sl.unsqueeze(0)).squeeze(0)
                        else:
                            if sl.shape[-1] < ws_dim:
                                sl = torch.nn.functional.pad(sl, (0, ws_dim - sl.shape[-1]))
                            elif sl.shape[-1] > ws_dim:
                                sl = sl[:ws_dim]
                        standings.append(sl)
                    if standings:
                        content = torch.stack(standings).unsqueeze(0)  # [1, n_organs, ws_dim]
                    else:
                        content = torch.zeros(1, 1, ws_dim, device=device)
                    ctx = content.clone()
                contents_per_ws.append(content)
                contexts_per_ws.append(ctx)

            # Build phase lists (split oscillator phases across workspaces)
            n_osc = c.num_oscillators
            osc_per_ws = max(1, n_osc // n_ws)
            phases_per_ws = []
            for wi in range(n_ws):
                start = wi * osc_per_ws
                end = min(start + osc_per_ws, n_osc)
                phases_per_ws.append(phases[start:end])

            ens_result = self._workspace_ensemble(
                contents_per_ws, contexts_per_ws, phases_per_ws
            )
            ensemble_info = {
                'ensemble_bleed_matrix': ens_result['bleed_matrix'].detach(),
                'ensemble_active_idx': ens_result['active_idx'],
                'ensemble_n_workspaces': n_ws,
            }

        # ── Step 8b: Optional Wilson-Cowan neural mass ──
        neural_mass_info = {}
        if self._neural_mass is not None:
            # Drive neural mass with oscillator coherence
            ext_input = torch.ones(c.num_populations, device=device) * coherence
            nm_result = self._neural_mass(steps=1, external_input=ext_input)
            neural_mass_info = {
                'E_states': nm_result['E_states'].detach(),
                'I_states': nm_result['I_states'].detach(),
                'nm_synchronization': nm_result['synchronization'],
                'nm_mean_E': nm_result['mean_E'],
            }

        # ── Step 9: Neural darwinism ──
        group_input = self.state_to_group(sensory_input)
        # Global coherence from oscillator order parameter
        global_coherence_signal = self.state_to_group(
            self.phase_to_state(torch.sin(phases).unsqueeze(0)).squeeze(0)
        )

        # GW → Arena wiring: blend workspace broadcast into coherence signal
        if gw_info and 'gw_broadcast' in gw_info:
            # Project workspace broadcast to group_dim and add to coherence
            gw_proj = gw_info['gw_broadcast'][:c.group_dim]
            if gw_proj.shape[0] < c.group_dim:
                gw_proj = torch.nn.functional.pad(gw_proj, (0, c.group_dim - gw_proj.shape[0]))
            # Weight by ignition strength (0.2 blend factor)
            global_coherence_signal = global_coherence_signal + 0.2 * gw_proj

        if self._tick_py % c.step_every_arena == 0 or self._cached_arena is None:
            arena_result = self.arena(group_input, global_coherence=global_coherence_signal)
            self.symbiogenesis(arena_result['group_outputs'])
            self._cached_arena = arena_result
        else:
            arena_result = self._cached_arena

        # ── Step 10: Motivation ──
        if self._tick_py % c.step_every_motivation == 0 or self._cached_motiv is None:
            motiv_result = self.motivation(
                sensory_input,
                order_parameter=coherence.detach(),
            )
            self._cached_motiv = motiv_result
        else:
            motiv_result = self._cached_motiv
        combined_motivation = motiv_result['motivation']  # keep tensor

        # ── Step 10b: Phase 15 — Cognitive appraisal (if enabled) ──
        appraisal_info = {}
        if self._appraisal is not None:
            appr_result = self._appraisal(sensory_input)
            occ_input = torch.stack([
                v['value'] for v in appr_result.values()
            ]).unsqueeze(0)  # (1, num_dims)
            occ_result = self._occ_model(occ_input)
            appraisal_info = {
                'dominant_emotion': occ_result['dominant_emotion'].detach(),
                'valence': occ_result['valence'].detach(),
                'arousal': occ_result['arousal'].detach(),
            }

        # ── Step 10c: Phase 13 — Deliberation (if enabled) ──
        deliberation_info = {}
        if self._deliberator is not None:
            coherence_f = float(coherence) if isinstance(coherence, torch.Tensor) else coherence
            delib_result = self._deliberator(
                sensory_input.unsqueeze(0),
                coherence=coherence_f,
            )
            deliberation_info = {
                'deliberation_confidence': delib_result['confidence'].detach(),
                'deliberation_corrections': delib_result['corrections_made'].detach(),
            }

        # ── Step 11: Self-model ──
        somatic_info = {}
        if self._tick_py % c.step_every_selfmodel == 0 or self._cached_self is None:
            sensory_for_blanket = self.state_to_sensory(sensory_input)
            blanket = self.markov_blanket(sensory_for_blanket)

            # Somatic → Self-Model wiring: enrich internal state with body signals
            internal_state = blanket['internal_state']
            if self._somatic is not None:
                som_result = self._somatic.forward()
                som_state = som_result['state']  # SomaticState dataclass
                stress_val = som_state.total_stress()
                coupling_mod = som_result.get('coupling_modulation', None)
                somatic_info = {
                    'somatic_stress': stress_val,
                    'somatic_pain': som_state.pain,
                    'somatic_fatigue': som_state.fatigue,
                    'should_repair': som_result.get('should_repair', False),
                }
                # Modulate internal state: high stress → dampen self-model updates
                stress_gate = max(0.3, 1.0 - stress_val * 0.5)  # [0.3, 1.0]
                internal_state = internal_state * stress_gate
                # Somatic coupling modulation → oscillator coupling
                if coupling_mod is not None and coupling_mod.shape[0] == c.num_oscillators:
                    with torch.no_grad():
                        self.oscillators.coupling_strength.mul_(0.9).add_(0.1 * coupling_mod.mean())

            self_result = self.self_model(internal_state)
            narr_result = self.narrative(self_result['self_summary'])
            self._cached_self = self_result
            self._cached_narr = narr_result
            self._cached_blanket = blanket
        else:
            self_result = self._cached_self
            narr_result = self._cached_narr
            blanket = self._cached_blanket

        # ── Step 11b: Three-factor learning ──
        three_factor_info = {}
        if self._three_factor is not None:
            # Use bottom-level HPC input as pre-synaptic activity
            pre_activity = hpc_input.squeeze(0) if hpc_input.dim() > 1 else hpc_input
            self._three_factor(pre_activity)
            # Modulate with dopamine level (first neuromodulator)
            da_level = levels[0]
            self._three_factor.modulated_update(da_level)
            three_factor_info = {
                'homeostatic_factor': self._three_factor.homeostatic_factor.mean(),
            }

        # ── Step 11c: Dendritic credit assignment ──
        dendritic_info = {}
        if self._dendritic_stack is not None:
            # Run dendritic stack without top-down predictions (avoids dim mismatch)
            # The dendritic layers learn their own top-down signals over time
            stack_input = hpc_input.squeeze(0) if hpc_input.dim() > 1 else hpc_input
            stack_result = self._dendritic_stack(stack_input)
            learn_result = self._dendritic_stack.learn()
            dendritic_info = {
                'dendritic_mean_delta': learn_result['mean_delta_norm'],
            }

        # ── Step 12: Memory (consolidation if sleeping) ──
        # Store experience
        self.episodic_buffer.store(
            sensory_input.detach(),
            reward=reward + combined_motivation,
        )

        consolidation_info = {'consolidated': False, 'num_traces': 0}
        src_info = {}
        if sleep_signal > 0.3:
            replay = self.sharp_wave_ripple(self.episodic_buffer, sleep_signal=sleep_signal)
            if replay['replayed_states'].shape[0] > 0:
                replay_rewards = replay['replayed_rewards']

                # Motivation → Memory Priority wiring: high-motivation memories
                # consolidate with amplified reward signal
                if c.use_motivation_priority:
                    # Scale rewards: motivation boosts consolidation priority.
                    # Clamp to >=1 so negative motivation doesn't shrink rewards.
                    priority_scale = 1.0 + combined_motivation.clamp(min=0.0) * 0.5
                    replay_rewards = replay_rewards * priority_scale

                # Circadian → Consolidation wiring: plasticity modulates consolidation
                if self._circadian is not None:
                    replay_rewards = replay_rewards * circadian_plasticity

                cons = self.consolidation(
                    replay['replayed_states'],
                    replay_rewards,
                )
                consolidation_info = {
                    'consolidated': cons['consolidated'],
                    'num_traces': cons['num_traces'],
                }

            # SRC Hebbian sleep consolidation
            if self._sleep_consolidation is not None:
                src_result = self._sleep_consolidation.sleep_step()
                src_delta_norm = sum(src_result['weight_delta_norms']) / len(src_result['weight_delta_norms'])
                src_info = {
                    'src_delta_norm': src_delta_norm,
                }
                # Consolidation → Identity boost: meaningful sleep consolidation
                # strengthens narrative identity (memories solidify self-concept).
                # Bonus proportional to consolidation magnitude, capped to prevent jumps.
                if src_delta_norm > 0.01:
                    consolidation_bonus = min(0.005, src_delta_norm * 0.01)
                    with torch.no_grad():
                        self.narrative.identity_strength.add_(consolidation_bonus)
                        self.narrative.identity_strength.clamp_(max=1.0)

        # ── Step 12b: Phase 14 — CLS memory (if enabled) ──
        cls_info = {}
        if self._hippocampal is not None:
            # Store in hippocampus (fast episodic)
            importance = abs(reward) + float(combined_motivation)
            self._hippocampal.store(
                sensory_input.detach(),
                importance=min(1.0, importance),
                novelty=float(sc['input_novelty']) if isinstance(sc['input_novelty'], torch.Tensor) else sc['input_novelty'],
            )
            cls_info['hippocampal_count'] = self._hippocampal.num_stored.item()

            # Consolidate to neocortex during sleep
            if sleep_signal > 0.3 and self._neocortical is not None:
                recall = self._hippocampal.recall(sensory_input.detach(), top_k=10)
                if recall['memories'].shape[0] > 0:
                    neo_result = self._neocortical.consolidate(recall['memories'])
                    cls_info['neocortical_updated'] = neo_result['num_updated']

        # ── Step 12c: Phase 15 — Causal reasoning (if enabled) ──
        causal_info = {}
        if self._causal_graph is not None and self._do_operator is not None:
            # Propagate causal influence through hierarchy levels
            # Errors have different dims per level — pad to max before stacking
            max_err_dim = max(err.shape[-1] for err in hpc_result['errors'])
            level_embeds = torch.stack([
                F.pad(err.detach(), (0, max_err_dim - err.shape[-1]))
                for err in hpc_result['errors']
            ])  # (n_levels, max_err_dim)
            # Pad/trim to state_dim for DoOperator
            if level_embeds.shape[-1] < c.state_dim:
                level_embeds = torch.nn.functional.pad(
                    level_embeds, (0, c.state_dim - level_embeds.shape[-1])
                )
            elif level_embeds.shape[-1] > c.state_dim:
                level_embeds = level_embeds[..., :c.state_dim]
            # Simple adjacency: each level feeds into the next
            n_lev = level_embeds.shape[0]
            adj = torch.zeros(n_lev, n_lev, device=device)
            for i in range(n_lev - 1):
                adj[i, i + 1] = 1.0
            propagated = self._do_operator.propagate(level_embeds, adj)
            causal_info['causal_propagation_norm'] = propagated.norm().detach()

        # ── Step 12d: Phase 13 — JEPA world model (if enabled) ──
        jepa_info = {}
        if self._jepa is not None and self._tick_py % 5 == 0:
            # JEPA predicts next latent from current state
            jepa_obs = sensory_input.detach().unsqueeze(0)
            dummy_action = torch.zeros(1, 4, device=device)
            jepa_pred = self._jepa.imagine(jepa_obs, dummy_action.unsqueeze(1))
            jepa_info['jepa_pred_norm'] = jepa_pred.norm().detach()

        # ── Step 12e: Phase 13 — Deep active inference (if enabled) ──
        deep_aif_info = {}
        if self._deep_aif is not None and self._tick_py % 10 == 0:
            aif_result = self._deep_aif.act(
                sensory_input.detach().unsqueeze(0),
                use_planner=False,
                deterministic=True,
            )
            deep_aif_info = {
                'aif_efe': aif_result['efe'].detach(),
                'aif_epistemic': aif_result['epistemic_value'].detach(),
            }

        # ── Step 12f: Phase 17 — Prospective learning (if enabled) ──
        prospective_info = {}
        if self._prospective is not None and self._tick_py % 5 == 0:
            prosp_input = hpc_input.squeeze(0) if hpc_input.dim() > 1 else hpc_input
            prosp_target = top_beliefs.squeeze(0) if top_beliefs.dim() > 1 else top_beliefs
            # Pad/trim target to match prospective output dim
            out_dim = c.hierarchy_dims[-1]
            if prosp_target.shape[-1] != out_dim:
                prosp_target = prosp_target[..., :out_dim] if prosp_target.shape[-1] > out_dim else torch.nn.functional.pad(prosp_target, (0, out_dim - prosp_target.shape[-1]))
            prosp_result = self._prospective.learn(prosp_input, prosp_target)
            prospective_info = {
                'prospective_improvement': prosp_result['improvement'],  # already float from .item()
                'prospective_loss': prosp_result['loss'].detach(),
            }

        # EP training (periodically, not every tick)
        ep_info = {}
        if self._ep_trainer is not None and self._tick_py % 10 == 0:
            # Use sensory input as external driving force (projected to oscillator dim)
            ext = self.phase_to_state.weight.T @ sensory_input.detach()
            # Simple loss: maximize order parameter
            def _ep_loss(ph, _tgt):
                complex_phases = torch.exp(1j * ph)
                r = torch.abs(complex_phases.mean())
                return (1.0 - r)  # Minimize 1-R
            ep_result = self._ep_trainer.train_step(ext, _ep_loss, torch.tensor(0.0, device=device))
            ep_info = {
                'ep_loss': ep_result['loss'],
                'ep_phase_shift': ep_result['phase_shift'],
            }

        # ── Step 12b: Phase 7 optional modules ──
        phase7_info = {}
        if self._no_progress is not None:
            np_result = self._no_progress(free_energy)
            phase7_info['progress_stalled'] = np_result['progress_stalled']
            phase7_info['stagnation_duration'] = np_result['stagnation_duration']
            phase7_info['trend'] = np_result['trend']

        if self.config.use_attractor_stability and self._tick_py % 50 == 0:
            asi = self.oscillators.compute_attractor_stability(n_trials=5, recovery_steps=10)
            phase7_info['attractor_stability'] = asi['stability_index']
            phase7_info['escape_probability'] = asi['escape_probability']

        if self._phase_cache is not None:
            cached = self._phase_cache.query(self.oscillators.phases)
            if cached is not None:
                phase7_info['phase_cache_hit'] = True
                phase7_info['phase_cache_outcome'] = cached['outcome']
                phase7_info['phase_cache_confidence'] = cached['confidence']
            else:
                phase7_info['phase_cache_hit'] = False

        # ── Step 12g: Metabolic budget (Enhancement C) ──
        metabolic_info = {}
        if c.use_metabolic_budget:
            with torch.no_grad():
                # Charge costs based on what happened this tick
                tick_cost = torch.tensor(0.0, device=device)

                # 1. Oscillation cost: scales with coherence (high R = locked = expensive)
                R_f = float(R_val) if not isinstance(R_val, float) else R_val
                tick_cost += c.metabolic_oscillation_cost * (0.5 + R_f)

                # 2. Workspace ignition cost (if GW fired this tick)
                if gw_info and gw_info.get('gw_winner', '') != '':
                    tick_cost += c.metabolic_ignition_cost

                # 3. Broadcast cost (ensemble cross-bleed writes)
                if ensemble_info:
                    tick_cost += c.metabolic_broadcast_cost

                # 4. HPC cost: scales with free energy (high FE = harder inference)
                fe_f = float(free_energy) if isinstance(free_energy, torch.Tensor) else free_energy
                tick_cost += 0.002 * min(2.0, abs(fe_f))

                # Deduct cost from budget
                self._energy_budget.sub_(tick_cost.clamp(min=0.0))

                # Recovery: passive energy restoration each tick
                self._energy_budget.add_(c.metabolic_recovery_rate)

                # Clamp to [0, capacity]
                self._energy_budget.clamp_(0.0, c.metabolic_capacity)

                # Track debt (how far below 50% we've gone)
                half = c.metabolic_capacity * 0.5
                if self._energy_budget < half:
                    self._metabolic_debt.copy_(half - self._energy_budget)
                else:
                    self._metabolic_debt.fill_(0.0)

                # Metabolic coupling to neuromodulation:
                # Low energy → reduce precision (save resources), increase NE (arousal)
                energy_ratio = self._energy_budget / c.metabolic_capacity
                if energy_ratio < 0.3:
                    # Low energy: dampen oscillator coupling (reduce synchronization cost)
                    damping = (0.7 + energy_ratio).clamp(0.7, 1.0)
                    self.oscillators.coupling_strength.mul_(damping)

                metabolic_info = {
                    'energy_budget': self._energy_budget.item(),
                    'metabolic_debt': self._metabolic_debt.item(),
                    'tick_cost': tick_cost.item(),
                    'energy_ratio': energy_ratio.item(),
                }

        # ── Step 13: Tick counter ──
        with torch.no_grad():
            self.tick_count.add_(1)
        self._tick_py += 1

        # ── Compute LLM modulation parameters ──
        # This is the critical output — what actually modulates the language model
        temperature = self._compute_temperature(coherence, criticality_distance)

        # Circadian → Temperature wiring: circadian rhythm modulates temperature
        if circadian_info:
            circ_temp = circadian_info.get('circadian_temperature', 1.0)
            # Blend: circadian warmth during rest, cooler during active
            temperature = temperature * (0.8 + 0.2 * circ_temp)
            temperature = max(0.1, min(2.0, temperature))

        personality_blend = self._compute_personality_blend(
            int(dominant_state_t) if isinstance(dominant_state_t, torch.Tensor) else dominant_state_t,
            float(narr_result['identity_strength']),
            attractor_state=personality_info.get('personality_state', None),
        )
        attention_bias = attention_map.detach()

        # ── Cache consciousness metrics (readable by bridge without diagnostics) ──
        with torch.no_grad():
            if isinstance(criticality_distance, torch.Tensor):
                self._cached_criticality_distance.copy_(criticality_distance)
            else:
                self._cached_criticality_distance.fill_(criticality_distance)
            if isinstance(free_energy, torch.Tensor):
                self._cached_free_energy.copy_(free_energy)
            else:
                self._cached_free_energy.fill_(free_energy)
            _id = narr_result['identity_strength']
            if isinstance(_id, torch.Tensor):
                self._cached_identity_strength.copy_(_id)
            else:
                self._cached_identity_strength.fill_(_id)

            # ── Fix 11: True theta-gamma PAC with Tort MI ──
            if self._theta_gamma_enabled:
                n_theta = min(c.theta_oscillators, c.num_oscillators - 1)
                theta_phases = phases[:n_theta]
                gamma_phases = phases[n_theta:]

                # Theta mean phase (circular mean)
                theta_complex = torch.exp(1j * theta_phases.to(torch.complex64))
                theta_mean_phase = torch.angle(theta_complex.mean()).remainder(2 * math.pi)

                # Gamma amplitude envelope: local phase coherence per gamma oscillator
                # Use R_gamma = |mean(e^{i*gamma_phase})| as gamma power proxy
                gamma_complex = torch.exp(1j * gamma_phases.to(torch.complex64))
                gamma_amplitude = torch.abs(gamma_complex.mean())

                # Push into Tort MI circular buffers
                buf_idx = self._tort_buf_ptr % 200
                self._tort_theta_buf[buf_idx] = theta_mean_phase.detach()
                self._tort_gamma_buf[buf_idx] = gamma_amplitude.detach()
                self._tort_buf_ptr += 1
                self._tort_buf_filled = min(self._tort_buf_filled + 1, 200)

                # Compute Tort MI every 50 ticks (requires enough samples)
                if self._tick_py % 50 == 0 and self._tort_buf_filled >= 36:
                    mi = self._compute_tort_mi()
                    self._cached_pac_mi.copy_(mi)

                # Also update legacy pac_strength for backward compat
                self._cached_pac_strength.copy_(self._cached_pac_mi)

                # Apply CFC if instantiated (theta modulates gamma coupling)
                if self._cfc is not None:
                    gamma_amps = torch.ones(c.num_oscillators - n_theta, device=phases.device)
                    modulated = self._cfc(gamma_amps.unsqueeze(0), theta_mean_phase)
                    pac_mod = modulated.squeeze(0).real.mean()
                    self._cached_pac_strength.copy_(pac_mod.clamp(0, 5))

                # ── Fix 16: Working memory capacity prediction (Lisman & Jensen 2013) ──
                # WM_cap = f_gamma / f_theta — number of gamma cycles nested per theta cycle.
                # With theta ~6 Hz and gamma ~40 Hz → ~6.7 items (matches Miller's "7 ± 2")
                theta_freqs = self.oscillators.natural_frequencies[:n_theta]  # rad/s
                gamma_freqs = self.oscillators.natural_frequencies[n_theta:]  # rad/s
                mean_theta_hz = theta_freqs.abs().mean() / (2 * math.pi)
                mean_gamma_hz = gamma_freqs.abs().mean() / (2 * math.pi)
                wm_capacity = mean_gamma_hz / mean_theta_hz.clamp(min=0.1)
                self._cached_wm_capacity.copy_(wm_capacity)

            # ── Fix 9 legacy: R-oscillation meta-coupling (when not using true PAC) ──
            elif self._cfc is not None:
                R_deviation = R_current - self._R_ema
                n_filled = min(self._R_history_ptr, 50)
                if n_filled >= 2:
                    prev_R = self._R_history[(self._R_history_ptr - 2) % 50]
                    R_velocity = R_current - prev_R
                else:
                    R_velocity = torch.zeros_like(R_current)
                theta_phase = torch.atan2(R_velocity, R_deviation).remainder(2 * math.pi)
                osc_amplitudes = torch.abs(torch.exp(1j * phases.to(torch.complex64)))
                modulated_amps = self._cfc(osc_amplitudes.unsqueeze(0), theta_phase)
                pac_mod = modulated_amps.squeeze(0).real.mean()
                self._cached_pac_strength.copy_(pac_mod.clamp(0, 5))

            # ── Fix 10+14: Consciousness quality metric Φ_m (v4) ──
            # Φ_m = metastability × PAC × R_window × crit_quality
            # Peaks when ALL four conditions are met:
            # - High metastability (SD(R) near target)
            # - Active PAC (theta-gamma or R-oscillation coupling)
            # - R in optimal consciousness range (0.2-0.6)
            # - Near-critical dynamics (σ ≈ 1.0)
            R_val = R_current.detach()
            R_window = (1.0 - 4.0 * (R_val - 0.4).pow(2)).clamp(min=0.0)
            pac_s = self._cached_pac_mi if self._theta_gamma_enabled else (
                self._cached_pac_strength if self._cfc is not None else torch.tensor(1.0, device=R_val.device)
            )
            # Fix 14: Criticality quality — 1.0 at σ=1.0, 0.0 far away
            crit_quality = (1.0 - (crit_sigma.detach() - 1.0).abs()).clamp(min=0.0, max=1.0)
            phi_m = self._cached_metastability * pac_s * R_window * crit_quality
            self._cached_phi_m.copy_(phi_m)

        # ── Fast return path ──
        # When return_diagnostics=False, return a tensor-only TickResult.
        # No .item() conversions, no dicts, no Python strings — fully
        # compatible with torch.compile and CUDA graphs.
        # Uses preallocated scratch buffers for scalar conversions to
        # avoid per-tick torch.tensor() allocations.
        if not return_diagnostics:
            # Temperature: in-place copy to scratch buffer
            if isinstance(temperature, torch.Tensor):
                self._scratch_temperature.copy_(temperature)
            else:
                self._scratch_temperature.fill_(temperature)

            # Personality blend: in-place copy to scratch buffer
            _pb = personality_blend
            _pb_weights = _pb['facet_weights'] if isinstance(_pb, dict) else _pb
            if isinstance(_pb_weights, torch.Tensor):
                self._scratch_personality.copy_(_pb_weights)
            else:
                for i, w in enumerate(_pb_weights):
                    self._scratch_personality[i] = w

            # Processing mode: in-place fill
            _pm_idx = _PROCESSING_MODE_FROM_STR.get(processing_mode, PROCESSING_MODE_BALANCED)
            self._scratch_processing_mode.fill_(_pm_idx)

            # Dominant state: in-place copy
            if isinstance(dominant_state_t, torch.Tensor):
                self._scratch_dominant_state.copy_(dominant_state_t)
            else:
                self._scratch_dominant_state.fill_(dominant_state_t)

            # Motivation: in-place copy
            if isinstance(combined_motivation, torch.Tensor):
                self._scratch_motivation.copy_(combined_motivation)
            else:
                self._scratch_motivation.fill_(combined_motivation)

            # Sleep signal: in-place fill
            if isinstance(sleep_signal, torch.Tensor):
                self._scratch_sleep_signal.copy_(sleep_signal)
            else:
                self._scratch_sleep_signal.fill_(sleep_signal)

            return TickResult(
                temperature=self._scratch_temperature,
                personality_blend=self._scratch_personality,
                attention_bias=attention_bias,
                processing_mode=self._scratch_processing_mode,
                coherence=coherence,
                dominant_state=self._scratch_dominant_state,
                motivation=self._scratch_motivation,
                sleep_signal=self._scratch_sleep_signal,
            )

        # ── Diagnostic path: build full dicts (backward compat) ──
        _coherence_f = float(coherence) if isinstance(coherence, torch.Tensor) else coherence
        _dominant_state_f = int(dominant_state_t) if isinstance(dominant_state_t, torch.Tensor) else dominant_state_t
        _comb_motiv_f = float(combined_motivation) if isinstance(combined_motivation, torch.Tensor) else combined_motivation

        modulation = {
            'temperature': temperature,
            'personality_blend': personality_blend,
            'attention_bias': attention_bias,
            'processing_mode': processing_mode,
            'coherence': _coherence_f,
            'dominant_state': _dominant_state_f,
            'motivation': _comb_motiv_f,
            'sleep_signal': sleep_signal,
        }

        # ── Batch scalar extraction ──
        # Most .item() calls are grouped here to minimize GPU sync points.
        # Each .item() forces a device sync on GPU; batching them at the
        # end of the tick avoids individual syncs scattered through
        # the computation path.
        _crit_dist = criticality_distance.item() if isinstance(criticality_distance, torch.Tensor) else criticality_distance
        _crit_br = crit_sigma.item() if isinstance(crit_sigma, torch.Tensor) else crit_sigma
        _crit_sus = crit['susceptibility'].item()
        _crit_Rm = crit['R_mean'].item()
        _crit_Rv = crit['R_var'].item()
        _pi_coup = crit_pi['coupling_adjustment'].item()
        _pi_noise = crit_pi['noise_adjustment'].item()
        _fe = free_energy.item()
        _sr_nl = sr_result['noise_level'].item()
        _sr_snr = sr_result['snr'].item()
        _id_str = narr_result['identity_strength'].item()
        _narr_dev = narr_result['deviation'].item()
        _self_pe = self_result['self_prediction_error_magnitude'].item()
        _sens_flow = blanket['sensory_flow'].item()
        _act_flow = blanket['active_flow'].item()
        _mean_fit = arena_result['mean_fitness'].item()
        _fit_var = arena_result['fitness_variance'].item()
        _n_mem = self.episodic_buffer.num_stored.item()
        _adapt_lr = adaptive_lr.mean().item()
        _mean_vol = vol_result['mean_volatility'].item()
        _tick_val = self.tick_count.item()

        # ── Neuroscience diagnostics (every 100 ticks when buffer is full) ──
        _neuro_metrics = {}
        if self._R_long_ptr >= self.config.diagnostics_buffer_len and _tick_val % 100 == 0:
            from pyquifer.neuro_diagnostics import (
                spectral_exponent, dfa_exponent, lempel_ziv_complexity,
                avalanche_statistics, complexity_entropy,
            )
            R_series = self._R_long_history.clone()
            # Use oscillator activity signal for avalanche detection (LFP analog)
            activity_series = self._activity_long_history.clone()
            _se = spectral_exponent(R_series)
            _dfa = dfa_exponent(R_series)
            _lz = lempel_ziv_complexity(R_series)
            _av = avalanche_statistics(activity_series)  # activity, NOT R
            _h, _c = complexity_entropy(R_series)
            _neuro_metrics = {
                'spectral_exponent': _se,
                'dfa_exponent': _dfa,
                'lempel_ziv': _lz,
                'avalanche_size_exp': _av['size_exponent'],
                'avalanche_duration_exp': _av['duration_exponent'],
                'avalanche_n': _av['n_avalanches'],
                'permutation_entropy': _h,
                'statistical_complexity': _c,
            }

        _reset_s = sc['reset_strength'].item() if isinstance(sc['reset_strength'], torch.Tensor) else sc['reset_strength']
        _coup_sc = sc['coupling_scale'].item() if isinstance(sc['coupling_scale'], torch.Tensor) else sc['coupling_scale']
        _dom_ratio = dominance_ratio.item() if isinstance(dominance_ratio, torch.Tensor) else dominance_ratio
        _coal_ent = coalition_entropy.item() if isinstance(coalition_entropy, torch.Tensor) else coalition_entropy

        if _sl_result is not None:
            stuart_landau_info = {
                'amplitudes': _sl_result['amplitudes'].detach(),
                'sl_order_parameter': _sl_result['order_parameter'].item(),
                'criticality_distance_sl': self._stuart_landau.get_criticality_distance().item(),
            }
        if _mf_result is not None:
            mean_field_info = {
                'mf_R': _mf_result['R'].item(),
                'mf_Psi': _mf_result['Psi'].item(),
                'mf_synchronized': self._mean_field.is_synchronized(),
            }
        if _koopman_result is not None:
            koopman_info = {
                'stability_margin': _koopman_result['stability_margin'].item(),
                'max_eigenvalue_mag': _koopman_result['max_eigenvalue_mag'].item(),
                'approaching_bifurcation': _koopman_result['approaching_bifurcation'],
            }

        # Fix 17: Per-module order parameters (Crowe et al. 2024)
        _mod_R = self.oscillators.get_module_order_parameters()

        return {
            'modulation': modulation,
            'consciousness': {
                'free_energy': _fe,
                'coherence': _coherence_f,
                'criticality_distance': _crit_dist,
                'criticality_sigma': _crit_br,
                'branching_ratio': _crit_br,
                'susceptibility': _crit_sus,
                'coalition_entropy': _coal_ent,
                'dominance_ratio': _dom_ratio,
                'processing_mode': processing_mode,
                'metastability_sd_R': self._cached_metastability.item(),
                'effective_target_meta': self._effective_target_meta,
                'dephasing_gain': self._cached_dephasing_gain.item(),
                'meta_r_coupling': self._cached_pac_strength.item(),  # R-oscillation coupling (legacy name: pac_strength)
                'pac_mi': self._cached_pac_mi.item(),  # Tort MI (true theta-gamma PAC)
                'wm_capacity': self._cached_wm_capacity.item(),  # Lisman-Jensen WM slots prediction
                'phi_m': self._cached_phi_m.item(),
                'sr_noise_level': _sr_nl,
                'sr_snr': _sr_snr,
            },
            'self_state': {
                'identity_strength': _id_str,
                'narrative_deviation': _narr_dev,
                'self_prediction_error': _self_pe,
                'sensory_flow': _sens_flow,
                'active_flow': _act_flow,
            },
            'learning': {
                'mean_fitness': _mean_fit,
                'fitness_variance': _fit_var,
                'num_memories': _n_mem,
                'consolidation': consolidation_info,
                'combined_motivation': _comb_motiv_f,
                'adaptive_lr': _adapt_lr,
                'mean_volatility': _mean_vol,
            },
            'diagnostics': {
                'tick': _tick_val,
                'phases': phases.detach(),
                'top_beliefs': top_beliefs.detach(),
                'precision': prec_result['precision'].detach(),
                'resources': arena_result['resources'].detach(),
                'neuromodulator_levels': levels.detach(),
                'criticality_sigma': _crit_br,
                'branching_ratio': _crit_br,
                'criticality_distance': _crit_dist,
                'susceptibility': _crit_sus,
                'R_mean_crit': _crit_Rm,
                'R_var_crit': _crit_Rv,
                'coupling_adjustment': _pi_coup,
                'noise_adjustment': _pi_noise,
                **stuart_landau_info,
                **mean_field_info,
                **koopman_info,
                **stp_info,
                **neural_mass_info,
                **phase7_info,
                **opc_info,
                **three_factor_info,
                **dendritic_info,
                **src_info,
                **ep_info,
                **gw_info,
                **ensemble_info,
                **circadian_info,
                **personality_info,
                **somatic_info,
                'theta_gate_value': theta_gate_value,
                'input_novelty': sc['input_novelty'],
                'phase_reset_strength': _reset_s,
                'coupling_scale': _coup_sc,
                'module_order_params': _mod_R.tolist() if _mod_R is not None else [],
                **complex_osc_info,
                **binding_info,
                **bus_info,
                **ssm_moe_info,
                **appraisal_info,
                **deliberation_info,
                **cls_info,
                **causal_info,
                **jepa_info,
                **deep_aif_info,
                **prospective_info,
                **metabolic_info,
                'neuro_metrics': _neuro_metrics,
            },
        }

    def _compute_tort_mi(self) -> torch.Tensor:
        """Tort et al. 2010 Modulation Index for phase-amplitude coupling.

        MI = (log(N_bins) - H(P)) / log(N_bins) where P is the normalized
        mean gamma amplitude per theta phase bin. MI=0 means uniform (no PAC),
        MI=1 means all gamma power concentrated in one phase bin (max PAC).

        Uses the circular buffers _tort_theta_buf and _tort_gamma_buf.

        Returns:
            Scalar tensor: Tort Modulation Index in [0, 1].
        """
        n = self._tort_buf_filled
        if n < self._tort_n_bins * 2:
            return torch.tensor(0.0, device=self._tort_theta_buf.device)

        theta = self._tort_theta_buf[:n]
        gamma = self._tort_gamma_buf[:n]

        # Bin theta phases into N_BINS equal-width bins (20° each)
        N = self._tort_n_bins
        bin_width = 2 * math.pi / N
        bin_indices = (theta / bin_width).long().clamp(0, N - 1)

        # Mean gamma amplitude per bin
        bin_sums = torch.zeros(N, device=theta.device)
        bin_counts = torch.zeros(N, device=theta.device)
        bin_sums.scatter_add_(0, bin_indices, gamma)
        bin_counts.scatter_add_(0, bin_indices, torch.ones_like(gamma))

        # Avoid empty bins: add small epsilon
        bin_means = bin_sums / bin_counts.clamp(min=1)

        # Normalize to probability distribution
        total = bin_means.sum()
        if total < 1e-8:
            return torch.tensor(0.0, device=theta.device)
        P = bin_means / total

        # Shannon entropy
        # H(P) = -sum(p * log(p)), with 0*log(0) = 0
        log_P = torch.log(P.clamp(min=1e-10))
        H = -(P * log_P).sum()

        # Modulation Index = (log(N) - H) / log(N)
        log_N = math.log(N)
        mi = (log_N - H) / log_N
        return mi.clamp(min=0.0, max=1.0)

    def _compute_temperature(self, coherence, criticality_distance) -> float:
        """
        Map consciousness state to LLM temperature.

        High coherence + near criticality → lower temperature (focused)
        Low coherence + far from criticality → higher temperature (creative)

        Accepts both float and tensor inputs. Returns float.
        """
        # Base temperature from coherence (inverted: high coherence = low temp)
        base_temp = 1.0 - 0.5 * coherence  # [0.5, 1.0]

        # Criticality modulation: near critical = slightly more creative
        if isinstance(criticality_distance, torch.Tensor):
            crit_mod = 0.1 * (1.0 - criticality_distance).clamp(min=0)
            temp = (base_temp + crit_mod).clamp(min=0.1, max=2.0).item()
        else:
            crit_mod = 0.1 * max(0, 1.0 - criticality_distance)
            temp = max(0.1, min(2.0, base_temp + crit_mod))
        return temp

    def _compute_personality_blend(self, dominant_state: int,
                                    identity_strength: float,
                                    attractor_state: Optional[torch.Tensor] = None,
                                    ) -> Dict[str, float]:
        """
        Map metastable dominant state to personality expression weights.

        Each population in the WinnerlessCompetition maps to a personality facet.
        Identity strength determines how much personality constrains the blend.
        When a PersonalityAttractor is active, its state biases the facet weights.
        """
        num_pop = self.config.num_populations

        # Base weights from dominant state (one-hot with softening)
        weights = [0.1] * num_pop
        weights[dominant_state % num_pop] = 1.0

        # Personality → Metastability wiring: attractor state biases facet weights
        if attractor_state is not None:
            # Map attractor dimensions to population weights via chunked mean
            chunk = max(1, attractor_state.shape[0] // num_pop)
            for i in range(num_pop):
                start = i * chunk
                end = min(start + chunk, attractor_state.shape[0])
                if end > start:
                    bias = attractor_state[start:end].mean().item()
                    weights[i] += abs(bias) * 0.3  # Gentle bias, not override

        # Normalize
        total = sum(weights)
        weights = [w / total for w in weights]

        # Identity strength determines how stable the blend is
        # High identity = less susceptible to state changes
        stability = min(1.0, identity_strength)

        return {
            'facet_weights': weights,
            'stability': stability,
            'dominant_facet': dominant_state % num_pop,
        }

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a compact summary of the current cognitive state."""
        coherence = self.oscillators.get_order_parameter()
        return {
            'tick': self.tick_count.item(),
            'coherence': coherence.item() if isinstance(coherence, torch.Tensor) else coherence,
            'num_memories': self.episodic_buffer.num_stored.item(),
            'semantic_traces': self.consolidation.num_traces.item(),
            'identity_strength': self.narrative.identity_strength.item(),
        }

    def reset(self):
        """Reset all module states."""
        self.tick_count.zero_()
        self._tick_py = 0
        self.hpc.reset()
        self.precision.reset()
        self.metastability.reset()
        self.arena.reset()
        self.episodic_buffer.reset()
        self.consolidation.reset()
        self.markov_blanket.reset()
        self.self_model.reset()
        self.narrative.reset()
        self.stochastic_resonance.reset()
        self.volatility_gate.reset()

        # Phase 5 optional modules
        if self._stuart_landau is not None:
            self._stuart_landau.reset()
        if self._mean_field is not None:
            self._mean_field.reset()
        if self._koopman is not None:
            self._koopman.reset()
        if self._stp is not None:
            self._stp.reset()
        if self._neural_mass is not None:
            self._neural_mass.reset()

        # Phase 8 training modules
        if self._oscillation_gated is not None:
            self._oscillation_gated.reset()
        if self._three_factor is not None:
            self._three_factor.reset()
        if self._oscillatory_predictive is not None:
            self._oscillatory_predictive.reset()
        if self._sleep_consolidation is not None:
            self._sleep_consolidation.reset()
        if self._dendritic_stack is not None:
            self._dendritic_stack.reset()
        # EP trainer has no state to reset (bank phases are reset with oscillators)

        # Phase 10 workspace ensemble
        if self._workspace_ensemble is not None:
            self._workspace_ensemble.reset()

        # Phase 9b cross-module wiring modules (no reset methods — state is organic)

        # Phase 7 optional modules
        if self._no_progress is not None:
            self._no_progress.reset()
        if self._evidence_aggregator is not None:
            self._evidence_aggregator.clear()
        # Phase cache is not reset (accumulated knowledge persists)

        # Phase 11-18 optional modules
        if self._complex_oscillators is not None:
            self._complex_oscillators.reset()
        if self._hippocampal is not None:
            self._hippocampal.reset()
        if self._neocortical is not None:
            self._neocortical.reset()

    @staticmethod
    def _detect_compile_backend(preferred: str, device: str = "cpu") -> str:
        """Choose the best available torch.compile backend.

        On CUDA, ``inductor`` generates fused Triton kernels — always
        available.  On CPU, it generates C++/OpenMP loops which require
        a C++ compiler (``cl.exe`` on Windows, ``gcc``/``clang`` on Linux).
        When no compiler is found, returns ``None`` to signal that
        compilation should be skipped entirely (``aot_eager`` adds
        overhead without kernel fusion).

        Returns:
            Backend name, or None if compilation is not beneficial.
        """
        if preferred not in ("inductor", "auto"):
            return preferred

        # CUDA always has Triton
        if device != "cpu" or torch.cuda.is_available():
            # If current device is CUDA or could be, inductor works
            if device != "cpu":
                return "inductor"

        # CPU: inductor needs a C++ compiler
        import sys
        if sys.platform == "win32":
            import shutil
            if shutil.which("cl") is None:
                logger.info(
                    "cl.exe not found — torch.compile skipped on CPU "
                    "(install MSVC Build Tools, or use .to('cuda') "
                    "for Triton-based compilation)"
                )
                return None
        elif sys.platform != "darwin":
            # Linux: check for gcc/g++
            import shutil
            if shutil.which("gcc") is None and shutil.which("g++") is None:
                logger.info("No C++ compiler found — torch.compile skipped")
                return None
        return "inductor"

    def compile_modules(self, mode: str = "default",
                        backend: str = "inductor") -> 'CognitiveCycle':
        """Apply torch.compile to performance-critical submodules.

        Fuses small tensor operations (sin, cos, clamp, mul, add) into
        fewer kernel launches.  On CPU, inductor generates C++/OpenMP loops.
        On CUDA, it generates fused Triton kernels or CUDA graphs.
        Skips compilation on Windows when MSVC (cl.exe) is unavailable.

        Modules compiled:
        - oscillators (LearnableKuramotoBank) — Kuramoto dynamics
        - _sensory_coupling (SensoryCoupling) — input-to-oscillator coupling
        - hpc (HierarchicalPredictiveCoding) — predictive coding hierarchy
        - precision (AttentionAsPrecision) — precision weighting
        - criticality feedback function — extracted pure-tensor loop
        - phase_to_state, error_to_levels, state_to_group — linear projections

        The tick() method itself is NOT compiled because it has extensive
        Python branching (optional modules, dict construction). Instead we
        compile the leaf modules; Dynamo traces through their forward()
        calls when they are invoked from tick().

        Args:
            mode: Compile mode.
                "default" — balanced compile time vs runtime speed
                "reduce-overhead" — uses CUDA graphs; best for CUDA with
                    static shapes (requires PyTorch >= 2.1)
                "max-autotune" — longest compile, fastest runtime
            backend: Compilation backend. "inductor" (default, auto-detected),
                "aot_eager" (no codegen, still optimizes graph), or
                "eager" (no-op, useful for debugging graph breaks).

        Returns:
            self (for chaining: ``cycle.compile_modules().to("cuda")``)
        """
        global _compiled_criticality_feedback

        if not hasattr(torch, 'compile'):
            logger.info("torch.compile unavailable (PyTorch < 2.0), skipping")
            return self

        # Auto-detect the best available backend based on current device
        device = str(next(self.parameters()).device)
        backend = self._detect_compile_backend(backend, device=device)
        if backend is None:
            return self

        # Smoke test: verify the backend can actually compile + run code.
        # This catches missing Triton (CUDA) or cl.exe (CPU) early,
        # before we wrap all modules and hit runtime failures.
        try:
            _test = torch.compile(lambda x: x + 1, backend=backend,
                                  fullgraph=True)
            _test(torch.tensor(1.0, device=device))
        except Exception as e:
            logger.warning(
                "torch.compile smoke test failed (%s backend): %s — "
                "skipping compilation", backend, e
            )
            return self

        # Suppress runtime compilation errors so a single module failure
        # doesn't crash the whole system — just falls back to eager.
        _dynamo = __import__('torch._dynamo', fromlist=['config'])
        _dynamo.config.suppress_errors = True
        # HPC has multiple PredictiveLevel instances with different dims
        # (e.g. [64, 32, 16]), each needing a separate specialization.
        # Default cache_size_limit=8 is too low; raise to 32.
        _dynamo.config.cache_size_limit = 32

        compiled = []
        failed = []

        def _try_compile(name, module):
            try:
                out = torch.compile(module, mode=mode, backend=backend,
                                    fullgraph=False)
                compiled.append(name)
                return out
            except Exception as e:
                failed.append((name, str(e)))
                return module

        # Core dynamics
        self.oscillators = _try_compile("oscillators", self.oscillators)
        self._sensory_coupling = _try_compile(
            "sensory_coupling", self._sensory_coupling
        )

        # Predictive coding
        self.hpc = _try_compile("hpc", self.hpc)
        self.precision = _try_compile("precision", self.precision)

        # Criticality monitors
        self._kuramoto_criticality = _try_compile(
            "kuramoto_criticality", self._kuramoto_criticality
        )
        self.criticality = _try_compile("criticality", self.criticality)

        # Stochastic resonance (has .item() graph breaks but partial
        # compilation of the detection kernel still helps)
        self.stochastic_resonance = _try_compile(
            "stochastic_resonance", self.stochastic_resonance
        )

        # Linear projections (trivially compiled)
        self.phase_to_state = _try_compile("phase_to_state", self.phase_to_state)
        self.error_to_levels = _try_compile("error_to_levels", self.error_to_levels)

        # Criticality feedback standalone function
        try:
            _compiled_criticality_feedback = torch.compile(
                _criticality_feedback, mode=mode, backend=backend,
                fullgraph=True,
            )
            compiled.append("criticality_feedback")
        except Exception as e:
            failed.append(("criticality_feedback", str(e)))

        # Neuromodulation + volatility (small but frequently called)
        self.neuromodulation = _try_compile("neuromodulation", self.neuromodulation)
        self.volatility_gate = _try_compile("volatility_gate", self.volatility_gate)

        # Phase 11-18 modules (conditionally present)
        if hasattr(self, '_complex_oscillators') and self._complex_oscillators is not None:
            self._complex_oscillators = _try_compile("complex_oscillators", self._complex_oscillators)

        if hasattr(self, '_selective_ssm') and self._selective_ssm is not None:
            self._selective_ssm = _try_compile("selective_ssm", self._selective_ssm)

        if hasattr(self, '_oscillatory_moe') and self._oscillatory_moe is not None:
            self._oscillatory_moe = _try_compile("oscillatory_moe", self._oscillatory_moe)

        if hasattr(self, '_deep_aif') and self._deep_aif is not None:
            self._deep_aif = _try_compile("deep_aif", self._deep_aif)

        if hasattr(self, '_jepa') and self._jepa is not None:
            self._jepa = _try_compile("jepa", self._jepa)

        if hasattr(self, '_appraisal') and self._appraisal is not None:
            self._appraisal = _try_compile("appraisal", self._appraisal)

        if hasattr(self, '_gated_memory') and self._gated_memory is not None:
            self._gated_memory = _try_compile("gated_memory", self._gated_memory)

        # Store compilation status for diagnostics
        self._compile_status = {
            'compiled': compiled,
            'failed': failed,
            'mode': mode,
            'backend': backend,
        }

        if compiled:
            logger.info("Compiled %d modules (%s backend): %s",
                        len(compiled), backend, ", ".join(compiled))
        if failed:
            logger.warning("Failed to compile %d modules: %s",
                           len(failed),
                           ", ".join(f"{n}: {e}" for n, e in failed))
        return self

    def explain_graph_breaks(self, sensory_input: torch.Tensor = None
                             ) -> Dict[str, Any]:
        """Analyze graph breaks in compiled leaf modules using torch._dynamo.explain.

        Returns a dict per module with break count and break reasons.
        Useful as a regression gate: minimal tick should have 0 breaks
        in critical modules (oscillators, precision, projections).

        Args:
            sensory_input: Optional (state_dim,) input for tracing.
                          If None, creates a random input.

        Returns:
            Dict mapping module name → ``{'break_count': int, 'reasons': list}``.
        """
        try:
            import torch._dynamo as dynamo
        except ImportError:
            return {'error': 'torch._dynamo not available'}

        c = self.config
        if sensory_input is None:
            device = next(self.parameters()).device
            sensory_input = torch.randn(c.state_dim, device=device)

        results = {}

        # Test key leaf modules with representative inputs
        test_cases = {
            'oscillators': (
                self.oscillators,
                (sensory_input[:c.num_oscillators],),
                {'steps': 1, 'use_precision': True},
            ),
            'phase_to_state': (
                self.phase_to_state,
                (torch.sin(self.oscillators.phases).unsqueeze(0),),
                {},
            ),
        }

        for name, (module, args, kwargs) in test_cases.items():
            try:
                explanation = dynamo.explain(module)(*args, **kwargs)
                results[name] = {
                    'break_count': explanation.break_count,
                    'reasons': [str(r) for r in explanation.break_reasons],
                }
            except Exception as e:
                results[name] = {
                    'break_count': -1,
                    'reasons': [f'explain failed: {e}'],
                }

        return results


if __name__ == '__main__':
    print("--- CognitiveCycle Integration Test ---\n")

    # Use small config for testing
    config = CycleConfig.small()
    cycle = CognitiveCycle(config)

    print(f"Config: state_dim={config.state_dim}, "
          f"hierarchy={config.hierarchy_dims}, "
          f"populations={config.num_populations}")

    total_params = sum(p.numel() for p in cycle.parameters())
    total_buffers = sum(b.numel() for b in cycle.buffers())
    print(f"Parameters: {total_params:,}")
    print(f"Buffers: {total_buffers:,}")
    print()

    # Run 50 ticks
    torch.manual_seed(42)
    for i in range(50):
        sensory = torch.randn(config.state_dim) * 0.5

        # Vary reward and sleep
        reward = math.sin(i * 0.1) * 2.0
        sleep = 1.0 if i > 40 else 0.0  # Sleep for last 10 ticks

        result = cycle.tick(sensory, reward=reward, sleep_signal=sleep)

        if i % 10 == 0:
            m = result['modulation']
            c = result['consciousness']
            s = result['self_state']
            l = result['learning']
            print(f"Tick {i:3d}: "
                  f"temp={m['temperature']:.2f} "
                  f"coherence={c['coherence']:.3f} "
                  f"FE={c['free_energy']:.3f} "
                  f"mode={c['processing_mode']:>11s} "
                  f"identity={s['identity_strength']:.3f} "
                  f"memories={l['num_memories']}")

    # Sleep consolidation summary
    print(f"\nAfter sleep: {result['learning']['consolidation']}")
    print(f"State summary: {cycle.get_state_summary()}")

    # Reset
    cycle.reset()
    print(f"\nAfter reset: {cycle.get_state_summary()}")

    print("\n[OK] CognitiveCycle integration test passed!")
