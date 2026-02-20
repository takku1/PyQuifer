"""CognitiveCycle configuration."""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


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
