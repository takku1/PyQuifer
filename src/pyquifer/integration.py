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
import math
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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

    # Phase 9b: Cross-module integration wiring
    use_circadian: bool = False            # ChronobiologicalSystem → consolidation timing
    use_personality_attractor: bool = False # PersonalityAttractor → metastability/personality
    use_somatic: bool = False              # SomaticManifold → self-model/narrative
    use_phase_dominance: bool = False      # Oscillator phases → causal flow dominance
    use_motivation_priority: bool = False  # Motivation → memory consolidation priority

    # Phase 6: Integration method (G-17)
    integration_method: str = 'euler'  # 'euler' or 'rk4' — passed to oscillators/neural_mass

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


def _criticality_feedback(
    osc_phases: torch.Tensor,
    osc_coupling_data: torch.Tensor,
    crit_sigma: torch.Tensor,
    R_current: torch.Tensor,
) -> torch.Tensor:
    """Pure-tensor criticality feedback loop (slow + fast paths).

    Extracted from tick() so torch.compile can fuse the ~15 element-wise
    ops into 1-2 kernels.  No .item() calls, no Python branching on
    tensor values, no dict construction — fully traceable.

    Args:
        osc_phases: Oscillator phase buffer (num_oscillators,)
        osc_coupling_data: coupling_strength.data (scalar tensor)
        crit_sigma: Branching ratio σ from KuramotoCriticalityMonitor
        R_current: Instantaneous order parameter R

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

    # ── FAST PATH: Inhibitory dephasing on instantaneous R ──
    R_excess = (R_current - 0.52).clamp(min=0.0)
    noise_scale = (R_excess * 1.5).clamp(max=0.8)
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

        # === Layer 3: Dynamical Core ===
        from pyquifer.oscillators import LearnableKuramotoBank, SensoryCoupling
        self.oscillators = LearnableKuramotoBank(
            num_oscillators=c.num_oscillators,
            dt=c.oscillator_dt,
            integration_method=c.integration_method,
        )

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
                                    phases: torch.Tensor) -> Dict[str, Any]:
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

            # Apply diversity pressure
            boost = self._diversity_tracker.get_boost(organ.organ_id)
            proposal.salience = proposal.salience * gate_val + boost

            proposals.append((organ, adapter, proposal))

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

        return {
            'gw_broadcast': broadcast.detach(),
            'gw_winner': winner_id,
            'gw_saliences': saliences.detach().squeeze(0),
            'gw_did_ignite': gw_result['did_ignite'].any().item(),
            'gw_cycle_consistency_loss': cc_loss,
        }

    def tick(self,
             sensory_input: torch.Tensor,
             reward: float = 0.0,
             sleep_signal: float = 0.0,
             ) -> Dict[str, Any]:
        """
        Run one cognitive tick.

        Args:
            sensory_input: Raw sensory input (state_dim,) or (batch, state_dim)
            reward: External reward signal (for motivation + memory priority)
            sleep_signal: 0.0 = awake, 1.0 = deep sleep (gates consolidation)

        Returns:
            Dict with:
            - modulation: Dict of LLM modulation parameters
            - consciousness: Dict of consciousness metrics
            - self_state: Dict of self-model state
            - diagnostics: Dict of internal metrics for monitoring
        """
        c = self.config
        was_1d = sensory_input.dim() == 1
        if was_1d:
            sensory_input = sensory_input.unsqueeze(0)

        device = sensory_input.device

        # ── Step 0: Sensory-oscillator coupling ──
        # Input drives oscillators via frequency entrainment, phase resets,
        # and coupling modulation (Lakatos et al. 2008; Fries 2005 CTC).
        # All modifications are within torch.no_grad — gradient flow from
        # the LLM to oscillators is severed by design.
        sc = self._sensory_coupling(sensory_input[0], self.oscillators.phases)

        with torch.no_grad():
            # Phase reset: novel stimuli partially reset phases toward
            # input-derived targets (event-related desynchronization)
            reset_s = sc['reset_strength']
            if reset_s > 0.05:
                blended = (
                    (1.0 - reset_s) * self.oscillators.phases
                    + reset_s * sc['phase_targets']
                ) % (2 * math.pi)
                self.oscillators.phases.copy_(blended)

            # Coupling modulation: input salience scales K
            self.oscillators.coupling_strength.data.mul_(
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
        coherence = order_param if isinstance(order_param, torch.Tensor) else torch.tensor(order_param, device=device)

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
            circadian_plasticity = circ_result['plasticity'].item() if isinstance(circ_result['plasticity'], torch.Tensor) else circ_result['plasticity']
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
        R_current = self.oscillators.get_order_parameter()
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
            # Criticality feedback: dual-timescale control extracted into a
            # standalone function so torch.compile can fuse all ~15 element-wise
            # ops into 1-2 kernels.  See _criticality_feedback() docstring.
            _crit_fn = _compiled_criticality_feedback or _criticality_feedback
            _crit_fn(
                self.oscillators.phases,
                self.oscillators.coupling_strength.data,
                crit_sigma,
                R_current,
            )

        # ── Step 2b: Optional Koopman bifurcation detection ──
        koopman_info = {}
        _koopman_result = None
        if self._koopman is not None:
            _koopman_result = self._koopman(sensory_input[0].detach())

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
                sensory_input.squeeze(0) if was_1d else sensory_input[0],
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
                'mean_facilitation': stp_result['u'].mean().item(),
                'mean_depression': stp_result['x'].mean().item(),
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
            theta_gate_value = self._oscillation_gated.gate_value.item()

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
                'gamma_power': opc_result['gamma_power'].item() if isinstance(opc_result['gamma_power'], torch.Tensor) else opc_result['gamma_power'],
                'alpha_beta_power': opc_result['alpha_beta_power'].item() if isinstance(opc_result['alpha_beta_power'], torch.Tensor) else opc_result['alpha_beta_power'],
            }

        if self._tick_py % c.step_every_hpc == 0 or self._cached_hpc is None:
            hpc_result = self.hpc(hpc_input)
            self._cached_hpc = hpc_result
        else:
            hpc_result = self._cached_hpc
        prediction_error = hpc_result['errors'][0]  # Bottom-level errors
        free_energy = hpc_result['free_energy']
        top_beliefs = hpc_result['top_level_beliefs']

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

        dom_result = self.dominance(level_activations, compute_every=50)
        dominance_ratio = dom_result['dominance_ratio']  # keep tensor
        processing_mode = dom_result['mode']

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
            gw_info = self._run_workspace_competition(sensory_input[0], phases)

        # ── Step 8c: Multi-Workspace Ensemble (if enabled) ──
        ensemble_info = {}
        if self._workspace_ensemble is not None and self.config.use_global_workspace:
            n_ws = self.config.n_workspaces
            ws_dim = self.config.workspace_dim
            # Build per-workspace contents from organ proposals + standing latents
            # Active workspace (idx 0) gets fresh proposals, background gets standings
            contents_per_ws = []
            contexts_per_ws = []
            for wi in range(n_ws):
                if wi == self._workspace_ensemble.active_idx.item() and gw_info:
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
                'nm_synchronization': nm_result['synchronization'].item(),
                'nm_mean_E': nm_result['mean_E'].item(),
            }

        # ── Step 9: Neural darwinism ──
        group_input = self.state_to_group(sensory_input[0])
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
                sensory_input[0],
                order_parameter=coherence.detach() if isinstance(coherence, torch.Tensor) else torch.tensor(coherence, device=device),
            )
            self._cached_motiv = motiv_result
        else:
            motiv_result = self._cached_motiv
        combined_motivation = motiv_result['motivation']  # keep tensor

        # ── Step 11: Self-model ──
        somatic_info = {}
        if self._tick_py % c.step_every_selfmodel == 0 or self._cached_self is None:
            sensory_for_blanket = self.state_to_sensory(sensory_input[0])
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
                'homeostatic_factor': self._three_factor.homeostatic_factor.mean().item(),
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
            sensory_input[0].detach(),
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
                if c.use_motivation_priority and combined_motivation > 0:
                    # Scale rewards: motivation boosts consolidation priority
                    priority_scale = 1.0 + combined_motivation * 0.5  # [1.0, ~2.0]
                    replay_rewards = replay_rewards * priority_scale

                # Circadian → Consolidation wiring: plasticity modulates consolidation
                if self._circadian is not None:
                    replay_rewards = replay_rewards * circadian_plasticity

                cons = self.consolidation(
                    replay['replayed_states'],
                    replay_rewards,
                )
                consolidation_info = {
                    'consolidated': cons['consolidated'].item() if isinstance(cons['consolidated'], torch.Tensor) else cons['consolidated'],
                    'num_traces': cons['num_traces'].item() if isinstance(cons['num_traces'], torch.Tensor) else cons['num_traces'],
                }

            # SRC Hebbian sleep consolidation
            if self._sleep_consolidation is not None:
                src_result = self._sleep_consolidation.sleep_step()
                src_info = {
                    'src_delta_norm': sum(src_result['weight_delta_norms']) / len(src_result['weight_delta_norms']),
                }

        # EP training (periodically, not every tick)
        ep_info = {}
        if self._ep_trainer is not None and self._tick_py % 10 == 0:
            # Use sensory input as external driving force (projected to oscillator dim)
            ext = self.phase_to_state.weight.T @ sensory_input[0].detach()
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
            phase7_info['progress_stalled'] = np_result['progress_stalled'].item() if isinstance(np_result['progress_stalled'], torch.Tensor) else np_result['progress_stalled']
            phase7_info['stagnation_duration'] = np_result['stagnation_duration'].item() if isinstance(np_result['stagnation_duration'], torch.Tensor) else np_result['stagnation_duration']
            phase7_info['trend'] = np_result['trend'].item() if isinstance(np_result['trend'], torch.Tensor) else np_result['trend']

        if self.config.use_attractor_stability and self._tick_py % 50 == 0:
            asi = self.oscillators.compute_attractor_stability(n_trials=5, recovery_steps=10)
            phase7_info['attractor_stability'] = asi['stability_index'].item()
            phase7_info['escape_probability'] = asi['escape_probability'].item()

        if self._phase_cache is not None:
            cached = self._phase_cache.query(self.oscillators.phases)
            if cached is not None:
                phase7_info['phase_cache_hit'] = True
                phase7_info['phase_cache_outcome'] = cached['outcome']
                phase7_info['phase_cache_confidence'] = cached['confidence']
            else:
                phase7_info['phase_cache_hit'] = False

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
            dominant_state_t.item() if isinstance(dominant_state_t, torch.Tensor) else dominant_state_t,
            narr_result['identity_strength'].item(),
            attractor_state=personality_info.get('personality_state', None),
        )
        attention_bias = attention_map.detach()

        # ── Batch scalar extraction ──
        # ALL .item() calls are grouped here to minimize GPU sync points.
        # Each .item() forces a device sync on GPU; batching them at the
        # end of the tick avoids ~25 individual syncs scattered through
        # the computation path.
        _coherence = coherence.item() if isinstance(coherence, torch.Tensor) else coherence
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
        _reset_s = sc['reset_strength'].item() if isinstance(sc['reset_strength'], torch.Tensor) else sc['reset_strength']
        _coup_sc = sc['coupling_scale'].item() if isinstance(sc['coupling_scale'], torch.Tensor) else sc['coupling_scale']
        _dom_ratio = dominance_ratio.item() if isinstance(dominance_ratio, torch.Tensor) else dominance_ratio
        _coal_ent = coalition_entropy.item() if isinstance(coalition_entropy, torch.Tensor) else coalition_entropy
        _dominant_state = dominant_state_t.item() if isinstance(dominant_state_t, torch.Tensor) else dominant_state_t
        _comb_motiv = combined_motivation.item() if isinstance(combined_motivation, torch.Tensor) else combined_motivation

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

        return {
            'modulation': {
                'temperature': temperature,
                'personality_blend': personality_blend,
                'attention_bias': attention_bias,
                'processing_mode': processing_mode,
                'coherence': _coherence,
                'dominant_state': _dominant_state,
                'motivation': _comb_motiv,
                'sleep_signal': sleep_signal,
            },
            'consciousness': {
                'free_energy': _fe,
                'coherence': _coherence,
                'criticality_distance': _crit_dist,
                'criticality_sigma': _crit_br,
                'branching_ratio': _crit_br,
                'susceptibility': _crit_sus,
                'coalition_entropy': _coal_ent,
                'dominance_ratio': _dom_ratio,
                'processing_mode': processing_mode,
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
                'combined_motivation': _comb_motiv,
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
            },
        }

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
        Falls back to ``aot_eager`` on Windows when MSVC is unavailable.

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

        if compiled:
            logger.info("Compiled %d modules (%s backend): %s",
                        len(compiled), backend, ", ".join(compiled))
        if failed:
            logger.warning("Failed to compile %d modules: %s",
                           len(failed),
                           ", ".join(f"{n}: {e}" for n, e in failed))
        return self


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
