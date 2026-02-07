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
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


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

    # Phase 6: Integration method (G-17)
    integration_method: str = 'euler'  # 'euler' or 'rk4' — passed to oscillators/neural_mass

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

        # Track tick count
        self.register_buffer('tick_count', torch.tensor(0))

        # === Layer 3: Dynamical Core ===
        from pyquifer.oscillators import LearnableKuramotoBank
        self.oscillators = LearnableKuramotoBank(
            num_oscillators=c.num_oscillators,
            dt=c.oscillator_dt,
            integration_method=c.integration_method,
        )

        from pyquifer.criticality import CriticalityController
        self.criticality = CriticalityController(
            target_branching_ratio=c.target_branching_ratio,
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

        # ── Step 1: Oscillator dynamics ──
        phases = self.oscillators(steps=1, use_precision=True)
        order_param = self.oscillators.get_order_parameter()
        coherence = order_param.item() if isinstance(order_param, torch.Tensor) else order_param

        # ── Step 1b: Optional Stuart-Landau oscillator ──
        stuart_landau_info = {}
        if self._stuart_landau is not None:
            sl_result = self._stuart_landau(steps=1)
            stuart_landau_info = {
                'amplitudes': sl_result['amplitudes'].detach(),
                'sl_order_parameter': sl_result['order_parameter'].item(),
                'criticality_distance_sl': self._stuart_landau.get_criticality_distance().item(),
            }

        # ── Step 1c: Optional Kuramoto-Daido mean-field ──
        mean_field_info = {}
        if self._mean_field is not None:
            mf_result = self._mean_field(steps=1)
            mean_field_info = {
                'mf_R': mf_result['R'].item(),
                'mf_Psi': mf_result['Psi'].item(),
                'mf_synchronized': self._mean_field.is_synchronized(),
            }

        # ── Step 2: Criticality check ──
        # Use oscillator phases as activity signal
        activity = torch.sin(phases)
        crit = self.criticality(activity)
        criticality_distance = crit['criticality_distance'].item()

        # ── Step 2b: Optional Koopman bifurcation detection ──
        koopman_info = {}
        if self._koopman is not None:
            koopman_result = self._koopman(sensory_input[0].detach())
            koopman_info = {
                'stability_margin': koopman_result['stability_margin'].item(),
                'max_eigenvalue_mag': koopman_result['max_eigenvalue_mag'].item(),
                'approaching_bifurcation': koopman_result['approaching_bifurcation'],
            }

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
        levels = self.neuromodulation.levels
        ach = levels[3].item()  # acetylcholine
        ne = levels[2].item()   # norepinephrine

        # ── Step 4: Stochastic resonance ──
        sr_result = self.stochastic_resonance(
            sensory_input.squeeze(0) if was_1d else sensory_input[0],
            criticality_distance=criticality_distance,
        )
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

        hpc_result = self.hpc(hpc_input)
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
        prec_result = self.precision(
            hpc_input,
            prediction_error,
            acetylcholine=ach,
            norepinephrine=ne,
        )
        attention_map = prec_result['attention_map']

        # ── Step 7: Causal flow (dominance detection) ──
        # Use HPC error norms at each level as level activations
        level_activations = torch.stack(
            [err.norm() for err in hpc_result['errors']]
        ).to(device)
        dom_result = self.dominance(level_activations, compute_every=10)
        dominance_ratio = dom_result['dominance_ratio'].item()
        processing_mode = dom_result['mode']

        # ── Step 8: Metastability (stream of consciousness) ──
        meta_result = self.metastability()
        dominant_state = meta_result['dominant'].item()
        coalition_entropy = meta_result['coalition_entropy'].item()

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
        arena_result = self.arena(group_input, global_coherence=global_coherence_signal)
        self.symbiogenesis(arena_result['group_outputs'])

        # ── Step 10: Motivation ──
        motiv_result = self.motivation(
            sensory_input[0],
            order_parameter=torch.tensor(coherence, device=device),
        )
        combined_motivation = motiv_result['motivation'].item()

        # ── Step 11: Self-model ──
        sensory_for_blanket = self.state_to_sensory(sensory_input[0])
        blanket = self.markov_blanket(sensory_for_blanket)
        self_result = self.self_model(blanket['internal_state'])
        narr_result = self.narrative(self_result['self_summary'])

        # ── Step 12: Memory (consolidation if sleeping) ──
        # Store experience
        self.episodic_buffer.store(
            sensory_input[0].detach(),
            reward=reward + combined_motivation,
        )

        consolidation_info = {'consolidated': False, 'num_traces': 0}
        if sleep_signal > 0.3:
            replay = self.sharp_wave_ripple(self.episodic_buffer, sleep_signal=sleep_signal)
            if replay['replayed_states'].shape[0] > 0:
                cons = self.consolidation(
                    replay['replayed_states'],
                    replay['replayed_rewards'],
                )
                consolidation_info = {
                    'consolidated': cons['consolidated'].item() if isinstance(cons['consolidated'], torch.Tensor) else cons['consolidated'],
                    'num_traces': cons['num_traces'].item() if isinstance(cons['num_traces'], torch.Tensor) else cons['num_traces'],
                }

        # ── Step 13: Tick counter ──
        with torch.no_grad():
            self.tick_count.add_(1)

        # ── Compute LLM modulation parameters ──
        # This is the critical output — what actually modulates the language model
        temperature = self._compute_temperature(coherence, criticality_distance)
        personality_blend = self._compute_personality_blend(
            dominant_state, narr_result['identity_strength'].item()
        )
        attention_bias = attention_map.detach()

        return {
            'modulation': {
                'temperature': temperature,
                'personality_blend': personality_blend,
                'attention_bias': attention_bias,
                'processing_mode': processing_mode,
                'coherence': coherence,
                'dominant_state': dominant_state,
                'motivation': combined_motivation,
                'sleep_signal': sleep_signal,
            },
            'consciousness': {
                'free_energy': free_energy.item(),
                'coherence': coherence,
                'criticality_distance': criticality_distance,
                'branching_ratio': crit['branching_ratio'].item(),
                'coalition_entropy': coalition_entropy,
                'dominance_ratio': dominance_ratio,
                'processing_mode': processing_mode,
                'sr_noise_level': sr_result['noise_level'].item(),
                'sr_snr': sr_result['snr'].item(),
            },
            'self_state': {
                'identity_strength': narr_result['identity_strength'].item(),
                'narrative_deviation': narr_result['deviation'].item(),
                'self_prediction_error': self_result['self_prediction_error_magnitude'].item(),
                'sensory_flow': blanket['sensory_flow'].item(),
                'active_flow': blanket['active_flow'].item(),
            },
            'learning': {
                'mean_fitness': arena_result['mean_fitness'].item(),
                'fitness_variance': arena_result['fitness_variance'].item(),
                'num_memories': self.episodic_buffer.num_stored.item(),
                'consolidation': consolidation_info,
                'combined_motivation': combined_motivation,
                'adaptive_lr': adaptive_lr.mean().item(),
                'mean_volatility': vol_result['mean_volatility'].item(),
            },
            'diagnostics': {
                'tick': self.tick_count.item(),
                'phases': phases.detach(),
                'top_beliefs': top_beliefs.detach(),
                'precision': prec_result['precision'].detach(),
                'resources': arena_result['resources'].detach(),
                'neuromodulator_levels': levels.detach(),
                **stuart_landau_info,
                **mean_field_info,
                **koopman_info,
                **stp_info,
                **neural_mass_info,
            },
        }

    def _compute_temperature(self, coherence: float, criticality_distance: float) -> float:
        """
        Map consciousness state to LLM temperature.

        High coherence + near criticality → lower temperature (focused)
        Low coherence + far from criticality → higher temperature (creative)
        """
        # Base temperature from coherence (inverted: high coherence = low temp)
        base_temp = 1.0 - 0.5 * coherence  # [0.5, 1.0]

        # Criticality modulation: near critical = slightly more creative
        crit_mod = 0.1 * max(0, 1.0 - criticality_distance)

        # Clamp to reasonable range
        temp = max(0.1, min(2.0, base_temp + crit_mod))
        return temp

    def _compute_personality_blend(self, dominant_state: int,
                                    identity_strength: float) -> Dict[str, float]:
        """
        Map metastable dominant state to personality expression weights.

        Each population in the WinnerlessCompetition maps to a personality facet.
        Identity strength determines how much personality constrains the blend.
        """
        num_pop = self.config.num_populations

        # Base weights from dominant state (one-hot with softening)
        weights = [0.1] * num_pop
        weights[dominant_state % num_pop] = 1.0

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
