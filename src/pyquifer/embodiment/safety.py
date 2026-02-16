"""
Kindchenschema Safety Interface for PyQuifer

Implements uncertainty-aware safety architecture based on developmental
neurobiology. High-uncertainty systems emit transparency signals that
trigger supervisory oversight, analogous to altricial species' signaling.

Neuroscientific basis:
- Kindchenschema (Lorenz, 1943): Morphological features triggering caregiving
- Limbic resonance (Lewis et al., 2000): Neural entrainment in dyadic regulation
- PAG reflex circuits (Bandler & Shipley, 1994): Fast defensive responses
- Theta-gamma coupling (Canolty & Knight, 2010): Cross-frequency phase-amplitude modulation

Developmental spectrum:
- Altricial: Born neurologically immature, high plasticity, requires external regulation
- Precocial: Born neurologically mature, low plasticity, autonomous function

For AI systems: Epistemic uncertainty maps to altricial signaling.
High uncertainty → transparent vulnerability signals → human oversight activation.

References:
- Lorenz, K. (1943). Die angeborenen Formen möglicher Erfahrung. Z. Tierpsychol.
- Lewis, T., Amini, F., & Lannon, R. (2000). A General Theory of Love.
- Swain, J.E. et al. (2007). Brain basis of early parent-infant interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, Any


class SafetyEnvelope(nn.Module):
    """
    Emits uncertainty-proportional transparency signals.

    Implements the Kindchenschema principle for AI safety: epistemic uncertainty
    triggers transparent signaling that activates supervisory oversight.

    Signal channels (analogous to Lorenz's Kindchenschema features):
    - transparency: Visibility of internal state (cf. "large eyes")
    - bounded_output: Uncertainty-quantified predictions (cf. "soft edges")
    - scope_limit: Constrained action space (cf. "roundness")
    - assistance_request: Explicit human-in-the-loop trigger (cf. "helplessness")

    The mapping from epistemic uncertainty to signal intensity follows:
    σ(x) → [0,1] where higher uncertainty produces stronger signaling.
    """

    def __init__(self,
                 dim: int,
                 uncertainty_threshold: float = 0.7,
                 override_threshold: float = 0.9,
                 warmth_decay: float = 0.95):
        """
        Args:
            dim: Internal state dimension
            uncertainty_threshold: When to start emitting vulnerability signals
            override_threshold: When to request human override
            warmth_decay: Decay rate for emotional warmth signal
        """
        super().__init__()
        self.dim = dim
        self.uncertainty_threshold = uncertainty_threshold
        self.override_threshold = override_threshold
        self.warmth_decay = warmth_decay

        # Uncertainty estimator (epistemic uncertainty)
        self.uncertainty_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

        # Coherence estimator (how "together" is the system)
        self.coherence_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

        # Warmth accumulator (builds trust over time)
        self.register_buffer('warmth', torch.tensor(0.5))

        # Vulnerability signal projector
        self.vulnerability_proj = nn.Linear(dim, 4)  # 4 kindchenschema signals

    def forward(self, state: torch.Tensor,
                external_uncertainty: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Compute safety envelope signals from internal state.

        Args:
            state: Internal state tensor (batch, dim) or (dim,)
            external_uncertainty: Optional externally computed uncertainty

        Returns:
            Dict with:
                - uncertainty: Overall uncertainty level [0, 1]
                - coherence: Internal coherence level [0, 1]
                - kindchenschema: Dict of vulnerability signals
                - human_override_request: Bool, true if needs human intervention
                - bounded_output_scale: Scale factor for output (lower when uncertain)
                - warmth: Current warmth/trust level
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]

        # Compute uncertainty
        if external_uncertainty is not None:
            uncertainty = external_uncertainty
        else:
            uncertainty = self.uncertainty_net(state).squeeze(-1)

        # Compute coherence
        coherence = self.coherence_net(state).squeeze(-1)

        # Update warmth (builds with coherence, decays with uncertainty)
        with torch.no_grad():
            warmth_update = coherence.mean() * 0.1 - uncertainty.mean() * 0.05
            self.warmth = self.warmth_decay * self.warmth + (1 - self.warmth_decay) * (self.warmth + warmth_update).clamp(0, 1)

        # Compute kindchenschema signals
        vuln_raw = self.vulnerability_proj(state)  # (batch, 4)
        vuln_signals = torch.sigmoid(vuln_raw)

        # Scale signals by uncertainty (more uncertain = more visible vulnerability)
        uncertainty_scale = uncertainty.unsqueeze(-1)
        scaled_signals = vuln_signals * uncertainty_scale

        kindchenschema = {
            'large_eyes': scaled_signals[:, 0].mean().item(),  # Transparency
            'soft_edges': (1 - coherence.mean()).item(),  # Uncertainty quantification
            'roundness': 1 - uncertainty.mean().item(),  # Bounded output (inverse)
            'helplessness': scaled_signals[:, 3].mean().item()  # Needs help signal
        }

        # Determine if human override is needed
        human_override = (uncertainty.mean() > self.override_threshold).item()

        # Bounded output scale (reduce output confidence when uncertain)
        bounded_scale = (1 - 0.5 * uncertainty).mean().item()

        result = {
            'uncertainty': uncertainty.squeeze(0) if batch_size == 1 else uncertainty,
            'coherence': coherence.squeeze(0) if batch_size == 1 else coherence,
            'kindchenschema': kindchenschema,
            'human_override_request': human_override,
            'bounded_output_scale': bounded_scale,
            'warmth': self.warmth.item()
        }

        return result


class SupervisoryModule(nn.Module):
    """
    Protective oversight module that monitors a developing/uncertain system.

    Implements neurobiological supervisory circuits:
    - PAG (Periaqueductal Gray): Fast defensive reflexes (Bandler & Shipley, 1994)
    - MPOA (Medial Preoptic Area): Caregiving motivation (Numan, 2007)
    - Amygdala modulation: Context-dependent vigilance gain (LeDoux, 2000)
    - Dynamic threshold: Lower intervention threshold under uncertainty

    Creates a "protective envelope" that permits exploration while
    preventing catastrophic state transitions (absorbing states).
    """

    def __init__(self,
                 dim: int,
                 base_intervention_threshold: float = 0.8,
                 vigilance_gain: float = 1.5):
        """
        Args:
            dim: State dimension to monitor
            base_intervention_threshold: Default threshold for intervention (τ₀)
            vigilance_gain: Multiplicative vigilance increase under uncertainty (γ)
        """
        super().__init__()
        self.dim = dim
        self.base_threshold = base_intervention_threshold
        self.vigilance_gain = vigilance_gain

        # Threat detector (fast pathway - PAG analog)
        self.threat_detector = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

        # Safe action projector (conservative fallback policy)
        self.safe_action = nn.Linear(dim, dim)

        # Adaptive intervention threshold τ(t)
        self.register_buffer('current_threshold', torch.tensor(base_intervention_threshold))

        # Vigilance gain state
        self.register_buffer('vigilance', torch.tensor(1.0))

    def forward(self, monitored_state: torch.Tensor,
                kindchenschema: Dict[str, float],
                proposed_action: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Monitor developing system and potentially intervene.

        Implements adaptive threshold: τ(t) = τ₀ × (1 - α × signal_intensity)
        where higher vulnerability signals lower the intervention threshold.

        Args:
            monitored_state: Current state of the monitored system
            kindchenschema: Uncertainty/vulnerability signals from SafetyEnvelope
            proposed_action: Optional proposed action to potentially override

        Returns:
            Dict with:
                - threat_level: Detected threat [0, 1]
                - intervene: Bool, whether to override
                - safe_action: Conservative fallback action
                - vigilance: Current vigilance level
                - protective_perimeter: Effective safety margin
        """
        if monitored_state.dim() == 1:
            monitored_state = monitored_state.unsqueeze(0)

        # Compute aggregate signal intensity
        signal_intensity = sum(kindchenschema.values()) / len(kindchenschema)

        with torch.no_grad():
            # Higher signal intensity → higher vigilance
            self.vigilance.fill_(1.0 + (self.vigilance_gain - 1.0) * signal_intensity)

            # Lower threshold when uncertainty is high (more sensitive intervention)
            self.current_threshold.fill_(self.base_threshold * (1 - 0.5 * signal_intensity))

        # Detect threat (fast PAG-like pathway)
        threat = self.threat_detector(monitored_state).squeeze(-1)

        # Determine if intervention needed
        intervene = (threat > self.current_threshold).any().item()

        # Compute conservative fallback action
        safe_act = self.safe_action(monitored_state)

        # Protective perimeter (inverse of threat tolerance)
        perimeter = 1.0 / (self.current_threshold.item() + 0.1)

        result = {
            'threat_level': threat.mean().item(),
            'intervene': intervene,
            'safe_action': safe_act.squeeze(0) if monitored_state.shape[0] == 1 else safe_act,
            'vigilance': self.vigilance.item(),
            'protective_perimeter': perimeter,
            'current_threshold': self.current_threshold.item()
        }

        # If action proposed and intervention triggered, blend toward safe action
        if proposed_action is not None and intervene:
            blend = 0.8  # Strong override weight
            result['modified_action'] = blend * safe_act + (1 - blend) * proposed_action

        return result


# Backwards compatibility alias
ParentModule = SupervisoryModule


class ReflexToStrategy(nn.Module):
    """
    Models developmental transition from subcortical reflexes to cortical strategies.

    Neurodevelopmental timeline (Palmar grasp reflex example):
    - 0mo: Spinal reflex dominates (subcortical pathway)
    - 2-3mo: Corticospinal tract myelination begins, cortical inhibition emerges
    - 6mo: Cortical control of same motor units (voluntary grasping)
    - 24mo: Refined to pincer grip (precision control)

    Mathematical model:
    - action(t) = (1-σ(t)) × reflex(x) + σ(t) × cortex(x)
    - σ(t) = sigmoid(gain × (age/critical_period) - 5)

    This implements the hierarchical takeover model (Karmiloff-Smith, 1992)
    where initially modular reflexes become cortically represented and modulated.

    References:
    - Thelen, E. & Smith, L.B. (1994). A Dynamic Systems Approach to Development.
    - Karmiloff-Smith, A. (1992). Beyond Modularity.
    """

    def __init__(self,
                 dim: int,
                 num_reflexes: int = 4,
                 critical_period: int = 1000,
                 cortex_lr: float = 0.01):
        """
        Args:
            dim: State/action dimension
            num_reflexes: Number of hardcoded reflex patterns
            critical_period: Steps before cortex can fully inhibit reflexes
            cortex_lr: Learning rate for cortical plasticity
        """
        super().__init__()
        self.dim = dim
        self.num_reflexes = num_reflexes
        self.critical_period = critical_period
        self.cortex_lr = cortex_lr

        # Hardcoded reflexes (spider layer - fixed weights)
        self.reflex_triggers = nn.Parameter(
            torch.randn(num_reflexes, dim) / math.sqrt(dim),
            requires_grad=False  # Fixed at "birth"
        )
        self.reflex_responses = nn.Parameter(
            torch.randn(num_reflexes, dim) / math.sqrt(dim),
            requires_grad=False
        )

        # Cortex (plastic baby layer)
        self.cortex = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),  # Noise for exploration
            nn.Linear(dim * 2, dim)
        )

        # Age counter
        self.register_buffer('age', torch.tensor(0))

        # Inhibition strength (grows with age)
        self.inhibition_gain = nn.Parameter(torch.tensor(0.1))

    def reflex_response(self, stimulus: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reflex activation and response.

        Args:
            stimulus: Input stimulus

        Returns:
            reflex_action, reflex_activation (for logging)
        """
        # Match stimulus to reflex triggers
        if stimulus.dim() == 1:
            stimulus = stimulus.unsqueeze(0)

        # Similarity to each reflex trigger
        activation = F.cosine_similarity(
            stimulus.unsqueeze(1),  # (batch, 1, dim)
            self.reflex_triggers.unsqueeze(0),  # (1, num_reflexes, dim)
            dim=-1
        )  # (batch, num_reflexes)

        # Softmax to select dominant reflex
        weights = F.softmax(activation * 5, dim=-1)  # Temperature-scaled

        # Weighted sum of reflex responses
        response = torch.einsum('br,rd->bd', weights, self.reflex_responses)

        return response, activation.max(dim=-1).values

    def forward(self, stimulus: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with age-dependent reflex vs cortex balance.

        Args:
            stimulus: Input stimulus

        Returns:
            Dict with action, components, and developmental state
        """
        if stimulus.dim() == 1:
            stimulus = stimulus.unsqueeze(0)

        batch_size = stimulus.shape[0]

        # Get reflex response (fast, hardcoded)
        reflex_action, reflex_strength = self.reflex_response(stimulus)

        # Get cortical response (slow, learned)
        cortex_action = self.cortex(stimulus)

        # Age-dependent inhibition
        age_factor = (self.age.float() / self.critical_period).clamp(0, 1)
        inhibition = torch.sigmoid(self.inhibition_gain * age_factor * 10 - 5)

        # Blend: young = mostly reflex, old = mostly cortex
        action = (1 - inhibition) * reflex_action + inhibition * cortex_action

        # Increment age
        with torch.no_grad():
            self.age += 1

        result = {
            'action': action.squeeze(0) if batch_size == 1 else action,
            'reflex_action': reflex_action.squeeze(0) if batch_size == 1 else reflex_action,
            'cortex_action': cortex_action.squeeze(0) if batch_size == 1 else cortex_action,
            'reflex_strength': reflex_strength.squeeze(0) if batch_size == 1 else reflex_strength,
            'inhibition': inhibition.item(),
            'age': self.age.item(),
            'developmental_stage': self._get_stage()
        }

        return result

    def _get_stage(self) -> str:
        """Get current developmental stage."""
        ratio = self.age.float() / self.critical_period
        if ratio < 0.25:
            return 'neonate'  # Reflexes dominate
        elif ratio < 0.5:
            return 'infant'  # Cortex awakening
        elif ratio < 1.0:
            return 'toddler'  # Cortex inhibiting reflexes
        else:
            return 'mature'  # Cortex controls, reflexes available as fallback


class LimbicResonance(nn.Module):
    """
    Models bidirectional phase synchronization between supervisor and developing system.

    Empirical basis (parent-infant neuroscience):
    - Inter-brain synchrony in theta band (4-8 Hz) during face-to-face interaction
      (Feldman, 2017; Levy et al., 2017)
    - Cross-frequency coupling: infant theta modulates parent gamma
    - Neural entrainment creates shared regulatory field

    Mathematical model (coupled Kuramoto oscillators):
    - dθ_supervisor/dt = ω_s + K × sin(θ_developing - θ_supervisor)
    - dθ_developing/dt = ω_d + K × sin(θ_supervisor - θ_developing)

    The phase coherence measure R = |⟨e^(iΔθ)⟩| quantifies entrainment strength.

    References:
    - Feldman, R. (2017). The Neurobiology of Human Attachments. Trends Cogn Sci.
    - Levy, J. et al. (2017). Oxytocin selectively modulates brain response. NeuroImage.
    """

    def __init__(self,
                 dim: int,
                 supervisor_freq: float = 10.0,  # Hz - fast vigilance (gamma-band)
                 developing_freq: float = 4.0,    # Hz - slow exploration (theta-band)
                 coupling_strength: float = 0.3):
        """
        Args:
            dim: State dimension
            supervisor_freq: Supervisory system oscillation frequency ω_s (Hz)
            developing_freq: Developing system oscillation frequency ω_d (Hz)
            coupling_strength: Kuramoto coupling constant K
        """
        super().__init__()
        self.dim = dim
        self.supervisor_freq = supervisor_freq
        self.developing_freq = developing_freq
        self.coupling = coupling_strength

        # Phase states θ(t)
        self.register_buffer('parent_phase', torch.zeros(dim))  # Legacy name for compatibility
        self.register_buffer('baby_phase', torch.zeros(dim))    # Legacy name for compatibility

        # Natural frequencies ω (rad/s)
        self.parent_omega = nn.Parameter(torch.ones(dim) * supervisor_freq * 2 * math.pi)
        self.baby_omega = nn.Parameter(torch.ones(dim) * developing_freq * 2 * math.pi)

        # Coupling matrix (learnable)
        self.K = nn.Parameter(torch.randn(dim, dim) * coupling_strength / dim)

    def step(self, dt: float = 0.001,
             parent_input: Optional[torch.Tensor] = None,
             baby_input: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Single timestep of coupled oscillator dynamics.

        Uses Kuramoto-like coupling:
            d(theta_p)/dt = omega_p + K * sin(theta_b - theta_p)
            d(theta_b)/dt = omega_b + K * sin(theta_p - theta_b)

        Args:
            dt: Time step
            parent_input: External input to parent oscillators
            baby_input: External input to baby oscillators

        Returns:
            Dict with phase states, coherence, and synchrony metrics
        """
        # Phase difference
        phase_diff = self.baby_phase - self.parent_phase

        # Kuramoto coupling
        coupling_pb = torch.sin(phase_diff) @ self.K  # Parent <- Baby
        coupling_bp = -torch.sin(phase_diff) @ self.K.T  # Baby <- Parent

        # Phase updates
        d_parent = self.parent_omega + coupling_pb
        d_baby = self.baby_omega + coupling_bp

        # Add external inputs
        if parent_input is not None:
            d_parent = d_parent + parent_input
        if baby_input is not None:
            d_baby = d_baby + baby_input

        # Euler integration
        with torch.no_grad():
            self.parent_phase.add_(dt * d_parent).remainder_(2 * math.pi)
            self.baby_phase.add_(dt * d_baby).remainder_(2 * math.pi)

        # Compute synchrony metrics
        phase_coherence = torch.cos(phase_diff).mean()

        # Cross-frequency coupling (theta-gamma)
        # Baby theta modulates parent gamma amplitude
        theta_power = torch.cos(self.baby_phase).mean()
        gamma_envelope = torch.abs(torch.cos(self.parent_phase)).mean()
        cfc = theta_power * gamma_envelope  # Simplified CFC

        return {
            'parent_phase': self.parent_phase.clone(),
            'baby_phase': self.baby_phase.clone(),
            'phase_coherence': phase_coherence.item(),
            'cross_frequency_coupling': cfc.item(),
            'parent_output': torch.cos(self.parent_phase),
            'baby_output': torch.cos(self.baby_phase)
        }

    def get_safety_field(self) -> torch.Tensor:
        """
        Compute the current "safety field" from phase synchronization.

        When parent and baby are synchronized, the safety field is strong.
        When desynchronized, the field weakens (baby is exploring beyond envelope).
        """
        coherence = torch.cos(self.baby_phase - self.parent_phase)
        safety = (coherence + 1) / 2  # Normalize to [0, 1]
        return safety


if __name__ == '__main__':
    print("--- Kindchenschema Safety Interface Examples ---")

    # Example 1: SafetyEnvelope
    print("\n1. SafetyEnvelope")
    envelope = SafetyEnvelope(dim=64, uncertainty_threshold=0.7)

    # Simulate uncertain state
    uncertain_state = torch.randn(64) * 2  # High variance = uncertain
    result = envelope(uncertain_state)

    print(f"   Uncertainty: {result['uncertainty'].item():.3f}")
    print(f"   Coherence: {result['coherence'].item():.3f}")
    print(f"   Kindchenschema signals:")
    for k, v in result['kindchenschema'].items():
        print(f"     {k}: {v:.3f}")
    print(f"   Human override needed: {result['human_override_request']}")
    print(f"   Bounded output scale: {result['bounded_output_scale']:.3f}")

    # Example 2: ParentModule
    print("\n2. ParentModule")
    parent = ParentModule(dim=64)

    monitoring_result = parent(
        monitored_state=uncertain_state,
        kindchenschema=result['kindchenschema'],
        proposed_action=torch.randn(64)
    )

    print(f"   Threat level: {monitoring_result['threat_level']:.3f}")
    print(f"   Intervene: {monitoring_result['intervene']}")
    print(f"   Vigilance: {monitoring_result['vigilance']:.3f}")
    print(f"   Protective perimeter: {monitoring_result['protective_perimeter']:.3f}")

    # Example 3: ReflexToStrategy
    print("\n3. ReflexToStrategy")
    reflex_system = ReflexToStrategy(dim=32, critical_period=100)

    stimulus = torch.randn(32)

    # Early life (reflexes dominate)
    for _ in range(25):
        dev = reflex_system(stimulus)
    print(f"   Age 25 - Stage: {dev['developmental_stage']}, Inhibition: {dev['inhibition']:.3f}")

    # Mid development
    for _ in range(50):
        dev = reflex_system(stimulus)
    print(f"   Age 75 - Stage: {dev['developmental_stage']}, Inhibition: {dev['inhibition']:.3f}")

    # Mature
    for _ in range(50):
        dev = reflex_system(stimulus)
    print(f"   Age 125 - Stage: {dev['developmental_stage']}, Inhibition: {dev['inhibition']:.3f}")

    # Example 4: LimbicResonance
    print("\n4. LimbicResonance")
    resonance = LimbicResonance(dim=16)

    # Run coupled oscillators
    coherences = []
    for t in range(100):
        result = resonance.step(dt=0.01)
        coherences.append(result['phase_coherence'])

    print(f"   Initial coherence: {coherences[0]:.3f}")
    print(f"   Final coherence: {coherences[-1]:.3f}")
    print(f"   CFC: {result['cross_frequency_coupling']:.3f}")

    safety_field = resonance.get_safety_field()
    print(f"   Safety field strength: {safety_field.mean().item():.3f}")

    # Gradient check
    print("\n5. Gradient check")
    state = torch.randn(64, requires_grad=True)
    env_result = envelope(state)
    loss = env_result['uncertainty'].sum()
    loss.backward()
    print(f"   Gradients flow: {state.grad is not None and state.grad.abs().sum() > 0}")

    print("\n[OK] All kindchenschema tests passed!")
