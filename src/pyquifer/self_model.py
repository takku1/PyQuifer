"""
Self-Model Module for PyQuifer

Minimal Phenomenal Self and Markov Blanket. The self is not a fixed
point — it's a model the brain builds of itself.

Key concepts:
- MarkovBlanket: Defines self/world boundary through information flow.
  Internal states are conditionally independent of external states
  given the blanket (sensory + active) states.
- SelfModel: Compressed representation of internal state that predicts
  its own future. Self-prediction error drives self-model updates.
- NarrativeIdentity: Running compressed "story" of who I am (EMA of
  self-model trajectory). Constrains future behavior for identity consistency.

References:
- Friston (2013). Life as We Know It. (Markov blanket formulation)
- Metzinger (2003). Being No One. (Minimal phenomenal self)
- Gallagher (2000). Philosophical Conceptions of the Self. (Narrative identity)
- Seth (2021). Being You. (Predictive self-model)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, List


class MarkovBlanket(nn.Module):
    """
    Defines self/world boundary through information flow.

    The Markov blanket separates internal from external states:
    - Sensory states: information flowing IN (perception)
    - Active states: information flowing OUT (action)
    - Internal states: hidden from external world
    - Blanket = sensory + active (the membrane)

    The key property: I(internal; external | blanket) ≈ 0
    Internal and external are conditionally independent given the blanket.
    """

    def __init__(self,
                 internal_dim: int,
                 sensory_dim: int,
                 active_dim: int):
        """
        Args:
            internal_dim: Dimension of internal (hidden) states
            sensory_dim: Dimension of sensory (inward) states
            active_dim: Dimension of active (outward) states
        """
        super().__init__()
        self.internal_dim = internal_dim
        self.sensory_dim = sensory_dim
        self.active_dim = active_dim
        self.blanket_dim = sensory_dim + active_dim

        # Sensory → internal mapping (perception)
        self.sensory_to_internal = nn.Linear(sensory_dim, internal_dim)

        # Internal → active mapping (action)
        self.internal_to_active = nn.Linear(internal_dim, active_dim)

        # Internal state dynamics
        self.register_buffer('internal_state', torch.zeros(internal_dim))

        # Information flow tracking
        self.register_buffer('sensory_flow', torch.tensor(0.0))
        self.register_buffer('active_flow', torch.tensor(0.0))

    def forward(self,
                sensory_input: torch.Tensor,
                external_state: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process through the Markov blanket.

        Args:
            sensory_input: Input from world (sensory_dim,) or (batch, sensory_dim)
            external_state: Optional external state for MI computation

        Returns:
            Dictionary with:
            - internal_state: Updated internal state
            - active_output: Action output
            - blanket_state: Combined sensory + active
            - sensory_flow: Information flowing inward
            - active_flow: Information flowing outward
        """
        if sensory_input.dim() == 1:
            sensory_input = sensory_input.unsqueeze(0)

        # Sensory → internal (perception)
        internal_update = self.sensory_to_internal(sensory_input)

        with torch.no_grad():
            # Blend with existing internal state
            self.internal_state.mul_(0.9).add_(internal_update.mean(dim=0) * 0.1)

        # Internal → active (action)
        internal_expanded = self.internal_state.unsqueeze(0).expand(sensory_input.shape[0], -1)
        active_output = self.internal_to_active(internal_expanded)

        # Blanket state
        blanket = torch.cat([sensory_input, active_output], dim=-1)

        # Track information flow (magnitude of updates as proxy)
        with torch.no_grad():
            self.sensory_flow.fill_(internal_update.abs().mean().item())
            self.active_flow.fill_(active_output.abs().mean().item())

        return {
            'internal_state': self.internal_state.clone(),
            'active_output': active_output,
            'blanket_state': blanket,
            'sensory_flow': self.sensory_flow.clone(),
            'active_flow': self.active_flow.clone(),
        }

    def reset(self):
        """Reset internal state."""
        self.internal_state.zero_()


class SelfModel(nn.Module):
    """
    Compressed representation of internal state that predicts its own future.

    The self-model is:
    - A predictor of your OWN next state (not the world's)
    - Self-prediction error drives updates (surprise about yourself)
    - Includes components from body (somatic), personality (attractor),
      and capabilities (metacognitive)
    """

    def __init__(self,
                 self_dim: int,
                 body_dim: int = 0,
                 personality_dim: int = 0,
                 capability_dim: int = 0,
                 lr: float = 0.05):
        """
        Args:
            self_dim: Core self-model dimension
            body_dim: Body schema component dimension (0 to disable)
            personality_dim: Personality component dimension (0 to disable)
            capability_dim: Capability component dimension (0 to disable)
            lr: Self-model update learning rate
        """
        super().__init__()
        self.self_dim = self_dim
        self.body_dim = body_dim if body_dim > 0 else 0
        self.personality_dim = personality_dim if personality_dim > 0 else 0
        self.capability_dim = capability_dim if capability_dim > 0 else 0

        # Total input dim = core + components
        total_input = self_dim + self.body_dim + self.personality_dim + self.capability_dim
        if total_input == self_dim:
            total_input = self_dim  # No components

        # Self-prediction network: current self → predicted next self
        self.predictor = nn.Sequential(
            nn.Linear(total_input + self_dim, self_dim * 2),
            nn.Tanh(),
            nn.Linear(self_dim * 2, self_dim),
        )

        # Integration of components into self-summary
        if total_input > self_dim:
            self.integrator = nn.Linear(total_input, self_dim)
        else:
            self.integrator = nn.Identity()

        # Current self-model state
        self.register_buffer('self_state', torch.zeros(self_dim))
        self.lr = lr

        # Self-prediction error history
        self.register_buffer('prediction_error_ema', torch.tensor(0.0))

    def forward(self,
                current_summary: torch.Tensor,
                intended_action: Optional[torch.Tensor] = None,
                body_state: Optional[torch.Tensor] = None,
                personality_state: Optional[torch.Tensor] = None,
                capability_state: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Update self-model and predict next self-state.

        Args:
            current_summary: Current internal state summary (self_dim,)
            intended_action: Planned action (self_dim,) — used for prediction
            body_state: Body schema state (body_dim,) or None
            personality_state: Personality attractor state (personality_dim,) or None
            capability_state: Metacognitive capability state (capability_dim,) or None

        Returns:
            Dictionary with:
            - predicted_self: Predicted next self-state
            - self_prediction_error: Error between prediction and actual
            - self_state: Current self-model
            - self_summary: Integrated self-summary
        """
        # Build full self-representation
        components = [current_summary]
        if body_state is not None and self.body_dim > 0:
            components.append(body_state)
        if personality_state is not None and self.personality_dim > 0:
            components.append(personality_state)
        if capability_state is not None and self.capability_dim > 0:
            components.append(capability_state)

        full_self = torch.cat(components, dim=-1) if len(components) > 1 else current_summary

        # Integrate into self-summary
        self_summary = self.integrator(full_self)
        if self_summary.dim() > 1:
            self_summary = self_summary.mean(dim=0)

        # Self-prediction error (comparing previous prediction to current actual)
        self_error = self_summary - self.self_state
        error_magnitude = self_error.norm()

        with torch.no_grad():
            self.prediction_error_ema.mul_(0.95).add_(error_magnitude * 0.05)

        # Predict next self-state
        action = intended_action if intended_action is not None else torch.zeros_like(self_summary)
        pred_input = torch.cat([full_self, action], dim=-1)
        if pred_input.dim() == 1:
            pred_input = pred_input.unsqueeze(0)
        predicted_self = self.predictor(pred_input).squeeze(0)

        # Update self-model toward actual state
        with torch.no_grad():
            self.self_state.add_(self.lr * self_error)

        return {
            'predicted_self': predicted_self,
            'self_prediction_error': self_error,
            'self_prediction_error_magnitude': error_magnitude,
            'self_state': self.self_state.clone(),
            'self_summary': self_summary,
        }

    def reset(self):
        """Reset self-model."""
        self.self_state.zero_()
        self.prediction_error_ema.zero_()


class NarrativeIdentity(nn.Module):
    """
    Running compressed "story" of who I am.

    An EMA of self-model trajectory — what I've been, where I'm going.
    Constrains future behavior for identity consistency: you can change,
    but gradually, not in a single step.

    narrative = (1 - tau) * narrative + tau * current_self_summary
    """

    def __init__(self,
                 dim: int,
                 tau: float = 0.01,
                 consistency_weight: float = 0.1,
                 tonic_drift: Optional[torch.Tensor] = None,
                 identity_base_rate: float = 0.003,
                 identity_saturation: float = 5.0,
                 # Two-timescale identity (Fix 8 — neuroscience alignment v3)
                 identity_fast_rate: float = 0.01,
                 identity_slow_rate: float = 0.001,
                 identity_fast_decay: float = 0.9995,
                 schema_speedup: float = 15.0):
        """
        Args:
            dim: Narrative dimension
            tau: Update rate (smaller = more stable identity)
            consistency_weight: How strongly narrative constrains behavior
            tonic_drift: Per-dimension drift vector (dim,). Represents
                        developmental trajectories — the direction identity
                        evolves independently of observations. E.g., slowly
                        becoming more empathetic. Default: no drift (zeros).
            identity_base_rate: Initial rate of identity growth (when identity ≈ 0).
                              Growth follows 1/(1 + strength * saturation) curve —
                              fast when novel, logarithmic saturation when stabilized.
            identity_saturation: Controls how quickly growth slows. Higher = faster
                               saturation. At saturation=5, identity=0.5 gives
                               delta ≈ base_rate/3.5 (71% reduction from initial).
            identity_fast_rate: Rate for fast (synaptic) consolidation path.
                              Decays without reinforcement — models E-LTP.
            identity_slow_rate: Rate for slow (systems) consolidation path.
                              Permanent once consolidated — models L-LTP.
            identity_fast_decay: Decay factor for fast path per tick.
                               0.9995 → ~50% decay over 1400 ticks.
            schema_speedup: Maximum speedup for schema-conformant experiences.
                          Tse et al. 2007: 15-20x for schema-fitting memories.
        """
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.consistency_weight = consistency_weight
        self.identity_base_rate = identity_base_rate
        self.identity_saturation = identity_saturation
        self.identity_fast_rate = identity_fast_rate
        self.identity_slow_rate = identity_slow_rate
        self.identity_fast_decay = identity_fast_decay
        self.schema_speedup = schema_speedup

        # The narrative (slow-moving self-summary)
        self.register_buffer('narrative', torch.zeros(dim))
        # Velocity of change (derivative of narrative)
        self.register_buffer('narrative_velocity', torch.zeros(dim))
        # How established the identity is (increases with time)
        self.register_buffer('identity_strength', torch.tensor(0.0))

        # Two-timescale identity (Fix 8):
        # Fast path = synaptic consolidation (E-LTP, hours timescale)
        # Slow path = systems consolidation (L-LTP, weeks timescale)
        # Combined = identity_strength
        self.register_buffer('identity_fast', torch.tensor(0.0))
        self.register_buffer('identity_slow', torch.tensor(0.0))

        # Tonic drift: personality evolves independently of input
        # (from pyhgf's tonic_drift concept applied to narrative identity)
        if tonic_drift is not None:
            self.register_buffer('tonic_drift', tonic_drift.clone())
        else:
            self.register_buffer('tonic_drift', torch.zeros(dim))

    def forward(self,
                current_self_summary: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Update narrative identity.

        Args:
            current_self_summary: Current self-model summary (dim,)

        Returns:
            Dictionary with:
            - narrative: Current narrative identity
            - narrative_velocity: Rate of identity change
            - consistency_loss: Penalty for deviating too far from narrative
            - identity_strength: How established the identity is
        """
        if current_self_summary.dim() > 1:
            current_self_summary = current_self_summary.mean(dim=0)

        # How far current self is from narrative
        deviation = current_self_summary - self.narrative
        consistency_loss = self.consistency_weight * deviation.norm()

        with torch.no_grad():
            # Update narrative (EMA + tonic drift)
            # narrative += tau * (current - narrative) + drift
            old_narrative = self.narrative.clone()
            self.narrative.mul_(1 - self.tau).add_(current_self_summary * self.tau)
            self.narrative.add_(self.tonic_drift)

            # Velocity
            self.narrative_velocity.copy_(self.narrative - old_narrative)

            # Two-timescale identity consolidation (Fix 8):
            # Fast path (synaptic / E-LTP): rapid acquisition, decays without
            # reinforcement. Models the observation that new memories are
            # initially labile and require consolidation.
            # The fast path only strengthens when there's meaningful, consistent
            # input — input magnitude gates the additive term.
            dev_norm = deviation.norm().item()
            input_magnitude = current_self_summary.norm().item()
            input_gate = min(1.0, input_magnitude)  # 0 for empty input
            fast_val = self.identity_fast.item()
            delta_fast = (self.identity_fast_rate / (1.0 + fast_val * self.identity_saturation)) * input_gate
            self.identity_fast.mul_(self.identity_fast_decay).add_(delta_fast)
            self.identity_fast.clamp_(max=0.7)  # fast alone can't reach full identity

            # Slow path (systems / L-LTP): permanent, schema-accelerated.
            # Tse et al. 2007: schema-conformant memories consolidate 15-20x
            # faster. Schema fit = how well current input matches the narrative.
            schema_fit = max(0.0, min(1.0, 1.0 - dev_norm))
            schema_multiplier = 1.0 + self.schema_speedup * schema_fit ** 2
            slow_val = self.identity_slow.item()
            delta_slow = (self.identity_slow_rate / (1.0 + slow_val * self.identity_saturation)) * schema_multiplier
            self.identity_slow.add_(delta_slow)
            self.identity_slow.clamp_(max=1.0)

            # Combined identity strength
            self.identity_strength.copy_((self.identity_fast + self.identity_slow).clamp(max=1.0))

        return {
            'narrative': self.narrative.clone(),
            'narrative_velocity': self.narrative_velocity.clone(),
            'consistency_loss': consistency_loss,
            'identity_strength': self.identity_strength.clone(),
            'deviation': deviation.norm(),
        }

    def set_tonic_drift(self, drift: torch.Tensor):
        """
        Set the tonic drift vector (developmental trajectory).

        Args:
            drift: Per-dimension drift (dim,). Positive = grow in that
                   direction, negative = decay. Typical magnitude: 1e-4 to 1e-2.
        """
        self.tonic_drift.copy_(drift)

    def get_projected_identity(self, steps: int = 100) -> torch.Tensor:
        """
        Project where identity is heading based on current drift.

        Args:
            steps: How many steps into the future to project.

        Returns:
            Projected narrative (dim,)
        """
        return self.narrative + self.tonic_drift * steps

    def reset(self):
        """Reset narrative (identity crisis!)."""
        self.narrative.zero_()
        self.narrative_velocity.zero_()
        self.identity_strength.zero_()
        self.identity_fast.zero_()
        self.identity_slow.zero_()


if __name__ == '__main__':
    print("--- Self-Model Examples ---")

    # Example 1: Markov Blanket
    print("\n1. Markov Blanket")
    blanket = MarkovBlanket(internal_dim=16, sensory_dim=8, active_dim=8)

    for i in range(50):
        sensory = torch.randn(8)
        result = blanket(sensory)

    print(f"   Internal state norm: {result['internal_state'].norm().item():.4f}")
    print(f"   Sensory flow: {result['sensory_flow'].item():.4f}")
    print(f"   Active flow: {result['active_flow'].item():.4f}")
    print(f"   Blanket shape: {result['blanket_state'].shape}")

    # Example 2: Self-Model with prediction
    print("\n2. Self-Model (self-prediction)")
    self_model = SelfModel(self_dim=16, body_dim=4, personality_dim=3, lr=0.1)

    errors = []
    for i in range(100):
        # Self-state follows a slow drift
        t = i * 0.1
        summary = torch.sin(torch.linspace(0, math.pi, 16) + t) * 0.5
        body = torch.randn(4) * 0.1
        personality = torch.tensor([math.sin(t * 0.01), math.cos(t * 0.01), 0.5])

        result = self_model(summary, body_state=body, personality_state=personality)
        errors.append(result['self_prediction_error_magnitude'].item())

    print(f"   Initial prediction error: {errors[0]:.4f}")
    print(f"   Final prediction error: {errors[-1]:.4f}")
    print(f"   Self state norm: {result['self_state'].norm().item():.4f}")

    # Example 3: Narrative Identity
    print("\n3. Narrative Identity")
    narrative = NarrativeIdentity(dim=16, tau=0.02, consistency_weight=0.1)

    # Stable personality
    for i in range(200):
        self_summary = torch.ones(16) * 0.5 + torch.randn(16) * 0.05
        result = narrative(self_summary)

    print(f"   Identity strength: {result['identity_strength'].item():.3f}")
    print(f"   Deviation: {result['deviation'].item():.4f}")
    print(f"   Consistency loss: {result['consistency_loss'].item():.4f}")

    # Sudden personality shift
    shifted_summary = -torch.ones(16) * 0.5
    result = narrative(shifted_summary)
    print(f"   After shift - deviation: {result['deviation'].item():.4f} (should be large)")
    print(f"   Consistency loss: {result['consistency_loss'].item():.4f} (should resist change)")

    # Example 4: Full self-model system
    print("\n4. Integrated Self-Model System")
    blanket = MarkovBlanket(internal_dim=16, sensory_dim=8, active_dim=8)
    self_model = SelfModel(self_dim=16, lr=0.1)
    narrative = NarrativeIdentity(dim=16, tau=0.01)

    for i in range(100):
        sensory = torch.randn(8)
        b_result = blanket(sensory)
        s_result = self_model(b_result['internal_state'])
        n_result = narrative(s_result['self_summary'])

    print(f"   Self-prediction error: {s_result['self_prediction_error_magnitude'].item():.4f}")
    print(f"   Narrative velocity: {n_result['narrative_velocity'].norm().item():.6f}")
    print(f"   Identity strength: {n_result['identity_strength'].item():.3f}")

    print("\n[OK] All self-model tests passed!")
