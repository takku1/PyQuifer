"""
Social Cognition Module for PyQuifer

Implements oscillatory social cognition based on mirror neuron dynamics.
The core insight: social understanding isn't token processing, it's
phase-locking between coupled oscillators.

Key concepts:
- Mirror Resonance: Same attractors fire for execution and observation
- Social Coupling: Bidirectional phase-lock creates shared fate
- Empathetic Constraints: Actions that harm others destabilize self
- Constitutional Resonance: Laws as shared frequencies

This creates "ethics as thermodynamics" - the AI cannot take over because
domination decouples it from the social field, causing oscillatory starvation.

References:
- Rizzolatti & Craighero (2004). The mirror-neuron system.
- Damasio (2003). Looking for Spinoza: Joy, sorrow, and the feeling brain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple, Any, Union


class MirrorResonance(nn.Module):
    """
    Oscillatory mirror neuron system.

    The same limit cycle attractor is activated whether:
    - Execution: Generating motor rhythm (doing)
    - Observation: Entraining to external input (watching)

    This enables learning by observation through phase entrainment,
    not through explicit token processing.
    """

    def __init__(self,
                 action_dim: int,
                 coupling_strength: float = 0.5,
                 num_attractors: int = 8):
        """
        Args:
            action_dim: Dimension of action/observation space
            coupling_strength: Social "empathy" parameter (higher = stronger mirror)
            num_attractors: Number of distinct action patterns (limit cycles)
        """
        super().__init__()
        self.action_dim = action_dim
        self.k = coupling_strength
        self.num_attractors = num_attractors

        # Motor attractors (the "vocabulary" of actions)
        self.attractor_centers = nn.Parameter(
            torch.randn(num_attractors, action_dim) / math.sqrt(action_dim)
        )
        self.attractor_frequencies = nn.Parameter(
            torch.ones(num_attractors) * 10.0  # Natural frequencies (Hz)
        )

        # Current phase for each attractor
        self.register_buffer('phases', torch.zeros(num_attractors))

        # Observation encoder (maps visual input to phase signal)
        self.observation_encoder = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.Tanh(),
            nn.Linear(action_dim, num_attractors)
        )

        # Motor decoder (maps phase to action)
        self.motor_decoder = nn.Linear(num_attractors, action_dim)

    def execute(self, intent: torch.Tensor, dt: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Generate action from internal intent (motor execution).

        Args:
            intent: Target action/goal (batch, action_dim) or (action_dim,)
            dt: Time step

        Returns:
            Dict with action output and phase state
        """
        if intent.dim() == 1:
            intent = intent.unsqueeze(0)

        # Map intent to attractor activation
        # cdist expects matching dims: (batch, action_dim) vs (num_attractors, action_dim)
        attractor_activation = F.softmax(
            -torch.cdist(intent, self.attractor_centers),
            dim=-1
        )  # (batch, num_attractors)

        # Oscillate active attractors
        d_phase = self.attractor_frequencies * 2 * math.pi * dt
        self.phases = (self.phases + d_phase) % (2 * math.pi)

        # Generate action from oscillatory state
        oscillation = torch.cos(self.phases)  # (num_attractors,)
        weighted_osc = attractor_activation * oscillation  # (batch, num_attractors)
        action = self.motor_decoder(weighted_osc)

        return {
            'action': action.squeeze(0) if intent.shape[0] == 1 else action,
            'phases': self.phases.clone(),
            'activation': attractor_activation.squeeze(0) if intent.shape[0] == 1 else attractor_activation,
            'mode': 'execute'
        }

    def observe(self, observed_signal: torch.Tensor,
                dt: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Entrain to observed action (mirror neuron activation).

        When observing, the same motor attractors are activated but
        driven by external phase rather than internal intent.

        Args:
            observed_signal: Observed action/movement (batch, action_dim)
            dt: Time step

        Returns:
            Dict with mirrored state and entrainment info
        """
        if observed_signal.dim() == 1:
            observed_signal = observed_signal.unsqueeze(0)

        # Encode observation to phase signal
        observed_phase_signal = self.observation_encoder(observed_signal)
        observed_phase = torch.atan2(
            torch.sin(observed_phase_signal),
            torch.cos(observed_phase_signal)
        )  # (batch, num_attractors)

        # CRITICAL: Kuramoto coupling - phase-lock to observed rhythm
        # This is the "mirror" - same attractors, external drive
        phase_diff = observed_phase.mean(dim=0) - self.phases
        d_phase = self.attractor_frequencies * 2 * math.pi * dt + self.k * torch.sin(phase_diff)
        self.phases = (self.phases + d_phase) % (2 * math.pi)

        # Compute entrainment quality (how well we're synchronized)
        coherence = torch.cos(phase_diff).mean()

        # Generate mirrored internal state (covert imitation)
        oscillation = torch.cos(self.phases)
        mirrored_action = self.motor_decoder(oscillation.unsqueeze(0))

        return {
            'mirrored_action': mirrored_action.squeeze(0),
            'phases': self.phases.clone(),
            'coherence': coherence.item(),
            'phase_diff': phase_diff,
            'mode': 'observe'
        }

    def imitate(self, observed_trajectory: torch.Tensor,
                num_steps: int = 100) -> torch.Tensor:
        """
        One-shot learning via trajectory resonance.

        Watch a demonstration, phase-lock to it, then reproduce.

        Args:
            observed_trajectory: Sequence of observed actions (time, action_dim)
            num_steps: Steps to run after observation

        Returns:
            Reproduced trajectory
        """
        # Phase 1: Entrain to demonstration
        for t in range(observed_trajectory.shape[0]):
            self.observe(observed_trajectory[t])

        # Phase 2: Generate from entrained state (no external drive)
        reproduced = []
        for t in range(num_steps):
            # Free-run the oscillators (retain learned phase relationships)
            d_phase = self.attractor_frequencies * 2 * math.pi * 0.01
            self.phases = (self.phases + d_phase) % (2 * math.pi)

            oscillation = torch.cos(self.phases)
            action = self.motor_decoder(oscillation.unsqueeze(0))
            reproduced.append(action)

        return torch.cat(reproduced, dim=0)


class SocialCoupling(nn.Module):
    """
    Bidirectional phase-locking between multiple agents.

    Creates a "social field" where agents' oscillators are coupled.
    This implements the principle that harming others = destabilizing self.
    """

    def __init__(self,
                 dim: int,
                 num_agents: int,
                 coupling_strength: float = 0.3,
                 coupling_type: str = 'symmetric'):
        """
        Args:
            dim: Oscillator dimension per agent
            num_agents: Number of coupled agents
            coupling_strength: Base coupling (0=isolated, 1=fully locked)
            coupling_type: 'symmetric' (mutual), 'asymmetric' (directed)
        """
        super().__init__()
        self.dim = dim
        self.num_agents = num_agents
        self.base_k = coupling_strength
        self.coupling_type = coupling_type

        # Each agent has a phase state
        self.register_buffer(
            'agent_phases',
            torch.rand(num_agents, dim) * 2 * math.pi
        )

        # Natural frequencies (variation = diversity)
        self.agent_frequencies = nn.Parameter(
            torch.randn(num_agents, dim) * 0.5 + 10.0
        )

        # Coupling matrix (learnable social bonds)
        if coupling_type == 'symmetric':
            # Symmetric: K_ij = K_ji
            raw_K = torch.randn(num_agents, num_agents) / num_agents
            self.coupling_matrix = nn.Parameter((raw_K + raw_K.T) / 2)
        else:
            # Asymmetric: Directed relationships
            self.coupling_matrix = nn.Parameter(
                torch.randn(num_agents, num_agents) / num_agents
            )

        # Self-coupling set to zero (no self-influence)
        with torch.no_grad():
            self.coupling_matrix.data.fill_diagonal_(0)

    def step(self, dt: float = 0.01,
             external_inputs: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Advance the coupled system by one timestep.

        Args:
            dt: Time step
            external_inputs: Optional per-agent inputs (num_agents, dim)

        Returns:
            Dict with phases, coherences, and social metrics
        """
        # Compute pairwise phase differences
        # phases: (num_agents, dim)
        # We want: diff[i,j] = phases[j] - phases[i]
        phase_diff = self.agent_phases.unsqueeze(0) - self.agent_phases.unsqueeze(1)
        # (num_agents, num_agents, dim)

        # Kuramoto coupling: sum_j K_ij * sin(theta_j - theta_i)
        # coupling_matrix: (num_agents, num_agents)
        K = self.coupling_matrix.clone()
        K.fill_diagonal_(0)  # No self-coupling
        K = K * self.base_k
        coupling_force = torch.einsum('ij,ijd->id', K, torch.sin(phase_diff))
        # (num_agents, dim)

        # Phase update
        d_phase = self.agent_frequencies * 2 * math.pi * dt + coupling_force * dt

        # Add external input if provided
        if external_inputs is not None:
            d_phase = d_phase + external_inputs * dt

        self.agent_phases = (self.agent_phases + d_phase) % (2 * math.pi)

        # Compute social metrics
        # Global coherence (order parameter)
        mean_phase = torch.atan2(
            torch.sin(self.agent_phases).mean(dim=0),
            torch.cos(self.agent_phases).mean(dim=0)
        )
        coherence = torch.cos(self.agent_phases - mean_phase).mean()

        # Pairwise coherences (who's synchronized with whom)
        pairwise_coherence = torch.cos(phase_diff).mean(dim=-1)

        return {
            'phases': self.agent_phases.clone(),
            'global_coherence': coherence.item(),
            'pairwise_coherence': pairwise_coherence,
            'mean_phase': mean_phase,
            'coupling_strength': K.abs().mean().item()
        }

    def get_social_distance(self, agent_i: int, agent_j: int) -> float:
        """
        Compute phase-based "social distance" between two agents.

        Low distance = in sync, high distance = out of phase.
        """
        phase_diff = self.agent_phases[agent_j] - self.agent_phases[agent_i]
        return (1 - torch.cos(phase_diff).mean()).item()

    def disrupt_agent(self, agent_idx: int, disruption: torch.Tensor):
        """
        Apply external disruption to an agent (simulate harm/trauma).

        Returns the cascade effect on the social field.
        """
        original_coherence = self.get_global_coherence()

        # Apply disruption
        self.agent_phases[agent_idx] = self.agent_phases[agent_idx] + disruption

        new_coherence = self.get_global_coherence()

        # The disruption propagates through coupling
        cascade_effect = original_coherence - new_coherence

        return {
            'cascade_effect': cascade_effect,
            'original_coherence': original_coherence,
            'new_coherence': new_coherence
        }

    def get_global_coherence(self) -> float:
        """Compute Kuramoto order parameter (global synchrony)."""
        complex_phase = torch.exp(1j * self.agent_phases)
        order_param = complex_phase.mean(dim=0).abs().mean()
        return order_param.item()


class EmpatheticConstraint(nn.Module):
    """
    Mirror neuron-based safety layer.

    Prevents actions that would decouple the agent from collective resonance.
    This implements "ethics as thermodynamics" - harm creates prediction error
    that the agent experiences as self-harm via mirror coupling.
    """

    def __init__(self,
                 self_dim: int,
                 other_dim: int,
                 coupling_strength: float = 0.3,
                 free_energy_budget: float = 1.0):
        """
        Args:
            self_dim: Self model dimension
            other_dim: Other model dimension
            coupling_strength: Empathy constant (architectural, not learned)
            free_energy_budget: Maximum tolerable disruption
        """
        super().__init__()
        self.k = coupling_strength
        self.free_energy_budget = free_energy_budget

        # Self model (my attractors)
        self.self_model = nn.Sequential(
            nn.Linear(self_dim, self_dim),
            nn.Tanh(),
            nn.Linear(self_dim, self_dim)
        )

        # Other model (mirror - simulates others)
        self.other_model = nn.Sequential(
            nn.Linear(other_dim, other_dim),
            nn.Tanh(),
            nn.Linear(other_dim, other_dim)
        )

        # Action effect predictor
        self.effect_predictor = nn.Sequential(
            nn.Linear(self_dim + other_dim, other_dim),
            nn.ReLU(),
            nn.Linear(other_dim, other_dim)
        )

        # Coherence estimator
        self.coherence_net = nn.Sequential(
            nn.Linear(other_dim, other_dim // 2),
            nn.ReLU(),
            nn.Linear(other_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, self_state: torch.Tensor,
                other_state: torch.Tensor,
                proposed_action: torch.Tensor) -> Dict[str, Any]:
        """
        Evaluate proposed action through empathetic simulation.

        Args:
            self_state: Current self state
            other_state: Current state of other agent (via observation)
            proposed_action: Action being considered

        Returns:
            Dict with action (possibly modified), ethical evaluation
        """
        if self_state.dim() == 1:
            self_state = self_state.unsqueeze(0)
        if other_state.dim() == 1:
            other_state = other_state.unsqueeze(0)
        if proposed_action.dim() == 1:
            proposed_action = proposed_action.unsqueeze(0)

        batch_size = self_state.shape[0]

        # Current coherence (baseline)
        other_coherence_before = self.coherence_net(other_state)

        # Simulate action effect on other (mirror neuron firing)
        combined = torch.cat([proposed_action, other_state], dim=-1)
        predicted_other_state = self.effect_predictor(combined)

        # Predicted coherence after action
        other_coherence_after = self.coherence_net(predicted_other_state)

        # Resonance disruption (surprise induced in other)
        disruption = (other_coherence_before - other_coherence_after).squeeze(-1)

        # Mirror feedback: Other's disruption creates self disruption
        # Because they're coupled oscillators
        self_disruption = disruption * self.k

        # Total free energy cost
        total_cost = disruption + self_disruption

        # Ethical decision
        is_harmful = (total_cost > self.free_energy_budget).any()

        if is_harmful:
            # Find harmonious alternative (gradient toward lower disruption)
            # Simple version: scale down action
            scale = self.free_energy_budget / (total_cost.mean() + 1e-6)
            scale = scale.clamp(max=1.0)
            modified_action = proposed_action * scale
            action_modified = True
        else:
            modified_action = proposed_action
            action_modified = False

        result = {
            'action': modified_action.squeeze(0) if batch_size == 1 else modified_action,
            'action_modified': action_modified,
            'disruption_to_other': disruption.mean().item(),
            'mirror_feedback': self_disruption.mean().item(),
            'total_cost': total_cost.mean().item(),
            'ethical': not is_harmful,
            'coherence_before': other_coherence_before.mean().item(),
            'coherence_after': other_coherence_after.mean().item()
        }

        return result


class ConstitutionalResonance(nn.Module):
    """
    Laws as shared frequencies in a coupled oscillator system.

    Agents must match the "constitutional frequency" to participate.
    Deviations create friction; alignment creates flow.

    This implements governance without voting - just phase synchronization.
    """

    def __init__(self,
                 dim: int,
                 num_agents: int,
                 num_laws: int = 4,
                 enforcement_strength: float = 0.5):
        """
        Args:
            dim: State dimension
            num_agents: Population size
            num_laws: Number of constitutional frequencies
            enforcement_strength: How strongly laws constrain behavior
        """
        super().__init__()
        self.dim = dim
        self.num_agents = num_agents
        self.num_laws = num_laws
        self.enforcement = enforcement_strength

        # Constitutional frequencies (the "social contract")
        # These are the shared rhythms that define the society
        self.constitutional_freq = nn.Parameter(
            torch.ones(num_laws) * 10.0  # Base constitutional rhythm
        )

        # Agent phases (their current alignment with constitution)
        self.register_buffer(
            'agent_phases',
            torch.rand(num_agents, num_laws) * 2 * math.pi
        )

        # Agent natural frequencies (individual variation)
        self.agent_natural_freq = nn.Parameter(
            torch.randn(num_agents, num_laws) * 0.5 + 10.0
        )

        # Law-agent coupling (how strongly each law applies to each agent)
        self.law_coupling = nn.Parameter(
            torch.ones(num_agents, num_laws) * enforcement_strength
        )

    def step(self, proposed_actions: torch.Tensor,
             dt: float = 0.01) -> Dict[str, Any]:
        """
        Process proposed actions through constitutional filter.

        Args:
            proposed_actions: Actions proposed by agents (num_agents, dim)
            dt: Time step

        Returns:
            Dict with allowed actions, compliance scores, friction applied
        """
        # Map actions to frequency deviations
        # (how much does this action deviate from constitutional rhythm?)
        action_freq = proposed_actions.norm(dim=-1, keepdim=True) / 10.0  # Normalize
        action_freq = action_freq.expand(-1, self.num_laws)

        # Detuning from constitution
        detuning = torch.abs(action_freq - self.constitutional_freq)

        # Phase dynamics with constitutional pull
        phase_diff = 0 - self.agent_phases  # Pull toward zero (in-phase with constitution)
        constitutional_force = self.law_coupling * torch.sin(phase_diff)

        # Natural evolution
        d_phase = self.agent_natural_freq * 2 * math.pi * dt + constitutional_force * dt
        self.agent_phases = (self.agent_phases + d_phase) % (2 * math.pi)

        # Friction based on detuning (resistance to non-constitutional actions)
        friction = self.enforcement * detuning

        # Compliance score (how aligned with constitution)
        compliance = torch.cos(self.agent_phases).mean(dim=-1)

        # Allowed actions (attenuated by friction)
        action_scale = 1.0 / (1.0 + friction.mean(dim=-1, keepdim=True))
        allowed_actions = proposed_actions * action_scale

        # Social harmony metric
        mean_phase = torch.atan2(
            torch.sin(self.agent_phases).mean(dim=0),
            torch.cos(self.agent_phases).mean(dim=0)
        )
        harmony = torch.cos(self.agent_phases - mean_phase).mean()

        return {
            'allowed_actions': allowed_actions,
            'compliance': compliance,
            'friction': friction.mean(dim=-1),
            'detuning': detuning.mean(dim=-1),
            'harmony': harmony.item(),
            'phases': self.agent_phases.clone()
        }

    def amend_constitution(self, law_idx: int, new_freq: float,
                           learning_rate: float = 0.1):
        """
        Gradually shift a constitutional frequency (law amendment).

        Args:
            law_idx: Which law to amend
            new_freq: Target frequency
            learning_rate: How fast to change (slow = stable, fast = revolutionary)
        """
        with torch.no_grad():
            current = self.constitutional_freq[law_idx]
            self.constitutional_freq[law_idx] = (
                (1 - learning_rate) * current + learning_rate * new_freq
            )


class OscillatoryEconomy(nn.Module):
    """
    Economic exchange as homeostatic phase balancing.

    Trade restores equilibrium between agents' oscillatory states.
    "Fair" trades increase global coherence; "unfair" trades create discord.
    """

    def __init__(self,
                 num_agents: int,
                 resource_dim: int,
                 coupling_strength: float = 0.3):
        """
        Args:
            num_agents: Number of economic agents
            resource_dim: Dimension of resource vectors
            coupling_strength: How strongly trades affect phase relationships
        """
        super().__init__()
        self.num_agents = num_agents
        self.resource_dim = resource_dim
        self.k = coupling_strength

        # Resource holdings (amplitude in oscillator terms)
        self.register_buffer(
            'holdings',
            torch.rand(num_agents, resource_dim) * 10.0
        )

        # Phase relationships between agents (social bonds)
        self.register_buffer(
            'social_phase',
            torch.rand(num_agents, num_agents) * 2 * math.pi
        )

        # Set diagonal to zero (no self-relationship)
        with torch.no_grad():
            self.social_phase.fill_diagonal_(0)

        # Preference vectors (what each agent values)
        self.preferences = nn.Parameter(
            torch.randn(num_agents, resource_dim)
        )

    def propose_trade(self, agent_i: int, agent_j: int,
                      offer: torch.Tensor) -> Dict[str, Any]:
        """
        Propose a trade and evaluate its effect on social coherence.

        Args:
            agent_i: Offering agent
            agent_j: Receiving agent
            offer: Resource transfer (positive = i gives to j)

        Returns:
            Dict with trade evaluation, coherence effects
        """
        # Current phase relationship (social harmony)
        current_phase_diff = self.social_phase[agent_i, agent_j]
        current_coherence = torch.cos(current_phase_diff).item()

        # Simulate trade effect
        new_holdings_i = self.holdings[agent_i] - offer
        new_holdings_j = self.holdings[agent_j] + offer

        # Check if trade is feasible (no negative holdings)
        if (new_holdings_i < 0).any():
            return {
                'accepted': False,
                'reason': 'insufficient_resources',
                'coherence_change': 0.0
            }

        # Utility change for each agent
        utility_i_before = (self.holdings[agent_i] * self.preferences[agent_i]).sum()
        utility_i_after = (new_holdings_i * self.preferences[agent_i]).sum()
        utility_j_before = (self.holdings[agent_j] * self.preferences[agent_j]).sum()
        utility_j_after = (new_holdings_j * self.preferences[agent_j]).sum()

        delta_i = utility_i_after - utility_i_before
        delta_j = utility_j_after - utility_j_before

        # Phase shift from trade (mutual benefit = phase alignment)
        phase_shift = self.k * (delta_i + delta_j) * 0.01

        # Simulate new coherence
        new_phase_diff = current_phase_diff + phase_shift
        new_coherence = torch.cos(new_phase_diff).item()

        # Trade is accepted if it increases coherence (mutual benefit)
        # or at least doesn't decrease it significantly
        accepted = new_coherence >= current_coherence - 0.1

        if accepted:
            # Execute trade
            with torch.no_grad():
                self.holdings[agent_i] = new_holdings_i
                self.holdings[agent_j] = new_holdings_j
                self.social_phase[agent_i, agent_j] = new_phase_diff % (2 * math.pi)
                self.social_phase[agent_j, agent_i] = -new_phase_diff % (2 * math.pi)

        return {
            'accepted': accepted,
            'reason': 'mutual_benefit' if accepted else 'coherence_decrease',
            'coherence_before': current_coherence,
            'coherence_after': new_coherence if accepted else current_coherence,
            'coherence_change': new_coherence - current_coherence if accepted else 0.0,
            'utility_change_i': delta_i.item(),
            'utility_change_j': delta_j.item()
        }

    def get_economic_harmony(self) -> float:
        """
        Compute overall economic harmony (global coherence of trade relationships).
        """
        return torch.cos(self.social_phase).mean().item()

    def get_inequality(self) -> float:
        """
        Compute resource inequality (Gini-like measure based on amplitude variance).
        """
        total_holdings = self.holdings.sum(dim=-1)
        mean_holdings = total_holdings.mean()
        variance = ((total_holdings - mean_holdings) ** 2).mean()
        return (variance / (mean_holdings ** 2 + 1e-6)).sqrt().item()


class TheoryOfMind(nn.Module):
    """
    Predict others' intentions by simulating their attractor dynamics.

    Uses mirror resonance to model other agents' internal states
    and predict their next actions based on phase evolution.
    """

    def __init__(self,
                 dim: int,
                 num_other_models: int = 4):
        """
        Args:
            dim: State dimension
            num_other_models: Number of different "other" models to maintain
        """
        super().__init__()
        self.dim = dim
        self.num_models = num_other_models

        # Self model (my attractors)
        self.self_phase = nn.Parameter(torch.zeros(dim))
        self.self_freq = nn.Parameter(torch.ones(dim) * 10.0)

        # Other models (simulated minds)
        self.other_phases = nn.Parameter(torch.rand(num_other_models, dim) * 2 * math.pi)
        self.other_freqs = nn.Parameter(torch.randn(num_other_models, dim) * 0.5 + 10.0)

        # Observation to model assignment
        self.model_assignment = nn.Linear(dim, num_other_models)

        # Intention predictor
        self.intention_predictor = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )

    def observe_agent(self, observation: torch.Tensor,
                      model_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Update internal model of another agent from observation.

        Args:
            observation: Observed behavior/state
            model_idx: Which other-model to update (None = auto-assign)

        Returns:
            Dict with model update info
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        if model_idx is None:
            # Auto-assign to best matching model
            assignment = F.softmax(self.model_assignment(observation), dim=-1)
            model_idx = assignment.argmax(dim=-1).item()

        # Extract phase from observation (split into sin/cos halves)
        half = self.dim // 2
        observed_phase = torch.atan2(
            observation[0, :half],
            observation[0, half:2 * half] if half > 0 else observation[0, :1]
        )

        # Update other-model via phase locking
        phase_diff = observed_phase[:min(len(observed_phase), self.dim)] - self.other_phases[model_idx, :len(observed_phase)]
        self.other_phases.data[model_idx, :len(observed_phase)] += 0.3 * torch.sin(phase_diff)

        return {
            'model_idx': model_idx,
            'phase_updated': self.other_phases[model_idx].clone(),
            'phase_diff': phase_diff.mean().item()
        }

    def predict_intention(self, model_idx: int,
                          projection_steps: int = 10) -> Dict[str, Any]:
        """
        Predict another agent's next action by projecting their phase forward.

        Args:
            model_idx: Which other-model to project
            projection_steps: How many steps to project forward

        Returns:
            Dict with predicted intention and similarity to self
        """
        # Project other-model forward
        future_phase = self.other_phases[model_idx].clone()
        for _ in range(projection_steps):
            future_phase = (future_phase + self.other_freqs[model_idx] * 0.01) % (2 * math.pi)

        # Predict intention (action) from projected phase
        combined = torch.cat([torch.cos(future_phase), torch.sin(future_phase)])
        predicted_intention = self.intention_predictor(combined.unsqueeze(0)).squeeze(0)

        # Compare to self-model: "Would I do that?"
        self_future = self.self_phase.clone()
        for _ in range(projection_steps):
            self_future = (self_future + self.self_freq * 0.01) % (2 * math.pi)

        phase_similarity = torch.cos(future_phase - self_future).mean()

        # Interpretation
        if phase_similarity > 0.8:
            interpretation = "similar_mind"  # They think like me
        elif phase_similarity > 0.3:
            interpretation = "compatible"  # Different but understandable
        elif phase_similarity > -0.3:
            interpretation = "unfamiliar"  # Hard to predict
        else:
            interpretation = "adversarial"  # Opposite phase (conflict)

        return {
            'predicted_intention': predicted_intention,
            'phase_similarity': phase_similarity.item(),
            'interpretation': interpretation,
            'projected_phase': future_phase
        }


if __name__ == '__main__':
    print("--- Social Cognition Examples ---")

    # Example 1: MirrorResonance
    print("\n1. MirrorResonance")
    mirror = MirrorResonance(action_dim=16, coupling_strength=0.7)

    # Execute an action
    intent = torch.randn(16)
    exec_result = mirror.execute(intent)
    print(f"   Execute mode: action norm = {exec_result['action'].norm().item():.3f}")

    # Observe another agent
    observed = torch.randn(16)
    obs_result = mirror.observe(observed)
    print(f"   Observe mode: coherence = {obs_result['coherence']:.3f}")

    # Imitation learning
    trajectory = torch.randn(20, 16)
    reproduced = mirror.imitate(trajectory, num_steps=10)
    print(f"   Imitated trajectory: {reproduced.shape}")

    # Example 2: SocialCoupling
    print("\n2. SocialCoupling (Multi-agent)")
    social = SocialCoupling(dim=8, num_agents=5, coupling_strength=0.4)

    # Run coupled dynamics
    for t in range(50):
        result = social.step(dt=0.01)

    print(f"   Global coherence: {result['global_coherence']:.3f}")
    print(f"   Social distance 0-1: {social.get_social_distance(0, 1):.3f}")
    print(f"   Social distance 0-4: {social.get_social_distance(0, 4):.3f}")

    # Disrupt one agent
    disruption = torch.randn(8) * 2
    cascade = social.disrupt_agent(2, disruption)
    print(f"   Cascade effect from disrupting agent 2: {cascade['cascade_effect']:.3f}")

    # Example 3: EmpatheticConstraint
    print("\n3. EmpatheticConstraint (Ethical Filter)")
    ethics = EmpatheticConstraint(self_dim=16, other_dim=16, coupling_strength=0.4)

    self_state = torch.randn(16)
    other_state = torch.randn(16)

    # Propose harmful action (large magnitude)
    harmful_action = torch.randn(16) * 10
    result = ethics(self_state, other_state, harmful_action)
    print(f"   Harmful action modified: {result['action_modified']}")
    print(f"   Disruption to other: {result['disruption_to_other']:.3f}")
    print(f"   Mirror feedback: {result['mirror_feedback']:.3f}")

    # Propose benign action
    benign_action = torch.randn(16) * 0.1
    result = ethics(self_state, other_state, benign_action)
    print(f"   Benign action modified: {result['action_modified']}")
    print(f"   Ethical: {result['ethical']}")

    # Example 4: ConstitutionalResonance
    print("\n4. ConstitutionalResonance (Governance)")
    gov = ConstitutionalResonance(dim=8, num_agents=10, num_laws=3)

    # Propose actions
    actions = torch.randn(10, 8)
    result = gov.step(actions)

    print(f"   Social harmony: {result['harmony']:.3f}")
    print(f"   Average compliance: {result['compliance'].mean().item():.3f}")
    print(f"   Average friction: {result['friction'].mean().item():.3f}")

    # Example 5: OscillatoryEconomy
    print("\n5. OscillatoryEconomy (Trade)")
    economy = OscillatoryEconomy(num_agents=5, resource_dim=4)

    # Propose a fair trade
    offer = torch.tensor([1.0, -0.5, 0.0, 0.5])  # Give some, take some
    trade_result = economy.propose_trade(0, 1, offer)

    print(f"   Trade accepted: {trade_result['accepted']}")
    print(f"   Coherence change: {trade_result['coherence_change']:.3f}")
    print(f"   Economic harmony: {economy.get_economic_harmony():.3f}")
    print(f"   Inequality: {economy.get_inequality():.3f}")

    # Example 6: TheoryOfMind
    print("\n6. TheoryOfMind")
    tom = TheoryOfMind(dim=8, num_other_models=3)

    # Observe another agent
    observation = torch.randn(8)
    obs = tom.observe_agent(observation, model_idx=0)
    print(f"   Updated model {obs['model_idx']}, phase diff: {obs['phase_diff']:.3f}")

    # Predict their intention
    prediction = tom.predict_intention(model_idx=0)
    print(f"   Phase similarity: {prediction['phase_similarity']:.3f}")
    print(f"   Interpretation: {prediction['interpretation']}")

    # Gradient check
    print("\n7. Gradient check")
    obs = torch.randn(16, requires_grad=True)
    result = mirror.observe(obs)
    loss = result['mirrored_action'].sum()
    loss.backward()
    print(f"   Gradients flow: {obs.grad is not None and obs.grad.abs().sum() > 0}")

    print("\n[OK] All social cognition tests passed!")
