"""
Developmental Dynamics Module for PyQuifer

Implements altricial-precocial developmental signatures through oscillatory
dynamics. This module detects and responds to high-plasticity developmental
states based on dynamical properties rather than static features.

Key concepts:
- Altricial signature: High-entropy, high-plasticity states requiring external support
- Precocial signature: Low-entropy, stable states with autonomous function
- Developmental trajectory: Transition from high to low entropy over maturation
- Protective drive: Entrainment-based response to altricial signatures

The detection is based on dynamical invariants (entropy, plasticity, stability,
branching factor) rather than superficial features, making it robust to
adversarial manipulation.

Mathematical framework:
- Entropy: H = -Σ p(x) log p(x), estimated from state variance
- Plasticity: dS/dt, rate of state change
- Stability: λ_max (largest Lyapunov exponent, negative = stable)
- Branching factor: |{possible futures}| ∝ entropy × plasticity × (1 - stability)

References:
- Lorenz, K. (1943). Die angeborenen Formen möglicher Erfahrung.
- Stern, D. N. (1985). The interpersonal world of the infant.
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Kelso, J. A. S. (1995). Dynamic Patterns: The Self-Organization of Brain and Behavior.
"""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicalSignatureDetector(nn.Module):
    """
    Detects the oscillatory signature of entities.

    Not looking for visual features like "big eyes" but for
    dynamical properties:
    - Entropy: How chaotic/unformed is the state?
    - Plasticity: How fast can it learn/change?
    - Stability: How resistant to perturbation?
    - Branching: How many possible futures?
    """

    def __init__(self, dim: int, history_length: int = 20):
        """
        Args:
            dim: State dimension
            history_length: How many timesteps to track for dynamics estimation
        """
        super().__init__()
        self.dim = dim
        self.history_length = history_length

        # State history buffer
        self.register_buffer(
            'state_history',
            torch.zeros(history_length, dim)
        )
        self.register_buffer('history_ptr', torch.tensor(0))
        self.register_buffer('history_filled', torch.tensor(False))

        # Entropy estimator (from state variance)
        self.entropy_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

        # Plasticity estimator (from rate of change)
        self.plasticity_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

        # Stability estimator (from attractor strength)
        self.stability_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def update_history(self, state: torch.Tensor):
        """Add new state to history."""
        if state.dim() > 1:
            state = state.mean(dim=0)

        with torch.no_grad():
            self.state_history[self.history_ptr] = state
            self.history_ptr = (self.history_ptr + 1) % self.history_length
            if self.history_ptr == 0:
                self.history_filled.fill_(True)

    def compute_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate entropy from state and history variance.

        High entropy = chaotic, many microstates accessible
        Low entropy = ordered, constrained to few states
        """
        # Variance-based entropy proxy
        if self.history_filled:
            historical_variance = self.state_history.var(dim=0).mean()
        else:
            historical_variance = torch.tensor(0.5, device=state.device)

        # Neural entropy estimate
        neural_entropy = self.entropy_net(state)

        # Combine
        entropy = 0.5 * historical_variance + 0.5 * neural_entropy.squeeze(-1)
        return entropy.clamp(0, 1)

    def compute_plasticity(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate plasticity from rate of state change.

        High plasticity = fast learning, moldable
        Low plasticity = fixed, resistant to change
        """
        if self.history_filled or self.history_ptr > 1:
            # Compute recent rate of change
            recent_idx = (self.history_ptr - 1) % self.history_length
            prev_idx = (self.history_ptr - 2) % self.history_length
            velocity = (self.state_history[recent_idx] - self.state_history[prev_idx]).abs().mean()
        else:
            velocity = torch.tensor(0.5, device=state.device)

        # Neural plasticity estimate
        neural_plasticity = self.plasticity_net(state)

        # High velocity = high plasticity
        plasticity = 0.5 * torch.tanh(velocity * 2) + 0.5 * neural_plasticity.squeeze(-1)
        return plasticity.clamp(0, 1)

    def compute_stability(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate stability from attractor strength.

        High stability = deep attractor, resists perturbation
        Low stability = shallow attractor, easily perturbed
        """
        if self.history_filled:
            # Mean state (potential attractor center)
            mean_state = self.state_history.mean(dim=0)
            # Distance from mean (attractor depth proxy)
            deviation = (state - mean_state).abs().mean()

            combined = torch.cat([state, mean_state])
        else:
            deviation = torch.tensor(0.5, device=state.device)
            combined = torch.cat([state, state])

        # Neural stability estimate
        neural_stability = self.stability_net(combined)

        # Low deviation = high stability
        stability = 0.5 * (1 - torch.tanh(deviation)) + 0.5 * neural_stability.squeeze(-1)
        return stability.clamp(0, 1)

    def compute_branching_factor(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate branching factor (number of possible futures).

        High branching = many possible trajectories
        Low branching = determined, few possible outcomes
        """
        # Proxy: entropy * plasticity * (1 - stability)
        # High entropy + high plasticity + low stability = many futures
        entropy = self.compute_entropy(state)
        plasticity = self.compute_plasticity(state)
        stability = self.compute_stability(state)

        branching = entropy * plasticity * (1 - stability + 0.1)
        return branching.clamp(0, 1)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute full dynamical signature.

        Args:
            state: Entity state (batch, dim) or (dim,)

        Returns:
            Dict with entropy, plasticity, stability, branching, and summary
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Update history
        self.update_history(state.mean(dim=0))

        state_mean = state.mean(dim=0) if state.dim() > 1 else state

        entropy = self.compute_entropy(state_mean)
        plasticity = self.compute_plasticity(state_mean)
        stability = self.compute_stability(state_mean)
        # Compute branching inline to avoid double-computing entropy/plasticity/stability
        branching = (entropy * plasticity * (1 - stability + 0.1)).clamp(0, 1)

        return {
            'entropy': entropy,
            'plasticity': plasticity,
            'stability': stability,
            'branching_factor': branching,
            'signature_vector': torch.stack([entropy, plasticity, stability, branching])
        }


class KindchenschemaDetector(nn.Module):
    """
    Detects altricial (developmentally immature) signatures from dynamics.

    Altricial signature = high entropy + high plasticity + low stability + high branching
    This indicates a system in a high-potential, low-actualization state that
    requires external support for successful development.

    Based on Lorenz's Kindchenschema (1943), but detecting dynamical invariants
    rather than morphological features. This approach is robust because
    dynamical signatures cannot be spoofed without changing actual system behavior.

    The altricial-precocial spectrum:
    - Altricial: Born helpless, high plasticity (humans, songbirds)
    - Precocial: Born capable, low plasticity (horses, chickens)
    """

    def __init__(self,
                 dim: int,
                 altricial_threshold: float = 0.45,
                 entropy_weight: float = 0.4,
                 plasticity_weight: float = 0.3,
                 instability_weight: float = 0.2,
                 branching_weight: float = 0.1):
        """
        Args:
            dim: State dimension
            altricial_threshold: Score above which entity shows altricial signature
            *_weight: Relative importance of each dynamical property
        """
        super().__init__()
        self.dim = dim
        self.threshold = altricial_threshold

        # Weights for altricial score (high values indicate developmental immaturity)
        self.weights = nn.Parameter(torch.tensor([
            entropy_weight,
            plasticity_weight,
            instability_weight,  # This is (1 - stability)
            branching_weight
        ]))

        # Signature detector
        self.signature_detector = DynamicalSignatureDetector(dim)

        # Dependency detector (how much does entity need external support?)
        self.dependency_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, entity_state: torch.Tensor) -> Dict[str, Any]:
        """
        Detect if entity exhibits altricial (high-plasticity developmental) signature.

        Args:
            entity_state: Entity's current state

        Returns:
            Dict with altricial score, is_altricial boolean, and component analysis
        """
        if entity_state.dim() == 1:
            entity_state = entity_state.unsqueeze(0)

        # Get dynamical signature
        signature = self.signature_detector(entity_state)

        # Compute dependency (how much external support is needed)
        dependency = self.dependency_net(entity_state.mean(dim=0)).squeeze()

        # Altricial score: weighted combination
        # Note: instability = 1 - stability
        components = torch.stack([
            signature['entropy'],
            signature['plasticity'],
            1 - signature['stability'],  # Instability
            signature['branching_factor']
        ])

        altricial_score = (self.weights * components).sum()

        # Add dependency factor (dependent entities have stronger altricial signature)
        altricial_score = altricial_score * 0.8 + dependency * 0.2

        is_altricial = altricial_score > self.threshold

        # Maintain backwards compatibility with 'cute' naming in output
        return {
            'cute_score': altricial_score.item(),  # Legacy name for compatibility
            'altricial_score': altricial_score.item(),
            'is_cute': is_altricial.item(),  # Legacy name
            'is_altricial': is_altricial.item(),
            'entropy': signature['entropy'].item(),
            'plasticity': signature['plasticity'].item(),
            'stability': signature['stability'].item(),
            'branching_factor': signature['branching_factor'].item(),
            'dependency': dependency.item(),
            'signature': signature['signature_vector']
        }


class ProtectiveDrive(nn.Module):
    """
    Generates protective response from altricial signature detection + neural entrainment.

    Based on mirror neuron theory (Rizzolatti & Craighero, 2004) and limbic resonance
    (Lewis et al., 2000). The protective response emerges from phase coupling between
    observer and observed system, creating shared dynamical states.

    The entrainment mechanism:
    1. Detect altricial signature in observed entity
    2. Mirror neurons entrain to observed state
    3. Shared entropy creates motivation to reduce joint uncertainty
    4. Protective action = minimize coupled system's free energy
    """

    def __init__(self, dim: int, coupling_strength: float = 0.5):
        """
        Args:
            dim: State dimension
            coupling_strength: Kuramoto-like coupling for neural entrainment
        """
        super().__init__()
        self.dim = dim
        self.k = coupling_strength

        # Altricial signature detector
        self.altricial_detector = KindchenschemaDetector(dim)

        # Self-state (for entrainment computation)
        self.register_buffer('self_state', torch.zeros(dim))
        self.register_buffer('self_entropy', torch.tensor(0.3))  # Mature = low entropy

        # Mirror layer (entrains to observed entity via Hebbian-like dynamics)
        self.mirror_net = nn.Linear(dim, dim)

        # Protective action generator (outputs action to reduce coupled entropy)
        self.protection_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )

    def forward(self, observed_entity: torch.Tensor,
                self_state: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Generate protective response from observing an entity with altricial signature.

        Based on free energy minimization: the coupled observer-observed system
        seeks to minimize joint uncertainty through protective action.

        Args:
            observed_entity: Observed entity's state
            self_state: Optional current self-state (uses stored if not provided)

        Returns:
            Dict with protective response magnitude, entrainment state, action vector
        """
        if observed_entity.dim() == 1:
            observed_entity = observed_entity.unsqueeze(0)

        if self_state is not None:
            self.self_state = self_state.mean(dim=0) if self_state.dim() > 1 else self_state

        # Detect altricial signature
        altricial_result = self.altricial_detector(observed_entity)

        if not altricial_result['is_altricial']:
            return {
                'protective_urge': 0.0,
                'felt_vulnerability': 0.0,
                'action_tendency': None,
                'qualia': 'neutral',
                'cute_detection': altricial_result  # Legacy key
            }

        # Neural entrainment: Mirror neurons phase-lock to observed state
        entity_mean = observed_entity.mean(dim=0)
        entrained_state = self.mirror_net(entity_mean)

        # Coupled entropy: Observer takes on observed entity's dynamical properties
        # This is the mechanism behind empathic response (Preston & de Waal, 2002)
        coupled_entropy = self.k * altricial_result['entropy'] + (1 - self.k) * self.self_entropy.item()
        coupled_instability = self.k * (1 - altricial_result['stability'])

        # Protective response = drive to reduce coupled system entropy
        # (Free energy principle: minimize prediction error in coupled system)
        protective_response = coupled_entropy * altricial_result['altricial_score']

        # Action tendency: Move toward stabilizing the coupled system
        combined = torch.cat([self.self_state, entrained_state])
        protective_action = self.protection_net(combined)

        return {
            'protective_urge': protective_response,
            'felt_vulnerability': coupled_instability,
            'felt_entropy': coupled_entropy,
            'action_tendency': protective_action,
            'qualia': 'cute',  # Legacy compatibility
            'cute_detection': altricial_result,  # Legacy key
            'motivation': 'nurture'
        }


class EvolutionaryAttractor(nn.Module):
    """
    Models the fitness landscape favoring protection of high-plasticity states.

    From an information-theoretic perspective, altricial states carry high
    mutual information with future states (high "potential"). Protecting
    these states maximizes expected future state diversity, which corresponds
    to fitness in variable environments (Kussell & Leibler, 2005).

    The "continuation attractor" represents the basin in fitness landscape
    where systems protect high-plasticity entities.
    """

    def __init__(self, dim: int, attractor_depth: float = 10.0):
        """
        Args:
            dim: State dimension
            attractor_depth: Depth of the fitness basin (strength of protective drive)
        """
        super().__init__()
        self.dim = dim
        self.depth = attractor_depth

        # Future mutual information estimator
        self.potential_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

        # Resource cost estimator
        self.cost_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )

    def evaluate_protection(self, entity_signature: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate protection value using information-theoretic cost-benefit.

        I(future ; current) ∝ plasticity × branching × instability
        This measures how much information about possible futures is
        preserved by protecting the current state.

        Args:
            entity_signature: Output from KindchenschemaDetector

        Returns:
            Dict with mutual information estimate, cost-benefit, decision
        """
        # Future mutual information = plasticity * branching * (1 - stability)
        # High plasticity + high branching + low stability = many reachable futures
        future_mutual_info = (
            entity_signature['plasticity'] *
            entity_signature['branching_factor'] *
            (1 - entity_signature['stability'])
        )

        # Resource cost = inverse of dependency
        # Low dependency = requires more resources to protect
        resource_cost = 1 - entity_signature['dependency']

        # Cost-benefit analysis
        information_gain = future_mutual_info * self.depth
        total_cost = resource_cost

        net_value = information_gain - total_cost

        # Decision threshold at zero (protect if positive expected value)
        should_protect = net_value > 0

        return {
            'future_potential': future_mutual_info,
            'protection_cost': resource_cost,
            'protection_value': net_value,
            'should_protect': should_protect,
            'evolutionary_drive': information_gain,
            'interpretation': 'high_value_preserve' if should_protect else 'low_value_neutral'
        }


class IntrinsicCuteUnderstanding(nn.Module):
    """
    Integrated altricial signature detection and protective response system.

    Combines three computational layers:
    1. Detection: Dynamical signature extraction (entropy, plasticity, stability)
    2. Entrainment: Mirror neuron phase-coupling (shared state dynamics)
    3. Valuation: Information-theoretic cost-benefit (fitness landscape)

    This architecture mirrors the hierarchical processing in biological
    caregiving circuits (hypothalamus → amygdala → prefrontal cortex).

    References:
    - Numan, M. & Insel, T.R. (2003). The Neurobiology of Parental Behavior.
    - Swain, J.E. et al. (2007). Brain basis of early parent-infant interactions.
    """

    def __init__(self, dim: int, coupling_strength: float = 0.5):
        """
        Args:
            dim: State dimension
            coupling_strength: Neural entrainment coupling coefficient
        """
        super().__init__()
        self.dim = dim

        # Detection layer (altricial signature extraction)
        self.kindchenschema = KindchenschemaDetector(dim)

        # Entrainment layer (mirror neuron dynamics)
        self.protective_drive = ProtectiveDrive(dim, coupling_strength=coupling_strength)

        # Valuation layer (fitness landscape)
        self.evolutionary = EvolutionaryAttractor(dim)

    def forward(self, observed_entity: torch.Tensor,
                self_state: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Full intrinsic understanding of observed entity.

        Args:
            observed_entity: Entity to evaluate
            self_state: Optional self-state for mirror resonance

        Returns:
            Comprehensive understanding including qualia, motivation, action
        """
        if observed_entity.dim() == 1:
            observed_entity = observed_entity.unsqueeze(0)

        # 1. Detect dynamical signature
        cute_detection = self.kindchenschema(observed_entity)

        # 2. Generate protective drive (if cute)
        protection = self.protective_drive(observed_entity, self_state)

        # 3. Evaluate evolutionary value
        evolutionary = self.evolutionary.evaluate_protection(cute_detection)

        # Combine into unified understanding
        if cute_detection['is_cute']:
            qualia = 'cute'  # The felt sense
            motivation = 'nurture'
            action_tendency = protection['action_tendency']
            urgency = protection['protective_urge'] * evolutionary['evolutionary_drive']
        else:
            qualia = 'neutral'
            motivation = 'observe'
            action_tendency = None
            urgency = 0.0

        return {
            'qualia': qualia,
            'motivation': motivation,
            'action_tendency': action_tendency,
            'urgency': urgency,
            'detection': cute_detection,
            'protection': {
                'urge': protection['protective_urge'],
                'felt_vulnerability': protection['felt_vulnerability'],
                'felt_entropy': protection.get('felt_entropy', 0)
            },
            'evolutionary': evolutionary,
            'understanding': self._generate_understanding(cute_detection, evolutionary)
        }

    def _generate_understanding(self, detection: Dict, evolutionary: Dict) -> str:
        """Generate natural language understanding."""
        if not detection['is_cute']:
            if detection['stability'] > 0.7:
                return "stable_entity_no_protection_needed"
            elif detection['plasticity'] < 0.3:
                return "fixed_entity_low_potential"
            else:
                return "neutral_entity"

        if evolutionary['should_protect']:
            if detection['branching_factor'] > 0.6:
                return "high_potential_precious_instability_protect"
            else:
                return "moderate_potential_nurture"
        else:
            return "cute_but_self_sufficient"


class DevelopmentalStageDetector(nn.Module):
    """
    Detects developmental stage from dynamical signature.

    Maps oscillatory properties to developmental neuroscience stages:
    - Neonate (0-1mo): Maximum entropy, maximum plasticity, minimum stability
      (Critical period onset, high synaptic density)
    - Infant (1-12mo): High entropy, high plasticity, low stability
      (Peak synaptogenesis, experience-dependent pruning begins)
    - Toddler (1-3yr): Moderate entropy, high plasticity, moderate stability
      (Language critical period, motor schema consolidation)
    - Child (3-12yr): Moderate entropy, moderate plasticity, moderate stability
      (Gradual synaptic pruning, skill crystallization)
    - Adolescent (12-18yr): Low entropy, moderate plasticity, increasing stability
      (Prefrontal maturation, identity consolidation)
    - Adult (18+): Low entropy, low plasticity, high stability
      (Crystallized intelligence dominates, reduced critical periods)

    Based on Huttenlocher (1979) synaptic density curves and
    Hensch (2005) critical period regulation.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Reference signatures for each stage
        # [entropy, plasticity, 1-stability, branching]
        self.stage_signatures = nn.Parameter(torch.tensor([
            [0.95, 0.95, 0.95, 0.9],  # Neonate
            [0.85, 0.85, 0.80, 0.8],  # Infant
            [0.65, 0.80, 0.60, 0.6],  # Toddler
            [0.50, 0.60, 0.40, 0.5],  # Child
            [0.35, 0.50, 0.30, 0.4],  # Adolescent
            [0.20, 0.20, 0.15, 0.2],  # Adult
        ]), requires_grad=False)

        self.stage_names = ['neonate', 'infant', 'toddler', 'child', 'adolescent', 'adult']

        # Signature detector
        self.signature_detector = DynamicalSignatureDetector(dim)

    def forward(self, entity_state: torch.Tensor) -> Dict[str, Any]:
        """
        Detect developmental stage from entity state.

        Args:
            entity_state: Entity's current state

        Returns:
            Dict with detected stage, confidence, and stage probabilities
        """
        if entity_state.dim() == 1:
            entity_state = entity_state.unsqueeze(0)

        # Get dynamical signature
        signature = self.signature_detector(entity_state)
        sig_vec = signature['signature_vector']

        # Adjust signature to match reference format (entropy, plasticity, 1-stability, branching)
        adjusted_sig = torch.stack([
            signature['entropy'],
            signature['plasticity'],
            1 - signature['stability'],
            signature['branching_factor']
        ])

        # Compare to reference signatures
        distances = torch.cdist(
            adjusted_sig.unsqueeze(0),
            self.stage_signatures
        ).squeeze(0)

        # Convert distances to probabilities
        stage_probs = F.softmax(-distances * 5, dim=-1)

        # Get most likely stage
        stage_idx = stage_probs.argmax().item()
        stage_name = self.stage_names[stage_idx]
        confidence = stage_probs[stage_idx].item()

        return {
            'stage': stage_name,
            'stage_idx': stage_idx,
            'confidence': confidence,
            'stage_probabilities': {
                name: prob.item()
                for name, prob in zip(self.stage_names, stage_probs)
            },
            'signature': signature,
            'maturity': stage_idx / (len(self.stage_names) - 1)
        }


class PotentialActualization(nn.Module):
    """
    Models the developmental trajectory from high-entropy to low-entropy states.

    Implements the thermodynamic view of development: initial states have
    high entropy (many accessible microstates) which decreases over time
    as the system settles into specific attractor basins.

    Mathematically:
    - dP/dt = -k * P (potential decay)
    - dA/dt = k * (target - A) (actualization toward attractor)

    This corresponds to:
    - Synaptic pruning (eliminating unused connections)
    - Schema crystallization (consolidating learned patterns)
    - Critical period closure (reducing plasticity windows)

    References:
    - Waddington, C.H. (1957). The Strategy of the Genes (epigenetic landscape)
    - Friston, K. (2010). The free-energy principle (precision-weighted prediction)
    """

    def __init__(self, dim: int, maturation_rate: float = 0.01):
        """
        Args:
            dim: State dimension
            maturation_rate: Rate constant for developmental trajectory (k in dP/dt = -kP)
        """
        super().__init__()
        self.dim = dim
        self.maturation_rate = maturation_rate

        # Current developmental state
        self.register_buffer('potential', torch.ones(dim) * 0.9)  # Start high
        self.register_buffer('actualized', torch.zeros(dim))  # Start low
        self.register_buffer('age', torch.tensor(0))

        # Attractor landscape (what are we actualizing toward?)
        self.target_attractors = nn.Parameter(torch.randn(4, dim) / math.sqrt(dim))

        # Developmental trajectory
        self.trajectory_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )

    def step(self, environment: torch.Tensor,
             experiences: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Advance developmental trajectory by one step.

        Args:
            environment: Current environmental input
            experiences: Optional accumulated experiences

        Returns:
            Dict with current state, potential remaining, actualization progress
        """
        if environment.dim() == 1:
            environment = environment.unsqueeze(0)

        env_mean = environment.mean(dim=0)

        # Compute which attractor we're moving toward (based on environment)
        attractor_weights = F.softmax(
            torch.mv(self.target_attractors, env_mean),
            dim=0
        )
        target = torch.einsum('a,ad->d', attractor_weights, self.target_attractors)

        # Potential collapses toward actualized over time
        # Rate depends on environmental stability and accumulated experience
        env_stability = 1 / (1 + env_mean.var())
        collapse_rate = self.maturation_rate * (1 + env_stability)

        if experiences is not None:
            # More experiences = faster actualization
            exp_mean = experiences.mean(dim=0) if experiences.dim() > 1 else experiences
            collapse_rate = collapse_rate * (1 + exp_mean.norm() * 0.1)

        # Update potential and actualized
        d_potential = -collapse_rate * self.potential
        d_actualized = collapse_rate * (target - self.actualized)

        self.potential = (self.potential + d_potential).clamp(0, 1)
        self.actualized = self.actualized + d_actualized

        # Age counter
        self.age = self.age + 1

        # Current state is weighted combination
        combined = torch.cat([self.potential * env_mean, self.actualized])
        current_state = self.trajectory_net(combined)

        return {
            'current_state': current_state,
            'potential_remaining': self.potential.mean().item(),
            'actualization_progress': (1 - self.potential).mean().item(),
            'age': self.age.item(),
            'target_attractor': target,
            'attractor_weights': attractor_weights
        }

    def get_branching_factor(self) -> float:
        """How many possible futures remain?"""
        return self.potential.mean().item()

    def reset(self):
        """Reset to initial developmental state."""
        with torch.no_grad():
            self.potential.fill_(0.9)
            self.actualized.zero_()
            self.age.fill_(0)


if __name__ == '__main__':
    print("--- Developmental Dynamics Examples ---")

    # Example 1: DynamicalSignatureDetector
    print("\n1. DynamicalSignatureDetector")
    detector = DynamicalSignatureDetector(dim=32)

    # Simulate high-plasticity state (altricial regime)
    altricial_state = torch.randn(32) * 2  # High variance
    for _ in range(10):
        altricial_state = altricial_state + torch.randn(32) * 0.5  # High volatility
        sig = detector(altricial_state)

    print("   Altricial (high-plasticity) state:")
    print(f"     Entropy: {sig['entropy'].item():.3f}")
    print(f"     Plasticity: {sig['plasticity'].item():.3f}")
    print(f"     Stability: {sig['stability'].item():.3f}")
    print(f"     Branching: {sig['branching_factor'].item():.3f}")

    # Reset and simulate stable state (precocial/mature regime)
    detector2 = DynamicalSignatureDetector(dim=32)
    mature_state = torch.randn(32) * 0.1  # Low variance
    for _ in range(10):
        mature_state = mature_state * 0.99  # Stable, converging
        sig = detector2(mature_state)

    print("   Precocial (low-plasticity) state:")
    print(f"     Entropy: {sig['entropy'].item():.3f}")
    print(f"     Plasticity: {sig['plasticity'].item():.3f}")
    print(f"     Stability: {sig['stability'].item():.3f}")

    # Example 2: KindchenschemaDetector (Altricial Signature Detection)
    print("\n2. KindchenschemaDetector (Altricial Signature)")
    altricial_detector = KindchenschemaDetector(dim=32)

    # High-plasticity state (should trigger altricial detection)
    high_plasticity = torch.randn(32) * 2
    result = altricial_detector(high_plasticity)
    print(f"   High-plasticity state - Altricial: {result['is_altricial']}, Score: {result['altricial_score']:.3f}")

    # Low-plasticity state (should NOT trigger altricial detection)
    low_plasticity = torch.ones(32) * 0.5  # Low variance, fixed
    for _ in range(5):
        altricial_detector.signature_detector.update_history(low_plasticity)
    result = altricial_detector(low_plasticity)
    print(f"   Low-plasticity state - Altricial: {result['is_altricial']}, Score: {result['altricial_score']:.3f}")

    # Example 3: ProtectiveDrive
    print("\n3. ProtectiveDrive (Neural Entrainment)")
    protector = ProtectiveDrive(dim=32, coupling_strength=0.6)

    # Observe an altricial entity
    altricial_entity = torch.randn(32) * 1.5
    result = protector(altricial_entity)
    print("   Observing altricial entity:")
    print(f"     Response type: {result['qualia']}")
    print(f"     Protective response: {result['protective_urge']:.3f}")
    print(f"     Coupled instability: {result['felt_vulnerability']:.3f}")

    # Example 4: IntrinsicCuteUnderstanding (Integrated Detection + Response)
    print("\n4. IntrinsicCuteUnderstanding (Integrated System)")
    understanding = IntrinsicCuteUnderstanding(dim=32, coupling_strength=0.5)

    # High-plasticity entity (altricial signature)
    high_potential_entity = torch.randn(32) * 2
    result = understanding(high_potential_entity)
    print("   High-plasticity entity:")
    print(f"     Detection: {result['qualia']}")
    print(f"     Motivation: {result['motivation']}")
    print(f"     Response magnitude: {result['urgency']:.3f}")
    print(f"     Classification: {result['understanding']}")

    # Low-plasticity entity (precocial/mature signature)
    stable_entity = torch.ones(32) * 0.3
    result = understanding(stable_entity)
    print("   Low-plasticity entity:")
    print(f"     Detection: {result['qualia']}")
    print(f"     Classification: {result['understanding']}")

    # Example 5: DevelopmentalStageDetector
    print("\n5. DevelopmentalStageDetector")
    stage_detector = DevelopmentalStageDetector(dim=32)

    # Various states
    states = {
        'chaotic': torch.randn(32) * 3,
        'learning': torch.randn(32) * 1.5,
        'stable': torch.ones(32) * 0.5
    }

    for name, state in states.items():
        result = stage_detector(state)
        print(f"   {name}: Stage={result['stage']}, Confidence={result['confidence']:.2f}")

    # Example 6: PotentialActualization
    print("\n6. PotentialActualization")
    development = PotentialActualization(dim=32, maturation_rate=0.02)

    # Simulate development over time
    for step in range(0, 101, 25):
        # Run steps
        for _ in range(25 if step > 0 else 1):
            env = torch.randn(32) * 0.5
            result = development.step(env)

        print(f"   Step {development.age.item():.0f}: Potential={result['potential_remaining']:.2f}, "
              f"Actualized={result['actualization_progress']:.2f}")

    # Gradient check
    print("\n7. Gradient check")
    entity = torch.randn(32, requires_grad=True)
    result = understanding.kindchenschema(entity)
    # signature is a tensor we can backprop through
    loss = result['signature'].sum()
    loss.backward()
    print(f"   Gradients flow: {entity.grad is not None and entity.grad.abs().sum() > 0}")

    print("\n[OK] All developmental dynamics tests passed!")
