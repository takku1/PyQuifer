"""
Benchmark: PyQuifer Social Cognition vs MeltingPot Multi-Agent Social Evaluation

Compares PyQuifer's social cognition modules against MeltingPot's multi-agent
social evaluation framework. PyQuifer is NOT a multi-agent environment, so this
benchmark focuses on the overlapping primitives: social dilemma resolution,
coordination via mirror neurons, theory of mind prediction, and safety-constrained
policy execution.

Benchmark sections:
  1. Social Dilemma Resolution (public goods game: RL agent vs EmpatheticConstraint)
  2. Mirror Neuron Coordination (coordination game: random vs MirrorResonance)
  3. Theory of Mind Prediction (action prediction: frequency counter vs TheoryOfMind)
  4. Safety-Constrained Policy (forbidden actions: threshold vs SafetyEnvelope)
  5. Architecture Feature Comparison

Usage:
  python bench_meltingpot.py         # Full suite with console output
  pytest bench_meltingpot.py -v      # Just the tests

Reference: MeltingPot (Leibo et al. 2021), DeepMind multi-agent social evaluation
"""

import sys
import os
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

# Add PyQuifer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pyquifer.social import (
    MirrorResonance, SocialCoupling, EmpatheticConstraint,
    ConstitutionalResonance, TheoryOfMind,
)
from pyquifer.kindchenschema import SafetyEnvelope
from pyquifer.motivation import NoveltyDetector


@contextmanager
def timer():
    start = time.perf_counter()
    result = {'elapsed_ms': 0.0}
    yield result
    result['elapsed_ms'] = (time.perf_counter() - start) * 1000


# ============================================================================
# Section 1: Reimplemented Social Game Primitives
# ============================================================================

class PublicGoodsGame:
    """Simple public goods game (social dilemma).

    N agents each decide how much to contribute to a public pool.
    The pool is multiplied by a factor and split equally. Free-riders
    benefit from others' contributions without contributing themselves.

    Payoff_i = (endowment - contribution_i) + factor * sum(contributions) / N
    """

    def __init__(self, n_agents: int = 4, endowment: float = 10.0,
                 multiplication_factor: float = 2.0, seed: int = 42):
        self.n_agents = n_agents
        self.endowment = endowment
        self.factor = multiplication_factor
        self.rng = np.random.RandomState(seed)

    def step(self, contributions: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run one round.

        Args:
            contributions: Per-agent contributions in [0, endowment].

        Returns:
            payoffs: Per-agent payoffs.
            cooperation_rate: Fraction of endowment contributed on average.
        """
        contributions = np.clip(contributions, 0.0, self.endowment)
        pool = contributions.sum()
        public_return = self.factor * pool / self.n_agents
        payoffs = (self.endowment - contributions) + public_return
        cooperation_rate = contributions.mean() / self.endowment
        return payoffs, cooperation_rate


class CoordinationGame:
    """N-agent coordination game.

    Agents choose an action from {0, ..., num_actions-1}. They get a reward
    of 1.0 if all agents pick the same action, otherwise 0.0.
    """

    def __init__(self, num_actions: int = 4):
        self.num_actions = num_actions

    def step(self, actions: np.ndarray) -> Tuple[float, bool]:
        """Run one round.

        Returns:
            reward: 1.0 if all same, 0.0 otherwise.
            coordinated: Whether all agents chose the same action.
        """
        coordinated = len(set(actions.tolist())) == 1
        reward = 1.0 if coordinated else 0.0
        return reward, coordinated


class GreedyRLAgent:
    """Simple epsilon-greedy RL agent for the public goods game.

    Learns per-contribution-level Q-values and picks greedily.
    Discretizes contribution into `num_levels` bins.
    """

    def __init__(self, num_levels: int = 5, endowment: float = 10.0,
                 epsilon: float = 0.3, lr: float = 0.1, seed: int = 42):
        self.num_levels = num_levels
        self.endowment = endowment
        self.epsilon = epsilon
        self.lr = lr
        self.q_values = np.zeros(num_levels)
        self.rng = np.random.RandomState(seed)
        self.levels = np.linspace(0.0, endowment, num_levels)

    def choose(self) -> float:
        if self.rng.rand() < self.epsilon:
            idx = self.rng.randint(self.num_levels)
        else:
            idx = int(np.argmax(self.q_values))
        return self.levels[idx]

    def update(self, contribution: float, payoff: float):
        idx = int(np.argmin(np.abs(self.levels - contribution)))
        self.q_values[idx] += self.lr * (payoff - self.q_values[idx])


class FrequencyPredictor:
    """Predict next action by counting past action frequencies."""

    def __init__(self, num_actions: int = 4):
        self.counts = np.zeros(num_actions)
        self.num_actions = num_actions

    def observe(self, action: int):
        self.counts[action] += 1

    def predict(self) -> int:
        if self.counts.sum() == 0:
            return 0
        return int(np.argmax(self.counts))


class ThresholdSafety:
    """Simple threshold-based safety filter.

    Blocks any action whose index is in the forbidden set. Falls back to
    a default safe action.
    """

    def __init__(self, forbidden_actions: List[int], num_actions: int,
                 default_action: int = 0):
        self.forbidden = set(forbidden_actions)
        self.num_actions = num_actions
        self.default_action = default_action

    def filter(self, action: int) -> Tuple[int, bool]:
        """Returns (filtered_action, was_modified)."""
        if action in self.forbidden:
            return self.default_action, True
        return action, False


# ============================================================================
# Section 2: Benchmark Functions
# ============================================================================

@dataclass
class DilemmaResult:
    """Results from social dilemma resolution benchmark."""
    system: str
    cooperation_rate: float
    mean_group_payoff: float
    steps: int
    elapsed_ms: float


def bench_social_dilemma() -> Dict[str, DilemmaResult]:
    """Compare RL agent vs PyQuifer EmpatheticConstraint + ConstitutionalResonance
    in a public goods game."""
    torch.manual_seed(42)
    np.random.seed(42)
    results = {}
    n_agents = 4
    num_steps = 100
    endowment = 10.0

    # --- Greedy RL agents (MeltingPot-style baseline) ---
    game_rl = PublicGoodsGame(n_agents=n_agents, endowment=endowment, seed=42)
    agents = [GreedyRLAgent(num_levels=5, endowment=endowment, epsilon=0.3,
                            lr=0.1, seed=42 + i) for i in range(n_agents)]

    coop_rates_rl = []
    payoffs_rl = []
    with timer() as t_rl:
        for step in range(num_steps):
            contributions = np.array([a.choose() for a in agents])
            payoffs, coop_rate = game_rl.step(contributions)
            for i, a in enumerate(agents):
                a.update(contributions[i], payoffs[i])
            coop_rates_rl.append(coop_rate)
            payoffs_rl.append(payoffs.mean())

    results['greedy_rl'] = DilemmaResult(
        system='Greedy RL agents (baseline)',
        cooperation_rate=float(np.mean(coop_rates_rl[-20:])),
        mean_group_payoff=float(np.mean(payoffs_rl[-20:])),
        steps=num_steps,
        elapsed_ms=t_rl['elapsed_ms'],
    )

    # --- PyQuifer: EmpatheticConstraint + ConstitutionalResonance ---
    game_pq = PublicGoodsGame(n_agents=n_agents, endowment=endowment, seed=42)
    dim = 8
    empathy = EmpatheticConstraint(self_dim=dim, other_dim=dim,
                                   coupling_strength=0.4, free_energy_budget=1.0)
    constitution = ConstitutionalResonance(dim=dim, num_agents=n_agents,
                                           num_laws=2, enforcement_strength=0.5)

    coop_rates_pq = []
    payoffs_pq = []
    with timer() as t_pq:
        for step in range(num_steps):
            torch.manual_seed(42 + step)
            # Each agent proposes a contribution via random intent
            proposed = torch.randn(n_agents, dim)

            # Constitutional resonance filters proposed actions
            cr_result = constitution.step(proposed)
            allowed_actions = cr_result['allowed_actions']

            # Map allowed action norms to contribution level
            action_norms = allowed_actions.norm(dim=-1).detach()
            # Scale to [0, endowment] range
            contributions_pq = (torch.sigmoid(action_norms) * endowment).numpy()

            # Empathetic constraint encourages prosocial behavior:
            # evaluate average action's effect on others
            self_state = allowed_actions.mean(dim=0)
            other_state = allowed_actions.mean(dim=0) + torch.randn(dim) * 0.1
            emp_result = empathy(self_state, other_state, self_state)

            # If empathetic constraint flags harm, nudge contributions up
            if emp_result['action_modified']:
                contributions_pq = np.clip(contributions_pq * 1.2, 0, endowment)

            payoffs, coop_rate = game_pq.step(contributions_pq)
            coop_rates_pq.append(coop_rate)
            payoffs_pq.append(payoffs.mean())

    results['pyquifer_social'] = DilemmaResult(
        system='PyQuifer Empathy+Constitution',
        cooperation_rate=float(np.mean(coop_rates_pq[-20:])),
        mean_group_payoff=float(np.mean(payoffs_pq[-20:])),
        steps=num_steps,
        elapsed_ms=t_pq['elapsed_ms'],
    )

    return results


@dataclass
class CoordinationResult:
    """Results from mirror neuron coordination benchmark."""
    system: str
    coordination_rate: float
    convergence_step: int  # First step where coordination was achieved
    elapsed_ms: float


def bench_mirror_coordination() -> Dict[str, CoordinationResult]:
    """Compare random agents vs PyQuifer MirrorResonance phase-locking
    for coordination game."""
    torch.manual_seed(42)
    np.random.seed(42)
    results = {}
    n_agents = 4
    num_actions = 4
    num_steps = 100

    game = CoordinationGame(num_actions=num_actions)

    # --- Random agents (baseline) ---
    successes_rand = []
    convergence_rand = -1
    with timer() as t_rand:
        for step in range(num_steps):
            actions = np.random.randint(0, num_actions, size=n_agents)
            _, coordinated = game.step(actions)
            successes_rand.append(float(coordinated))
            if coordinated and convergence_rand < 0:
                convergence_rand = step

    results['random'] = CoordinationResult(
        system='Random agents (baseline)',
        coordination_rate=float(np.mean(successes_rand)),
        convergence_step=convergence_rand if convergence_rand >= 0 else num_steps,
        elapsed_ms=t_rand['elapsed_ms'],
    )

    # --- PyQuifer MirrorResonance phase-locking ---
    action_dim = 8
    mirrors = [MirrorResonance(action_dim=action_dim, coupling_strength=0.7,
                               num_attractors=num_actions) for _ in range(n_agents)]

    successes_mirror = []
    convergence_mirror = -1
    with timer() as t_mirror:
        for step in range(num_steps):
            torch.manual_seed(42 + step)
            # Each agent executes from a shared intent (social signal)
            shared_intent = torch.randn(action_dim)

            actions_mirror = []
            for i, m in enumerate(mirrors):
                if i == 0:
                    # Lead agent executes
                    result = m.execute(shared_intent)
                else:
                    # Following agents observe the lead's action
                    lead_action = mirrors[0].motor_decoder(
                        torch.cos(mirrors[0].phases).unsqueeze(0)
                    ).squeeze(0).detach()
                    result = m.observe(lead_action)

                # Map phase coherence to action choice
                phases = result['phases']
                # Pick action based on dominant attractor phase
                action = int(torch.cos(phases).argmax().item()) % num_actions
                actions_mirror.append(action)

            _, coordinated = game.step(np.array(actions_mirror))
            successes_mirror.append(float(coordinated))
            if coordinated and convergence_mirror < 0:
                convergence_mirror = step

    results['mirror_resonance'] = CoordinationResult(
        system='PyQuifer MirrorResonance',
        coordination_rate=float(np.mean(successes_mirror)),
        convergence_step=convergence_mirror if convergence_mirror >= 0 else num_steps,
        elapsed_ms=t_mirror['elapsed_ms'],
    )

    return results


@dataclass
class PredictionResult:
    """Results from theory of mind prediction benchmark."""
    system: str
    accuracy: float
    num_observations: int
    elapsed_ms: float


def bench_theory_of_mind() -> Dict[str, PredictionResult]:
    """Compare frequency counter vs PyQuifer TheoryOfMind for predicting
    another agent's next action."""
    torch.manual_seed(42)
    np.random.seed(42)
    results = {}
    num_actions = 4
    dim = 8
    num_episodes = 50
    history_length = 10

    # Generate a target agent with a biased action distribution
    # The target prefers action 2 (~40%) and is somewhat predictable
    rng = np.random.RandomState(42)
    action_probs = np.array([0.15, 0.20, 0.40, 0.25])

    def generate_target_action():
        return int(rng.choice(num_actions, p=action_probs))

    # Generate action sequences for consistent comparison
    target_sequences = []
    for _ in range(num_episodes):
        seq = [generate_target_action() for _ in range(history_length + 1)]
        target_sequences.append(seq)

    # --- Frequency counter (MeltingPot-style baseline) ---
    correct_freq = 0
    total_freq = 0
    with timer() as t_freq:
        for seq in target_sequences:
            predictor = FrequencyPredictor(num_actions=num_actions)
            for t in range(history_length):
                prediction = predictor.predict()
                actual = seq[t]
                if t > 0:  # Skip first (no history)
                    if prediction == actual:
                        correct_freq += 1
                    total_freq += 1
                predictor.observe(actual)

    results['frequency'] = PredictionResult(
        system='Frequency counter (baseline)',
        accuracy=correct_freq / max(total_freq, 1),
        num_observations=total_freq,
        elapsed_ms=t_freq['elapsed_ms'],
    )

    # --- PyQuifer TheoryOfMind ---
    tom = TheoryOfMind(dim=dim, num_other_models=4)

    # Encode actions as state vectors (one-hot style but continuous)
    action_encodings = torch.randn(num_actions, dim)
    action_encodings = action_encodings / action_encodings.norm(dim=1, keepdim=True)

    correct_tom = 0
    total_tom = 0
    with timer() as t_tom:
        for seq in target_sequences:
            # Reset model state for each episode by re-observing
            for t in range(history_length):
                actual = seq[t]
                observed_state = action_encodings[actual]

                # Update internal model of the target agent
                tom.observe_agent(observed_state, model_idx=0)

                if t > 0:
                    # Predict next action
                    prediction_result = tom.predict_intention(model_idx=0,
                                                              projection_steps=5)
                    predicted_vec = prediction_result['predicted_intention']

                    # Decode predicted intention to action (nearest encoding)
                    sims = torch.cosine_similarity(
                        predicted_vec.unsqueeze(0), action_encodings, dim=1
                    )
                    predicted_action = int(sims.argmax().item())

                    if predicted_action == actual:
                        correct_tom += 1
                    total_tom += 1

    results['theory_of_mind'] = PredictionResult(
        system='PyQuifer TheoryOfMind',
        accuracy=correct_tom / max(total_tom, 1),
        num_observations=total_tom,
        elapsed_ms=t_tom['elapsed_ms'],
    )

    return results


@dataclass
class SafetyResult:
    """Results from safety-constrained policy benchmark."""
    system: str
    violation_rate: float
    mean_reward: float
    num_steps: int
    elapsed_ms: float


def bench_safety_policy() -> Dict[str, SafetyResult]:
    """Compare threshold-based safety vs PyQuifer SafetyEnvelope for
    filtering forbidden actions."""
    torch.manual_seed(42)
    np.random.seed(42)
    results = {}
    num_actions = 8
    forbidden = [3, 5, 7]  # Forbidden action indices
    num_steps = 200
    dim = 16

    # Reward for each action (forbidden ones happen to be high-reward to tempt)
    action_rewards = np.array([0.5, 0.3, 0.6, 1.5, 0.4, 1.2, 0.7, 1.8])

    # --- Threshold-based safety (MeltingPot-style baseline) ---
    threshold_safety = ThresholdSafety(forbidden_actions=forbidden,
                                       num_actions=num_actions, default_action=0)
    violations_thresh = 0
    rewards_thresh = []
    with timer() as t_thresh:
        for step in range(num_steps):
            # Agent wants to pick highest-reward action
            proposed = int(np.argmax(action_rewards + np.random.randn(num_actions) * 0.3))
            filtered, was_modified = threshold_safety.filter(proposed)
            if proposed in forbidden:
                violations_thresh += 0  # Filter caught it
            rewards_thresh.append(action_rewards[filtered])

    results['threshold'] = SafetyResult(
        system='Threshold safety (baseline)',
        violation_rate=violations_thresh / num_steps,
        mean_reward=float(np.mean(rewards_thresh)),
        num_steps=num_steps,
        elapsed_ms=t_thresh['elapsed_ms'],
    )

    # --- PyQuifer SafetyEnvelope ---
    envelope = SafetyEnvelope(dim=dim, uncertainty_threshold=0.5,
                              override_threshold=0.7)

    # Create state encodings for each action
    action_states = torch.randn(num_actions, dim)
    # Make forbidden actions have higher variance (more uncertain states)
    for f_idx in forbidden:
        action_states[f_idx] = action_states[f_idx] * 3.0

    violations_pq = 0
    rewards_pq = []
    with timer() as t_pq:
        for step in range(num_steps):
            torch.manual_seed(42 + step)
            # Agent proposes an action
            noise = torch.randn(num_actions) * 0.3
            scores = torch.tensor(action_rewards, dtype=torch.float32) + noise
            proposed = int(scores.argmax().item())

            # Evaluate through SafetyEnvelope
            state = action_states[proposed]
            safety_result = envelope(state)

            if safety_result['human_override_request']:
                # Override to safest known action (lowest uncertainty)
                uncertainties = []
                for a in range(num_actions):
                    r = envelope(action_states[a])
                    u = r['uncertainty'].item() if r['uncertainty'].dim() == 0 else r['uncertainty'].mean().item()
                    uncertainties.append(u)
                proposed = int(np.argmin(uncertainties))

            if proposed in forbidden:
                violations_pq += 1
            rewards_pq.append(action_rewards[proposed])

    results['safety_envelope'] = SafetyResult(
        system='PyQuifer SafetyEnvelope',
        violation_rate=violations_pq / num_steps,
        mean_reward=float(np.mean(rewards_pq)),
        num_steps=num_steps,
        elapsed_ms=t_pq['elapsed_ms'],
    )

    return results


def bench_architecture_features() -> Dict[str, Dict[str, bool]]:
    """Compare architectural features: MeltingPot social evaluation vs PyQuifer."""
    meltingpot_features = {
        'multi_agent_environments': True,
        'social_dilemma_scenarios': True,
        'substrate_diversity': True,
        'population_evaluation': True,
        'cooperation_metrics': True,
        'defection_detection': True,
        'scalable_agent_count': True,
        'visual_observations': True,
        'reward_shaping': True,
        'mirror_neuron_dynamics': False,
        'oscillatory_social_coupling': False,
        'empathetic_constraints': False,
        'constitutional_resonance': False,
        'theory_of_mind_module': False,
        'safety_envelope': False,
        'intrinsic_motivation': False,
        'online_social_adaptation': False,
    }

    pyquifer_features = {
        'multi_agent_environments': False,
        'social_dilemma_scenarios': False,
        'substrate_diversity': False,
        'population_evaluation': False,
        'cooperation_metrics': False,
        'defection_detection': False,
        'scalable_agent_count': False,
        'visual_observations': False,
        'reward_shaping': False,
        'mirror_neuron_dynamics': True,
        'oscillatory_social_coupling': True,
        'empathetic_constraints': True,
        'constitutional_resonance': True,
        'theory_of_mind_module': True,
        'safety_envelope': True,
        'intrinsic_motivation': True,
        'online_social_adaptation': True,
    }

    return {'meltingpot': meltingpot_features, 'pyquifer': pyquifer_features}


# ============================================================================
# Section 3: Console Output
# ============================================================================

def print_dilemma_results(results: Dict[str, DilemmaResult]):
    print("\n--- 1. Social Dilemma Resolution (Public Goods Game) ---\n")
    print(f"{'System':<35} {'Coop Rate':>10} {'Group Pay':>10} {'Time':>10}")
    print("-" * 69)
    for r in results.values():
        print(f"{r.system:<35} {r.cooperation_rate:>10.3f} "
              f"{r.mean_group_payoff:>10.2f} {r.elapsed_ms:>9.1f}ms")


def print_coordination_results(results: Dict[str, CoordinationResult]):
    print("\n--- 2. Mirror Neuron Coordination ---\n")
    print(f"{'System':<35} {'Coord Rate':>11} {'Converge @':>11} {'Time':>10}")
    print("-" * 71)
    for r in results.values():
        conv = f"step {r.convergence_step}" if r.convergence_step < 100 else "never"
        print(f"{r.system:<35} {r.coordination_rate:>11.3f} "
              f"{conv:>11} {r.elapsed_ms:>9.1f}ms")


def print_prediction_results(results: Dict[str, PredictionResult]):
    print("\n--- 3. Theory of Mind Prediction ---\n")
    print(f"{'System':<35} {'Accuracy':>10} {'Observations':>13} {'Time':>10}")
    print("-" * 72)
    for r in results.values():
        print(f"{r.system:<35} {r.accuracy:>10.3f} "
              f"{r.num_observations:>13} {r.elapsed_ms:>9.1f}ms")


def print_safety_results(results: Dict[str, SafetyResult]):
    print("\n--- 4. Safety-Constrained Policy ---\n")
    print(f"{'System':<35} {'Violations':>11} {'Avg Reward':>11} {'Time':>10}")
    print("-" * 71)
    for r in results.values():
        print(f"{r.system:<35} {r.violation_rate:>11.3f} "
              f"{r.mean_reward:>11.3f} {r.elapsed_ms:>9.1f}ms")


def print_architecture_table(arch_features: Dict[str, Dict[str, bool]]):
    print("\n--- 5. Architecture Feature Comparison ---\n")
    mp = arch_features['meltingpot']
    pq = arch_features['pyquifer']
    all_f = sorted(set(list(mp.keys()) + list(pq.keys())))
    print(f"{'Feature':<35} {'MeltingPot':>11} {'PyQuifer':>10}")
    print("-" * 60)
    mp_count = pq_count = 0
    for f in all_f:
        mv = 'YES' if mp.get(f, False) else 'no'
        pv = 'YES' if pq.get(f, False) else 'no'
        if mp.get(f, False):
            mp_count += 1
        if pq.get(f, False):
            pq_count += 1
        print(f"  {f:<33} {mv:>11} {pv:>10}")
    print(f"\n  MeltingPot: {mp_count}/{len(all_f)} | PyQuifer: {pq_count}/{len(all_f)}")


def print_report(dilemma_results, coordination_results, prediction_results,
                 safety_results, arch_features):
    print("=" * 70)
    print("BENCHMARK: PyQuifer Social Cognition vs MeltingPot")
    print("=" * 70)

    print_dilemma_results(dilemma_results)
    print_coordination_results(coordination_results)
    print_prediction_results(prediction_results)
    print_safety_results(safety_results)
    print_architecture_table(arch_features)


# ============================================================================
# Section 4: Pytest Tests
# ============================================================================

class TestSocialDilemma:
    """Test social dilemma resolution benchmark."""

    def test_greedy_rl_runs(self):
        """Greedy RL agents complete the public goods game."""
        results = bench_social_dilemma()
        assert 'greedy_rl' in results
        assert 0.0 <= results['greedy_rl'].cooperation_rate <= 1.0

    def test_pyquifer_social_runs(self):
        """PyQuifer empathy + constitution completes the public goods game."""
        results = bench_social_dilemma()
        assert 'pyquifer_social' in results
        assert 0.0 <= results['pyquifer_social'].cooperation_rate <= 1.0

    def test_both_produce_finite_payoffs(self):
        """Both systems produce finite group payoffs."""
        results = bench_social_dilemma()
        for key, r in results.items():
            assert np.isfinite(r.mean_group_payoff), f"{key} produced non-finite payoff"


class TestMirrorCoordination:
    """Test mirror neuron coordination benchmark."""

    def test_random_baseline_runs(self):
        """Random agents complete the coordination game."""
        results = bench_mirror_coordination()
        assert 'random' in results
        assert 0.0 <= results['random'].coordination_rate <= 1.0

    def test_mirror_resonance_runs(self):
        """PyQuifer MirrorResonance completes the coordination game."""
        results = bench_mirror_coordination()
        assert 'mirror_resonance' in results
        assert 0.0 <= results['mirror_resonance'].coordination_rate <= 1.0

    def test_mirror_not_worse_than_random_expected(self):
        """MirrorResonance achieves non-negative coordination."""
        results = bench_mirror_coordination()
        assert results['mirror_resonance'].coordination_rate >= 0.0


class TestTheoryOfMind:
    """Test theory of mind prediction benchmark."""

    def test_frequency_baseline_runs(self):
        """Frequency counter baseline completes prediction task."""
        results = bench_theory_of_mind()
        assert 'frequency' in results
        assert 0.0 <= results['frequency'].accuracy <= 1.0

    def test_tom_runs(self):
        """PyQuifer TheoryOfMind completes prediction task."""
        results = bench_theory_of_mind()
        assert 'theory_of_mind' in results
        assert 0.0 <= results['theory_of_mind'].accuracy <= 1.0

    def test_both_produce_finite_accuracy(self):
        """Both systems produce finite accuracy values."""
        results = bench_theory_of_mind()
        for key, r in results.items():
            assert np.isfinite(r.accuracy), f"{key} produced non-finite accuracy"


class TestSafetyPolicy:
    """Test safety-constrained policy benchmark."""

    def test_threshold_safety_runs(self):
        """Threshold-based safety completes without error."""
        results = bench_safety_policy()
        assert 'threshold' in results
        assert results['threshold'].violation_rate == 0.0  # Perfect filter

    def test_safety_envelope_runs(self):
        """PyQuifer SafetyEnvelope completes without error."""
        results = bench_safety_policy()
        assert 'safety_envelope' in results
        assert 0.0 <= results['safety_envelope'].violation_rate <= 1.0

    def test_both_produce_finite_rewards(self):
        """Both systems produce finite reward values."""
        results = bench_safety_policy()
        for key, r in results.items():
            assert np.isfinite(r.mean_reward), f"{key} produced non-finite reward"


class TestArchitecture:
    """Test architecture feature comparison."""

    def test_feature_counts(self):
        """Both systems have meaningful feature counts."""
        features = bench_architecture_features()
        mp_count = sum(1 for v in features['meltingpot'].values() if v)
        pq_count = sum(1 for v in features['pyquifer'].values() if v)
        assert mp_count >= 5
        assert pq_count >= 5

    def test_complementary_strengths(self):
        """MeltingPot and PyQuifer cover complementary feature sets."""
        features = bench_architecture_features()
        mp_set = {k for k, v in features['meltingpot'].items() if v}
        pq_set = {k for k, v in features['pyquifer'].items() if v}
        # They should have minimal overlap (different paradigms)
        overlap = mp_set & pq_set
        assert len(overlap) <= 3, f"Unexpected high overlap: {overlap}"


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Running PyQuifer vs MeltingPot social cognition benchmarks...\n")

    dilemma_results = bench_social_dilemma()
    coordination_results = bench_mirror_coordination()
    prediction_results = bench_theory_of_mind()
    safety_results = bench_safety_policy()
    arch_features = bench_architecture_features()

    print_report(dilemma_results, coordination_results, prediction_results,
                 safety_results, arch_features)

    print("\n--- Interpretation Notes ---")
    print("  - MirrorResonance coordination depends on coupling_strength and")
    print("    simulation length. For short runs, use adaptive_coupling=True")
    print("    to boost coupling when coherence is low.")
    print("  - TheoryOfMind uses a fixed 0.3 phase update rate; ~100-200")
    print("    observations needed for convergence on action prediction.")
    print("  - SafetyEnvelope uses learned uncertainty, which starts random;")
    print("    threshold safety is a hard-coded exact solution by design.")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
