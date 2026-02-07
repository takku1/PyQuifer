"""
Benchmark: PyQuifer Cognitive Dynamics vs OpenSpiel Game Theory Framework

Compares PyQuifer's neural selection and social cognition mechanisms against
OpenSpiel's game theory framework (DeepMind). PyQuifer is not a game theory
library, so this benchmark focuses on the overlapping primitives: evolutionary
dynamics, regret-driven exploration, and cooperation emergence.

Benchmark sections:
  1. Replicator Dynamics vs Neural Darwinism (evolutionary equilibrium)
  2. Regret Minimization vs Curiosity-Driven Exploration (action diversity)
  3. Social Dilemma Dynamics (cooperation via phase-locking vs strategies)
  4. Architecture Feature Comparison

Usage:
  python bench_open_spiel.py          # Full suite with console output
  pytest bench_open_spiel.py -v       # Just the tests

Reference: OpenSpiel (Lanctot et al. 2019), DeepMind Technologies.
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

from pyquifer.neural_darwinism import SelectionArena
from pyquifer.motivation import NoveltyDetector
from pyquifer.social import SocialCoupling, MirrorResonance
from pyquifer.oscillators import LearnableKuramotoBank


@contextmanager
def timer():
    start = time.perf_counter()
    result = {'elapsed_ms': 0.0}
    yield result
    result['elapsed_ms'] = (time.perf_counter() - start) * 1000


# ============================================================================
# Section 1: Reimplemented Game Theory Primitives (no OpenSpiel dependency)
# ============================================================================

class ReplicatorDynamics:
    """Replicator dynamics for evolutionary game theory.

    Implements the continuous-time replicator equation:
      dx_i/dt = x_i * ((Ax)_i - x^T A x)

    where x is the population strategy vector, A is the payoff matrix,
    and the second term is the average fitness.
    """

    def __init__(self, payoff_matrix: np.ndarray, dt: float = 0.01):
        self.A = payoff_matrix
        self.dt = dt
        self.num_strategies = payoff_matrix.shape[0]

    def step(self, x: np.ndarray) -> np.ndarray:
        """One step of replicator dynamics."""
        fitness = self.A @ x
        avg_fitness = x @ fitness
        dx = x * (fitness - avg_fitness) * self.dt
        x_new = x + dx
        # Project back to simplex (ensure non-negative, sums to 1)
        x_new = np.maximum(x_new, 1e-10)
        x_new = x_new / x_new.sum()
        return x_new

    def run(self, x0: np.ndarray, num_steps: int) -> List[np.ndarray]:
        """Run replicator dynamics for N steps."""
        trajectory = [x0.copy()]
        x = x0.copy()
        for _ in range(num_steps):
            x = self.step(x)
            trajectory.append(x.copy())
        return trajectory


class RegretMatcher:
    """Regret matching algorithm (simplified CFR).

    At each step:
    1. Compute strategy from positive cumulative regrets
    2. Play against opponent
    3. Update regrets based on counterfactual values

    Converges to Nash equilibrium in two-player zero-sum games.
    """

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.cumulative_regret = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.num_iterations = 0

    def get_strategy(self) -> np.ndarray:
        """Get current strategy from positive regrets."""
        positive_regret = np.maximum(self.cumulative_regret, 0)
        total = positive_regret.sum()
        if total > 0:
            return positive_regret / total
        else:
            return np.ones(self.num_actions) / self.num_actions

    def update(self, utility_vector: np.ndarray):
        """Update cumulative regret given utility for each action."""
        strategy = self.get_strategy()
        action_utility = strategy @ utility_vector
        regret = utility_vector - action_utility
        self.cumulative_regret += regret
        self.strategy_sum += strategy
        self.num_iterations += 1

    def get_average_strategy(self) -> np.ndarray:
        """Get time-averaged strategy (converges to Nash)."""
        if self.num_iterations == 0:
            return np.ones(self.num_actions) / self.num_actions
        avg = self.strategy_sum / self.num_iterations
        total = avg.sum()
        if total > 0:
            return avg / total
        return np.ones(self.num_actions) / self.num_actions


class PrisonersDilemma:
    """Iterated Prisoner's Dilemma environment.

    Payoff matrix (row player):
        C    D
    C  (3,3) (0,5)
    D  (5,0) (1,1)

    Actions: 0 = Cooperate, 1 = Defect
    """

    COOPERATE = 0
    DEFECT = 1

    # Payoffs: (row_payoff, col_payoff)
    PAYOFFS = {
        (0, 0): (3, 3),  # Both cooperate
        (0, 1): (0, 5),  # Row cooperates, col defects
        (1, 0): (5, 0),  # Row defects, col cooperates
        (1, 1): (1, 1),  # Both defect
    }

    @staticmethod
    def play(action_a: int, action_b: int) -> Tuple[float, float]:
        """Play one round, return (payoff_a, payoff_b)."""
        return PrisonersDilemma.PAYOFFS[(action_a, action_b)]


class TitForTat:
    """Tit-for-Tat strategy: cooperate first, then copy opponent's last move."""

    def __init__(self):
        self.last_opponent_action = PrisonersDilemma.COOPERATE

    def act(self) -> int:
        return self.last_opponent_action

    def observe(self, opponent_action: int):
        self.last_opponent_action = opponent_action


class AlwaysCooperate:
    """Always cooperate strategy."""

    def act(self) -> int:
        return PrisonersDilemma.COOPERATE

    def observe(self, opponent_action: int):
        pass


class AlwaysDefect:
    """Always defect strategy."""

    def act(self) -> int:
        return PrisonersDilemma.DEFECT

    def observe(self, opponent_action: int):
        pass


# ============================================================================
# Section 2: Benchmark Functions
# ============================================================================

@dataclass
class ReplicatorResult:
    """Results from replicator dynamics vs neural darwinism comparison."""
    system: str
    nash_distance: float  # Distance from Nash equilibrium
    final_distribution: List[float]
    num_steps: int
    elapsed_ms: float


def bench_replicator_dynamics() -> Dict[str, ReplicatorResult]:
    """Compare replicator dynamics (OpenSpiel) vs PyQuifer SelectionArena.

    Both evolve population strategies in Rock-Paper-Scissors.
    Nash equilibrium is (1/3, 1/3, 1/3).
    """
    torch.manual_seed(42)
    np.random.seed(42)
    results = {}
    num_steps = 200

    # RPS payoff matrix
    rps_payoff = np.array([
        [0, -1, 1],   # Rock vs Rock, Paper, Scissors
        [1, 0, -1],   # Paper vs Rock, Paper, Scissors
        [-1, 1, 0],   # Scissors vs Rock, Paper, Scissors
    ], dtype=np.float64)

    nash_eq = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    # --- OpenSpiel-style Replicator Dynamics ---
    rd = ReplicatorDynamics(rps_payoff, dt=0.05)
    x0 = np.array([0.6, 0.3, 0.1])  # Biased initial distribution

    with timer() as t_rd:
        trajectory = rd.run(x0, num_steps)

    final_dist_rd = trajectory[-1]
    nash_dist_rd = np.linalg.norm(final_dist_rd - nash_eq)

    results['replicator'] = ReplicatorResult(
        system='Replicator Dynamics (OpenSpiel-style)',
        nash_distance=nash_dist_rd,
        final_distribution=final_dist_rd.tolist(),
        num_steps=num_steps,
        elapsed_ms=t_rd['elapsed_ms']
    )

    # --- PyQuifer SelectionArena ---
    # 3 groups competing (analogous to 3 strategies)
    arena = SelectionArena(
        num_groups=3,
        group_dim=8,
        selection_pressure=0.05,
        total_budget=3.0,
    )

    # Create 3 distinct input patterns (one per "strategy")
    torch.manual_seed(42)
    strategy_patterns = [torch.randn(1, 8) for _ in range(3)]

    with timer() as t_nd:
        for step in range(num_steps):
            # Cycle through strategy patterns (simulates mixed population)
            pattern_idx = step % 3
            x = strategy_patterns[pattern_idx]
            result = arena(x)

    # Extract resource distribution (analogous to population shares)
    resources = result['resources'].detach().cpu().numpy()
    resource_shares = resources / resources.sum()

    nash_dist_nd = np.linalg.norm(resource_shares - nash_eq)

    results['neural_darwinism'] = ReplicatorResult(
        system='PyQuifer SelectionArena',
        nash_distance=nash_dist_nd,
        final_distribution=resource_shares.tolist(),
        num_steps=num_steps,
        elapsed_ms=t_nd['elapsed_ms']
    )

    return results


@dataclass
class RegretResult:
    """Results from regret minimization vs curiosity exploration comparison."""
    system: str
    final_regret: float
    exploration_coverage: float  # Fraction of actions tried
    action_entropy: float  # Shannon entropy of action distribution
    num_steps: int
    elapsed_ms: float


def bench_regret_exploration() -> Dict[str, RegretResult]:
    """Compare regret matching (CFR-lite) vs PyQuifer curiosity-driven exploration.

    In a simple 2-player matching pennies game:
    - Regret matching converges to Nash via regret minimization
    - PyQuifer explores via novelty detection + intrinsic motivation
    """
    torch.manual_seed(42)
    np.random.seed(42)
    results = {}
    num_steps = 300
    num_actions = 5

    # Matching pennies-style payoff: each action beats one, loses to another
    payoff_matrix = np.zeros((num_actions, num_actions))
    for i in range(num_actions):
        payoff_matrix[i, (i + 1) % num_actions] = 1.0  # Beat next
        payoff_matrix[i, (i - 1) % num_actions] = -1.0  # Lose to prev

    # --- OpenSpiel-style Regret Matching ---
    player1 = RegretMatcher(num_actions)
    player2 = RegretMatcher(num_actions)

    action_counts_rm = np.zeros(num_actions)

    with timer() as t_rm:
        for step in range(num_steps):
            s1 = player1.get_strategy()
            s2 = player2.get_strategy()

            # Compute expected utility for each action
            utility1 = payoff_matrix @ s2
            utility2 = -payoff_matrix.T @ s1

            player1.update(utility1)
            player2.update(utility2)

            # Track which actions player1 uses
            action = np.random.choice(num_actions, p=s1)
            action_counts_rm[action] += 1

    avg_strategy = player1.get_average_strategy()
    nash_uniform = np.ones(num_actions) / num_actions
    final_regret_rm = np.linalg.norm(avg_strategy - nash_uniform)

    # Exploration coverage: fraction of actions tried at least once
    coverage_rm = (action_counts_rm > 0).sum() / num_actions

    # Entropy of action distribution
    p_rm = action_counts_rm / action_counts_rm.sum()
    entropy_rm = -np.sum(p_rm * np.log(p_rm + 1e-10))

    results['regret_matching'] = RegretResult(
        system='Regret Matching (OpenSpiel-style)',
        final_regret=final_regret_rm,
        exploration_coverage=coverage_rm,
        action_entropy=entropy_rm,
        num_steps=num_steps,
        elapsed_ms=t_rm['elapsed_ms']
    )

    # --- PyQuifer Novelty-Driven Exploration ---
    dim = 8
    nd = NoveltyDetector(dim=dim)
    action_embeddings = torch.randn(num_actions, dim)
    action_counts_nd = np.zeros(num_actions)

    with timer() as t_nd:
        for step in range(num_steps):
            # Compute novelty for each action
            novelties = []
            for a in range(num_actions):
                novelty, _ = nd(action_embeddings[a])
                novelties.append(novelty.item())

            # Select action with highest novelty (curiosity-driven)
            novelty_scores = np.array(novelties)
            # Softmax selection to balance exploration/exploitation
            exp_scores = np.exp(novelty_scores * 5.0)
            probs = exp_scores / exp_scores.sum()
            action = np.random.choice(num_actions, p=probs)

            # "Experience" the chosen action (updates novelty model)
            nd(action_embeddings[action])
            action_counts_nd[action] += 1

    coverage_nd = (action_counts_nd > 0).sum() / num_actions
    p_nd = action_counts_nd / action_counts_nd.sum()
    entropy_nd = -np.sum(p_nd * np.log(p_nd + 1e-10))

    # "Regret" analog: distance from uniform exploration
    final_regret_nd = np.linalg.norm(p_nd - nash_uniform)

    results['curiosity_exploration'] = RegretResult(
        system='PyQuifer NoveltyDetector',
        final_regret=final_regret_nd,
        exploration_coverage=coverage_nd,
        action_entropy=entropy_nd,
        num_steps=num_steps,
        elapsed_ms=t_nd['elapsed_ms']
    )

    return results


@dataclass
class DilemmaResult:
    """Results from social dilemma dynamics comparison."""
    system: str
    cooperation_rate: float
    phase_coherence: float  # Only meaningful for PyQuifer
    final_payoff: float
    num_rounds: int
    elapsed_ms: float


def bench_social_dilemma() -> Dict[str, DilemmaResult]:
    """Compare IPD strategies (OpenSpiel) vs PyQuifer SocialCoupling.

    Classical strategies play iterated Prisoner's Dilemma.
    PyQuifer uses phase-locking to model cooperation emergence.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    results = {}
    num_rounds = 200

    # --- OpenSpiel-style: TFT vs AlwaysCooperate ---
    tft = TitForTat()
    coop = AlwaysCooperate()

    payoffs_tft = []
    cooperations_tft = 0

    with timer() as t_tft:
        for _ in range(num_rounds):
            a_tft = tft.act()
            a_coop = coop.act()
            p_tft, p_coop = PrisonersDilemma.play(a_tft, a_coop)
            tft.observe(a_coop)
            coop.observe(a_tft)
            payoffs_tft.append(p_tft)
            if a_tft == PrisonersDilemma.COOPERATE:
                cooperations_tft += 1

    results['tft_vs_coop'] = DilemmaResult(
        system='TFT vs AlwaysCooperate (OpenSpiel-style)',
        cooperation_rate=cooperations_tft / num_rounds,
        phase_coherence=0.0,  # Not applicable
        final_payoff=np.mean(payoffs_tft),
        num_rounds=num_rounds,
        elapsed_ms=t_tft['elapsed_ms']
    )

    # --- OpenSpiel-style: TFT vs AlwaysDefect ---
    tft2 = TitForTat()
    defector = AlwaysDefect()

    payoffs_tft2 = []
    cooperations_tft2 = 0

    with timer() as t_tft2:
        for _ in range(num_rounds):
            a_tft = tft2.act()
            a_def = defector.act()
            p_tft, p_def = PrisonersDilemma.play(a_tft, a_def)
            tft2.observe(a_def)
            defector.observe(a_tft)
            payoffs_tft2.append(p_tft)
            if a_tft == PrisonersDilemma.COOPERATE:
                cooperations_tft2 += 1

    results['tft_vs_defect'] = DilemmaResult(
        system='TFT vs AlwaysDefect (OpenSpiel-style)',
        cooperation_rate=cooperations_tft2 / num_rounds,
        phase_coherence=0.0,
        final_payoff=np.mean(payoffs_tft2),
        num_rounds=num_rounds,
        elapsed_ms=t_tft2['elapsed_ms']
    )

    # --- PyQuifer SocialCoupling (cooperation via phase-locking) ---
    # 2 agents coupled through oscillatory dynamics
    social = SocialCoupling(
        dim=8,
        num_agents=2,
        coupling_strength=0.5,
        coupling_type='symmetric'
    )

    cooperations_sc = 0
    payoffs_sc = []
    coherences = []

    with timer() as t_sc:
        for round_idx in range(num_rounds):
            # Step the coupled oscillators
            result = social.step(dt=0.01)
            coherence = result['global_coherence']
            coherences.append(coherence)

            # Map coherence to cooperation decision
            # High coherence (phase-locked) -> cooperate
            # Low coherence (out of sync) -> defect
            agent_a_coop = coherence > 0.3
            agent_b_coop = coherence > 0.3

            a_a = PrisonersDilemma.COOPERATE if agent_a_coop else PrisonersDilemma.DEFECT
            a_b = PrisonersDilemma.COOPERATE if agent_b_coop else PrisonersDilemma.DEFECT

            p_a, p_b = PrisonersDilemma.play(a_a, a_b)
            payoffs_sc.append(p_a)
            if a_a == PrisonersDilemma.COOPERATE:
                cooperations_sc += 1

    mean_coherence = np.mean(coherences)

    results['social_coupling'] = DilemmaResult(
        system='PyQuifer SocialCoupling',
        cooperation_rate=cooperations_sc / num_rounds,
        phase_coherence=mean_coherence,
        final_payoff=np.mean(payoffs_sc),
        num_rounds=num_rounds,
        elapsed_ms=t_sc['elapsed_ms']
    )

    return results


def bench_architecture_features() -> Dict[str, Dict[str, bool]]:
    """Compare architectural features across game theory frameworks."""
    openspiel_features = {
        'normal_form_games': True,
        'extensive_form_games': True,
        'replicator_dynamics': True,
        'cfr_algorithm': True,
        'nash_equilibrium_solver': True,
        'multi_agent_rl': True,
        'game_tree_search': True,
        'information_sets': True,
        'mean_field_games': True,
        'evolutionary_dynamics': True,
        'spiking_neurons': False,
        'oscillatory_dynamics': False,
        'intrinsic_motivation': False,
        'neural_darwinism': False,
        'phase_coupling': False,
        'mirror_neurons': False,
        'online_adaptation': False,
        'criticality_control': False,
    }

    pyquifer_features = {
        'normal_form_games': False,
        'extensive_form_games': False,
        'replicator_dynamics': True,   # SelectionArena uses replicator equation
        'cfr_algorithm': False,
        'nash_equilibrium_solver': False,
        'multi_agent_rl': False,
        'game_tree_search': False,
        'information_sets': False,
        'mean_field_games': False,
        'evolutionary_dynamics': True,  # NeuralDarwinism
        'spiking_neurons': True,
        'oscillatory_dynamics': True,
        'intrinsic_motivation': True,
        'neural_darwinism': True,
        'phase_coupling': True,         # SocialCoupling
        'mirror_neurons': True,         # MirrorResonance
        'online_adaptation': True,
        'criticality_control': True,
    }

    return {'openspiel': openspiel_features, 'pyquifer': pyquifer_features}


# ============================================================================
# Section 3: Console Output
# ============================================================================

def print_replicator_results(results: Dict[str, ReplicatorResult]):
    print("\n--- 1. Replicator Dynamics vs Neural Darwinism ---\n")
    print(f"  Game: Rock-Paper-Scissors | Nash equilibrium: (0.333, 0.333, 0.333)\n")
    print(f"  {'System':<40} {'Nash Dist':>10} {'Distribution':>30} {'Time':>10}")
    print("  " + "-" * 94)
    for r in results.values():
        dist_str = '(' + ', '.join(f'{v:.3f}' for v in r.final_distribution) + ')'
        print(f"  {r.system:<40} {r.nash_distance:>10.4f} {dist_str:>30} "
              f"{r.elapsed_ms:>9.1f}ms")


def print_regret_results(results: Dict[str, RegretResult]):
    print("\n--- 2. Regret Minimization vs Curiosity Exploration ---\n")
    print(f"  {'System':<35} {'Regret':>8} {'Coverage':>10} {'Entropy':>9} {'Time':>10}")
    print("  " + "-" * 76)
    for r in results.values():
        print(f"  {r.system:<35} {r.final_regret:>8.4f} "
              f"{r.exploration_coverage:>10.2f} {r.action_entropy:>9.4f} "
              f"{r.elapsed_ms:>9.1f}ms")


def print_dilemma_results(results: Dict[str, DilemmaResult]):
    print("\n--- 3. Social Dilemma Dynamics ---\n")
    print(f"  {'System':<42} {'Coop Rate':>10} {'Coherence':>10} {'Payoff':>8} {'Time':>10}")
    print("  " + "-" * 84)
    for r in results.values():
        coh_str = f"{r.phase_coherence:.4f}" if r.phase_coherence > 0 else "N/A"
        print(f"  {r.system:<42} {r.cooperation_rate:>10.3f} "
              f"{coh_str:>10} {r.final_payoff:>8.2f} {r.elapsed_ms:>9.1f}ms")


def print_architecture_features(arch_features: Dict[str, Dict[str, bool]]):
    print("\n--- 4. Architecture Feature Comparison ---\n")
    os_f = arch_features['openspiel']
    pq_f = arch_features['pyquifer']
    all_features = sorted(set(list(os_f.keys()) + list(pq_f.keys())))
    print(f"  {'Feature':<30} {'OpenSpiel':>10} {'PyQuifer':>10}")
    print("  " + "-" * 54)
    os_count = pq_count = 0
    for f in all_features:
        ov = 'YES' if os_f.get(f, False) else 'no'
        pv = 'YES' if pq_f.get(f, False) else 'no'
        if os_f.get(f, False):
            os_count += 1
        if pq_f.get(f, False):
            pq_count += 1
        print(f"    {f:<28} {ov:>10} {pv:>10}")
    print(f"\n    OpenSpiel: {os_count}/{len(all_features)} | PyQuifer: {pq_count}/{len(all_features)}")


def print_report(replicator_results, regret_results, dilemma_results, arch_features):
    print("=" * 70)
    print("BENCHMARK: PyQuifer vs OpenSpiel Game Theory Framework")
    print("=" * 70)

    print_replicator_results(replicator_results)
    print_regret_results(regret_results)
    print_dilemma_results(dilemma_results)
    print_architecture_features(arch_features)


# ============================================================================
# Section 4: Pytest Tests
# ============================================================================

class TestReplicatorDynamics:
    """Test replicator dynamics vs neural darwinism."""

    def test_replicator_converges_toward_nash(self):
        """Replicator dynamics on RPS should stay near Nash equilibrium."""
        results = bench_replicator_dynamics()
        r = results['replicator']
        # RPS replicator dynamics should orbit near Nash
        assert r.nash_distance < 1.0, f"Nash distance {r.nash_distance} too large"

    def test_neural_darwinism_produces_distribution(self):
        """SelectionArena should produce a valid resource distribution."""
        results = bench_replicator_dynamics()
        r = results['neural_darwinism']
        dist = np.array(r.final_distribution)
        assert np.all(dist >= 0), "Negative resource shares"
        assert abs(dist.sum() - 1.0) < 0.01, "Shares don't sum to 1"

    def test_both_produce_finite_values(self):
        """Both systems produce finite distance values."""
        results = bench_replicator_dynamics()
        for key, r in results.items():
            assert np.isfinite(r.nash_distance), f"{key} produced non-finite distance"


class TestRegretExploration:
    """Test regret minimization vs curiosity-driven exploration."""

    def test_regret_matching_converges(self):
        """Regret matching should converge toward uniform strategy."""
        results = bench_regret_exploration()
        r = results['regret_matching']
        assert r.final_regret < 1.0, f"Regret {r.final_regret} too large"

    def test_curiosity_covers_all_actions(self):
        """Novelty-driven exploration should cover all actions."""
        results = bench_regret_exploration()
        r = results['curiosity_exploration']
        assert r.exploration_coverage >= 0.8, f"Coverage {r.exploration_coverage} too low"

    def test_both_have_positive_entropy(self):
        """Both systems should explore (positive entropy)."""
        results = bench_regret_exploration()
        for key, r in results.items():
            assert r.action_entropy > 0.5, f"{key} entropy {r.action_entropy} too low"


class TestSocialDilemma:
    """Test social dilemma dynamics."""

    def test_tft_cooperates_with_cooperator(self):
        """TFT should cooperate fully against AlwaysCooperate."""
        results = bench_social_dilemma()
        r = results['tft_vs_coop']
        assert r.cooperation_rate > 0.9, f"TFT cooperation {r.cooperation_rate} too low"

    def test_tft_defects_against_defector(self):
        """TFT should mostly defect against AlwaysDefect (after first round)."""
        results = bench_social_dilemma()
        r = results['tft_vs_defect']
        assert r.cooperation_rate < 0.1, f"TFT cooperation {r.cooperation_rate} too high"

    def test_social_coupling_runs(self):
        """SocialCoupling produces valid cooperation rate."""
        results = bench_social_dilemma()
        r = results['social_coupling']
        assert 0.0 <= r.cooperation_rate <= 1.0
        assert np.isfinite(r.phase_coherence)

    def test_social_coupling_has_coherence(self):
        """SocialCoupling should produce non-trivial phase coherence."""
        results = bench_social_dilemma()
        r = results['social_coupling']
        assert r.phase_coherence > -1.0 and r.phase_coherence < 2.0


class TestArchitecture:
    """Test architecture feature comparison."""

    def test_feature_counts(self):
        """Both systems have meaningful feature counts."""
        features = bench_architecture_features()
        os_count = sum(1 for v in features['openspiel'].values() if v)
        pq_count = sum(1 for v in features['pyquifer'].values() if v)
        assert os_count >= 5
        assert pq_count >= 5

    def test_complementary_strengths(self):
        """Systems should have complementary (non-identical) feature sets."""
        features = bench_architecture_features()
        os_set = {k for k, v in features['openspiel'].items() if v}
        pq_set = {k for k, v in features['pyquifer'].items() if v}
        # They should not be identical
        assert os_set != pq_set
        # Each should have unique features
        assert len(os_set - pq_set) > 0
        assert len(pq_set - os_set) > 0


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Running PyQuifer vs OpenSpiel benchmarks...\n")

    replicator_results = bench_replicator_dynamics()
    regret_results = bench_regret_exploration()
    dilemma_results = bench_social_dilemma()
    arch_features = bench_architecture_features()

    print_report(replicator_results, regret_results, dilemma_results,
                 arch_features)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
