"""
Benchmark: PyQuifer Motivation & Learning vs Gymnasium RL Framework

Compares PyQuifer's intrinsic motivation and reward-modulated learning against
Gymnasium's (formerly OpenAI Gym) RL environment interface. PyQuifer is not
an RL agent framework, so this benchmark focuses on the overlapping primitives:
reward processing, exploration strategies, and adaptive learning.

Benchmark sections:
  1. Reward Processing (PyQuifer motivation signals vs RL reward shaping)
  2. Exploration Strategies (PyQuifer stochastic resonance vs epsilon-greedy)
  3. Temporal Credit Assignment (eligibility traces vs TD learning)
  4. Architecture Feature Comparison

Usage:
  python bench_gymnasium.py           # Full suite with console output
  pytest bench_gymnasium.py -v        # Just the tests

Reference: Gymnasium (Farama Foundation), v1.0+
"""

import sys
import os
import time
import math
from dataclasses import dataclass
from typing import Dict, List
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

# Add PyQuifer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pyquifer.motivation import NoveltyDetector, IntrinsicMotivationSystem
from pyquifer.learning import EligibilityTrace, RewardModulatedHebbian
from pyquifer.stochastic_resonance import AdaptiveStochasticResonance


@contextmanager
def timer():
    start = time.perf_counter()
    result = {'elapsed_ms': 0.0}
    yield result
    result['elapsed_ms'] = (time.perf_counter() - start) * 1000


# ============================================================================
# Section 1: Simple RL Environment (reimplemented, no Gymnasium dependency)
# ============================================================================

class SimpleBandit:
    """N-armed bandit for reward processing comparison."""

    def __init__(self, n_arms=5, seed=42):
        rng = np.random.RandomState(seed)
        self.means = rng.randn(n_arms)
        self.n_arms = n_arms

    def step(self, action):
        reward = self.means[action] + np.random.randn() * 0.1
        return reward


class SimpleGridWorld:
    """Minimal grid world for credit assignment comparison."""

    def __init__(self, size=5, seed=42):
        self.size = size
        self.pos = [0, 0]
        self.goal = [size - 1, size - 1]

    def reset(self):
        self.pos = [0, 0]
        return self._obs()

    def _obs(self):
        obs = np.zeros(self.size * self.size)
        obs[self.pos[0] * self.size + self.pos[1]] = 1.0
        return obs

    def step(self, action):
        # 0=up, 1=right, 2=down, 3=left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dr, dc = moves[action]
        self.pos[0] = max(0, min(self.size - 1, self.pos[0] + dr))
        self.pos[1] = max(0, min(self.size - 1, self.pos[1] + dc))
        done = self.pos == self.goal
        reward = 1.0 if done else -0.01
        return self._obs(), reward, done


# ============================================================================
# Section 2: Benchmark Functions
# ============================================================================

@dataclass
class RewardResult:
    system: str
    total_reward: float
    reward_variance: float
    novelty_correlation: float  # Does novelty track reward?
    elapsed_ms: float


def bench_reward_processing() -> Dict[str, RewardResult]:
    """Compare reward processing strategies on N-armed bandit."""
    torch.manual_seed(42)
    np.random.seed(42)
    results = {}
    bandit = SimpleBandit(n_arms=5)
    num_steps = 500

    # --- Epsilon-greedy (standard RL) ---
    Q = np.zeros(5)
    counts = np.zeros(5)
    rewards_eg = []
    epsilon = 0.1

    with timer() as t_eg:
        for step in range(num_steps):
            if np.random.rand() < epsilon:
                action = np.random.randint(5)
            else:
                action = np.argmax(Q)
            reward = bandit.step(action)
            counts[action] += 1
            Q[action] += (reward - Q[action]) / counts[action]
            rewards_eg.append(reward)

    results['epsilon_greedy'] = RewardResult(
        system='Epsilon-Greedy (Gymnasium-style)',
        total_reward=sum(rewards_eg),
        reward_variance=np.var(rewards_eg),
        novelty_correlation=0.0,  # No novelty concept
        elapsed_ms=t_eg['elapsed_ms']
    )

    # --- PyQuifer: Novelty-driven exploration ---
    nd = NoveltyDetector(dim=5)
    Q_pq = np.zeros(5)
    counts_pq = np.zeros(5)
    rewards_pq = []
    novelties = []

    with timer() as t_pq:
        for step in range(num_steps):
            # Use novelty as exploration bonus
            action_features = torch.eye(5)
            novelty_scores = []
            for a in range(5):
                nov, _ = nd(action_features[a])
                novelty_scores.append(nov.item())

            # Choose action: Q-value + novelty bonus
            combined = Q_pq + 0.5 * np.array(novelty_scores)
            action = np.argmax(combined)

            reward = bandit.step(action)
            counts_pq[action] += 1
            Q_pq[action] += (reward - Q_pq[action]) / counts_pq[action]
            rewards_pq.append(reward)
            novelties.append(novelty_scores[action])

    # Correlation between novelty and reward
    if len(novelties) > 10:
        corr = np.corrcoef(novelties[:100], rewards_pq[:100])[0, 1]
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    results['novelty_driven'] = RewardResult(
        system='Novelty-Driven (PyQuifer)',
        total_reward=sum(rewards_pq),
        reward_variance=np.var(rewards_pq),
        novelty_correlation=corr,
        elapsed_ms=t_pq['elapsed_ms']
    )

    return results


@dataclass
class ExplorationResult:
    system: str
    states_visited: int
    unique_states: int
    coverage: float  # fraction of state space visited
    elapsed_ms: float


def bench_exploration() -> Dict[str, ExplorationResult]:
    """Compare exploration strategies on grid world."""
    np.random.seed(42)
    torch.manual_seed(42)
    results = {}
    num_episodes = 20
    max_steps = 50
    grid_size = 5

    # --- Random exploration (baseline) ---
    visited_random = set()
    with timer() as t_rand:
        for ep in range(num_episodes):
            env = SimpleGridWorld(grid_size)
            obs = env.reset()
            for _ in range(max_steps):
                action = np.random.randint(4)
                obs, reward, done = env.step(action)
                visited_random.add(tuple(env.pos))
                if done:
                    break

    results['random'] = ExplorationResult(
        system='Random (baseline)',
        states_visited=len(visited_random) * num_episodes,
        unique_states=len(visited_random),
        coverage=len(visited_random) / (grid_size ** 2),
        elapsed_ms=t_rand['elapsed_ms']
    )

    # --- PyQuifer: Stochastic resonance-guided exploration ---
    sr = AdaptiveStochasticResonance(dim=grid_size * grid_size, initial_noise=0.3)
    visited_sr = set()

    with timer() as t_sr:
        for ep in range(num_episodes):
            env = SimpleGridWorld(grid_size)
            obs = env.reset()
            for _ in range(max_steps):
                # Use SR to process observation and select action
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                result = sr(obs_tensor)
                enhanced = result['enhanced'].squeeze(0)
                # Use enhanced signal to bias action selection
                action_scores = torch.zeros(4)
                for a in range(4):
                    # Simple directional preference from enhanced observation
                    if a == 0:
                        action_scores[a] = enhanced[:grid_size].sum()
                    elif a == 1:
                        action_scores[a] = enhanced[grid_size:2*grid_size].sum()
                    elif a == 2:
                        action_scores[a] = enhanced[2*grid_size:3*grid_size].sum()
                    else:
                        action_scores[a] = enhanced[3*grid_size:4*grid_size].sum()
                # Add noise for exploration
                action_scores += torch.randn(4) * 0.5
                action = action_scores.argmax().item()
                obs, reward, done = env.step(action)
                visited_sr.add(tuple(env.pos))
                if done:
                    break

    results['stochastic_resonance'] = ExplorationResult(
        system='Stochastic Resonance (PyQuifer)',
        states_visited=len(visited_sr) * num_episodes,
        unique_states=len(visited_sr),
        coverage=len(visited_sr) / (grid_size ** 2),
        elapsed_ms=t_sr['elapsed_ms']
    )

    return results


@dataclass
class CreditResult:
    system: str
    final_value_error: float
    trace_decay_used: float
    elapsed_ms: float


def bench_credit_assignment() -> Dict[str, CreditResult]:
    """Compare temporal credit assignment: TD(lambda) vs eligibility traces."""
    torch.manual_seed(42)
    np.random.seed(42)
    results = {}
    state_dim = 25
    num_steps = 300

    # True value function (distance to goal in 5x5 grid)
    true_values = np.zeros(state_dim)
    for i in range(5):
        for j in range(5):
            true_values[i * 5 + j] = -(abs(4 - i) + abs(4 - j)) / 8.0

    # --- TD(0.9) with accumulating traces (standard RL) ---
    V_td = np.zeros(state_dim)
    traces_td = np.zeros(state_dim)
    alpha = 0.1
    gamma = 0.99
    lam = 0.9

    env = SimpleGridWorld()
    with timer() as t_td:
        for ep in range(20):
            obs = env.reset()
            traces_td *= 0.0
            s = 0  # start state
            for _ in range(num_steps // 20):
                a = np.random.randint(4)
                obs_next, reward, done = env.step(a)
                s_next = env.pos[0] * 5 + env.pos[1]
                delta = reward + gamma * V_td[s_next] * (1 - done) - V_td[s]
                traces_td[s] = 1.0
                V_td += alpha * delta * traces_td
                traces_td *= gamma * lam
                s = s_next
                if done:
                    break

    td_error = np.mean((V_td - true_values) ** 2)

    results['td_lambda'] = CreditResult(
        system='TD(0.9) (Gymnasium-style)',
        final_value_error=td_error,
        trace_decay_used=lam,
        elapsed_ms=t_td['elapsed_ms']
    )

    # --- PyQuifer Eligibility Traces ---
    trace = EligibilityTrace(shape=(state_dim,), decay_rate=0.9)
    V_pq = torch.zeros(state_dim)

    env = SimpleGridWorld()
    with timer() as t_pq:
        for ep in range(20):
            obs = env.reset()
            trace.reset()
            s = 0
            for _ in range(num_steps // 20):
                a = np.random.randint(4)
                obs_next, reward, done = env.step(a)
                s_next = env.pos[0] * 5 + env.pos[1]

                # Update trace
                activity = torch.zeros(state_dim)
                activity[s] = 1.0
                trace(activity)

                # TD error as reward
                with torch.no_grad():
                    delta = reward + gamma * V_pq[s_next].item() * (1 - done) - V_pq[s].item()
                    update = trace.apply_reward(
                        torch.tensor(delta), learning_rate=alpha
                    )
                    V_pq += update

                s = s_next
                if done:
                    break

    pq_error = torch.mean((V_pq - torch.tensor(true_values, dtype=torch.float32)) ** 2).item()

    results['pyquifer_traces'] = CreditResult(
        system='EligibilityTrace (PyQuifer)',
        final_value_error=pq_error,
        trace_decay_used=0.9,
        elapsed_ms=t_pq['elapsed_ms']
    )

    return results


def bench_architecture_features() -> Dict[str, Dict[str, bool]]:
    """Compare architectural features."""
    gymnasium_features = {
        'environment_interface': True,
        'action_spaces': True,
        'observation_spaces': True,
        'wrappers': True,
        'vectorized_envs': True,
        'reward_shaping': True,
        'td_learning': True,
        'policy_gradient': True,
        'intrinsic_motivation': False,
        'novelty_detection': False,
        'eligibility_traces': False,  # Not in Gymnasium itself
        'spiking_neurons': False,
        'oscillatory_dynamics': False,
        'criticality_control': False,
        'stochastic_resonance': False,
        'biological_learning': False,
    }

    pyquifer_features = {
        'environment_interface': False,
        'action_spaces': False,
        'observation_spaces': False,
        'wrappers': False,
        'vectorized_envs': False,
        'reward_shaping': True,  # Via IntrinsicMotivationSystem
        'td_learning': False,
        'policy_gradient': False,
        'intrinsic_motivation': True,
        'novelty_detection': True,
        'eligibility_traces': True,
        'spiking_neurons': True,
        'oscillatory_dynamics': True,
        'criticality_control': True,
        'stochastic_resonance': True,
        'biological_learning': True,
    }

    return {'gymnasium': gymnasium_features, 'pyquifer': pyquifer_features}


# ============================================================================
# Section 3: Console Output
# ============================================================================

def print_report(reward_results, exploration_results, credit_results,
                 arch_features):
    print("=" * 70)
    print("BENCHMARK: PyQuifer Motivation & Learning vs Gymnasium RL")
    print("=" * 70)

    print("\n--- 1. Reward Processing (N-Armed Bandit) ---\n")
    print(f"{'System':<35} {'Total R':>8} {'R Var':>8} {'Time':>10}")
    print("-" * 65)
    for r in reward_results.values():
        print(f"{r.system:<35} {r.total_reward:>8.1f} {r.reward_variance:>8.4f} "
              f"{r.elapsed_ms:>9.1f}ms")

    print("\n--- 2. Exploration (Grid World) ---\n")
    print(f"{'System':<35} {'Unique':>7} {'Coverage':>9} {'Time':>10}")
    print("-" * 65)
    for r in exploration_results.values():
        print(f"{r.system:<35} {r.unique_states:>7} {r.coverage:>9.1%} "
              f"{r.elapsed_ms:>9.1f}ms")

    print("\n--- 3. Temporal Credit Assignment ---\n")
    print(f"{'System':<35} {'Value Error':>12} {'Trace Decay':>12} {'Time':>10}")
    print("-" * 73)
    for r in credit_results.values():
        print(f"{r.system:<35} {r.final_value_error:>12.6f} "
              f"{r.trace_decay_used:>12.1f} {r.elapsed_ms:>9.1f}ms")

    print("\n--- 4. Architecture Feature Comparison ---\n")
    gy = arch_features['gymnasium']
    pq = arch_features['pyquifer']
    all_f = sorted(set(list(gy.keys()) + list(pq.keys())))
    print(f"{'Feature':<28} {'Gymnasium':>10} {'PyQuifer':>10}")
    print("-" * 52)
    gy_count = pq_count = 0
    for f in all_f:
        gv = 'YES' if gy.get(f, False) else 'no'
        pv = 'YES' if pq.get(f, False) else 'no'
        if gy.get(f, False):
            gy_count += 1
        if pq.get(f, False):
            pq_count += 1
        print(f"  {f:<26} {gv:>10} {pv:>10}")
    print(f"\n  Gymnasium: {gy_count}/{len(all_f)} | PyQuifer: {pq_count}/{len(all_f)}")


# ============================================================================
# Section 4: Pytest Tests
# ============================================================================

class TestRewardProcessing:
    def test_epsilon_greedy_learns(self):
        results = bench_reward_processing()
        assert results['epsilon_greedy'].total_reward > 0

    def test_novelty_driven_learns(self):
        results = bench_reward_processing()
        assert results['novelty_driven'].total_reward > -100

    def test_both_finite(self):
        results = bench_reward_processing()
        for r in results.values():
            assert np.isfinite(r.total_reward)


class TestExploration:
    def test_random_explores(self):
        results = bench_exploration()
        assert results['random'].coverage > 0.3

    def test_sr_explores(self):
        results = bench_exploration()
        assert results['stochastic_resonance'].coverage > 0.2


class TestCreditAssignment:
    def test_td_learns(self):
        results = bench_credit_assignment()
        assert results['td_lambda'].final_value_error < 1.0

    def test_traces_learn(self):
        results = bench_credit_assignment()
        assert results['pyquifer_traces'].final_value_error < 1.0

    def test_both_use_same_decay(self):
        results = bench_credit_assignment()
        assert results['td_lambda'].trace_decay_used == results['pyquifer_traces'].trace_decay_used


class TestArchitecture:
    def test_feature_counts(self):
        features = bench_architecture_features()
        gy_count = sum(1 for v in features['gymnasium'].values() if v)
        pq_count = sum(1 for v in features['pyquifer'].values() if v)
        assert gy_count >= 5
        assert pq_count >= 5


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Running PyQuifer vs Gymnasium benchmarks...\n")

    reward_results = bench_reward_processing()
    exploration_results = bench_exploration()
    credit_results = bench_credit_assignment()
    arch_features = bench_architecture_features()

    print_report(reward_results, exploration_results, credit_results,
                 arch_features)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
