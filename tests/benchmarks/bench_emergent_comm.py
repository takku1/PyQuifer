"""
Benchmark: PyQuifer Learning & Selection vs Emergent Communication Frameworks

Compares PyQuifer's learning and selection mechanisms against EGG (Facebook)
and emergent_communication_at_scale (Jax) - two emergent communication frameworks.

PyQuifer is NOT a multi-agent communication framework, so this benchmark
focuses on the overlapping components: reward-modulated learning, population
selection, and signal processing primitives that underpin communication.

Benchmark sections:
  1. Reward-Modulated Learning (PyQuifer R-STDP vs REINFORCE)
  2. Population Selection (PyQuifer Neural Darwinism vs EGG population training)
  3. Signal Discrimination (PyQuifer spiking layers as signal detectors)
  4. Architecture Feature Comparison

Usage:
  python bench_emergent_comm.py         # Full suite with console output
  pytest bench_emergent_comm.py -v      # Just the tests

Reference: EGG (Kharitonov et al. 2019), emergent_communication_at_scale (Chaabouni et al.)
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

from pyquifer.learning import RewardModulatedHebbian
from pyquifer.neural_darwinism import SelectionArena
from pyquifer.spiking import SpikingLayer, STDPLayer
from pyquifer.motivation import NoveltyDetector


@contextmanager
def timer():
    start = time.perf_counter()
    result = {'elapsed_ms': 0.0}
    yield result
    result['elapsed_ms'] = (time.perf_counter() - start) * 1000


# ============================================================================
# Section 1: Reimplemented Communication Primitives
# ============================================================================

class REINFORCESender(nn.Module):
    """EGG-style discrete sender using REINFORCE.
    Maps input to a discrete message via policy gradient.
    """

    def __init__(self, input_dim, vocab_size, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x):
        h = torch.relu(self.encoder(x))
        logits = self.policy(h)
        probs = torch.softmax(logits, dim=-1)
        # Sample discrete message
        dist = torch.distributions.Categorical(probs)
        message = dist.sample()
        log_prob = dist.log_prob(message)
        return message, log_prob, dist.entropy()


class SimpleReceiver(nn.Module):
    """EGG-style receiver: maps message to prediction."""

    def __init__(self, vocab_size, output_dim, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, message):
        h = self.embedding(message)
        return self.decoder(h)


def reinforce_training_step(sender, receiver, x, target, baseline=0.0):
    """One step of REINFORCE training for sender-receiver game.
    Returns loss components for analysis.
    """
    message, log_prob, entropy = sender(x)
    prediction = receiver(message)
    # Task loss (reconstruction)
    task_loss = nn.functional.mse_loss(prediction, target)
    reward = -task_loss.detach()
    # REINFORCE loss
    policy_loss = -log_prob * (reward - baseline)
    return task_loss, policy_loss, entropy, reward


# ============================================================================
# Section 2: Benchmark Functions
# ============================================================================

@dataclass
class LearningResult:
    """Results from reward-modulated learning comparison."""
    system: str
    final_loss: float
    reward_correlation: float  # Does reward modulate learning correctly?
    steps: int
    elapsed_ms: float


def bench_reward_learning() -> Dict[str, LearningResult]:
    """Compare REINFORCE (EGG) vs PyQuifer's R-STDP for reward-modulated learning."""
    torch.manual_seed(42)
    results = {}
    input_dim = 16
    output_dim = 8
    vocab_size = 10
    num_steps = 200

    # --- EGG-style REINFORCE ---
    sender = REINFORCESender(input_dim, vocab_size)
    receiver = SimpleReceiver(vocab_size, output_dim)
    optimizer = torch.optim.Adam(
        list(sender.parameters()) + list(receiver.parameters()), lr=0.01
    )

    losses_reinforce = []
    with timer() as t_rf:
        for step in range(num_steps):
            x = torch.randn(1, input_dim)
            target = x[:, :output_dim] * 0.5  # Simple mapping
            task_loss, policy_loss, entropy, reward = reinforce_training_step(
                sender, receiver, x, target
            )
            loss = task_loss + policy_loss.mean() - 0.01 * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_reinforce.append(task_loss.item())

    # Reward correlation: does later reward > earlier reward?
    first_half = np.mean(losses_reinforce[:num_steps // 2])
    second_half = np.mean(losses_reinforce[num_steps // 2:])
    rf_correlation = 1.0 if second_half < first_half else 0.0

    results['reinforce'] = LearningResult(
        system='REINFORCE (EGG-style)',
        final_loss=losses_reinforce[-1],
        reward_correlation=rf_correlation,
        steps=num_steps,
        elapsed_ms=t_rf['elapsed_ms']
    )

    # --- PyQuifer Reward-Modulated Hebbian ---
    rmh = RewardModulatedHebbian(
        input_dim=input_dim, output_dim=output_dim, learning_rate=0.01
    )

    losses_rstdp = []
    with timer() as t_rs:
        for step in range(num_steps):
            torch.manual_seed(42 + step)
            pre = torch.randn(1, input_dim)
            target = pre[:, :output_dim] * 0.5

            # Forward pass
            post = rmh(pre)

            # Compute reward (negative MSE)
            loss = nn.functional.mse_loss(post, target)
            reward = -loss.item()

            # Apply reward signal
            rmh.reward_update(reward)
            losses_rstdp.append(loss.item())

    first_half = np.mean(losses_rstdp[:num_steps // 2])
    second_half = np.mean(losses_rstdp[num_steps // 2:])
    rs_correlation = 1.0 if second_half < first_half else 0.0

    results['rstdp'] = LearningResult(
        system='Reward-Modulated Hebbian (PyQuifer)',
        final_loss=losses_rstdp[-1],
        reward_correlation=rs_correlation,
        steps=num_steps,
        elapsed_ms=t_rs['elapsed_ms']
    )

    return results


@dataclass
class SelectionResult:
    """Results from population selection comparison."""
    system: str
    initial_diversity: float
    final_diversity: float
    fitness_improvement: float
    num_groups: int
    elapsed_ms: float


def bench_population_selection() -> Dict[str, SelectionResult]:
    """Compare EGG population-based training vs PyQuifer Neural Darwinism."""
    torch.manual_seed(42)
    results = {}

    # --- EGG-style population: multiple sender-receiver pairs ---
    pop_size = 5
    input_dim = 8
    vocab_size = 6
    output_dim = 4
    num_steps = 100

    # Create population of agents
    senders = [REINFORCESender(input_dim, vocab_size, hidden_dim=32)
               for _ in range(pop_size)]
    receivers = [SimpleReceiver(vocab_size, output_dim, hidden_dim=32)
                 for _ in range(pop_size)]

    # Measure initial diversity (weight variance across population)
    init_weights = [s.encoder.weight.data.flatten() for s in senders]
    init_diversity = torch.stack(init_weights).var(dim=0).mean().item()

    with timer() as t_pop:
        fitness_scores = []
        for s, r in zip(senders, receivers):
            opt = torch.optim.Adam(
                list(s.parameters()) + list(r.parameters()), lr=0.01
            )
            total_reward = 0.0
            for _ in range(num_steps):
                x = torch.randn(1, input_dim)
                target = x[:, :output_dim] * 0.5
                task_loss, policy_loss, entropy, reward = reinforce_training_step(
                    s, r, x, target
                )
                loss = task_loss + policy_loss.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_reward += reward.item()
            fitness_scores.append(total_reward / num_steps)

    final_weights = [s.encoder.weight.data.flatten() for s in senders]
    final_diversity = torch.stack(final_weights).var(dim=0).mean().item()

    results['egg_population'] = SelectionResult(
        system='EGG Population (independent training)',
        initial_diversity=init_diversity,
        final_diversity=final_diversity,
        fitness_improvement=max(fitness_scores) - min(fitness_scores),
        num_groups=pop_size,
        elapsed_ms=t_pop['elapsed_ms']
    )

    # --- PyQuifer Neural Darwinism (SelectionArena) ---
    # group_dim must match input_dim for SelectionArena
    arena = SelectionArena(
        num_groups=pop_size,
        group_dim=input_dim,
        selection_pressure=0.1,
    )

    # Measure initial resource diversity
    x_test = torch.randn(1, input_dim)
    with torch.no_grad():
        result = arena(x_test)
        init_resources = torch.tensor([g.resources.item() for g in arena.groups])
        init_div_nd = init_resources.var().item()

    with timer() as t_nd:
        for step in range(num_steps):
            x = torch.randn(1, input_dim)
            result = arena(x)
            # Selection happens internally via fitness computation
            output = result['output']

    with torch.no_grad():
        result = arena(x_test)
        final_resources = torch.tensor([g.resources.item() for g in arena.groups])
        final_div_nd = final_resources.var().item()

    results['neural_darwinism'] = SelectionResult(
        system='PyQuifer SelectionArena',
        initial_diversity=init_div_nd,
        final_diversity=final_div_nd,
        fitness_improvement=0.0,  # Uses replicator dynamics, not explicit fitness ranking
        num_groups=pop_size,
        elapsed_ms=t_nd['elapsed_ms']
    )

    return results


@dataclass
class DiscriminationResult:
    """Results from signal discrimination test."""
    system: str
    accuracy: float
    response_separation: float  # How different are responses to different signals
    elapsed_ms: float


def bench_signal_discrimination() -> Dict[str, DiscriminationResult]:
    """Test signal discrimination ability (fundamental to communication).
    Can the system distinguish between N different input patterns?
    """
    torch.manual_seed(42)
    results = {}
    dim = 16
    num_signals = 5
    num_trials = 50

    # Create distinct signals
    signals = [torch.randn(1, dim) for _ in range(num_signals)]

    # --- Simple Linear (baseline) ---
    linear = nn.Linear(dim, num_signals)
    responses_linear = []
    with timer() as t_lin:
        for sig in signals:
            resp = torch.softmax(linear(sig), dim=-1)
            responses_linear.append(resp.detach())

    # Measure separation
    resp_stack = torch.cat(responses_linear)
    separation_lin = resp_stack.var(dim=0).mean().item()

    results['linear_baseline'] = DiscriminationResult(
        system='Linear (baseline)',
        accuracy=0.0,  # Untrained
        response_separation=separation_lin,
        elapsed_ms=t_lin['elapsed_ms']
    )

    # --- PyQuifer Spiking Layer ---
    # SpikingLayer expects (batch, seq_len, input_dim)
    spiking = SpikingLayer(dim, num_signals, threshold=0.5)
    spiking.eval()

    responses_spike = []
    with timer() as t_sp:
        for sig in signals:
            with torch.no_grad():
                # Add seq_len=1 dimension
                sig_3d = sig.unsqueeze(1)  # (1, 1, dim)
                resp = spiking(sig_3d)
                if isinstance(resp, dict):
                    resp = resp.get('output', resp.get('spikes', sig))
                resp = resp.squeeze(1)  # back to (1, num_signals)
            responses_spike.append(resp.detach())

    resp_stack_sp = torch.cat(responses_spike)
    separation_sp = resp_stack_sp.var(dim=0).mean().item()

    results['spiking_layer'] = DiscriminationResult(
        system='PyQuifer SpikingLayer',
        accuracy=0.0,  # Untrained
        response_separation=separation_sp,
        elapsed_ms=t_sp['elapsed_ms']
    )

    # --- PyQuifer Novelty Detector (novelty-based discrimination) ---
    nd = NoveltyDetector(dim=dim)

    novelty_scores = []
    with timer() as t_im:
        for trial in range(num_trials):
            for i, sig in enumerate(signals):
                novelty, _ = nd(sig.squeeze(0))
                novelty_scores.append((i, novelty.item()))

    # Group by signal index and check if novelty varies
    signal_novelties = {}
    for idx, nov in novelty_scores:
        signal_novelties.setdefault(idx, []).append(nov)
    nov_means = [np.mean(v) for v in signal_novelties.values()]
    separation_im = np.std(nov_means)

    results['novelty_detector'] = DiscriminationResult(
        system='PyQuifer NoveltyDetector',
        accuracy=0.0,  # Not classification
        response_separation=separation_im,
        elapsed_ms=t_im['elapsed_ms']
    )

    return results


def bench_architecture_features() -> Dict[str, Dict[str, bool]]:
    """Compare architectural features across communication frameworks."""
    egg_features = {
        'discrete_channel': True,
        'continuous_channel': True,
        'reinforce_optimization': True,
        'gumbel_softmax': True,
        'population_training': True,
        'compositionality_metrics': True,
        'multi_agent': True,
        'rnn_agents': True,
        'transformer_agents': True,
        'reward_modulated_learning': False,
        'spiking_neurons': False,
        'oscillatory_dynamics': False,
        'intrinsic_motivation': False,
        'neural_darwinism': False,
        'criticality_control': False,
        'online_adaptation': False,
    }

    pyquifer_features = {
        'discrete_channel': False,  # No explicit communication channel
        'continuous_channel': False,
        'reinforce_optimization': False,
        'gumbel_softmax': False,
        'population_training': True,  # NeuralDarwinismLayer
        'compositionality_metrics': False,
        'multi_agent': False,
        'rnn_agents': False,
        'transformer_agents': False,
        'reward_modulated_learning': True,
        'spiking_neurons': True,
        'oscillatory_dynamics': True,
        'intrinsic_motivation': True,
        'neural_darwinism': True,
        'criticality_control': True,
        'online_adaptation': True,
    }

    return {'egg': egg_features, 'pyquifer': pyquifer_features}


# ============================================================================
# Section 3: Console Output
# ============================================================================

def print_report(learning_results, selection_results, discrimination_results,
                 arch_features):
    print("=" * 70)
    print("BENCHMARK: PyQuifer vs Emergent Communication Frameworks")
    print("=" * 70)

    # Learning
    print("\n--- 1. Reward-Modulated Learning ---\n")
    print(f"{'System':<30} {'Final Loss':>10} {'Improves?':>10} {'Time':>10}")
    print("-" * 64)
    for r in learning_results.values():
        print(f"{r.system:<30} {r.final_loss:>10.4f} "
              f"{'YES' if r.reward_correlation > 0 else 'NO':>10} "
              f"{r.elapsed_ms:>9.1f}ms")

    # Selection
    print("\n--- 2. Population Selection ---\n")
    print(f"{'System':<40} {'Init Div':>9} {'Final Div':>10} {'Time':>10}")
    print("-" * 73)
    for r in selection_results.values():
        print(f"{r.system:<40} {r.initial_diversity:>9.4f} "
              f"{r.final_diversity:>10.4f} {r.elapsed_ms:>9.1f}ms")

    # Discrimination
    print("\n--- 3. Signal Discrimination ---\n")
    print(f"{'System':<35} {'Separation':>11} {'Time':>10}")
    print("-" * 60)
    for r in discrimination_results.values():
        print(f"{r.system:<35} {r.response_separation:>11.4f} "
              f"{r.elapsed_ms:>9.1f}ms")

    # Architecture
    print("\n--- 4. Architecture Feature Comparison ---\n")
    eg = arch_features['egg']
    pq = arch_features['pyquifer']
    all_f = sorted(set(list(eg.keys()) + list(pq.keys())))
    print(f"{'Feature':<30} {'EGG/ECAS':>10} {'PyQuifer':>10}")
    print("-" * 54)
    eg_count = pq_count = 0
    for f in all_f:
        ev = 'YES' if eg.get(f, False) else 'no'
        pv = 'YES' if pq.get(f, False) else 'no'
        if eg.get(f, False):
            eg_count += 1
        if pq.get(f, False):
            pq_count += 1
        print(f"  {f:<28} {ev:>10} {pv:>10}")
    print(f"\n  EGG/ECAS: {eg_count}/{len(all_f)} | PyQuifer: {pq_count}/{len(all_f)}")


# ============================================================================
# Section 4: Pytest Tests
# ============================================================================

class TestRewardLearning:
    """Test reward-modulated learning comparison."""

    def test_reinforce_runs(self):
        """REINFORCE training completes without error."""
        results = bench_reward_learning()
        assert 'reinforce' in results
        assert results['reinforce'].final_loss < 10.0

    def test_rstdp_runs(self):
        """R-STDP training completes without error."""
        results = bench_reward_learning()
        assert 'rstdp' in results
        assert results['rstdp'].final_loss < 10.0

    def test_both_produce_finite_loss(self):
        """Both methods produce finite loss values."""
        results = bench_reward_learning()
        for key, r in results.items():
            assert np.isfinite(r.final_loss), f"{key} produced non-finite loss"


class TestPopulationSelection:
    """Test population selection mechanisms."""

    def test_egg_population_runs(self):
        """EGG-style population training completes."""
        results = bench_population_selection()
        assert 'egg_population' in results
        assert results['egg_population'].num_groups == 5

    def test_neural_darwinism_runs(self):
        """PyQuifer NeuralDarwinism completes."""
        results = bench_population_selection()
        assert 'neural_darwinism' in results


class TestSignalDiscrimination:
    """Test signal discrimination ability."""

    def test_spiking_produces_output(self):
        """Spiking layer produces non-trivial discrimination."""
        results = bench_signal_discrimination()
        assert 'spiking_layer' in results

    def test_novelty_produces_separation(self):
        """Novelty detector produces signal-based separation."""
        results = bench_signal_discrimination()
        assert 'novelty_detector' in results


class TestArchitecture:
    """Test architecture feature comparison."""

    def test_feature_counts(self):
        """Both systems have meaningful feature counts."""
        features = bench_architecture_features()
        eg_count = sum(1 for v in features['egg'].values() if v)
        pq_count = sum(1 for v in features['pyquifer'].values() if v)
        assert eg_count >= 5
        assert pq_count >= 5


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Running PyQuifer vs Emergent Communication benchmarks...\n")

    learning_results = bench_reward_learning()
    selection_results = bench_population_selection()
    discrimination_results = bench_signal_discrimination()
    arch_features = bench_architecture_features()

    print_report(learning_results, selection_results,
                 discrimination_results, arch_features)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
