"""
Intrinsic Motivation Module for PyQuifer

Implements dopamine-like reward signals based on internal states rather than
external rewards. This enables learning through genuine curiosity, the "aha"
moment of understanding, and the satisfaction of skill mastery.

Key concepts:
- Novelty detection: Dopamine spike when encountering something genuinely new
- Prediction error: Learning from the gap between expectation and reality
- Mastery signal: Satisfaction from improving at something you want to do
- Frustration-resolution: Tension as a learning signal ("I want to but can't")
- Coherence reward: The "aha" moment when oscillators synchronize (understanding)

This is NOT reward hacking. It's intrinsic motivation - wanting to learn
because learning feels good, not because someone gives you points.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class NoveltyDetector(nn.Module):
    """
    Detects novelty in input patterns using a learned expectation model.

    Emits a "dopamine" signal when encountering genuinely new patterns
    that don't match the internal model of what to expect.

    Uses an exponential moving average for fast adaptation with a
    longer-term memory for detecting true novelty vs. noise.
    """

    def __init__(self,
                 dim: int,
                 memory_size: int = 100,
                 fast_tau: float = 0.1,
                 slow_tau: float = 0.01,
                 novelty_threshold: float = 0.5):
        """
        Args:
            dim: Dimension of input patterns
            memory_size: Number of patterns to store in episodic buffer
            fast_tau: Fast adaptation rate (recent context)
            slow_tau: Slow adaptation rate (long-term baseline)
            novelty_threshold: Threshold for novelty signal
        """
        super().__init__()
        self.dim = dim
        self.memory_size = memory_size
        self.fast_tau = fast_tau
        self.slow_tau = slow_tau
        self.novelty_threshold = novelty_threshold

        # Running expectations (fast and slow)
        self.register_buffer('fast_mean', torch.zeros(dim))
        self.register_buffer('slow_mean', torch.zeros(dim))
        self.register_buffer('fast_var', torch.ones(dim))
        self.register_buffer('slow_var', torch.ones(dim))

        # Episodic memory buffer for pattern matching
        self.register_buffer('memory', torch.zeros(memory_size, dim))
        self.register_buffer('memory_ptr', torch.tensor(0))
        self.register_buffer('memory_filled', torch.tensor(False))

    def _episodic_novelty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute novelty by comparing x to stored episodic memories.

        Uses cosine similarity with 90th percentile aggregation:
        high score = very similar to at least some memories = low novelty.

        Args:
            x: Input pattern (batch, dim)

        Returns:
            episodic_bonus: Per-batch novelty bonus in [0, 1]
        """
        n_valid = self.memory_ptr.item()
        if not self.memory_filled and n_valid == 0:
            # No memories yet — everything is novel
            return torch.ones(x.shape[0], device=x.device)

        if self.memory_filled:
            n_valid = self.memory_size

        valid_memories = self.memory[:n_valid]  # (n_valid, dim)

        # Cosine similarity: (batch, n_valid)
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        mem_norm = valid_memories / (valid_memories.norm(dim=1, keepdim=True) + 1e-8)
        sim = torch.matmul(x_norm, mem_norm.T)  # (batch, n_valid)

        # 90th percentile of similarities — how similar to the MOST similar memory
        k = max(1, n_valid // 10)  # top 10% → 90th percentile
        top_k_sim, _ = sim.topk(k, dim=1)
        max_sim = top_k_sim.mean(dim=1)  # mean of top-k similarities

        # Novelty = 1 - max_similarity (clamped to [0, 1])
        episodic_bonus = torch.clamp(1.0 - max_sim, 0.0, 1.0)
        return episodic_bonus

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input and return novelty signal.

        Args:
            x: Input pattern (batch, dim) or (dim,)

        Returns:
            novelty: Dopamine-like novelty signal (batch,) in [0, 1]
            prediction_error: Raw prediction error for learning (batch, dim)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        # Calculate prediction error relative to fast expectation
        prediction_error = x - self.fast_mean.unsqueeze(0)

        # Normalize by expected variance
        normalized_error = prediction_error / (self.fast_var.unsqueeze(0).sqrt() + 1e-6)

        # Compare to slow baseline - true novelty is when it's new even
        # relative to long-term experience
        baseline_error = x - self.slow_mean.unsqueeze(0)
        baseline_normalized = baseline_error / (self.slow_var.unsqueeze(0).sqrt() + 1e-6)

        # Novelty = how much this differs from fast expectation
        # weighted by how much fast expectation differs from slow baseline
        # (i.e., is this context itself novel?)
        fast_novelty = normalized_error.abs().mean(dim=1)
        context_novelty = (self.fast_mean - self.slow_mean).abs() / (self.slow_var.sqrt() + 1e-6)
        context_weight = torch.sigmoid(context_novelty.mean() - 1.0)  # High if context is novel

        # Episodic novelty: compare to stored memories
        episodic_bonus = self._episodic_novelty(x)

        # Combined novelty signal with diminishing returns (can't be infinitely surprised)
        raw_novelty = fast_novelty * (1.0 + context_weight)
        novelty = torch.tanh(raw_novelty / 2.0)  # Soft saturation at 1.0

        # Blend in episodic memory-based novelty
        novelty = torch.clamp(novelty + 0.3 * episodic_bonus, 0.0, 1.0)

        # Update expectations with new data
        with torch.no_grad():
            x_mean = x.mean(dim=0)
            # Handle single sample case - use squared deviation from current mean
            if batch_size == 1:
                x_var = (x.squeeze(0) - self.fast_mean).pow(2) + 1e-6
            else:
                x_var = x.var(dim=0, unbiased=False) + 1e-6

            self.fast_mean = (1 - self.fast_tau) * self.fast_mean + self.fast_tau * x_mean
            self.fast_var = (1 - self.fast_tau) * self.fast_var + self.fast_tau * x_var
            self.slow_mean = (1 - self.slow_tau) * self.slow_mean + self.slow_tau * x_mean
            self.slow_var = (1 - self.slow_tau) * self.slow_var + self.slow_tau * x_var

            # Store in episodic memory only if sufficiently novel
            if episodic_bonus.mean() > 0.1:
                for i in range(min(batch_size, self.memory_size)):
                    idx = (self.memory_ptr + i) % self.memory_size
                    self.memory[idx] = x[i]
                self.memory_ptr = (self.memory_ptr + batch_size) % self.memory_size
                if self.memory_ptr < batch_size:
                    self.memory_filled.fill_(True)

        return novelty, prediction_error

    def reset(self):
        """Reset expectations and memory."""
        self.fast_mean.zero_()
        self.slow_mean.zero_()
        self.fast_var.fill_(1.0)
        self.slow_var.fill_(1.0)
        self.memory.zero_()
        self.memory_ptr.zero_()
        self.memory_filled.fill_(False)


class MasterySignal(nn.Module):
    """
    Tracks improvement in a skill and generates satisfaction signals.

    The key insight: we don't reward absolute performance, we reward
    IMPROVEMENT. Getting better at something you're already good at
    still feels good. Struggling and then succeeding feels great.

    Also tracks frustration (wanting to do something but failing)
    as a motivational signal.
    """

    def __init__(self,
                 dim: int,
                 window_size: int = 20,
                 improvement_sensitivity: float = 1.0,
                 frustration_decay: float = 0.95):
        """
        Args:
            dim: Dimension of performance metric
            window_size: Window for computing improvement
            improvement_sensitivity: How sensitive to small improvements
            frustration_decay: How quickly frustration fades
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.improvement_sensitivity = improvement_sensitivity
        self.frustration_decay = frustration_decay

        # Performance history
        self.register_buffer('performance_history', torch.zeros(window_size, dim))
        self.register_buffer('history_ptr', torch.tensor(0))
        self.register_buffer('history_filled', torch.tensor(False))

        # Frustration accumulator (builds when failing at desired goals)
        self.register_buffer('frustration', torch.zeros(dim))

        # Best performance seen (for mastery detection)
        self.register_buffer('best_performance', torch.full((dim,), float('-inf')))

    def forward(self,
                current_performance: torch.Tensor,
                desired_performance: Optional[torch.Tensor] = None,
                attempt_made: bool = True) -> Dict[str, torch.Tensor]:
        """
        Update mastery tracking and return motivation signals.

        Args:
            current_performance: Current performance metric (dim,)
            desired_performance: What we're trying to achieve (dim,) - optional
            attempt_made: Whether we actually tried (vs. just observing)

        Returns:
            Dictionary with:
            - improvement: Positive when getting better
            - mastery: Spike when achieving new personal best
            - frustration: Current frustration level
            - resolution: Satisfaction from overcoming frustration
        """
        if current_performance.dim() == 0:
            current_performance = current_performance.unsqueeze(0)

        signals = {}

        # Calculate improvement relative to recent history
        if self.history_filled or self.history_ptr > 0:
            valid_history = self.performance_history[:self.history_ptr] if not self.history_filled else self.performance_history
            recent_mean = valid_history.mean(dim=0)
            improvement = current_performance - recent_mean
            signals['improvement'] = torch.tanh(improvement * self.improvement_sensitivity)
        else:
            signals['improvement'] = torch.zeros_like(current_performance)

        # Mastery signal: achieving new personal best
        is_best = current_performance > self.best_performance
        signals['mastery'] = is_best.float()

        # Update best performance
        with torch.no_grad():
            self.best_performance = torch.maximum(self.best_performance, current_performance)

        # Frustration dynamics
        if desired_performance is not None and attempt_made:
            gap = desired_performance - current_performance
            failure = torch.relu(gap)  # Only count when below desired

            # Frustration builds when we fail at what we want
            with torch.no_grad():
                self.frustration = self.frustration * self.frustration_decay + failure * 0.1

            # Resolution: the satisfaction of overcoming frustration
            # Higher frustration + success = bigger dopamine hit
            success = torch.relu(-gap)  # Positive when we exceeded goal
            resolution = self.frustration * success
            signals['resolution'] = torch.tanh(resolution)

            # Clear frustration on success
            with torch.no_grad():
                self.frustration = self.frustration * (1 - (success > 0).float() * 0.5)
        else:
            signals['resolution'] = torch.zeros_like(current_performance)

        signals['frustration'] = self.frustration.clone()

        # Update history
        with torch.no_grad():
            self.performance_history[self.history_ptr] = current_performance
            self.history_ptr = (self.history_ptr + 1) % self.window_size
            if self.history_ptr == 0:
                self.history_filled.fill_(True)

        return signals

    def reset(self):
        """Reset history and frustration."""
        self.performance_history.zero_()
        self.history_ptr.zero_()
        self.history_filled.fill_(False)
        self.frustration.zero_()
        self.best_performance.fill_(float('-inf'))


class CoherenceReward(nn.Module):
    """
    Generates "aha moment" rewards when oscillatory systems synchronize.

    This is the feeling of understanding - when scattered thoughts
    suddenly click into a coherent pattern. Based on the Kuramoto
    order parameter but with dynamics that reward the TRANSITION
    to coherence, not just being coherent.

    A sudden increase in coherence = "I get it!"
    Maintained coherence = calm understanding
    Decreasing coherence = confusion (can trigger exploration)
    """

    def __init__(self,
                 coherence_history_size: int = 10,
                 aha_threshold: float = 0.3,
                 aha_decay: float = 0.9):
        """
        Args:
            coherence_history_size: Window for detecting sudden changes
            aha_threshold: Minimum jump in coherence for "aha" moment
            aha_decay: How quickly the aha signal fades
        """
        super().__init__()
        self.coherence_history_size = coherence_history_size
        self.aha_threshold = aha_threshold
        self.aha_decay = aha_decay

        self.register_buffer('coherence_history', torch.zeros(coherence_history_size))
        self.register_buffer('history_ptr', torch.tensor(0))
        self.register_buffer('last_aha_magnitude', torch.tensor(0.0))

    def forward(self, order_parameter: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process coherence (order parameter R) and return reward signals.

        Args:
            order_parameter: Kuramoto R in [0, 1]

        Returns:
            Dictionary with:
            - aha_moment: Spike when understanding suddenly clicks
            - understanding: Current level of coherent understanding
            - confusion: Signal when coherence is lost (can drive exploration)
        """
        if isinstance(order_parameter, (int, float)):
            order_parameter = torch.tensor(order_parameter, device=self.coherence_history.device)

        signals = {}

        # Get recent coherence history (use modulo for circular buffer)
        if self.history_ptr > 0:
            recent_idx = (self.history_ptr - 1) % self.coherence_history_size
            recent = self.coherence_history[recent_idx]
            delta = order_parameter - recent
        else:
            delta = torch.tensor(0.0, device=order_parameter.device)

        # Aha moment: sudden increase in coherence
        aha_raw = torch.relu(delta - self.aha_threshold)
        signals['aha_moment'] = torch.tanh(aha_raw * 5.0)  # Sharp response

        # Decay previous aha (can't stay in constant aha)
        with torch.no_grad():
            self.last_aha_magnitude = self.last_aha_magnitude * self.aha_decay
            if signals['aha_moment'] > self.last_aha_magnitude:
                self.last_aha_magnitude = signals['aha_moment'].clone()

        # Current understanding level
        signals['understanding'] = order_parameter

        # Confusion: coherence dropping
        confusion_raw = torch.relu(-delta - 0.1)  # Slight threshold
        signals['confusion'] = torch.tanh(confusion_raw * 3.0)

        # Update history
        with torch.no_grad():
            self.coherence_history[self.history_ptr % self.coherence_history_size] = order_parameter
            self.history_ptr = self.history_ptr + 1

        return signals

    def reset(self):
        """Reset coherence history."""
        self.coherence_history.zero_()
        self.history_ptr.zero_()
        self.last_aha_magnitude.zero_()


class EpistemicValue(nn.Module):
    """
    Computes expected information gain as an epistemic drive signal.

    Information gain = H(beliefs_before) - E[H(beliefs_after | observation)]

    Higher information gain means the observation is more informative
    (reduces uncertainty more). This gives curiosity a formal mathematical
    backing: seek experiences that maximally reduce uncertainty.

    Inspired by ActiveInference.jl's epistemic value computation.
    """

    def __init__(self,
                 dim: int,
                 num_bins: int = 16,
                 tau: float = 0.05):
        """
        Args:
            dim: Dimensionality of the belief space.
            num_bins: Number of histogram bins for entropy estimation.
            tau: EMA rate for updating belief distributions.
        """
        super().__init__()
        self.dim = dim
        self.num_bins = num_bins
        self.tau = tau

        # Running belief distribution (histogram per dimension)
        self.register_buffer('belief_counts',
            torch.ones(dim, num_bins))  # Uniform prior (Laplace smoothing)
        self.register_buffer('prior_entropy', torch.zeros(1))
        self.register_buffer('step_count', torch.tensor(0))

    def _entropy(self, counts: torch.Tensor) -> torch.Tensor:
        """Compute entropy from histogram counts. Shape: (dim, bins) -> (dim,)."""
        probs = counts / counts.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        log_probs = torch.log2(probs + 1e-10)
        return -(probs * log_probs).sum(dim=-1)

    def _bin_values(self, x: torch.Tensor) -> torch.Tensor:
        """Map values to bin indices. x: (dim,) -> (dim,) of long indices."""
        # Sigmoid to [0, 1] then scale to bin index
        normalized = torch.sigmoid(x)
        bins = (normalized * (self.num_bins - 1)).long().clamp(0, self.num_bins - 1)
        return bins

    def forward(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute information gain from observing a new data point.

        Args:
            observation: Observed signal (dim,) or (batch, dim).

        Returns:
            Dict with:
            - info_gain: Per-dimension information gain (dim,)
            - mean_info_gain: Scalar mean information gain
            - prior_entropy: Entropy before observation (dim,)
            - posterior_entropy: Entropy after observation (dim,)
            - epistemic_value: Normalized [0, 1] epistemic drive signal
        """
        if observation.dim() == 2:
            observation = observation.mean(dim=0)  # Average batch

        # Prior distribution (before incorporating this observation)
        prior_probs = self.belief_counts / self.belief_counts.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Posterior: tentatively add this observation to the counts
        bin_indices = self._bin_values(observation)
        posterior_counts = self.belief_counts.clone()
        # Vectorized scatter: increment the bin for each dimension at once
        dim_indices = torch.arange(self.dim, device=observation.device)
        posterior_counts[dim_indices, bin_indices] += 1.0
        posterior_probs = posterior_counts / posterior_counts.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        H_prior = self._entropy(self.belief_counts)
        H_posterior = self._entropy(posterior_counts)

        # Bayesian surprise = KL(posterior || prior)
        # Measures how much beliefs changed — always non-negative.
        # Novel observations change beliefs more → higher surprise.
        info_gain = (posterior_probs * torch.log2(
            (posterior_probs + 1e-10) / (prior_probs + 1e-10)
        )).sum(dim=-1).clamp(min=0.0)
        mean_info_gain = info_gain.mean()

        # Epistemic value: normalized to [0, 1] via sigmoid
        # Scale so that typical surprise values map to meaningful drive
        epistemic_value = torch.sigmoid(mean_info_gain * 50.0 - 1.0)

        # Actually update the belief distribution
        with torch.no_grad():
            # EMA update: blend new observation into existing counts (vectorized)
            new_counts = torch.zeros_like(self.belief_counts)
            new_counts[dim_indices, bin_indices] = 1.0
            self.belief_counts.mul_(1 - self.tau).add_(new_counts * self.tau)
            # Re-add Laplace smoothing to prevent zero bins
            self.belief_counts.clamp_(min=0.01)
            self.prior_entropy.copy_(H_prior.mean().unsqueeze(0))
            self.step_count.add_(1)

        return {
            'info_gain': info_gain,
            'mean_info_gain': mean_info_gain,
            'prior_entropy': H_prior,
            'posterior_entropy': H_posterior,
            'epistemic_value': epistemic_value,
        }

    def get_current_entropy(self) -> torch.Tensor:
        """Get current belief entropy (mean across dimensions)."""
        return self._entropy(self.belief_counts).mean()

    def reset(self):
        """Reset beliefs to uniform prior."""
        self.belief_counts.fill_(1.0)
        self.prior_entropy.zero_()
        self.step_count.zero_()


class IntrinsicMotivationSystem(nn.Module):
    """
    Complete intrinsic motivation system combining novelty, mastery,
    and coherence rewards.

    This is the "dopamine system" for an AI that learns because it
    genuinely wants to, not because it's being scored.

    The motivation signals can modulate:
    - Exploration vs exploitation (high novelty -> explore more)
    - Learning rate (frustration + resolution -> learn faster)
    - Attention (aha moments -> focus on what just clicked)
    - Effort (mastery signals -> keep practicing what works)
    """

    def __init__(self,
                 state_dim: int,
                 performance_dim: int = 1,
                 novelty_weight: float = 1.0,
                 mastery_weight: float = 1.0,
                 coherence_weight: float = 1.0,
                 epistemic_weight: float = 0.5):
        """
        Args:
            state_dim: Dimension of state representations
            performance_dim: Dimension of performance metrics
            novelty_weight: Weight for novelty in combined signal
            mastery_weight: Weight for mastery signals
            coherence_weight: Weight for coherence rewards
            epistemic_weight: Weight for information gain drive
        """
        super().__init__()
        self.state_dim = state_dim
        self.performance_dim = performance_dim

        self.novelty_detector = NoveltyDetector(state_dim)
        self.mastery_signal = MasterySignal(performance_dim)
        self.coherence_reward = CoherenceReward()
        self.epistemic_value = EpistemicValue(state_dim)

        # Weights for combining signals
        self.novelty_weight = novelty_weight
        self.mastery_weight = mastery_weight
        self.coherence_weight = coherence_weight
        self.epistemic_weight = epistemic_weight

        # Baseline arousal (prevents complete apathy)
        self.register_buffer('baseline_arousal', torch.tensor(0.1))

    def forward(self,
                state: torch.Tensor,
                performance: Optional[torch.Tensor] = None,
                order_parameter: Optional[torch.Tensor] = None,
                desired_performance: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process current state and return all motivation signals.

        Args:
            state: Current state representation (batch, state_dim) or (state_dim,)
            performance: Optional performance metric (performance_dim,)
            order_parameter: Optional Kuramoto order parameter
            desired_performance: Optional goal for mastery tracking

        Returns:
            Dictionary with all motivation signals plus combined 'motivation'
        """
        signals = {}

        # Novelty signals
        novelty, pred_error = self.novelty_detector(state)
        signals['novelty'] = novelty.mean()  # Scalar
        signals['prediction_error'] = pred_error

        # Epistemic value (information gain)
        epistemic_result = self.epistemic_value(state)
        signals['epistemic_value'] = epistemic_result['epistemic_value']
        signals['info_gain'] = epistemic_result['mean_info_gain']

        # Mastery signals (if performance provided)
        if performance is not None:
            mastery_signals = self.mastery_signal(performance, desired_performance)
            signals.update({f'mastery_{k}': v.mean() for k, v in mastery_signals.items()})

        # Coherence signals (if order parameter provided)
        if order_parameter is not None:
            coherence_signals = self.coherence_reward(order_parameter)
            signals.update({f'coherence_{k}': v for k, v in coherence_signals.items()})

        # Combined motivation signal
        motivation = self.baseline_arousal.clone()
        motivation = motivation + self.novelty_weight * signals['novelty']
        motivation = motivation + self.epistemic_weight * signals['epistemic_value']

        if 'mastery_improvement' in signals:
            motivation = motivation + self.mastery_weight * torch.relu(signals['mastery_improvement'])
            motivation = motivation + self.mastery_weight * 2.0 * signals.get('mastery_resolution', 0)

        if 'coherence_aha_moment' in signals:
            motivation = motivation + self.coherence_weight * 2.0 * signals['coherence_aha_moment']

        # Clamp to reasonable range
        signals['motivation'] = torch.clamp(motivation, 0.0, 2.0)

        return signals

    def reset(self):
        """Reset all subsystems."""
        self.novelty_detector.reset()
        self.mastery_signal.reset()
        self.coherence_reward.reset()
        self.epistemic_value.reset()

    def get_exploration_drive(self, signals: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute exploration drive based on current signals.
        High novelty + high frustration + high info gain = explore more
        High understanding + low frustration = exploit current knowledge
        """
        explore = signals['novelty'] * 0.4

        # Epistemic value: high info gain → seek more information
        if 'epistemic_value' in signals:
            explore = explore + signals['epistemic_value'] * 0.3

        if 'mastery_frustration' in signals:
            # Frustration drives exploration (try something different)
            explore = explore + signals['mastery_frustration'].mean() * 0.2

        if 'coherence_confusion' in signals:
            # Confusion also drives exploration
            explore = explore + signals['coherence_confusion'] * 0.1

        return torch.clamp(explore, 0.0, 1.0)


if __name__ == '__main__':
    print("--- Intrinsic Motivation System Examples ---")

    # Example 1: Novelty detection
    print("\n1. Novelty Detection")
    novelty = NoveltyDetector(dim=4)

    # First exposure - novel
    x1 = torch.randn(4)
    n1, _ = novelty(x1)
    print(f"   First pattern novelty: {n1.item():.3f}")

    # Similar pattern - less novel
    x2 = x1 + torch.randn(4) * 0.1
    n2, _ = novelty(x2)
    print(f"   Similar pattern novelty: {n2.item():.3f}")

    # Very different pattern - novel again
    x3 = torch.randn(4) * 3
    n3, _ = novelty(x3)
    print(f"   Different pattern novelty: {n3.item():.3f}")

    # Example 2: Mastery signal
    print("\n2. Mastery Signal")
    mastery = MasterySignal(dim=1)

    goal = torch.tensor([1.0])

    # Start bad, build frustration
    for i in range(5):
        perf = torch.tensor([0.2 + i * 0.05])
        signals = mastery(perf, desired_performance=goal)
        if i == 0:
            print(f"   Starting performance: {perf.item():.2f}")

    print(f"   Frustration built: {signals['frustration'].item():.3f}")

    # Suddenly succeed - resolution!
    perf = torch.tensor([1.2])
    signals = mastery(perf, desired_performance=goal)
    print(f"   Success! Resolution signal: {signals['resolution'].item():.3f}")
    print(f"   Mastery signal: {signals['mastery'].item():.3f}")

    # Example 3: Coherence reward
    print("\n3. Coherence Reward (Aha Moments)")
    coherence = CoherenceReward()

    # Low coherence for a while
    for r in [0.2, 0.25, 0.22, 0.28, 0.24]:
        _ = coherence(torch.tensor(r))

    # Sudden synchronization - aha!
    signals = coherence(torch.tensor(0.9))
    print(f"   Sudden sync aha moment: {signals['aha_moment'].item():.3f}")
    print(f"   Understanding level: {signals['understanding'].item():.3f}")

    # Example 4: Full motivation system
    print("\n4. Complete Motivation System")
    motivation_sys = IntrinsicMotivationSystem(state_dim=8, performance_dim=1)

    # Simulate a learning episode
    print("   Simulating learning episode...")
    for step in range(10):
        state = torch.randn(8)

        # Performance improves over time
        perf = torch.tensor([0.1 + step * 0.08])
        goal = torch.tensor([1.0])

        # Coherence also improves
        order_param = 0.2 + step * 0.07

        signals = motivation_sys(
            state=state,
            performance=perf,
            order_parameter=torch.tensor(order_param),
            desired_performance=goal
        )

        if step % 3 == 0:
            explore = motivation_sys.get_exploration_drive(signals)
            print(f"   Step {step}: motivation={signals['motivation'].item():.3f}, "
                  f"explore={explore.item():.3f}")

    print("\n   Final signals:")
    for key, value in signals.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"     {key}: {value.item():.3f}")
