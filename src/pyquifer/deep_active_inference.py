"""
Deep Active Inference - Multi-step latent rollout with EFE gradient flow to policy.

Extends the single-step active_inference.py to production-grade deep active
inference with:
- GRU-based latent transition model for temporal consistency
- Learned policy network trained via EFE gradients (continuous + discrete)
- Cross-Entropy Method (CEM) planning for K-step lookahead
- Experience replay for alternating world-model / policy optimization

Reference:
    Fountas, Z., Sajid, N., Mediano, P.A.M., & Friston, K. (2020).
    "Deep active inference agents using Monte-Carlo methods."
    Advances in Neural Information Processing Systems (NeurIPS).

Compatible with existing active_inference.py classes (PredictiveEncoder,
PredictiveDecoder, ExpectedFreeEnergy).
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from .active_inference import (
    PredictiveEncoder,
    PredictiveDecoder,
    ExpectedFreeEnergy,
    reparameterize,
    kl_divergence_gaussian,
    constrained_activation,
)


class LatentTransitionModel(nn.Module):
    """
    Learned dynamics model in latent space with GRU-based recurrence.

    Models the transition p(z_{t+1} | z_t, a_t) using a GRU cell for
    temporal consistency across multi-step rollouts. Outputs a Gaussian
    distribution over the next latent state, enabling uncertainty-aware
    planning.

    The GRU hidden state provides an implicit memory of the rollout
    trajectory, preventing error accumulation that plagues purely
    feed-forward transition models over long horizons.

    Reference:
        Fountas et al. (2020), Section 3.2 - Transition model.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        """
        Args:
            latent_dim: Dimension of the latent state z.
            action_dim: Dimension of the action space.
            hidden_dim: Dimension of the GRU hidden state and MLP layers.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Pre-projection: map (z, a) to GRU input dimension
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ELU(),
        )

        # GRU cell for recurrent temporal dynamics
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        # Post-projection: map GRU hidden state to next latent distribution
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean + logvar
        )

    def forward(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next latent state distribution given current state and action.

        Args:
            z_t: Current latent state [B, latent_dim].
            a_t: Current action [B, action_dim].
            h: Optional GRU hidden state [B, hidden_dim]. If None, initialized
               to zeros.

        Returns:
            z_next: Sampled next latent state [B, latent_dim].
            z_dist: Tuple-like via dict — use z_mean, z_logvar from return.
            h_next: Updated GRU hidden state [B, hidden_dim].

        Note:
            Returns (z_next, (z_mean, z_logvar), h_next) as a 3-tuple.
        """
        batch_size = z_t.shape[0]

        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=z_t.device)

        # Project (z, a) to GRU input
        x = torch.cat([z_t, a_t], dim=-1)
        x = self.input_proj(x)

        # GRU update
        h_next = self.gru_cell(x, h)

        # Predict next latent distribution
        out = self.output_proj(h_next)
        z_mean, z_logvar = constrained_activation(out, mean_bound=2.0, var_max=1.0)

        # Sample via reparameterization
        z_next = reparameterize(z_mean, z_logvar)

        return z_next, (z_mean, z_logvar), h_next

    def multi_step_predict(
        self,
        z_0: torch.Tensor,
        actions: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Roll out the transition model for multiple steps.

        Args:
            z_0: Initial latent state [B, latent_dim].
            actions: Sequence of actions [B, K, action_dim] where K is horizon.
            h: Optional initial GRU hidden state [B, hidden_dim].

        Returns:
            z_sequence: List of K sampled latent states, each [B, latent_dim].
            dist_sequence: List of K (mean, logvar) tuples for each step.
        """
        z_sequence: List[torch.Tensor] = []
        dist_sequence: List[Tuple[torch.Tensor, torch.Tensor]] = []

        z_t = z_0
        h_t = h
        K = actions.shape[1]

        for k in range(K):
            a_t = actions[:, k, :]
            z_next, (z_mean, z_logvar), h_t = self.forward(z_t, a_t, h_t)
            z_sequence.append(z_next)
            dist_sequence.append((z_mean, z_logvar))
            z_t = z_next

        return z_sequence, dist_sequence


class PolicyNetwork(nn.Module):
    """
    Policy network trained via Expected Free Energy gradients.

    Supports both continuous and discrete action spaces:
    - Continuous: outputs mean and log_std of a diagonal Gaussian,
      actions sampled via the reparameterization trick with tanh squashing.
    - Discrete: outputs logits over action categories.

    The policy is trained to minimize EFE rather than maximize reward,
    making it an active inference policy that balances pragmatic (goal)
    and epistemic (exploration) value.

    Reference:
        Fountas et al. (2020), Section 3.3 - Policy network.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        discrete: bool = False,
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
    ):
        """
        Args:
            latent_dim: Dimension of the latent state input.
            action_dim: Dimension of the action output.
            hidden_dim: Hidden layer dimension.
            discrete: If True, output categorical logits; else Gaussian params.
            log_std_bounds: (min, max) clamp for log_std in continuous mode.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.log_std_min, self.log_std_max = log_std_bounds

        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        if discrete:
            self.head = nn.Linear(hidden_dim, action_dim)
        else:
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, z: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Compute action distribution parameters from latent state.

        Args:
            z: Latent state [B, latent_dim].

        Returns:
            If continuous: (mean, log_std) each [B, action_dim].
            If discrete: logits [B, action_dim].
        """
        h = self.trunk(z)

        if self.discrete:
            logits = self.head(h)
            return logits
        else:
            mean = self.mean_head(h)
            log_std = self.log_std_head(h)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            return mean, log_std

    def sample(
        self, z: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action with reparameterization trick.

        Args:
            z: Latent state [B, latent_dim].
            deterministic: If True, return the mean/mode (no sampling).

        Returns:
            (action, log_prob): Sampled action [B, action_dim] and its
            log probability [B].
        """
        if self.discrete:
            logits = self.forward(z)
            if deterministic:
                action_idx = logits.argmax(dim=-1)
                action = F.one_hot(action_idx, self.action_dim).float()
                log_prob = F.log_softmax(logits, dim=-1).gather(
                    1, action_idx.unsqueeze(-1)
                ).squeeze(-1)
            else:
                dist = Categorical(logits=logits)
                action_idx = dist.sample()
                action = F.one_hot(action_idx, self.action_dim).float()
                log_prob = dist.log_prob(action_idx)
            return action, log_prob
        else:
            mean, log_std = self.forward(z)
            if deterministic:
                action = torch.tanh(mean)
                # Log prob of tanh-squashed Gaussian at the mean
                log_prob = Normal(mean, log_std.exp()).log_prob(mean)
                log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6)
                log_prob = log_prob.sum(dim=-1)
                return action, log_prob

            std = log_std.exp()
            dist = Normal(mean, std)
            # Reparameterized sample
            x_t = dist.rsample()
            action = torch.tanh(x_t)

            # Log prob with tanh squashing correction (SAC-style)
            log_prob = dist.log_prob(x_t)
            log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1)

            return action, log_prob

    def log_prob(
        self, z: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of a given action under the current policy.

        Args:
            z: Latent state [B, latent_dim].
            action: Action [B, action_dim].

        Returns:
            Log probability [B].
        """
        if self.discrete:
            logits = self.forward(z)
            # action is one-hot; recover index
            action_idx = action.argmax(dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs.gather(1, action_idx.unsqueeze(-1)).squeeze(-1)
        else:
            mean, log_std = self.forward(z)
            std = log_std.exp()
            dist = Normal(mean, std)

            # Invert tanh to get pre-squash value
            # atanh(action) — clamp for numerical stability
            action_clamped = action.clamp(-0.999, 0.999)
            x_t = torch.atanh(action_clamped)

            log_prob = dist.log_prob(x_t)
            log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6)
            return log_prob.sum(dim=-1)

    def entropy(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of the policy distribution.

        Args:
            z: Latent state [B, latent_dim].

        Returns:
            Entropy [B].
        """
        if self.discrete:
            logits = self.forward(z)
            dist = Categorical(logits=logits)
            return dist.entropy()
        else:
            _, log_std = self.forward(z)
            # Gaussian entropy: 0.5 * ln(2*pi*e) * D + sum(log_std)
            return (0.5 * math.log(2 * math.pi * math.e) + log_std).sum(dim=-1)


class ReplayBuffer:
    """
    Experience replay buffer for alternating world-model / policy optimization.

    Stores (state, action, reward, next_state, done) transitions as contiguous
    tensors for efficient batched sampling. Uses a circular buffer with uniform
    random sampling.

    This is not an nn.Module since it holds no learnable parameters, but it
    manages tensors on the appropriate device for zero-copy sampling.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            capacity: Maximum number of transitions to store.
            state_dim: Dimension of state/observation vectors.
            action_dim: Dimension of action vectors.
            device: Device for storage tensors. Defaults to CPU.
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or torch.device("cpu")

        self._states = torch.zeros(capacity, state_dim, device=self.device)
        self._actions = torch.zeros(capacity, action_dim, device=self.device)
        self._rewards = torch.zeros(capacity, 1, device=self.device)
        self._next_states = torch.zeros(capacity, state_dim, device=self.device)
        self._dones = torch.zeros(capacity, 1, device=self.device)

        self._ptr = 0
        self._size = 0

    @property
    def size(self) -> int:
        """Current number of stored transitions."""
        return self._size

    @property
    def is_full(self) -> bool:
        """Whether the buffer has reached capacity."""
        return self._size >= self.capacity

    def store(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: Union[float, torch.Tensor],
        next_state: torch.Tensor,
        done: Union[bool, float, torch.Tensor],
    ) -> None:
        """
        Store a single transition.

        Args:
            state: Observation [state_dim] or [1, state_dim].
            action: Action [action_dim] or [1, action_dim].
            reward: Scalar reward.
            next_state: Next observation [state_dim] or [1, state_dim].
            done: Episode termination flag (True/1.0 if terminal).
        """
        idx = self._ptr

        self._states[idx] = (
            state.detach().to(self.device).view(-1)[:self.state_dim]
        )
        self._actions[idx] = (
            action.detach().to(self.device).view(-1)[:self.action_dim]
        )
        self._next_states[idx] = (
            next_state.detach().to(self.device).view(-1)[:self.state_dim]
        )

        if isinstance(reward, torch.Tensor):
            self._rewards[idx, 0] = reward.detach().to(self.device).item()
        else:
            self._rewards[idx, 0] = float(reward)

        if isinstance(done, torch.Tensor):
            self._dones[idx, 0] = done.detach().to(self.device).float().item()
        else:
            self._dones[idx, 0] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def store_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """
        Store a batch of transitions efficiently.

        Args:
            states: [N, state_dim]
            actions: [N, action_dim]
            rewards: [N] or [N, 1]
            next_states: [N, state_dim]
            dones: [N] or [N, 1]
        """
        n = states.shape[0]
        rewards = rewards.detach().to(self.device).view(n, 1)
        dones = dones.detach().to(self.device).float().view(n, 1)
        states = states.detach().to(self.device)
        actions = actions.detach().to(self.device)
        next_states = next_states.detach().to(self.device)

        if self._ptr + n <= self.capacity:
            self._states[self._ptr : self._ptr + n] = states
            self._actions[self._ptr : self._ptr + n] = actions
            self._rewards[self._ptr : self._ptr + n] = rewards
            self._next_states[self._ptr : self._ptr + n] = next_states
            self._dones[self._ptr : self._ptr + n] = dones
        else:
            # Wrap around
            first = self.capacity - self._ptr
            self._states[self._ptr :] = states[:first]
            self._actions[self._ptr :] = actions[:first]
            self._rewards[self._ptr :] = rewards[:first]
            self._next_states[self._ptr :] = next_states[:first]
            self._dones[self._ptr :] = dones[:first]

            remaining = n - first
            self._states[:remaining] = states[first:]
            self._actions[:remaining] = actions[first:]
            self._rewards[:remaining] = rewards[first:]
            self._next_states[:remaining] = next_states[first:]
            self._dones[:remaining] = dones[first:]

        self._ptr = (self._ptr + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a uniformly random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dict with keys: 'states', 'actions', 'rewards',
            'next_states', 'dones' — each [batch_size, dim].

        Raises:
            ValueError: If batch_size exceeds current buffer size.
        """
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer "
                f"with only {self._size} stored."
            )

        indices = torch.randint(0, self._size, (batch_size,), device=self.device)

        return {
            "states": self._states[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_states": self._next_states[indices],
            "dones": self._dones[indices],
        }

    def clear(self) -> None:
        """Reset the buffer."""
        self._ptr = 0
        self._size = 0


class MultiStepPlanner:
    """
    K-step lookahead planner using Cross-Entropy Method (CEM).

    Instead of exhaustive tree search over the action space, CEM
    iteratively refines a distribution over action sequences by:
    1. Sampling N candidate action sequences from a Gaussian.
    2. Evaluating cumulative EFE for each via latent rollout.
    3. Fitting a new Gaussian to the top-J elite sequences.
    4. Repeating for I iterations.

    The policy network provides the initial action distribution,
    warm-starting CEM for faster convergence.

    Reference:
        Fountas et al. (2020), Section 3.4 - Planning.
        Botev et al. (2013). "The cross-entropy method for optimization."
    """

    def __init__(
        self,
        transition_model: LatentTransitionModel,
        policy: PolicyNetwork,
        efe_computer: ExpectedFreeEnergy,
        horizon: int = 5,
        num_candidates: int = 64,
        num_elites: int = 8,
        cem_iterations: int = 3,
        discrete: bool = False,
    ):
        """
        Args:
            transition_model: Learned latent dynamics model.
            policy: Policy network for warm-starting CEM.
            efe_computer: EFE computation module.
            horizon: Planning horizon K (number of lookahead steps).
            num_candidates: Number of candidate action sequences per CEM iter.
            num_elites: Number of top sequences kept as elites.
            cem_iterations: Number of CEM refinement iterations.
            discrete: Whether the action space is discrete.
        """
        self.transition_model = transition_model
        self.policy = policy
        self.efe_computer = efe_computer
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.cem_iterations = cem_iterations
        self.discrete = discrete

    @torch.no_grad()
    def plan(
        self,
        current_latent: torch.Tensor,
        decoder: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Plan the best first action via K-step CEM lookahead.

        Args:
            current_latent: Current latent state [B, latent_dim] or [latent_dim].
            decoder: Optional decoder for observation-model entropy in EFE.

        Returns:
            best_action: Optimal first action [B, action_dim].
            info: Dict with 'best_efe', 'mean_efe', 'elite_actions'.
        """
        if current_latent.dim() == 1:
            current_latent = current_latent.unsqueeze(0)

        batch_size = current_latent.shape[0]
        latent_dim = current_latent.shape[1]
        action_dim = self.policy.action_dim
        device = current_latent.device
        N = self.num_candidates
        K = self.horizon

        if self.discrete:
            return self._plan_discrete(current_latent, decoder)

        # Initialize CEM distribution from policy
        # Get policy's mean and std for the current latent
        with torch.no_grad():
            policy_mean, policy_log_std = self.policy.forward(current_latent)
            policy_std = policy_log_std.exp()

        # CEM mean/std for full action sequence [B, K, action_dim]
        # Initialize all steps with the policy's output at current state
        cem_mean = policy_mean.unsqueeze(1).expand(batch_size, K, action_dim).clone()
        cem_std = policy_std.unsqueeze(1).expand(batch_size, K, action_dim).clone()
        # Slightly inflate std for exploration
        cem_std = cem_std * 1.5

        best_efe = torch.full((batch_size,), float("inf"), device=device)
        best_actions = torch.zeros(batch_size, K, action_dim, device=device)

        for iteration in range(self.cem_iterations):
            # Sample N candidate action sequences: [B, N, K, action_dim]
            eps = torch.randn(batch_size, N, K, action_dim, device=device)
            candidates = (
                cem_mean.unsqueeze(1) + cem_std.unsqueeze(1) * eps
            )
            # Squash to valid range
            candidates = torch.tanh(candidates)

            # Evaluate each candidate: [B, N]
            efes = self._evaluate_candidates(
                current_latent, candidates, decoder
            )

            # Select elites
            _, elite_idx = efes.topk(self.num_elites, dim=1, largest=False)
            # elite_idx: [B, J]

            # Gather elite action sequences: [B, J, K, action_dim]
            elite_idx_expanded = elite_idx.unsqueeze(-1).unsqueeze(-1).expand(
                batch_size, self.num_elites, K, action_dim
            )
            elites = candidates.gather(1, elite_idx_expanded)

            # Update CEM distribution (atanh to go back to pre-squash space)
            elites_presquash = torch.atanh(elites.clamp(-0.999, 0.999))
            cem_mean = elites_presquash.mean(dim=1)  # [B, K, action_dim]
            cem_std = elites_presquash.std(dim=1).clamp(min=0.01)

            # Track best
            batch_best_efe, batch_best_idx = efes.min(dim=1)
            improved = batch_best_efe < best_efe
            if improved.any():
                best_efe[improved] = batch_best_efe[improved]
                batch_best_idx_expanded = (
                    batch_best_idx[improved]
                    .unsqueeze(-1).unsqueeze(-1)
                    .expand(-1, K, action_dim)
                    .unsqueeze(1)
                )
                best_actions[improved] = candidates[improved].gather(
                    1, batch_best_idx_expanded
                ).squeeze(1)

        # Return the first action of the best sequence
        first_action = best_actions[:, 0, :]  # [B, action_dim]

        info = {
            "best_efe": best_efe,
            "mean_efe": efes.mean(dim=1) if 'efes' in dir() else best_efe,
            "elite_actions": elites[:, 0, 0, :] if 'elites' in dir() else first_action,
        }

        return first_action, info

    @torch.no_grad()
    def _plan_discrete(
        self,
        current_latent: torch.Tensor,
        decoder: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Planning for discrete action spaces using policy-guided sampling.

        For discrete actions, CEM uses categorical re-weighting rather
        than Gaussian fitting.

        Args:
            current_latent: [B, latent_dim]
            decoder: Optional decoder.

        Returns:
            best_action: [B, action_dim] one-hot.
            info: Dict with planning statistics.
        """
        batch_size = current_latent.shape[0]
        action_dim = self.policy.action_dim
        device = current_latent.device
        N = self.num_candidates
        K = self.horizon

        # Get policy logits for initial sampling distribution
        logits = self.policy.forward(current_latent)  # [B, action_dim]

        # Initialize categorical probabilities for each step
        # [B, K, action_dim]
        probs = F.softmax(logits, dim=-1).unsqueeze(1).expand(
            batch_size, K, action_dim
        ).clone()

        best_efe = torch.full((batch_size,), float("inf"), device=device)
        best_actions = torch.zeros(batch_size, K, action_dim, device=device)

        for iteration in range(self.cem_iterations):
            # Sample N candidate sequences: [B, N, K]
            flat_probs = probs.unsqueeze(1).expand(batch_size, N, K, action_dim)
            flat_probs = flat_probs.reshape(-1, action_dim)
            action_indices = Categorical(probs=flat_probs).sample()
            action_indices = action_indices.reshape(batch_size, N, K)

            # Convert to one-hot: [B, N, K, action_dim]
            candidates = F.one_hot(action_indices, action_dim).float()

            # Evaluate
            efes = self._evaluate_candidates(
                current_latent, candidates, decoder
            )

            # Select elites
            _, elite_idx = efes.topk(self.num_elites, dim=1, largest=False)

            # Gather elite sequences: [B, J, K]
            elite_idx_k = elite_idx.unsqueeze(-1).expand(
                batch_size, self.num_elites, K
            )
            elite_indices = action_indices.gather(1, elite_idx_k)

            # Update categorical by counting elite action frequencies
            for k in range(K):
                counts = torch.zeros(batch_size, action_dim, device=device)
                for j in range(self.num_elites):
                    idx = elite_indices[:, j, k]  # [B]
                    counts.scatter_add_(
                        1, idx.unsqueeze(-1),
                        torch.ones(batch_size, 1, device=device),
                    )
                # Mix with uniform for exploration
                probs[:, k, :] = 0.8 * (
                    counts / counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
                ) + 0.2 / action_dim

            # Track best
            batch_best_efe, batch_best_idx = efes.min(dim=1)
            improved = batch_best_efe < best_efe
            if improved.any():
                best_efe[improved] = batch_best_efe[improved]
                best_idx_expanded = (
                    batch_best_idx[improved]
                    .unsqueeze(-1).unsqueeze(-1)
                    .expand(-1, K, action_dim)
                    .unsqueeze(1)
                )
                best_actions[improved] = candidates[improved].gather(
                    1, best_idx_expanded
                ).squeeze(1)

        first_action = best_actions[:, 0, :]

        info = {
            "best_efe": best_efe,
            "mean_efe": efes.mean(dim=1),
            "elite_actions": best_actions[:, 0, :],
        }

        return first_action, info

    def _evaluate_candidates(
        self,
        z_0: torch.Tensor,
        candidates: torch.Tensor,
        decoder: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Evaluate cumulative EFE for candidate action sequences.

        Args:
            z_0: Initial latent [B, latent_dim].
            candidates: Action sequences [B, N, K, action_dim].
            decoder: Optional decoder for observation entropy.

        Returns:
            Cumulative EFE for each candidate [B, N].
        """
        batch_size, N, K, action_dim = candidates.shape
        latent_dim = z_0.shape[1]
        device = z_0.device

        # Expand z_0 for all candidates: [B*N, latent_dim]
        z_t = z_0.unsqueeze(1).expand(batch_size, N, latent_dim)
        z_t = z_t.reshape(batch_size * N, latent_dim)

        cumulative_efe = torch.zeros(batch_size * N, device=device)
        h_t = None  # GRU hidden state

        discount = 1.0

        for k in range(K):
            a_t = candidates[:, :, k, :].reshape(batch_size * N, action_dim)

            z_next, (z_mean, z_logvar), h_t = self.transition_model.forward(
                z_t, a_t, h_t
            )

            efe_result = self.efe_computer(z_mean, z_logvar, decoder)
            cumulative_efe = cumulative_efe + discount * efe_result["efe"]

            z_t = z_next
            discount *= 0.99  # Temporal discounting

        return cumulative_efe.reshape(batch_size, N)

    def evaluate_trajectory(
        self,
        z_0: torch.Tensor,
        actions: torch.Tensor,
        decoder: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Evaluate cumulative EFE for a single action trajectory.

        Args:
            z_0: Initial latent state [B, latent_dim].
            actions: Action sequence [B, K, action_dim].
            decoder: Optional decoder.

        Returns:
            Cumulative EFE [B].
        """
        # Treat as a single candidate per batch element
        candidates = actions.unsqueeze(1)  # [B, 1, K, action_dim]
        return self._evaluate_candidates(z_0, candidates, decoder).squeeze(1)


class DeepAIF(nn.Module):
    """
    Full Deep Active Inference agent with multi-step planning.

    Integrates:
    - PredictiveEncoder / PredictiveDecoder: VAE-style perception
    - LatentTransitionModel: GRU-based latent dynamics
    - PolicyNetwork: EFE-gradient-trained action selection
    - ExpectedFreeEnergy: Epistemic + pragmatic value computation
    - MultiStepPlanner: CEM-based K-step lookahead
    - ReplayBuffer: Experience storage for alternating optimization

    Training alternates between:
    1. World model update (encoder + decoder + transition): minimize
       reconstruction error + KL + transition prediction error.
    2. Policy update: minimize multi-step EFE via gradient descent
       through the differentiable planning rollout.

    Reference:
        Fountas, Z., Sajid, N., Mediano, P.A.M., & Friston, K. (2020).
        "Deep active inference agents using Monte-Carlo methods."
        NeurIPS 2020.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        action_dim: int,
        horizon: int = 5,
        hidden_dim: int = 256,
        discrete_actions: bool = False,
        replay_capacity: int = 100_000,
        efe_pragmatic_weight: float = 1.0,
        efe_epistemic_weight: float = 1.0,
        efe_num_samples: int = 10,
        cem_candidates: int = 64,
        cem_elites: int = 8,
        cem_iterations: int = 3,
        entropy_coeff: float = 0.01,
        kl_coeff: float = 0.1,
        transition_coeff: float = 1.0,
    ):
        """
        Args:
            obs_dim: Observation/state dimension.
            latent_dim: Latent representation dimension.
            action_dim: Action dimension (num categories if discrete).
            horizon: Planning horizon K for multi-step EFE.
            hidden_dim: Hidden layer size for all networks.
            discrete_actions: True for categorical, False for continuous.
            replay_capacity: Maximum replay buffer size.
            efe_pragmatic_weight: Weight for pragmatic (goal) value in EFE.
            efe_epistemic_weight: Weight for epistemic (exploration) value.
            efe_num_samples: Monte Carlo samples for EFE estimation.
            cem_candidates: Number of CEM candidate sequences.
            cem_elites: Number of CEM elite sequences.
            cem_iterations: Number of CEM refinement iterations.
            entropy_coeff: Entropy bonus coefficient for policy.
            kl_coeff: KL divergence weight in world model loss.
            transition_coeff: Transition prediction weight in world model loss.
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.discrete_actions = discrete_actions
        self.entropy_coeff = entropy_coeff
        self.kl_coeff = kl_coeff
        self.transition_coeff = transition_coeff

        # --- World model components ---
        self.encoder = PredictiveEncoder(obs_dim, latent_dim, hidden_dim)
        self.decoder = PredictiveDecoder(latent_dim, obs_dim, hidden_dim)
        self.transition_model = LatentTransitionModel(
            latent_dim, action_dim, hidden_dim
        )

        # --- EFE computation ---
        self.efe_computer = ExpectedFreeEnergy(
            latent_dim,
            num_samples=efe_num_samples,
            pragmatic_weight=efe_pragmatic_weight,
            epistemic_weight=efe_epistemic_weight,
        )

        # --- Policy ---
        self.policy = PolicyNetwork(
            latent_dim, action_dim, hidden_dim, discrete=discrete_actions
        )

        # --- Planner ---
        self.planner = MultiStepPlanner(
            transition_model=self.transition_model,
            policy=self.policy,
            efe_computer=self.efe_computer,
            horizon=horizon,
            num_candidates=cem_candidates,
            num_elites=cem_elites,
            cem_iterations=cem_iterations,
            discrete=discrete_actions,
        )

        # --- Replay buffer (not an nn.Module, managed externally) ---
        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
            state_dim=obs_dim,
            action_dim=action_dim,
        )

        # Step counter for alternating optimization
        self._train_step_count = 0

    def set_preference(
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> None:
        """
        Set the preferred (goal) latent state distribution for EFE.

        Args:
            mean: Preferred state mean [latent_dim].
            logvar: Preferred state log-variance [latent_dim].
        """
        self.efe_computer.set_preference(mean, logvar)

    def encode(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode observation to latent distribution and sample.

        Args:
            observation: [B, obs_dim]

        Returns:
            (z, z_mean, z_logvar)
        """
        z_mean, z_logvar = self.encoder(observation)
        z = reparameterize(z_mean, z_logvar)
        return z, z_mean, z_logvar

    @torch.no_grad()
    def act(
        self,
        observation: torch.Tensor,
        use_planner: bool = True,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Select an action given an observation (inference mode).

        Args:
            observation: Current observation [obs_dim] or [B, obs_dim].
            use_planner: If True, use CEM multi-step planning; else use
                policy network directly.
            deterministic: If True, take the mode of the policy.

        Returns:
            Dict with:
            - 'action': Selected action [B, action_dim].
            - 'latent': Current latent state [B, latent_dim].
            - 'efe': Expected free energy estimate [B].
            - 'epistemic_value': Epistemic component [B].
            - 'pragmatic_value': Pragmatic component [B].
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        # Encode
        z, z_mean, z_logvar = self.encode(observation)

        if use_planner:
            action, plan_info = self.planner.plan(z, self.decoder)
            efe_val = plan_info["best_efe"]
        else:
            action, _ = self.policy.sample(z, deterministic=deterministic)
            # Single-step EFE for diagnostics
            if self.discrete_actions:
                a_for_trans = action
            else:
                a_for_trans = action
            z_next, (z_next_mean, z_next_logvar), _ = self.transition_model(
                z, a_for_trans
            )
            efe_result = self.efe_computer(z_next_mean, z_next_logvar, self.decoder)
            efe_val = efe_result["efe"]

        # Compute EFE components at current state for diagnostics
        # (single-step from current z with selected action)
        z_pred, (z_pred_mean, z_pred_logvar), _ = self.transition_model(z, action)
        efe_components = self.efe_computer(z_pred_mean, z_pred_logvar, self.decoder)

        return {
            "action": action,
            "latent": z,
            "efe": efe_val,
            "epistemic_value": efe_components["epistemic"],
            "pragmatic_value": efe_components["pragmatic"],
        }

    def train_step(
        self,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: int = 64,
        world_model_lr: float = 1e-3,
        policy_lr: float = 3e-4,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one training step with alternating optimization.

        Even steps: update world model (encoder + decoder + transition).
        Odd steps: update policy via multi-step EFE gradient.

        Args:
            batch: Optional pre-sampled batch dict. If None, samples from
                replay buffer.
            batch_size: Batch size if sampling from buffer.
            world_model_lr: Learning rate for world model (used to create
                optimizers on first call if needed).
            policy_lr: Learning rate for policy.

        Returns:
            Dict with loss components:
            - 'total_loss': Combined training loss.
            - 'recon_loss': Reconstruction loss (world model steps).
            - 'kl_loss': KL divergence loss (world model steps).
            - 'transition_loss': Transition prediction loss (world model steps).
            - 'policy_loss': EFE-based policy loss (policy steps).
            - 'entropy': Policy entropy (policy steps).
            - 'efe': Mean EFE over batch.
            - 'step_type': 'world_model' or 'policy'.
        """
        if batch is None:
            if self.replay_buffer.size < batch_size:
                return {
                    "total_loss": torch.tensor(0.0),
                    "recon_loss": torch.tensor(0.0),
                    "kl_loss": torch.tensor(0.0),
                    "transition_loss": torch.tensor(0.0),
                    "policy_loss": torch.tensor(0.0),
                    "entropy": torch.tensor(0.0),
                    "efe": torch.tensor(0.0),
                    "step_type": "insufficient_data",
                }
            batch = self.replay_buffer.sample(batch_size)

        # Ensure optimizers exist (lazy creation so device is correct)
        if not hasattr(self, "_world_model_opt"):
            world_params = (
                list(self.encoder.parameters())
                + list(self.decoder.parameters())
                + list(self.transition_model.parameters())
            )
            self._world_model_opt = torch.optim.Adam(
                world_params, lr=world_model_lr
            )
            self._policy_opt = torch.optim.Adam(
                self.policy.parameters(), lr=policy_lr
            )

        self._train_step_count += 1

        if self._train_step_count % 2 == 1:
            return self._world_model_step(batch)
        else:
            return self._policy_step(batch)

    def _world_model_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Update world model: encoder, decoder, transition.

        Loss = recon_loss + kl_coeff * kl_loss + transition_coeff * trans_loss
        """
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]

        # Encode current state
        z_mean, z_logvar = self.encoder(states)
        z = reparameterize(z_mean, z_logvar)

        # Decode (reconstruction)
        recon = self.decoder(z)
        recon_loss = F.mse_loss(recon, states)

        # KL to standard normal prior
        kl_loss = kl_divergence_gaussian(
            z_mean,
            z_logvar,
            torch.zeros_like(z_mean),
            torch.zeros_like(z_logvar),
        ).mean()

        # Transition prediction
        z_next_pred, (z_next_mean, z_next_logvar), _ = self.transition_model(
            z.detach(), actions
        )

        # Target: encode next state (stop gradient to avoid encoder collapse)
        with torch.no_grad():
            z_next_target_mean, z_next_target_logvar = self.encoder(next_states)

        trans_loss = kl_divergence_gaussian(
            z_next_mean,
            z_next_logvar,
            z_next_target_mean,
            z_next_target_logvar,
        ).mean()

        total = recon_loss + self.kl_coeff * kl_loss + self.transition_coeff * trans_loss

        self._world_model_opt.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.transition_model.parameters()),
            max_norm=10.0,
        )
        self._world_model_opt.step()

        return {
            "total_loss": total.detach(),
            "recon_loss": recon_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "transition_loss": trans_loss.detach(),
            "policy_loss": torch.tensor(0.0),
            "entropy": torch.tensor(0.0),
            "efe": torch.tensor(0.0),
            "step_type": "world_model",
        }

    def _policy_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Update policy by minimizing multi-step EFE via gradient descent.

        The key insight: we roll out the differentiable transition model
        for K steps using actions sampled from the policy (with
        reparameterization), accumulate EFE, and backpropagate the
        gradient through the entire rollout to the policy parameters.
        """
        states = batch["states"]

        # Encode (detach: policy step does not update encoder)
        with torch.no_grad():
            z_mean, z_logvar = self.encoder(states)
            z = reparameterize(z_mean, z_logvar)

        # Multi-step rollout with policy actions
        cumulative_efe = torch.zeros(states.shape[0], device=states.device)
        z_t = z
        h_t = None
        discount = 1.0
        total_log_prob = torch.zeros(states.shape[0], device=states.device)
        total_entropy = torch.zeros(states.shape[0], device=states.device)

        for k in range(self.horizon):
            # Sample action from policy (differentiable)
            action, log_prob = self.policy.sample(z_t)
            total_log_prob = total_log_prob + log_prob
            total_entropy = total_entropy + self.policy.entropy(z_t)

            # Predict next state
            z_next, (z_next_mean, z_next_logvar), h_t = self.transition_model(
                z_t, action, h_t
            )

            # Compute EFE at predicted state
            efe_result = self.efe_computer(z_next_mean, z_next_logvar, self.decoder)
            cumulative_efe = cumulative_efe + discount * efe_result["efe"]

            z_t = z_next
            discount *= 0.99

        # Policy loss: minimize EFE - entropy bonus
        # The EFE gradient flows through the transition model rollout
        # to the policy via reparameterized actions
        mean_efe = cumulative_efe.mean()
        mean_entropy = total_entropy.mean()
        policy_loss = mean_efe - self.entropy_coeff * mean_entropy

        self._policy_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=5.0)
        self._policy_opt.step()

        return {
            "total_loss": policy_loss.detach(),
            "recon_loss": torch.tensor(0.0),
            "kl_loss": torch.tensor(0.0),
            "transition_loss": torch.tensor(0.0),
            "policy_loss": policy_loss.detach(),
            "entropy": mean_entropy.detach(),
            "efe": mean_efe.detach(),
            "step_type": "policy",
        }

    def forward(
        self, observation: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode, select action, predict, compute EFE.

        For training/analysis — use act() for inference.

        Args:
            observation: Current observation [B, obs_dim].

        Returns:
            Dict with z_mean, z_logvar, z, reconstruction, action,
            next_z_mean, next_z_logvar, efe, pragmatic, epistemic.
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        # Encode
        z_mean, z_logvar = self.encoder(observation)
        z = reparameterize(z_mean, z_logvar)

        # Reconstruct
        reconstruction = self.decoder(z)

        # Select action from policy
        action, log_prob = self.policy.sample(z)

        # Predict next latent state
        z_next, (z_next_mean, z_next_logvar), _ = self.transition_model(z, action)

        # Compute EFE
        efe_result = self.efe_computer(z_next_mean, z_next_logvar, self.decoder)

        return {
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "z": z,
            "reconstruction": reconstruction,
            "action": action,
            "log_prob": log_prob,
            "next_z_mean": z_next_mean,
            "next_z_logvar": z_next_logvar,
            "efe": efe_result["efe"],
            "pragmatic": efe_result["pragmatic"],
            "epistemic": efe_result["epistemic"],
        }
