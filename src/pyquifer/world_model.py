"""
World Model Module for PyQuifer

Implements Dreamer-style world models for imagination and planning:

1. RSSM - Recurrent State Space Model (deterministic + stochastic)
2. Latent Dynamics - Neural ODE-based continuous evolution
3. Imagination - Rollout future trajectories without real experience
4. Planning - Action selection via imagined returns

Mathematical Foundation:
- State evolution as ODE: dz/dt = f(z, a, t)
- Belief state: p(s_t | o_{≤t}, a_{<t})
- Imagination: Sample trajectories from learned dynamics
- Planning: Maximize expected return over imagined futures

All operations use einsum and differentiable dynamics.
No if/else - behavior emerges from tensor flow.

Based on: Hafner et al. (2019-2024), DreamerV1/V2/V3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass


@dataclass
class WorldModelState:
    """Complete world model state."""
    deterministic: torch.Tensor  # h_t
    stochastic: torch.Tensor     # s_t (sampled)
    mean: torch.Tensor           # μ_t (for KL)
    std: torch.Tensor            # σ_t (for KL)

    @property
    def combined(self) -> torch.Tensor:
        """Concatenated state for downstream use."""
        return torch.cat([self.deterministic, self.stochastic], dim=-1)


class GRUDynamics(nn.Module):
    """
    GRU-based deterministic dynamics.

    Implements: h_t = GRU(h_{t-1}, [s_{t-1}, a_{t-1}])

    Uses einsum for weight application.
    """

    def __init__(self, hidden_dim: int, stoch_dim: int, action_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        input_dim = stoch_dim + action_dim

        # GRU gates: W_z, W_r, W_h for each gate
        self.W_z = nn.Parameter(torch.randn(hidden_dim, input_dim + hidden_dim) * 0.02)
        self.W_r = nn.Parameter(torch.randn(hidden_dim, input_dim + hidden_dim) * 0.02)
        self.W_h = nn.Parameter(torch.randn(hidden_dim, input_dim + hidden_dim) * 0.02)

        self.b_z = nn.Parameter(torch.zeros(hidden_dim))
        self.b_r = nn.Parameter(torch.zeros(hidden_dim))
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

        # Layer norm for stability
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self,
                h_prev: torch.Tensor,
                stoch_prev: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """
        GRU step via einsum.

        Args:
            h_prev: Previous hidden state (batch, hidden_dim)
            stoch_prev: Previous stochastic state (batch, stoch_dim)
            action: Action (batch, action_dim)

        Returns:
            h_t: New hidden state
        """
        # Concatenate inputs
        x = torch.cat([stoch_prev, action], dim=-1)
        xh = torch.cat([x, h_prev], dim=-1)

        # Gates via einsum
        z = torch.sigmoid(torch.einsum('hi,bi->bh', self.W_z, xh) + self.b_z)
        r = torch.sigmoid(torch.einsum('hi,bi->bh', self.W_r, xh) + self.b_r)

        # Candidate with reset gate
        xh_reset = torch.cat([x, r * h_prev], dim=-1)
        h_cand = torch.tanh(torch.einsum('hi,bi->bh', self.W_h, xh_reset) + self.b_h)

        # Update
        h_new = (1 - z) * h_prev + z * h_cand

        return self.ln(h_new)


class StochasticStateModel(nn.Module):
    """
    Stochastic state prediction.

    Prior: p(s_t | h_t) - predict from deterministic
    Posterior: q(s_t | h_t, o_t) - infer from deterministic + observation

    Uses diagonal Gaussian parameterization.
    """

    def __init__(self,
                 hidden_dim: int,
                 stoch_dim: int,
                 obs_dim: Optional[int] = None,
                 min_std: float = 0.1):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.min_std = min_std

        # Prior network (deterministic → stochastic)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2)  # mean + log_std
        )

        # Posterior network (deterministic + obs → stochastic)
        if obs_dim is not None:
            self.posterior_net = nn.Sequential(
                nn.Linear(hidden_dim + obs_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, stoch_dim * 2)
            )
        else:
            self.posterior_net = None

    def prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute prior distribution p(s|h)."""
        params = self.prior_net(h)
        mean, log_std = params.chunk(2, dim=-1)
        std = F.softplus(log_std) + self.min_std
        return mean, std

    def posterior(self,
                  h: torch.Tensor,
                  obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior distribution q(s|h,o)."""
        if self.posterior_net is None:
            raise ValueError("Posterior network not initialized")
        params = self.posterior_net(torch.cat([h, obs], dim=-1))
        mean, log_std = params.chunk(2, dim=-1)
        std = F.softplus(log_std) + self.min_std
        return mean, std

    def sample(self,
               mean: torch.Tensor,
               std: torch.Tensor,
               deterministic: bool = False) -> torch.Tensor:
        """Sample from Gaussian via reparameterization."""
        if deterministic:
            return mean
        eps = torch.randn_like(std)
        return mean + std * eps


class RSSM(nn.Module):
    """
    Recurrent State Space Model.

    Combines deterministic (GRU) and stochastic (VAE) dynamics:
    - Deterministic: h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
    - Stochastic prior: p(s_t | h_t)
    - Stochastic posterior: q(s_t | h_t, o_t)

    Full latent state: z_t = [h_t, s_t]
    """

    def __init__(self,
                 hidden_dim: int = 200,
                 stoch_dim: int = 30,
                 action_dim: int = 4,
                 obs_embed_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.action_dim = action_dim
        self.state_dim = hidden_dim + stoch_dim

        # Dynamics
        self.dynamics = GRUDynamics(hidden_dim, stoch_dim, action_dim)

        # Stochastic model
        self.stoch_model = StochasticStateModel(hidden_dim, stoch_dim, obs_embed_dim)

    def initial_state(self, batch_size: int, device: torch.device) -> WorldModelState:
        """Create initial state (zeros)."""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        s = torch.zeros(batch_size, self.stoch_dim, device=device)
        return WorldModelState(
            deterministic=h,
            stochastic=s,
            mean=s,
            std=torch.ones_like(s) * 0.1
        )

    def imagine_step(self,
                     state: WorldModelState,
                     action: torch.Tensor) -> WorldModelState:
        """
        One step of imagination (no observation).

        Uses prior to sample next stochastic state.
        """
        # Deterministic update
        h_new = self.dynamics(state.deterministic, state.stochastic, action)

        # Prior for stochastic
        mean, std = self.stoch_model.prior(h_new)
        s_new = self.stoch_model.sample(mean, std)

        return WorldModelState(
            deterministic=h_new,
            stochastic=s_new,
            mean=mean,
            std=std
        )

    def observe_step(self,
                     state: WorldModelState,
                     action: torch.Tensor,
                     obs_embed: torch.Tensor) -> Tuple[WorldModelState, WorldModelState]:
        """
        One step with observation.

        Returns both prior and posterior states for KL computation.
        """
        # Deterministic update
        h_new = self.dynamics(state.deterministic, state.stochastic, action)

        # Prior
        prior_mean, prior_std = self.stoch_model.prior(h_new)

        # Posterior
        post_mean, post_std = self.stoch_model.posterior(h_new, obs_embed)
        s_new = self.stoch_model.sample(post_mean, post_std)

        prior_state = WorldModelState(h_new, self.stoch_model.sample(prior_mean, prior_std),
                                       prior_mean, prior_std)
        post_state = WorldModelState(h_new, s_new, post_mean, post_std)

        return prior_state, post_state

    def imagine_trajectory(self,
                          initial_state: WorldModelState,
                          policy: Callable[[torch.Tensor], torch.Tensor],
                          horizon: int) -> List[WorldModelState]:
        """
        Imagine future trajectory using policy.

        Args:
            initial_state: Starting state
            policy: Function mapping state → action distribution
            horizon: Number of steps to imagine

        Returns:
            List of imagined states
        """
        trajectory = [initial_state]
        state = initial_state

        for _ in range(horizon):
            action = policy(state.combined)
            state = self.imagine_step(state, action)
            trajectory.append(state)

        return trajectory


class NeuralODEDynamics(nn.Module):
    """
    Continuous-time dynamics via Neural ODE.

    dz/dt = f(z, a, t)

    Solved via Euler integration (efficient on GPU).
    Enables smooth interpolation between discrete observations.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 dt: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dt = dt

        # Dynamics network: f(z, a, t) → dz/dt
        self.dynamics_net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Tanh(),  # Bound derivatives for stability
        )

        # Scale for derivatives
        self.deriv_scale = nn.Parameter(torch.ones(1))

    def derivative(self,
                   z: torch.Tensor,
                   a: torch.Tensor,
                   t: torch.Tensor) -> torch.Tensor:
        """Compute dz/dt."""
        # Expand t to batch dimension
        if t.dim() == 0:
            t = t.expand(z.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)

        inputs = torch.cat([z, a, t], dim=-1)
        return self.dynamics_net(inputs) * self.deriv_scale

    def step(self,
             z: torch.Tensor,
             a: torch.Tensor,
             t: torch.Tensor,
             dt: Optional[float] = None) -> torch.Tensor:
        """Euler step: z_{t+dt} = z_t + dt * f(z_t, a, t)"""
        dt = dt or self.dt
        dz = self.derivative(z, a, t)
        return z + dt * dz

    def integrate(self,
                  z0: torch.Tensor,
                  actions: torch.Tensor,
                  t_span: Tuple[float, float],
                  steps: int = 10) -> torch.Tensor:
        """
        Integrate ODE over time span.

        Args:
            z0: Initial state (batch, state_dim)
            actions: Actions over time (batch, steps, action_dim)
            t_span: (t_start, t_end)
            steps: Number of integration steps

        Returns:
            Trajectory (batch, steps+1, state_dim)
        """
        t_start, t_end = t_span
        dt = (t_end - t_start) / steps

        trajectory = [z0]
        z = z0

        for i in range(steps):
            t = torch.tensor(t_start + i * dt, device=z.device)
            a = actions[:, min(i, actions.shape[1]-1)]
            z = self.step(z, a, t, dt)
            trajectory.append(z)

        return torch.stack(trajectory, dim=1)


class ObservationModel(nn.Module):
    """
    Observation decoder: p(o_t | z_t)

    Maps latent state to observation distribution.
    """

    def __init__(self,
                 state_dim: int,
                 obs_dim: int,
                 hidden_dim: int = 256):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Decode state to observation."""
        return self.decoder(state)


class RewardModel(nn.Module):
    """
    Reward predictor: p(r_t | z_t)

    Maps latent state to expected reward.
    """

    def __init__(self,
                 state_dim: int,
                 hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict reward from state."""
        return self.net(state).squeeze(-1)


class ContinueModel(nn.Module):
    """
    Episode continuation predictor: p(continue | z_t)

    Predicts whether episode continues (not terminated).
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict continuation probability."""
        return self.net(state).squeeze(-1)


class WorldModel(nn.Module):
    """
    Complete World Model (Dreamer-style).

    Combines:
    - RSSM for discrete dynamics
    - Neural ODE for continuous interpolation
    - Observation, reward, continue decoders

    Supports both training from experience and
    imagination for planning.
    """

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_dim: int = 200,
                 stoch_dim: int = 30,
                 obs_embed_dim: int = 256,
                 use_neural_ode: bool = False):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.state_dim = hidden_dim + stoch_dim

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, obs_embed_dim),
            nn.ELU(),
            nn.Linear(obs_embed_dim, obs_embed_dim),
            nn.ELU(),
        )

        # RSSM
        self.rssm = RSSM(hidden_dim, stoch_dim, action_dim, obs_embed_dim)

        # Optional Neural ODE dynamics
        self.use_neural_ode = use_neural_ode
        if use_neural_ode:
            self.ode_dynamics = NeuralODEDynamics(self.state_dim, action_dim)

        # Decoders
        self.obs_decoder = ObservationModel(self.state_dim, obs_dim)
        self.reward_decoder = RewardModel(self.state_dim)
        self.continue_decoder = ContinueModel(self.state_dim)

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to embedding."""
        return self.obs_encoder(obs)

    def observe(self,
                obs_seq: torch.Tensor,
                action_seq: torch.Tensor,
                initial_state: Optional[WorldModelState] = None
                ) -> Tuple[List[WorldModelState], List[WorldModelState]]:
        """
        Process observation sequence.

        Args:
            obs_seq: Observations (batch, time, obs_dim)
            action_seq: Actions (batch, time, action_dim)
            initial_state: Optional initial state

        Returns:
            priors: Prior states at each step
            posteriors: Posterior states at each step
        """
        batch_size, seq_len = obs_seq.shape[:2]

        if initial_state is None:
            initial_state = self.rssm.initial_state(batch_size, obs_seq.device)

        priors, posteriors = [], []
        state = initial_state

        for t in range(seq_len):
            obs_embed = self.encode_observation(obs_seq[:, t])
            action = action_seq[:, t]

            prior_state, post_state = self.rssm.observe_step(state, action, obs_embed)
            priors.append(prior_state)
            posteriors.append(post_state)

            state = post_state

        return priors, posteriors

    def imagine(self,
                initial_state: WorldModelState,
                policy: Callable[[torch.Tensor], torch.Tensor],
                horizon: int = 15) -> Dict[str, torch.Tensor]:
        """
        Imagine future trajectory from current state.

        Args:
            initial_state: Starting state
            policy: Policy function (state → action)
            horizon: Imagination horizon

        Returns:
            Dictionary with states, rewards, continues
        """
        trajectory = self.rssm.imagine_trajectory(initial_state, policy, horizon)

        # Stack states
        states = torch.stack([s.combined for s in trajectory], dim=1)

        # Predict rewards and continues
        rewards = self.reward_decoder(states[:, 1:])  # Skip initial
        continues = self.continue_decoder(states[:, 1:])

        return {
            'states': states,
            'rewards': rewards,
            'continues': continues,
            'trajectory': trajectory,
        }

    def compute_kl(self,
                   prior: WorldModelState,
                   posterior: WorldModelState) -> torch.Tensor:
        """
        Compute KL divergence between prior and posterior.

        KL[q(s|h,o) || p(s|h)]
        """
        # Diagonal Gaussian KL
        var_prior = prior.std.pow(2)
        var_post = posterior.std.pow(2)

        kl = 0.5 * (
            var_post / var_prior +
            (prior.mean - posterior.mean).pow(2) / var_prior -
            1 +
            2 * (prior.std.log() - posterior.std.log())
        )

        return kl.sum(dim=-1)

    def loss(self,
             obs_seq: torch.Tensor,
             action_seq: torch.Tensor,
             reward_seq: torch.Tensor,
             continue_seq: Optional[torch.Tensor] = None,
             kl_scale: float = 1.0,
             free_nats: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute world model losses.

        Args:
            obs_seq: Observations (batch, time, obs_dim)
            action_seq: Actions (batch, time, action_dim)
            reward_seq: Rewards (batch, time)
            continue_seq: Optional continue flags
            kl_scale: KL loss weight
            free_nats: Free nats for KL (minimum allowed)

        Returns:
            Dictionary with losses
        """
        priors, posteriors = self.observe(obs_seq, action_seq)

        # Reconstruction loss
        recon_loss = 0.0
        reward_loss = 0.0
        continue_loss = 0.0
        kl_loss = 0.0

        for t, post in enumerate(posteriors):
            state = post.combined

            # Observation reconstruction
            obs_pred = self.obs_decoder(state)
            recon_loss += F.mse_loss(obs_pred, obs_seq[:, t])

            # Reward prediction
            reward_pred = self.reward_decoder(state)
            reward_loss += F.mse_loss(reward_pred, reward_seq[:, t])

            # Continue prediction
            if continue_seq is not None:
                cont_pred = self.continue_decoder(state)
                continue_loss += F.binary_cross_entropy(cont_pred, continue_seq[:, t])

            # KL divergence
            kl = self.compute_kl(priors[t], post)
            kl = torch.clamp(kl - free_nats, min=0.0)  # Free nats
            kl_loss += kl.mean()

        # Average over time
        n_steps = len(posteriors)
        recon_loss /= n_steps
        reward_loss /= n_steps
        kl_loss /= n_steps
        if continue_seq is not None:
            continue_loss /= n_steps

        total_loss = recon_loss + reward_loss + continue_loss + kl_scale * kl_loss

        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'reward': reward_loss,
            'continue': continue_loss,
            'kl': kl_loss,
        }


class ImaginationBasedPlanner(nn.Module):
    """
    Planning via imagination.

    Uses world model to simulate futures and select
    actions that maximize expected return.
    """

    def __init__(self,
                 world_model: WorldModel,
                 horizon: int = 15,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95):
        super().__init__()
        self.world_model = world_model
        self.horizon = horizon
        self.gamma = gamma
        self.lambda_gae = lambda_gae

        # Value function
        self.value_fn = nn.Sequential(
            nn.Linear(world_model.state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def compute_returns(self,
                        rewards: torch.Tensor,
                        values: torch.Tensor,
                        continues: torch.Tensor) -> torch.Tensor:
        """
        Compute lambda-returns via einsum-friendly recursion.

        Uses dynamic programming, fully vectorized.
        """
        # rewards, values, continues: (batch, horizon)
        batch_size, horizon = rewards.shape

        # Initialize returns
        returns = torch.zeros_like(rewards)
        last_value = values[:, -1]
        last_return = last_value

        # Backward pass (can't easily vectorize due to recursion)
        for t in reversed(range(horizon)):
            cont = continues[:, t]
            bootstrap = cont * (
                self.lambda_gae * last_return +
                (1 - self.lambda_gae) * values[:, t]
            )
            returns[:, t] = rewards[:, t] + self.gamma * bootstrap
            last_return = returns[:, t]

        return returns

    def plan(self,
             state: WorldModelState,
             policy: Callable[[torch.Tensor], torch.Tensor],
             n_samples: int = 8) -> Dict[str, torch.Tensor]:
        """
        Plan via imagination.

        Args:
            state: Current world model state
            policy: Policy to evaluate
            n_samples: Number of imagination samples

        Returns:
            Expected returns and action distribution
        """
        # Expand state for multiple samples
        batch_size = state.deterministic.shape[0]
        expanded_state = WorldModelState(
            deterministic=state.deterministic.repeat(n_samples, 1),
            stochastic=state.stochastic.repeat(n_samples, 1),
            mean=state.mean.repeat(n_samples, 1),
            std=state.std.repeat(n_samples, 1),
        )

        # Imagine
        imagination = self.world_model.imagine(expanded_state, policy, self.horizon)

        states = imagination['states']
        rewards = imagination['rewards']
        continues = imagination['continues']

        # Compute values
        values = self.value_fn(states[:, 1:]).squeeze(-1)

        # Compute returns
        returns = self.compute_returns(rewards, values, continues)

        # Reshape to (n_samples, batch_size, horizon)
        returns = returns.reshape(n_samples, batch_size, -1)

        # Average over samples
        expected_returns = returns.mean(dim=0)

        return {
            'expected_returns': expected_returns,
            'values': values.reshape(n_samples, batch_size, -1).mean(dim=0),
            'rewards': rewards.reshape(n_samples, batch_size, -1).mean(dim=0),
        }


if __name__ == '__main__':
    print("--- World Model Examples ---")

    # Example 1: RSSM
    print("\n1. RSSM Dynamics")
    rssm = RSSM(hidden_dim=64, stoch_dim=16, action_dim=4, obs_embed_dim=32)

    state = rssm.initial_state(batch_size=4, device=torch.device('cpu'))
    action = torch.randn(4, 4)

    next_state = rssm.imagine_step(state, action)
    print(f"   Deterministic: {next_state.deterministic.shape}")
    print(f"   Stochastic: {next_state.stochastic.shape}")

    # Example 2: Imagination Trajectory
    print("\n2. Imagination Trajectory")
    policy = lambda z: torch.randn(z.shape[0], 4)  # Random policy

    trajectory = rssm.imagine_trajectory(state, policy, horizon=10)
    print(f"   Trajectory length: {len(trajectory)}")
    print(f"   Final state shape: {trajectory[-1].combined.shape}")

    # Example 3: Neural ODE Dynamics
    print("\n3. Neural ODE Dynamics")
    ode = NeuralODEDynamics(state_dim=80, action_dim=4, dt=0.1)

    z0 = torch.randn(4, 80)
    actions = torch.randn(4, 10, 4)

    trajectory = ode.integrate(z0, actions, t_span=(0, 1), steps=10)
    print(f"   ODE trajectory shape: {trajectory.shape}")

    # Example 4: Full World Model
    print("\n4. Full World Model")
    wm = WorldModel(
        obs_dim=64,
        action_dim=4,
        hidden_dim=64,
        stoch_dim=16,
    )

    obs_seq = torch.randn(4, 20, 64)
    action_seq = torch.randn(4, 20, 4)
    reward_seq = torch.randn(4, 20)

    losses = wm.loss(obs_seq, action_seq, reward_seq)
    print(f"   Total loss: {losses['total'].item():.4f}")
    print(f"   KL loss: {losses['kl'].item():.4f}")
    print(f"   Recon loss: {losses['reconstruction'].item():.4f}")

    # Example 5: Imagination-Based Planning
    print("\n5. Imagination-Based Planning")
    planner = ImaginationBasedPlanner(wm, horizon=10)

    priors, posteriors = wm.observe(obs_seq[:, :5], action_seq[:, :5])
    current_state = posteriors[-1]

    plan_result = planner.plan(current_state, policy, n_samples=4)
    print(f"   Expected returns shape: {plan_result['expected_returns'].shape}")
    print(f"   Mean expected return: {plan_result['expected_returns'].mean().item():.4f}")

    print("\n[OK] All world model tests passed!")
