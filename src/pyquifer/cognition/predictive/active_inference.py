"""
Active Inference - Predictive coding and expected free energy.

Extracted from Deep_AIF and pymdp patterns. Key concepts:
- Agents minimize surprise (prediction error)
- Expected Free Energy guides action selection
- Beliefs updated via variational inference

This enables oscillatory systems to act as active inference agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick for sampling from Gaussian.

    Allows backprop through stochastic sampling.

    Args:
        mu: Mean of distribution
        logvar: Log variance

    Returns:
        Sample from N(mu, exp(logvar))
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence_gaussian(mu_q: torch.Tensor, logvar_q: torch.Tensor,
                          mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between two diagonal Gaussians.

    KL(q || p) where q = N(mu_q, var_q), p = N(mu_p, var_p)

    Args:
        mu_q, logvar_q: Parameters of q distribution
        mu_p, logvar_p: Parameters of p distribution

    Returns:
        KL divergence (scalar per batch)
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    kl = 0.5 * (
        var_q / var_p +
        (mu_p - mu_q).pow(2) / var_p -
        1.0 +
        logvar_p - logvar_q
    )

    return kl.sum(dim=-1)


def constrained_activation(x: torch.Tensor,
                          mean_bound: float = 1.0,
                          var_max: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constrained activation for stable latent representations.

    From Deep_AIF: ensures bounded means and positive variances.

    Args:
        x: Input tensor [B, 2*D] (first half = mean, second half = logvar)
        mean_bound: Maximum absolute mean value
        var_max: Maximum variance

    Returns:
        (mean, logvar) with proper constraints
    """
    d = x.shape[-1] // 2

    # Mean bounded to [-mean_bound, mean_bound]
    mean = mean_bound * torch.tanh(x[..., :d])

    # Variance bounded to (0, var_max)
    logvar = torch.log(var_max * torch.sigmoid(x[..., d:]) + 1e-8)

    return mean, logvar


class PredictiveEncoder(nn.Module):
    """
    Encoder that produces predictions about latent states.

    Maps observations to a distribution over latent states,
    enabling predictive coding where the system maintains
    beliefs about hidden causes.
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean + logvar
        )

        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation to latent distribution.

        Args:
            x: Observation [B, input_dim]

        Returns:
            (mean, logvar) of latent distribution
        """
        h = self.net(x)
        mean, logvar = constrained_activation(h)
        return mean, logvar

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Sample latent state from encoded distribution."""
        mean, logvar = self.forward(x)
        return reparameterize(mean, logvar)


class PredictiveDecoder(nn.Module):
    """
    Decoder that generates predictions from latent states.

    The decoder's output represents the system's prediction
    of what observations should look like given latent beliefs.
    """

    def __init__(self,
                 latent_dim: int,
                 output_dim: int,
                 hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent state to observation prediction.

        Args:
            z: Latent state [B, latent_dim]

        Returns:
            Predicted observation [B, output_dim]
        """
        return self.net(z)


class TransitionModel(nn.Module):
    """
    Models state transitions given actions/policies.

    Predicts next latent state distribution given current
    state and action, enabling planning and prediction.
    """

    def __init__(self,
                 latent_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

        self.latent_dim = latent_dim

    def forward(self,
               z: torch.Tensor,
               action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state distribution.

        Args:
            z: Current latent state [B, latent_dim]
            action: Action/policy [B, action_dim]

        Returns:
            (mean, logvar) of predicted next state
        """
        x = torch.cat([z, action], dim=-1)
        h = self.net(x)
        mean, logvar = constrained_activation(h)
        return mean, logvar


class ExpectedFreeEnergy(nn.Module):
    """
    Compute Expected Free Energy (EFE) for action selection.

    EFE decomposes into:
    1. Pragmatic value (achieving preferred outcomes)
    2. Epistemic value (reducing uncertainty)
    3. Entropy of predictions

    Actions that minimize EFE are selected.

    From Deep_AIF paper's formulation.
    """

    def __init__(self,
                 latent_dim: int,
                 num_samples: int = 10,
                 pragmatic_weight: float = 1.0,
                 epistemic_weight: float = 1.0):
        """
        Args:
            latent_dim: Dimension of latent states
            num_samples: Monte Carlo samples for EFE estimation
            pragmatic_weight: Weight for goal-directed term
            epistemic_weight: Weight for information-seeking term
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.pragmatic_weight = pragmatic_weight
        self.epistemic_weight = epistemic_weight

        # Preferred state prior (learnable)
        self.register_buffer(
            'preferred_mean',
            torch.zeros(latent_dim)
        )
        self.register_buffer(
            'preferred_logvar',
            torch.zeros(latent_dim)
        )

    def set_preference(self, mean: torch.Tensor, logvar: torch.Tensor):
        """Set preferred state distribution."""
        self.preferred_mean = mean
        self.preferred_logvar = logvar

    def forward(self,
               predicted_mean: torch.Tensor,
               predicted_logvar: torch.Tensor,
               decoder: Optional[nn.Module] = None,
               observation_model_entropy: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute Expected Free Energy.

        Args:
            predicted_mean: Mean of predicted state [B, latent_dim]
            predicted_logvar: Log variance of predicted state
            decoder: Optional decoder for reconstruction term
            observation_model_entropy: Optional pre-computed entropy

        Returns:
            Dict with EFE components:
            - efe: Total expected free energy
            - pragmatic: Goal-directed component
            - epistemic: Information-seeking component
            - entropy: Predictive entropy
        """
        batch_size = predicted_mean.shape[0]

        # Sample predicted states (vectorized)
        std = torch.exp(0.5 * predicted_logvar)
        eps = torch.randn(predicted_mean.shape[0], self.num_samples, predicted_mean.shape[-1],
                          device=predicted_mean.device)
        samples = predicted_mean.unsqueeze(1) + std.unsqueeze(1) * eps  # [B, S, D]

        # 1. Pragmatic value: KL from predicted to preferred
        pragmatic = kl_divergence_gaussian(
            predicted_mean, predicted_logvar,
            self.preferred_mean.expand(batch_size, -1),
            self.preferred_logvar.expand(batch_size, -1)
        )

        # 2. Epistemic value: expected information gain
        # Approximated as entropy of predicted distribution
        epistemic = 0.5 * predicted_logvar.sum(dim=-1)  # Gaussian entropy

        # 3. Observation entropy (if decoder provided)
        if decoder is not None and observation_model_entropy is None:
            # Estimate via sample variance of decoded predictions
            decoded = decoder(samples.reshape(-1, self.latent_dim))
            decoded = decoded.reshape(batch_size, self.num_samples, -1)
            observation_model_entropy = decoded.var(dim=1).sum(dim=-1)
        elif observation_model_entropy is None:
            observation_model_entropy = torch.zeros(batch_size, device=predicted_mean.device)

        # Total EFE
        efe = (
            self.pragmatic_weight * pragmatic -
            self.epistemic_weight * epistemic +
            observation_model_entropy
        )

        return {
            'efe': efe,
            'pragmatic': pragmatic,
            'epistemic': epistemic,
            'entropy': observation_model_entropy
        }


class BeliefUpdate(nn.Module):
    """
    Fixed-point iteration for belief updating.

    From pymdp's inference algorithms. Iteratively
    updates beliefs until convergence based on:
    - Prior beliefs
    - Likelihood of observations
    - Messages from connected nodes

    Suitable for oscillator networks where each oscillator
    maintains beliefs about its phase.
    """

    def __init__(self,
                 state_dim: int,
                 num_iterations: int = 10,
                 convergence_threshold: float = 1e-4,
                 learning_rate: float = 1.0):
        """
        Args:
            state_dim: Dimension of state beliefs
            num_iterations: Maximum iterations
            convergence_threshold: Stop when change < threshold
            learning_rate: Step size for updates
        """
        super().__init__()

        self.state_dim = state_dim
        self.num_iterations = num_iterations
        self.threshold = convergence_threshold
        self.lr = learning_rate

    def forward(self,
               prior: torch.Tensor,
               likelihood: torch.Tensor,
               messages: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, int]:
        """
        Update beliefs via fixed-point iteration.

        Args:
            prior: Prior belief [B, state_dim]
            likelihood: Observation likelihood [B, state_dim]
            messages: Optional list of messages from neighbors

        Returns:
            (posterior, num_iterations) converged belief and iterations used
        """
        # Initialize with prior
        belief = prior.clone()

        for i in range(self.num_iterations):
            old_belief = belief.clone()

            # Log belief
            log_belief = torch.log(belief + 1e-10)

            # Add log likelihood
            log_posterior = log_belief + torch.log(likelihood + 1e-10)

            # Add messages from neighbors
            if messages is not None:
                for msg in messages:
                    log_posterior = log_posterior + torch.log(msg + 1e-10)

            # Softmax to get normalized belief
            new_belief = F.softmax(log_posterior, dim=-1)

            # Gradient step toward new belief
            belief = belief + self.lr * (new_belief - belief)

            # Check convergence
            change = (belief - old_belief).abs().max()
            if change < self.threshold:
                return belief, i + 1

        return belief, self.num_iterations


class ActiveInferenceAgent(nn.Module):
    """
    Complete Active Inference agent.

    Combines:
    - Predictive encoder (perception)
    - Transition model (prediction)
    - Decoder (generation)
    - EFE computation (action selection)

    The agent perceives, predicts, and acts to minimize
    expected free energy.
    """

    def __init__(self,
                 observation_dim: int,
                 latent_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()

        self.encoder = PredictiveEncoder(observation_dim, latent_dim, hidden_dim)
        self.decoder = PredictiveDecoder(latent_dim, observation_dim, hidden_dim)
        self.transition = TransitionModel(latent_dim, action_dim, hidden_dim)
        self.efe = ExpectedFreeEnergy(latent_dim)

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def perceive(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to latent belief."""
        return self.encoder(observation)

    def predict(self,
               z: torch.Tensor,
               action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next state given action."""
        return self.transition(z, action)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """Generate observation prediction from latent state."""
        return self.decoder(z)

    def select_action(self,
                     observation: torch.Tensor,
                     evaluate_all: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Select action based on EFE.

        Args:
            observation: Current observation [B, obs_dim]
            evaluate_all: If True, evaluate EFE for all actions

        Returns:
            (action, info) selected action and EFE information
        """
        # Get current belief
        z_mean, z_logvar = self.perceive(observation)
        z = reparameterize(z_mean, z_logvar)

        if evaluate_all:
            # Evaluate EFE for each discrete action
            batch_size = observation.shape[0]
            efes = []

            for a in range(self.action_dim):
                action = torch.zeros(batch_size, self.action_dim, device=observation.device)
                action[:, a] = 1.0

                # Predict next state
                next_mean, next_logvar = self.predict(z, action)

                # Compute EFE
                efe_result = self.efe(next_mean, next_logvar, self.decoder)
                efes.append(efe_result['efe'])

            efes = torch.stack(efes, dim=1)  # [B, A]

            # Select action with minimum EFE
            action_idx = efes.argmin(dim=1)
            action = F.one_hot(action_idx, self.action_dim).float()

            return action, {'efes': efes, 'selected': action_idx}

        else:
            # Use policy network
            action_probs = self.policy(observation)
            action = F.gumbel_softmax(action_probs.log(), hard=True)

            return action, {'probs': action_probs}

    def forward(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: perceive, predict, act.

        Args:
            observation: Current observation [B, obs_dim]

        Returns:
            Dict with latent state, reconstruction, action, EFE
        """
        # Perceive (once — select_action reuses this via policy network)
        z_mean, z_logvar = self.perceive(observation)
        z = reparameterize(z_mean, z_logvar)

        # Reconstruct
        reconstruction = self.generate(z)

        # Select action (uses policy network directly to avoid re-encoding)
        action_probs = self.policy(observation)
        action = F.gumbel_softmax(action_probs.log(), hard=True)
        action_info = {'probs': action_probs}

        # Predict next state
        next_mean, next_logvar = self.predict(z, action)

        # Compute EFE
        efe_result = self.efe(next_mean, next_logvar, self.decoder)

        return {
            'z_mean': z_mean,
            'z_logvar': z_logvar,
            'z': z,
            'reconstruction': reconstruction,
            'action': action,
            'next_mean': next_mean,
            'next_logvar': next_logvar,
            'efe': efe_result['efe'],
            'pragmatic': efe_result['pragmatic'],
            'epistemic': efe_result['epistemic']
        }

    def compute_loss(self,
                    observation: torch.Tensor,
                    next_observation: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            observation: Current observation
            next_observation: Optional next observation for transition learning

        Returns:
            Dict with loss components
        """
        result = self.forward(observation)

        # Reconstruction loss
        recon_loss = F.mse_loss(result['reconstruction'], observation)

        # KL loss (regularize latent)
        kl_loss = kl_divergence_gaussian(
            result['z_mean'], result['z_logvar'],
            torch.zeros_like(result['z_mean']),
            torch.zeros_like(result['z_logvar'])
        ).mean()

        # Transition loss (if next observation provided)
        if next_observation is not None:
            next_z_mean, next_z_logvar = self.perceive(next_observation)
            trans_loss = kl_divergence_gaussian(
                result['next_mean'], result['next_logvar'],
                next_z_mean.detach(), next_z_logvar.detach()
            ).mean()
        else:
            trans_loss = torch.tensor(0.0, device=observation.device)

        total_loss = recon_loss + 0.1 * kl_loss + trans_loss

        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'kl': kl_loss,
            'transition': trans_loss
        }
