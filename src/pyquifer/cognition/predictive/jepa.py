"""
Joint-Embedding Predictive Architecture (JEPA).

Predicts in latent space rather than observation space — enables a world
model that imagines "meaning" rather than "pixels." Uses VICReg or Barlow
regularization to prevent representation collapse.

Key classes:
- JEPAEncoder: Encode observations to latent space
- JEPAPredictor: Predict target embedding from context
- VICRegLoss: Variance-Invariance-Covariance regularization
- BarlowLoss: Barlow Twins redundancy reduction
- ActionJEPA: Action-conditioned variant for embodied planning

References:
- LeCun (2022). A Path Towards Autonomous Machine Intelligence.
- Assran et al. (2023). Self-Supervised Learning from Images with a
  Joint-Embedding Predictive Architecture. CVPR 2023.
- Bardes et al. (2024). V-JEPA: Latent Video Prediction for Visual Representation.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAEncoder(nn.Module):
    """Encode observations to latent representations.

    Produces embeddings that capture semantic content while being
    amenable to prediction in latent space.

    Args:
        input_dim: Observation dimension.
        latent_dim: Latent representation dimension.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of encoder layers.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent space.

        Args:
            x: (..., input_dim) observations

        Returns:
            (..., latent_dim) latent representations
        """
        return self.encoder(x)


class JEPAPredictor(nn.Module):
    """Predict target embedding from context embedding.

    A lightweight predictor that maps context embeddings to target
    embeddings in latent space. Asymmetric design prevents collapse.

    Args:
        latent_dim: Latent representation dimension.
        hidden_dim: Predictor hidden dimension.
        num_layers: Number of predictor layers.
        context_dim: Optional additional context dimension.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        context_dim: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        in_dim = latent_dim + context_dim

        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.predictor = nn.Sequential(*layers)

    def forward(
        self,
        context: torch.Tensor,
        additional_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict target embedding.

        Args:
            context: (..., latent_dim) context embedding
            additional_context: Optional (..., context_dim) extra context

        Returns:
            (..., latent_dim) predicted target embedding
        """
        if additional_context is not None:
            x = torch.cat([context, additional_context], dim=-1)
        else:
            x = context
        return self.predictor(x)


class VICRegLoss(nn.Module):
    """Variance-Invariance-Covariance regularization.

    Prevents representation collapse in JEPA by maintaining:
    - Variance: each dimension stays active (non-constant)
    - Invariance: matched pairs have similar representations
    - Covariance: different dimensions are decorrelated

    Args:
        sim_weight: Weight for invariance (MSE) term.
        var_weight: Weight for variance term.
        cov_weight: Weight for covariance term.
        var_target: Target standard deviation per dimension.
    """

    def __init__(
        self,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        var_target: float = 1.0,
    ):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_target = var_target

    def forward(
        self, z_pred: torch.Tensor, z_target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute VICReg loss.

        Args:
            z_pred: (B, D) predicted embeddings
            z_target: (B, D) target embeddings (detached from encoder)

        Returns:
            Dict with 'loss', 'invariance', 'variance', 'covariance'
        """
        B, D = z_pred.shape

        # Invariance: MSE between matched pairs
        inv_loss = F.mse_loss(z_pred, z_target)

        # Variance: keep std > target for each dimension
        std_pred = z_pred.std(dim=0)
        std_target = z_target.std(dim=0)
        var_loss = (
            F.relu(self.var_target - std_pred).mean() +
            F.relu(self.var_target - std_target).mean()
        )

        # Covariance: decorrelate dimensions
        z_pred_c = z_pred - z_pred.mean(dim=0)
        z_target_c = z_target - z_target.mean(dim=0)
        cov_pred = (z_pred_c.T @ z_pred_c) / (B - 1)
        cov_target = (z_target_c.T @ z_target_c) / (B - 1)
        # Off-diagonal elements should be zero
        off_diag_pred = cov_pred.fill_diagonal_(0).pow(2).sum() / D
        off_diag_target = cov_target.fill_diagonal_(0).pow(2).sum() / D
        cov_loss = off_diag_pred + off_diag_target

        loss = (
            self.sim_weight * inv_loss +
            self.var_weight * var_loss +
            self.cov_weight * cov_loss
        )

        return {
            'loss': loss,
            'invariance': inv_loss,
            'variance': var_loss,
            'covariance': cov_loss,
        }


class BarlowLoss(nn.Module):
    """Barlow Twins redundancy reduction loss.

    Makes the cross-correlation matrix between embeddings approach
    the identity matrix.

    Args:
        lambda_off_diag: Weight for off-diagonal terms.
    """

    def __init__(self, lambda_off_diag: float = 0.005):
        super().__init__()
        self.lambda_off_diag = lambda_off_diag

    def forward(
        self, z_pred: torch.Tensor, z_target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute Barlow Twins loss.

        Args:
            z_pred: (B, D) predicted embeddings (batch-normalized)
            z_target: (B, D) target embeddings (batch-normalized)

        Returns:
            Dict with 'loss', 'on_diagonal', 'off_diagonal'
        """
        B, D = z_pred.shape

        # Normalize each dimension to zero mean, unit variance
        z_pred_n = (z_pred - z_pred.mean(dim=0)) / (z_pred.std(dim=0) + 1e-5)
        z_target_n = (z_target - z_target.mean(dim=0)) / (z_target.std(dim=0) + 1e-5)

        # Cross-correlation matrix
        cc = (z_pred_n.T @ z_target_n) / B  # (D, D)

        # Loss: diagonal should be 1, off-diagonal should be 0
        on_diag = (1 - cc.diagonal()).pow(2).sum()
        off_diag = cc.fill_diagonal_(0).pow(2).sum()

        loss = on_diag + self.lambda_off_diag * off_diag

        return {
            'loss': loss,
            'on_diagonal': on_diag,
            'off_diagonal': off_diag,
        }


class ActionJEPA(nn.Module):
    """Action-conditioned JEPA for embodied planning.

    Predicts how actions transform latent representations. Enables
    planning by imagining future latent states given action sequences.

    z_{t+1} = Predictor(Encoder(x_t), a_t)

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        latent_dim: Latent representation dimension.
        hidden_dim: Hidden layer dimension.
        reg_type: Regularization type ('vicreg' or 'barlow').
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        reg_type: str = 'vicreg',
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Context encoder (online, updated by gradient)
        self.context_encoder = JEPAEncoder(obs_dim, latent_dim, hidden_dim)

        # Target encoder (EMA updated, no gradient)
        self.target_encoder = JEPAEncoder(obs_dim, latent_dim, hidden_dim)
        # Initialize target with same weights
        for p_t, p_c in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.copy_(p_c.data)
            p_t.requires_grad_(False)

        # Action-conditioned predictor
        self.predictor = JEPAPredictor(latent_dim, hidden_dim, context_dim=action_dim)

        # Regularization
        if reg_type == 'vicreg':
            self.reg_loss = VICRegLoss()
        else:
            self.reg_loss = BarlowLoss()

        self._ema_decay = 0.996

    @torch.no_grad()
    def update_target(self):
        """EMA update of target encoder."""
        for p_t, p_c in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.lerp_(p_c.data, 1.0 - self._ema_decay)

    def forward(
        self,
        obs_t: torch.Tensor,
        action: torch.Tensor,
        obs_tp1: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            obs_t: (B, obs_dim) current observation
            action: (B, action_dim) action taken
            obs_tp1: (B, obs_dim) next observation

        Returns:
            Dict with 'loss', 'z_context', 'z_pred', 'z_target', regularization terms
        """
        # Encode
        z_context = self.context_encoder(obs_t)

        with torch.no_grad():
            z_target = self.target_encoder(obs_tp1)

        # Predict next latent given context + action
        z_pred = self.predictor(z_context, action)

        # Compute regularized loss
        reg = self.reg_loss(z_pred, z_target.detach())

        return {
            'loss': reg['loss'],
            'z_context': z_context,
            'z_pred': z_pred,
            'z_target': z_target,
            **{k: v for k, v in reg.items() if k != 'loss'},
        }

    @torch.no_grad()
    def imagine(
        self,
        obs: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Imagine future states given action sequence.

        Args:
            obs: (B, obs_dim) starting observation
            action_sequence: (B, T, action_dim) planned actions

        Returns:
            (B, T+1, latent_dim) predicted latent trajectory
        """
        z = self.context_encoder(obs)
        trajectory = [z]

        for t in range(action_sequence.shape[1]):
            z = self.predictor(z, action_sequence[:, t])
            trajectory.append(z)

        return torch.stack(trajectory, dim=1)
