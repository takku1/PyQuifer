"""
Workspace competition dynamics for Global Workspace Theory.

Contains the core competition mechanisms that determine which content
gains access to the global workspace:

- ContentType: Enum for classifying content sources
- WorkspaceItem: Dataclass for items competing for workspace access
- SalienceComputer: Attention-based salience computation
- IgnitionDynamics: Soft phase-transition ignition mechanism
- CompetitionDynamics: Hopfield-like winner-take-all competition
- PrecisionWeighting: Bayesian precision estimation and error weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from dataclasses import dataclass
from enum import Enum


class ContentType(Enum):
    """Types of content that can enter workspace."""
    PERCEPTUAL = 0
    MEMORY = 1
    THOUGHT = 2
    MOTOR = 3
    EMOTIONAL = 4
    METACOGNITIVE = 5


@dataclass
class WorkspaceItem:
    """Item competing for workspace access."""
    content: torch.Tensor       # The actual content
    salience: torch.Tensor      # Current salience/activation
    precision: torch.Tensor     # Reliability/confidence
    source_id: int              # Which module it came from
    content_type: ContentType   # Type of content
    timestamp: int              # When it was created


class SalienceComputer(nn.Module):
    """
    Compute salience of content for workspace access.

    Salience = f(content, context, precision)

    Uses attention mechanism - content queries context
    to determine relevance.
    """

    def __init__(self,
                 content_dim: int,
                 context_dim: int,
                 num_heads: int = 4):
        super().__init__()
        self.content_dim = content_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = content_dim // num_heads

        # Query from content, Key/Value from context
        self.W_q = nn.Parameter(torch.randn(num_heads, content_dim, self.head_dim) * 0.02)
        self.W_k = nn.Parameter(torch.randn(num_heads, context_dim, self.head_dim) * 0.02)

        # Salience projection
        self.salience_proj = nn.Linear(content_dim, 1)

        # Precision integration
        self.precision_scale = nn.Parameter(torch.ones(1))

    def forward(self,
                content: torch.Tensor,
                context: torch.Tensor,
                precision: torch.Tensor) -> torch.Tensor:
        """
        Compute salience of content given context.

        Args:
            content: Content vectors (batch, n_items, content_dim)
            context: Context vectors (batch, n_context, context_dim)
            precision: Precision weights (batch, n_items)

        Returns:
            salience: Salience scores (batch, n_items)
        """
        batch_size, n_items = content.shape[:2]
        n_context = context.shape[1]

        # Multi-head queries and keys via einsum
        # content: (B, I, D), W_q: (H, D, O) -> Q: (B, I, H, O)
        Q = torch.einsum('bid,hdo->biho', content, self.W_q)
        # context: (B, C, D), W_k: (H, D, O) -> K: (B, C, H, O)
        K = torch.einsum('bcd,hdo->bcho', context, self.W_k)

        # Attention scores (batch, n_items, heads, n_context)
        # Q: (B, I, H, O), K: (B, C, H, O) -> attn: (B, I, H, C)
        attn = torch.einsum('biho,bcho->bihc', Q, K) / math.sqrt(self.head_dim)

        # Max attention as relevance (batch, n_items, heads)
        relevance = attn.max(dim=-1)[0]

        # Average over heads (batch, n_items)
        relevance = relevance.mean(dim=-1)

        # Base salience from content
        base_salience = self.salience_proj(content).squeeze(-1)

        # Precision-weighted salience
        salience = (base_salience + relevance) * (precision * self.precision_scale)

        return salience


class IgnitionDynamics(nn.Module):
    """
    Ignition dynamics for global broadcast.

    When salience crosses threshold, content "ignites" and
    becomes globally available. This is a soft phase transition.

    Implements via sigmoid with temperature (sharpness).
    At low temperature, becomes sharp threshold (step function).
    """

    def __init__(self,
                 threshold: float = 0.5,
                 temperature: float = 0.1,
                 refractory_period: int = 5):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.refractory_period = refractory_period

        # Track ignition history
        self.register_buffer('last_ignition', torch.tensor(-1000))
        self.register_buffer('ignition_count', torch.tensor(0))

    def forward(self,
                salience: torch.Tensor,
                current_time: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply ignition dynamics.

        Args:
            salience: Salience scores (batch, n_items)
            current_time: Current timestep

        Returns:
            ignition: Ignition probabilities (batch, n_items)
            is_refractory: Whether in refractory period
        """
        # Refractory check (soft)
        time_since_ignition = current_time - self.last_ignition.float()
        refractory_gate = torch.sigmoid(
            (time_since_ignition - self.refractory_period) / 2.0
        )

        # Ignition probability (soft threshold)
        ignition = torch.sigmoid(
            (salience - self.threshold) / self.temperature
        )

        # Apply refractory gating
        ignition = ignition * refractory_gate

        return ignition, refractory_gate < 0.5

    def update_history(self, did_ignite: bool, current_time: int):
        """Update ignition history."""
        if did_ignite:
            self.last_ignition.fill_(current_time)
            self.ignition_count += 1


class CompetitionDynamics(nn.Module):
    """
    Winner-take-all competition for workspace access.

    Only one (or few) items can occupy the workspace.
    Uses Hopfield-like energy dynamics.

    E(x) = -1/2 x^T W x + b^T x

    Minima correspond to winner states.
    """

    def __init__(self,
                 n_slots: int,
                 inhibition_strength: float = 1.0,
                 n_winners: int = 1):
        super().__init__()
        self.n_slots = n_slots
        self.inhibition_strength = inhibition_strength
        self.n_winners = n_winners

        # Mutual inhibition matrix (off-diagonal negative)
        W = -inhibition_strength * (torch.ones(n_slots, n_slots) - torch.eye(n_slots))
        self.register_buffer('W', W)

    def energy(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute Hopfield energy of activation state.

        Lower energy = more stable configuration.
        Useful for diagnostics and monitoring convergence.
        """
        # E = -1/2 x^T W x
        return -0.5 * torch.einsum('bi,ij,bj->b', activations, self.W, activations)

    def forward(self,
                salience: torch.Tensor,
                temperature: float = 1.0,
                steps: int = 5) -> torch.Tensor:
        """
        Run competition dynamics.

        Args:
            salience: Initial salience (batch, n_items)
            temperature: Softmax temperature
            steps: Number of relaxation steps

        Returns:
            winners: Winning activations (batch, n_items)
        """
        # Pad/truncate to n_slots
        batch_size = salience.shape[0]
        n_items = salience.shape[1]

        if n_items < self.n_slots:
            salience = F.pad(salience, (0, self.n_slots - n_items))
        elif n_items > self.n_slots:
            salience = salience[:, :self.n_slots]

        # Initialize activations
        x = salience.clone()

        # Relaxation dynamics
        for _ in range(steps):
            # Hopfield update
            h = torch.einsum('ij,bj->bi', self.W, x) + salience
            x = torch.softmax(h / temperature, dim=-1)

        # Take top-k winners
        if self.n_winners < self.n_slots:
            topk_vals, topk_idx = x.topk(self.n_winners, dim=-1)
            mask = torch.zeros_like(x)
            mask.scatter_(1, topk_idx, 1.0)
            x = x * mask

        # Normalize winners
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)

        # Return to original n_items size
        if n_items < self.n_slots:
            return x[:, :n_items]
        elif n_items > self.n_slots:
            # Pad back with zeros for items that didn't compete
            return F.pad(x, (0, n_items - self.n_slots))
        return x


class PrecisionWeighting(nn.Module):
    """
    Precision weighting for prediction errors.

    Precision = inverse variance = reliability/confidence.

    High precision errors are more salient.
    Low precision errors are down-weighted.

    Implements optimal Bayesian precision estimation.
    """

    def __init__(self,
                 dim: int,
                 min_precision: float = 0.1,
                 max_precision: float = 10.0):
        super().__init__()
        self.dim = dim
        self.min_precision = min_precision
        self.max_precision = max_precision

        # Precision estimator network
        self.precision_net = nn.Sequential(
            nn.Linear(dim * 2, dim),  # prediction + error
            nn.ELU(),
            nn.Linear(dim, 1),
            nn.Softplus(),  # Ensure positive
        )

        # Running statistics for normalization
        self.register_buffer('error_mean', torch.zeros(dim))
        self.register_buffer('error_var', torch.ones(dim))
        self.register_buffer('n_samples', torch.tensor(0.0))

    def estimate_precision(self,
                          prediction: torch.Tensor,
                          error: torch.Tensor) -> torch.Tensor:
        """
        Estimate precision from prediction and error.

        Args:
            prediction: Model prediction (batch, dim)
            error: Prediction error (batch, dim)

        Returns:
            precision: Estimated precision (batch,)
        """
        # Concatenate prediction and error
        combined = torch.cat([prediction, error], dim=-1)

        # Network estimate
        precision = self.precision_net(combined).squeeze(-1)

        # Clamp to valid range
        precision = torch.clamp(precision, self.min_precision, self.max_precision)

        return precision

    def weight_error(self,
                     error: torch.Tensor,
                     precision: torch.Tensor) -> torch.Tensor:
        """
        Apply precision weighting to error.

        Weighted error = sqrt(precision) * error
        """
        return error * torch.sqrt(precision).unsqueeze(-1)

    def update_statistics(self, error: torch.Tensor):
        """Update running error statistics."""
        with torch.no_grad():
            batch_mean = error.mean(dim=0)
            batch_var = error.var(dim=0)
            batch_size = error.shape[0]

            # Online update
            self.n_samples += batch_size
            delta = batch_mean - self.error_mean
            self.error_mean += delta * batch_size / self.n_samples
            self.error_var = (
                self.error_var * (self.n_samples - batch_size) +
                batch_var * batch_size +
                delta.pow(2) * (self.n_samples - batch_size) * batch_size / self.n_samples
            ) / self.n_samples
