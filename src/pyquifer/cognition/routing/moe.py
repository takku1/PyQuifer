"""
Oscillatory Mixture of Experts — oscillator-state-driven expert routing.

Routes tokens to specialized expert sub-networks based on oscillator band
activity. High gamma coherence → focused specialist expert, low coherence →
broad generalist mixture. Band dominance drives domain routing.

Key classes:
- ExpertPool: Collection of specialized sub-networks
- OscillatorRouter: Route tokens based on oscillator state
- LoadBalancer: Prevent expert collapse
- SparseMoE: Top-k routing with differentiable selection

References:
- Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer.
- Fedus et al. (2022). Switch Transformers: Scaling to Trillion Parameter Models.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertPool(nn.Module):
    """Collection of specialized sub-networks.

    Each expert is a small feedforward network that specializes
    in a particular domain or processing mode.

    Args:
        num_experts: Number of expert networks.
        d_model: Input/output dimension.
        d_expert: Expert hidden dimension.
        dropout: Dropout rate within experts.
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_expert: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        d_expert = d_expert if d_expert > 0 else d_model * 4

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_expert),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_expert, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        expert_idx: int,
    ) -> torch.Tensor:
        """Run input through a specific expert.

        Args:
            x: (..., d_model) input
            expert_idx: Which expert to use

        Returns:
            (..., d_model) expert output
        """
        return self.experts[expert_idx](x)


class LoadBalancer:
    """Prevent expert collapse by tracking and balancing load.

    Computes auxiliary loss that encourages uniform expert utilization.
    Without this, a few experts would handle all tokens.

    Args:
        num_experts: Number of experts.
        balance_weight: Weight for balance auxiliary loss.
    """

    def __init__(self, num_experts: int, balance_weight: float = 0.01):
        self.num_experts = num_experts
        self.balance_weight = balance_weight

    def compute_balance_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute load balancing auxiliary loss.

        Args:
            router_probs: (B*T, num_experts) routing probabilities
            expert_indices: (B*T, k) selected expert indices

        Returns:
            Scalar balance loss
        """
        num_tokens = router_probs.shape[0]

        # Fraction of tokens routed to each expert (vectorized)
        expert_counts = torch.zeros(
            self.num_experts, device=router_probs.device
        )
        # Flatten expert_indices and scatter all at once
        flat_indices = expert_indices.reshape(-1)
        expert_counts.scatter_add_(
            0, flat_indices,
            torch.ones(flat_indices.shape[0], device=router_probs.device),
        )
        fraction_tokens = expert_counts / num_tokens

        # Mean routing probability per expert
        mean_probs = router_probs.mean(dim=0)

        # Balance loss: dot product of fractions and mean probs
        # Minimized when both are uniform (1/num_experts)
        balance_loss = (
            self.num_experts *
            (fraction_tokens * mean_probs).sum()
        )

        return self.balance_weight * balance_loss


class OscillatorRouter(nn.Module):
    """Route tokens to experts based on oscillator band activity.

    Routing logic:
    - High gamma coherence (>0.7) → specialist expert (top-1)
    - Low coherence (<0.3) → generalist mixture (top-k)
    - Band dominance selects domain:
      - theta dominant → memory expert
      - beta dominant → reasoning expert
      - gamma dominant → perception expert

    Args:
        d_model: Model dimension.
        num_experts: Number of experts.
        num_oscillator_features: Oscillator feature dimension.
        top_k: Default number of experts per token.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_oscillator_features: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # Token-based routing
        self.token_gate = nn.Linear(d_model, num_experts, bias=False)

        # Oscillator-based routing modulation
        self.osc_gate = nn.Sequential(
            nn.Linear(num_oscillator_features, num_experts),
            nn.Softmax(dim=-1),
        )

        # Coherence-based k selection
        self.k_predictor = nn.Sequential(
            nn.Linear(num_oscillator_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Mixing weight between token and oscillator routing
        self.mix_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,
        oscillator_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute routing decisions.

        Args:
            x: (B, T, d_model) token features
            oscillator_state: Optional (num_osc_features,) oscillator features
                (coherence values, band powers, etc.)

        Returns:
            Dict with 'router_probs', 'expert_indices', 'expert_weights',
            'effective_k'
        """
        B, T, D = x.shape
        x_flat = x.view(B * T, D)

        # Token-based logits
        token_logits = self.token_gate(x_flat)  # (B*T, E)

        if oscillator_state is not None:
            # Oscillator-based routing bias
            osc_probs = self.osc_gate(oscillator_state)  # (E,)
            osc_bias = osc_probs.unsqueeze(0).expand(B * T, -1)

            # Mix token and oscillator routing
            alpha = torch.sigmoid(self.mix_weight)
            logits = alpha * token_logits + (1 - alpha) * osc_bias * 10.0

            # Adaptive k based on coherence
            k_raw = self.k_predictor(oscillator_state)
            effective_k = max(1, min(
                self.num_experts,
                int(1 + k_raw.item() * (self.top_k - 1) + 0.5)
            ))
        else:
            logits = token_logits
            effective_k = self.top_k

        # Softmax routing probabilities
        router_probs = F.softmax(logits, dim=-1)

        # Top-k selection
        k = min(effective_k, self.num_experts)
        top_weights, top_indices = router_probs.topk(k, dim=-1)

        # Renormalize weights
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)

        return {
            'router_probs': router_probs,
            'expert_indices': top_indices,
            'expert_weights': top_weights,
            'effective_k': effective_k,
        }


class SparseMoE(nn.Module):
    """Sparse Mixture of Experts with oscillator-aware routing.

    Full MoE layer: router selects top-k experts per token,
    experts process their assigned tokens, outputs are
    weighted-summed.

    Args:
        d_model: Model dimension.
        num_experts: Number of expert networks.
        d_expert: Expert hidden dimension.
        top_k: Number of experts per token.
        num_oscillator_features: Oscillator feature dimension.
        dropout: Expert dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        d_expert: int = 0,
        top_k: int = 2,
        num_oscillator_features: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.norm = nn.LayerNorm(d_model)
        self.router = OscillatorRouter(d_model, num_experts, num_oscillator_features, top_k)
        self.experts = ExpertPool(num_experts, d_model, d_expert, dropout)
        self.load_balancer = LoadBalancer(num_experts)

    def forward(
        self,
        x: torch.Tensor,
        oscillator_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Apply sparse MoE layer.

        Args:
            x: (B, T, d_model) input tokens
            oscillator_state: Optional oscillator features for routing

        Returns:
            Dict with 'output', 'balance_loss', 'router_probs', 'effective_k'
        """
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        # Get routing
        routing = self.router(x, oscillator_state)
        indices = routing['expert_indices']    # (B*T, k)
        weights = routing['expert_weights']    # (B*T, k)

        x_flat = x.view(B * T, D)
        output = torch.zeros_like(x_flat)

        # Dispatch to experts
        k = indices.shape[1]
        for i in range(k):
            for e in range(self.num_experts):
                mask = (indices[:, i] == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts(expert_input, e)
                    output[mask] += weights[mask, i].unsqueeze(-1) * expert_output

        output = output.view(B, T, D)

        # Balance loss
        balance_loss = self.load_balancer.compute_balance_loss(
            routing['router_probs'], indices
        )

        return {
            'output': residual + output,
            'balance_loss': balance_loss,
            'router_probs': routing['router_probs'],
            'effective_k': routing['effective_k'],
        }
