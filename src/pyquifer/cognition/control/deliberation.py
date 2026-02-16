"""
Test-Time Compute / Deliberative Reasoning.

Orchestrates search and verification at inference time, allocating
more thinking steps for harder problems. Oscillator coherence serves
as the difficulty signal: low coherence → "think harder" → more compute.

Key classes:
- ProcessRewardModel: Step-level reward estimation
- BeamSearchReasoner: Generate N chains, score, return best
- SelfCorrectionLoop: Detect errors, backtrack, retry
- ComputeBudget: Allocate compute based on difficulty
- Deliberator: Full orchestrator

References:
- Snell et al. (2025). Scaling LLM Test-Time Compute Optimally. ICLR 2025.
- Lightman et al. (2023). Let's Verify Step by Step.
- Wang et al. (2023). Self-Consistency Improves Chain of Thought Reasoning.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


class ProcessRewardModel(nn.Module):
    """Step-level reward estimation for reasoning chains.

    Evaluates the quality of each reasoning step using a small transformer
    that attends to all previous steps. Higher rewards indicate more
    correct/useful reasoning steps.

    Args:
        dim: Embedding dimension per step.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
    """

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.reward_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, step_embeddings: torch.Tensor) -> torch.Tensor:
        """Score each reasoning step.

        Args:
            step_embeddings: (B, T, D) embeddings of reasoning steps

        Returns:
            (B, T) per-step rewards in [0, 1]
        """
        B, T, D = step_embeddings.shape
        # Causal mask: each step only sees previous steps
        mask = torch.triu(
            torch.ones(T, T, device=step_embeddings.device, dtype=torch.bool),
            diagonal=1,
        )
        encoded = self.encoder(step_embeddings, mask=mask)
        rewards = self.reward_head(encoded).squeeze(-1)  # (B, T)
        return rewards


class BeamSearchReasoner(nn.Module):
    """Generate multiple reasoning chains and select the best one.

    Uses beam search with process reward model scoring to maintain
    multiple candidate chains and prune low-quality branches.

    Args:
        dim: Step embedding dimension.
        beam_width: Number of beams to maintain.
        max_steps: Maximum reasoning steps.
        step_generator: Module that generates next step embedding given history.
    """

    def __init__(
        self,
        dim: int,
        beam_width: int = 4,
        max_steps: int = 16,
    ):
        super().__init__()
        self.dim = dim
        self.beam_width = beam_width
        self.max_steps = max_steps

        # Step generator: GRU that predicts next step from history
        self.step_gru = nn.GRU(dim, dim, batch_first=True)
        self.step_proj = nn.Linear(dim, dim)

        # Termination predictor
        self.terminate_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

        # Process reward model
        self.prm = ProcessRewardModel(dim)

    def forward(
        self,
        initial_state: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run beam search reasoning.

        Args:
            initial_state: (B, D) initial reasoning state
            num_steps: Override max_steps if provided

        Returns:
            Dict with:
            - best_chain: (B, T, D) best reasoning chain
            - best_scores: (B, T) per-step rewards of best chain
            - all_chains: (B, beam_width, T, D) all beam chains
            - all_scores: (B, beam_width) total scores
        """
        B = initial_state.shape[0]
        device = initial_state.device
        max_s = num_steps or self.max_steps
        W = self.beam_width

        # Initialize beams: (B * W, 1, D)
        beams = initial_state.unsqueeze(1).unsqueeze(1).expand(B, W, 1, -1)
        beams = beams.reshape(B * W, 1, self.dim)
        beam_scores = torch.zeros(B * W, device=device)
        beam_lengths = torch.ones(B * W, device=device, dtype=torch.long)
        active = torch.ones(B * W, device=device, dtype=torch.bool)

        # Hidden state for GRU
        h = torch.zeros(1, B * W, self.dim, device=device)

        all_steps = [beams]

        for step in range(max_s - 1):
            if not active.any():
                break

            # Generate next step for active beams
            last = beams[:, -1:, :]  # (B*W, 1, D)
            out, h = self.step_gru(last, h)
            next_step = self.step_proj(out)  # (B*W, 1, D)

            # Check termination
            term_prob = self.terminate_head(next_step.squeeze(1))  # (B*W, 1)
            should_terminate = (term_prob.squeeze(-1) > 0.5) | (~active)

            # Append step
            beams = torch.cat([beams, next_step], dim=1)  # (B*W, t+1, D)
            beam_lengths[active] += 1

            # Score with PRM
            rewards = self.prm(beams)  # (B*W, t+1)
            # Update cumulative score
            beam_scores = rewards.mean(dim=-1)

            # Mark terminated beams
            active = active & ~should_terminate

            all_steps.append(next_step)

        # Reshape beams back to (B, W, T, D)
        T = beams.shape[1]
        all_chains = beams.view(B, W, T, self.dim)
        all_scores = beam_scores.view(B, W)

        # Select best beam per batch
        best_idx = all_scores.argmax(dim=-1)  # (B,)
        best_chain = all_chains[
            torch.arange(B, device=device), best_idx
        ]  # (B, T, D)
        best_scores = self.prm(best_chain)  # (B, T)

        return {
            'best_chain': best_chain,
            'best_scores': best_scores,
            'all_chains': all_chains,
            'all_scores': all_scores,
        }


class SelfCorrectionLoop(nn.Module):
    """Detect errors in reasoning and attempt correction.

    Identifies low-scoring steps in a reasoning chain, generates
    alternative continuations, and replaces bad steps with better ones.

    Args:
        dim: Step embedding dimension.
        max_corrections: Maximum correction attempts.
        error_threshold: Steps scoring below this are flagged for correction.
    """

    def __init__(
        self,
        dim: int,
        max_corrections: int = 3,
        error_threshold: float = 0.3,
    ):
        super().__init__()
        self.dim = dim
        self.max_corrections = max_corrections
        self.error_threshold = error_threshold

        # Correction generator: takes bad step + context → corrected step
        self.corrector = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )

        # Process reward model for re-evaluation
        self.prm = ProcessRewardModel(dim)

    def forward(
        self,
        chain: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Attempt to correct low-scoring reasoning steps.

        Args:
            chain: (B, T, D) reasoning chain embeddings
            scores: Optional (B, T) pre-computed step scores

        Returns:
            Dict with:
            - corrected_chain: (B, T, D) corrected chain
            - corrections_made: (B,) number of corrections per sample
            - final_scores: (B, T) scores of corrected chain
            - improved: (B,) whether chain was improved
        """
        B, T, D = chain.shape
        device = chain.device

        if scores is None:
            scores = self.prm(chain)

        corrected = chain.clone()
        corrections_made = torch.zeros(B, device=device, dtype=torch.long)

        for attempt in range(self.max_corrections):
            current_scores = self.prm(corrected)

            # Find worst step per batch below threshold
            below_threshold = current_scores < self.error_threshold
            if not below_threshold.any():
                break

            for b in range(B):
                bad_steps = below_threshold[b].nonzero(as_tuple=False)
                if bad_steps.numel() == 0:
                    continue

                # Fix the worst step
                worst_idx = bad_steps[current_scores[b][bad_steps.squeeze(-1)].argmin()]

                # Context: mean of surrounding steps
                ctx_start = max(0, worst_idx - 1)
                ctx_end = min(T, worst_idx + 2)
                context = corrected[b, ctx_start:ctx_end].mean(dim=0)

                # Generate correction
                correction_input = torch.cat([
                    corrected[b, worst_idx], context
                ], dim=-1)
                new_step = self.corrector(correction_input.unsqueeze(0)).squeeze(0)

                # Apply correction
                corrected[b, worst_idx] = new_step
                corrections_made[b] += 1

        final_scores = self.prm(corrected)
        improved = final_scores.mean(dim=-1) > scores.mean(dim=-1)

        return {
            'corrected_chain': corrected,
            'corrections_made': corrections_made,
            'final_scores': final_scores,
            'improved': improved,
        }


@dataclass
class ComputeBudgetAllocation:
    """Compute budget allocation for deliberation."""
    num_steps: int
    beam_width: int
    num_corrections: int
    difficulty: float


class ComputeBudget:
    """Allocate compute based on problem difficulty.

    Uses oscillator coherence as a proxy for difficulty:
    - High coherence → easy problem → fewer steps
    - Low coherence → hard problem → more steps

    Can also accept explicit task_complexity as override.

    Args:
        min_steps: Minimum reasoning steps.
        max_steps: Maximum reasoning steps.
        min_beam: Minimum beam width.
        max_beam: Maximum beam width.
        coherence_threshold: Below this coherence, allocate max compute.
    """

    def __init__(
        self,
        min_steps: int = 4,
        max_steps: int = 32,
        min_beam: int = 1,
        max_beam: int = 8,
        coherence_threshold: float = 0.3,
    ):
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.min_beam = min_beam
        self.max_beam = max_beam
        self.coherence_threshold = coherence_threshold

    def allocate(
        self,
        coherence: float = 0.5,
        task_complexity: Optional[float] = None,
    ) -> ComputeBudgetAllocation:
        """Allocate compute budget.

        Args:
            coherence: Oscillator coherence in [0, 1].
            task_complexity: Optional explicit complexity in [0, 1].
                If provided, overrides coherence-based estimate.

        Returns:
            ComputeBudgetAllocation with steps, beam width, corrections.
        """
        if task_complexity is not None:
            difficulty = task_complexity
        else:
            # Invert coherence: low coherence = high difficulty
            difficulty = 1.0 - coherence

        # Scale compute with difficulty
        frac = max(0.0, min(1.0, difficulty))

        num_steps = int(self.min_steps + frac * (self.max_steps - self.min_steps))
        beam_width = int(self.min_beam + frac * (self.max_beam - self.min_beam))
        num_corrections = max(1, int(3 * frac))

        return ComputeBudgetAllocation(
            num_steps=num_steps,
            beam_width=beam_width,
            num_corrections=num_corrections,
            difficulty=difficulty,
        )


class Deliberator(nn.Module):
    """Full deliberative reasoning orchestrator.

    Pipeline: ComputeBudget → BeamSearchReasoner → SelfCorrectionLoop → answer

    Oscillator coherence drives how much compute to allocate:
    - Low coherence → hard problem → wide beam + many steps
    - High coherence → easy problem → narrow beam + few steps

    Args:
        dim: Embedding dimension.
        min_steps: Minimum reasoning steps.
        max_steps: Maximum reasoning steps.
        beam_width: Maximum beam width.
        num_layers: Transformer layers in process reward model.
    """

    def __init__(
        self,
        dim: int,
        min_steps: int = 4,
        max_steps: int = 32,
        beam_width: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.dim = dim

        self.budget = ComputeBudget(
            min_steps=min_steps, max_steps=max_steps,
            min_beam=1, max_beam=beam_width,
        )

        self.reasoner = BeamSearchReasoner(
            dim=dim, beam_width=beam_width, max_steps=max_steps,
        )

        self.corrector = SelfCorrectionLoop(
            dim=dim, max_corrections=3,
        )

        # Final answer extraction
        self.answer_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        query_embedding: torch.Tensor,
        coherence: Optional[float] = None,
        task_complexity: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run full deliberative reasoning.

        Args:
            query_embedding: (B, D) initial query/problem embedding
            coherence: Optional oscillator coherence for budget allocation
            task_complexity: Optional explicit complexity override

        Returns:
            Dict with:
            - result: (B, D) final answer embedding
            - reasoning_chain: (B, T, D) reasoning trace
            - confidence: (B,) geometric mean of step rewards
            - compute_used: Dict with budget details
            - corrections_made: (B,) number of corrections
        """
        # Allocate compute budget
        coh = coherence if coherence is not None else 0.5
        budget = self.budget.allocate(coh, task_complexity)

        # Run beam search
        beam_result = self.reasoner(
            query_embedding,
            num_steps=budget.num_steps,
        )

        # Self-correction
        correction_result = self.corrector(
            beam_result['best_chain'],
            beam_result['best_scores'],
        )

        chain = correction_result['corrected_chain']
        scores = correction_result['final_scores']

        # Extract final answer from last step
        result = self.answer_head(chain[:, -1, :])

        # Confidence: geometric mean of step rewards
        log_scores = torch.log(scores.clamp(min=1e-8))
        confidence = torch.exp(log_scores.mean(dim=-1))

        return {
            'result': result,
            'reasoning_chain': chain,
            'confidence': confidence,
            'step_scores': scores,
            'compute_used': {
                'num_steps': budget.num_steps,
                'beam_width': budget.beam_width,
                'corrections': budget.num_corrections,
                'difficulty': budget.difficulty,
            },
            'corrections_made': correction_result['corrections_made'],
        }
