"""
Basal Ganglia-Thalamocortical Gating Loop for PyQuifer.

Implements a unified BG-thalamic loop with three pathways:
- Direct (Go): DA(D1) excitation → disinhibit thalamus → ACTION
- Indirect (NoGo): DA(D2) inhibition → maintain thalamic block
- Hyperdirect: Surprise + low coherence → STN emergency stop ALL channels

Operates on organ proposals, outputs salience priors for workspace competition
and processing mode bias.  Reuses existing patterns from SelectionArena
(neural_darwinism.py), NeuromodulatorDynamics (neuromodulation.py), and
OscillatorRouter (oscillatory_moe.py).

References:
- Bogacz & Gurney (2007): Short-circuit in basal ganglia.
- Wiecki & Frank (2013): Computational model of STN conflict.
- Gurney, Prescott & Redgrave (2001): BG as action selection device.
"""

import torch
import torch.nn as nn
import math
from typing import NamedTuple, Optional, List, Set


class GatingOutput(NamedTuple):
    """Result of one BG-thalamic gating step."""
    selected_channel: int          # Winner index (-1 if STN active)
    channel_activations: torch.Tensor  # Post-competition levels (num_channels,)
    salience_prior: torch.Tensor   # Boost per channel for workspace (num_channels,)
    thalamic_gate: float           # Relay strength 0-1
    processing_mode_bias: int      # 0=perception, 1=imagination, 2=balanced
    stn_active: bool               # Hyperdirect engaged
    da_bias: float                 # DA modulation strength
    switching_cost: float          # Penalty for channel change


class BasalGangliaLoop(nn.Module):
    """
    Basal ganglia-thalamocortical gating loop.

    Three pathways:
    - Direct (Go): DA(D1)↑ → channel excitation → disinhibit thalamus → ACTION
    - Indirect (NoGo): DA(D2)↓ → channel inhibition → maintain thalamic block
    - Hyperdirect: Surprise + low coherence → STN → emergency stop ALL channels

    Operates on organ proposals, outputs salience priors for workspace competition.

    Args:
        max_channels: Maximum number of parallel channels (organs).
        channel_dim: Unused placeholder for future per-channel embeddings.
        da_go_weight: D1 pathway DA sensitivity (higher = more Go excitation).
        da_nogo_weight: D2 pathway DA sensitivity (higher = more NoGo inhibition).
        stn_surprise_threshold: Novelty level that triggers STN hyperdirect.
        stn_coherence_threshold: Coherence below which STN can activate.
        stn_refractory_ticks: How many ticks STN suppression persists.
        switching_penalty: Hysteresis cost for non-incumbent channels.
        tonic_inhibition: Baseline GPi inhibition (learnable Parameter).
    """

    def __init__(
        self,
        max_channels: int = 8,
        channel_dim: int = 64,
        da_go_weight: float = 2.0,
        da_nogo_weight: float = 1.5,
        stn_surprise_threshold: float = 0.8,
        stn_coherence_threshold: float = 0.3,
        stn_refractory_ticks: int = 3,
        switching_penalty: float = 0.1,
        tonic_inhibition: float = 0.5,
    ):
        super().__init__()
        self.max_channels = max_channels
        self.channel_dim = channel_dim
        self.da_go_weight = da_go_weight
        self.da_nogo_weight = da_nogo_weight
        self.stn_surprise_threshold = stn_surprise_threshold
        self.stn_coherence_threshold = stn_coherence_threshold
        self.stn_refractory_ticks = stn_refractory_ticks
        self.switching_penalty = switching_penalty

        # Tonic inhibition is learnable — can be tuned by training or manually.
        self.tonic_inhibition = nn.Parameter(torch.tensor(tonic_inhibition))

        # Internal state
        self._prev_winner: int = -1
        self._stn_refractory_counter: int = 0

    def step(
        self,
        saliences: torch.Tensor,
        costs: torch.Tensor,
        tags_list: List[Set[str]],
        neuro_levels: torch.Tensor,
        coherence: float,
        novelty: float = 0.0,
    ) -> GatingOutput:
        """Run one BG-thalamic gating step.

        Args:
            saliences: Per-channel salience values (num_channels,).
            costs: Per-channel predicted compute cost (num_channels,).
            tags_list: Per-channel domain tags (e.g. {'external'}, {'internal'}).
            neuro_levels: Neuromodulator levels [DA, 5HT, NE, ACh, Cortisol].
            coherence: Oscillator order parameter R (0-1).
            novelty: Input novelty signal (0-1), used for STN trigger.

        Returns:
            GatingOutput with winner, salience priors, gate strength, etc.
        """
        device = saliences.device
        num_channels = saliences.shape[0]
        da = float(neuro_levels[0])  # Dopamine is levels[0]

        # ── Direct pathway (Go): D1-type excitation ──
        # High DA → amplifies salient channels
        go = saliences * (1.0 + self.da_go_weight * da)

        # ── Indirect pathway (NoGo): D2-type inhibition ──
        # Low DA → amplifies cost-based suppression
        nogo = costs * (1.0 + self.da_nogo_weight * (1.0 - da))

        # ── Net activation: Go - NoGo - tonic inhibition ──
        activation = go - nogo - self.tonic_inhibition

        # ── Hyperdirect pathway (STN): emergency stop ──
        stn_active = False
        if self._stn_refractory_counter > 0:
            # Still in refractory — suppress all
            stn_active = True
            self._stn_refractory_counter -= 1
            activation = torch.zeros_like(activation)
        elif (novelty > self.stn_surprise_threshold
              and coherence < self.stn_coherence_threshold):
            # Trigger: high surprise + low coherence = conflict
            stn_active = True
            self._stn_refractory_counter = self.stn_refractory_ticks - 1
            activation = torch.zeros_like(activation)

        # ── Hysteresis: incumbent bonus ──
        actual_switching_cost = 0.0
        if not stn_active and self._prev_winner >= 0 and self._prev_winner < num_channels:
            activation = self._compute_hysteresis(activation, self._prev_winner)
            if num_channels > 1:
                # switching_cost is the penalty that WAS applied to the runner-up
                actual_switching_cost = self.switching_penalty

        # ── Winner selection (GPi output) ──
        if stn_active or num_channels == 0:
            selected = -1
            thalamic_gate = 0.0
        else:
            selected = int(activation.argmax().item())
            # Thalamic gate = sigmoid of winner activation above tonic inhibition
            winner_act = activation[selected]
            thalamic_gate = float(torch.sigmoid(winner_act).item())
            self._prev_winner = selected

        # ── Salience prior: boost for workspace competition ──
        # Softmax over activations gives relative salience boost
        if stn_active:
            salience_prior = torch.zeros(num_channels, device=device)
        else:
            # Use softmax to create a smooth prior (not hard winner-take-all)
            salience_prior = torch.softmax(activation, dim=0)

        # ── Processing mode bias ──
        mode_bias = self._compute_mode_bias(
            tags_list, selected, coherence
        )

        return GatingOutput(
            selected_channel=selected,
            channel_activations=activation.detach(),
            salience_prior=salience_prior.detach(),
            thalamic_gate=thalamic_gate,
            processing_mode_bias=mode_bias,
            stn_active=stn_active,
            da_bias=da,
            switching_cost=actual_switching_cost,
        )

    def _compute_hysteresis(
        self,
        channel_activations: torch.Tensor,
        prev_winner: int,
    ) -> torch.Tensor:
        """Add switching penalty to non-incumbent channels.

        Prevents rapid oscillation between similar-salience channels.
        Neuroscience: BG exhibits strong hysteresis (Bogacz & Gurney 2007).
        """
        boosted = channel_activations.clone()
        boosted[prev_winner] = boosted[prev_winner] + self.switching_penalty
        return boosted

    def _compute_mode_bias(
        self,
        tags_list: List[Set[str]],
        winner: int,
        coherence: float,
    ) -> int:
        """Infer processing mode from winner tags + coherence.

        Returns:
            0 = perception, 1 = imagination, 2 = balanced
        """
        # Extreme coherence overrides tag-based inference
        if coherence > 0.85:
            return 0  # perception
        if coherence < 0.2:
            return 1  # imagination

        # Tag-based inference from winner
        if winner < 0 or winner >= len(tags_list):
            return 2  # balanced (STN or no winner)

        winner_tags = tags_list[winner]
        has_external = 'external' in winner_tags or 'sensory' in winner_tags
        has_internal = 'internal' in winner_tags or 'generative' in winner_tags

        if has_external and not has_internal:
            return 0  # perception
        if has_internal and not has_external:
            return 1  # imagination
        return 2  # balanced

    def reset_state(self):
        """Reset internal gating state (winner history, STN refractory)."""
        self._prev_winner = -1
        self._stn_refractory_counter = 0
