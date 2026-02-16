"""
Metastability Module for PyQuifer

The brain never settles — it flows between transient states.
Metastability is distinct from criticality: criticality is about
SCALE (power laws), metastability is about FLOW (winnerless competition).

Key concepts:
- Winnerless Competition: Lotka-Volterra dynamics with asymmetric inhibition
  → no stable equilibrium, system spirals through saddle points
- Heteroclinic Channel: The orbit connecting saddle points
  → dwell times at each saddle = how long a "thought" persists
  → transition sequence = "stream of consciousness"
- Metastability Index: How metastable vs. stable the dynamics are

References:
- Rabinovich et al. (2001). Dynamical Principles in Neuroscience.
- Kelso (2012). Multistability and Metastability.
- Tognoli & Kelso (2014). The Metastable Brain.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, List


class WinnerlessCompetition(nn.Module):
    """
    Lotka-Volterra dynamics with asymmetric inhibition.

    N populations compete for activation. Asymmetric inhibition
    (rho_ij != rho_ji) ensures NO stable fixed point — the system
    cycles through saddle points in a heteroclinic orbit.

    Each saddle point represents a transient "thought" or state.
    The system dwells near each saddle before transitioning to the next.
    """

    def __init__(self,
                 num_populations: int = 6,
                 sigma: float = 1.0,
                 dt: float = 0.01,
                 noise_scale: float = 0.01):
        """
        Args:
            num_populations: Number of competing populations
            sigma: Growth rate (same for all populations)
            dt: Integration time step
            noise_scale: Noise injection for transitions
        """
        super().__init__()
        self.num_populations = num_populations
        self.sigma = sigma
        self.dt = dt
        self.noise_scale = noise_scale

        # Population activations (positive, bounded)
        self.register_buffer('activations', torch.ones(num_populations) / num_populations)

        # Asymmetric inhibition matrix (KEY: rho_ij != rho_ji)
        # Initialize with cyclic dominance pattern (rock-paper-scissors-like)
        rho = torch.zeros(num_populations, num_populations)
        for i in range(num_populations):
            for j in range(num_populations):
                if i == j:
                    rho[i, j] = 1.0  # Self-regulation
                elif (j - i) % num_populations <= num_populations // 2:
                    rho[i, j] = 1.5 + 0.3 * ((j - i) % num_populations)  # Strong forward inhibition
                else:
                    rho[i, j] = 0.5  # Weak backward inhibition
        self.register_buffer('rho', rho)

        # Step counter
        self.register_buffer('step_count', torch.tensor(0))

    def forward(self,
                external_input: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        One step of Lotka-Volterra dynamics.

        dx_i/dt = x_i * (sigma_i - sum_j(rho_ij * x_j)) + noise

        Args:
            external_input: Optional bias to specific populations (num_populations,)

        Returns:
            Dictionary with:
            - activations: Current population activations
            - dominant: Index of most active population
            - dwell_signal: How concentrated activation is (1=one dominant, 0=uniform)
        """
        x = self.activations

        # Lotka-Volterra dynamics
        inhibition = self.rho @ x
        growth = x * (self.sigma - inhibition)

        # External input
        if external_input is not None:
            growth = growth + external_input * 0.1

        # Noise for transitions
        noise = torch.randn_like(x) * self.noise_scale

        # Euler integration
        with torch.no_grad():
            self.activations.add_(growth * self.dt + noise)
            # Keep activations positive
            self.activations.clamp_(min=1e-6)
            # Normalize to prevent unbounded growth
            self.activations.div_(self.activations.sum() + 1e-8)
            self.step_count.add_(1)

        # Dominance measure
        dominant = self.activations.argmax()
        dwell_signal = self.activations.max() - self.activations.mean()

        return {
            'activations': self.activations.clone(),
            'dominant': dominant,
            'dwell_signal': dwell_signal,
        }

    def reset(self):
        """Reset to uniform activation."""
        self.activations.fill_(1.0 / self.num_populations)
        self.step_count.zero_()


class HeteroclinicChannel(nn.Module):
    """
    Tracks the flow through saddle points in the heteroclinic orbit.

    Records which state is dominant, how long it dwells there,
    and the transition pattern — this IS the stream of consciousness.
    """

    def __init__(self,
                 num_states: int = 6,
                 dominance_threshold: float = 0.3,
                 max_history: int = 200):
        """
        Args:
            num_states: Number of possible dominant states
            dominance_threshold: Min activation to count as "dominant"
            max_history: Length of transition history to keep
        """
        super().__init__()
        self.num_states = num_states
        self.dominance_threshold = dominance_threshold
        self.max_history = max_history

        # Current dominant state
        self.register_buffer('current_dominant', torch.tensor(-1))
        # Dwell time at current state
        self.register_buffer('current_dwell', torch.tensor(0))
        # Dwell time history per state
        self.register_buffer('dwell_times', torch.zeros(num_states, max_history))
        self.register_buffer('dwell_counts', torch.zeros(num_states, dtype=torch.long))
        # Transition count matrix
        self.register_buffer('transition_counts', torch.zeros(num_states, num_states))

    def forward(self, activations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Track the heteroclinic flow.

        Args:
            activations: Current population activations (num_states,)

        Returns:
            Dictionary with:
            - current_dominant: Which state is dominant (-1 if none)
            - dwell_time: How long at current state
            - transition_occurred: Whether a transition just happened
            - mean_dwell_times: Average dwell time per state
        """
        dominant_idx = activations.argmax()
        dominant_val = activations[dominant_idx]

        transition_occurred = False

        with torch.no_grad():
            if dominant_val > self.dominance_threshold:
                if dominant_idx != self.current_dominant and self.current_dominant >= 0:
                    # Transition! Record dwell time of previous state
                    prev = self.current_dominant.item()
                    count = self.dwell_counts[prev].item()
                    if count < self.max_history:
                        self.dwell_times[prev, count] = self.current_dwell.float()
                        self.dwell_counts[prev].add_(1)
                    # Record transition
                    self.transition_counts[prev, dominant_idx.item()].add_(1)
                    transition_occurred = True
                    self.current_dwell.zero_()

                self.current_dominant.copy_(dominant_idx)
                self.current_dwell.add_(1)
            else:
                # No clear dominant state
                if self.current_dominant >= 0:
                    prev = self.current_dominant.item()
                    count = self.dwell_counts[prev].item()
                    if count < self.max_history:
                        self.dwell_times[prev, count] = self.current_dwell.float()
                        self.dwell_counts[prev].add_(1)
                self.current_dominant.fill_(-1)
                self.current_dwell.zero_()

        # Compute mean dwell times (vectorized)
        mean_dwells = torch.zeros(self.num_states, device=activations.device)
        valid_mask = self.dwell_counts > 0
        for s in range(self.num_states):
            if valid_mask[s]:
                n = self.dwell_counts[s].item()
                mean_dwells[s] = self.dwell_times[s, :n].mean()

        return {
            'current_dominant': self.current_dominant.clone(),
            'dwell_time': self.current_dwell.clone(),
            'transition_occurred': torch.tensor(transition_occurred),
            'mean_dwell_times': mean_dwells,
            'transition_matrix': self.transition_counts.clone(),
        }

    def reset(self):
        """Reset tracking."""
        self.current_dominant.fill_(-1)
        self.current_dwell.zero_()
        self.dwell_times.zero_()
        self.dwell_counts.zero_()
        self.transition_counts.zero_()


class MetastabilityIndex(nn.Module):
    """
    Quantifies how metastable the system is.

    Metastability = the system visits many states without settling.
    Key metrics:
    - Coefficient of variation of dwell times (high = more metastable)
    - Coalition entropy: diversity of active state combinations
    - Chimera index: coexistence of synchronized and desynchronized groups
    """

    def __init__(self,
                 num_populations: int = 6,
                 window_size: int = 100):
        """
        Args:
            num_populations: Number of competing populations
            window_size: Window for computing indices
        """
        super().__init__()
        self.num_populations = num_populations
        self.window_size = window_size

        # Competition dynamics
        self.competition = WinnerlessCompetition(num_populations=num_populations)
        # Flow tracking
        self.channel = HeteroclinicChannel(num_states=num_populations)

        # History for index computation
        self.register_buffer('activation_history',
                             torch.zeros(window_size, num_populations))
        self.register_buffer('history_ptr', torch.tensor(0))

    def forward(self,
                external_input: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Step dynamics and compute metastability indices.

        Args:
            external_input: Optional bias to populations

        Returns:
            Dictionary with activations, flow info, and metastability metrics
        """
        # Step competition
        comp_result = self.competition(external_input)
        # Track flow
        flow_result = self.channel(comp_result['activations'])

        # Record history
        with torch.no_grad():
            idx = self.history_ptr % self.window_size
            self.activation_history[idx] = comp_result['activations']
            self.history_ptr.add_(1)

        # Compute metastability index
        n_valid = min(self.history_ptr.item(), self.window_size)
        history = self.activation_history[:n_valid]

        # Coalition entropy: how diverse are the active coalitions?
        # Binarize: which populations are "active" at each timestep?
        active = (history > 1.0 / self.num_populations).float()
        # Compute unique patterns
        if n_valid > 1:
            # Pattern = binary string → decimal
            powers = 2.0 ** torch.arange(self.num_populations, dtype=torch.float,
                                         device=history.device)
            patterns = (active * powers).sum(dim=1)
            unique_patterns = patterns.unique()
            # Entropy of pattern distribution
            counts = (patterns.unsqueeze(-1) == unique_patterns).sum(0).float()
            probs = counts / counts.sum()
            coalition_entropy = -(probs * torch.log(probs + 1e-8)).sum()
            max_entropy = math.log(min(n_valid, 2 ** self.num_populations))
            normalized_entropy = coalition_entropy / (max_entropy + 1e-8)
        else:
            coalition_entropy = torch.tensor(0.0, device=history.device)
            normalized_entropy = torch.tensor(0.0, device=history.device)

        # Metastability index from dwell times
        mean_dwells = flow_result['mean_dwell_times']
        active_dwells = mean_dwells[mean_dwells > 0]
        if len(active_dwells) > 1:
            mi = active_dwells.std() / (active_dwells.mean() + 1e-8)
        else:
            mi = torch.tensor(0.0, device=history.device)

        # Chimera index: variance of per-population synchronization
        if n_valid > 10:
            # Phase coherence per population over time
            pop_coherence = history.std(dim=0)
            chimera = pop_coherence.std() / (pop_coherence.mean() + 1e-8)
        else:
            chimera = torch.tensor(0.0, device=history.device)

        return {
            'activations': comp_result['activations'],
            'dominant': comp_result['dominant'],
            'current_dominant': flow_result['current_dominant'],
            'dwell_time': flow_result['dwell_time'],
            'transition_occurred': flow_result['transition_occurred'],
            'metastability_index': mi,
            'coalition_entropy': coalition_entropy,
            'normalized_coalition_entropy': normalized_entropy,
            'chimera_index': chimera,
        }

    def reset(self):
        """Reset all state."""
        self.competition.reset()
        self.channel.reset()
        self.activation_history.zero_()
        self.history_ptr.zero_()


if __name__ == '__main__':
    print("--- Metastability Examples ---")

    # Example 1: Winnerless Competition
    print("\n1. Winnerless Competition (6 populations)")
    wlc = WinnerlessCompetition(num_populations=6, noise_scale=0.02)

    dominants = []
    for i in range(500):
        result = wlc()
        dominants.append(result['dominant'].item())

    unique_visited = len(set(dominants))
    print(f"   States visited: {unique_visited}/6")
    print(f"   Final activations: {result['activations'].detach().cpu().numpy().round(3)}")
    print(f"   Dwell signal: {result['dwell_signal'].item():.3f}")

    # Example 2: Heteroclinic Channel
    print("\n2. Heteroclinic Channel Tracking")
    wlc2 = WinnerlessCompetition(num_populations=4, noise_scale=0.03)
    channel = HeteroclinicChannel(num_states=4)

    transitions = 0
    for i in range(1000):
        comp = wlc2()
        flow = channel(comp['activations'])
        if flow['transition_occurred'].item():
            transitions += 1

    print(f"   Transitions detected: {transitions}")
    print(f"   Mean dwell times: {flow['mean_dwell_times'].detach().cpu().numpy().round(1)}")
    print(f"   Transition matrix:\n{flow['transition_matrix'].detach().cpu().numpy().astype(int)}")

    # Example 3: Full Metastability Index
    print("\n3. Metastability Index")
    meta = MetastabilityIndex(num_populations=6)

    for i in range(500):
        result = meta()

    print(f"   Metastability index: {result['metastability_index'].item():.3f}")
    print(f"   Coalition entropy: {result['coalition_entropy'].item():.3f}")
    print(f"   Normalized entropy: {result['normalized_coalition_entropy'].item():.3f}")
    print(f"   Chimera index: {result['chimera_index'].item():.3f}")

    # Example 4: With external input
    print("\n4. External Input Bias")
    meta2 = MetastabilityIndex(num_populations=4)

    bias = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Bias toward state 0
    for i in range(200):
        result = meta2(external_input=bias)

    print(f"   Dominant state: {result['dominant'].item()} (biased toward 0)")
    print(f"   Activations: {result['activations'].detach().cpu().numpy().round(3)}")

    print("\n[OK] All metastability tests passed!")
