"""
Oscillatory Multiplexing and Temporal Coding Module for PyQuifer

Implements phase-based information gating, cross-frequency coupling,
and temporal coding mechanisms inspired by how the brain multiplexes
competing information streams through precise timing relationships.

Key concepts:
- Phase-gated attention: Different phases open/close processing windows
- Cross-frequency coupling: Slow oscillation phase modulates fast amplitude
- Temporal coding: Information encoded in timing relative to oscillations
- Nested oscillations: Fast rhythms organized by slow rhythm structure

Based on research by Pascal Fries, Gyorgy Buzsaki, Wolf Singer.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List, Literal


class PhaseGate(nn.Module):
    """
    Phase-dependent gating of information flow.

    Uses the phase of a reference oscillation to selectively gate
    information. Different phases can "open" or "close" processing
    windows, enabling attention-like mechanisms and temporal multiplexing.

    Inspired by Pascal Fries' "Communication Through Coherence" theory.
    """

    def __init__(self,
                 dim: int,
                 num_phases: int = 4,
                 gate_sharpness: float = 2.0,
                 learnable_preferences: bool = True):
        """
        Args:
            dim: Dimension of gated signal
            num_phases: Number of discrete phase bins (e.g., 4 = quadrants)
            gate_sharpness: How sharply gates open/close (higher = more binary)
            learnable_preferences: Whether phase preferences are learnable
        """
        super().__init__()
        self.dim = dim
        self.num_phases = num_phases
        self.gate_sharpness = gate_sharpness

        # Learnable phase preference for each dimension
        # Each dimension can prefer to be "open" at different phases
        if learnable_preferences:
            self.phase_preferences = nn.Parameter(
                torch.rand(dim) * 2 * math.pi  # Random initial preferences
            )
        else:
            self.register_buffer('phase_preferences', torch.zeros(dim))

        # Learnable gate width (how long each dimension stays open)
        self.log_gate_width = nn.Parameter(torch.zeros(dim))  # log for positivity

    @property
    def gate_width(self) -> torch.Tensor:
        """Gate width in radians (always positive)."""
        return torch.sigmoid(self.log_gate_width) * math.pi  # 0 to pi

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Gate input based on current oscillation phase.

        Args:
            x: Input tensor (batch, dim) or (batch, seq, dim)
            phase: Current oscillation phase in [0, 2*pi] - scalar or (batch,)

        Returns:
            Gated output with same shape as input
        """
        if phase.dim() == 0:
            phase = phase.unsqueeze(0)
        if x.dim() == 2:
            # (batch, dim)
            phase = phase.unsqueeze(1)  # (batch, 1)
        elif x.dim() == 3:
            # (batch, seq, dim)
            phase = phase.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)

        # Compute phase distance from each dimension's preferred phase
        # Using circular distance
        phase_diff = torch.abs(phase - self.phase_preferences)
        phase_diff = torch.minimum(phase_diff, 2 * math.pi - phase_diff)

        # Gate value: high when phase is near preference, low otherwise
        # Uses a smooth cosine-like function
        gate = torch.exp(-self.gate_sharpness * (phase_diff / self.gate_width) ** 2)

        return x * gate

    def get_phase_profile(self, num_points: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the gating profile across all phases for visualization.

        Returns:
            phases: (num_points,) tensor of phase values
            gates: (num_points, dim) tensor of gate values
        """
        phases = torch.linspace(0, 2 * math.pi, num_points)
        gates = []
        for p in phases:
            gate = self.forward(torch.ones(1, self.dim), p.unsqueeze(0))
            gates.append(gate.squeeze(0))
        return phases, torch.stack(gates)


class CrossFrequencyCoupling(nn.Module):
    """
    Cross-frequency coupling between slow and fast oscillations.

    Implements phase-amplitude coupling where the phase of a slow
    oscillation modulates the amplitude of faster oscillations.
    This creates hierarchical temporal structure for information processing.

    Based on theta-gamma coupling observed in hippocampus (Buzsaki).
    """

    def __init__(self,
                 num_fast_oscillators: int,
                 coupling_strength: float = 1.0,
                 preferred_phases: Optional[torch.Tensor] = None):
        """
        Args:
            num_fast_oscillators: Number of fast oscillators to modulate
            coupling_strength: How strongly slow phase affects fast amplitude
            preferred_phases: Preferred slow phase for each fast oscillator
        """
        super().__init__()
        self.num_fast = num_fast_oscillators
        self.coupling_strength = nn.Parameter(torch.tensor(coupling_strength))

        # Each fast oscillator has a preferred phase of the slow oscillation
        # where its amplitude is maximal
        if preferred_phases is not None:
            self.preferred_phases = nn.Parameter(preferred_phases)
        else:
            # Distribute evenly across slow cycle by default
            self.preferred_phases = nn.Parameter(
                torch.linspace(0, 2 * math.pi, num_fast_oscillators + 1)[:-1]
            )

        # Phase spread (how sharply tuned to preferred phase)
        self.log_phase_spread = nn.Parameter(torch.zeros(num_fast_oscillators))

    @property
    def phase_spread(self) -> torch.Tensor:
        """Phase spread (kappa parameter of von Mises)."""
        return torch.exp(self.log_phase_spread) + 0.5

    def forward(self,
                fast_amplitude: torch.Tensor,
                slow_phase: torch.Tensor) -> torch.Tensor:
        """
        Modulate fast oscillator amplitudes based on slow phase.

        Args:
            fast_amplitude: Amplitude of fast oscillators (batch, num_fast) or (num_fast,)
            slow_phase: Phase of slow oscillation in [0, 2*pi] - scalar or (batch,)

        Returns:
            Modulated fast amplitudes
        """
        if slow_phase.dim() == 0:
            slow_phase = slow_phase.unsqueeze(0)
        if fast_amplitude.dim() == 1:
            fast_amplitude = fast_amplitude.unsqueeze(0)

        # Von Mises-like modulation
        # Each fast oscillator's amplitude peaks at its preferred slow phase
        phase_diff = slow_phase.unsqueeze(1) - self.preferred_phases.unsqueeze(0)

        # Circular modulation function
        modulation = torch.exp(
            self.phase_spread * (torch.cos(phase_diff) - 1)
        )

        # Apply modulation
        modulated = fast_amplitude * (1 + self.coupling_strength * modulation)

        return modulated

    def get_coupling_profile(self, num_points: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get modulation profile across slow phases.

        Returns:
            phases: (num_points,) slow phases
            modulation: (num_points, num_fast) modulation for each fast oscillator
        """
        phases = torch.linspace(0, 2 * math.pi, num_points)
        base_amp = torch.ones(1, self.num_fast)
        mods = []
        for p in phases:
            mod = self.forward(base_amp, p.unsqueeze(0))
            mods.append(mod.squeeze(0))
        return phases, torch.stack(mods)


class TemporalMultiplexer(nn.Module):
    """
    Multiplexes multiple input streams into phase-locked temporal slots.

    Takes N input streams and assigns each to a different phase of
    a reference oscillation, allowing them to be processed sequentially
    without interference. Useful for handling competing sensory inputs.
    """

    def __init__(self,
                 num_streams: int,
                 stream_dim: int,
                 oscillator_freq: float = 1.0,
                 dt: float = 0.01):
        """
        Args:
            num_streams: Number of input streams to multiplex
            stream_dim: Dimension of each input stream
            oscillator_freq: Frequency of multiplexing oscillation (Hz)
            dt: Time step
        """
        super().__init__()
        self.num_streams = num_streams
        self.stream_dim = stream_dim
        self.dt = dt

        # Each stream gets a phase slot
        slot_width = 2 * math.pi / num_streams
        self.register_buffer(
            'stream_phases',
            torch.linspace(slot_width / 2, 2 * math.pi - slot_width / 2, num_streams)
        )

        # Oscillator for multiplexing
        self.omega = nn.Parameter(torch.tensor(oscillator_freq * 2 * math.pi))
        self.register_buffer('current_phase', torch.tensor(0.0))

        # Phase gates for each stream
        self.gates = nn.ModuleList([
            PhaseGate(stream_dim, gate_sharpness=3.0, learnable_preferences=False)
            for _ in range(num_streams)
        ])
        # Set each gate to prefer its assigned phase slot
        for i, gate in enumerate(self.gates):
            with torch.no_grad():
                gate.phase_preferences.fill_(self.stream_phases[i].item())

    def forward(self,
                streams: List[torch.Tensor],
                steps: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multiplex input streams over time.

        Args:
            streams: List of input tensors, each (batch, stream_dim)
            steps: Number of oscillation steps to run

        Returns:
            output: Combined multiplexed output (batch, steps, stream_dim)
            phases: Phase at each step (steps,)
        """
        assert len(streams) == self.num_streams

        batch_size = streams[0].shape[0]
        outputs = []
        phases = []

        for _ in range(steps):
            # Gate each stream based on current phase
            gated_streams = []
            for i, (stream, gate) in enumerate(zip(streams, self.gates)):
                gated = gate(stream, self.current_phase)
                gated_streams.append(gated)

            # Combine gated streams (they're mostly non-overlapping in time)
            combined = sum(gated_streams)
            outputs.append(combined)
            phases.append(self.current_phase.clone())

            # Advance oscillator
            with torch.no_grad():
                self.current_phase = (self.current_phase + self.omega * self.dt) % (2 * math.pi)

        return torch.stack(outputs, dim=1), torch.stack(phases)

    def reset_phase(self):
        """Reset oscillator phase to zero."""
        self.current_phase.zero_()


class PhaseEncoder(nn.Module):
    """
    Encodes information in the phase of spikes/events relative to oscillation.

    Implements phase-of-firing coding where the precise phase at which
    an event occurs carries information. Earlier phase = stronger/more salient.

    Based on hippocampal phase precession (O'Keefe, Buzsaki).
    """

    def __init__(self,
                 input_dim: int,
                 num_phase_bins: int = 8,
                 reference_freq: float = 8.0):
        """
        Args:
            input_dim: Dimension of input to encode
            num_phase_bins: Number of discrete phase bins for encoding
            reference_freq: Reference oscillation frequency (theta ~ 8 Hz)
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_phase_bins = num_phase_bins
        self.reference_freq = reference_freq

        # Learnable mapping from input value to preferred phase
        # Higher values -> earlier phase (phase precession)
        self.value_to_phase = nn.Linear(input_dim, input_dim)

        # Phase bin centers
        self.register_buffer(
            'bin_centers',
            torch.linspace(0, 2 * math.pi, num_phase_bins + 1)[:-1]
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input values as phases.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            phases: Encoded phases (batch, input_dim) in [0, 2*pi]
            phase_bins: Discretized phase bins (batch, input_dim) as integers
        """
        # Map input to phase (sigmoid to [0, 2*pi])
        phase_logits = self.value_to_phase(x)
        phases = torch.sigmoid(phase_logits) * 2 * math.pi

        # Discretize to bins
        bin_width = 2 * math.pi / self.num_phase_bins
        phase_bins = (phases / bin_width).long() % self.num_phase_bins

        return phases, phase_bins

    def decode(self, phases: torch.Tensor, reference_phase: torch.Tensor) -> torch.Tensor:
        """
        Decode phase-encoded information relative to reference oscillation.

        Args:
            phases: Encoded phases (batch, input_dim)
            reference_phase: Current reference oscillation phase

        Returns:
            activation: How strongly each input "fires" at current phase
        """
        if reference_phase.dim() == 0:
            reference_phase = reference_phase.unsqueeze(0)

        # Circular distance from encoded phase to reference
        phase_diff = torch.abs(phases - reference_phase.unsqueeze(1))
        phase_diff = torch.minimum(phase_diff, 2 * math.pi - phase_diff)

        # Activation peaks when reference phase matches encoded phase
        activation = torch.exp(-2.0 * phase_diff)

        return activation


class NestedOscillator(nn.Module):
    """
    Nested oscillatory dynamics with fast rhythms organized by slow rhythms.

    Implements theta-gamma nesting where multiple gamma cycles occur
    within each theta cycle, with the gamma phase resetting at theta troughs.

    This creates a hierarchical temporal structure for organizing
    information across multiple timescales.
    """

    def __init__(self,
                 slow_freq: float = 8.0,
                 fast_freq: float = 40.0,
                 num_fast_per_slow: int = 5,
                 dt: float = 0.001):
        """
        Args:
            slow_freq: Slow (theta) frequency in Hz
            fast_freq: Fast (gamma) frequency in Hz
            num_fast_per_slow: Target number of fast cycles per slow cycle
            dt: Time step in seconds
        """
        super().__init__()
        self.dt = dt
        self.num_fast_per_slow = num_fast_per_slow

        # Learnable frequencies
        self.log_slow_omega = nn.Parameter(torch.tensor(math.log(slow_freq * 2 * math.pi)))
        self.log_fast_omega = nn.Parameter(torch.tensor(math.log(fast_freq * 2 * math.pi)))

        # Coupling: how much slow phase resets fast phase
        self.reset_strength = nn.Parameter(torch.tensor(0.5))

        # Phase states
        self.register_buffer('slow_phase', torch.tensor(0.0))
        self.register_buffer('fast_phase', torch.tensor(0.0))

        # Track slow phase for reset detection
        self.register_buffer('prev_slow_phase', torch.tensor(0.0))

    @property
    def slow_omega(self) -> torch.Tensor:
        return torch.exp(self.log_slow_omega)

    @property
    def fast_omega(self) -> torch.Tensor:
        return torch.exp(self.log_fast_omega)

    def forward(self, steps: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run nested oscillation for given number of steps.

        Args:
            steps: Number of time steps

        Returns:
            slow_phases: Slow oscillation phases (steps,)
            fast_phases: Fast oscillation phases (steps,)
            fast_in_slow: Position of each fast phase within slow cycle (steps,)
        """
        slow_phases = []
        fast_phases = []
        fast_in_slow = []

        for _ in range(steps):
            # Store current phases
            slow_phases.append(self.slow_phase.clone())
            fast_phases.append(self.fast_phase.clone())

            # Compute position of fast phase within current slow cycle
            # (useful for phase-amplitude coupling analysis)
            slow_cycle_position = self.slow_phase / (2 * math.pi)
            fast_in_slow.append(slow_cycle_position.clone())

            # Detect slow phase reset (crossing through 0)
            crossed_zero = (self.prev_slow_phase > math.pi) and (self.slow_phase < math.pi)

            # Update phases
            with torch.no_grad():
                self.prev_slow_phase.copy_(self.slow_phase)
                self.slow_phase.add_(self.slow_omega * self.dt).remainder_(2 * math.pi)
                self.fast_phase.add_(self.fast_omega * self.dt).remainder_(2 * math.pi)

                # Reset fast phase at slow trough (with some noise/flexibility)
                if crossed_zero:
                    reset_target = 0.0
                    self.fast_phase.mul_(1 - self.reset_strength).add_(self.reset_strength * reset_target).remainder_(2 * math.pi)

        return torch.stack(slow_phases), torch.stack(fast_phases), torch.stack(fast_in_slow)

    def reset(self):
        """Reset both oscillators to phase 0."""
        self.slow_phase.zero_()
        self.fast_phase.zero_()
        self.prev_slow_phase.zero_()


if __name__ == '__main__':
    print("--- Oscillatory Multiplexing Examples ---")

    # Example 1: Phase Gate
    print("\n1. Phase Gate")
    gate = PhaseGate(dim=4, num_phases=4, gate_sharpness=2.0)

    x = torch.ones(2, 4)  # Constant input
    for phase in [0, math.pi/2, math.pi, 3*math.pi/2]:
        gated = gate(x, torch.tensor(phase))
        print(f"   Phase {phase:.2f}: gate = {gated[0].tolist()}")

    # Example 2: Cross-Frequency Coupling
    print("\n2. Cross-Frequency Coupling (Theta-Gamma)")
    cfc = CrossFrequencyCoupling(num_fast_oscillators=4, coupling_strength=0.5)

    fast_amp = torch.ones(4)  # Base gamma amplitude
    for theta_phase in [0, math.pi/2, math.pi, 3*math.pi/2]:
        modulated = cfc(fast_amp, torch.tensor(theta_phase))
        print(f"   Theta phase {theta_phase:.2f}: gamma amp = {modulated.squeeze().tolist()}")

    # Example 3: Temporal Multiplexer
    print("\n3. Temporal Multiplexer")
    mux = TemporalMultiplexer(num_streams=3, stream_dim=2, oscillator_freq=10.0)

    streams = [
        torch.tensor([[1.0, 0.0]]),  # Stream 1
        torch.tensor([[0.0, 1.0]]),  # Stream 2
        torch.tensor([[0.5, 0.5]]),  # Stream 3
    ]

    output, phases = mux(streams, steps=20)
    print(f"   Output shape: {output.shape}")
    print(f"   Phases range: {phases.min().item():.2f} to {phases.max().item():.2f}")

    # Example 4: Phase Encoder
    print("\n4. Phase Encoder")
    encoder = PhaseEncoder(input_dim=3, num_phase_bins=8)

    x = torch.tensor([[0.1, 0.5, 0.9]])  # Low, medium, high values
    phases, bins = encoder.encode(x)
    print(f"   Input: {x.squeeze().tolist()}")
    print(f"   Encoded phases: {phases.squeeze().tolist()}")
    print(f"   Phase bins: {bins.squeeze().tolist()}")

    # Example 5: Nested Oscillator
    print("\n5. Nested Oscillator (Theta-Gamma)")
    nested = NestedOscillator(slow_freq=8.0, fast_freq=40.0, dt=0.001)

    slow, fast, positions = nested(steps=500)  # 500ms
    print(f"   Slow oscillation cycles: {(slow[-1] / (2*math.pi)).item():.1f}")
    print(f"   Fast oscillation cycles: {(fast.diff().abs().sum() / (2*math.pi)).item():.0f}")

    # Count fast cycles per slow cycle
    slow_resets = (slow[1:] < slow[:-1]).sum().item()
    fast_cycles = (fast[1:] < fast[:-1]).sum().item()
    print(f"   Ratio fast/slow: {fast_cycles / max(slow_resets, 1):.1f}")
