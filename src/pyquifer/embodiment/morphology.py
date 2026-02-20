"""
Morphological Computation Module for PyQuifer

Implements embodied cognition patterns where computation is stored in
physical state rather than just weights. Inspired by:
- Fascia: Viscoelastic tissue that holds tension patterns
- Peripheral ganglia: Local processing nodes (heart, gut, dorsal root)
- Morphological computation: Body stores computation in material physics

Key concepts:
- TensionField: Persistent state that resists perturbation (fascia analog)
- PeripheralGanglion: Autonomous local processing (heart/gut brain analog)
- SleepWakeController: Attention/arousal gating (cortical wake cycles)
- MorphologicalMemory: Stores skills in oscillatory phase topology

This module is hardware-agnostic (pure PyTorch). For persistent GPU daemons
and hardware-specific implementations, see your deployment project.
"""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TensionField(nn.Module):
    """
    Viscoelastic state that holds tension patterns.

    Analogous to fascial tissue in biology:
    - Stores "tension topology" that persists across processing
    - Resists perturbation with configurable viscoelasticity
    - Integrates with motor commands to create stable postures

    The tension field represents procedural memory - the "muscle memory"
    that holds motor patterns even when the cortex isn't attending.
    """

    def __init__(self,
                 dim: int,
                 persistence: float = 0.99,
                 stiffness: float = 0.1,
                 damping: float = 0.05):
        """
        Args:
            dim: Dimension of the tension field
            persistence: Viscoelastic decay rate (higher = more persistent)
            stiffness: Spring constant for restoring force
            damping: Damping coefficient for oscillation
        """
        super().__init__()
        self.dim = dim
        self.persistence = persistence
        self.stiffness = stiffness
        self.damping = damping

        # Tension state (the "strain" in the fascia)
        self.register_buffer('tension', torch.zeros(dim))

        # Velocity for second-order dynamics
        self.register_buffer('velocity', torch.zeros(dim))

        # Rest position (learned "home" configuration)
        self.rest_position = nn.Parameter(torch.zeros(dim))

        # Trainable viscoelastic properties
        self.local_stiffness = nn.Parameter(torch.ones(dim) * stiffness)

    def forward(self, motor_command: torch.Tensor,
                dt: float = 0.01) -> torch.Tensor:
        """
        Apply motor command and return current tension state.

        Uses spring-damper dynamics:
            F = -k(x - x_rest) - c*v + command
            a = F
            v += a*dt
            x += v*dt

        Args:
            motor_command: External force/command (batch, dim) or (dim,)
            dt: Time step

        Returns:
            Current tension state
        """
        if motor_command.dim() == 1:
            motor_command = motor_command.unsqueeze(0)

        batch_size = motor_command.shape[0]

        # Spring force toward rest position
        displacement = self.tension - self.rest_position
        spring_force = -self.local_stiffness * displacement

        # Damping force
        damping_force = -self.damping * self.velocity

        # Total force = spring + damping + external command
        # Average command over batch for state update
        total_force = spring_force + damping_force + motor_command.mean(dim=0)

        # Update velocity and position
        with torch.no_grad():
            self.velocity.mul_(self.persistence).add_(dt * total_force)
            self.tension.add_(dt * self.velocity)

        # Return tension (expanded to batch if needed)
        if batch_size == 1:
            return self.tension.clone()
        else:
            return self.tension.unsqueeze(0).expand(batch_size, -1).clone()

    def get_strain_energy(self) -> torch.Tensor:
        """Compute stored elastic energy (0.5 * k * x^2)."""
        displacement = self.tension - self.rest_position
        return 0.5 * (self.local_stiffness * displacement ** 2).sum()

    def release(self, fraction: float = 0.5):
        """Release tension (like stretching/relaxation)."""
        with torch.no_grad():
            self.tension = (1 - fraction) * self.tension + fraction * self.rest_position
            self.velocity = self.velocity * (1 - fraction)


class PeripheralGanglion(nn.Module):
    """
    Autonomous local processing node.

    Analogous to peripheral ganglia in biology:
    - Heart: 40,000+ intrinsic cardiac neurons
    - Gut: 200M neurons in enteric nervous system
    - Dorsal root ganglia: Pre-process sensory before spinal cord

    Key property: Can process locally without central (cortical) involvement.
    Only relays to central system when thresholds exceeded.
    """

    def __init__(self,
                 input_dim: int,
                 local_dim: int,
                 output_dim: int,
                 location: str = 'generic',
                 reflex_threshold: float = 0.5,
                 relay_threshold: float = 0.8):
        """
        Args:
            input_dim: Sensory input dimension
            local_dim: Local processing dimension (local "gray matter")
            output_dim: Output dimension
            location: Name/type of ganglion ('cardiac', 'enteric', 'dorsal')
            reflex_threshold: Threshold for local reflex response
            relay_threshold: Threshold for upward relay to CNS
        """
        super().__init__()
        self.location = location
        self.reflex_threshold = reflex_threshold
        self.relay_threshold = relay_threshold

        # Local processing (the "mini-brain")
        self.local_network = nn.Sequential(
            nn.Linear(input_dim, local_dim),
            nn.Tanh(),  # Bounded activation
            nn.Linear(local_dim, local_dim),
            nn.Tanh()
        )

        # Reflex arc (fast, hardcoded response)
        self.reflex_weights = nn.Parameter(
            torch.randn(output_dim, input_dim) / math.sqrt(input_dim)
        )

        # Output projection
        self.output_proj = nn.Linear(local_dim, output_dim)

        # Local state (persistent activity)
        self.register_buffer('local_activity', torch.zeros(local_dim))

        # Statistics for deciding reflex vs relay
        self.register_buffer('input_history', torch.zeros(10, input_dim))
        self.register_buffer('history_ptr', torch.tensor(0))

    def forward(self, sensory_input: torch.Tensor) -> Dict[str, Any]:
        """
        Process sensory input locally.

        Returns dict with:
            - action: Local response (reflex or processed)
            - relay: Signal to send to CNS (or None)
            - mode: 'reflex', 'local', or 'relay'

        Args:
            sensory_input: Input from sensors (batch, input_dim) or (input_dim,)

        Returns:
            Dict with action, relay signal, and processing mode
        """
        if sensory_input.dim() == 1:
            sensory_input = sensory_input.unsqueeze(0)

        batch_size = sensory_input.shape[0]

        # Compute input strength (magnitude)
        input_strength = sensory_input.norm(dim=-1)

        # Update history
        with torch.no_grad():
            self.input_history[self.history_ptr] = sensory_input.mean(dim=0)
            self.history_ptr = (self.history_ptr + 1) % 10

        # Decision logic
        mean_strength = input_strength.mean()

        if mean_strength < self.reflex_threshold:
            # Weak input: Reflex arc only (no local processing)
            action = F.linear(sensory_input, self.reflex_weights)
            mode = 'reflex'
            relay = None

        elif mean_strength < self.relay_threshold:
            # Medium input: Local processing
            local_output = self.local_network(sensory_input)

            # Update local activity (leaky integration)
            self.local_activity = 0.9 * self.local_activity + 0.1 * local_output.mean(dim=0)

            action = self.output_proj(local_output)
            mode = 'local'
            relay = None

        else:
            # Strong input: Process and relay to CNS
            local_output = self.local_network(sensory_input)
            self.local_activity = 0.9 * self.local_activity + 0.1 * local_output.mean(dim=0)

            action = self.output_proj(local_output)
            mode = 'relay'
            # Relay includes both raw input and local interpretation
            relay = {
                'raw': sensory_input,
                'processed': local_output,
                'urgency': mean_strength.item()
            }

        result = {
            'action': action.squeeze(0) if batch_size == 1 else action,
            'relay': relay,
            'mode': mode,
            'local_activity': self.local_activity.clone(),
            'location': self.location
        }

        return result


class SleepWakeController(nn.Module):
    """
    Controls arousal/attention state transitions.

    Models the cortical sleep/wake cycle:
    - Sleep: Low power, no high-level processing, only reflexes active
    - Drowsy: Partial processing, slow responses
    - Wake: Full processing, high power consumption
    - Alert: Maximum processing, depletes energy

    The controller decides when to "wake up" based on input salience
    and when to "sleep" based on energy depletion.
    """

    def __init__(self,
                 dim: int,
                 wake_threshold: float = 0.5,
                 sleep_threshold: float = 0.2,
                 energy_capacity: float = 100.0,
                 wake_cost: float = 1.0,
                 recovery_rate: float = 0.5):
        """
        Args:
            dim: Input/output dimension
            wake_threshold: Salience threshold to wake from sleep
            sleep_threshold: Activity threshold to enter sleep
            energy_capacity: Maximum energy before exhaustion
            wake_cost: Energy cost per wake step
            recovery_rate: Energy recovery per sleep step
        """
        super().__init__()
        self.dim = dim
        self.wake_threshold = wake_threshold
        self.sleep_threshold = sleep_threshold
        self.energy_capacity = energy_capacity
        self.wake_cost = wake_cost
        self.recovery_rate = recovery_rate

        # Current state
        self.register_buffer('state', torch.tensor(0))  # 0=sleep, 1=drowsy, 2=wake, 3=alert
        self.register_buffer('energy', torch.tensor(energy_capacity))

        # State names for logging
        self.state_names = ['sleep', 'drowsy', 'wake', 'alert']

        # Salience detector
        self.salience_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

        # Processing network (only active when awake)
        self.wake_processor = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )

        # Reflex network (active even in sleep)
        self.reflex_processor = nn.Linear(dim, dim)

    def forward(self, input_signal: torch.Tensor) -> Dict[str, Any]:
        """
        Process input based on current arousal state.

        Args:
            input_signal: Input to process (batch, dim) or (dim,)

        Returns:
            Dict with output, state info, and energy level
        """
        if input_signal.dim() == 1:
            input_signal = input_signal.unsqueeze(0)

        batch_size = input_signal.shape[0]

        # Compute salience (always runs - it's subcortical)
        salience = self.salience_net(input_signal).squeeze(-1)
        mean_salience = salience.mean()

        # State transition logic
        with torch.no_grad():
            old_state = self.state.item()

            if old_state == 0:  # Sleep
                if mean_salience > self.wake_threshold:
                    self.state.fill_(2)  # Jump to wake
                self.energy = (self.energy + self.recovery_rate).clamp(max=self.energy_capacity)

            elif old_state == 1:  # Drowsy
                if mean_salience > self.wake_threshold:
                    self.state.fill_(2)
                elif mean_salience < self.sleep_threshold:
                    self.state.fill_(0)
                self.energy = (self.energy + self.recovery_rate * 0.5).clamp(max=self.energy_capacity)

            elif old_state == 2:  # Wake
                if mean_salience > 0.8:
                    self.state.fill_(3)  # Alert
                elif mean_salience < self.sleep_threshold:
                    self.state.fill_(1)  # Drowsy
                self.energy = self.energy - self.wake_cost

            elif old_state == 3:  # Alert
                if mean_salience < 0.6:
                    self.state.fill_(2)  # Back to wake
                self.energy = self.energy - self.wake_cost * 2

            # Exhaustion check
            if self.energy <= 0:
                self.state.fill_(0)  # Forced sleep
                self.energy.fill_(0)

        # Process based on current state
        current_state = self.state.item()

        if current_state == 0:  # Sleep - reflex only
            output = self.reflex_processor(input_signal) * 0.1  # Dampened
            processing_depth = 0.0

        elif current_state == 1:  # Drowsy - partial processing
            reflex_out = self.reflex_processor(input_signal)
            wake_out = self.wake_processor(input_signal)
            output = 0.7 * reflex_out + 0.3 * wake_out
            processing_depth = 0.3

        elif current_state == 2:  # Wake - full processing
            output = self.wake_processor(input_signal)
            processing_depth = 1.0

        else:  # Alert - enhanced processing
            wake_out = self.wake_processor(input_signal)
            # Alert mode: more gain, sharper responses
            output = wake_out * 1.5
            processing_depth = 1.5

        result = {
            'output': output.squeeze(0) if batch_size == 1 else output,
            'state': self.state_names[current_state],
            'state_id': current_state,
            'energy': self.energy.item(),
            'energy_ratio': (self.energy / self.energy_capacity).item(),
            'salience': mean_salience.item(),
            'processing_depth': processing_depth
        }

        return result

    def force_sleep(self):
        """Force the system into sleep state."""
        with torch.no_grad():
            self.state.fill_(0)

    def force_wake(self):
        """Force the system awake (costs energy)."""
        with torch.no_grad():
            if self.energy > 10:
                self.state.fill_(2)


class MorphologicalMemory(nn.Module):
    """
    Stores procedural skills in oscillatory phase topology.

    Unlike weight-based memory, morphological memory stores skills
    as attractor configurations - the "shape" of the phase space
    rather than explicit weights.

    This is analogous to how a musician's hands "know" a piece
    through motor patterns, not declarative recall.
    """

    def __init__(self,
                 dim: int,
                 num_skills: int,
                 attractor_depth: float = 1.0):
        """
        Args:
            dim: State dimension
            num_skills: Number of procedural skills to store
            attractor_depth: Depth of skill attractors (deeper = more robust)
        """
        super().__init__()
        self.dim = dim
        self.num_skills = num_skills
        self.attractor_depth = attractor_depth

        # Each skill is an attractor configuration
        # Stored as phase pattern + coupling pattern
        self.skill_phases = nn.Parameter(
            torch.randn(num_skills, dim) * 0.1
        )
        self.skill_couplings = nn.Parameter(
            torch.randn(num_skills, dim, dim) / dim
        )

        # Current phase state
        self.register_buffer('phase', torch.zeros(dim))

        # Active skill (index or None)
        self.register_buffer('active_skill', torch.tensor(-1))

    def activate_skill(self, skill_idx: int):
        """Activate a stored skill (set attractor target)."""
        if skill_idx < 0 or skill_idx >= self.num_skills:
            raise ValueError(f"Skill index {skill_idx} out of range [0, {self.num_skills})")
        self.active_skill.fill_(skill_idx)

    def deactivate(self):
        """Deactivate current skill (free exploration)."""
        self.active_skill.fill_(-1)

    def forward(self, perturbation: Optional[torch.Tensor] = None,
                dt: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Evolve phase state, pulled by active skill attractor.

        Args:
            perturbation: External perturbation (batch, dim) or (dim,)
            dt: Time step

        Returns:
            Dict with phase, output, and skill activation
        """
        skill_idx = self.active_skill.item()

        if skill_idx >= 0:
            # Get active skill attractor
            target_phase = self.skill_phases[skill_idx]
            coupling = self.skill_couplings[skill_idx]

            # Attractor dynamics: d(phase)/dt = K * sin(target - phase)
            phase_error = target_phase - self.phase
            attractor_force = self.attractor_depth * torch.sin(phase_error)

            # Coupling contribution
            coupled_force = torch.tanh(coupling @ self.phase) * 0.1

            # Update phase
            d_phase = attractor_force + coupled_force
        else:
            # No active skill - free dynamics
            d_phase = torch.zeros_like(self.phase)

        # Add perturbation if provided
        if perturbation is not None:
            if perturbation.dim() > 1:
                perturbation = perturbation.mean(dim=0)
            d_phase = d_phase + perturbation * 0.1

        # Integrate
        with torch.no_grad():
            self.phase.add_(dt * d_phase).remainder_(2 * math.pi)

        # Output is oscillatory signal from phase
        output = torch.cos(self.phase)

        # Compute "skill activation" - how close to attractor
        if skill_idx >= 0:
            target_phase = self.skill_phases[skill_idx]
            alignment = torch.cos(target_phase - self.phase).mean()
        else:
            alignment = 0.0

        return {
            'phase': self.phase.clone(),
            'output': output,
            'active_skill': skill_idx,
            'skill_alignment': alignment.item() if isinstance(alignment, torch.Tensor) else alignment
        }

    def imprint_skill(self, skill_idx: int, trajectory: torch.Tensor,
                      learning_rate: float = 0.1):
        """
        Learn a skill from demonstrated trajectory.

        Args:
            skill_idx: Which skill slot to use
            trajectory: Sequence of states (time, dim)
            learning_rate: Learning rate for imprinting
        """
        with torch.no_grad():
            # Extract final phase as target
            final_state = trajectory[-1]
            self.skill_phases[skill_idx] = (
                (1 - learning_rate) * self.skill_phases[skill_idx] +
                learning_rate * final_state
            )

            # Extract coupling from trajectory correlations
            if trajectory.shape[0] > 1:
                # Compute phase velocity correlations
                velocities = trajectory[1:] - trajectory[:-1]
                cov = velocities.T @ velocities / velocities.shape[0]
                self.skill_couplings[skill_idx] = (
                    (1 - learning_rate) * self.skill_couplings[skill_idx] +
                    learning_rate * cov * 0.1
                )


class DistributedBody(nn.Module):
    """
    Full embodied cognition system with heterogeneous components.

    Combines:
    - TensionField: Fascial "muscle memory"
    - PeripheralGanglia: Local processing nodes (heart, gut, etc.)
    - SleepWakeController: Cortical attention gating
    - MorphologicalMemory: Skill storage in phase topology

    This is the abstract "body" that PyQuifer provides.
    Hardware-specific implementations go in your deployment project.
    """

    def __init__(self,
                 dim: int,
                 num_ganglia: int = 3,
                 num_skills: int = 8):
        """
        Args:
            dim: Base dimension
            num_ganglia: Number of peripheral processing nodes
            num_skills: Number of procedural skills to store
        """
        super().__init__()
        self.dim = dim

        # Fascial layer (persistent tension)
        self.fascia = TensionField(dim)

        # Peripheral ganglia (local processing)
        ganglion_names = ['cardiac', 'enteric', 'dorsal'][:num_ganglia]
        self.ganglia = nn.ModuleDict({
            name: PeripheralGanglion(
                input_dim=dim,
                local_dim=dim // 2,
                output_dim=dim // 4,
                location=name,
                reflex_threshold=0.3 + 0.1 * i,
                relay_threshold=0.7 + 0.1 * i
            )
            for i, name in enumerate(ganglion_names)
        })

        # Cortical controller (sleep/wake)
        self.cortex = SleepWakeController(dim)

        # Procedural memory (skills in phase space)
        self.motor_memory = MorphologicalMemory(dim, num_skills)

        # Integration weights
        self.integration = nn.Linear(dim + dim // 4 * num_ganglia, dim)

    def forward(self, sensory_input: torch.Tensor,
                motor_command: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Full embodied processing cycle.

        Args:
            sensory_input: External sensory input
            motor_command: Optional motor command

        Returns:
            Comprehensive state dict
        """
        if sensory_input.dim() == 1:
            sensory_input = sensory_input.unsqueeze(0)

        batch_size = sensory_input.shape[0]

        # 1. Process through peripheral ganglia (parallel, local)
        ganglion_outputs = {}
        relay_signals = []

        for name, ganglion in self.ganglia.items():
            result = ganglion(sensory_input)
            ganglion_outputs[name] = result
            if result['relay'] is not None:
                relay_signals.append(result['relay'])

        # 2. Update fascial tension (motor integration)
        if motor_command is None:
            # Default: gentle tension update from sensory
            motor_command = sensory_input * 0.1
        tension_state = self.fascia(motor_command)

        # 3. Process through cortex (if awake)
        # Combine ganglia outputs + tension as cortical input
        ganglion_actions = torch.cat([
            ganglion_outputs[name]['action'].unsqueeze(0)
            if ganglion_outputs[name]['action'].dim() == 1
            else ganglion_outputs[name]['action']
            for name in self.ganglia
        ], dim=-1)

        # Expand ganglion_actions to match batch size if needed
        if ganglion_actions.shape[0] == 1 and batch_size > 1:
            ganglion_actions = ganglion_actions.expand(batch_size, -1)

        cortical_input = torch.cat([tension_state, ganglion_actions], dim=-1)
        cortical_input = self.integration(cortical_input)

        cortex_result = self.cortex(cortical_input)

        # 4. Evolve motor memory (skill attractors)
        memory_result = self.motor_memory(perturbation=sensory_input.mean(dim=0))

        # 5. Combine outputs
        output = {
            'cortical_output': cortex_result['output'],
            'fascial_tension': tension_state,
            'ganglion_outputs': ganglion_outputs,
            'motor_memory': memory_result,
            'arousal_state': cortex_result['state'],
            'energy': cortex_result['energy'],
            'relay_count': len(relay_signals),
            'relays': relay_signals if relay_signals else None
        }

        return output


if __name__ == '__main__':
    print("--- Morphological Computation Examples ---")

    # Example 1: TensionField
    print("\n1. TensionField (Fascial Memory)")
    fascia = TensionField(dim=32, persistence=0.99, stiffness=0.1)

    # Apply motor commands
    for t in range(50):
        command = torch.randn(32) * (0.5 if t < 25 else 0.1)
        tension = fascia(command, dt=0.02)

    print(f"   Stored strain energy: {fascia.get_strain_energy().item():.4f}")
    print(f"   Tension norm: {tension.norm().item():.4f}")

    # Release
    fascia.release(fraction=0.5)
    print(f"   After release: {fascia.tension.norm().item():.4f}")

    # Example 2: PeripheralGanglion
    print("\n2. PeripheralGanglion (Heart Brain)")
    heart = PeripheralGanglion(
        input_dim=16, local_dim=32, output_dim=8,
        location='cardiac', reflex_threshold=0.3, relay_threshold=0.7
    )

    # Test different input strengths
    for strength, label in [(0.1, 'weak'), (0.5, 'medium'), (0.9, 'strong')]:
        input_signal = torch.randn(16) * strength
        result = heart(input_signal)
        print(f"   {label} input -> mode: {result['mode']}, relay: {result['relay'] is not None}")

    # Example 3: SleepWakeController
    print("\n3. SleepWakeController")
    cortex = SleepWakeController(dim=32, wake_threshold=0.5, energy_capacity=50)

    # Simulate day/night cycle
    states_seen = []
    for t in range(100):
        # High salience during "day", low during "night"
        salience = 0.7 if 20 < t < 80 else 0.1
        input_signal = torch.randn(32) * salience
        result = cortex(input_signal)
        states_seen.append(result['state'])

    from collections import Counter
    state_counts = Counter(states_seen)
    print(f"   State distribution: {dict(state_counts)}")
    print(f"   Final energy: {cortex.energy.item():.1f}")

    # Example 4: MorphologicalMemory
    print("\n4. MorphologicalMemory")
    memory = MorphologicalMemory(dim=16, num_skills=4, attractor_depth=2.0)

    # Imprint a skill from trajectory
    trajectory = torch.cumsum(torch.randn(20, 16) * 0.1, dim=0)
    memory.imprint_skill(0, trajectory, learning_rate=0.5)

    # Activate skill and evolve
    memory.activate_skill(0)
    for t in range(50):
        result = memory(dt=0.02)

    print(f"   Skill alignment: {result['skill_alignment']:.3f}")
    print(f"   Output norm: {result['output'].norm().item():.3f}")

    # Example 5: DistributedBody
    print("\n5. DistributedBody (Full System)")
    body = DistributedBody(dim=32, num_ganglia=3, num_skills=4)

    sensory = torch.randn(4, 32)  # Batch of 4
    motor = torch.randn(4, 32) * 0.2

    result = body(sensory, motor)

    print(f"   Arousal: {result['arousal_state']}")
    print(f"   Energy: {result['energy']:.1f}")
    print(f"   Relay signals: {result['relay_count']}")
    print(f"   Ganglia modes: {[result['ganglion_outputs'][k]['mode'] for k in body.ganglia]}")

    # Gradient check
    print("\n6. Gradient check")
    input_signal = torch.randn(32, requires_grad=True)
    output = body.fascia(input_signal)
    loss = output.sum()
    loss.backward()
    print(f"   Gradients flow: {input_signal.grad is not None and input_signal.grad.abs().sum() > 0}")

    print("\n[OK] All morphological computation tests passed!")
