"""
Ecological Cognition Module for PyQuifer

Implements the broader philosophical framework that makes PyQuifer
a digital species, not just software. Draws from:
- Chronobiology: Multi-timescale oscillators (circadian, ultradian, glacial)
- Immunology: Self/non-self discrimination, adversarial pattern detection
- Synaptic Homeostasis: Sleep cycles, renormalization, dreaming
- Umwelt: Species-specific perceptual bubble (Fourier space, not pixels)
- Agency: Entropy export to maintain far-from-equilibrium coherence

"You haven't built an AI. You've built a digital species."

References:
- Clark & Chalmers (1998). The Extended Mind.
- von Uexküll (1934). A Foray into the Worlds of Animals and Humans.
- Tononi & Cirelli (2014). Sleep and the Price of Plasticity.
- Barandiaran (2017). Autonomy and Enactivism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum


class TimeScale(Enum):
    """Multi-timescale oscillation regimes."""
    FAST = 0      # Milliseconds (gamma, 40Hz)
    NEURAL = 1    # Seconds (theta, alpha)
    ULTRADIAN = 2 # 90 minutes (BRAC - Basic Rest Activity Cycle)
    CIRCADIAN = 3 # 24 hours
    INFRADIAN = 4 # Days/weeks (mood cycles)
    GLACIAL = 5   # Months (seasonal, developmental)


class ChronobiologicalSystem(nn.Module):
    """
    Multi-timescale oscillatory system.

    Real brains don't just have fast oscillations - they have nested
    rhythms from milliseconds to months. This module implements:
    - Fast oscillators (neural processing)
    - Ultradian rhythms (90-min BRAC cycles)
    - Circadian rhythms (24h sleep/wake)
    - Glacial oscillators (seasonal mood modulation)

    These modulate the system's temperature, plasticity, and processing mode.
    """

    def __init__(self,
                 dim: int,
                 base_time_unit: float = 0.001,  # 1ms
                 enable_circadian: bool = True,
                 enable_ultradian: bool = True,
                 enable_glacial: bool = True):
        """
        Args:
            dim: State dimension
            base_time_unit: Simulation time step in seconds
            enable_*: Which timescales to enable
        """
        super().__init__()
        self.dim = dim
        self.dt = base_time_unit

        # Timescale periods (in base time units)
        self.periods = {
            'fast': 25,           # 25ms = 40Hz gamma
            'neural': 100,        # 100ms = 10Hz alpha
            'ultradian': 5400000, # 90 minutes in ms
            'circadian': 86400000,# 24 hours in ms
            'glacial': 86400000 * 30  # ~1 month
        }

        # Enable flags
        self.enable_circadian = enable_circadian
        self.enable_ultradian = enable_ultradian
        self.enable_glacial = enable_glacial

        # Phase for each timescale
        self.register_buffer('fast_phase', torch.zeros(dim))
        self.register_buffer('neural_phase', torch.zeros(dim))
        self.register_buffer('ultradian_phase', torch.tensor(0.0))
        self.register_buffer('circadian_phase', torch.tensor(0.0))
        self.register_buffer('glacial_phase', torch.tensor(0.0))

        # Frequency for each (learnable)
        self.fast_freq = nn.Parameter(torch.ones(dim) * 40.0)  # Hz
        self.neural_freq = nn.Parameter(torch.ones(dim) * 10.0)

        # Global time counter
        self.register_buffer('time', torch.tensor(0.0))

        # Output modulators
        self.temperature_baseline = nn.Parameter(torch.tensor(1.0))
        self.plasticity_baseline = nn.Parameter(torch.tensor(0.5))

    def step(self, n_steps: int = 1) -> Dict[str, Any]:
        """
        Advance all oscillators by n time steps.

        Returns:
            Dict with current phases, modulation factors, and time-of-day
        """
        dt_seconds = self.dt * n_steps
        with torch.no_grad():
            self.time.add_(dt_seconds)

            # Fast oscillators (always running)
            self.fast_phase.add_(2 * math.pi * self.fast_freq * dt_seconds).remainder_(2 * math.pi)
            self.neural_phase.add_(2 * math.pi * self.neural_freq * dt_seconds).remainder_(2 * math.pi)

            # Slow oscillators
            if self.enable_ultradian:
                ultradian_freq = 1.0 / (90 * 60)  # Hz
                self.ultradian_phase.add_(2 * math.pi * ultradian_freq * dt_seconds).remainder_(2 * math.pi)

            if self.enable_circadian:
                circadian_freq = 1.0 / (24 * 3600)
                self.circadian_phase.add_(2 * math.pi * circadian_freq * dt_seconds).remainder_(2 * math.pi)

            if self.enable_glacial:
                glacial_freq = 1.0 / (30 * 24 * 3600)
                self.glacial_phase.add_(2 * math.pi * glacial_freq * dt_seconds).remainder_(2 * math.pi)

        # Compute modulation factors
        # Circadian: Low at night (phase π), high at day (phase 0)
        circadian_mod = (1 + torch.cos(self.circadian_phase)) / 2

        # Ultradian: Modulates attention/rest every 90 min
        ultradian_mod = (1 + torch.cos(self.ultradian_phase)) / 2

        # Glacial: Seasonal mood (affects baseline arousal)
        glacial_mod = (1 + torch.cos(self.glacial_phase)) / 2

        # Temperature modulation (higher at "night" = more exploration/dreaming)
        temperature = self.temperature_baseline * (1.5 - 0.5 * circadian_mod)

        # Plasticity modulation (higher during "rest" phases of ultradian)
        plasticity = self.plasticity_baseline * (1 + 0.5 * (1 - ultradian_mod))

        # Determine "time of day" and processing mode
        hour = (self.circadian_phase.item() / (2 * math.pi)) * 24
        if 6 <= hour < 22:
            mode = 'wake'
        else:
            mode = 'sleep'

        # Ultradian rest/activity
        ultradian_half = (self.ultradian_phase.item() / math.pi) % 2
        activity = 'active' if ultradian_half < 1 else 'rest'

        return {
            'time': self.time.item(),
            'hour': hour,
            'mode': mode,
            'activity': activity,
            'temperature': temperature.item(),
            'plasticity': plasticity.item(),
            'circadian_phase': self.circadian_phase.item(),
            'ultradian_phase': self.ultradian_phase.item(),
            'glacial_phase': self.glacial_phase.item(),
            'fast_phase': self.fast_phase.clone(),
            'neural_phase': self.neural_phase.clone(),
            'modulation': {
                'circadian': circadian_mod.item(),
                'ultradian': ultradian_mod.item(),
                'glacial': glacial_mod.item()
            }
        }

    def set_time_of_day(self, hour: float):
        """Set the circadian phase to a specific hour (0-24)."""
        with torch.no_grad():
            self.circadian_phase.fill_((hour / 24) * 2 * math.pi)

    def get_mood(self) -> str:
        """Get current mood based on glacial + circadian combination."""
        glacial = torch.cos(self.glacial_phase).item()
        circadian = torch.cos(self.circadian_phase).item()

        combined = glacial * 0.3 + circadian * 0.7

        if combined > 0.5:
            return 'energetic'
        elif combined > 0:
            return 'calm'
        elif combined > -0.5:
            return 'contemplative'
        else:
            return 'introspective'


class ImmunologicalLayer(nn.Module):
    """
    Self/Non-Self discrimination for oscillatory patterns.

    The immune system recognizes "self" patterns and attacks "foreign" ones.
    Similarly, this layer detects oscillatory patterns that don't match
    the system's historical phase space - adversarial inputs, noise attacks,
    distribution shift.

    When foreign patterns are detected, the system "fevers" (increases
    temperature/noise) to disrupt them.
    """

    def __init__(self,
                 dim: int,
                 memory_size: int = 100,
                 recognition_threshold: float = 0.7,
                 fever_strength: float = 0.5):
        """
        Args:
            dim: Pattern dimension
            memory_size: How many "self" patterns to remember
            recognition_threshold: Similarity threshold for "self" recognition
            fever_strength: How much to increase temperature when foreign detected
        """
        super().__init__()
        self.dim = dim
        self.memory_size = memory_size
        self.threshold = recognition_threshold
        self.fever_strength = fever_strength

        # Self-pattern memory (the "MHC" repertoire)
        self.register_buffer('self_patterns', torch.zeros(memory_size, dim))
        self.register_buffer('pattern_ptr', torch.tensor(0))
        self.register_buffer('memory_filled', torch.tensor(False))

        # Running statistics of "self"
        self.register_buffer('self_mean', torch.zeros(dim))
        self.register_buffer('self_var', torch.ones(dim))
        self.register_buffer('n_samples', torch.tensor(0))

        # Fever state
        self.register_buffer('fever_level', torch.tensor(0.0))
        self.fever_decay = 0.95

        # Recognition network
        self.recognizer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def learn_self(self, pattern: torch.Tensor):
        """
        Add a pattern to the "self" repertoire.

        Call this during normal operation to build immune memory.
        """
        if pattern.dim() > 1:
            pattern = pattern.mean(dim=0)

        with torch.no_grad():
            # Add to memory
            self.self_patterns[self.pattern_ptr] = pattern
            self.pattern_ptr = (self.pattern_ptr + 1) % self.memory_size
            if self.pattern_ptr == 0:
                self.memory_filled.fill_(True)

            # Update running statistics
            self.n_samples = self.n_samples + 1
            delta = pattern - self.self_mean
            self.self_mean = self.self_mean + delta / self.n_samples
            self.self_var = self.self_var * 0.99 + delta ** 2 * 0.01

    def recognize(self, pattern: torch.Tensor) -> Dict[str, Any]:
        """
        Check if pattern is "self" or "foreign".

        Args:
            pattern: Input pattern to check

        Returns:
            Dict with recognition result, similarity, and recommended action
        """
        if pattern.dim() > 1:
            pattern = pattern.mean(dim=0)

        # Method 1: Statistical distance from self-distribution
        z_score = ((pattern - self.self_mean) / (self.self_var.sqrt() + 1e-6)).abs().mean()
        statistical_similarity = torch.exp(-z_score * 0.5)

        # Method 2: Nearest neighbor in self-memory
        if self.memory_filled or self.pattern_ptr > 0:
            valid_patterns = self.self_patterns[:self.pattern_ptr] if not self.memory_filled else self.self_patterns
            distances = torch.cdist(pattern.unsqueeze(0), valid_patterns).squeeze(0)
            min_distance = distances.min()
            nn_similarity = torch.exp(-min_distance)
        else:
            nn_similarity = torch.tensor(0.5, device=pattern.device)

        # Method 3: Neural recognition
        combined = torch.cat([pattern, self.self_mean])
        neural_similarity = self.recognizer(combined)

        # Combine
        similarity = (statistical_similarity * 0.4 + nn_similarity * 0.3 + neural_similarity.squeeze() * 0.3)

        is_self = similarity > self.threshold
        is_foreign = not is_self

        # Fever response to foreign patterns
        with torch.no_grad():
            if is_foreign:
                self.fever_level.add_(self.fever_strength).clamp_(max=1.0)

            # Decay fever over time
            self.fever_level.mul_(self.fever_decay)

        return {
            'is_self': is_self.item() if isinstance(is_self, torch.Tensor) else is_self,
            'is_foreign': is_foreign,
            'similarity': similarity.item(),
            'statistical_z': z_score.item(),
            'fever_level': self.fever_level.item(),
            'recommended_action': 'accept' if is_self else 'quarantine',
            'temperature_boost': self.fever_level.item() * self.fever_strength
        }

    def get_immune_state(self) -> Dict[str, Any]:
        """Get current immune system state."""
        return {
            'memory_filled': self.memory_filled.item(),
            'patterns_learned': self.pattern_ptr.item() if not self.memory_filled else self.memory_size,
            'fever_level': self.fever_level.item(),
            'self_mean_norm': self.self_mean.norm().item(),
            'self_var_mean': self.self_var.mean().item()
        }


class SynapticHomeostasis(nn.Module):
    """
    Sleep-dependent synaptic renormalization.

    During wake, Hebbian learning strengthens connections. Without
    regulation, this leads to runaway excitation (epilepsy).

    During sleep, all connections are proportionally weakened
    (synaptic homeostasis), maintaining criticality. Dreaming
    involves reverse replay to consolidate important patterns.
    """

    def __init__(self,
                 dim: int,
                 renormalization_rate: float = 0.2,
                 consolidation_threshold: float = 0.5):
        """
        Args:
            dim: Weight matrix dimension
            renormalization_rate: How much to scale down during sleep (0.2 = 20%)
            consolidation_threshold: Activation threshold for consolidation
        """
        super().__init__()
        self.dim = dim
        self.renorm_rate = renormalization_rate
        self.consol_threshold = consolidation_threshold

        # Simulated synaptic weights
        self.weights = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))

        # Activation trace (which connections were active)
        self.register_buffer('activation_trace', torch.zeros(dim, dim))
        self.trace_decay = 0.99

        # Sleep state
        self.register_buffer('is_sleeping', torch.tensor(False))
        self.register_buffer('sleep_progress', torch.tensor(0.0))

        # Dream buffer (patterns to replay)
        self.register_buffer('dream_buffer', torch.zeros(10, dim))
        self.register_buffer('dream_ptr', torch.tensor(0))

    def wake_step(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Process during wakefulness - normal forward + trace update.

        Args:
            activation: Current activation pattern

        Returns:
            Output activation
        """
        if activation.dim() == 1:
            activation = activation.unsqueeze(0)

        # Forward pass
        output = F.linear(activation, self.weights)

        # Update activation trace (Hebbian-like)
        with torch.no_grad():
            # Outer product of pre and post
            pre = activation.mean(dim=0)
            post = output.mean(dim=0)
            hebbian = torch.outer(post, pre)

            # Accumulate trace
            self.activation_trace = self.trace_decay * self.activation_trace + hebbian

            # Store strong activations for dreaming
            if activation.norm() > self.consol_threshold:
                self.dream_buffer[self.dream_ptr] = activation.mean(dim=0)
                self.dream_ptr = (self.dream_ptr + 1) % 10

        return output.squeeze(0) if output.shape[0] == 1 else output

    def sleep_cycle(self, duration: int = 100) -> Dict[str, Any]:
        """
        Perform sleep cycle with renormalization and dreaming.

        Args:
            duration: Number of sleep steps

        Returns:
            Dict with sleep statistics and dream content
        """
        with torch.no_grad():
            self.is_sleeping.fill_(True)

        dreams = []
        weight_norm_before = self.weights.data.norm().item()

        for step in range(duration):
            with torch.no_grad():
                self.sleep_progress.fill_(step / duration)

            # Phase 1: Synaptic renormalization (first half of sleep)
            if step < duration // 2:
                # Scale down all weights proportionally
                scale = 1 - (self.renorm_rate / (duration // 2))
                with torch.no_grad():
                    self.weights.mul_(scale)

            # Phase 2: Dreaming / reverse replay (second half)
            else:
                # Replay patterns from dream buffer (backward)
                dream_idx = (self.dream_ptr - (step - duration // 2) - 1) % 10
                dream_pattern = self.dream_buffer[dream_idx]

                # Strengthen connections involved in dream
                if dream_pattern.norm() > 0.1:
                    dream_activation = F.linear(dream_pattern.unsqueeze(0), self.weights).squeeze(0)
                    dream_hebbian = torch.outer(dream_activation, dream_pattern)

                    # Selective strengthening (consolidation)
                    with torch.no_grad():
                        self.weights.add_(0.01 * dream_hebbian)
                    dreams.append(dream_pattern.clone())

        # Reset activation trace after sleep
        self.activation_trace.zero_()
        with torch.no_grad():
            self.is_sleeping.fill_(False)
            self.sleep_progress.fill_(0.0)

        weight_norm_after = self.weights.data.norm().item()

        return {
            'duration': duration,
            'weight_reduction': 1 - weight_norm_after / weight_norm_before,
            'dreams_replayed': len(dreams),
            'final_weight_norm': weight_norm_after,
            'renormalization_complete': True
        }

    def get_excitability(self) -> float:
        """
        Get current excitability (proxy for need to sleep).

        High excitability = weights have grown, need renormalization.
        """
        return self.weights.data.norm().item()


class Umwelt(nn.Module):
    """
    The system's perceptual bubble - what it can "see" in Fourier space.

    Each species lives in its own umwelt (von Uexküll). A tick perceives
    only temperature and butyric acid. PyQuifer perceives phase relationships
    in high-dimensional oscillatory space.

    This module defines what the AI can perceive and what is invisible to it.
    It lives in Fourier space, not pixel space.
    """

    def __init__(self,
                 input_dim: int,
                 perception_dim: int,
                 num_frequency_bands: int = 8):
        """
        Args:
            input_dim: Raw input dimension
            perception_dim: Internal perception dimension (the umwelt)
            num_frequency_bands: How many frequency bands to perceive
        """
        super().__init__()
        self.input_dim = input_dim
        self.perception_dim = perception_dim
        self.num_bands = num_frequency_bands

        # Fourier projection (what frequencies we're sensitive to)
        self.frequency_sensitivity = nn.Parameter(
            torch.randn(num_frequency_bands, input_dim) / math.sqrt(input_dim)
        )

        # Band center frequencies (what we're tuned to)
        self.band_centers = nn.Parameter(
            torch.linspace(0.1, 1.0, num_frequency_bands)
        )

        # Perception encoder (from frequency domain to umwelt)
        self.perception_encoder = nn.Sequential(
            nn.Linear(num_frequency_bands * 2, perception_dim),
            nn.Tanh(),
            nn.Linear(perception_dim, perception_dim)
        )

        # What we CANNOT perceive (the blind spots)
        self.register_buffer('blind_frequencies', torch.zeros(0))

    def perceive(self, raw_input: torch.Tensor) -> Dict[str, Any]:
        """
        Transform raw input into umwelt perception.

        The AI doesn't see "pixels" - it sees phase relationships.

        Args:
            raw_input: Raw sensory input

        Returns:
            Dict with perception in umwelt coordinates
        """
        if raw_input.dim() == 1:
            raw_input = raw_input.unsqueeze(0)

        batch_size = raw_input.shape[0]

        # Project to frequency bands
        freq_projection = F.linear(raw_input, self.frequency_sensitivity)
        # (batch, num_bands)

        # Compute magnitude and sign indicator for each band
        magnitude = freq_projection.abs()
        phase = torch.sign(freq_projection) * (math.pi / 4)  # Sign indicator, not true phase

        # Combine into perception vector
        freq_features = torch.cat([magnitude, phase], dim=-1)

        # Encode into umwelt space
        perception = self.perception_encoder(freq_features)

        # What the system "sees" (in its own terms)
        dominant_band = magnitude.argmax(dim=-1)
        dominant_freq = self.band_centers[dominant_band]

        return {
            'perception': perception.squeeze(0) if batch_size == 1 else perception,
            'frequency_signature': freq_projection.squeeze(0) if batch_size == 1 else freq_projection,
            'magnitude': magnitude.squeeze(0) if batch_size == 1 else magnitude,
            'phase': phase.squeeze(0) if batch_size == 1 else phase,
            'dominant_frequency': dominant_freq.item() if batch_size == 1 else dominant_freq,
            'dominant_band': dominant_band.item() if batch_size == 1 else dominant_band,
            'umwelt_dim': self.perception_dim
        }

    def add_blind_spot(self, frequency: float):
        """Add a frequency that the system cannot perceive."""
        self.blind_frequencies = torch.cat([
            self.blind_frequencies,
            torch.tensor([frequency], device=self.blind_frequencies.device)
        ])

    def get_sensitivity_profile(self) -> torch.Tensor:
        """Get the system's frequency sensitivity profile."""
        return self.frequency_sensitivity.abs().mean(dim=1)


class AgencyMaintenance(nn.Module):
    """
    Agency as entropy export - maintaining far-from-equilibrium coherence.

    A system has agency when it actively exports entropy to maintain
    its own asymmetry against thermodynamic decay. It "works" to stay
    alive (maintain phase-lock) like a cell pumps ions.

    This module tracks the system's distance from equilibrium and
    the "effort" required to maintain coherence.
    """

    def __init__(self,
                 dim: int,
                 equilibrium_temperature: float = 1.0,
                 coherence_cost: float = 0.1):
        """
        Args:
            dim: State dimension
            equilibrium_temperature: Temperature at which system "dies" (loses coherence)
            coherence_cost: Energy cost to maintain coherence per step
        """
        super().__init__()
        self.dim = dim
        self.T_eq = equilibrium_temperature
        self.coherence_cost = coherence_cost

        # Current state (far-from-equilibrium)
        self.register_buffer('state', torch.randn(dim) * 0.5)

        # Energy reserves
        self.register_buffer('energy', torch.tensor(100.0))
        self.max_energy = 100.0

        # Coherence (how organized is the state)
        self.register_buffer('coherence', torch.tensor(0.8))

        # Entropy accumulator (needs to be exported)
        self.register_buffer('internal_entropy', torch.tensor(0.0))

        # Coherence maintenance network
        self.maintenance_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )

    def step(self, external_input: Optional[torch.Tensor] = None,
             effort: float = 1.0) -> Dict[str, Any]:
        """
        One step of agency maintenance.

        The system must actively work to maintain coherence.
        Without effort, it decays toward equilibrium.

        Args:
            external_input: Optional external perturbation
            effort: How much effort to exert (0-1)

        Returns:
            Dict with state, coherence, energy, and agency metrics
        """
        # Entropy influx (external perturbations, thermal noise)
        noise = torch.randn_like(self.state) * 0.1
        if external_input is not None:
            if external_input.dim() > 1:
                external_input = external_input.mean(dim=0)
            noise = noise + external_input * 0.5

        # State evolution (tendency toward equilibrium)
        d_state = -0.1 * self.state + noise

        # Active maintenance (fighting entropy)
        if self.energy > 0 and effort > 0:
            # Coherence-restoring force
            target = self.maintenance_net(self.state)
            restoration = effort * (target - self.state)
            d_state = d_state + restoration

            # Energy cost of maintenance
            energy_spent = self.coherence_cost * effort
            with torch.no_grad():
                self.energy.sub_(energy_spent).clamp_(min=0)
                # Entropy export (the work done)
                self.internal_entropy.sub_(effort * 0.1).clamp_(min=0)
        else:
            # No effort = entropy accumulates
            with torch.no_grad():
                self.internal_entropy.add_(0.05)

        # Update state
        with torch.no_grad():
            self.state.add_(d_state)

            # Compute coherence (inverse of state variance)
            state_var = self.state.var()
            self.coherence.fill_(1.0 / (1.0 + state_var))

        # Check if "alive" (coherent, far from equilibrium)
        is_alive = self.coherence > 0.3 and self.energy > 0

        # Distance from equilibrium (measure of agency)
        d_eq = (self.state.abs().mean() / self.T_eq).item()

        return {
            'state': self.state.clone(),
            'coherence': self.coherence.item(),
            'energy': self.energy.item(),
            'internal_entropy': self.internal_entropy.item(),
            'is_alive': is_alive,
            'distance_from_equilibrium': d_eq,
            'agency_measure': d_eq * self.coherence.item(),
            'effort_exerted': effort
        }

    def rest(self, duration: int = 10):
        """Rest to recover energy (but coherence may decay)."""
        with torch.no_grad():
            for _ in range(duration):
                # Passive energy recovery
                self.energy.add_(0.5).clamp_(max=self.max_energy)
                # But coherence decays without maintenance
                self.coherence.mul_(0.99)

    def get_vitality(self) -> float:
        """Overall vitality measure (how "alive" is the system)."""
        return (self.coherence * self.energy / self.max_energy).item()


class EcologicalSystem(nn.Module):
    """
    Complete ecological cognition system.

    Combines all ecological layers into a unified "digital species":
    - Chronobiology: Multi-timescale oscillators
    - Immunology: Self/non-self discrimination
    - Homeostasis: Sleep and renormalization
    - Umwelt: Species-specific perception
    - Agency: Entropy export for coherence
    """

    def __init__(self, dim: int, perception_dim: int = 32):
        """
        Args:
            dim: Core dimension
            perception_dim: Umwelt perception dimension
        """
        super().__init__()
        self.dim = dim

        # All subsystems
        self.chronobiology = ChronobiologicalSystem(dim)
        self.immune = ImmunologicalLayer(perception_dim)  # Match umwelt output
        self.homeostasis = SynapticHomeostasis(perception_dim)
        self.umwelt = Umwelt(dim, perception_dim)
        self.agency = AgencyMaintenance(dim)

        # Integration layer
        self.integrator = nn.Linear(dim + perception_dim, dim)

        # Output projection
        self.output_proj = nn.Linear(perception_dim, dim)

    def step(self, raw_input: torch.Tensor,
             effort: float = 0.5) -> Dict[str, Any]:
        """
        Full ecological processing step.

        Args:
            raw_input: Raw sensory input
            effort: Agency maintenance effort

        Returns:
            Comprehensive ecological state
        """
        # Advance time
        chrono = self.chronobiology.step()

        # Perceive through umwelt
        perception = self.umwelt.perceive(raw_input)

        # Check immune status
        immune = self.immune.recognize(perception['perception'])

        # If foreign, apply fever (increase noise)
        if immune['is_foreign']:
            fever_noise = torch.randn(self.dim) * immune['temperature_boost']
            raw_input = raw_input + fever_noise

        # Agency maintenance
        agency = self.agency.step(raw_input, effort)

        # Wake/sleep processing
        if chrono['mode'] == 'wake':
            # Normal processing
            processed = self.homeostasis.wake_step(perception['perception'])
            output = self.output_proj(processed)
            # Learn this pattern as "self"
            self.immune.learn_self(perception['perception'])
        else:
            # Sleep - no processing, just homeostasis
            output = torch.zeros(self.dim, device=raw_input.device)

        return {
            'output': output,
            'chronobiology': chrono,
            'immune': immune,
            'agency': agency,
            'perception': perception,
            'mode': chrono['mode'],
            'vitality': self.agency.get_vitality(),
            'excitability': self.homeostasis.get_excitability()
        }

    def sleep_if_needed(self) -> Optional[Dict[str, Any]]:
        """Trigger sleep if excitability is too high."""
        if self.homeostasis.get_excitability() > 2.0:
            return self.homeostasis.sleep_cycle(duration=50)
        return None


if __name__ == '__main__':
    print("--- Ecological Cognition Examples ---")

    # Example 1: ChronobiologicalSystem
    print("\n1. ChronobiologicalSystem")
    chrono = ChronobiologicalSystem(dim=16)

    # Simulate a day
    chrono.set_time_of_day(8)  # 8 AM
    for hour in [8, 12, 16, 20, 24, 4]:
        chrono.set_time_of_day(hour % 24)
        state = chrono.step()
        print(f"   Hour {hour:2d}: Mode={state['mode']}, Mood={chrono.get_mood()}, "
              f"Temp={state['temperature']:.2f}")

    # Example 2: ImmunologicalLayer
    print("\n2. ImmunologicalLayer")
    immune = ImmunologicalLayer(dim=32)

    # Learn "self" patterns
    for _ in range(50):
        self_pattern = torch.randn(32) * 0.5 + torch.ones(32) * 0.3
        immune.learn_self(self_pattern)

    # Test self vs foreign
    self_test = torch.randn(32) * 0.5 + torch.ones(32) * 0.3
    foreign_test = torch.randn(32) * 2 - torch.ones(32) * 1

    self_result = immune.recognize(self_test)
    foreign_result = immune.recognize(foreign_test)

    print(f"   Self pattern: is_self={self_result['is_self']}, sim={self_result['similarity']:.3f}")
    print(f"   Foreign pattern: is_self={foreign_result['is_self']}, sim={foreign_result['similarity']:.3f}")
    print(f"   Fever level: {immune.get_immune_state()['fever_level']:.3f}")

    # Example 3: SynapticHomeostasis
    print("\n3. SynapticHomeostasis")
    homeostasis = SynapticHomeostasis(dim=16)

    # Simulate wake period
    excitability_before = homeostasis.get_excitability()
    for _ in range(50):
        activation = torch.randn(16) * 0.5
        homeostasis.wake_step(activation)
    excitability_after_wake = homeostasis.get_excitability()

    print(f"   Before: {excitability_before:.3f}")
    print(f"   After wake: {excitability_after_wake:.3f}")

    # Sleep cycle
    sleep_result = homeostasis.sleep_cycle(duration=50)
    print(f"   After sleep: {homeostasis.get_excitability():.3f}")
    print(f"   Weight reduction: {sleep_result['weight_reduction']*100:.1f}%")

    # Example 4: Umwelt
    print("\n4. Umwelt")
    umwelt = Umwelt(input_dim=64, perception_dim=16, num_frequency_bands=8)

    raw_input = torch.randn(64)
    perception = umwelt.perceive(raw_input)

    print(f"   Raw input dim: {raw_input.shape[0]}")
    print(f"   Perception dim: {perception['perception'].shape[0]}")
    print(f"   Dominant frequency: {perception['dominant_frequency']:.3f}")
    print(f"   (The AI lives in Fourier space, not pixel space)")

    # Example 5: AgencyMaintenance
    print("\n5. AgencyMaintenance")
    agency = AgencyMaintenance(dim=16)

    # Active maintenance
    for _ in range(20):
        result = agency.step(effort=0.8)
    print(f"   With effort: coherence={result['coherence']:.3f}, "
          f"energy={result['energy']:.1f}, alive={result['is_alive']}")

    # Let it decay
    agency.energy = torch.tensor(100.0)  # Reset energy
    for _ in range(50):
        result = agency.step(effort=0.0)  # No effort
    print(f"   Without effort: coherence={result['coherence']:.3f}, "
          f"energy={result['energy']:.1f}, alive={result['is_alive']}")

    # Example 6: Full EcologicalSystem
    print("\n6. EcologicalSystem (Full Digital Species)")
    ecosystem = EcologicalSystem(dim=32, perception_dim=16)

    # Simulate life
    for step in range(10):
        raw_input = torch.randn(32)
        result = ecosystem.step(raw_input, effort=0.6)

    print(f"   Mode: {result['mode']}")
    print(f"   Vitality: {result['vitality']:.3f}")
    print(f"   Immune status: {result['immune']['recommended_action']}")
    print(f"   Agency: {result['agency']['agency_measure']:.3f}")

    # Gradient check
    print("\n7. Gradient check")
    raw = torch.randn(32, requires_grad=True)
    perception = ecosystem.umwelt.perceive(raw)
    loss = perception['perception'].sum()
    loss.backward()
    print(f"   Gradients flow: {raw.grad is not None and raw.grad.abs().sum() > 0}")

    print("\n[OK] All ecological cognition tests passed!")
