"""
Voice Dynamics - Oscillator patterns for speech prosody.

Maps neural oscillations to voice characteristics based on
neuroscience of speech production.

Key concepts:
- Theta oscillations (4-8 Hz) track syllabic rhythm
- Gamma oscillations modulate clarity/articulation
- Neuromodulators color emotional expression
- Phase coupling coordinates speech timing

References:
- Giraud & Poeppel (2012): Cortical oscillations and speech processing
- Keitel et al. (2018): Speech rhythms and oscillatory tracking
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from .neuromodulation import NeuromodulatorState


@dataclass
class SpeechRhythm:
    """Speech rhythm parameters derived from oscillators."""
    syllable_rate: float      # Hz (typically 4-6 Hz, theta range)
    stress_pattern: float     # Emphasis strength (0-1)
    pause_tendency: float     # Likelihood of pauses (0-1)
    breathing_sync: float     # Coordination with breath (-1 to 1)
    phrase_boundary: float    # Probability of phrase end (0-1)


@dataclass
class VoiceEffects:
    """Voice modulation effects from neuromodulators."""
    pitch_range: float        # Pitch variation multiplier
    energy: float             # Volume/energy level
    warmth: float             # Breathiness/warmth
    rate_modifier: float      # Speech rate multiplier
    attack: float             # Consonant crispness
    urgency: float            # Perceived urgency
    emphasis: float           # Word stress strength
    clarity: float            # Articulation clarity


class SpeechOscillator(nn.Module):
    """
    Specialized oscillator for speech timing.

    Based on research showing theta oscillations (4-8 Hz)
    track syllabic rhythm in speech production.

    The oscillator couples to main band oscillators to
    maintain coherent speech timing across the system.
    """

    def __init__(
        self,
        base_rate: float = 5.0,  # Syllables per second (theta range)
        dt: float = 0.01
    ):
        """
        Args:
            base_rate: Base syllable rate in Hz
            dt: Time step for integration
        """
        super().__init__()
        self.base_rate = base_rate
        self.dt = dt

        # Phase for syllable timing
        self.register_buffer('phase', torch.zeros(1))

        # Coupling strengths to main oscillator bands
        self.theta_coupling = nn.Parameter(torch.tensor(0.3))
        self.gamma_coupling = nn.Parameter(torch.tensor(0.1))
        self.alpha_coupling = nn.Parameter(torch.tensor(0.15))

        # Phrase boundary detector
        self.phrase_accumulator = 0.0
        self.phrase_threshold = 2 * math.pi * 3  # ~3 syllables per phrase

    def step(
        self,
        theta_phase: Optional[torch.Tensor] = None,
        gamma_phase: Optional[torch.Tensor] = None,
        alpha_phase: Optional[torch.Tensor] = None
    ) -> SpeechRhythm:
        """
        Compute speech rhythm from oscillator state.

        Args:
            theta_phase: Theta band phase (receptive state)
            gamma_phase: Gamma band phase (articulation)
            alpha_phase: Alpha band phase (reflection)

        Returns:
            SpeechRhythm with timing parameters
        """
        # Base phase evolution (as tensor for consistent handling)
        d_phase = torch.tensor(2 * math.pi * self.base_rate * self.dt)

        # Coupling to main bands (if provided)
        if theta_phase is not None:
            if theta_phase.dim() > 0:
                theta_phase = theta_phase.mean()
            d_phase = d_phase + self.theta_coupling * torch.sin(theta_phase - self.phase)

        if gamma_phase is not None:
            if gamma_phase.dim() > 0:
                gamma_phase = gamma_phase.mean()
            d_phase = d_phase + self.gamma_coupling * torch.sin(gamma_phase - self.phase)

        if alpha_phase is not None:
            if alpha_phase.dim() > 0:
                alpha_phase = alpha_phase.mean()
            d_phase = d_phase + self.alpha_coupling * torch.sin(alpha_phase - self.phase)

        # Update phase
        with torch.no_grad():
            self.phase.add_(d_phase).remainder_(2 * math.pi)

        # Track phrase accumulator
        self.phrase_accumulator += d_phase.item()

        # Derive rhythm parameters from phase
        phase_val = self.phase.item()

        syllable_rate = self.base_rate * (1 + 0.2 * math.sin(phase_val))
        stress_pattern = 0.5 + 0.3 * math.cos(phase_val * 2)
        pause_tendency = max(0, math.sin(phase_val - math.pi / 4))
        breathing_sync = math.cos(phase_val / 4)

        # Phrase boundary probability
        if self.phrase_accumulator > self.phrase_threshold:
            phrase_boundary = 0.8
            self.phrase_accumulator = 0.0
        else:
            phrase_boundary = 0.1 * pause_tendency

        return SpeechRhythm(
            syllable_rate=syllable_rate,
            stress_pattern=stress_pattern,
            pause_tendency=pause_tendency,
            breathing_sync=breathing_sync,
            phrase_boundary=phrase_boundary
        )

    def reset(self):
        """Reset oscillator state."""
        self.phase.zero_()
        self.phrase_accumulator = 0.0


class VoiceNeuromodulation:
    """
    Maps neuromodulator state to voice modulation effects.

    Based on how neurotransmitters affect speech production:
    - Dopamine: Enthusiasm, pitch variation
    - Serotonin: Calmness, warmth
    - Norepinephrine: Alertness, crispness
    - Acetylcholine: Attention, emphasis
    """

    @staticmethod
    def get_voice_effects(state: NeuromodulatorState) -> VoiceEffects:
        """
        Map neuromodulator state to voice effects.

        Args:
            state: Current neuromodulator levels

        Returns:
            VoiceEffects for prosody modulation
        """
        return VoiceEffects(
            # Dopamine: reward/enthusiasm → pitch variation, energy
            pitch_range=1.0 + state.dopamine * 0.3,
            energy=0.8 + state.dopamine * 0.4,

            # Serotonin: contentment → warmth, slower pace
            warmth=state.serotonin,
            rate_modifier=1.0 - state.serotonin * 0.15,

            # Norepinephrine: alertness → crispness, urgency
            attack=0.5 + state.norepinephrine * 0.5,
            urgency=state.norepinephrine,

            # Acetylcholine: attention → emphasis, clarity
            emphasis=0.5 + state.acetylcholine * 0.3,
            clarity=0.7 + state.acetylcholine * 0.3
        )


class ProsodyModulator(nn.Module):
    """
    Modulates TTS prosody based on consciousness state.

    Integrates:
    - Oscillator phases → timing
    - Neuromodulators → emotional color
    - Coherence → confidence

    Outputs a style vector suitable for TTS conditioning.
    """

    def __init__(self, style_dim: int = 256):
        """
        Args:
            style_dim: Dimension of output style vector
        """
        super().__init__()
        self.style_dim = style_dim

        # Input projections
        # 4 bands × 4 features (phase, coherence, amplitude, frequency)
        self.oscillator_proj = nn.Linear(16, 64)
        # 5 neuromodulators
        self.neuromod_proj = nn.Linear(5, 32)
        # Global coherence
        self.coherence_proj = nn.Linear(1, 16)

        # Style generation network
        # Input: osc(64) + neuro(32) + coherence(16) + rhythm(16) = 128
        self.style_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, style_dim),
            nn.Tanh()
        )

        # Speech rhythm integration
        self.rhythm_proj = nn.Linear(5, 16)

    def forward(
        self,
        band_phases: torch.Tensor,      # [4] theta, alpha, beta, gamma
        band_coherences: torch.Tensor,  # [4]
        band_amplitudes: torch.Tensor,  # [4]
        band_frequencies: torch.Tensor, # [4]
        neuromodulators: torch.Tensor,  # [5] DA, 5-HT, NE, ACh, Cortisol
        global_coherence: torch.Tensor, # [1]
        speech_rhythm: Optional[SpeechRhythm] = None
    ) -> torch.Tensor:
        """
        Generate style vector from oscillator state.

        Args:
            band_phases: Phase of each frequency band
            band_coherences: Coherence of each band
            band_amplitudes: Amplitude of each band
            band_frequencies: Frequency of each band
            neuromodulators: Neuromodulator levels
            global_coherence: Overall system coherence
            speech_rhythm: Optional speech rhythm state

        Returns:
            Style vector for TTS conditioning
        """
        # Stack oscillator features
        osc_features = torch.cat([
            band_phases, band_coherences, band_amplitudes, band_frequencies
        ])

        # Project each feature group
        osc_emb = self.oscillator_proj(osc_features)
        neuro_emb = self.neuromod_proj(neuromodulators)
        coh_emb = self.coherence_proj(global_coherence.unsqueeze(0) if global_coherence.dim() == 0 else global_coherence)

        # Combine base features
        combined = torch.cat([osc_emb, neuro_emb, coh_emb])

        # Add speech rhythm (zero if not provided)
        if speech_rhythm is not None:
            rhythm_tensor = torch.tensor([
                speech_rhythm.syllable_rate / 10.0,  # Normalize
                speech_rhythm.stress_pattern,
                speech_rhythm.pause_tendency,
                speech_rhythm.breathing_sync,
                speech_rhythm.phrase_boundary
            ], device=combined.device)
        else:
            rhythm_tensor = torch.zeros(5, device=combined.device)
        rhythm_emb = self.rhythm_proj(rhythm_tensor)
        combined = torch.cat([combined, rhythm_emb])

        # Generate style vector
        style = self.style_net(combined)

        return style

    def get_prosody_params(
        self,
        style: torch.Tensor
    ) -> Dict[str, float]:
        """
        Extract interpretable prosody parameters from style vector.

        Args:
            style: Style vector from forward()

        Returns:
            Dictionary of prosody parameters
        """
        # First 8 dimensions encode specific prosody features
        return {
            'pitch_shift': style[0].item() * 6,  # -6 to +6 semitones
            'pitch_variation': 0.5 + style[1].item() * 0.5,
            'rate': 0.85 + style[2].item() * 0.3,  # 0.7 to 1.3
            'energy': 0.85 + style[3].item() * 0.3,
            'warmth': 0.5 + style[4].item() * 0.5,
            'emphasis': 0.5 + style[5].item() * 0.3,
            'clarity': 0.7 + style[6].item() * 0.3,
            'breathiness': max(0, style[7].item()),
        }


class VoiceDynamicsSystem(nn.Module):
    """
    Complete voice dynamics system integrating all components.

    Combines:
    - SpeechOscillator for timing
    - VoiceNeuromodulation for emotional color
    - ProsodyModulator for style vector generation
    """

    def __init__(self, style_dim: int = 256):
        super().__init__()

        self.speech_osc = SpeechOscillator()
        self.prosody_mod = ProsodyModulator(style_dim)

    def step(
        self,
        theta_phase: Optional[torch.Tensor] = None,
        alpha_phase: Optional[torch.Tensor] = None,
        gamma_phase: Optional[torch.Tensor] = None,
        neuromodulator_state: Optional[NeuromodulatorState] = None,
        global_coherence: float = 0.5
    ) -> Tuple[torch.Tensor, SpeechRhythm, VoiceEffects]:
        """
        Update voice dynamics and generate style.

        Args:
            theta_phase: Theta oscillator phase
            alpha_phase: Alpha oscillator phase
            gamma_phase: Gamma oscillator phase
            neuromodulator_state: Current neuromodulator levels
            global_coherence: System coherence

        Returns:
            (style_vector, speech_rhythm, voice_effects)
        """
        # Get speech rhythm from oscillator
        rhythm = self.speech_osc.step(theta_phase, gamma_phase, alpha_phase)

        # Get voice effects from neuromodulators
        if neuromodulator_state is not None:
            effects = VoiceNeuromodulation.get_voice_effects(neuromodulator_state)
            neuro_tensor = torch.tensor([
                neuromodulator_state.dopamine,
                neuromodulator_state.serotonin,
                neuromodulator_state.norepinephrine,
                neuromodulator_state.acetylcholine,
                neuromodulator_state.cortisol
            ])
        else:
            effects = VoiceEffects(
                pitch_range=1.0, energy=1.0, warmth=0.5,
                rate_modifier=1.0, attack=0.5, urgency=0.3,
                emphasis=0.5, clarity=0.7
            )
            neuro_tensor = torch.ones(5) * 0.5

        # Generate style vector
        # Create dummy band tensors if not provided
        dummy = torch.zeros(4)
        band_phases = torch.stack([
            theta_phase.mean() if theta_phase is not None else torch.tensor(0.0),
            alpha_phase.mean() if alpha_phase is not None else torch.tensor(0.0),
            torch.tensor(0.0),  # beta
            gamma_phase.mean() if gamma_phase is not None else torch.tensor(0.0),
        ])

        style = self.prosody_mod(
            band_phases=band_phases,
            band_coherences=dummy + 0.5,
            band_amplitudes=dummy + 0.5,
            band_frequencies=torch.tensor([6.0, 10.0, 18.0, 40.0]),
            neuromodulators=neuro_tensor,
            global_coherence=torch.tensor(global_coherence),
            speech_rhythm=rhythm
        )

        return style, rhythm, effects

    def reset(self):
        """Reset all internal state."""
        self.speech_osc.reset()


if __name__ == '__main__':
    print("--- Voice Dynamics Examples ---")

    # Example 1: Speech Oscillator
    print("\n1. Speech Oscillator")
    osc = SpeechOscillator(base_rate=5.0)

    for i in range(20):
        rhythm = osc.step(
            theta_phase=torch.tensor(i * 0.3),
            gamma_phase=torch.tensor(i * 0.8)
        )
        if i % 5 == 0:
            print(f"   Step {i}: rate={rhythm.syllable_rate:.2f} Hz, "
                  f"stress={rhythm.stress_pattern:.2f}, "
                  f"phrase_end={rhythm.phrase_boundary:.2f}")

    # Example 2: Voice Neuromodulation
    print("\n2. Voice Neuromodulation")
    nm_state = NeuromodulatorState(
        dopamine=0.8,
        serotonin=0.4,
        norepinephrine=0.6,
        acetylcholine=0.7,
        cortisol=0.2
    )
    effects = VoiceNeuromodulation.get_voice_effects(nm_state)
    print(f"   Pitch range: {effects.pitch_range:.2f}")
    print(f"   Energy: {effects.energy:.2f}")
    print(f"   Warmth: {effects.warmth:.2f}")
    print(f"   Clarity: {effects.clarity:.2f}")

    # Example 3: Prosody Modulator
    print("\n3. Prosody Modulator")
    mod = ProsodyModulator(style_dim=256)

    style = mod(
        band_phases=torch.tensor([0.5, 1.0, 1.5, 2.0]),
        band_coherences=torch.tensor([0.6, 0.7, 0.5, 0.8]),
        band_amplitudes=torch.tensor([0.5, 0.6, 0.4, 0.7]),
        band_frequencies=torch.tensor([6.0, 10.0, 18.0, 40.0]),
        neuromodulators=torch.tensor([0.8, 0.4, 0.6, 0.7, 0.2]),
        global_coherence=torch.tensor(0.7)
    )
    print(f"   Style vector shape: {style.shape}")
    params = mod.get_prosody_params(style)
    print(f"   Pitch shift: {params['pitch_shift']:.2f} semitones")
    print(f"   Rate: {params['rate']:.2f}")
    print(f"   Warmth: {params['warmth']:.2f}")

    # Example 4: Full Voice Dynamics System
    print("\n4. Voice Dynamics System")
    system = VoiceDynamicsSystem(style_dim=256)

    style, rhythm, effects = system.step(
        theta_phase=torch.tensor([0.5, 0.6, 0.7, 0.8]),
        alpha_phase=torch.tensor([1.0, 1.1]),
        gamma_phase=torch.tensor([2.0, 2.1, 2.2, 2.3]),
        neuromodulator_state=nm_state,
        global_coherence=0.75
    )
    print(f"   Style shape: {style.shape}")
    print(f"   Rhythm: {rhythm.syllable_rate:.2f} Hz")
    print(f"   Effects: pitch_range={effects.pitch_range:.2f}")

    print("\n[OK] All voice dynamics tests passed!")
