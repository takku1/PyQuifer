"""
PyQuifer Bridge: Clean API for plugging PyQuifer into any LLM.

This is the external-facing interface. One class, three methods:
  1. bridge.step(input) -> ModulationState
  2. bridge.modulate_logits(logits, state) -> modified_logits
  3. bridge.modulate_hidden(hidden, state) -> modified_hidden

The coupling equation (from design doc):
  Modified_Hidden = Original + A * sin(wt + phi) * Trait_Vector

Where:
  A = amplitude (coherence * neuromodulator gain)
  w, phi = oscillator frequency and phase
  Trait_Vector = personality direction in embedding space

IMPORTANT DESIGN CONSTRAINT:
  LLM -> Oscillator gradient flow is SEVERED by design.
  Oscillators evolve through their own physics (Kuramoto, PLL),
  NOT through backprop from the language model. The oscillators
  are the "soul" — the LLM is the "body."

Usage:
    from pyquifer.bridge import PyQuiferBridge

    bridge = PyQuiferBridge.default()
    for tokens in stream:
        state = bridge.step(tokens.mean(dim=-1))
        logits = model(tokens)
        logits = bridge.modulate_logits(logits, state)

References:
- PyQuifer design doc: Modified_Hidden equation
- Friston (2010): Free-energy principle
- Edelman (1987): Neural Darwinism
"""

import torch
import torch.nn as nn
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class ModulationState:
    """
    Everything an LLM needs from one cognitive tick.

    This is the output contract — any LLM integration reads from this.
    """
    # === Generation parameters ===
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    top_p: float = 0.9

    # === Personality ===
    dominant_facet: int = 0
    facet_weights: List[float] = field(default_factory=lambda: [1.0])
    personality_stability: float = 0.5

    # === Attention ===
    attention_bias: Optional[torch.Tensor] = None  # (dim,) precision weights

    # === Cognitive mode ===
    processing_mode: str = "balanced"  # "perception", "imagination", "balanced"
    coherence: float = 0.5
    motivation: float = 0.5

    # === Internal state (for monitoring, not generation) ===
    free_energy: float = 0.0
    criticality_distance: float = 0.5
    identity_strength: float = 0.0
    tick: int = 0

    # === Oscillator state (for hidden state modulation) ===
    phases: Optional[torch.Tensor] = None  # (num_oscillators,)
    neuromodulator_levels: Optional[torch.Tensor] = None  # [DA, 5HT, NE, ACh, Cortisol]

    # === Latency ===
    step_latency_ms: float = 0.0


class PyQuiferBridge(nn.Module):
    """
    Bridge between PyQuifer's cognitive engine and any LLM.

    Three-level integration:
      Level 1 (easy): Temperature/top_p modulation only
      Level 2 (medium): + logit biasing from personality/motivation
      Level 3 (deep): + hidden state injection via coupling equation

    Example (Level 1 - HuggingFace):
        bridge = PyQuiferBridge.default()
        for step in range(100):
            state = bridge.step(torch.randn(32))
            output = model.generate(
                input_ids,
                temperature=state.temperature,
                top_p=state.top_p,
                repetition_penalty=state.repetition_penalty,
            )
    """

    def __init__(self, config=None):
        """
        Args:
            config: CycleConfig for the cognitive engine.
                    Use PyQuiferBridge.default() or .small() for presets.
        """
        super().__init__()
        from pyquifer.integration import CognitiveCycle, CycleConfig
        self.config = config or CycleConfig.default()
        self.cycle = CognitiveCycle(self.config)

        # Trait vectors for hidden state modulation (Level 3)
        # Each facet gets a unit-norm direction in embedding space.
        # With coherence~0.5 and gain~1.0, perturbation is ~2-5% of
        # hidden state norm (visible in generation but not destructive).
        # These are learnable IF you want to train the bridge, but
        # gradient does NOT flow back to oscillators.
        _raw_traits = torch.randn(self.config.num_populations, self.config.state_dim)
        _raw_traits = _raw_traits / _raw_traits.norm(dim=-1, keepdim=True)
        self.trait_vectors = nn.Parameter(_raw_traits)
        # Learnable modulation gain (allows fine-tuning perturbation strength)
        self.modulation_gain = nn.Parameter(torch.tensor(1.0))

        # Latency tracking
        self.register_buffer('_latency_ema', torch.tensor(0.0))
        self.register_buffer('_latency_count', torch.tensor(0))

    @staticmethod
    def default():
        """Standard bridge with default cognitive engine."""
        from pyquifer.integration import CycleConfig
        return PyQuiferBridge(CycleConfig.default())

    @staticmethod
    def small():
        """Lightweight bridge for testing or low-resource environments."""
        from pyquifer.integration import CycleConfig
        return PyQuiferBridge(CycleConfig.small())

    def step(self,
             sensory_input: torch.Tensor,
             reward: float = 0.0,
             sleep_signal: float = 0.0) -> ModulationState:
        """
        Run one cognitive tick. Call this once per generation step.

        Args:
            sensory_input: Any signal the LLM provides (state_dim,).
                          Could be: mean hidden state, token embedding,
                          attention pattern summary, or synthetic signal.
            reward: Scalar reward (e.g., from RLHF, user feedback, task success).
            sleep_signal: 0.0=awake, 1.0=consolidating (e.g., between conversations).

        Returns:
            ModulationState with all parameters the LLM needs.
        """
        t0 = time.perf_counter()

        # Ensure correct dimension
        if sensory_input.shape[-1] != self.config.state_dim:
            # Project to state_dim if needed
            sensory_input = self._project_input(sensory_input)

        # Run cognitive cycle
        # NOTE: Do NOT wrap in torch.no_grad() — the HPC module uses
        # autograd internally for generative model learning (gen_lr).
        # The cycle manages its own no_grad blocks for buffer updates.
        result = self.cycle.tick(sensory_input, reward=reward, sleep_signal=sleep_signal)

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000

        # Update latency EMA
        with torch.no_grad():
            alpha = 0.1 if self._latency_count > 0 else 1.0
            self._latency_ema.mul_(1 - alpha).add_(alpha * latency_ms)
            self._latency_count.add_(1)

        m = result['modulation']
        c = result['consciousness']
        s = result['self_state']
        d = result['diagnostics']

        # Map cognitive state to generation parameters
        temperature = m['temperature']
        repetition_penalty = self._compute_repetition_penalty(c['coherence'], m['motivation'])
        top_p = self._compute_top_p(c['coherence'], c['criticality_distance'])

        return ModulationState(
            # Generation params
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            # Personality
            dominant_facet=m['dominant_state'],
            facet_weights=m['personality_blend']['facet_weights'],
            personality_stability=m['personality_blend']['stability'],
            # Attention
            attention_bias=m['attention_bias'],
            # Cognitive mode
            processing_mode=m['processing_mode'],
            coherence=m['coherence'],
            motivation=m['motivation'],
            # Internal state
            free_energy=c['free_energy'],
            criticality_distance=c['criticality_distance'],
            identity_strength=s['identity_strength'],
            tick=d['tick'],
            # Oscillator state
            phases=d['phases'],
            neuromodulator_levels=d['neuromodulator_levels'],
            # Latency
            step_latency_ms=latency_ms,
        )

    def modulate_logits(self,
                        logits: torch.Tensor,
                        state: ModulationState) -> torch.Tensor:
        """
        Level 2: Modify logits based on cognitive state.

        Applies temperature scaling, coherence-based sharpening, and
        motivation-based diversity. These produce visible changes in
        token distribution (1-10% logit shifts).

        Args:
            logits: Raw logits from LLM (batch, vocab_size)
            state: ModulationState from step()

        Returns:
            Modified logits (same shape)
        """
        # Temperature scaling (standard)
        modified = logits / max(state.temperature, 0.01)

        # Coherence sharpening: high coherence (focused) → sharpen distribution
        # by scaling logits away from mean. Low coherence → flatten.
        # Effect: 5-15% change at extreme coherence values.
        if state.coherence > 0.1:
            logit_mean = modified.mean(dim=-1, keepdim=True)
            # coherence 0.5 → scale 1.0 (neutral), 0.8 → 1.06, 0.2 → 0.94
            sharpening = 1.0 + (state.coherence - 0.5) * 0.2
            modified = logit_mean + (modified - logit_mean) * sharpening

        # Motivation diversity: higher motivation → more exploratory
        # (flatten the distribution slightly to increase sampling entropy)
        if state.motivation > 0.3:
            flatten_factor = 1.0 + (state.motivation - 0.3) * 0.15
            logit_mean = modified.mean(dim=-1, keepdim=True)
            modified = logit_mean + (modified - logit_mean) / flatten_factor

        return modified

    def modulate_hidden(self,
                        hidden_states: torch.Tensor,
                        state: ModulationState,
                        layer_idx: int = -1) -> torch.Tensor:
        """
        Level 3: Inject oscillator state into transformer hidden states.

        The coupling equation:
          Modified = Original + A * sin(phase) * Trait_Vector

        Where:
          A = coherence * neuromodulator_gain (detached from LLM grad)
          phase = oscillator phases (from Kuramoto dynamics)
          Trait_Vector = personality direction in embedding space

        IMPORTANT: Gradient does NOT flow back to oscillators.
        The oscillators evolve through their own physics.

        Args:
            hidden_states: Transformer hidden states (batch, seq_len, hidden_dim)
            state: ModulationState from step()
            layer_idx: Which transformer layer (-1 = last)

        Returns:
            Modified hidden states (same shape)
        """
        if state.phases is None:
            return hidden_states

        batch, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        # Amplitude: coherence * neuromodulator gain * learnable gain
        # Detached — oscillator state is read-only from LLM's perspective
        coherence = state.coherence
        if state.neuromodulator_levels is not None:
            nm_gain = state.neuromodulator_levels.mean()
        else:
            nm_gain = torch.tensor(0.5, device=device)
        amplitude = coherence * nm_gain * self.modulation_gain

        # Phase contribution: sin(phases) gives per-oscillator modulation
        # Instead of averaging (which cancels to ~0), tile phases across
        # hidden_dim so different dimensions get different phase signals.
        # This is the key coupling equation from the design doc:
        #   Modified = Original + A * sin(wt + phi) * Trait_Vector
        phases = state.phases.detach().to(device)
        phase_signal = torch.sin(phases)  # (num_oscillators,)

        # Tile phase signal to match hidden_dim
        num_osc = phase_signal.shape[0]
        if num_osc < hidden_dim:
            repeats = hidden_dim // num_osc + 1
            phase_tiled = phase_signal.repeat(repeats)[:hidden_dim]
        else:
            phase_tiled = phase_signal[:hidden_dim]

        # Weighted trait vector: blend personality facets by their weights
        weights = torch.tensor(state.facet_weights, device=device, dtype=torch.float32)
        trait_vecs = self.trait_vectors.to(device)
        blended_trait = (weights.unsqueeze(-1) * trait_vecs).sum(dim=0)  # (state_dim,)

        # Project trait vector to hidden_dim if needed
        if blended_trait.shape[0] != hidden_dim:
            if blended_trait.shape[0] < hidden_dim:
                repeats = hidden_dim // blended_trait.shape[0] + 1
                blended_trait = blended_trait.repeat(repeats)[:hidden_dim]
            else:
                blended_trait = blended_trait[:hidden_dim]

        # The coupling equation (per-dimension phase modulation):
        #   perturbation_d = A * sin(phase_d) * trait_d
        # This gives structured perturbation where each hidden dimension
        # is modulated by a different oscillator's phase.
        perturbation = amplitude * phase_tiled * blended_trait
        modified = hidden_states + perturbation.unsqueeze(0).unsqueeze(0)

        return modified

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        return {
            'mean_latency_ms': self._latency_ema.item(),
            'num_steps': self._latency_count.item(),
        }

    def get_state_summary(self) -> Dict[str, Any]:
        """Compact cognitive state summary."""
        return self.cycle.get_state_summary()

    def reset(self):
        """Reset cognitive engine state."""
        self.cycle.reset()
        self._latency_ema.zero_()
        self._latency_count.zero_()

    def _project_input(self, x: torch.Tensor) -> torch.Tensor:
        """Project arbitrary input to state_dim."""
        target = self.config.state_dim
        if x.shape[-1] > target:
            return x[..., :target]
        elif x.shape[-1] < target:
            pad = torch.zeros(*x.shape[:-1], target - x.shape[-1], device=x.device)
            return torch.cat([x, pad], dim=-1)
        return x

    def _compute_repetition_penalty(self, coherence: float, motivation: float) -> float:
        """
        Map cognitive state to repetition penalty.

        High coherence = focused, less repetitive -> higher penalty
        High motivation = exploring new ideas -> higher penalty
        """
        base = 1.0
        coherence_bonus = coherence * 0.2  # Up to 0.2 extra
        motivation_bonus = max(0, motivation - 0.5) * 0.1  # Up to 0.1 extra
        return min(1.5, base + coherence_bonus + motivation_bonus)

    def _compute_top_p(self, coherence: float, criticality_distance: float) -> float:
        """
        Map cognitive state to nucleus sampling threshold.

        Near criticality = more creative = higher top_p
        High coherence = more focused = lower top_p
        """
        base = 0.9
        coherence_mod = -coherence * 0.1  # High coherence tightens
        crit_mod = max(0, 1.0 - criticality_distance) * 0.05  # Near critical loosens
        return max(0.5, min(1.0, base + coherence_mod + crit_mod))


if __name__ == '__main__':
    import math

    print("=== PyQuiferBridge Demo ===\n")

    # Create bridge
    bridge = PyQuiferBridge.small()
    total_params = sum(p.numel() for p in bridge.parameters())
    total_buffers = sum(b.numel() for b in bridge.buffers())
    print(f"Parameters: {total_params:,}")
    print(f"Buffers: {total_buffers:,}")

    # --- Level 1: Generation parameter modulation ---
    print("\n--- Level 1: Generation Parameters ---")
    torch.manual_seed(42)
    for i in range(20):
        sensory = torch.randn(bridge.config.state_dim) * 0.5
        reward = math.sin(i * 0.2)
        state = bridge.step(sensory, reward=reward)

        if i % 5 == 0:
            print(f"  Tick {state.tick:3d}: "
                  f"temp={state.temperature:.2f} "
                  f"top_p={state.top_p:.2f} "
                  f"rep_pen={state.repetition_penalty:.2f} "
                  f"mode={state.processing_mode:>11s} "
                  f"latency={state.step_latency_ms:.1f}ms")

    # --- Level 2: Logit modulation ---
    print("\n--- Level 2: Logit Modulation ---")
    fake_logits = torch.randn(1, 50000)  # (batch=1, vocab=50k)
    modified = bridge.modulate_logits(fake_logits, state)
    print(f"  Original logit range: [{fake_logits.min():.2f}, {fake_logits.max():.2f}]")
    print(f"  Modified logit range: [{modified.min():.2f}, {modified.max():.2f}]")

    # --- Level 3: Hidden state injection ---
    print("\n--- Level 3: Hidden State Injection ---")
    fake_hidden = torch.randn(1, 128, bridge.config.state_dim)  # (batch, seq, dim)
    modified_hidden = bridge.modulate_hidden(fake_hidden, state)
    diff = (modified_hidden - fake_hidden).norm().item()
    print(f"  Hidden state perturbation norm: {diff:.6f}")
    print(f"  Relative perturbation: {diff / fake_hidden.norm().item():.6f}")

    # --- Latency ---
    print(f"\n--- Latency ---")
    stats = bridge.get_latency_stats()
    print(f"  Mean latency: {stats['mean_latency_ms']:.2f}ms")
    print(f"  Steps: {stats['num_steps']}")

    # --- HuggingFace integration example ---
    print("\n--- HuggingFace Example (pseudocode) ---")
    print("""
    from pyquifer.bridge import PyQuiferBridge

    bridge = PyQuiferBridge.default()

    # In your generation loop:
    state = bridge.step(last_hidden.mean(dim=1).squeeze())
    output = model.generate(
        input_ids,
        temperature=state.temperature,
        top_p=state.top_p,
        repetition_penalty=state.repetition_penalty,
    )
    """)

    print("\n[OK] PyQuiferBridge demo passed!")
