"""
PyQuifer Bridge: Clean API for plugging PyQuifer into any LLM.

This is the **canonical integration point** for Mizuki and any other
LLM application.  All new integrations should use this module.

One class, three methods:
  1. bridge.step(input) -> ModulationState
  2. bridge.modulate_logits(logits, state) -> modified_logits
  3. bridge.modulate_hidden(hidden, state) -> modified_hidden

Canonical Call Path (Mizuki)
----------------------------
For per-token streaming (the primary production path)::

    from pyquifer.bridge import PyQuiferBridge, SteppedModulator

    # One-time setup
    bridge = PyQuiferBridge.realtime()   # or .interactive() / .default()
    stepper = SteppedModulator(bridge, step_every=8)

    # Per-token loop
    for token_embedding in stream:
        sensory = bridge.prepare_sensory_input(token_embedding)
        state = stepper.step(sensory)
        logits = model(token_embedding)
        logits = bridge.modulate_logits(logits, state)

For non-streaming (one tick per turn)::

    bridge = PyQuiferBridge.default()
    state = bridge.step(sensory_input, return_diagnostics=False)
    # Use state.temperature, state.top_p, state.repetition_penalty

The older ``PyQuifer`` class (core.py) is the data-science API for
standalone actualization/training.  ``PyQuiferBridge`` wraps the full
``CognitiveCycle`` and is the correct entry point for LLM integration.

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

References:
- PyQuifer design doc: Modified_Hidden equation
- Friston (2010): Free-energy principle
- Edelman (1987): Neural Darwinism
"""

import torch
import torch.nn as nn
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@contextmanager
def sync_debug_mode(mode: str = "warn"):
    """Context manager for CUDA synchronization detection during profiling.

    Args:
        mode: ``"warn"`` to print warnings on sync, ``"error"`` to raise.
              On CPU this is a no-op.

    Usage::

        with sync_debug_mode("error"):
            bridge.step(sensory)  # raises if any .item() sync occurs
    """
    if torch.cuda.is_available():
        prev = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode(mode)
        try:
            yield
        finally:
            torch.cuda.set_sync_debug_mode(prev)
    else:
        yield


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

    Training policy
    ---------------
    ``trait_vectors`` and ``modulation_gain`` are ``nn.Parameter`` and
    **can** be fine-tuned when the bridge is used inside a training loop
    (e.g. RLHF, SFT).  However, **gradient does NOT flow back to the
    oscillators** — the CognitiveCycle is always run inside
    ``torch.no_grad()`` for its buffer updates, and ``ModulationState``
    tensors (phases, neuromodulator_levels) are ``.detach()``-ed before
    injection into hidden states.

    In the benchmark suite the bridge runs in **eval-only mode**:
    ``bridge.eval()`` is not strictly required (the cycle already
    manages its own no_grad blocks) but is recommended for clarity.
    No optimizer is created for bridge parameters during benchmarking.

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

        # Last diagnostic snapshot (updated by step_diagnostic / step(diagnostics='always'))
        self._last_diagnostics: dict = {}

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

    @staticmethod
    def interactive():
        """Bridge tuned for real-time streaming (<=2ms target).

        Uses hierarchical timestepping and reduced HPC iterations.
        """
        from pyquifer.integration import CycleConfig
        return PyQuiferBridge(CycleConfig.interactive())

    @staticmethod
    def realtime():
        """Bridge for absolute minimum latency (<1.5ms target).

        Maximum hierarchical timestepping. Best for high-frequency
        tick loops (10+ Hz) where responsive feel matters more
        than per-tick accuracy.
        """
        from pyquifer.integration import CycleConfig
        return PyQuiferBridge(CycleConfig.realtime())

    @staticmethod
    def neuroscience():
        """Bridge tuned for maximum neuroscience alignment.

        Uses modular oscillator topology, true theta-gamma PAC,
        size-normalized metastability, and avalanche-consciousness linking.
        """
        from pyquifer.integration import CycleConfig
        return PyQuiferBridge(CycleConfig.neuroscience())

    def compile(self, mode: str = "default",
                backend: str = "inductor") -> 'PyQuiferBridge':
        """Apply torch.compile to performance-critical submodules.

        Fuses small tensor operations into fewer kernel launches for
        lower tick latency.  Safe to call on any platform — silently
        falls back to eager mode when compilation is unavailable.

        Typical usage::

            bridge = PyQuiferBridge.realtime().compile()

        Args:
            mode: "default", "reduce-overhead" (CUDA graphs), or
                  "max-autotune".
            backend: "inductor" (default) or "eager" (debugging).

        Returns:
            self (for chaining)
        """
        self.cycle.compile_modules(mode=mode, backend=backend)
        return self

    def enable_cuda_graphs(self,
                           n_warmup: int = 3,
                           latency_threshold_ms: float = 5.0,
                           ) -> bool:
        """Capture the minimal tick as a CUDA graph for replay.

        Prerequisites (enforced):
        - Device must be CUDA.
        - Minimal tick must be static (fixed shapes, no sync).

        The method captures the graph only if measured eager latency
        exceeds ``latency_threshold_ms`` — otherwise CUDA graphs add
        overhead without benefit for small workloads.

        Args:
            n_warmup: Warmup iterations before capture.
            latency_threshold_ms: Only capture if eager p50 > this value.

        Returns:
            True if graph was captured, False if skipped/failed.
        """
        device = next(self.parameters()).device
        if not device.type == 'cuda':
            return False

        sd = self.config.state_dim
        # Allocate static buffers
        self._cg_input = torch.randn(sd, device=device)
        self._cg_output = None
        self._cuda_graph = None

        # Measure eager baseline
        import time as _time
        for _ in range(n_warmup):
            self.cycle.tick(self._cg_input, return_diagnostics=False)
        torch.cuda.synchronize()

        lats = []
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = _time.perf_counter()
            self.cycle.tick(self._cg_input, return_diagnostics=False)
            torch.cuda.synchronize()
            lats.append((_time.perf_counter() - t0) * 1000)
        lats.sort()
        eager_p50 = lats[len(lats) // 2]

        if eager_p50 < latency_threshold_ms:
            # Not worth capturing — eager is fast enough
            return False

        # Capture
        try:
            g = torch.cuda.CUDAGraph()
            # Side-channel capture: run tick, record
            with torch.cuda.graph(g):
                self._cg_output = self.cycle.tick(
                    self._cg_input, return_diagnostics=False,
                )
            self._cuda_graph = g
            return True
        except Exception:
            self._cuda_graph = None
            return False

    def disable_cuda_graphs(self):
        """Disable CUDA graph replay and revert to eager execution."""
        self._cuda_graph = None
        self._cg_input = None
        self._cg_output = None

    def step(self,
             sensory_input: torch.Tensor,
             reward: float = 0.0,
             sleep_signal: float = 0.0,
             diagnostics: str = 'never') -> ModulationState:
        """
        Run one cognitive tick. Call this once per generation step.

        By default runs the **minimal tick** (no ``.item()`` calls, no dict
        construction, tensor-only return from the cycle).  This is the
        recommended hot path for production use.

        Args:
            sensory_input: Any signal the LLM provides (state_dim,).
                          Could be: mean hidden state, token embedding,
                          attention pattern summary, or synthetic signal.
            reward: Scalar reward (e.g., from RLHF, user feedback, task success).
            sleep_signal: 0.0=awake, 1.0=consolidating (e.g., between conversations).
            diagnostics: ``'never'`` (default) — minimal tick, no diagnostics.
                         ``'always'`` — full diagnostics every step (legacy behavior).
                         ``'auto'`` — same as ``'never'`` (reserved for future
                         decimated diagnostics).

        Returns:
            ModulationState with all parameters the LLM needs.
        """
        want_diag = diagnostics == 'always'
        return self._step_impl(sensory_input, reward, sleep_signal, want_diag)

    def step_diagnostic(self,
                        sensory_input: torch.Tensor,
                        reward: float = 0.0,
                        sleep_signal: float = 0.0,
                        ) -> 'tuple[ModulationState, dict]':
        """Run one cognitive tick with full diagnostics.

        Convenience wrapper that always requests the diagnostic path.
        Use this for debugging, monitoring dashboards, or benchmark
        analysis — NOT on the hot path.

        Returns:
            ``(ModulationState, diagnostics_dict)`` where diagnostics_dict
            has keys ``'consciousness'``, ``'self_state'``, ``'learning'``,
            ``'diagnostics'``.
        """
        return self._step_impl(sensory_input, reward, sleep_signal, True)

    def _step_impl(self,
                   sensory_input: torch.Tensor,
                   reward: float,
                   sleep_signal: float,
                   return_diagnostics: bool):
        """Shared implementation for step() and step_diagnostic()."""
        t0 = time.perf_counter()

        # Ensure correct dimension
        if sensory_input.shape[-1] != self.config.state_dim:
            sensory_input = self._project_input(sensory_input)

        # CUDA graph replay path (minimal tick only, default reward/sleep)
        _cg = getattr(self, '_cuda_graph', None)
        if _cg is not None and not return_diagnostics and reward == 0.0 and sleep_signal == 0.0:
            self._cg_input.copy_(sensory_input)
            _cg.replay()
            result = self._cg_output
        else:
            # Standard eager path
            # NOTE: Do NOT wrap in torch.no_grad() — the HPC module uses
            # autograd internally for generative model learning (gen_lr).
            result = self.cycle.tick(
                sensory_input, reward=reward, sleep_signal=sleep_signal,
                return_diagnostics=return_diagnostics,
            )

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000

        # Update latency EMA
        with torch.no_grad():
            alpha = 0.1 if self._latency_count > 0 else 1.0
            self._latency_ema.mul_(1 - alpha).add_(alpha * latency_ms)
            self._latency_count.add_(1)

        if return_diagnostics:
            # Full diagnostic path — unpack from dict
            m = result['modulation']
            c = result['consciousness']
            s = result['self_state']
            d = result['diagnostics']

            temperature = m['temperature']
            repetition_penalty = self._compute_repetition_penalty(c['coherence'], m['motivation'])
            top_p = self._compute_top_p(c['coherence'], c['criticality_distance'])

            mod_state = ModulationState(
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                dominant_facet=m['dominant_state'],
                facet_weights=m['personality_blend']['facet_weights'],
                personality_stability=m['personality_blend']['stability'],
                attention_bias=m['attention_bias'],
                processing_mode=m['processing_mode'],
                coherence=m['coherence'],
                motivation=m['motivation'],
                free_energy=c['free_energy'],
                criticality_distance=c['criticality_distance'],
                identity_strength=s['identity_strength'],
                tick=d['tick'],
                phases=d['phases'],
                neuromodulator_levels=d['neuromodulator_levels'],
                step_latency_ms=latency_ms,
            )
            self._last_diagnostics = {
                'consciousness': c,
                'self_state': s,
                'learning': result.get('learning', {}),
                'diagnostics': d,
            }
            return mod_state, self._last_diagnostics
        else:
            # Minimal path — unpack from TickResult (tensor-only NamedTuple)
            from pyquifer.integration import PROCESSING_MODE_NAMES
            coherence_f = result.coherence.item()
            motivation_f = result.motivation.item()
            temperature_f = result.temperature.item()
            processing_mode_str = PROCESSING_MODE_NAMES.get(
                int(result.processing_mode), "balanced"
            )

            repetition_penalty = self._compute_repetition_penalty(coherence_f, motivation_f)

            # Read real cached metrics from cycle (updated every tick, no .item() overhead)
            crit_dist_f = self.cycle._cached_criticality_distance.item()
            free_energy_f = self.cycle._cached_free_energy.item()
            identity_str_f = self.cycle._cached_identity_strength.item()

            top_p = self._compute_top_p(coherence_f, crit_dist_f)

            # Personality blend: convert tensor → list
            pb_weights = result.personality_blend.tolist()

            return ModulationState(
                temperature=temperature_f,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                dominant_facet=int(result.dominant_state),
                facet_weights=pb_weights,
                personality_stability=0.5,  # Not available in minimal path
                attention_bias=result.attention_bias,
                processing_mode=processing_mode_str,
                coherence=coherence_f,
                motivation=motivation_f,
                free_energy=free_energy_f,
                criticality_distance=crit_dist_f,
                identity_strength=identity_str_f,
                tick=self.cycle._tick_py,
                phases=self.cycle.oscillators.phases.detach(),
                neuromodulator_levels=self.cycle.neuromodulation.levels.detach(),
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
            logits: Raw logits from LLM.  Supports both
                    ``(batch, vocab_size)`` and ``(batch, seq_len, vocab_size)``
                    shapes.  The last dimension is always treated as vocab.
            state: ModulationState from step()

        Returns:
            Modified logits (same shape as input)
        """
        assert logits.dim() in (2, 3), (
            f"modulate_logits expects 2D (B,V) or 3D (B,T,V) logits, got {logits.dim()}D"
        )
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
            nm_gain = state.neuromodulator_levels.to(device).mean()
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

    def profile_step(self,
                     sensory_input: torch.Tensor,
                     n_warmup: int = 5,
                     n_measure: int = 10,
                     sync_mode: str = "warn") -> Dict[str, Any]:
        """Run minimal ticks with sync detection and latency profiling.

        Useful for regression testing: verifies no ``.item()`` syncs on
        CUDA and measures tail latency.

        Args:
            sensory_input: (state_dim,) input tensor.
            n_warmup: Warmup iterations (not measured).
            n_measure: Measurement iterations.
            sync_mode: ``"warn"`` or ``"error"`` for CUDA sync detection.

        Returns:
            Dict with ``latencies_ms`` (list), ``p50_ms``, ``p95_ms``,
            ``p99_ms``, ``sync_warnings`` (count, CUDA only).
        """
        import warnings

        # Warmup
        for _ in range(n_warmup):
            self.step(sensory_input)

        latencies = []
        sync_warn_count = 0

        with sync_debug_mode(sync_mode):
            for _ in range(n_measure):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    t0 = time.perf_counter()
                    self.step(sensory_input)
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000)
                    sync_warn_count += len(caught)

        latencies.sort()
        n = len(latencies)
        return {
            'latencies_ms': latencies,
            'p50_ms': latencies[n // 2] if n else 0,
            'p95_ms': latencies[int(n * 0.95)] if n else 0,
            'p99_ms': latencies[int(n * 0.99)] if n else 0,
            'mean_ms': sum(latencies) / n if n else 0,
            'sync_warnings': sync_warn_count,
        }

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        return {
            'mean_latency_ms': self._latency_ema.item(),
            'num_steps': self._latency_count.item(),
        }

    def get_state_summary(self) -> Dict[str, Any]:
        """Compact cognitive state summary."""
        return self.cycle.get_state_summary()

    def get_diagnostics(self) -> dict:
        """Return the last diagnostic snapshot.

        Updated only when ``step_diagnostic()`` or
        ``step(diagnostics='always')`` is called.  Returns ``{}`` if
        no diagnostic step has been run yet.
        """
        return self._last_diagnostics

    def reset(self):
        """Reset cognitive engine state."""
        self.cycle.reset()
        self._latency_ema.zero_()
        self._latency_count.zero_()
        self._last_diagnostics = {}

    def prepare_sensory_input(self,
                              x: torch.Tensor,
                              device: torch.device | str | None = None,
                              dtype: torch.dtype | None = None) -> torch.Tensor:
        """Standardize a raw signal into a valid sensory input for step().

        Handles:
        - Multi-dimensional input: flattens to 1D via mean over leading dims
        - Dimension mismatch: projects to ``config.state_dim``
        - Device transfer: moves to *device* (defaults to bridge's device)
        - Dtype cast: casts to *dtype* (defaults to float32)

        Adapters should use this instead of manually slicing / padding::

            sensory = bridge.prepare_sensory_input(
                hidden_states.mean(dim=(0, 1)),  # e.g. (hidden_dim,)
                device=hidden_states.device,
            )
            state = bridge.step(sensory)

        Args:
            x: Arbitrary tensor — (D,), (B, D), or (B, T, D).
            device: Target device.  Defaults to the bridge module's device.
            dtype: Target dtype.  Defaults to torch.float32.

        Returns:
            1-D tensor of shape ``(state_dim,)`` ready for ``step()``.
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = torch.float32

        # Collapse to 1-D
        if x.dim() > 1:
            x = x.mean(dim=tuple(range(x.dim() - 1)))  # keep last dim

        x = x.to(device=device, dtype=dtype)
        return self._project_input(x)

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


class PyQuiferLogitsProcessor:
    """HuggingFace-compatible LogitsProcessor for bridge modulation.

    Plugs into ``model.generate(logits_processor=[...])`` so you don't
    need to override ``_model_generate()``.  Calls ``bridge.modulate_logits()``
    on every token.

    Usage::

        from transformers import LogitsProcessorList
        from pyquifer.bridge import PyQuiferBridge, PyQuiferLogitsProcessor

        bridge = PyQuiferBridge.default()
        state = bridge.step(sensory)
        processor = PyQuiferLogitsProcessor(bridge, state)

        output = model.generate(
            input_ids,
            logits_processor=LogitsProcessorList([processor]),
        )

    For stepped modulation (bridge.step() every N tokens), combine with
    :class:`SteppedModulator`::

        stepper = SteppedModulator(bridge, step_every=8)
        processor = PyQuiferLogitsProcessor(bridge, stepper=stepper, sensory=sensory)
    """

    def __init__(self,
                 bridge: 'PyQuiferBridge',
                 state: Optional[ModulationState] = None,
                 stepper: Optional['SteppedModulator'] = None,
                 sensory: Optional[torch.Tensor] = None):
        """
        Args:
            bridge: The PyQuiferBridge instance.
            state: Fixed ModulationState to use (if no stepper).
            stepper: Optional SteppedModulator for per-token stepping.
            sensory: Sensory input for the stepper (required if stepper is set).
        """
        self.bridge = bridge
        self._state = state
        self._stepper = stepper
        self._sensory = sensory

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply bridge modulation to scores (logits).

        Args:
            input_ids: (batch, seq_len) — generated tokens so far.
            scores: (batch, vocab_size) — logits for next token.

        Returns:
            Modified scores (same shape).
        """
        if self._stepper is not None and self._sensory is not None:
            state = self._stepper.step(self._sensory)
        elif self._state is not None:
            state = self._state
        else:
            return scores  # No state available, pass through
        return self.bridge.modulate_logits(scores, state)


class SteppedModulator:
    """Amortizes ``bridge.step()`` over multiple tokens via interpolation.

    Instead of calling the full cognitive cycle every token (which costs
    ~45ms on CPU), this helper calls ``step()`` once every *step_every*
    tokens and returns linearly-interpolated :class:`ModulationState`
    objects in between.

    With ``step_every=16``, per-token overhead drops from ~45ms to ~0.3ms
    (only modulate_logits/modulate_hidden run on non-step tokens).

    Usage::

        bridge = PyQuiferBridge.default()
        stepper = SteppedModulator(bridge, step_every=8)
        for token_idx in range(num_tokens):
            state = stepper.step(sensory_input)
            logits = bridge.modulate_logits(raw_logits, state)

    Args:
        bridge: The :class:`PyQuiferBridge` to wrap.
        step_every: Run ``bridge.step()`` once per this many tokens.
    """

    def __init__(self, bridge: PyQuiferBridge, step_every: int = 8):
        self.bridge = bridge
        self.step_every = max(1, step_every)
        self._token_count: int = 0
        self._prev_state: Optional[ModulationState] = None
        self._curr_state: Optional[ModulationState] = None

    def step(self,
             sensory_input: torch.Tensor,
             reward: float = 0.0,
             sleep_signal: float = 0.0) -> ModulationState:
        """Return the current (possibly interpolated) modulation state.

        Calls ``bridge.step()`` on every *step_every*-th invocation.
        On other calls, returns a linearly-interpolated state.
        """
        if self._token_count % self.step_every == 0:
            self._prev_state = self._curr_state
            self._curr_state = self.bridge.step(
                sensory_input, reward=reward, sleep_signal=sleep_signal,
            )
        self._token_count += 1

        if self._prev_state is None or self.step_every <= 1:
            return self._curr_state  # type: ignore[return-value]

        alpha = (self._token_count % self.step_every) / self.step_every
        if alpha == 0.0:
            # Exactly on step boundary — return fresh state
            return self._curr_state  # type: ignore[return-value]
        return _interpolate_state(self._prev_state, self._curr_state, alpha)

    @property
    def current_state(self) -> Optional[ModulationState]:
        """Last computed (non-interpolated) state."""
        return self._curr_state

    def reset(self):
        """Reset token counter and cached states."""
        self._token_count = 0
        self._prev_state = None
        self._curr_state = None
        self.bridge.reset()


def _interpolate_state(s1: ModulationState, s2: ModulationState,
                       alpha: float) -> ModulationState:
    """Linear interpolation between two ModulationStates.

    Scalar fields are lerped.  Tensor fields use ``torch.lerp``.
    Categorical/string fields take the value from *s2* (the newer state).
    """
    def _lerp_f(a: float, b: float) -> float:
        return a * (1 - alpha) + b * alpha

    def _lerp_t(a: Optional[torch.Tensor],
                b: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if a is None or b is None:
            return b
        return torch.lerp(a, b, alpha)

    # Interpolate facet_weights (both are lists of the same length)
    if len(s1.facet_weights) == len(s2.facet_weights):
        blended_weights = [
            w1 * (1 - alpha) + w2 * alpha
            for w1, w2 in zip(s1.facet_weights, s2.facet_weights)
        ]
    else:
        blended_weights = s2.facet_weights

    return ModulationState(
        temperature=_lerp_f(s1.temperature, s2.temperature),
        repetition_penalty=_lerp_f(s1.repetition_penalty, s2.repetition_penalty),
        top_p=_lerp_f(s1.top_p, s2.top_p),
        dominant_facet=s2.dominant_facet,
        facet_weights=blended_weights,
        personality_stability=_lerp_f(s1.personality_stability, s2.personality_stability),
        attention_bias=_lerp_t(s1.attention_bias, s2.attention_bias),
        processing_mode=s2.processing_mode,
        coherence=_lerp_f(s1.coherence, s2.coherence),
        motivation=_lerp_f(s1.motivation, s2.motivation),
        free_energy=_lerp_f(s1.free_energy, s2.free_energy),
        criticality_distance=_lerp_f(s1.criticality_distance, s2.criticality_distance),
        identity_strength=_lerp_f(s1.identity_strength, s2.identity_strength),
        tick=s2.tick,
        phases=_lerp_t(s1.phases, s2.phases),
        neuromodulator_levels=_lerp_t(s1.neuromodulator_levels, s2.neuromodulator_levels),
        step_latency_ms=s2.step_latency_ms,
    )


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
