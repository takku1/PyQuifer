"""
Oscillator-Driven LoRA Adapter Manager.

Bridges PyQuifer's oscillator state (ModulationState) with PEFT's LoRA
adapter system.  The oscillators control which personality adapters are
active and how they blend — turning system-prompt personality hints into
actual weight-level modulation.

Three integration levels:
  1. Discrete switching:  dominant facet → swap adapter (simplest)
  2. Weighted blending:   facet_weights → linear adapter combination (per-tick)
  3. Gated routing:       oscillator features → learned gate → per-layer scaling (deepest)

Design constraint (inherited from bridge.py):
  LLM → Oscillator gradient flow is SEVERED.  Adapter weights are learned
  through standard fine-tuning (LoRA/QLoRA), but the *selection* of which
  adapter is active comes from oscillator dynamics, not backprop.

Usage::

    from pyquifer.adapter_manager import AdapterManager
    from pyquifer.bridge import PyQuiferBridge

    bridge = PyQuiferBridge.interactive()
    manager = AdapterManager(bridge)

    # Register personality adapters (paths to PEFT adapter dirs or state_dicts)
    manager.register_adapter("curious", path="adapters/curious_lora")
    manager.register_adapter("warm", path="adapters/warm_lora")
    manager.register_adapter("analytical", path="adapters/analytical_lora")

    # Attach to a PEFT-wrapped model
    manager.attach(model)

    # Each tick: oscillator state automatically selects/blends adapters
    state = bridge.step(sensory_input)
    manager.update(state)  # updates adapter weights based on oscillator state

    # Generate with oscillator-modulated personality at the weight level
    output = model.generate(input_ids, ...)

References:
- Ha et al. (2016). HyperNetworks.
- Hu et al. (2021). LoRA: Low-Rank Adaptation.
- Buehler et al. (2024). X-LoRA: Mixture of LoRA Experts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Literal, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AdapterConfig:
    """Configuration for a registered personality adapter."""
    name: str
    path: Optional[str] = None       # Path to PEFT adapter dir
    state_dict: Optional[dict] = None  # Or raw state_dict
    facet_index: int = -1             # Which personality facet this maps to (-1 = unmapped)
    base_weight: float = 0.0         # Default weight when no oscillator signal
    rank: int = 16                   # LoRA rank (for validation)
    description: str = ""


@dataclass
class BlendState:
    """Current adapter blend state — what's active and at what weight."""
    active_adapters: List[str] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    blend_name: str = "oscillator_blend"
    coherence: float = 0.5
    temperature: float = 1.0  # Blend temperature (from oscillator coherence)


class AdapterGate(nn.Module):
    """Learned gate that maps oscillator features to adapter blend weights.

    This is the Level 3 (deepest) integration — a small network that
    takes oscillator state features and produces per-adapter scaling
    coefficients.  Similar to X-LoRA's gating but driven by oscillator
    dynamics instead of hidden states.

    The gate is trained separately (e.g., on preference data) — NOT
    through the oscillator gradient path.

    Args:
        num_adapters: Number of registered adapters.
        oscillator_dim: Dimension of oscillator feature vector.
            Default 13 = 5 neuromodulators + 1 coherence + 1 motivation
            + 1 criticality + 1 free_energy + 4 band powers.
        hidden_dim: Gate hidden layer size.
        temperature: Softmax temperature (lower = sharper selection).
    """

    def __init__(
        self,
        num_adapters: int,
        oscillator_dim: int = 13,
        hidden_dim: int = 32,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_adapters = num_adapters
        self.temperature = temperature

        self.gate = nn.Sequential(
            nn.Linear(oscillator_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_adapters),
        )

        # Initialize near-uniform output
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, oscillator_features: torch.Tensor) -> torch.Tensor:
        """Compute adapter blend weights from oscillator features.

        Args:
            oscillator_features: (oscillator_dim,) feature vector.

        Returns:
            (num_adapters,) softmax weights summing to 1.
        """
        logits = self.gate(oscillator_features)
        return F.softmax(logits / self.temperature, dim=-1)


class AdapterManager:
    """Manages LoRA adapter selection and blending driven by oscillator state.

    This is the glue between PyQuifer's cognitive engine and PEFT's adapter
    system.  It reads ModulationState from the bridge and translates it into
    adapter operations (switch, blend, or gate).

    Args:
        bridge: PyQuiferBridge instance (for reading config like num_populations).
        mode: Integration level.
            - ``'switch'``: Discrete adapter switching on dominant facet change.
            - ``'blend'``: Weighted linear combination every tick.
            - ``'gate'``: Learned gate network for continuous routing.
        blend_momentum: Exponential moving average for blend weight smoothing.
            0.0 = instant switch, 0.9 = very smooth transitions.
            Prevents jarring personality jumps between ticks.
        coherence_sharpening: If True, high oscillator coherence sharpens the
            blend toward the dominant adapter (more focused personality).
            Low coherence spreads weight more evenly (exploratory personality).
    """

    def __init__(
        self,
        bridge: Any = None,
        mode: Literal['switch', 'blend', 'gate'] = 'blend',
        blend_momentum: float = 0.3,
        coherence_sharpening: bool = True,
    ):
        self.bridge = bridge
        self.mode = mode
        self.blend_momentum = blend_momentum
        self.coherence_sharpening = coherence_sharpening

        # Registered adapters
        self._adapters: Dict[str, AdapterConfig] = {}
        self._facet_to_adapter: Dict[int, str] = {}  # facet_index -> adapter name

        # Current state
        self._current_blend = BlendState()
        self._prev_weights: Optional[torch.Tensor] = None
        self._model = None  # Attached PEFT model

        # Learned gate (Level 3 only)
        self._gate: Optional[AdapterGate] = None

        # Stats
        self._update_count = 0
        self._switch_count = 0

    def register_adapter(
        self,
        name: str,
        path: Optional[str] = None,
        state_dict: Optional[dict] = None,
        facet_index: int = -1,
        base_weight: float = 0.0,
        rank: int = 16,
        description: str = "",
    ) -> 'AdapterManager':
        """Register a personality adapter.

        Args:
            name: Unique adapter name (e.g., "curious", "warm").
            path: Path to PEFT adapter directory.
            state_dict: Alternative: raw adapter state_dict.
            facet_index: Which personality facet this maps to.
                Maps to ModulationState.facet_weights[facet_index].
            base_weight: Default weight when no oscillator signal.
            rank: LoRA rank for validation.
            description: Human-readable description.

        Returns:
            self (for chaining).
        """
        config = AdapterConfig(
            name=name,
            path=path,
            state_dict=state_dict,
            facet_index=facet_index,
            base_weight=base_weight,
            rank=rank,
            description=description,
        )
        self._adapters[name] = config

        if facet_index >= 0:
            self._facet_to_adapter[facet_index] = name

        logger.info(f"Registered adapter '{name}' (facet={facet_index}, rank={rank})")
        return self

    def attach(self, model: Any) -> 'AdapterManager':
        """Attach to a PEFT-wrapped model.

        Loads all registered adapters into the model.  The model must
        already be wrapped with PeftModel (first adapter loaded).

        Args:
            model: A PeftModel instance.

        Returns:
            self (for chaining).
        """
        self._model = model

        # Load adapters that aren't already in the model
        for name, config in self._adapters.items():
            if config.path:
                try:
                    model.load_adapter(config.path, adapter_name=name)
                    logger.info(f"Loaded adapter '{name}' from {config.path}")
                except Exception as e:
                    logger.warning(f"Failed to load adapter '{name}': {e}")
            elif config.state_dict:
                try:
                    from peft.utils.hotswap import hotswap_adapter_from_state_dict
                    hotswap_adapter_from_state_dict(
                        model, config.state_dict, adapter_name=name
                    )
                    logger.info(f"Loaded adapter '{name}' from state_dict")
                except Exception as e:
                    logger.warning(f"Failed to load adapter '{name}' from state_dict: {e}")

        # Initialize gate if in gate mode
        if self.mode == 'gate' and self._gate is None:
            self._gate = AdapterGate(num_adapters=len(self._adapters))
            logger.info(f"Initialized AdapterGate with {len(self._adapters)} adapters")

        return self

    def update(self, mod_state: Any) -> BlendState:
        """Update adapter blend based on current oscillator state.

        This is the main per-tick method.  Call after bridge.step().

        Args:
            mod_state: ModulationState from bridge.step().

        Returns:
            BlendState with current adapter weights.
        """
        self._update_count += 1
        adapter_names = list(self._adapters.keys())
        if not adapter_names:
            return self._current_blend

        if self.mode == 'switch':
            return self._update_switch(mod_state, adapter_names)
        elif self.mode == 'blend':
            return self._update_blend(mod_state, adapter_names)
        elif self.mode == 'gate':
            return self._update_gate(mod_state, adapter_names)
        else:
            return self._current_blend

    def _update_switch(self, mod_state: Any, adapter_names: List[str]) -> BlendState:
        """Discrete switching: pick one adapter based on dominant facet."""
        dominant = mod_state.dominant_facet
        target_name = self._facet_to_adapter.get(dominant)

        if target_name is None:
            # Fall back to first adapter
            target_name = adapter_names[0]

        # Only switch if facet actually changed
        if (self._current_blend.active_adapters
                and self._current_blend.active_adapters[0] == target_name):
            return self._current_blend

        # Apply switch
        if self._model is not None:
            try:
                self._model.set_adapter(target_name)
                self._switch_count += 1
                logger.debug(f"Switched to adapter '{target_name}' (facet={dominant})")
            except Exception as e:
                logger.warning(f"Failed to switch adapter: {e}")

        self._current_blend = BlendState(
            active_adapters=[target_name],
            weights=[1.0],
            coherence=mod_state.coherence,
        )
        return self._current_blend

    def _update_blend(self, mod_state: Any, adapter_names: List[str]) -> BlendState:
        """Weighted blending: combine adapters using facet_weights."""
        n = len(adapter_names)

        # Build raw weights from facet_weights
        raw_weights = torch.zeros(n)
        facet_weights = mod_state.facet_weights
        for i, name in enumerate(adapter_names):
            config = self._adapters[name]
            if config.facet_index >= 0 and config.facet_index < len(facet_weights):
                raw_weights[i] = facet_weights[config.facet_index]
            else:
                raw_weights[i] = config.base_weight

        # Coherence sharpening: high coherence → near one-hot selection
        if self.coherence_sharpening:
            coherence = mod_state.coherence
            # Temperature: coherence 0.8 → temp 0.2 (sharp), coherence 0.2 → temp 2.0 (flat)
            temp = max(0.1, 2.0 - coherence * 2.0)
            weights = F.softmax(raw_weights / temp, dim=-1)
        else:
            # Plain normalization
            total = raw_weights.sum()
            weights = raw_weights / total if total > 0 else torch.ones(n) / n

        # EMA smoothing to prevent jarring transitions
        if self._prev_weights is not None and self._prev_weights.shape == weights.shape:
            m = self.blend_momentum
            weights = m * self._prev_weights + (1 - m) * weights
            # Renormalize after smoothing
            weights = weights / weights.sum()
        self._prev_weights = weights.clone()

        # Apply to model
        weight_list = weights.tolist()
        if self._model is not None:
            try:
                self._model.add_weighted_adapter(
                    adapters=adapter_names,
                    weights=weight_list,
                    adapter_name="oscillator_blend",
                    combination_type="linear",
                )
                self._model.set_adapter("oscillator_blend")
            except Exception as e:
                logger.warning(f"Failed to blend adapters: {e}")

        self._current_blend = BlendState(
            active_adapters=adapter_names,
            weights=weight_list,
            blend_name="oscillator_blend",
            coherence=mod_state.coherence,
            temperature=max(0.1, 2.0 - mod_state.coherence * 2.0),
        )
        return self._current_blend

    def _update_gate(self, mod_state: Any, adapter_names: List[str]) -> BlendState:
        """Learned gate: oscillator features → per-adapter scaling."""
        if self._gate is None:
            return self._update_blend(mod_state, adapter_names)

        # Build oscillator feature vector
        features = self._extract_oscillator_features(mod_state)
        with torch.no_grad():
            weights = self._gate(features)

        # EMA smoothing
        if self._prev_weights is not None and self._prev_weights.shape == weights.shape:
            m = self.blend_momentum
            weights = m * self._prev_weights + (1 - m) * weights
            weights = weights / weights.sum()
        self._prev_weights = weights.clone()

        # Apply
        weight_list = weights.tolist()
        if self._model is not None:
            try:
                self._model.add_weighted_adapter(
                    adapters=adapter_names,
                    weights=weight_list,
                    adapter_name="oscillator_blend",
                    combination_type="linear",
                )
                self._model.set_adapter("oscillator_blend")
            except Exception as e:
                logger.warning(f"Failed to apply gated blend: {e}")

        self._current_blend = BlendState(
            active_adapters=adapter_names,
            weights=weight_list,
            blend_name="oscillator_blend",
            coherence=mod_state.coherence,
        )
        return self._current_blend

    def _extract_oscillator_features(self, mod_state: Any) -> torch.Tensor:
        """Extract a fixed-size feature vector from ModulationState.

        Returns a 13-dim vector:
          [5 neuromodulators, coherence, motivation, criticality_distance,
           free_energy, 4 facet weights (padded/truncated)]
        """
        features = []

        # Neuromodulator levels (5)
        if mod_state.neuromodulator_levels is not None:
            nm = mod_state.neuromodulator_levels.detach().float()
            if nm.numel() >= 5:
                features.append(nm[:5])
            else:
                features.append(F.pad(nm, (0, 5 - nm.numel())))
        else:
            features.append(torch.zeros(5))

        # Scalar features (4)
        features.append(torch.tensor([
            mod_state.coherence,
            mod_state.motivation,
            mod_state.criticality_distance,
            mod_state.free_energy,
        ]))

        # Facet weights (4, padded/truncated)
        fw = mod_state.facet_weights
        if len(fw) >= 4:
            features.append(torch.tensor(fw[:4]))
        else:
            features.append(torch.tensor(fw + [0.0] * (4 - len(fw))))

        return torch.cat(features)  # (13,)

    @property
    def current_blend(self) -> BlendState:
        """Current adapter blend state."""
        return self._current_blend

    @property
    def adapter_names(self) -> List[str]:
        """List of registered adapter names."""
        return list(self._adapters.keys())

    @property
    def gate(self) -> Optional[AdapterGate]:
        """The learned gate network (None if mode != 'gate')."""
        return self._gate

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter manager statistics."""
        return {
            'mode': self.mode,
            'num_adapters': len(self._adapters),
            'update_count': self._update_count,
            'switch_count': self._switch_count,
            'current_blend': {
                'adapters': self._current_blend.active_adapters,
                'weights': self._current_blend.weights,
                'coherence': self._current_blend.coherence,
            },
        }

    def __repr__(self) -> str:
        adapters = ", ".join(self._adapters.keys()) or "none"
        return f"AdapterManager(mode={self.mode!r}, adapters=[{adapters}])"


if __name__ == '__main__':
    print("=== AdapterManager Demo ===\n")

    # Simulate without a real PEFT model
    from pyquifer.bridge import ModulationState

    # Create manager in blend mode
    manager = AdapterManager(mode='blend', coherence_sharpening=True)

    # Register personality adapters mapped to facets
    manager.register_adapter("curious", facet_index=0, description="Exploratory, questioning")
    manager.register_adapter("warm", facet_index=1, description="Caring, empathetic")
    manager.register_adapter("analytical", facet_index=2, description="Logical, precise")
    manager.register_adapter("playful", facet_index=3, description="Fun, creative")

    print(f"Manager: {manager}")
    print(f"Adapters: {manager.adapter_names}")

    # Simulate tick updates with different oscillator states
    test_states = [
        ModulationState(
            dominant_facet=0,
            facet_weights=[0.6, 0.2, 0.1, 0.1],
            coherence=0.8,  # High coherence → sharp selection
            motivation=0.5,
        ),
        ModulationState(
            dominant_facet=1,
            facet_weights=[0.2, 0.5, 0.2, 0.1],
            coherence=0.3,  # Low coherence → broad blend
            motivation=0.7,
        ),
        ModulationState(
            dominant_facet=2,
            facet_weights=[0.1, 0.1, 0.7, 0.1],
            coherence=0.9,  # Very high → near one-hot
            motivation=0.3,
        ),
    ]

    print("\nBlend evolution across ticks:")
    for i, state in enumerate(test_states):
        blend = manager.update(state)
        weights_str = ", ".join(f"{n}={w:.3f}" for n, w in zip(blend.active_adapters, blend.weights))
        print(f"  Tick {i}: coherence={state.coherence:.1f} -> [{weights_str}]")

    # Show gate mode
    print("\n--- Gate Mode ---")
    gate_manager = AdapterManager(mode='gate')
    gate_manager.register_adapter("curious", facet_index=0)
    gate_manager.register_adapter("warm", facet_index=1)
    gate_manager.register_adapter("analytical", facet_index=2)

    # Initialize gate manually (normally done by attach())
    gate_manager._gate = AdapterGate(num_adapters=3)

    state = ModulationState(
        dominant_facet=0,
        facet_weights=[0.6, 0.2, 0.2],
        coherence=0.7,
        motivation=0.5,
        neuromodulator_levels=torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]),
    )
    blend = gate_manager.update(state)
    weights_str = ", ".join(f"{n}={w:.3f}" for n, w in zip(blend.active_adapters, blend.weights))
    print(f"  Gate output: [{weights_str}]")

    print(f"\nStats: {manager.get_stats()}")
    print("\n[OK] AdapterManager demo passed!")
