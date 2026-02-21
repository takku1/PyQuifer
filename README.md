# PyQuifer

**Oscillatory neural computation for emergent cognition.**

Cognition as temporal dynamics — coupled oscillators synchronizing and competing at the edge of chaos. Not static weight matrices.

```
Modified_Hidden = Original + A · sin(ωt + φ) · Trait_Vector
```

**Dynamics over weights** — temporal patterns carry information static weights cannot.
**Severed gradient flow** — oscillators evolve through their own physics (Kuramoto coupling, PLL entrainment), not LLM backprop. The dynamical substrate is separate from the language model by design.
**Criticality** — σ = 1.0, the edge of chaos. Not seizure, not coma. The sweet spot where computation is maximized.

---

## Architecture

Seven layers, each grounded in neuroscience and dynamical systems theory:

```
7  SELF-MODEL       Narrative identity, Markov blanket, self-prediction
6  METACOGNITION    Confidence estimation, uncertainty, reasoning monitor
5  CONSCIOUSNESS    Global workspace, IIT phi, predictive coding, free energy
4  INFORMATION      Cross-frequency coupling, causal flow, precision weighting
3  DYNAMICS         Kuramoto, attractors, reservoirs, metastability, criticality
2  EMBODIMENT       Somatic markers, social coupling, circadian rhythms
1  LEARNING         STDP, Hebbian, consolidation, neural darwinism
```

Layers form a closed loop — oscillators (3) generate dynamics that consciousness (5) measures, metacognition (6) monitors, and learning (1) reshapes over time.

### Core Modules

| Layer | Modules | Key Classes |
|-------|---------|-------------|
| **Learning** | `learning`, `spiking`, `memory_consolidation` | `EligibilityTrace`, `SharpWaveRipple`, `ConsolidationEngine` |
| **Embodiment** | `somatic`, `social`, `ecology` | `SomaticManifold`, `TheoryOfMind`, `ChronobiologicalSystem` |
| **Dynamics** | `oscillators`, `metastability`, `reservoir` | `LearnableKuramotoBank`, `WinnerlessCompetition`, `CriticalReservoir` |
| **Information** | `multiplexing`, `phase_attention`, `causal_flow` | `CrossFrequencyCoupling`, `PhaseAttention`, `TransferEntropyEstimator` |
| **Consciousness** | `consciousness`, `active_inference`, `criticality` | `PerturbationalComplexity`, `ActiveInferenceAgent`, `AvalancheDetector` |
| **Metacognition** | `metacognitive`, `motivation` | `ConfidenceEstimator`, `ReasoningMonitor`, `IntrinsicMotivationSystem` |
| **Self-Model** | `self_model`, `core` | `MarkovBlanket`, `NarrativeIdentity`, `PyQuifer` |

---

## Installation

```bash
git clone https://github.com/takku1/PyQuifer.git
cd PyQuifer
pip install -e .
pip install -e ".[dev]"  # tests + visualization
```

Python 3.10+ · PyTorch 2.0+

---

## Quick Start

```python
import torch
from pyquifer import LearnableKuramotoBank, ConsciousnessMonitor

oscillators = LearnableKuramotoBank(num_oscillators=64, dt=0.01, initial_coupling=0.5)

for _ in range(100):
    oscillators(steps=10)

print(f"Coherence: {oscillators.get_order_parameter():.3f}")

monitor = ConsciousnessMonitor(state_dim=64)
metrics = monitor(oscillators.get_complex_state().unsqueeze(0))
print(f"Integration: {metrics.get('integration', 0):.3f}")
```

```python
# The "stream of consciousness" — no winner, just flow
from pyquifer.metastability import MetastabilityIndex

mi = MetastabilityIndex(num_populations=6)
for _ in range(500):
    result = mi()
    # System cycles through saddle points (heteroclinic orbit)
    print(f"Dominant: {result['dominant'].item()}  "
          f"Entropy: {result['coalition_entropy'].item():.3f}")
```

```python
# Sleep consolidation — episodic memory becomes semantic knowledge
from pyquifer.memory_consolidation import EpisodicBuffer, SharpWaveRipple, ConsolidationEngine

buffer = EpisodicBuffer(state_dim=64, capacity=1000)
ripple = SharpWaveRipple(state_dim=64)
engine = ConsolidationEngine(state_dim=64, semantic_dim=32)

replay = ripple(buffer, sleep_signal=0.9)
engine(replay['replayed_states'], replay['replay_counts'])
```

---

## Theoretical Foundations

| Theory | Source | Module |
|--------|--------|--------|
| Phase synchronization | Kuramoto (1984) | `oscillators`, `spherical` |
| Integrated information (IIT) | Tononi (2004) | `iit_metrics`, `consciousness` |
| Free energy principle | Friston (2010) | `active_inference`, `hierarchical_predictive` |
| Self-organized criticality | Beggs & Plenz (2003) | `criticality`, `reservoir` |
| Winnerless competition | Rabinovich et al. (2001) | `metastability` |
| Neural darwinism | Edelman (1987) | `neural_darwinism` |
| Transfer entropy | Schreiber (2000) | `causal_flow` |
| Somatic markers | Damasio (1994) | `somatic` |
| Markov blankets | Friston (2013) | `self_model` |
| Liquid time-constant networks | Hasani et al. (2021) | `liquid_networks` |
| Sleep consolidation | Born & Wilhelm (2012) | `memory_consolidation` |

---

## Testing

```bash
pytest tests/ -v
python tests/generate_visualizations.py  # diagnostic plots → tests/test_output/
```

---

*"Each thought: a saddle point on a heteroclinic orbit. Each moment of awareness: a brief coalition of synchronized oscillators at the edge of chaos."*
