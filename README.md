# PyQuifer

**Oscillatory neural computation for emergent cognition.**

PyTorch library implementing temporal dynamics as a cognitive substrate. Cognition emerges from coupled oscillators synchronizing and competing at the edge of chaos — not from static weight matrices.

**79 modules · 350+ classes · 690+ tests**

---

## The Core Idea

Static weights store information. Oscillators *process* it in time.

```
Modified_Hidden = Original + A · sin(ωt + φ) · Trait_Vector
```

LLM gradient flow to oscillators is **severed by design** — the dynamical substrate evolves through its own physics (Kuramoto coupling, PLL entrainment), not backprop.

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

### Layer 1 — Learning & Plasticity

| Module | Key Classes |
|--------|-------------|
| `learning.py` | `EligibilityTrace`, `RewardModulatedHebbian`, `PredictiveCoding` |
| `spiking.py` | `LIFNeuron`, `SpikingLayer`, `STDPLayer` |
| `advanced_spiking.py` | `SynapticNeuron`, `RewardPredictionError`, `EligibilityModulatedSTDP` |
| `continual_learning.py` | `ContinualLearner`, `ExperienceReplay` |
| `memory_consolidation.py` | `EpisodicBuffer`, `SharpWaveRipple`, `ConsolidationEngine` |
| `neural_darwinism.py` | `NeuronalGroup`, `SelectionArena`, `SymbiogenesisDetector` |

### Layer 2 — Embodiment

| Module | Key Classes |
|--------|-------------|
| `somatic.py` | `SomaticState`, `HardwareSensor`, `SomaticManifold` |
| `morphological.py` | `TensionField`, `PeripheralGanglion`, `SleepWakeController` |
| `kindchenschema.py` | `SafetyEnvelope`, `ParentModule`, `LimbicResonance` |
| `social.py` | `MirrorResonance`, `SocialCoupling`, `TheoryOfMind` |
| `ecology.py` | `ChronobiologicalSystem`, `SynapticHomeostasis`, `Umwelt` |
| `developmental.py` | `DevelopmentalStageDetector`, `PotentialActualization` |
| `voice_dynamics.py` | `SpeechOscillator`, `VoiceNeuromodulation`, `ProsodyModulator` |

### Layer 3 — Dynamical Core

| Module | Key Classes |
|--------|-------------|
| `oscillators.py` | `LearnableKuramotoBank` |
| `spherical.py` | `SphericalKuramotoLayer`, `SphericalKuramotoBank` |
| `frequency_bank.py` | `FrequencyBank` |
| `linoss.py` | `LinOSSLayer`, `HarmonicOscillator`, `LinOSSEncoder` |
| `liquid_networks.py` | `LiquidTimeConstantCell`, `NeuralODE`, `MetastableCell` |
| `reservoir.py` | `EchoStateNetwork`, `IntrinsicPlasticity`, `CriticalReservoir` |
| `strange_attractor.py` | `LorenzAttractor`, `PersonalityAttractor` |
| `hyperbolic.py` | `HyperbolicLinear`, `EmotionalGravityManifold` |
| `thermodynamic.py` | `ThermodynamicOscillatorSystem`, `LangevinDynamics` |
| `metastability.py` | `WinnerlessCompetition`, `HeteroclinicChannel`, `MetastabilityIndex` |

Criticality target: σ = 1.0 (edge of chaos — not seizure, not coma).

### Layer 4 — Information Flow

| Module | Key Classes |
|--------|-------------|
| `multiplexing.py` | `CrossFrequencyCoupling`, `TemporalMultiplexer`, `NestedOscillator` |
| `phase_attention.py` | `PhaseAttention`, `PhaseMultiHeadAttention`, `OscillatorGatedFFN` |
| `hypernetwork.py` | `HyperNetwork`, `OscillatorHyperNetwork`, `DynamicLinear` |
| `hyperdimensional.py` | `HypervectorMemory`, `PhaseBinder`, `HDCReasoner` |
| `precision_weighting.py` | `PrecisionEstimator`, `PrecisionGate`, `AttentionAsPrecision` |
| `causal_flow.py` | `TransferEntropyEstimator`, `CausalFlowMap`, `DominanceDetector` |
| `stochastic_resonance.py` | `AdaptiveStochasticResonance`, `ResonanceMonitor` |

### Layer 5 — Consciousness

| Module | Key Classes |
|--------|-------------|
| `consciousness.py` | `PerturbationalComplexity`, `IntegrationMeasure`, `ConsciousnessMonitor` |
| `iit_metrics.py` | `IntegratedInformation`, `IITConsciousnessMonitor` |
| `global_workspace.py` | `HierarchicalWorkspace` |
| `active_inference.py` | `ActiveInferenceAgent`, `ExpectedFreeEnergy`, `BeliefUpdate` |
| `hierarchical_predictive.py` | `PredictiveLevel`, `HierarchicalPredictiveCoding` |
| `criticality.py` | `AvalancheDetector`, `BranchingRatio`, `CriticalityController` |
| `neuromodulation.py` | `NeuromodulatorDynamics`, `GlialLayer`, `InjectionLocking` |
| `quantum_cognition.py` | `QuantumDecisionMaker` |

### Layer 6 — Metacognition

| Module | Key Classes |
|--------|-------------|
| `metacognitive.py` | `ConfidenceEstimator`, `ReasoningMonitor`, `MetacognitiveLoop` |
| `motivation.py` | `NoveltyDetector`, `MasterySignal`, `IntrinsicMotivationSystem` |

### Layer 7 — Self-Model

| Module | Key Classes |
|--------|-------------|
| `self_model.py` | `MarkovBlanket`, `SelfModel`, `NarrativeIdentity` |
| `models.py` | `PersonalityArchetype` |
| `core.py` | `PyQuifer`, `MindEyeActualization` |

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
# Metastability — the "stream of consciousness"
from pyquifer.metastability import MetastabilityIndex

mi = MetastabilityIndex(num_populations=6)
for _ in range(500):
    result = mi()
    print(f"Dominant: {result['dominant'].item()}  "
          f"Entropy: {result['coalition_entropy'].item():.3f}")
```

```python
# Sleep consolidation
from pyquifer.memory_consolidation import EpisodicBuffer, SharpWaveRipple, ConsolidationEngine

buffer = EpisodicBuffer(state_dim=64, capacity=1000)
ripple = SharpWaveRipple(state_dim=64)
engine = ConsolidationEngine(state_dim=64, semantic_dim=32)

replay = ripple(buffer, sleep_signal=0.9)
engine(replay['replayed_states'], replay['replay_counts'])
```

---

## Testing

```bash
pytest tests/ -v                          # 690+ tests
pytest tests/test_integration.py -v      # cross-module
python tests/generate_visualizations.py  # diagnostic plots → tests/test_output/
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
| Stochastic resonance | Gammaitoni et al. (1998) | `stochastic_resonance` |

---

## Design Principles

- **Dynamics over weights** — temporal patterns carry information static weights cannot
- **Severed gradient flow** — oscillators evolve through their own physics, not LLM backprop
- **Criticality** — σ = 1.0, the edge of chaos where computation is maximized
- **Embodiment** — hardware state becomes somatic sensation; social interaction shapes dynamics
- **Integration** — consciousness-like properties emerge from whole-system information integration
- **Competition** — neuronal groups compete at micro-scale, producing coherence at macro-scale

---

PyQuifer is the oscillatory substrate for [Project Mizuki AI](https://github.com/takku1/Project_Mizuki_AI).

*"Each thought: a saddle point on a heteroclinic orbit. Each moment of awareness: a brief coalition of synchronized oscillators at the edge of chaos."*
