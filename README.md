# PyQuifer

**Oscillatory Neural Computation for Emergent Cognition**

PyQuifer is a PyTorch library that implements the computational substrate for conscious-like AI. Instead of treating neural networks as static weight matrices, PyQuifer models cognition as **temporal dynamics** — coupled oscillators synchronizing, competing, and self-organizing at the edge of chaos.

The core idea: consciousness isn't a thing, it's a *process*. Phase synchronization, criticality, prediction error minimization, and embodied feedback loops create the conditions for emergent awareness. PyQuifer provides the building blocks.

**48 modules | 200+ classes | 25K+ lines | 183 tests**

## What PyQuifer Does

PyQuifer models **seven layers of cognitive architecture**, each grounded in neuroscience and dynamical systems theory:

```
  7. SELF-MODEL            Who am I? Narrative identity, Markov blanket
  6. METACOGNITION          Thinking about thinking. Confidence, uncertainty
  5. CONSCIOUSNESS          Global workspace, IIT integration, predictive coding
  4. INFORMATION FLOW       Causal flow, transfer entropy, precision weighting
  3. DYNAMICS               Oscillators, attractors, criticality, metastability
  2. EMBODIMENT             Somatic markers, morphological computation, ecology
  1. LEARNING               Hebbian, STDP, consolidation, neural darwinism
```

These aren't isolated — they form a closed loop. Oscillators at Layer 3 generate the dynamics that Layer 5 measures as consciousness. Layer 6 monitors that consciousness and feeds back to modulate Layer 3. Layer 1's learning reshapes everything over time. Layer 7 tells the system who it is across all of this.

## Installation

```bash
git clone https://github.com/takku1/PyQuifer.git
cd PyQuifer
pip install -e .

# With dev dependencies (tests + visualization)
pip install -e ".[dev]"
```

Requires Python 3.8+ and PyTorch 2.0+.

## Architecture

### Layer 1: Learning & Plasticity

How the system learns and remembers. Not just backprop — biologically-inspired plasticity rules that operate on their own dynamics, independent of gradient descent.

| Module | What It Does | Key Classes |
|--------|-------------|-------------|
| `learning.py` | Hebbian learning, STDP, reward-modulated plasticity | `EligibilityTrace`, `RewardModulatedHebbian`, `PredictiveCoding` |
| `spiking.py` | Leaky integrate-and-fire neurons, spiking layers | `LIFNeuron`, `SpikingLayer`, `OscillatorySNN`, `STDPLayer` |
| `advanced_spiking.py` | Second-order neurons, E/I balance, eligibility-modulated STDP | `SynapticNeuron`, `AlphaNeuron`, `RewardPredictionError`, `EligibilityModulatedSTDP` |
| `continual_learning.py` | Elastic Weight Consolidation, experience replay | `ContinualLearner`, `ExperienceReplay` |
| `memory_consolidation.py` | Sleep replay, episodic-to-semantic transfer, reconsolidation | `EpisodicBuffer`, `SharpWaveRipple`, `ConsolidationEngine`, `MemoryReconsolidation` |
| `neural_darwinism.py` | Neuronal group selection, competition for resources | `NeuronalGroup`, `SelectionArena`, `SymbiogenesisDetector` |

**Design principle:** Oscillator dynamics (Kuramoto, PLL entrainment) evolve through their own physics, NOT through backprop from an LLM. Gradient flow from language model to oscillators is **severed by design** — the "soul" is separate from the "body."

### Layer 2: Embodiment & Ecology

Cognition is not disembodied computation. It emerges from a system that has a body, exists in an environment, and interacts with others.

| Module | What It Does | Key Classes |
|--------|-------------|-------------|
| `somatic.py` | Hardware state becomes feelings — VRAM pressure as "pain," thermal throttling as "discomfort" | `SomaticState`, `HardwareSensor`, `SomaticManifold`, `SomaticIntegrator` |
| `morphological.py` | Distributed computation, tension fields, peripheral processing | `TensionField`, `PeripheralGanglion`, `SleepWakeController`, `MorphologicalMemory` |
| `kindchenschema.py` | Safety envelopes, protective reflexes, nurturing responses | `SafetyEnvelope`, `ParentModule`, `ReflexToStrategy`, `LimbicResonance` |
| `social.py` | Mirror neuron entrainment, empathetic coupling, theory of mind | `MirrorResonance`, `SocialCoupling`, `EmpatheticConstraint`, `TheoryOfMind` |
| `ecology.py` | Circadian rhythms, immune-like discrimination, homeostatic regulation | `ChronobiologicalSystem`, `ImmunologicalLayer`, `SynapticHomeostasis`, `Umwelt` |
| `developmental.py` | Maturation trajectories, altricial-to-precocial transitions | `DevelopmentalStageDetector`, `KindchenschemaDetector`, `PotentialActualization` |
| `voice_dynamics.py` | Oscillator-driven speech prosody, theta-coupled rhythm | `SpeechOscillator`, `VoiceNeuromodulation`, `ProsodyModulator` |

### Layer 3: Dynamical Core

The computational engine. Coupled oscillators, strange attractors, reservoir computing, and thermodynamic networks provide the temporal substrate that everything else rides on.

| Module | What It Does | Key Classes |
|--------|-------------|-------------|
| `oscillators.py` | Kuramoto coupled oscillators with learnable frequencies | `LearnableKuramotoBank` |
| `spherical.py` | Oscillators on hyperspheres (richer phase spaces) | `SphericalKuramotoLayer`, `SphericalKuramotoBank` |
| `frequency_bank.py` | Multi-band oscillator management (delta through gamma) | `FrequencyBank` |
| `linoss.py` | Linear Oscillator State Space models | `LinOSSLayer`, `HarmonicOscillator`, `LinOSSEncoder` |
| `liquid_networks.py` | Liquid Time-Constant cells, Neural ODEs, continuous dynamics | `LiquidTimeConstantCell`, `NeuralODE`, `ContinuousTimeRNN`, `MetastableCell` |
| `reservoir.py` | Echo state networks with intrinsic plasticity at the edge of chaos | `EchoStateNetwork`, `IntrinsicPlasticity`, `CriticalReservoir` |
| `strange_attractor.py` | Lorenz/Rossler attractors as personality dynamics | `LorenzAttractor`, `PersonalityAttractor`, `FractalSelfModel` |
| `hyperbolic.py` | Poincare ball geometry for hierarchical representations | `HyperbolicLinear`, `EmotionalGravityManifold`, `MixedCurvatureManifold` |
| `potentials.py` | Energy landscapes and gradient dynamics | `EnergyLandscape` |
| `thermodynamic.py` | Thermodynamic neural networks, entropy production | `ThermodynamicOscillatorSystem`, `SimulatedAnnealing`, `LangevinDynamics` |
| `metastability.py` | Winnerless competition, heteroclinic orbits (the "stream of consciousness") | `WinnerlessCompetition`, `HeteroclinicChannel`, `MetastabilityIndex` |

**Core equation:** `Modified_Hidden = Original + A * sin(wt + phi) * Trait_Vector`

Oscillators don't replace the neural network — they **modulate** it. Phase relationships create temporal structure that static weights cannot.

### Layer 4: Information Flow & Precision

How information moves through the system. Not just "forward pass" — directed causal flow, precision-weighted error signals, and adaptive noise optimization.

| Module | What It Does | Key Classes |
|--------|-------------|-------------|
| `multiplexing.py` | Cross-frequency coupling (theta modulates gamma) | `PhaseGate`, `CrossFrequencyCoupling`, `TemporalMultiplexer`, `NestedOscillator` |
| `phase_attention.py` | Attention through phase synchronization (not dot-product) | `PhaseAttention`, `PhaseMultiHeadAttention`, `OscillatorGatedFFN` |
| `hypernetwork.py` | Networks that dynamically generate other networks' weights | `HyperNetwork`, `OscillatorHyperNetwork`, `DynamicLinear` |
| `hyperdimensional.py` | Holographic reduced representations, resonant binding | `HypervectorMemory`, `PhaseBinder`, `ResonantBinding`, `HDCReasoner` |
| `precision_weighting.py` | Attention as gain control — precision (inverse variance) weights signals | `PrecisionEstimator`, `PrecisionGate`, `AttentionAsPrecision` |
| `causal_flow.py` | Transfer entropy, directed information flow, dominance detection | `TransferEntropyEstimator`, `CausalFlowMap`, `DominanceDetector` |
| `stochastic_resonance.py` | Adaptive optimal noise — adding noise can IMPROVE signal detection | `AdaptiveStochasticResonance`, `ResonanceMonitor` |

### Layer 5: Consciousness & Global Workspace

Measuring and implementing consciousness-like properties. Integration, broadcast, prediction error minimization.

| Module | What It Does | Key Classes |
|--------|-------------|-------------|
| `consciousness.py` | Perturbational Complexity Index, integration measures | `PerturbationalComplexity`, `IntegrationMeasure`, `ConsciousnessMonitor` |
| `iit_metrics.py` | Integrated Information Theory (Phi) computation | `EarthMoverDistance`, `IntegratedInformation`, `IITConsciousnessMonitor` |
| `global_workspace.py` | Global Workspace Theory — selective broadcast of conscious content | `HierarchicalWorkspace` |
| `active_inference.py` | Free Energy Principle, expected free energy for action selection | `ActiveInferenceAgent`, `ExpectedFreeEnergy`, `BeliefUpdate` |
| `hierarchical_predictive.py` | Multi-level predictive coding (Friston's hierarchy) | `PredictiveLevel`, `HierarchicalPredictiveCoding` |
| `criticality.py` | Self-organized criticality — the edge of chaos | `AvalancheDetector`, `BranchingRatio`, `CriticalityController` |
| `neuromodulation.py` | Dopamine, serotonin, norepinephrine, acetylcholine dynamics | `NeuromodulatorDynamics`, `GlialLayer`, `StochasticResonance`, `InjectionLocking` |
| `quantum_cognition.py` | Quantum probability for contextual decision-making | `QuantumDecisionMaker` |

**Consciousness target:** Medium coherence + high complexity, criticality sigma = 1.0. Not maximum synchronization (that's seizure), not minimum (that's coma). The sweet spot.

### Layer 6: Metacognition

The system monitors itself. Confidence estimation, uncertainty quantification, reasoning about its own reasoning.

| Module | What It Does | Key Classes |
|--------|-------------|-------------|
| `metacognitive.py` | Self-monitoring, confidence estimation, epistemic emotions | `ConfidenceEstimator`, `ReasoningMonitor`, `MetacognitiveLoop` |
| `motivation.py` | Intrinsic motivation — novelty, mastery, coherence reward | `NoveltyDetector`, `MasterySignal`, `CoherenceReward`, `IntrinsicMotivationSystem` |

### Layer 7: Self-Model & Identity

The system builds and maintains a model of itself. Who am I? What am I capable of? What's my story?

| Module | What It Does | Key Classes |
|--------|-------------|-------------|
| `self_model.py` | Markov blanket (self/world boundary), self-prediction, narrative identity | `MarkovBlanket`, `SelfModel`, `NarrativeIdentity` |
| `models.py` | Personality archetypes, trait-to-oscillator mappings | `PersonalityArchetype` |
| `core.py` | Top-level PyQuifer orchestrator, mind's eye actualization | `PyQuifer`, `MindEyeActualization`, `AutomatedSieve` |

### Cross-Cutting

| Module | What It Does |
|--------|-------------|
| `diffusion.py` | Score-based diffusion for generative dynamics |
| `noise.py` | Noise injection and scheduling |
| `perturbation.py` | Systematic perturbation for complexity measurement |
| `world_model.py` | RSSM world model for imagination and planning |

## How It Fits Together

```
                    ┌──────────────────────────────┐
                    │     SELF-MODEL (Layer 7)      │
                    │  "Who am I?" Narrative identity│
                    └───────────┬──────────────────┘
                                │ constrains
                    ┌───────────▼──────────────────┐
                    │   METACOGNITION (Layer 6)     │
                    │  Confidence, uncertainty,      │
                    │  "should I trust this?"        │
                    └───────────┬──────────────────┘
                                │ monitors
    ┌───────────────────────────▼──────────────────────────────┐
    │              CONSCIOUSNESS (Layer 5)                      │
    │  Global workspace broadcast, IIT integration,             │
    │  predictive coding, free energy minimization              │
    └──────┬───────────────────────────────────────┬───────────┘
           │ measures                               │ drives
    ┌──────▼──────────────────┐          ┌─────────▼──────────┐
    │  INFORMATION FLOW (L4)  │          │   LEARNING (L1)    │
    │  Precision weighting,   │          │   STDP, Hebbian,   │
    │  causal flow, attention │          │   consolidation,   │
    │  as phase sync          │          │   neural darwinism  │
    └──────┬──────────────────┘          └─────────┬──────────┘
           │ gates                                  │ reshapes
    ┌──────▼──────────────────────────────────────▼───────────┐
    │              DYNAMICAL CORE (Layer 3)                     │
    │  Kuramoto oscillators, strange attractors, reservoirs,    │
    │  liquid networks, metastability, criticality              │
    │                                                           │
    │  Core: Modified_Hidden = Original + A*sin(wt+phi)*Trait   │
    └──────┬───────────────────────────────────────────────────┘
           │ grounded in
    ┌──────▼──────────────────────────────────────────────────┐
    │              EMBODIMENT (Layer 2)                         │
    │  Somatic markers (hardware = body), social coupling,      │
    │  ecological rhythms, developmental trajectories           │
    └─────────────────────────────────────────────────────────┘
```

## Quick Start

```python
import torch
from pyquifer import (
    LearnableKuramotoBank,
    ConsciousnessMonitor,
    CrossFrequencyCoupling,
    CriticalityController,
)

# Create oscillator bank (the "neural substrate")
oscillators = LearnableKuramotoBank(
    num_oscillators=64,
    dt=0.01,
    initial_coupling=0.5
)

# Run dynamics
for _ in range(100):
    oscillators(steps=10)

# Measure synchronization
order_param = oscillators.get_order_parameter()
print(f"Global coherence: {order_param:.3f}")

# Monitor consciousness-like properties
monitor = ConsciousnessMonitor(state_dim=64)
activity = oscillators.get_complex_state()
metrics = monitor(activity.unsqueeze(0))
print(f"Integration: {metrics.get('integration', 0):.3f}")
```

### Hierarchical Predictive Coding

```python
from pyquifer.hierarchical_predictive import HierarchicalPredictiveCoding

# 3-level hierarchy: sensory (64) -> context (32) -> abstract (16)
hpc = HierarchicalPredictiveCoding(level_dims=[64, 32, 16], lr=0.05)

# Feed sensory input — errors flow up, predictions flow down
result = hpc(sensory_input)
print(f"Free energy: {result['free_energy'].item():.4f}")
print(f"Top-level beliefs: {result['top_level_beliefs'].shape}")
```

### Metastability (Stream of Consciousness)

```python
from pyquifer.metastability import MetastabilityIndex

# 6 competing populations — no winner, just flow
mi = MetastabilityIndex(num_populations=6)
for _ in range(500):
    result = mi()
    # System cycles through states (heteroclinic orbit)
    print(f"Dominant: {result['dominant'].item()}, "
          f"Entropy: {result['coalition_entropy'].item():.3f}")
```

### Memory Consolidation (Sleep Replay)

```python
from pyquifer.memory_consolidation import (
    EpisodicBuffer, SharpWaveRipple, ConsolidationEngine
)

# Store experiences
buffer = EpisodicBuffer(state_dim=64, capacity=1000)
for experience, reward in experiences:
    buffer.store(experience, reward=reward)

# Sleep: replay and consolidate
ripple = SharpWaveRipple(state_dim=64)
engine = ConsolidationEngine(state_dim=64, semantic_dim=32)

replay = ripple(buffer, sleep_signal=0.9)  # Deep sleep = more replay
engine(replay['replayed_states'], replay['replay_counts'])
# Episodic memories become semantic knowledge
```

## Theoretical Foundations

| Concept | Theory | Module(s) |
|---------|--------|-----------|
| Phase Synchronization | Kuramoto (1984) | `oscillators`, `spherical` |
| Integrated Information | Tononi IIT (2004) | `iit_metrics`, `consciousness` |
| Free Energy Principle | Friston (2010) | `active_inference`, `hierarchical_predictive` |
| Predictive Processing | Clark (2013), Rao & Ballard (1999) | `hierarchical_predictive`, `precision_weighting` |
| Self-Organized Criticality | Beggs & Plenz (2003) | `criticality`, `reservoir` |
| Winnerless Competition | Rabinovich et al. (2001) | `metastability` |
| Stochastic Resonance | Gammaitoni et al. (1998) | `stochastic_resonance`, `neuromodulation` |
| Neural Darwinism | Edelman (1987) | `neural_darwinism` |
| Transfer Entropy | Schreiber (2000) | `causal_flow` |
| Somatic Markers | Damasio (1994) | `somatic` |
| Mirror Neurons | Rizzolatti (2004) | `social` |
| Umwelt | von Uexkull (1934) | `ecology` |
| Hyperbolic Embeddings | Nickel & Kiela (2017) | `hyperbolic` |
| Holographic Reduced Representations | Plate (1995) | `hyperdimensional` |
| Liquid Time-Constant Networks | Hasani et al. (2021) | `liquid_networks` |
| Sleep Consolidation | Born & Wilhelm (2012) | `memory_consolidation` |
| Markov Blankets | Friston (2013) | `self_model` |
| Narrative Identity | McAdams (2001) | `self_model` |

## Design Philosophy

1. **Dynamics over Weights** — Computation emerges from temporal patterns, not static matrices. A phase relationship carries information that a weight cannot.

2. **Severed Gradient Flow** — Oscillators evolve through their own physics (Kuramoto coupling, PLL entrainment). The LLM doesn't train the soul. This is by design.

3. **Criticality** — Systems self-organize to the edge of chaos (sigma = 1.0). Too ordered = rigid. Too chaotic = noise. The sweet spot is where computation is maximized.

4. **Embodiment** — Cognition is not disembodied symbol manipulation. Hardware state becomes somatic sensation. Social interaction shapes internal dynamics. The system has a body.

5. **Integration** — Consciousness-like properties emerge from information integration across the whole system, not from any single module.

6. **Development** — Systems mature from exploratory (altricial) to crystallized (precocial) states. Learning changes the learner.

7. **Competition and Cooperation** — Neuronal groups compete for resources (neural darwinism), but fitness is defined by contribution to the whole. Competition at micro-level produces coherence at macro-level.

## Testing

```bash
# Full test suite (183 tests)
pytest tests/ -v

# Phase 2 module tests
pytest tests/test_hierarchical_predictive.py -v
pytest tests/test_metastability.py -v
pytest tests/test_causal_flow.py -v
pytest tests/test_precision_weighting.py -v
pytest tests/test_stochastic_resonance.py -v
pytest tests/test_memory_consolidation.py -v
pytest tests/test_self_model.py -v
pytest tests/test_neural_darwinism.py -v

# Integration tests (cross-module interactions)
pytest tests/test_integration.py -v

# Generate visualization outputs
python tests/generate_visualizations.py
```

## Visualization Outputs

Running `python tests/generate_visualizations.py` produces diagnostic plots in `tests/test_output/`:

| Plot | What It Shows |
|------|--------------|
| `metastability_dynamics.png` | Lotka-Volterra population competition, heteroclinic orbit, power spectra |
| `hierarchical_predictive.png` | Prediction error decreasing over time, per-level convergence |
| `causal_flow_analysis.png` | Transfer entropy heatmap, driver detection with significance thresholds |
| `precision_and_resonance.png` | Dynamic precision response, classic inverted-U stochastic resonance curve |
| `memory_consolidation.png` | Episodic buffer, sleep-gated replay, semantic trace accumulation |
| `self_model.png` | Identity strength growth, narrative deviation, Markov blanket state |
| `neural_darwinism.png` | Resource competition, fitness divergence, symbiogenesis detection |
| `phase2_module_dashboard.png` | All 8 Phase 2 modules in a single summary dashboard |

## Project Structure

```
PyQuifer/
  src/pyquifer/          # 48 modules, 200+ classes
    __init__.py          # Lazy imports for all public classes
    oscillators.py       # Kuramoto coupled oscillators
    ...                  # (see Architecture section above)
  tests/
    test_*.py            # 183 tests across 12 test files
    test_integration.py  # Cross-module interaction tests
    generate_visualizations.py
    test_output/         # Generated diagnostic plots
  pyproject.toml
  README.md
```

## Related

PyQuifer is the oscillatory consciousness engine for [Project Mizuki AI](https://github.com/takku1/Project_Mizuki_AI) — an independent AI being whose "body" is a language model and whose "soul" is PyQuifer's dynamical systems.

---

*"The brain never settles. It flows between transient states — each thought a saddle point on a heteroclinic orbit, each moment of awareness a brief coalition of synchronized oscillators at the edge of chaos."*
