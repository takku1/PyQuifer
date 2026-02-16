# PyQuifer Source Reorganization Plan

Last updated: February 15, 2026
Scope: Structural reorganization, refactoring, and optimization plan for `src/pyquifer`. All changes must preserve existing functionality — every public class, method signature, and behavioral contract is retained.

## 1. Audit Summary

- Modules audited: 84 Python files under `src/pyquifer` (48,801 total lines)
- Symbols detected: 364 classes, 47 top-level functions/defs
- Tests: 98 passing (74 core + 8 metabolic + 16 neuroscience validation)
- Largest concentration points:
  - `integration.py` (2986 lines) ← primary split target
  - `oscillators.py` (1541 lines)
  - `deep_active_inference.py` (1297 lines)
  - `__init__.py` (1055 lines)
  - `criticality.py` (1035 lines)
  - `bridge.py` (1027 lines)
  - `global_workspace.py` (1017 lines)

High-level read:
- PyQuifer has strong conceptual coverage and many usable primitives.
- The main maintainability risk is structural sprawl, not missing ideas.
- API discoverability and naming coherence are now the biggest bottlenecks.

## 2. Key Structural Issues

1. Orchestration monolith:
- `integration.py` is carrying too many responsibilities (runtime loop, optional features, wiring, diagnostics, policy logic).

2. Public API overexpansion:
- `__init__.py` exposes a very large flat surface through `__all__` and lazy imports.
- This makes stability, discoverability, and versioning harder.

3. Domain overlap and ambiguity:
- `models.py` and `world_model.py` represent different concepts but have similar naming.
- `precision_weighting.py` and `global_workspace.PrecisionWeighting` coexist with overlapping semantics.
- `stochastic_resonance.py` and `neuromodulation.StochasticResonance` create conceptual duplication.

4. Naming consistency issues:
- Mixed styles (`OCC_Model`, `Kindchenschema`, `FlashLTC`, `LinOSS`, etc.).
- Historical aliases in exports (example mismatch risk: names exported vs class identifiers in module).

5. Experimental vs production pathways are interleaved:
- Stable runtime pieces (bridge/cycle/workspace) and experimental modules (quantum, exotic attractors, some social/developmental components) live in one namespace tier.

## 3. Proposed Target Package Layout

```text
pyquifer/
  api/
    bridge.py              ← bridge.py (PyQuiferBridge, ModulationState, SteppedModulator)
    presets.py             ← CycleConfig presets (interactive, realtime, neuroscience)
    legacy.py              ← core.py (PyQuifer class — backward compat entrypoint)
  runtime/
    cycle.py               ← integration.py CognitiveCycle class (~2400 lines)
    tick_result.py         ← TickResult NamedTuple, PROCESSING_MODE_* constants
    config.py              ← CycleConfig dataclass + all field defaults
    routing.py             ← workspace/organ/ensemble competition wiring
    criticality_feedback.py ← _criticality_feedback() standalone (torch.compile target)
    diagnostics.py         ← diagnostics dict assembly, metabolic_info, neuro_metrics
  workspace/
    workspace.py           ← GlobalWorkspace facade (ContentType, WorkspaceItem)
    competition.py         ← SalienceComputer, IgnitionDynamics, CompetitionDynamics, PrecisionWeighting
    broadcast.py           ← GlobalBroadcast, StandingBroadcast
    organ_base.py          ← Proposal, Organ, OscillatoryWriteGate, PreGWAdapter
    organ_builtin.py       ← HPCOrgan, MotivationOrgan, SelectionOrgan
    organ_mcp.py           ← MCPOrganConfig, MCPOrgan
    ensemble.py            ← CrossBleedGate, WorkspaceEnsemble, DiversityTracker, HierarchicalWorkspace
  dynamics/
    oscillators/
      kuramoto.py          ← Snake, LearnableKuramotoBank, SensoryCoupling, _rk4_step
      spherical.py         ← SphericalKuramotoBank, SphericalKuramotoLayer
      complex.py           ← ComplexKuramotoBank, ComplexCoupling, ModReLU
      stuart_landau.py     ← StuartLandauOscillator
      mean_field.py        ← KuramotoDaidoMeanField, PhaseTopologyCache
      frequency_bank.py    ← FrequencyBank (unchanged)
      coupling.py          ← CrossFrequencyCoupling, NestedOscillator (from multiplexing.py)
    continuous/
      ode_solvers.py       ← all ODE solver classes (unchanged)
      liquid.py            ← LiquidTimeConstantCell, ContinuousTimeRNN, MetastableCell
      neural_mass.py       ← WilsonCowanPopulation, WilsonCowanNetwork (unchanged)
      thermodynamic.py     ← LangevinDynamics, SimulatedAnnealing, PhaseTransitionDetector
      linoss.py            ← HarmonicOscillator, LinOSSLayer, LinOSSEncoder
    criticality/
      monitors.py          ← AvalancheDetector, BranchingRatio, KuramotoCriticalityMonitor, KoopmanBifurcationDetector
      controllers.py       ← CriticalityController, HomeostaticRegulator, NoProgressDetector
    spiking/
      neurons.py           ← LIFNeuron, AdExNeuron, SynapticNeuron, AlphaNeuron
      layers.py            ← SpikingLayer, OscillatorySNN, STDPLayer, SynapticDelay
      advanced.py          ← EpropSTDP, EligibilityModulatedSTDP, RewardPredictionError
      energy.py            ← EnergyOptimizedSNN, MultiCompartmentSpikingPC, EnergyLandscape
    stochastic.py          ← OrnsteinUhlenbeckNoise, AdaptiveStochasticResonance, ResonanceMonitor
    metastability.py       ← WinnerlessCompetition, HeteroclinicChannel, MetastabilityIndex
    neuromodulation.py     ← NeuromodulatorDynamics, GlialLayer, StochasticResonance, InjectionLocking
    phase_lock_bus.py      ← PhaseLockBus (unchanged — cross-modal coherence)
    basal_ganglia.py       ← BasalGangliaGating (unchanged — BG-thalamus analogue)
  learning/
    synaptic.py            ← EligibilityTrace, RewardModulatedHebbian, ContrastiveHebbian
    plasticity.py          ← DifferentiablePlasticity, OscillationGatedPlasticity, ThreeFactorRule
    stp.py                 ← TsodyksMarkramSynapse, STPLayer (short-term plasticity)
    continual.py           ← EWC, SynapticIntelligence, ContinualBackprop, MESU, ExperienceReplay
    consolidation.py       ← EpisodicBuffer, SharpWaveRipple, ConsolidationEngine, SleepReplayConsolidation
    dendritic.py           ← DendriticNeuron, DendriticStack, PyramidalNeuron, DendriticLocalizedLearning
    equilibrium_prop.py    ← EquilibriumPropagationTrainer, EPKuramotoClassifier
    prospective.py         ← ProspectiveInference, ProspectiveHebbian, InferThenModify
  cognition/
    predictive/
      hierarchical.py      ← PredictiveLevel, HierarchicalPredictiveCoding, OscillatoryPredictiveCoding
      active_inference.py  ← ActiveInferenceAgent, PredictiveEncoder/Decoder, BeliefUpdate
      deep_active_inference.py ← DeepAIF, LatentTransitionModel, PolicyNetwork, MultiStepPlanner
      jepa.py              ← JEPAEncoder, JEPAPredictor, VICRegLoss, BarlowLoss, ActionJEPA
    control/
      motivation.py        ← NoveltyDetector, MasterySignal, CoherenceReward, EpistemicValue
      metacognitive.py     ← ConfidenceEstimator, EvidenceAggregator, ReasoningMonitor
      deliberation.py      ← Deliberator, BeamSearchReasoner, SelfCorrectionLoop, ComputeBudget
      volatility.py        ← HierarchicalVolatilityFilter, VolatilityGatedLearning
    reasoning/
      causal_flow.py       ← TransferEntropyEstimator, CausalFlowMap, DominanceDetector
      causal_reasoning.py  ← CausalGraph, DoOperator, CounterfactualEngine
      graph_reasoning.py   ← DynamicGraphAttention, MessagePassingWithPhase, TemporalGraphTransformer
      temporal_graph.py    ← TemporalKnowledgeGraph, EventTimeline, TemporalReasoner
    attention/
      phase_attention.py   ← PhaseAttention, PhaseMultiHeadAttention, OscillatorGatedFFN
      precision_weighting.py ← PrecisionEstimator, PrecisionGate, AttentionAsPrecision
    binding/
      visual.py            ← AKOrNLayer, AKOrNBlock, AKOrNEncoder, OscillatorySegmenter
      temporal.py          ← SequenceAKOrN, PhaseGrouping, OscillatoryChunking
      sensory.py           ← MultimodalBinder, BindingStrength, CrossModalAttention
    routing/
      ssm.py               ← SelectiveStateSpace, MambaLayer, OscillatorySSM
      moe.py               ← ExpertPool, OscillatorRouter, SparseMoE
      flash_rnn.py         ← FlashLTC, FlashCfC, FlashLTCLayer
  memory/
    gated_memory.py        ← NMDAGate, DifferentiableMemoryBank, MemoryConsolidationLoop
    cls.py                 ← HippocampalModule, NeocorticalModule, ForgettingCurve
    generative_world_model.py ← GenerativeWorldModel (from models.py)
    latent_world_model.py  ← RSSM, WorldModel, ImaginationBasedPlanner (from world_model.py)
  embodiment/
    somatic.py             ← SomaticState, SomaticManifold, SomaticIntegrator
    morphology.py          ← TensionField, MorphologicalMemory, DistributedBody
    ecology.py             ← ChronobiologicalSystem, AgencyMaintenance, EcologicalSystem
    social.py              ← MirrorResonance, TheoryOfMind, EmpatheticConstraint
    developmental.py       ← KindchenschemaDetector, DevelopmentalStageDetector, PotentialActualization
    safety.py              ← SafetyEnvelope, SupervisoryModule, ReflexToStrategy (from kindchenschema.py)
    voice.py               ← SpeechOscillator, ProsodyModulator, VoiceDynamicsSystem
  identity/
    self_model.py          ← MarkovBlanket, SelfModel, NarrativeIdentity
    strange_attractor.py   ← PersonalityAttractor, FractalSelfModel, MultiScaleFractalPersonality
    consciousness.py       ← PerturbationalComplexity, ConsciousnessMonitor, IIT metrics
    adapter_manager.py     ← AdapterConfig, AdapterGate, AdapterManager
    hypernetwork.py        ← HyperNetwork, OscillatorHyperNetwork, ContextualReservoir
    neural_darwinism.py    ← SelectionArena, SymbiogenesisDetector, SpeciatedSelectionArena
  diagnostics/
    neuroscience.py        ← spectral_exponent, dfa_exponent, lempel_ziv_complexity, etc.
    profiling.py           ← bridge.profile_step() helpers, sync_debug_mode
  accel/
    cuda/
      kuramoto_kernel.py   ← KuramotoCUDAKernel, TensorDiagnostics (unchanged)
  experimental/
    quantum.py             ← QuantumState, QuantumDecisionMaker, QuantumMemory
    attractors.py          ← LorenzAttractor (from strange_attractor.py)
    hyperdimensional.py    ← HypervectorMemory, HDCReasoner, FHRREncoder
    hyperbolic.py          ← HyperbolicOperations, EmotionalGravityManifold
    reservoir.py           ← EchoStateNetwork, CriticalReservoir
```

**Coverage:** Every class from the 84-module inventory is accounted for above. No class is dropped.
**Rule:** If a module has < 100 lines and a single class, it merges into the nearest thematic file rather than getting its own file.

Design rule:
- `api` and `runtime` are stability-first.
- `experimental` is opt-in and explicitly non-stable.

## 4. Module Migration Map (Complete)

### 4.1 Files that SPLIT (monolith decomposition)

| Current file | Split into | Lines saved |
|---|---|---|
| `integration.py` (2986) | `runtime/cycle.py` + `runtime/config.py` + `runtime/tick_result.py` + `runtime/criticality_feedback.py` + `runtime/diagnostics.py` | ~600 from cycle.py |
| `oscillators.py` (1541) | `dynamics/oscillators/kuramoto.py` + `stuart_landau.py` + `mean_field.py` | ~450 from kuramoto.py |
| `global_workspace.py` (1017) | `workspace/competition.py` + `broadcast.py` + `ensemble.py` + `workspace.py` | ~0 (pure split) |
| `criticality.py` (1035) | `dynamics/criticality/monitors.py` + `controllers.py` | ~0 (pure split) |
| `spiking.py` (911) | `dynamics/spiking/neurons.py` + `layers.py` | ~0 (pure split) |
| `organ.py` (370) | `workspace/organ_base.py` + `organ_builtin.py` | ~0 (pure split) |

### 4.2 Files that MOVE (rename only, no content change)

| Current | New location | Reason |
|---|---|---|
| `core.py` | `api/legacy.py` | Marks as legacy entrypoint |
| `bridge.py` | `api/bridge.py` | Canonical external interface |
| `mcp_organ.py` | `workspace/organ_mcp.py` | First-class workspace extension |
| `models.py` | `memory/generative_world_model.py` | Disambiguate from world_model.py |
| `world_model.py` | `memory/latent_world_model.py` | Clarifies RSSM/planner role |
| `neuro_diagnostics.py` | `diagnostics/neuroscience.py` | Better discoverability |
| `kindchenschema.py` | `embodiment/safety.py` | Removes obscure naming |
| `basal_ganglia.py` | `dynamics/basal_ganglia.py` | Groups with dynamics |
| `phase_lock_bus.py` | `dynamics/phase_lock_bus.py` | Groups with dynamics |
| `self_model.py` | `identity/self_model.py` | Groups with identity |
| `strange_attractor.py` | `identity/strange_attractor.py` | Personality attractor → identity |
| `consciousness.py` | `identity/consciousness.py` | Groups with identity |
| `neural_darwinism.py` | `identity/neural_darwinism.py` | Selection arenas → identity |
| `adapter_manager.py` | `identity/adapter_manager.py` | LoRA adapters → identity modulation |
| `hypernetwork.py` | `identity/hypernetwork.py` | Oscillator-driven weight modulation |
| `quantum_cognition.py` | `experimental/quantum.py` | Experimental, not in hot path |
| `hyperdimensional.py` | `experimental/hyperdimensional.py` | Experimental |
| `hyperbolic.py` | `experimental/hyperbolic.py` | Experimental |
| `reservoir.py` | `experimental/reservoir.py` | Experimental |
| `fhrr.py` | `experimental/hyperdimensional.py` (merge) | Same domain |

### 4.3 Files that MERGE (small files → thematic groupings)

| Current files | Merge into | Reason |
|---|---|---|
| `dendritic.py` + `dendritic_learning.py` | `learning/dendritic.py` | Same concept, 236+325=561 lines |
| `advanced_spiking.py` | `dynamics/spiking/advanced.py` | Groups with spiking |
| `energy_spiking.py` | `dynamics/spiking/energy.py` | Groups with spiking |
| `short_term_plasticity.py` | `learning/stp.py` | Short file (209 lines) |
| `multiplexing.py` | `dynamics/oscillators/coupling.py` | Cross-frequency coupling → oscillator domain |
| `noise.py` (8 lines, empty) | DELETE | Dead file |
| `diffusion.py` + `perturbation.py` + `potentials.py` | Keep in `memory/` near generative_world_model | Only used by GenerativeWorldModel |

### 4.4 Files that STAY (already well-placed or standalone)

| File | Reason to keep |
|---|---|
| `frequency_bank.py` | Clean, 163 lines, single responsibility |
| `ode_solvers.py` | Clean, well-structured |
| `neural_mass.py` | Clean, 308 lines |
| `flash_rnn.py` | Clean, 309 lines |
| `selective_ssm.py` | Clean, 385 lines |
| `oscillatory_moe.py` | Clean, 326 lines |
| `appraisal.py` | Clean, 326 lines |
| `causal_reasoning.py` | Clean, standalone |
| `graph_reasoning.py` | Clean, standalone |
| `temporal_graph.py` | Clean, standalone |
| `cls_memory.py` | Clean, 500 lines |
| `gated_memory.py` | Clean, 335 lines |
| `jepa.py` | Clean, 361 lines |
| `deep_active_inference.py` | Large (1297) but cohesive — don't split |
| `iit_metrics.py` | Large (1003) but cohesive — don't split |

## 5. Class and Method Naming Policy

### 5.1 Naming standard
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Acronyms in classes: normalize (`OCCModel`, `LTCCell`, `MCPOrgan`)

### 5.2 Specific cleanup candidates
- `OCC_Model` -> `OCCModel`
- Keep `MCPOrgan` as-is (already good)
- Normalize any legacy exports that diverge from class names in-source
- Reserve suffixes:
  - `*Monitor` for read-only metrics
  - `*Controller` for closed-loop control
  - `*Adapter` for interface/protocol bridges
  - `*Engine` for multi-step orchestrators

### 5.3 Public API curation
- Move from one giant flat export list to:
  - Stable: `pyquifer.api.*`
  - Advanced: submodule imports
  - Experimental: explicit `pyquifer.experimental.*`

## 6. Runtime-Oriented Refactor Priority

Priority order for maintainability (without changing behavior):

1. Split `integration.py` into runtime components:
- `tick_result.py` (return contracts)
- `config.py` (all presets and flags)
- `cycle.py` (execution loop only)
- `routing.py` (workspace/organ/ensemble competition)

2. Split `global_workspace.py` into:
- competition primitives
- broadcast/ignition
- ensemble and cross-bleed

3. Reduce `__init__.py` surface:
- keep only stable, high-traffic symbols
- route long-tail symbols to submodule imports

4. Separate stable vs experimental namespaces:
- move speculative modules to `experimental/`
- keep bridge/cycle/workspace/dynamics core in stable paths

## 7. Backward Compatibility Strategy

- Keep import shims for one full minor version.
- Emit deprecation warnings with exact replacement import paths.
- Maintain a migration table in `docs/README.md` and changelog.
- Add tests asserting old imports still resolve during transition.

Example shim pattern:
```python
# old module path
from pyquifer.runtime.cycle import CognitiveCycle  # new source
```

## 8. Risk Register

1. Risk: namespace churn breaks user scripts
- Mitigation: explicit shims + deprecation window + migration docs

2. Risk: hidden coupling in `integration.py`
- Mitigation: behavior lock tests before and after file splits

3. Risk: performance regressions from refactor
- Mitigation: keep latency benchmark suite as gating check

4. Risk: conceptual drift in neuroscience claims
- Mitigation: keep neuroscience diagnostics and target envelopes under CI

## 9. Phase Plan (Documentation-Level)

### Phase 1: API and runtime boundaries
- Define stable namespaces and import policy
- Split runtime monolith layout on paper first

### Phase 2: Workspace and organ boundary cleanup
- Separate protocol, built-in organs, and MCP organ integration
- Isolate ensemble/cross-bleed internals

### Phase 3: Dynamics and learning decomposition
- Split high-line-count dynamics files by concern
- Stabilize naming conventions

### Phase 4: Experimental segregation
- Move speculative modules under explicit experimental namespace
- Keep stable core minimal and well-documented

## 10. Refactoring Plan: `integration.py` (2986 lines → 4 files)

**Goal:** Split the monolith without changing any public API or behavior.

### 10.1 Extract `runtime/config.py` (~350 lines)
Move out of `integration.py`:
- `TickResult` NamedTuple (lines ~55-75)
- `CycleConfig` dataclass (lines ~77-220) including all presets (`interactive()`, `realtime()`, `neuroscience()`)
- `PROCESSING_MODE_*` constants

**Zero-loss check:** `from pyquifer.integration import CycleConfig, TickResult` stays valid via re-export.

### 10.2 Extract `runtime/criticality_feedback.py` (~50 lines)
Move out:
- `_criticality_feedback()` standalone function (already extracted for torch.compile)

### 10.3 Extract `runtime/diagnostics.py` (~200 lines)
Move out:
- The diagnostics dict assembly block (currently lines ~2400-2490)
- Neuro-metrics computation calls
- `metabolic_info` assembly

### 10.4 Remaining `runtime/cycle.py` (~2400 lines)
- `CognitiveCycle` class stays here but is ~15% lighter
- All `tick()` wiring stays intact — no step reordering

### 10.5 Split validation
- Before split: run `pytest tests/ -x` → capture pass count
- After split: same command → same pass count + same latency benchmarks
- Add import shim test: `test_import_compat.py` asserting `from pyquifer.integration import CognitiveCycle, CycleConfig, TickResult` still works

---

## 11. Refactoring Plan: `oscillators.py` (1541 lines → 3 files)

### 11.1 Extract `dynamics/oscillators/stuart_landau.py` (~200 lines)
- `StuartLandauOscillator` class
- Self-contained, no cross-dependencies back to Kuramoto

### 11.2 Extract `dynamics/oscillators/mean_field.py` (~250 lines)
- `KuramotoDaidoMeanField` class
- `PhaseTopologyCache` class

### 11.3 Remaining `dynamics/oscillators/kuramoto.py` (~1100 lines)
- `Snake` activation
- `LearnableKuramotoBank` (core)
- `SensoryCoupling`
- `_rk4_step`

---

## 12. Refactoring Plan: `global_workspace.py` (1017 lines → 3 files)

### 12.1 `workspace/competition.py` (~350 lines)
- `SalienceComputer`, `IgnitionDynamics`, `CompetitionDynamics`, `PrecisionWeighting`

### 12.2 `workspace/broadcast.py` (~200 lines)
- `GlobalBroadcast`, `StandingBroadcast`

### 12.3 `workspace/ensemble.py` (~350 lines)
- `CrossBleedGate`, `WorkspaceEnsemble`, `DiversityTracker`, `HierarchicalWorkspace`

### 12.4 `workspace/workspace.py` (~120 lines)
- `ContentType`, `WorkspaceItem`, `GlobalWorkspace` (facade re-exporting from above)

---

## 13. Optimization Plan (Hot-Path Performance)

### 13.1 Eliminate Python loops in tick()

**Target locations in `integration.py` tick():**

| Location | Current pattern | Replacement | Est. speedup |
|---|---|---|---|
| Organ proposal collection (~line 1700) | `for organ in self._organs: proposals.append(organ.propose(...))` | Batch `torch.stack` + vectorized salience sort | ~0.1ms |
| Workspace competition (~line 1750) | `for item in proposals: if item.salience > threshold` | `torch.topk` on stacked salience tensor | ~0.05ms |
| Diagnostics dict build (~line 2400) | 50+ `.item()` calls for dict values | Tensor-only diagnostics path (already partially done via `TickResult`) | ~0.5ms on CUDA |
| R-history buffer append (~line 1400) | `self._R_history[self._R_ptr] = R_val` with Python int ptr | Keep as-is (already O(1) ring buffer) | N/A |

### 13.2 Vectorize neuro_diagnostics.py

**Current:** `lempel_ziv_complexity()` uses pure-Python substring search (O(n²) inner loop).
**Target:** Keep pure-Python for correctness but add optional `_lz_fast()` using numpy bit-packing + sliding window.

**Current:** `avalanche_statistics()` iterates Python list with `for i in range(T)`.
**Target:** Replace with `torch.where` for threshold detection + `torch.diff` for run-length encoding.

**Current:** `complexity_entropy()` uses `itertools.permutations` + dict counting.
**Target:** Replace with `torch.argsort` + tuple hashing via tensor ops for the ordinal pattern counting.

### 13.3 Reduce `.item()` calls in CUDA path

**Inventory of `.item()` in tick() hot path:** ~30 calls for diagnostics dict.
**Plan:**
1. When `return_diagnostics=False`, skip all `.item()` calls entirely (already partially done)
2. When `return_diagnostics=True` on CUDA, batch all scalar tensors into one `torch.stack([...]).cpu()` call, then unpack — single sync point instead of 30

### 13.4 torch.compile coverage expansion

**Currently compiled:** 12 submodules via `cycle.compile_modules()`
**Expand to:**
- `_criticality_feedback()` (already standalone)
- `SensoryCoupling.forward()` (fixed `.item()` graph break)
- `PrecisionWeighting.forward()` (pure tensor math)
- `SalienceComputer.forward()` (pure tensor math)

### 13.5 Memory allocation reduction

| Allocation | Current | Fix |
|---|---|---|
| `torch.zeros(...)` per tick for scratch | New tensor each tick | Preallocate in `__init__`, reuse with `.zero_()` |
| Diagnostics dict creation | New dict each tick | Preallocate dict template, `.update()` values |
| Organ proposal list | `[]` + `.append()` per tick | Preallocated fixed-size list with index counter |

---

## 14. Code Style Optimization Rules

### 14.1 Replace long if/elif chains with dispatch tables
```python
# BAD
if mode == 'perception':
    result = self._perception_step(x)
elif mode == 'imagination':
    result = self._imagination_step(x)
elif mode == 'balanced':
    result = self._balanced_step(x)

# GOOD
_MODE_DISPATCH = {
    'perception': self._perception_step,
    'imagination': self._imagination_step,
    'balanced': self._balanced_step,
}
result = _MODE_DISPATCH[mode](x)
```

### 14.2 Replace Python for-loops with vectorized ops
```python
# BAD
for i in range(n):
    output[i] = weights[i] * inputs[i]

# GOOD
output = weights * inputs  # element-wise broadcast
```

### 14.3 Replace sequential list building with comprehensions or torch.stack
```python
# BAD
results = []
for bank in self.banks:
    results.append(bank.get_phases())
all_phases = torch.cat(results, dim=0)

# GOOD
all_phases = torch.cat([b.get_phases() for b in self.banks], dim=0)
```

### 14.4 Use early returns over deep nesting
```python
# BAD
def process(x):
    if x is not None:
        if x.dim() == 2:
            if x.shape[0] > 0:
                return self._compute(x)
    return default

# GOOD
def process(x):
    if x is None or x.dim() != 2 or x.shape[0] == 0:
        return default
    return self._compute(x)
```

---

## 15. Naming Cleanup Candidates (Specific)

| Current name | New name | File | Reason |
|---|---|---|---|
| `OCC_Model` | `OCCModel` | `appraisal.py` | PascalCase convention |
| `noise.py` (8 lines, empty) | DELETE | `noise.py` | Dead file |
| `ParentModule` alias | Remove alias, use `SupervisoryModule` | `kindchenschema.py` | Clarity |
| `use_cross_freq_coupling` | Remove (keep `use_meta_r_coupling` only) | `integration.py` | Eliminate legacy alias |

---

## 16. Functionality Preservation Checklist

Before ANY refactor step, verify:

- [ ] `pytest tests/ -x --timeout=300` → same pass count (currently 98)
- [ ] `python -m pyquifer` smoke test passes
- [ ] `from pyquifer import CognitiveCycle, CycleConfig, PyQuiferBridge` works
- [ ] `from pyquifer.integration import CognitiveCycle` works (compat shim)
- [ ] Bridge latency benchmark: `bridge.profile_step()` p50 ≤ 2ms (realtime mode)
- [ ] All 349 symbols in `__all__` remain importable
- [ ] Neuroscience validation: `test_neuro_validation.py` 16/16 pass
- [ ] No class removed, no method signature changed, no return type changed

After EACH refactor step:
- [ ] Run full suite
- [ ] Run import shim test
- [ ] Verify `git diff --stat` matches expected file moves only

---

## 17. Full Module/Symbol Inventory (Read-Only Reference)

The following inventory was auto-derived from source (`class`/`def` extraction).
- `__init__.py`
  - classes: (none)
  - defs: __getattr__, __dir__
- `__main__.py`
  - classes: (none)
  - defs: smoke_test, main
- `_cuda\__init__.py`
  - classes: (none)
  - defs: (none)
- `_cuda\kuramoto_kernel.py`
  - classes: KuramotoCUDAKernel, TensorDiagnostics
  - defs: try_load_cuda_kernels, is_available
- `active_inference.py`
  - classes: PredictiveEncoder, PredictiveDecoder, TransitionModel, ExpectedFreeEnergy, BeliefUpdate, ActiveInferenceAgent
  - defs: reparameterize, kl_divergence_gaussian, constrained_activation
- `adapter_manager.py`
  - classes: AdapterConfig, BlendState, AdapterGate, AdapterManager
  - defs: (none)
- `advanced_spiking.py`
  - classes: SynapticNeuron, AlphaNeuron, RecurrentSynapticLayer, RewardPredictionError, EligibilityModulatedSTDP, EpropSTDP
  - defs: (none)
- `appraisal.py`
  - classes: AppraisalResult, EmotionState, AppraisalDimension, AppraisalChain, OCC_Model, EmotionAttribution
  - defs: (none)
- `bridge.py`
  - classes: ModulationState, PyQuiferBridge, PyQuiferLogitsProcessor, SteppedModulator
  - defs: sync_debug_mode, _interpolate_state
- `causal_flow.py`
  - classes: TransferEntropyEstimator, CausalFlowMap, DominanceDetector
  - defs: (none)
- `causal_reasoning.py`
  - classes: CausalVariable, CausalGraph, DoOperator, CounterfactualEngine, CausalDiscovery, InterventionalQuery
  - defs: (none)
- `cls_memory.py`
  - classes: MemoryTrace, ForgettingCurve, ImportanceScorer, HippocampalModule, NeocorticalModule, MemoryInterference, ConsolidationScheduler
  - defs: (none)
- `complex_oscillators.py`
  - classes: ComplexCoupling, ComplexKuramotoBank, ModReLU, ComplexLinear, ComplexBatchNorm
  - defs: to_complex, from_complex, complex_order_parameter
- `consciousness.py`
  - classes: PerturbationalComplexity, IntegrationMeasure, DifferentiationMeasure, ConsciousnessMonitor
  - defs: (none)
- `continual_learning.py`
  - classes: TaskInfo, ElasticWeightConsolidation, SynapticIntelligence, ContinualBackprop, MESU, ExperienceReplay, ContinualLearner
  - defs: (none)
- `core.py`
  - classes: PyQuifer
  - defs: (none)
- `criticality.py`
  - classes: AvalancheDetector, BranchingRatio, NoProgressDetector, KuramotoCriticalityMonitor, CriticalityController, HomeostaticRegulator, KoopmanBifurcationDetector
  - defs: phase_activity_to_spikes
- `deep_active_inference.py`
  - classes: LatentTransitionModel, PolicyNetwork, ReplayBuffer, MultiStepPlanner, DeepAIF
  - defs: (none)
- `deliberation.py`
  - classes: ProcessRewardModel, BeamSearchReasoner, SelfCorrectionLoop, ComputeBudgetAllocation, ComputeBudget, Deliberator
  - defs: (none)
- `dendritic.py`
  - classes: DendriticNeuron, DendriticStack
  - defs: (none)
- `dendritic_learning.py`
  - classes: DendriticErrorSignal, PyramidalNeuron, DendriticLocalizedLearning
  - defs: (none)
- `developmental.py`
  - classes: DynamicalSignatureDetector, KindchenschemaDetector, ProtectiveDrive, EvolutionaryAttractor, IntrinsicCuteUnderstanding, DevelopmentalStageDetector, PotentialActualization
  - defs: (none)
- `diffusion.py`
  - classes: MindEyeActualization
  - defs: (none)
- `ecology.py`
  - classes: TimeScale, ChronobiologicalSystem, ImmunologicalLayer, SynapticHomeostasis, Umwelt, AgencyMaintenance, EcologicalSystem
  - defs: (none)
- `energy_spiking.py`
  - classes: SpikingPCNeuron, EnergyOptimizedSNN, MultiCompartmentSpikingPC, EnergyLandscape
  - defs: (none)
- `equilibrium_propagation.py`
  - classes: EquilibriumPropagationTrainer, EPKuramotoClassifier
  - defs: (none)
- `fhrr.py`
  - classes: FHRREncoder, LatencyEncoder, SpikeVSAOps, NeuromorphicExporter
  - defs: (none)
- `flash_rnn.py`
  - classes: FlashLTC, FlashCfC, FlashLTCLayer
  - defs: is_flash_available
- `frequency_bank.py`
  - classes: FrequencyBank
  - defs: (none)
- `gated_memory.py`
  - classes: NMDAGate, DifferentiableMemoryBank, MemoryConsolidationLoop
  - defs: (none)
- `global_workspace.py`
  - classes: ContentType, WorkspaceItem, SalienceComputer, IgnitionDynamics, CompetitionDynamics, PrecisionWeighting, GlobalBroadcast, GlobalWorkspace, DiversityTracker, HierarchicalWorkspace, StandingBroadcast, CrossBleedGate, WorkspaceEnsemble
  - defs: (none)
- `graph_reasoning.py`
  - classes: DynamicGraphAttention, MessagePassingWithPhase, TemporalGraphTransformer
  - defs: (none)
- `hierarchical_predictive.py`
  - classes: PredictiveLevel, HierarchicalPredictiveCoding, OscillatoryPredictiveCoding
  - defs: (none)
- `hyperbolic.py`
  - classes: HyperbolicOperations, HyperbolicLinear, EmotionalGravityManifold, MixedCurvatureManifold
  - defs: (none)
- `hyperdimensional.py`
  - classes: HypervectorMemory, PhaseBinder, ResonantBinding, HDCReasoner
  - defs: circular_convolution, circular_correlation, normalize_hd
- `hypernetwork.py`
  - classes: InputEncoding, HyperNetwork, OscillatorHyperNetwork, DynamicLinear, ContextualReservoir
  - defs: encode_input
- `iit_metrics.py`
  - classes: Concept, SystemIrreducibilityAnalysis, EarthMoverDistance, InformationDensity, KLDivergence, L1Distance, PartitionedInformation, IntegratedInformation, CauseEffectRepertoire, IITConsciousnessMonitor
  - defs: generate_bipartitions, hamming_distance_matrix
- `integration.py`
  - classes: TickResult, CycleConfig, CognitiveCycle
  - defs: _criticality_feedback
- `jepa.py`
  - classes: JEPAEncoder, JEPAPredictor, VICRegLoss, BarlowLoss, ActionJEPA
  - defs: (none)
- `kindchenschema.py`
  - classes: SafetyEnvelope, SupervisoryModule, ReflexToStrategy, LimbicResonance
  - defs: (none)
- `learning.py`
  - classes: EligibilityTrace, RewardModulatedHebbian, ContrastiveHebbian, PredictiveCoding, DifferentiablePlasticity, LearnableEligibilityTrace, OscillationGatedPlasticity, ThreeFactorRule
  - defs: (none)
- `linoss.py`
  - classes: HarmonicOscillator, LinOSSLayer, LinOSSEncoder
  - defs: (none)
- `liquid_networks.py`
  - classes: LiquidTimeConstantCell, NeuralODE, ODEFunc, ContinuousTimeRNN, MetastableCell
  - defs: (none)
- `mcp_organ.py`
  - classes: MCPOrganConfig, MCPOrgan
  - defs: (none)
- `memory_consolidation.py`
  - classes: EpisodicBuffer, SharpWaveRipple, ConsolidationEngine, MemoryReconsolidation, SleepReplayConsolidation
  - defs: (none)
- `metacognitive.py`
  - classes: ConfidenceLevel, ReasoningStep, MetacognitiveState, ConfidenceEstimator, EvidenceSource, EvidenceAggregator, ReasoningMonitor, MetacognitiveLoop
  - defs: (none)
- `metastability.py`
  - classes: WinnerlessCompetition, HeteroclinicChannel, MetastabilityIndex
  - defs: (none)
- `models.py`
  - classes: GenerativeWorldModel
  - defs: (none)
- `morphological.py`
  - classes: TensionField, PeripheralGanglion, SleepWakeController, MorphologicalMemory, DistributedBody
  - defs: (none)
- `motivation.py`
  - classes: NoveltyDetector, MasterySignal, CoherenceReward, EpistemicValue, IntrinsicMotivationSystem
  - defs: (none)
- `multiplexing.py`
  - classes: PhaseGate, CrossFrequencyCoupling, TemporalMultiplexer, PhaseEncoder, NestedOscillator
  - defs: (none)
- `neural_darwinism.py`
  - classes: HypothesisProfile, NeuronalGroup, SelectionArena, SymbiogenesisDetector, SpeciatedSelectionArena
  - defs: (none)
- `neural_mass.py`
  - classes: WilsonCowanPopulation, WilsonCowanNetwork
  - defs: (none)
- `neuro_diagnostics.py`
  - classes: (none)
  - defs: spectral_exponent, dfa_exponent, lempel_ziv_complexity, avalanche_statistics, complexity_entropy
- `neuromodulation.py`
  - classes: NeuromodulatorState, NeuromodulatorDynamics, GlialLayer, StochasticResonance, InjectionLocking, ThreeTimescaleNetwork
  - defs: (none)
- `noise.py`
  - classes: (none)
  - defs: (none)
- `ode_solvers.py`
  - classes: SolverConfig, SolverResult, BaseSolver, EulerSolver, RK4Solver, DopriSolver
  - defs: create_solver, solve_ivp, rk4_step, euler_step
- `organ.py`
  - classes: Proposal, Organ, OscillatoryWriteGate, PreGWAdapter, HPCOrgan, MotivationOrgan, SelectionOrgan
  - defs: (none)
- `oscillators.py`
  - classes: Snake, LearnableKuramotoBank, SensoryCoupling, StuartLandauOscillator, KuramotoDaidoMeanField, PhaseTopologyCache
  - defs: _rk4_step
- `oscillatory_moe.py`
  - classes: ExpertPool, LoadBalancer, OscillatorRouter, SparseMoE
  - defs: (none)
- `perturbation.py`
  - classes: PerlinNoise, PerturbationLayer
  - defs: (none)
- `phase_attention.py`
  - classes: PhaseAttention, PhaseMultiHeadAttention, HybridPhaseAttention, OscillatorGatedFFN
  - defs: (none)
- `potentials.py`
  - classes: MultiAttractorPotential
  - defs: (none)
- `precision_weighting.py`
  - classes: PrecisionEstimator, PrecisionGate, AttentionAsPrecision
  - defs: (none)
- `prospective_config.py`
  - classes: ProspectiveInference, ProspectiveHebbian, InferThenModify
  - defs: (none)
- `quantum_cognition.py`
  - classes: QuantumState, UnitaryTransform, ProjectiveMeasurement, QuantumInterference, QuantumEntanglement, QuantumDecisionMaker, QuantumMemory
  - defs: complex_mm, complex_conj, complex_norm_sq, complex_inner
- `reservoir.py`
  - classes: EchoStateNetwork, IntrinsicPlasticity, ReservoirWithIP, CriticalReservoir
  - defs: spectral_radius, scale_spectral_radius
- `selective_ssm.py`
  - classes: SelectiveScan, SelectiveStateSpace, SSMBlock, MambaLayer, OscillatorySSM
  - defs: (none)
- `self_model.py`
  - classes: MarkovBlanket, SelfModel, NarrativeIdentity
  - defs: (none)
- `sensory_binding.py`
  - classes: ModalityEncoder, BindingStrength, CrossModalAttention, MultimodalBinder
  - defs: (none)
- `short_term_plasticity.py`
  - classes: TsodyksMarkramSynapse, STPLayer
  - defs: (none)
- `social.py`
  - classes: MirrorResonance, SocialCoupling, EmpatheticConstraint, ConstitutionalResonance, OscillatoryEconomy, TheoryOfMind
  - defs: (none)
- `somatic.py`
  - classes: SomaticState, HardwareSensor, SomaticManifold, SomaticIntegrator
  - defs: (none)
- `spherical.py`
  - classes: LearnableOmega, TangentProjection, SphericalKuramotoLayer, SphericalKuramotoBank
  - defs: reshape_to_groups, reshape_from_groups, l2_normalize, normalize_oscillators, exponential_map, _params_to_skew_symmetric
- `spiking.py`
  - classes: SurrogateSpike, LIFNeuron, SpikingLayer, OscillatorySNN, STDPLayer, AdExNeuron, SpikeEncoder, SpikeDecoder, SynapticDelay
  - defs: surrogate_spike
- `stochastic_resonance.py`
  - classes: OrnsteinUhlenbeckNoise, AdaptiveStochasticResonance, ResonanceMonitor
  - defs: (none)
- `strange_attractor.py`
  - classes: AttractorState, LorenzAttractor, PersonalityAttractor, FractalSelfModel, MultiScaleFractalPersonality, FractalPatternLearner
  - defs: create_enhanced_fractal_self
- `temporal_binding.py`
  - classes: SequenceAKOrN, PhaseGrouping, OscillatoryChunking
  - defs: (none)
- `temporal_graph.py`
  - classes: TemporalNode, TemporalEdge, TemporalKnowledgeGraph, Event, EventTimeline, TemporalReasoner
  - defs: (none)
- `thermodynamic.py`
  - classes: TemperatureSchedule, LangevinDynamics, SimulatedAnnealing, PhaseTransitionDetector, ThermodynamicOscillatorSystem
  - defs: (none)
- `visual_binding.py`
  - classes: AKOrNLayer, AKOrNBlock, AKOrNEncoder, OscillatorySegmenter, BindingReadout
  - defs: (none)
- `voice_dynamics.py`
  - classes: SpeechRhythm, VoiceEffects, SpeechOscillator, VoiceNeuromodulation, ProsodyModulator, VoiceDynamicsSystem
  - defs: (none)
- `volatility_filter.py`
  - classes: VolatilityNode, HierarchicalVolatilityFilter, VolatilityGatedLearning
  - defs: (none)
- `world_model.py`
  - classes: WorldModelState, GRUDynamics, StochasticStateModel, RSSM, NeuralODEDynamics, ObservationModel, RewardModel, ContinueModel, WorldModel, ImaginationBasedPlanner
  - defs: (none)

