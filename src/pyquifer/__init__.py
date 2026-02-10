# src/pyquifer/__init__.py

"""
PyQuifer: A PyTorch library for simulating consciousness through generative world models.

This package provides tools for building "Physical Intelligence" models that
"resonate" with data rather than merely calculating answers, bridging
Physical Intelligence with Cognitive Architecture.

OPTIMIZED: Lazy loading of all modules - import time reduced from 2300ms to <50ms.
Components are loaded on first access, not at import time.
"""

__version__ = "0.1.0"

# Library-level logging: attach NullHandler so users aren't forced into our config.
# Users call logging.basicConfig() or configure handlers in their own code.
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Define what can be imported - actual loading happens lazily
__all__ = [
    "PyQuifer",
    "PerturbationLayer",
    "LearnableKuramotoBank",
    "Snake",
    "MultiAttractorPotential",
    "MindEyeActualization",
    "GenerativeWorldModel",
    "FrequencyBank",
    "HarmonicOscillator",
    "LinOSSLayer",
    "LinOSSEncoder",
    "LIFNeuron",
    "SpikingLayer",
    "OscillatorySNN",
    "STDPLayer",
    "surrogate_spike",
    "NoveltyDetector",
    "MasterySignal",
    "CoherenceReward",
    "IntrinsicMotivationSystem",
    "EpistemicValue",
    # Multiplexing
    "PhaseGate",
    "CrossFrequencyCoupling",
    "TemporalMultiplexer",
    "PhaseEncoder",
    "NestedOscillator",
    # Consciousness
    "PerturbationalComplexity",
    "IntegrationMeasure",
    "DifferentiationMeasure",
    "ConsciousnessMonitor",
    # Criticality
    "AvalancheDetector",
    "BranchingRatio",
    "CriticalityController",
    "HomeostaticRegulator",
    "KuramotoCriticalityMonitor",
    # Learning
    "EligibilityTrace",
    "RewardModulatedHebbian",
    "ContrastiveHebbian",
    "PredictiveCoding",
    # Spherical Kuramoto (from akorn)
    "SphericalKuramotoLayer",
    "SphericalKuramotoBank",
    "LearnableOmega",
    "TangentProjection",
    "normalize_oscillators",
    "exponential_map",
    # Active Inference (from Deep_AIF/pymdp)
    "ActiveInferenceAgent",
    "ExpectedFreeEnergy",
    "PredictiveEncoder",
    "PredictiveDecoder",
    "TransitionModel",
    "BeliefUpdate",
    "reparameterize",
    "kl_divergence_gaussian",
    # Thermodynamic (from generative_thermodynamic_computing/Neuroca)
    "ThermodynamicOscillatorSystem",
    "SimulatedAnnealing",
    "LangevinDynamics",
    "TemperatureSchedule",
    "PhaseTransitionDetector",
    # IIT Metrics (from pyphi)
    "EarthMoverDistance",
    "KLDivergence",
    "L1Distance",
    "InformationDensity",
    "PartitionedInformation",
    "IntegratedInformation",
    "CauseEffectRepertoire",
    "IITConsciousnessMonitor",
    "hamming_distance_matrix",
    "generate_bipartitions",
    "Concept",
    "SystemIrreducibilityAnalysis",
    # Hyperdimensional Computing (HDC/HRR)
    "circular_convolution",
    "circular_correlation",
    "normalize_hd",
    "HypervectorMemory",
    "PhaseBinder",
    "ResonantBinding",
    "HDCReasoner",
    # Neuromodulation (three-timescale dynamics)
    "NeuromodulatorState",
    "NeuromodulatorDynamics",
    "GlialLayer",
    "StochasticResonance",
    "InjectionLocking",
    "ThreeTimescaleNetwork",
    # Liquid Networks (LTC, Neural ODEs)
    "LiquidTimeConstantCell",
    "NeuralODE",
    "ODEFunc",
    "ContinuousTimeRNN",
    "MetastableCell",
    # Reservoir Computing (ESN)
    "EchoStateNetwork",
    "IntrinsicPlasticity",
    "ReservoirWithIP",
    "CriticalReservoir",
    "spectral_radius",
    "scale_spectral_radius",
    # Phase Attention (Kuramoto-based transformers)
    "PhaseAttention",
    "PhaseMultiHeadAttention",
    "HybridPhaseAttention",
    "OscillatorGatedFFN",
    # Hypernetworks (dynamic weight generation)
    "HyperNetwork",
    "OscillatorHyperNetwork",
    "DynamicLinear",
    "ContextualReservoir",
    # Advanced Spiking (2nd order LIF, E/I balance, reward modulation)
    "SynapticNeuron",
    "AlphaNeuron",
    "RecurrentSynapticLayer",
    "RewardPredictionError",
    "EligibilityModulatedSTDP",
    # Kindchenschema Safety Interface (neoteny-inspired safety)
    "SafetyEnvelope",
    "ParentModule",
    "ReflexToStrategy",
    "LimbicResonance",
    # Morphological Computation (embodied cognition)
    "TensionField",
    "PeripheralGanglion",
    "SleepWakeController",
    "MorphologicalMemory",
    "DistributedBody",
    # Social Cognition (mirror neurons, ethics as thermodynamics)
    "MirrorResonance",
    "SocialCoupling",
    "EmpatheticConstraint",
    "ConstitutionalResonance",
    "OscillatoryEconomy",
    "TheoryOfMind",
    # Developmental Cognition (cute as dynamics, not pixels)
    "DynamicalSignatureDetector",
    "KindchenschemaDetector",
    "ProtectiveDrive",
    "EvolutionaryAttractor",
    "IntrinsicCuteUnderstanding",
    "DevelopmentalStageDetector",
    "PotentialActualization",
    # Ecological Cognition (digital species)
    "TimeScale",
    "ChronobiologicalSystem",
    "ImmunologicalLayer",
    "SynapticHomeostasis",
    "Umwelt",
    "AgencyMaintenance",
    "EcologicalSystem",
    # Somatic Layer (hardware → feelings)
    "SomaticState",
    "HardwareSensor",
    "SomaticManifold",
    "SomaticIntegrator",
    # Metacognitive Loop (thinking about thinking)
    "ConfidenceLevel",
    "ReasoningStep",
    "MetacognitiveState",
    "ConfidenceEstimator",
    "ReasoningMonitor",
    "MetacognitiveLoop",
    # Hyperbolic Geometry (curved thought space)
    "HyperbolicOperations",
    "HyperbolicLinear",
    "EmotionalGravityManifold",
    "MixedCurvatureManifold",
    # Strange Attractors (fractal personality)
    "LorenzAttractor",
    "PersonalityAttractor",
    "FractalSelfModel",
    "AttractorState",
    # X-16/X-17: Enhanced fractal personality and learning
    "MultiScaleFractalPersonality",
    "FractalPatternLearner",
    "create_enhanced_fractal_self",
    # Continual Learning (anti-forgetting)
    "ElasticWeightConsolidation",
    "SynapticIntelligence",
    "ContinualBackprop",
    "MESU",
    "ExperienceReplay",
    "ContinualLearner",
    # Quantum Cognition (superposition decisions)
    "QuantumState",
    "UnitaryTransform",
    "ProjectiveMeasurement",
    "QuantumInterference",
    "QuantumEntanglement",
    "QuantumDecisionMaker",
    "QuantumMemory",
    # World Models (imagination and planning)
    "WorldModelState",
    "RSSM",
    "NeuralODEDynamics",
    "WorldModel",
    "ImaginationBasedPlanner",
    # Global Workspace (consciousness bottleneck)
    "SalienceComputer",
    "IgnitionDynamics",
    "CompetitionDynamics",
    "PrecisionWeighting",
    "GlobalBroadcast",
    "GlobalWorkspace",
    "HierarchicalWorkspace",
    # Voice Dynamics (oscillator → prosody)
    "SpeechRhythm",
    "VoiceEffects",
    "SpeechOscillator",
    "VoiceNeuromodulation",
    "ProsodyModulator",
    "VoiceDynamicsSystem",
    # Precision Weighting (attention as gain control)
    "PrecisionEstimator",
    "PrecisionGate",
    "AttentionAsPrecision",
    # Hierarchical Predictive Coding (multi-level generative model)
    "PredictiveLevel",
    "HierarchicalPredictiveCoding",
    # Metastability (winnerless competition, heteroclinic channels)
    "WinnerlessCompetition",
    "HeteroclinicChannel",
    "MetastabilityIndex",
    # Adaptive Stochastic Resonance (optimal noise finding)
    "AdaptiveStochasticResonance",
    "ResonanceMonitor",
    # Causal Flow (transfer entropy, directed information)
    "TransferEntropyEstimator",
    "CausalFlowMap",
    "DominanceDetector",
    # Memory Consolidation (sleep replay, episodic→semantic)
    "EpisodicBuffer",
    "SharpWaveRipple",
    "ConsolidationEngine",
    "MemoryReconsolidation",
    # Self-Model (minimal phenomenal self, Markov blanket)
    "MarkovBlanket",
    "SelfModel",
    "NarrativeIdentity",
    # Neural Darwinism (neuronal group selection)
    "NeuronalGroup",
    "SelectionArena",
    "SymbiogenesisDetector",
    # Integration (cognitive cycle)
    "CycleConfig",
    "CognitiveCycle",
    # Bridge (LLM modulation API)
    "PyQuiferBridge",
    "ModulationState",
    # Volatility Filter (adaptive learning rates from HGF)
    "VolatilityNode",
    "HierarchicalVolatilityFilter",
    "VolatilityGatedLearning",
    # Phase 5: OU Noise (stochastic resonance)
    "OrnsteinUhlenbeckNoise",
    # Phase 5: Short-Term Plasticity (Tsodyks-Markram)
    "TsodyksMarkramSynapse",
    "STPLayer",
    # Phase 5: Kuramoto-Daido Mean-Field
    "KuramotoDaidoMeanField",
    # Phase 5: Stuart-Landau Oscillator
    "StuartLandauOscillator",
    # Sensory-Oscillator Coupling
    "SensoryCoupling",
    # Phase 5: E-prop Dual Eligibility
    "EpropSTDP",
    # Phase 5: Differentiable Plasticity + Learnable Eligibility
    "DifferentiablePlasticity",
    "LearnableEligibilityTrace",
    # Phase 5: Speciated Neural Darwinism
    "SpeciatedSelectionArena",
    # Phase 5: AdEx Neuron
    "AdExNeuron",
    # Phase 5: Koopman Bifurcation Detection
    "KoopmanBifurcationDetector",
    # Phase 5: Wilson-Cowan Neural Mass
    "WilsonCowanPopulation",
    "WilsonCowanNetwork",
    # Phase 6: New spiking primitives
    "SpikeEncoder",
    "SpikeDecoder",
    "SynapticDelay",
    # Phase 7: Benchmark gap-closure
    "PhaseTopologyCache",
    "NoProgressDetector",
    "EvidenceSource",
    "EvidenceAggregator",
    "HypothesisProfile",
    # Phase 8: Training core
    "EquilibriumPropagationTrainer",
    "EPKuramotoClassifier",
    "OscillationGatedPlasticity",
    "ThreeFactorRule",
    "OscillatoryPredictiveCoding",
    "SleepReplayConsolidation",
    "DendriticNeuron",
    "DendriticStack",
    # Phase 9: Organ protocol + GWT wiring
    "Organ",
    "Proposal",
    "OscillatoryWriteGate",
    "PreGWAdapter",
    "HPCOrgan",
    "MotivationOrgan",
    "SelectionOrgan",
    "DiversityTracker",
    # Phase 10: Multi-workspace ensemble + cross-bleed
    "StandingBroadcast",
    "CrossBleedGate",
    "WorkspaceEnsemble",
    # Phase 11: ODE Solvers
    "SolverConfig",
    "EulerSolver",
    "RK4Solver",
    "DopriSolver",
    "solve_ivp",
    "create_solver",
    # Phase 11: Complex Oscillators
    "ComplexKuramotoBank",
    "ComplexCoupling",
    "ModReLU",
    "ComplexLinear",
    "ComplexBatchNorm",
    "complex_order_parameter",
    # Phase 11: CUDA Kernels
    "KuramotoCUDAKernel",
    "TensorDiagnostics",
    # Phase 12: Visual Binding (AKOrN)
    "AKOrNLayer",
    "AKOrNBlock",
    "AKOrNEncoder",
    "OscillatorySegmenter",
    "BindingReadout",
    # Phase 12: Temporal Binding
    "SequenceAKOrN",
    "PhaseGrouping",
    "OscillatoryChunking",
    # Phase 12: Sensory Binding
    "MultimodalBinder",
    "BindingStrength",
    "CrossModalAttention",
    "ModalityEncoder",
    # Phase 13: Deep Active Inference
    "DeepAIF",
    "LatentTransitionModel",
    "PolicyNetwork",
    "MultiStepPlanner",
    # Phase 13: JEPA
    "JEPAEncoder",
    "JEPAPredictor",
    "VICRegLoss",
    "BarlowLoss",
    "ActionJEPA",
    # Phase 13: Deliberation
    "Deliberator",
    "ProcessRewardModel",
    "BeamSearchReasoner",
    "SelfCorrectionLoop",
    "ComputeBudget",
    # Phase 14: CLS Memory
    "HippocampalModule",
    "NeocorticalModule",
    "ConsolidationScheduler",
    "ForgettingCurve",
    "ImportanceScorer",
    "MemoryInterference",
    # Phase 14: Temporal Knowledge Graph
    "TemporalNode",
    "TemporalEdge",
    "TemporalKnowledgeGraph",
    "EventTimeline",
    "TemporalReasoner",
    # Phase 14: Gated Memory
    "NMDAGate",
    "DifferentiableMemoryBank",
    "MemoryConsolidationLoop",
    # Phase 15: Cognitive Appraisal
    "AppraisalDimension",
    "AppraisalChain",
    "OCC_Model",
    "EmotionAttribution",
    # Phase 15: Causal Reasoning
    "CausalGraph",
    "DoOperator",
    "CounterfactualEngine",
    "CausalDiscovery",
    "InterventionalQuery",
    # Phase 16: Selective SSM
    "SelectiveStateSpace",
    "SelectiveScan",
    "SSMBlock",
    "MambaLayer",
    "OscillatorySSM",
    # Phase 16: Oscillatory MoE
    "ExpertPool",
    "OscillatorRouter",
    "SparseMoE",
    # Phase 16: Graph Reasoning
    "DynamicGraphAttention",
    "MessagePassingWithPhase",
    "TemporalGraphTransformer",
    # Phase 17: Prospective Configuration
    "ProspectiveInference",
    "ProspectiveHebbian",
    "InferThenModify",
    # Phase 17: Dendritic Learning
    "PyramidalNeuron",
    "DendriticLocalizedLearning",
    "DendriticErrorSignal",
    # Phase 17: Energy Spiking PC
    "EnergyOptimizedSNN",
    "MultiCompartmentSpikingPC",
    "EnergyLandscape",
    # Phase 18: FHRR
    "FHRREncoder",
    "LatencyEncoder",
    "SpikeVSAOps",
    "NeuromorphicExporter",
    # Phase 18: FlashRNN
    "FlashLTC",
    "FlashCfC",
    "FlashLTCLayer",
]

# Mapping from name to (module, attribute) for lazy loading
_LAZY_IMPORTS = {
    # Core
    "PyQuifer": (".core", "PyQuifer"),
    "PerturbationLayer": (".perturbation", "PerturbationLayer"),
    "LearnableKuramotoBank": (".oscillators", "LearnableKuramotoBank"),
    "Snake": (".oscillators", "Snake"),
    "FrequencyBank": (".frequency_bank", "FrequencyBank"),
    "MultiAttractorPotential": (".potentials", "MultiAttractorPotential"),
    "MindEyeActualization": (".diffusion", "MindEyeActualization"),
    "GenerativeWorldModel": (".models", "GenerativeWorldModel"),

    # LinOSS
    "HarmonicOscillator": (".linoss", "HarmonicOscillator"),
    "LinOSSLayer": (".linoss", "LinOSSLayer"),
    "LinOSSEncoder": (".linoss", "LinOSSEncoder"),

    # Spiking
    "LIFNeuron": (".spiking", "LIFNeuron"),
    "SpikingLayer": (".spiking", "SpikingLayer"),
    "OscillatorySNN": (".spiking", "OscillatorySNN"),
    "STDPLayer": (".spiking", "STDPLayer"),
    "surrogate_spike": (".spiking", "surrogate_spike"),

    # Motivation
    "NoveltyDetector": (".motivation", "NoveltyDetector"),
    "MasterySignal": (".motivation", "MasterySignal"),
    "CoherenceReward": (".motivation", "CoherenceReward"),
    "IntrinsicMotivationSystem": (".motivation", "IntrinsicMotivationSystem"),
    "EpistemicValue": (".motivation", "EpistemicValue"),

    # Multiplexing
    "PhaseGate": (".multiplexing", "PhaseGate"),
    "CrossFrequencyCoupling": (".multiplexing", "CrossFrequencyCoupling"),
    "TemporalMultiplexer": (".multiplexing", "TemporalMultiplexer"),
    "PhaseEncoder": (".multiplexing", "PhaseEncoder"),
    "NestedOscillator": (".multiplexing", "NestedOscillator"),

    # Consciousness
    "PerturbationalComplexity": (".consciousness", "PerturbationalComplexity"),
    "IntegrationMeasure": (".consciousness", "IntegrationMeasure"),
    "DifferentiationMeasure": (".consciousness", "DifferentiationMeasure"),
    "ConsciousnessMonitor": (".consciousness", "ConsciousnessMonitor"),

    # Criticality
    "AvalancheDetector": (".criticality", "AvalancheDetector"),
    "BranchingRatio": (".criticality", "BranchingRatio"),
    "CriticalityController": (".criticality", "CriticalityController"),
    "HomeostaticRegulator": (".criticality", "HomeostaticRegulator"),
    "KuramotoCriticalityMonitor": (".criticality", "KuramotoCriticalityMonitor"),

    # Learning
    "EligibilityTrace": (".learning", "EligibilityTrace"),
    "RewardModulatedHebbian": (".learning", "RewardModulatedHebbian"),
    "ContrastiveHebbian": (".learning", "ContrastiveHebbian"),
    "PredictiveCoding": (".learning", "PredictiveCoding"),

    # Spherical
    "SphericalKuramotoLayer": (".spherical", "SphericalKuramotoLayer"),
    "SphericalKuramotoBank": (".spherical", "SphericalKuramotoBank"),
    "LearnableOmega": (".spherical", "LearnableOmega"),
    "TangentProjection": (".spherical", "TangentProjection"),
    "normalize_oscillators": (".spherical", "normalize_oscillators"),
    "exponential_map": (".spherical", "exponential_map"),

    # Active Inference
    "ActiveInferenceAgent": (".active_inference", "ActiveInferenceAgent"),
    "ExpectedFreeEnergy": (".active_inference", "ExpectedFreeEnergy"),
    "PredictiveEncoder": (".active_inference", "PredictiveEncoder"),
    "PredictiveDecoder": (".active_inference", "PredictiveDecoder"),
    "TransitionModel": (".active_inference", "TransitionModel"),
    "BeliefUpdate": (".active_inference", "BeliefUpdate"),
    "reparameterize": (".active_inference", "reparameterize"),
    "kl_divergence_gaussian": (".active_inference", "kl_divergence_gaussian"),

    # Thermodynamic
    "ThermodynamicOscillatorSystem": (".thermodynamic", "ThermodynamicOscillatorSystem"),
    "SimulatedAnnealing": (".thermodynamic", "SimulatedAnnealing"),
    "LangevinDynamics": (".thermodynamic", "LangevinDynamics"),
    "TemperatureSchedule": (".thermodynamic", "TemperatureSchedule"),
    "PhaseTransitionDetector": (".thermodynamic", "PhaseTransitionDetector"),

    # IIT Metrics
    "EarthMoverDistance": (".iit_metrics", "EarthMoverDistance"),
    "KLDivergence": (".iit_metrics", "KLDivergence"),
    "L1Distance": (".iit_metrics", "L1Distance"),
    "InformationDensity": (".iit_metrics", "InformationDensity"),
    "PartitionedInformation": (".iit_metrics", "PartitionedInformation"),
    "IntegratedInformation": (".iit_metrics", "IntegratedInformation"),
    "CauseEffectRepertoire": (".iit_metrics", "CauseEffectRepertoire"),
    "IITConsciousnessMonitor": (".iit_metrics", "IITConsciousnessMonitor"),
    "hamming_distance_matrix": (".iit_metrics", "hamming_distance_matrix"),
    "generate_bipartitions": (".iit_metrics", "generate_bipartitions"),
    "Concept": (".iit_metrics", "Concept"),
    "SystemIrreducibilityAnalysis": (".iit_metrics", "SystemIrreducibilityAnalysis"),

    # Hyperdimensional
    "circular_convolution": (".hyperdimensional", "circular_convolution"),
    "circular_correlation": (".hyperdimensional", "circular_correlation"),
    "normalize_hd": (".hyperdimensional", "normalize_hd"),
    "HypervectorMemory": (".hyperdimensional", "HypervectorMemory"),
    "PhaseBinder": (".hyperdimensional", "PhaseBinder"),
    "ResonantBinding": (".hyperdimensional", "ResonantBinding"),
    "HDCReasoner": (".hyperdimensional", "HDCReasoner"),

    # Neuromodulation
    "NeuromodulatorState": (".neuromodulation", "NeuromodulatorState"),
    "NeuromodulatorDynamics": (".neuromodulation", "NeuromodulatorDynamics"),
    "GlialLayer": (".neuromodulation", "GlialLayer"),
    "StochasticResonance": (".neuromodulation", "StochasticResonance"),
    "InjectionLocking": (".neuromodulation", "InjectionLocking"),
    "ThreeTimescaleNetwork": (".neuromodulation", "ThreeTimescaleNetwork"),

    # Liquid Networks
    "LiquidTimeConstantCell": (".liquid_networks", "LiquidTimeConstantCell"),
    "NeuralODE": (".liquid_networks", "NeuralODE"),
    "ODEFunc": (".liquid_networks", "ODEFunc"),
    "ContinuousTimeRNN": (".liquid_networks", "ContinuousTimeRNN"),
    "MetastableCell": (".liquid_networks", "MetastableCell"),

    # Reservoir
    "EchoStateNetwork": (".reservoir", "EchoStateNetwork"),
    "IntrinsicPlasticity": (".reservoir", "IntrinsicPlasticity"),
    "ReservoirWithIP": (".reservoir", "ReservoirWithIP"),
    "CriticalReservoir": (".reservoir", "CriticalReservoir"),
    "spectral_radius": (".reservoir", "spectral_radius"),
    "scale_spectral_radius": (".reservoir", "scale_spectral_radius"),

    # Phase Attention
    "PhaseAttention": (".phase_attention", "PhaseAttention"),
    "PhaseMultiHeadAttention": (".phase_attention", "PhaseMultiHeadAttention"),
    "HybridPhaseAttention": (".phase_attention", "HybridPhaseAttention"),
    "OscillatorGatedFFN": (".phase_attention", "OscillatorGatedFFN"),

    # Hypernetwork
    "HyperNetwork": (".hypernetwork", "HyperNetwork"),
    "OscillatorHyperNetwork": (".hypernetwork", "OscillatorHyperNetwork"),
    "DynamicLinear": (".hypernetwork", "DynamicLinear"),
    "ContextualReservoir": (".hypernetwork", "ContextualReservoir"),

    # Advanced Spiking
    "SynapticNeuron": (".advanced_spiking", "SynapticNeuron"),
    "AlphaNeuron": (".advanced_spiking", "AlphaNeuron"),
    "RecurrentSynapticLayer": (".advanced_spiking", "RecurrentSynapticLayer"),
    "RewardPredictionError": (".advanced_spiking", "RewardPredictionError"),
    "EligibilityModulatedSTDP": (".advanced_spiking", "EligibilityModulatedSTDP"),

    # Kindchenschema
    "SafetyEnvelope": (".kindchenschema", "SafetyEnvelope"),
    "ParentModule": (".kindchenschema", "ParentModule"),
    "ReflexToStrategy": (".kindchenschema", "ReflexToStrategy"),
    "LimbicResonance": (".kindchenschema", "LimbicResonance"),

    # Morphological
    "TensionField": (".morphological", "TensionField"),
    "PeripheralGanglion": (".morphological", "PeripheralGanglion"),
    "SleepWakeController": (".morphological", "SleepWakeController"),
    "MorphologicalMemory": (".morphological", "MorphologicalMemory"),
    "DistributedBody": (".morphological", "DistributedBody"),

    # Social
    "MirrorResonance": (".social", "MirrorResonance"),
    "SocialCoupling": (".social", "SocialCoupling"),
    "EmpatheticConstraint": (".social", "EmpatheticConstraint"),
    "ConstitutionalResonance": (".social", "ConstitutionalResonance"),
    "OscillatoryEconomy": (".social", "OscillatoryEconomy"),
    "TheoryOfMind": (".social", "TheoryOfMind"),

    # Developmental
    "DynamicalSignatureDetector": (".developmental", "DynamicalSignatureDetector"),
    "KindchenschemaDetector": (".developmental", "KindchenschemaDetector"),
    "ProtectiveDrive": (".developmental", "ProtectiveDrive"),
    "EvolutionaryAttractor": (".developmental", "EvolutionaryAttractor"),
    "IntrinsicCuteUnderstanding": (".developmental", "IntrinsicCuteUnderstanding"),
    "DevelopmentalStageDetector": (".developmental", "DevelopmentalStageDetector"),
    "PotentialActualization": (".developmental", "PotentialActualization"),

    # Ecology
    "TimeScale": (".ecology", "TimeScale"),
    "ChronobiologicalSystem": (".ecology", "ChronobiologicalSystem"),
    "ImmunologicalLayer": (".ecology", "ImmunologicalLayer"),
    "SynapticHomeostasis": (".ecology", "SynapticHomeostasis"),
    "Umwelt": (".ecology", "Umwelt"),
    "AgencyMaintenance": (".ecology", "AgencyMaintenance"),
    "EcologicalSystem": (".ecology", "EcologicalSystem"),
    # Somatic Layer
    "SomaticState": (".somatic", "SomaticState"),
    "HardwareSensor": (".somatic", "HardwareSensor"),
    "SomaticManifold": (".somatic", "SomaticManifold"),
    "SomaticIntegrator": (".somatic", "SomaticIntegrator"),
    # Metacognitive Loop
    "ConfidenceLevel": (".metacognitive", "ConfidenceLevel"),
    "ReasoningStep": (".metacognitive", "ReasoningStep"),
    "MetacognitiveState": (".metacognitive", "MetacognitiveState"),
    "ConfidenceEstimator": (".metacognitive", "ConfidenceEstimator"),
    "ReasoningMonitor": (".metacognitive", "ReasoningMonitor"),
    "MetacognitiveLoop": (".metacognitive", "MetacognitiveLoop"),
    # Hyperbolic Geometry
    "HyperbolicOperations": (".hyperbolic", "HyperbolicOperations"),
    "HyperbolicLinear": (".hyperbolic", "HyperbolicLinear"),
    "EmotionalGravityManifold": (".hyperbolic", "EmotionalGravityManifold"),
    "MixedCurvatureManifold": (".hyperbolic", "MixedCurvatureManifold"),
    # Strange Attractors
    "LorenzAttractor": (".strange_attractor", "LorenzAttractor"),
    "PersonalityAttractor": (".strange_attractor", "PersonalityAttractor"),
    "FractalSelfModel": (".strange_attractor", "FractalSelfModel"),
    "AttractorState": (".strange_attractor", "AttractorState"),
    # X-16: Multi-Scale Fractal Personality
    "MultiScaleFractalPersonality": (".strange_attractor", "MultiScaleFractalPersonality"),
    # X-17: Fractal Pattern Learning
    "FractalPatternLearner": (".strange_attractor", "FractalPatternLearner"),
    "create_enhanced_fractal_self": (".strange_attractor", "create_enhanced_fractal_self"),

    # Continual Learning
    "ElasticWeightConsolidation": (".continual_learning", "ElasticWeightConsolidation"),
    "SynapticIntelligence": (".continual_learning", "SynapticIntelligence"),
    "ContinualBackprop": (".continual_learning", "ContinualBackprop"),
    "MESU": (".continual_learning", "MESU"),
    "ExperienceReplay": (".continual_learning", "ExperienceReplay"),
    "ContinualLearner": (".continual_learning", "ContinualLearner"),

    # Quantum Cognition
    "QuantumState": (".quantum_cognition", "QuantumState"),
    "UnitaryTransform": (".quantum_cognition", "UnitaryTransform"),
    "ProjectiveMeasurement": (".quantum_cognition", "ProjectiveMeasurement"),
    "QuantumInterference": (".quantum_cognition", "QuantumInterference"),
    "QuantumEntanglement": (".quantum_cognition", "QuantumEntanglement"),
    "QuantumDecisionMaker": (".quantum_cognition", "QuantumDecisionMaker"),
    "QuantumMemory": (".quantum_cognition", "QuantumMemory"),

    # World Models
    "WorldModelState": (".world_model", "WorldModelState"),
    "RSSM": (".world_model", "RSSM"),
    "NeuralODEDynamics": (".world_model", "NeuralODEDynamics"),
    "WorldModel": (".world_model", "WorldModel"),
    "ImaginationBasedPlanner": (".world_model", "ImaginationBasedPlanner"),

    # Global Workspace
    "SalienceComputer": (".global_workspace", "SalienceComputer"),
    "IgnitionDynamics": (".global_workspace", "IgnitionDynamics"),
    "CompetitionDynamics": (".global_workspace", "CompetitionDynamics"),
    "PrecisionWeighting": (".global_workspace", "PrecisionWeighting"),
    "GlobalBroadcast": (".global_workspace", "GlobalBroadcast"),
    "GlobalWorkspace": (".global_workspace", "GlobalWorkspace"),
    "HierarchicalWorkspace": (".global_workspace", "HierarchicalWorkspace"),

    # Voice Dynamics
    "SpeechRhythm": (".voice_dynamics", "SpeechRhythm"),
    "VoiceEffects": (".voice_dynamics", "VoiceEffects"),
    "SpeechOscillator": (".voice_dynamics", "SpeechOscillator"),
    "VoiceNeuromodulation": (".voice_dynamics", "VoiceNeuromodulation"),
    "ProsodyModulator": (".voice_dynamics", "ProsodyModulator"),
    "VoiceDynamicsSystem": (".voice_dynamics", "VoiceDynamicsSystem"),

    # Precision Weighting
    "PrecisionEstimator": (".precision_weighting", "PrecisionEstimator"),
    "PrecisionGate": (".precision_weighting", "PrecisionGate"),
    "AttentionAsPrecision": (".precision_weighting", "AttentionAsPrecision"),

    # Hierarchical Predictive Coding
    "PredictiveLevel": (".hierarchical_predictive", "PredictiveLevel"),
    "HierarchicalPredictiveCoding": (".hierarchical_predictive", "HierarchicalPredictiveCoding"),

    # Metastability
    "WinnerlessCompetition": (".metastability", "WinnerlessCompetition"),
    "HeteroclinicChannel": (".metastability", "HeteroclinicChannel"),
    "MetastabilityIndex": (".metastability", "MetastabilityIndex"),

    # Adaptive Stochastic Resonance
    "AdaptiveStochasticResonance": (".stochastic_resonance", "AdaptiveStochasticResonance"),
    "ResonanceMonitor": (".stochastic_resonance", "ResonanceMonitor"),

    # Causal Flow
    "TransferEntropyEstimator": (".causal_flow", "TransferEntropyEstimator"),
    "CausalFlowMap": (".causal_flow", "CausalFlowMap"),
    "DominanceDetector": (".causal_flow", "DominanceDetector"),

    # Memory Consolidation
    "EpisodicBuffer": (".memory_consolidation", "EpisodicBuffer"),
    "SharpWaveRipple": (".memory_consolidation", "SharpWaveRipple"),
    "ConsolidationEngine": (".memory_consolidation", "ConsolidationEngine"),
    "MemoryReconsolidation": (".memory_consolidation", "MemoryReconsolidation"),

    # Self-Model
    "MarkovBlanket": (".self_model", "MarkovBlanket"),
    "SelfModel": (".self_model", "SelfModel"),
    "NarrativeIdentity": (".self_model", "NarrativeIdentity"),

    # Neural Darwinism
    "NeuronalGroup": (".neural_darwinism", "NeuronalGroup"),
    "SelectionArena": (".neural_darwinism", "SelectionArena"),
    "SymbiogenesisDetector": (".neural_darwinism", "SymbiogenesisDetector"),

    # Integration
    "CycleConfig": (".integration", "CycleConfig"),
    "CognitiveCycle": (".integration", "CognitiveCycle"),

    # Bridge
    "PyQuiferBridge": (".bridge", "PyQuiferBridge"),

    # Volatility Filter
    "VolatilityNode": (".volatility_filter", "VolatilityNode"),
    "HierarchicalVolatilityFilter": (".volatility_filter", "HierarchicalVolatilityFilter"),
    "VolatilityGatedLearning": (".volatility_filter", "VolatilityGatedLearning"),
    "ModulationState": (".bridge", "ModulationState"),

    # Phase 5: OU Noise
    "OrnsteinUhlenbeckNoise": (".stochastic_resonance", "OrnsteinUhlenbeckNoise"),

    # Phase 5: Short-Term Plasticity
    "TsodyksMarkramSynapse": (".short_term_plasticity", "TsodyksMarkramSynapse"),
    "STPLayer": (".short_term_plasticity", "STPLayer"),

    # Phase 5: Kuramoto-Daido Mean-Field
    "KuramotoDaidoMeanField": (".oscillators", "KuramotoDaidoMeanField"),

    # Phase 5: Stuart-Landau Oscillator
    "StuartLandauOscillator": (".oscillators", "StuartLandauOscillator"),

    # Sensory-Oscillator Coupling
    "SensoryCoupling": (".oscillators", "SensoryCoupling"),

    # Phase 5: E-prop Dual Eligibility
    "EpropSTDP": (".advanced_spiking", "EpropSTDP"),

    # Phase 5: Differentiable Plasticity + Learnable Eligibility
    "DifferentiablePlasticity": (".learning", "DifferentiablePlasticity"),
    "LearnableEligibilityTrace": (".learning", "LearnableEligibilityTrace"),

    # Phase 5: Speciated Neural Darwinism
    "SpeciatedSelectionArena": (".neural_darwinism", "SpeciatedSelectionArena"),

    # Phase 5: AdEx Neuron
    "AdExNeuron": (".spiking", "AdExNeuron"),

    # Phase 5: Koopman Bifurcation Detection
    "KoopmanBifurcationDetector": (".criticality", "KoopmanBifurcationDetector"),

    # Phase 5: Wilson-Cowan Neural Mass
    "WilsonCowanPopulation": (".neural_mass", "WilsonCowanPopulation"),
    "WilsonCowanNetwork": (".neural_mass", "WilsonCowanNetwork"),

    # Phase 6: New spiking primitives
    "SpikeEncoder": (".spiking", "SpikeEncoder"),
    "SpikeDecoder": (".spiking", "SpikeDecoder"),
    "SynapticDelay": (".spiking", "SynapticDelay"),

    # Phase 7: Benchmark gap-closure
    "PhaseTopologyCache": (".oscillators", "PhaseTopologyCache"),
    "NoProgressDetector": (".criticality", "NoProgressDetector"),
    "EvidenceSource": (".metacognitive", "EvidenceSource"),
    "EvidenceAggregator": (".metacognitive", "EvidenceAggregator"),
    "HypothesisProfile": (".neural_darwinism", "HypothesisProfile"),

    # Phase 8: Training core
    "EquilibriumPropagationTrainer": (".equilibrium_propagation", "EquilibriumPropagationTrainer"),
    "EPKuramotoClassifier": (".equilibrium_propagation", "EPKuramotoClassifier"),
    "OscillationGatedPlasticity": (".learning", "OscillationGatedPlasticity"),
    "ThreeFactorRule": (".learning", "ThreeFactorRule"),
    "OscillatoryPredictiveCoding": (".hierarchical_predictive", "OscillatoryPredictiveCoding"),
    "SleepReplayConsolidation": (".memory_consolidation", "SleepReplayConsolidation"),
    "DendriticNeuron": (".dendritic", "DendriticNeuron"),
    "DendriticStack": (".dendritic", "DendriticStack"),

    # Phase 9: Organ protocol + GWT wiring
    "Organ": (".organ", "Organ"),
    "Proposal": (".organ", "Proposal"),
    "OscillatoryWriteGate": (".organ", "OscillatoryWriteGate"),
    "PreGWAdapter": (".organ", "PreGWAdapter"),
    "HPCOrgan": (".organ", "HPCOrgan"),
    "MotivationOrgan": (".organ", "MotivationOrgan"),
    "SelectionOrgan": (".organ", "SelectionOrgan"),
    "DiversityTracker": (".global_workspace", "DiversityTracker"),

    # Phase 10: Multi-workspace ensemble + cross-bleed
    "StandingBroadcast": (".global_workspace", "StandingBroadcast"),
    "CrossBleedGate": (".global_workspace", "CrossBleedGate"),
    "WorkspaceEnsemble": (".global_workspace", "WorkspaceEnsemble"),

    # Phase 11: ODE Solvers
    "SolverConfig": (".ode_solvers", "SolverConfig"),
    "EulerSolver": (".ode_solvers", "EulerSolver"),
    "RK4Solver": (".ode_solvers", "RK4Solver"),
    "DopriSolver": (".ode_solvers", "DopriSolver"),
    "solve_ivp": (".ode_solvers", "solve_ivp"),
    "create_solver": (".ode_solvers", "create_solver"),

    # Phase 11: Complex Oscillators
    "ComplexKuramotoBank": (".complex_oscillators", "ComplexKuramotoBank"),
    "ComplexCoupling": (".complex_oscillators", "ComplexCoupling"),
    "ModReLU": (".complex_oscillators", "ModReLU"),
    "ComplexLinear": (".complex_oscillators", "ComplexLinear"),
    "ComplexBatchNorm": (".complex_oscillators", "ComplexBatchNorm"),
    "complex_order_parameter": (".complex_oscillators", "complex_order_parameter"),

    # Phase 11: CUDA Kernels
    "KuramotoCUDAKernel": ("._cuda.kuramoto_kernel", "KuramotoCUDAKernel"),
    "TensorDiagnostics": ("._cuda.kuramoto_kernel", "TensorDiagnostics"),

    # Phase 12: Visual Binding (AKOrN)
    "AKOrNLayer": (".visual_binding", "AKOrNLayer"),
    "AKOrNBlock": (".visual_binding", "AKOrNBlock"),
    "AKOrNEncoder": (".visual_binding", "AKOrNEncoder"),
    "OscillatorySegmenter": (".visual_binding", "OscillatorySegmenter"),
    "BindingReadout": (".visual_binding", "BindingReadout"),

    # Phase 12: Temporal Binding
    "SequenceAKOrN": (".temporal_binding", "SequenceAKOrN"),
    "PhaseGrouping": (".temporal_binding", "PhaseGrouping"),
    "OscillatoryChunking": (".temporal_binding", "OscillatoryChunking"),

    # Phase 12: Sensory Binding
    "MultimodalBinder": (".sensory_binding", "MultimodalBinder"),
    "BindingStrength": (".sensory_binding", "BindingStrength"),
    "CrossModalAttention": (".sensory_binding", "CrossModalAttention"),
    "ModalityEncoder": (".sensory_binding", "ModalityEncoder"),

    # Phase 13: Deep Active Inference
    "DeepAIF": (".deep_active_inference", "DeepAIF"),
    "LatentTransitionModel": (".deep_active_inference", "LatentTransitionModel"),
    "PolicyNetwork": (".deep_active_inference", "PolicyNetwork"),
    "MultiStepPlanner": (".deep_active_inference", "MultiStepPlanner"),

    # Phase 13: JEPA
    "JEPAEncoder": (".jepa", "JEPAEncoder"),
    "JEPAPredictor": (".jepa", "JEPAPredictor"),
    "VICRegLoss": (".jepa", "VICRegLoss"),
    "BarlowLoss": (".jepa", "BarlowLoss"),
    "ActionJEPA": (".jepa", "ActionJEPA"),

    # Phase 13: Deliberation
    "Deliberator": (".deliberation", "Deliberator"),
    "ProcessRewardModel": (".deliberation", "ProcessRewardModel"),
    "BeamSearchReasoner": (".deliberation", "BeamSearchReasoner"),
    "SelfCorrectionLoop": (".deliberation", "SelfCorrectionLoop"),
    "ComputeBudget": (".deliberation", "ComputeBudget"),

    # Phase 14: CLS Memory
    "HippocampalModule": (".cls_memory", "HippocampalModule"),
    "NeocorticalModule": (".cls_memory", "NeocorticalModule"),
    "ConsolidationScheduler": (".cls_memory", "ConsolidationScheduler"),
    "ForgettingCurve": (".cls_memory", "ForgettingCurve"),
    "ImportanceScorer": (".cls_memory", "ImportanceScorer"),
    "MemoryInterference": (".cls_memory", "MemoryInterference"),

    # Phase 14: Temporal Knowledge Graph
    "TemporalNode": (".temporal_graph", "TemporalNode"),
    "TemporalEdge": (".temporal_graph", "TemporalEdge"),
    "TemporalKnowledgeGraph": (".temporal_graph", "TemporalKnowledgeGraph"),
    "EventTimeline": (".temporal_graph", "EventTimeline"),
    "TemporalReasoner": (".temporal_graph", "TemporalReasoner"),

    # Phase 14: Gated Memory
    "NMDAGate": (".gated_memory", "NMDAGate"),
    "DifferentiableMemoryBank": (".gated_memory", "DifferentiableMemoryBank"),
    "MemoryConsolidationLoop": (".gated_memory", "MemoryConsolidationLoop"),

    # Phase 15: Cognitive Appraisal
    "AppraisalDimension": (".appraisal", "AppraisalDimension"),
    "AppraisalChain": (".appraisal", "AppraisalChain"),
    "OCC_Model": (".appraisal", "OCC_Model"),
    "EmotionAttribution": (".appraisal", "EmotionAttribution"),

    # Phase 15: Causal Reasoning
    "CausalGraph": (".causal_reasoning", "CausalGraph"),
    "DoOperator": (".causal_reasoning", "DoOperator"),
    "CounterfactualEngine": (".causal_reasoning", "CounterfactualEngine"),
    "CausalDiscovery": (".causal_reasoning", "CausalDiscovery"),
    "InterventionalQuery": (".causal_reasoning", "InterventionalQuery"),

    # Phase 16: Selective SSM
    "SelectiveStateSpace": (".selective_ssm", "SelectiveStateSpace"),
    "SelectiveScan": (".selective_ssm", "SelectiveScan"),
    "SSMBlock": (".selective_ssm", "SSMBlock"),
    "MambaLayer": (".selective_ssm", "MambaLayer"),
    "OscillatorySSM": (".selective_ssm", "OscillatorySSM"),

    # Phase 16: Oscillatory MoE
    "ExpertPool": (".oscillatory_moe", "ExpertPool"),
    "OscillatorRouter": (".oscillatory_moe", "OscillatorRouter"),
    "SparseMoE": (".oscillatory_moe", "SparseMoE"),

    # Phase 16: Graph Reasoning
    "DynamicGraphAttention": (".graph_reasoning", "DynamicGraphAttention"),
    "MessagePassingWithPhase": (".graph_reasoning", "MessagePassingWithPhase"),
    "TemporalGraphTransformer": (".graph_reasoning", "TemporalGraphTransformer"),

    # Phase 17: Prospective Configuration
    "ProspectiveInference": (".prospective_config", "ProspectiveInference"),
    "ProspectiveHebbian": (".prospective_config", "ProspectiveHebbian"),
    "InferThenModify": (".prospective_config", "InferThenModify"),

    # Phase 17: Dendritic Learning
    "PyramidalNeuron": (".dendritic_learning", "PyramidalNeuron"),
    "DendriticLocalizedLearning": (".dendritic_learning", "DendriticLocalizedLearning"),
    "DendriticErrorSignal": (".dendritic_learning", "DendriticErrorSignal"),

    # Phase 17: Energy Spiking PC
    "EnergyOptimizedSNN": (".energy_spiking", "EnergyOptimizedSNN"),
    "MultiCompartmentSpikingPC": (".energy_spiking", "MultiCompartmentSpikingPC"),
    "EnergyLandscape": (".energy_spiking", "EnergyLandscape"),

    # Phase 18: FHRR
    "FHRREncoder": (".fhrr", "FHRREncoder"),
    "LatencyEncoder": (".fhrr", "LatencyEncoder"),
    "SpikeVSAOps": (".fhrr", "SpikeVSAOps"),
    "NeuromorphicExporter": (".fhrr", "NeuromorphicExporter"),

    # Phase 18: FlashRNN
    "FlashLTC": (".flash_rnn", "FlashLTC"),
    "FlashCfC": (".flash_rnn", "FlashCfC"),
    "FlashLTCLayer": (".flash_rnn", "FlashLTCLayer"),
}

# Cache for loaded attributes
_loaded = {}


def __getattr__(name: str):
    """Lazy load attributes on first access."""
    if name in _loaded:
        return _loaded[name]

    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        # Import the module
        import importlib
        module = importlib.import_module(module_path, package=__name__)
        # Get the attribute
        attr = getattr(module, attr_name)
        # Cache it
        _loaded[name] = attr
        return attr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available names for autocomplete."""
    return __all__
