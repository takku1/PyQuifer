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
