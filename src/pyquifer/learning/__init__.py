"""Learning rules: synaptic plasticity, continual learning, consolidation, and dendritic credit."""
from pyquifer.learning.synaptic import (
    EligibilityTrace,
    RewardModulatedHebbian,
    ContrastiveHebbian,
    PredictiveCoding,
    DifferentiablePlasticity,
    ThreeFactorRule,
    OscillationGatedPlasticity,
)
from pyquifer.learning.continual import (
    ElasticWeightConsolidation,
    ContinualLearner,
)
from pyquifer.learning.consolidation import (
    EpisodicBuffer,
    ConsolidationEngine,
    SleepReplayConsolidation,
)
from pyquifer.learning.equilibrium_prop import EPKuramotoClassifier
from pyquifer.learning.stp import TsodyksMarkramSynapse, STPLayer
from pyquifer.learning.dendritic import DendriticNeuron, DendriticStack

__all__ = [
    "EligibilityTrace",
    "RewardModulatedHebbian",
    "ContrastiveHebbian",
    "PredictiveCoding",
    "DifferentiablePlasticity",
    "ThreeFactorRule",
    "OscillationGatedPlasticity",
    "ElasticWeightConsolidation",
    "ContinualLearner",
    "EpisodicBuffer",
    "ConsolidationEngine",
    "SleepReplayConsolidation",
    "EPKuramotoClassifier",
    "TsodyksMarkramSynapse",
    "STPLayer",
    "DendriticNeuron",
    "DendriticStack",
]
