"""Learning rules: synaptic plasticity, continual learning, consolidation, and dendritic credit."""
from pyquifer.learning.consolidation import (
    ConsolidationEngine,
    EpisodicBuffer,
    SleepReplayConsolidation,
)
from pyquifer.learning.continual import (
    ContinualLearner,
    ElasticWeightConsolidation,
)
from pyquifer.learning.dendritic import DendriticNeuron, DendriticStack
from pyquifer.learning.equilibrium_prop import EPKuramotoClassifier
from pyquifer.learning.stp import STPLayer, TsodyksMarkramSynapse
from pyquifer.learning.synaptic import (
    ContrastiveHebbian,
    DifferentiablePlasticity,
    EligibilityTrace,
    OscillationGatedPlasticity,
    PredictiveCoding,
    RewardModulatedHebbian,
    ThreeFactorRule,
)

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
