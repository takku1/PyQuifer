"""Predictive processing: hierarchical prediction, active inference, JEPA world models."""
from pyquifer.cognition.predictive.active_inference import ActiveInferenceAgent
from pyquifer.cognition.predictive.deep_active_inference import DeepAIF
from pyquifer.cognition.predictive.hierarchical import (
    HierarchicalPredictiveCoding,
    OscillatoryPredictiveCoding,
)
from pyquifer.cognition.predictive.jepa import ActionJEPA, JEPAEncoder

__all__ = [
    "HierarchicalPredictiveCoding",
    "OscillatoryPredictiveCoding",
    "ActiveInferenceAgent",
    "DeepAIF",
    "JEPAEncoder",
    "ActionJEPA",
]
