"""Experimental modules: quantum, hyperdimensional, hyperbolic, reservoir computing, FHRR."""
from pyquifer.experimental.fhrr import FHRREncoder
from pyquifer.experimental.hyperbolic import (
    EmotionalGravityManifold,
    HyperbolicOperations,
)
from pyquifer.experimental.hyperdimensional import HDCReasoner, HypervectorMemory
from pyquifer.experimental.quantum import QuantumDecisionMaker
from pyquifer.experimental.reservoir import CriticalReservoir, EchoStateNetwork

__all__ = [
    "QuantumDecisionMaker",
    "HypervectorMemory",
    "HDCReasoner",
    "HyperbolicOperations",
    "EmotionalGravityManifold",
    "EchoStateNetwork",
    "CriticalReservoir",
    "FHRREncoder",
]
