"""Test that all 49 PyQuifer modules import cleanly."""
import os
import pytest


def get_module_names():
    """Get all module names from the pyquifer package."""
    module_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'pyquifer')
    return sorted([
        f[:-3] for f in os.listdir(module_dir)
        if f.endswith('.py') and not f.startswith('_')
    ])


def test_module_count():
    """We should have at least 48 modules."""
    assert len(get_module_names()) >= 48


@pytest.mark.parametrize("module_name", get_module_names())
def test_module_imports(module_name):
    """Each module should import without errors."""
    __import__(f'pyquifer.{module_name}')


PHASE2_CLASSES = [
    'PrecisionEstimator', 'PrecisionGate', 'AttentionAsPrecision',
    'PredictiveLevel', 'HierarchicalPredictiveCoding',
    'WinnerlessCompetition', 'HeteroclinicChannel', 'MetastabilityIndex',
    'AdaptiveStochasticResonance', 'ResonanceMonitor',
    'TransferEntropyEstimator', 'CausalFlowMap', 'DominanceDetector',
    'EpisodicBuffer', 'SharpWaveRipple', 'ConsolidationEngine', 'MemoryReconsolidation',
    'MarkovBlanket', 'SelfModel', 'NarrativeIdentity',
    'NeuronalGroup', 'SelectionArena', 'SymbiogenesisDetector',
]


@pytest.mark.parametrize("class_name", PHASE2_CLASSES)
def test_lazy_import(class_name):
    """Each Phase 2 class should be accessible via pyquifer.ClassName."""
    import pyquifer
    cls = getattr(pyquifer, class_name)
    assert cls is not None
