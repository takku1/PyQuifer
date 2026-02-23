"""
NeuroBench adapter for PyQuifer's spiking neural network stack.

Wraps PyQuifer's SpikingLayer / OscillatorySNN in the NeuroBenchModel interface
so neurobench's Benchmark harness can measure:

  Static:   ParameterCount, Footprint, ConnectionSparsity
  Workload: ActivationSparsity, ClassificationAccuracy

Gap note
--------
MembraneUpdates and SynapticOperations both call ``requires_hooks=True`` and
look for ``snn.SpikingNeuron`` (snntorch) instances on the hook objects.
PyQuifer's LIFNeuron is a custom implementation that does NOT inherit from
``snntorch.SpikingNeuron``.  Those two metrics are therefore skipped in the
current adapter; see gap_neurobench_snn_hooks in the eval gap registry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from neurobench.models import NeuroBenchModel

from pyquifer.dynamics.spiking.neurons import OscillatorySNN, SpikingLayer


# ---------------------------------------------------------------------------
# Composite network: SpikingLayer backbone + linear readout
# ---------------------------------------------------------------------------

class _SpikingClassifier(nn.Module):
    """
    SpikingLayer → rate-decode → Linear readout.

    Input:  (batch, seq_len, input_dim) — spike-encoded or continuous features.
    Output: (batch, num_classes)         — raw logits.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 tau: float = 10.0, threshold: float = 0.3, recurrent: bool = True):
        super().__init__()
        # threshold=0.3: at Kaiming init, the projected current per step is
        # ≈ sqrt(input_dim) * mean_input * weight_std * dt/tau.  With input_dim=16,
        # mean Bernoulli rate ~0.3, weight_std ~1/sqrt(input_dim)=0.25, dt/tau=0.1:
        # ≈ 4 * 0.3 * 0.25 * 0.1 ≈ 0.03 per step.  At threshold=1.0 the steady-state
        # membrane ~0.3 never fires.  threshold=0.3 gives sparse but real spiking.
        spiking_layer = SpikingLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            tau=tau,
            recurrent=recurrent,
        )
        # Override the LIF threshold to a calibrated value
        with torch.no_grad():
            spiking_layer.lif.threshold.fill_(threshold)
        self.spiking = spiking_layer
        self.readout = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spikes = self.spiking(x)          # (batch, seq_len, hidden_dim)
        rate = spikes.mean(dim=1)          # (batch, hidden_dim)  — rate decode
        return self.readout(rate)          # (batch, num_classes)


class _OscillatoryClassifier(nn.Module):
    """
    OscillatorySNN backbone + linear readout.

    Input:  (batch, seq_len, input_dim)
    Output: (batch, num_classes)
    """

    def __init__(self, input_dim: int, num_excitatory: int, num_inhibitory: int,
                 num_classes: int, steps_per_input: int = 5):
        super().__init__()
        self.osnn = OscillatorySNN(
            input_dim=input_dim,
            num_excitatory=num_excitatory,
            num_inhibitory=num_inhibitory,
        )
        self.steps_per_input = steps_per_input
        self.readout = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _rates = self.osnn(x, steps_per_input=self.steps_per_input)
        # Pool over time
        pooled = output.mean(dim=1)       # (batch, input_dim)
        return self.readout(pooled)        # (batch, num_classes)


# ---------------------------------------------------------------------------
# NeuroBenchModel adapter
# ---------------------------------------------------------------------------

class PyQuiferSNNModel(NeuroBenchModel):
    """
    NeuroBenchModel adapter for PyQuifer's SpikingLayer-based classifiers.

    Usage::

        net = _SpikingClassifier(input_dim=16, hidden_dim=64, num_classes=4)
        model = PyQuiferSNNModel(net)
        benchmark = Benchmark(model, dataloader, [], [], [[static_metrics], [workload_metrics]])
        results = benchmark.run()

    Notes
    -----
    * ``SpikingLayer`` is added as a recognized activation module so that
      neurobench's hook system captures spike tensors for ActivationSparsity.
    * The model returns *class index* predictions (argmax) so ClassificationAccuracy
      can compare directly against integer targets.
    * MembraneUpdates / SynapticOperations are not supported — see module docstring.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self._net = net
        # Tell neurobench to attach hooks to SpikingLayer instances so that
        # activation_outputs captures the (batch, seq_len, hidden) spike tensor.
        self.add_activation_module(SpikingLayer)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Run inference on one batch.

        Neurobench's Benchmark calls this with the preprocessed data tensor
        (output of processor_manager.preprocess), NOT the (data, targets) tuple.

        Args:
            batch: Data tensor ``(batch, seq_len, features)``.

        Returns:
            Predicted class indices, shape ``(batch,)``.
        """
        with torch.no_grad():
            logits = self._net(batch)
        return logits.argmax(dim=-1)

    def __net__(self) -> nn.Module:
        return self._net


class PyQuiferOscillatoryModel(PyQuiferSNNModel):
    """
    NeuroBenchModel adapter for PyQuifer's OscillatorySNN.

    Adds OscillatorySNN to recognized activation modules so that the
    oscillatory output can be measured for sparsity.
    """

    def __init__(self, net: _OscillatoryClassifier):
        super().__init__(net)
        self.add_activation_module(OscillatorySNN)


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def make_spiking_model(input_dim: int = 16, hidden_dim: int = 64,
                       num_classes: int = 4) -> PyQuiferSNNModel:
    """Build a default SpikingLayer-based NeuroBenchModel."""
    net = _SpikingClassifier(input_dim=input_dim, hidden_dim=hidden_dim,
                             num_classes=num_classes)
    return PyQuiferSNNModel(net)


def make_oscillatory_model(input_dim: int = 16, num_excitatory: int = 64,
                           num_inhibitory: int = 16,
                           num_classes: int = 4) -> PyQuiferOscillatoryModel:
    """Build a default OscillatorySNN-based NeuroBenchModel."""
    net = _OscillatoryClassifier(
        input_dim=input_dim,
        num_excitatory=num_excitatory,
        num_inhibitory=num_inhibitory,
        num_classes=num_classes,
    )
    return PyQuiferOscillatoryModel(net)
