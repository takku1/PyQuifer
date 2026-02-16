"""
Energy-Optimized Spiking Predictive Coding.

Unifies spiking dynamics and predictive coding through an energy
minimization principle. Spiking activity emerges from energy
optimization, and prediction errors are naturally represented
in multi-compartment neuron dynamics.

Key classes:
- EnergyOptimizedSNN: Spiking network where PC emerges from energy minimization
- MultiCompartmentSpikingPC: Multi-compartment spiking + predictive coding
- EnergyLandscape: Visualize and analyze energy surface

References:
- Rao & Ballard (1999). Predictive coding in the visual cortex.
- Bogacz (2017). A tutorial on the free-energy framework for modelling
  perception and learning. Journal of Mathematical Psychology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List


class SpikingPCNeuron(nn.Module):
    """Single spiking predictive coding neuron.

    Combines LIF spiking dynamics with energy-based predictive coding.
    The membrane potential represents the energy state, and spikes
    occur when the energy exceeds a threshold.

    Prediction error = input - top_down_prediction
    Energy = prediction_error^2 + regularization

    Args:
        dim: Neuron dimension.
        threshold: Spike threshold.
        tau_mem: Membrane time constant.
        tau_syn: Synaptic time constant.
    """

    def __init__(
        self,
        dim: int,
        threshold: float = 1.0,
        tau_mem: float = 20.0,
        tau_syn: float = 5.0,
    ):
        super().__init__()
        self.dim = dim
        self.threshold = threshold
        self.alpha = math.exp(-1.0 / tau_mem)
        self.beta = math.exp(-1.0 / tau_syn)

        self.register_buffer('membrane', torch.zeros(dim))
        self.register_buffer('synaptic', torch.zeros(dim))

    def reset_state(self, batch_size: int = 1, device: torch.device = None):
        """Reset neuron state."""
        dev = device or self.membrane.device
        self.membrane = torch.zeros(batch_size, self.dim, device=dev)
        self.synaptic = torch.zeros(batch_size, self.dim, device=dev)

    def forward(
        self,
        input_current: torch.Tensor,
        prediction: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Step spiking PC neuron.

        Args:
            input_current: (..., dim) input current
            prediction: Optional (..., dim) top-down prediction

        Returns:
            Dict with 'spikes', 'membrane', 'prediction_error', 'energy'
        """
        # Prediction error drives the neuron
        if prediction is not None:
            error = input_current - prediction
        else:
            error = input_current

        # Synaptic dynamics
        self.synaptic = self.beta * self.synaptic + error

        # Membrane dynamics (LIF with prediction error input)
        self.membrane = self.alpha * self.membrane + self.synaptic

        # Spike generation (surrogate gradient for training)
        spikes = (self.membrane >= self.threshold).float()

        # Soft reset (subtract threshold on spike)
        self.membrane = self.membrane - spikes * self.threshold

        # Energy: prediction error squared
        energy = (error ** 2).sum(dim=-1)

        return {
            'spikes': spikes,
            'membrane': self.membrane.clone(),
            'prediction_error': error,
            'energy': energy,
        }


class EnergyOptimizedSNN(nn.Module):
    """Spiking network where predictive coding emerges from energy minimization.

    Multi-layer spiking network where each layer:
    1. Receives feedforward input (bottom-up)
    2. Receives predictions from higher layer (top-down)
    3. Minimizes prediction error through spiking dynamics
    4. Sends residual errors upward

    The network naturally implements predictive coding through
    its energy-based dynamics.

    Args:
        dims: List of layer dimensions [input, hidden..., output].
        threshold: Spike threshold.
        tau_mem: Membrane time constant.
        num_steps: Number of spiking timesteps per forward pass.
    """

    def __init__(
        self,
        dims: List[int],
        threshold: float = 1.0,
        tau_mem: float = 20.0,
        num_steps: int = 25,
    ):
        super().__init__()
        self.dims = dims
        self.num_layers = len(dims) - 1
        self.num_steps = num_steps

        # Spiking PC neurons per layer
        self.neurons = nn.ModuleList([
            SpikingPCNeuron(d, threshold, tau_mem)
            for d in dims[1:]
        ])

        # Feedforward connections (bottom-up)
        self.ff_connections = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=False)
            for i in range(self.num_layers)
        ])

        # Feedback connections (top-down predictions)
        self.fb_connections = nn.ModuleList([
            nn.Linear(dims[i + 1], dims[i], bias=False)
            for i in range(self.num_layers)
        ])

    def reset(self, batch_size: int = 1, device: torch.device = None):
        """Reset all neuron states."""
        for neuron in self.neurons:
            neuron.reset_state(batch_size, device)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run spiking predictive coding network.

        Args:
            x: (B, dims[0]) input features (encoded as rates)

        Returns:
            Dict with 'output_spikes', 'spike_counts', 'total_energy',
            'layer_energies', 'spike_rates'
        """
        B = x.shape[0]
        self.reset(B, x.device)

        all_spike_counts = [
            torch.zeros(B, d, device=x.device) for d in self.dims[1:]
        ]
        total_energy = torch.zeros(B, device=x.device)

        for t in range(self.num_steps):
            # Forward pass: compute feedforward currents
            currents = []
            h = x
            for i in range(self.num_layers):
                # Use previous layer's membrane state for feedforward
                if i > 0:
                    h = self.neurons[i - 1].membrane.detach()
                current = self.ff_connections[i](h)
                currents.append(current)

            # Feedback pass: compute top-down predictions
            predictions = [None] * self.num_layers
            for i in range(self.num_layers - 1, 0, -1):
                predictions[i - 1] = self.fb_connections[i](
                    self.neurons[i].membrane.detach()
                )

            # Update each layer
            for i in range(self.num_layers):
                result = self.neurons[i](currents[i], predictions[i])
                all_spike_counts[i] += result['spikes']
                total_energy += result['energy']

        # Compute spike rates
        spike_rates = [counts / self.num_steps for counts in all_spike_counts]

        return {
            'output_spikes': all_spike_counts[-1],
            'spike_counts': all_spike_counts,
            'total_energy': total_energy / self.num_steps,
            'spike_rates': spike_rates,
            'output_rate': spike_rates[-1],
        }


class MultiCompartmentSpikingPC(nn.Module):
    """Multi-compartment spiking predictive coding.

    Each neuron has separate compartments for:
    - Basal: feedforward prediction errors
    - Apical: feedback predictions
    - Soma: integration and spike generation

    The energy landscape of the network implements hierarchical
    predictive coding through spiking dynamics.

    Args:
        dim: Layer dimension.
        num_compartments: Number of dendritic compartments.
        threshold: Spike threshold.
        tau_mem: Membrane time constant.
    """

    def __init__(
        self,
        dim: int,
        num_compartments: int = 3,
        threshold: float = 1.0,
        tau_mem: float = 20.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_compartments = num_compartments
        self.threshold = threshold
        self.alpha = math.exp(-1.0 / tau_mem)

        # Per-compartment processing
        self.compartment_weights = nn.Parameter(
            torch.ones(num_compartments) / num_compartments
        )

        # Compartment interactions
        self.interaction = nn.Linear(dim * num_compartments, dim)

        # State
        self.register_buffer(
            'compartment_voltages',
            torch.zeros(num_compartments, dim),
        )
        self.register_buffer('soma_voltage', torch.zeros(dim))

    def reset_state(self, batch_size: int = 1, device: torch.device = None):
        """Reset compartment states."""
        dev = device or self.soma_voltage.device
        self.compartment_voltages = torch.zeros(
            batch_size, self.num_compartments, self.dim, device=dev,
        )
        self.soma_voltage = torch.zeros(batch_size, self.dim, device=dev)

    def forward(
        self,
        compartment_inputs: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Step multi-compartment neuron.

        Args:
            compartment_inputs: List of (B, dim) inputs, one per compartment

        Returns:
            Dict with 'spikes', 'soma_voltage', 'compartment_energies',
            'total_energy'
        """
        B = compartment_inputs[0].shape[0]

        # Ensure state is right size
        if self.soma_voltage.shape[0] != B:
            self.reset_state(B, compartment_inputs[0].device)

        # Update each compartment
        comp_energies = []
        for c in range(min(self.num_compartments, len(compartment_inputs))):
            self.compartment_voltages[:, c] = (
                self.alpha * self.compartment_voltages[:, c] +
                compartment_inputs[c]
            )
            comp_energies.append(
                (compartment_inputs[c] ** 2).sum(dim=-1)
            )

        # Somatic integration
        weights = F.softmax(self.compartment_weights, dim=0)
        weighted = torch.zeros(B, self.dim, device=compartment_inputs[0].device)
        for c in range(self.num_compartments):
            weighted += weights[c] * self.compartment_voltages[:, c]

        # Nonlinear interaction
        flat_comp = self.compartment_voltages.view(B, -1)
        interaction = self.interaction(flat_comp)

        self.soma_voltage = self.alpha * self.soma_voltage + weighted + interaction

        # Spike
        spikes = (self.soma_voltage >= self.threshold).float()
        self.soma_voltage = self.soma_voltage - spikes * self.threshold

        total_energy = sum(comp_energies)

        return {
            'spikes': spikes,
            'soma_voltage': self.soma_voltage.clone(),
            'compartment_energies': comp_energies,
            'total_energy': total_energy,
        }


class EnergyLandscape(nn.Module):
    """Analyze and characterize the energy surface of spiking dynamics.

    Computes energy metrics that characterize the network's
    computational state: local minima count, barrier heights,
    basin sizes, curvature statistics.

    Args:
        dim: State dimension.
        num_probes: Number of random probe directions.
    """

    def __init__(self, dim: int, num_probes: int = 32):
        super().__init__()
        self.dim = dim
        self.num_probes = num_probes

        # Energy function (Hopfield-like)
        self.W = nn.Parameter(torch.randn(dim, dim) * 0.01)
        # Make symmetric
        with torch.no_grad():
            self.W.copy_((self.W + self.W.T) / 2)
            self.W.fill_diagonal_(0)

        self.bias = nn.Parameter(torch.zeros(dim))

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """Compute energy of a state.

        Args:
            state: (..., dim)

        Returns:
            (...,) energy
        """
        Ws = F.linear(state, self.W)
        return -0.5 * (state * Ws).sum(dim=-1) - (self.bias * state).sum(dim=-1)

    def analyze(
        self,
        state: torch.Tensor,
        probe_radius: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Analyze energy landscape around a state.

        Args:
            state: (B, dim) states to analyze
            probe_radius: Radius for landscape probing

        Returns:
            Dict with 'energy', 'gradient_norm', 'curvature_estimate',
            'local_roughness', 'is_minimum'
        """
        B = state.shape[0]
        device = state.device

        # Current energy
        current_energy = self.energy(state)

        # Gradient
        state_grad = state.clone().requires_grad_(True)
        e = self.energy(state_grad)
        grad = torch.autograd.grad(e.sum(), state_grad)[0]
        grad_norm = grad.norm(dim=-1)

        # Probe landscape with random directions
        probes = torch.randn(self.num_probes, self.dim, device=device)
        probes = probes / probes.norm(dim=-1, keepdim=True) * probe_radius

        probe_energies = []
        for p in range(self.num_probes):
            probed = state + probes[p].unsqueeze(0)
            probe_energies.append(self.energy(probed))

        probe_stack = torch.stack(probe_energies, dim=1)  # (B, num_probes)

        # Curvature estimate: mean second difference
        curvature = (probe_stack - current_energy.unsqueeze(1)).mean(dim=1) / (probe_radius ** 2)

        # Roughness: variance of nearby energies
        roughness = probe_stack.var(dim=1)

        # Is minimum: all probes have higher energy
        is_minimum = (probe_stack >= current_energy.unsqueeze(1) - 1e-6).all(dim=1)

        return {
            'energy': current_energy,
            'gradient_norm': grad_norm,
            'curvature_estimate': curvature,
            'local_roughness': roughness,
            'is_minimum': is_minimum,
        }
