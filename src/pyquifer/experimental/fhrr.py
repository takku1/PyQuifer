"""
Fourier Holographic Reduced Representations (FHRR) — spike-timing VSA.

Encodes information in the timing of spikes using Fourier-domain
holographic representations. Operations like binding (circular
convolution) and superposition (element-wise sum) are performed
on spike trains, enabling Vector Symbolic Architecture on
neuromorphic hardware.

Key classes:
- FHRREncoder: Encode values as complex phasors → spike times
- LatencyEncoder: Encode values as time-to-first-spike
- SpikeVSAOps: Binding/superposition/similarity on spike trains
- NeuromorphicExporter: Convert to Lava/Loihi format (stub)

References:
- Plate (2003). Holographic Reduced Representations.
- Frady et al. (2022). Computing with high-dimensional vectors using
  thinned element-wise transformations. Frontiers in Neuroscience.
- Kleyko et al. (2023). A Survey on Hyperdimensional Computing. ACM Survey.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple, List


class FHRREncoder(nn.Module):
    """Encode values as Fourier Holographic Reduced Representations.

    Each symbol is represented as a complex phasor vector:
    x = [exp(i*phi_1), exp(i*phi_2), ..., exp(i*phi_D)]

    Binding: element-wise complex multiplication (circular convolution)
    Unbinding: complex conjugate multiplication
    Superposition: element-wise sum + normalization

    Args:
        dim: Hypervector dimension.
        num_symbols: Size of codebook for discrete symbols.
    """

    def __init__(self, dim: int, num_symbols: int = 256):
        super().__init__()
        self.dim = dim
        self.num_symbols = num_symbols

        # Random phase codebook for discrete symbols
        self.register_buffer(
            'codebook_phases',
            torch.rand(num_symbols, dim) * 2 * math.pi,
        )

        # Continuous encoder
        self.continuous_encoder = nn.Sequential(
            nn.Linear(1, dim),
            nn.Tanh(),
        )

    def encode_discrete(self, indices: torch.Tensor) -> torch.Tensor:
        """Encode discrete symbols as phasor vectors.

        Args:
            indices: (...,) integer symbol indices

        Returns:
            (..., dim) complex phasor vectors (as phases in [0, 2pi])
        """
        return self.codebook_phases[indices]

    def encode_continuous(self, values: torch.Tensor) -> torch.Tensor:
        """Encode continuous values as phasor vectors.

        Args:
            values: (...,) scalar values

        Returns:
            (..., dim) phase vectors
        """
        v = values.unsqueeze(-1) if values.dim() == 0 or values.shape[-1] != 1 else values
        phases = self.continuous_encoder(v) * math.pi  # Map to [-pi, pi]
        return phases % (2 * math.pi)

    def to_complex(self, phases: torch.Tensor) -> torch.Tensor:
        """Convert phase representation to complex phasors.

        Args:
            phases: (..., dim) phase angles

        Returns:
            (..., dim) complex tensor
        """
        return torch.exp(1j * phases.to(torch.float32)).to(torch.complex64)

    def from_complex(self, z: torch.Tensor) -> torch.Tensor:
        """Extract phases from complex phasors.

        Args:
            z: (..., dim) complex tensor

        Returns:
            (..., dim) phase angles in [0, 2pi]
        """
        return torch.angle(z) % (2 * math.pi)


class LatencyEncoder(nn.Module):
    """Encode values as time-to-first-spike latencies.

    Higher values produce earlier spikes (shorter latency).
    This is the standard encoding for neuromorphic hardware
    where information is in spike timing.

    Args:
        dim: Output spike dimension.
        max_latency: Maximum spike latency (timesteps).
        min_latency: Minimum spike latency.
    """

    def __init__(
        self,
        dim: int,
        max_latency: float = 100.0,
        min_latency: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_latency = max_latency
        self.min_latency = min_latency

        self.encoder = nn.Linear(1, dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Encode values as spike latencies.

        Args:
            values: (...,) values in [0, 1]

        Returns:
            (..., dim) spike latencies
        """
        v = values.unsqueeze(-1) if values.dim() == 0 or values.shape[-1] != 1 else values
        # Project and map to latency range
        projected = torch.sigmoid(self.encoder(v))
        latencies = self.max_latency - projected * (self.max_latency - self.min_latency)
        return latencies

    def decode(self, latencies: torch.Tensor) -> torch.Tensor:
        """Decode spike latencies back to values.

        Args:
            latencies: (..., dim) spike latencies

        Returns:
            (...,) decoded values
        """
        normalized = (self.max_latency - latencies) / (self.max_latency - self.min_latency)
        return normalized.mean(dim=-1).clamp(0, 1)


class SpikeVSAOps(nn.Module):
    """Vector Symbolic Architecture operations on spike representations.

    Implements the three core VSA operations using spike-compatible
    representations:
    1. Binding: combine two concepts (role + filler)
    2. Superposition: create sets/bundles
    3. Similarity: compare representations

    Args:
        dim: Hypervector dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def bind(
        self,
        a_phases: torch.Tensor,
        b_phases: torch.Tensor,
    ) -> torch.Tensor:
        """Bind two FHRR vectors (circular convolution in phase domain).

        In frequency domain, circular convolution = element-wise multiplication.
        For phasors: binding = phase addition.

        Args:
            a_phases: (..., dim) phases of first vector
            b_phases: (..., dim) phases of second vector

        Returns:
            (..., dim) bound phases
        """
        return (a_phases + b_phases) % (2 * math.pi)

    def unbind(
        self,
        bound_phases: torch.Tensor,
        key_phases: torch.Tensor,
    ) -> torch.Tensor:
        """Unbind (retrieve filler from bound representation).

        Inverse of binding: subtract key phases.

        Args:
            bound_phases: (..., dim) bound representation
            key_phases: (..., dim) key to unbind with

        Returns:
            (..., dim) retrieved phases
        """
        return (bound_phases - key_phases) % (2 * math.pi)

    def superpose(
        self,
        vectors: List[torch.Tensor],
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Superpose multiple FHRR vectors (bundle/set creation).

        Sum complex phasors and extract resulting phases.

        Args:
            vectors: List of (..., dim) phase vectors
            weights: Optional (..., N) weights for each vector

        Returns:
            (..., dim) superposed phases
        """
        complex_sum = torch.zeros(
            *vectors[0].shape, dtype=torch.complex64,
            device=vectors[0].device,
        )

        for i, v in enumerate(vectors):
            phasor = torch.exp(1j * v.to(torch.float32)).to(torch.complex64)
            if weights is not None:
                w = weights[..., i:i+1].to(torch.complex64)
                phasor = phasor * w
            complex_sum = complex_sum + phasor

        return torch.angle(complex_sum) % (2 * math.pi)

    def similarity(
        self,
        a_phases: torch.Tensor,
        b_phases: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity between FHRR vectors.

        Args:
            a_phases: (..., dim) first phases
            b_phases: (..., dim) second phases

        Returns:
            (...,) similarity in [-1, 1]
        """
        diff = a_phases - b_phases
        return torch.cos(diff).mean(dim=-1)

    def resonator_decode(
        self,
        superposed: torch.Tensor,
        codebook: torch.Tensor,
        num_iterations: int = 50,
    ) -> torch.Tensor:
        """Decode superposition using resonator network.

        Iterative factorization: find which codebook elements
        were superposed.

        Args:
            superposed: (dim,) superposed phase vector
            codebook: (K, dim) codebook of possible components
            num_iterations: Resonator iterations

        Returns:
            (K,) similarity scores for each codebook entry
        """
        K = codebook.shape[0]

        # Initialize estimate
        estimate = superposed.clone()

        for _ in range(num_iterations):
            # Compute similarity with all codebook entries
            sims = torch.cos(estimate.unsqueeze(0) - codebook).mean(dim=-1)
            # Sharpen
            weights = F.softmax(sims * 10, dim=0)
            # Update estimate toward weighted codebook
            complex_est = torch.zeros(self.dim, dtype=torch.complex64, device=superposed.device)
            for k in range(K):
                phasor = torch.exp(1j * codebook[k].to(torch.float32)).to(torch.complex64)
                complex_est = complex_est + weights[k] * phasor
            estimate = torch.angle(complex_est) % (2 * math.pi)

        # Final similarity scores
        return torch.cos(estimate.unsqueeze(0) - codebook).mean(dim=-1)


class NeuromorphicExporter:
    """Export PyQuifer SNN models to neuromorphic formats.

    Provides conversion stubs for Intel Loihi (Lava) and
    other neuromorphic hardware platforms.

    Args:
        target: Target platform ('loihi', 'spinnaker', 'generic').
    """

    def __init__(self, target: str = 'generic'):
        self.target = target

    def export_weights(
        self,
        model: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Extract integer-quantized weights for neuromorphic hardware.

        Args:
            model: PyTorch model to export

        Returns:
            Dict of quantized weight tensors
        """
        weights = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Quantize to 8-bit integers (Loihi-compatible)
                scale = param.abs().max() / 127.0
                if scale > 0:
                    quantized = (param / scale).round().clamp(-128, 127).to(torch.int8)
                else:
                    quantized = torch.zeros_like(param, dtype=torch.int8)
                weights[name] = quantized
                weights[f'{name}_scale'] = scale

        return weights

    def export_config(
        self,
        model: nn.Module,
    ) -> Dict:
        """Export model configuration for hardware mapping.

        Args:
            model: PyTorch model

        Returns:
            Dict with hardware configuration
        """
        config = {
            'target': self.target,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'layers': [],
        }

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                config['layers'].append({
                    'name': name,
                    'type': 'dense',
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                })

        return config
