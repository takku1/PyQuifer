"""
Spherical Kuramoto - Oscillators on hyperspheres with tangent space projection.

Extracted from akorn repository patterns. Key innovations:
- Oscillators are points on a hypersphere (unit vectors)
- Updates projected onto tangent space to stay on manifold
- Learnable per-channel natural frequencies
- Conv or attention-based coupling

This is more geometrically correct than standard Kuramoto.
"""

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


def reshape_to_groups(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Reshape tensor to group oscillators.

    Args:
        x: Input tensor [B, C, ...] or [B, T, C]
        n: Number of components per oscillator (e.g., 2 for phases on circle)

    Returns:
        Reshaped tensor with oscillator groups
    """
    if x.ndim == 3:  # [B, T, C] - sequence
        return x.transpose(1, 2).unflatten(1, (-1, n))
    else:  # [B, C, ...] - spatial
        return x.unflatten(1, (-1, n))


def reshape_from_groups(x: torch.Tensor) -> torch.Tensor:
    """Inverse of reshape_to_groups."""
    if x.ndim == 4:  # Tokens
        return x.flatten(1, 2).transpose(1, 2)
    else:
        return x.flatten(1, 2)


def l2_normalize(x: torch.Tensor, dim: int = 2) -> torch.Tensor:
    """L2 normalize along dimension."""
    return torch.nn.functional.normalize(x, dim=dim)


def normalize_oscillators(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Normalize oscillators to unit hypersphere.

    Args:
        x: Oscillator states [B, C, ...]
        n: Components per oscillator

    Returns:
        Normalized states on unit hypersphere
    """
    x = reshape_to_groups(x, n)
    x = l2_normalize(x)
    x = reshape_from_groups(x)
    return x


def exponential_map(x: torch.Tensor, v: torch.Tensor, n: int) -> torch.Tensor:
    """
    Exponential map on hypersphere - move along geodesic.

    Args:
        x: Current position on sphere [B, C, ...]
        v: Tangent vector (velocity) [B, C, ...]
        n: Components per oscillator

    Returns:
        New position after geodesic step
    """
    v = reshape_to_groups(v, n)
    x = reshape_to_groups(x, n)

    # Magnitude of velocity (geodesic distance)
    norm = torch.linalg.norm(v, dim=2, keepdim=True)
    norm = torch.clamp(norm, 0, math.pi)  # Clamp to avoid wrapping

    # Geodesic step: cos(||v||)*x + sin(||v||)*(v/||v||)
    new_x = torch.cos(norm) * x + torch.sin(norm) * (v / (norm + 1e-8))

    return reshape_from_groups(new_x)


def _params_to_skew_symmetric(params: torch.Tensor, n: int) -> torch.Tensor:
    """Convert flat parameter vector to n x n skew-symmetric matrix.

    For oscillators on S^(n-1), the natural frequency is an element of so(n),
    the Lie algebra of skew-symmetric matrices. This has n*(n-1)/2 free
    parameters (the upper triangle).

    Args:
        params: [..., n*(n-1)//2] flat upper-triangle entries
        n: Matrix dimension

    Returns:
        [..., n, n] skew-symmetric matrix (A = -A^T)
    """
    batch_shape = params.shape[:-1]
    mat = torch.zeros(*batch_shape, n, n, device=params.device, dtype=params.dtype)
    idx = torch.triu_indices(n, n, offset=1)
    mat[..., idx[0], idx[1]] = params
    return mat - mat.transpose(-2, -1)


class LearnableOmega(nn.Module):
    """
    Learnable natural frequencies for oscillators on S^(n-1).

    Each oscillator's natural frequency is a skew-symmetric matrix in so(n),
    the Lie algebra of the rotation group SO(n). The velocity at point x
    on the sphere is v = Omega @ x where Omega is skew-symmetric.

    Dimension-specific behavior:
    - n=2 (S^1, circle): 1 free parameter, recovers standard (-omega*y, omega*x)
    - n=3 (S^2, sphere): 3 free parameters (3D rotation generators)
    - n=4 (S^3, hypersphere): 6 free parameters (quaternionic rotations)
    - General S^(n-1): n*(n-1)/2 free parameters

    Extracted from akorn's OmegaLayer, generalized to arbitrary dimension.
    """

    def __init__(self,
                 num_oscillators: int,
                 components_per_oscillator: int = 2,
                 init_omega: float = 0.1,
                 global_omega: bool = False,
                 learnable: bool = True):
        """
        Args:
            num_oscillators: Number of oscillators (channels // components)
            components_per_oscillator: Dimensionality of each oscillator (2 for circle)
            init_omega: Initial frequency magnitude
            global_omega: If True, single frequency for all; else per-oscillator
            learnable: Whether frequencies are trainable
        """
        super().__init__()
        self.num_oscillators = num_oscillators
        self.n = components_per_oscillator
        self.global_omega = global_omega

        # Number of free parameters in so(n)
        n_params = self.n * (self.n - 1) // 2

        # Initialize so each skew-symmetric matrix has Frobenius norm ~ init_omega * sqrt(2)
        # For n=2 this gives a single scalar whose abs is init_omega (backward compat)
        scale = init_omega / max(math.sqrt(n_params), 1.0)

        if global_omega:
            self.omega_param = nn.Parameter(
                scale * torch.ones(n_params),
                requires_grad=learnable
            )
        else:
            self.omega_param = nn.Parameter(
                scale * torch.ones(num_oscillators, n_params),
                requires_grad=learnable
            )

    def get_frequencies(self) -> torch.Tensor:
        """Get frequency magnitudes (Frobenius norm / sqrt(2) of each Omega)."""
        if self.n == 2:
            # Fast path: single parameter per oscillator, norm = |param|
            if self.global_omega:
                return self.omega_param.abs().repeat(self.num_oscillators)
            else:
                return self.omega_param.abs().squeeze(-1)

        # General case: build skew-symmetric, compute Frobenius norm / sqrt(2)
        if self.global_omega:
            omega_mat = _params_to_skew_symmetric(self.omega_param, self.n)
            freq = torch.linalg.norm(omega_mat) / math.sqrt(2)
            return freq.repeat(self.num_oscillators)
        else:
            omega_mat = _params_to_skew_symmetric(self.omega_param, self.n)
            # Frobenius norm per oscillator: sqrt(sum of squares) over last two dims
            freq = torch.linalg.norm(omega_mat.flatten(-2), dim=-1) / math.sqrt(2)
            return freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply natural frequency rotation: v = Omega @ x.

        Args:
            x: Oscillator states [B, C, ...]

        Returns:
            Rotation velocity in tangent space
        """
        _x = reshape_to_groups(x, self.n)
        # _x shape: [B, num_osc, n, ...] for spatial, [B, num_osc, n] for flat

        if self.n == 2:
            # Fast path for n=2: Omega = [[0, -w], [w, 0]], so Omega@x = (-w*x1, w*x0)
            if self.global_omega:
                omega = self.omega_param[0].expand(_x.shape[1])
            else:
                omega = self.omega_param[:, 0]  # [num_osc]

            omega = omega[None]  # [1, num_osc]
            for _ in range(_x.ndim - 3):
                omega = omega.unsqueeze(-1)

            omega_x = torch.stack([-omega * _x[:, :, 1], omega * _x[:, :, 0]], dim=2)
            return reshape_from_groups(omega_x)

        # General case: build skew-symmetric matrix and do matrix-vector product
        if self.global_omega:
            # Single omega for all oscillators: [n, n]
            omega_mat = _params_to_skew_symmetric(self.omega_param, self.n)
            if _x.ndim == 3:
                # [B, num_osc, n] — contract last dim
                omega_x = torch.einsum('ij,boj->boi', omega_mat, _x)
            else:
                # Spatial: [B, O, n, H, W...] — flatten spatial, contract n, restore
                spatial_shape = _x.shape[3:]
                flat = _x.reshape(_x.shape[0], _x.shape[1], self.n, -1)  # [B, O, n, S]
                result = torch.einsum('ij,bosj->bosi',
                                      omega_mat,
                                      flat.permute(0, 1, 3, 2))  # [B, O, S, n]
                omega_x = result.permute(0, 1, 3, 2).reshape_as(_x)
        else:
            # Per-oscillator: [num_osc, n_params] -> [num_osc, n, n]
            omega_mat = _params_to_skew_symmetric(self.omega_param, self.n)
            if _x.ndim == 3:
                # [B, num_osc, n] — standard case
                omega_x = torch.einsum('oij,boj->boi', omega_mat, _x)
            else:
                # Spatial: [B, O, n, H, W...] — flatten spatial, contract n, restore
                spatial_shape = _x.shape[3:]
                flat = _x.reshape(_x.shape[0], _x.shape[1], self.n, -1)  # [B, O, n, S]
                flat_t = flat.permute(0, 1, 3, 2)  # [B, O, S, n]
                result_t = torch.einsum('oij,bosj->bosi', omega_mat, flat_t)  # [B, O, S, n]
                omega_x = result_t.permute(0, 1, 3, 2).reshape_as(_x)

        return reshape_from_groups(omega_x)


class TangentProjection(nn.Module):
    """
    Project vectors onto tangent space of hypersphere.

    Given a point x on the sphere and a vector y,
    computes the component of y perpendicular to x.
    This keeps updates on the manifold.
    """

    def __init__(self, components_per_oscillator: int = 2):
        super().__init__()
        self.n = components_per_oscillator

    def forward(self,
                y: torch.Tensor,
                x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project y onto tangent space at x.

        Args:
            y: Vector to project [B, num_osc, n, ...]
            x: Point on sphere [B, num_osc, n, ...]

        Returns:
            (projected_y, similarity) where:
            - projected_y is perpendicular to x
            - similarity is the dot product (for energy computation)
        """
        # Dot product (similarity)
        sim = x * y

        # Component along x
        y_parallel = torch.sum(sim, dim=2, keepdim=True) * x

        # Perpendicular component (tangent)
        y_tangent = y - y_parallel

        return y_tangent, sim


class SphericalKuramotoLayer(nn.Module):
    """
    Kuramoto oscillator layer on hypersphere.

    Key differences from standard Kuramoto:
    1. States are unit vectors on S^{n-1}
    2. Coupling projected onto tangent space
    3. Learnable natural frequencies
    4. Can use conv or attention for coupling

    Extracted from akorn's KLayer.
    """

    def __init__(self,
                 num_channels: int,
                 components_per_oscillator: int = 2,
                 coupling_type: Literal["conv", "linear"] = "conv",
                 kernel_size: int = 3,
                 use_omega: bool = True,
                 init_omega: float = 1.0,
                 global_omega: bool = False,
                 learn_omega: bool = True,
                 use_projection: bool = True,
                 normalization: Literal["group", "none"] = "group"):
        """
        Args:
            num_channels: Total channels (must be divisible by components)
            components_per_oscillator: Dimensionality (2 for circle, 3 for sphere)
            coupling_type: "conv" for local, "linear" for global coupling
            kernel_size: Kernel size if using conv coupling
            use_omega: Whether to use learnable frequencies
            init_omega: Initial frequency magnitude
            global_omega: Single frequency for all oscillators
            learn_omega: Whether frequencies are trainable
            use_projection: Whether to project onto tangent space
            normalization: Type of normalization ("group" or "none")
        """
        super().__init__()

        assert num_channels % components_per_oscillator == 0

        self.num_channels = num_channels
        self.n = components_per_oscillator
        self.num_oscillators = num_channels // components_per_oscillator
        self.use_omega = use_omega
        self.use_projection = use_projection

        # Natural frequencies
        if use_omega:
            self.omega = LearnableOmega(
                num_oscillators=self.num_oscillators,
                components_per_oscillator=components_per_oscillator,
                init_omega=init_omega,
                global_omega=global_omega,
                learnable=learn_omega
            )
        else:
            self.omega = None

        # Coupling connectivity
        if coupling_type == "conv":
            self.coupling = nn.Conv2d(
                num_channels, num_channels,
                kernel_size, 1, kernel_size // 2
            )
        else:  # linear
            self.coupling = nn.Linear(num_channels, num_channels)

        # Normalization
        if normalization == "group":
            self.norm = nn.GroupNorm(self.num_oscillators, num_channels)
        else:
            self.norm = nn.Identity()

        # Tangent projection
        if use_projection:
            self.projection = TangentProjection(components_per_oscillator)

    def kuramoto_update(self,
                       x: torch.Tensor,
                       external_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute one Kuramoto update step.

        Args:
            x: Current oscillator states [B, C, H, W] or [B, C]
            external_input: External driving signal (same shape)

        Returns:
            (dx/dt, energy) where:
            - dx/dt is the state derivative
            - energy is the coupling energy (for monitoring)
        """
        # Coupling force from neighbors
        y = self.coupling(x)

        # Add external input
        y = y + external_input

        # Natural frequency contribution
        if self.omega is not None:
            omega_x = self.omega(x)
        else:
            omega_x = torch.zeros_like(x)

        # Reshape for per-oscillator operations
        y = reshape_to_groups(y, self.n)
        x_grouped = reshape_to_groups(x, self.n)

        # Project onto tangent space
        if self.use_projection:
            y_tangent, similarity = self.projection(y, x_grouped)
        else:
            y_tangent = y
            similarity = y * x_grouped

        # Total derivative
        dxdt = omega_x + reshape_from_groups(y_tangent)

        # Energy (negative similarity = lower energy when aligned)
        energy = reshape_from_groups(similarity)

        return dxdt, energy

    def forward(self,
               x: torch.Tensor,
               external_input: torch.Tensor,
               num_steps: int,
               step_size: float) -> Tuple[list, list]:
        """
        Run Kuramoto dynamics for multiple steps.

        Args:
            x: Initial states [B, C, ...]
            external_input: External input (normalized internally)
            num_steps: Number of integration steps
            step_size: Integration step size (gamma)

        Returns:
            (states, energies) where:
            - states is list of states at each step
            - energies is list of energy at each step
        """
        states = []
        energies = []

        # Normalize external input
        c = self.norm(external_input)

        # Ensure x starts on sphere
        x = normalize_oscillators(x, self.n)

        # Initialize energy
        energies.append(torch.zeros(x.shape[0], device=x.device))

        # Iterate
        for t in range(num_steps):
            dxdt, energy = self.kuramoto_update(x, c)

            # Euler step + renormalize to sphere
            x = normalize_oscillators(x + step_size * dxdt, self.n)

            states.append(x)
            energies.append((-energy).reshape(x.shape[0], -1).sum(-1))

        return states, energies

    def get_order_parameter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Kuramoto order parameter (synchronization measure).

        Args:
            x: Oscillator states

        Returns:
            Order parameter in [0, 1], higher = more synchronized
        """
        x_grouped = reshape_to_groups(x, self.n)

        # Mean oscillator state
        mean_state = x_grouped.mean(dim=1, keepdim=True)

        # Order parameter is magnitude of mean — [B] or scalar
        r = torch.linalg.norm(mean_state, dim=2).squeeze(-1).squeeze(-1)
        # Ensure at least 1-dim for stacking in SphericalKuramotoBank
        if r.ndim == 0:
            r = r.unsqueeze(0)

        return r


class SphericalKuramotoBank(nn.Module):
    """
    Bank of Spherical Kuramoto oscillators for frequency-band processing.

    Multiple independent oscillator groups at different frequencies,
    similar to brain frequency bands (theta, alpha, beta, gamma).
    """

    def __init__(self,
                 num_bands: int = 4,
                 oscillators_per_band: int = 16,
                 components: int = 2,
                 frequency_range: Tuple[float, float] = (4.0, 40.0)):
        """
        Args:
            num_bands: Number of frequency bands
            oscillators_per_band: Oscillators in each band
            components: Components per oscillator (2 for circle)
            frequency_range: (min_freq, max_freq) for initialization
        """
        super().__init__()

        self.num_bands = num_bands
        self.oscillators_per_band = oscillators_per_band

        # Create bands at different frequencies
        self.bands = nn.ModuleList()

        freq_min, freq_max = frequency_range
        frequencies = torch.linspace(freq_min, freq_max, num_bands)

        for i, freq in enumerate(frequencies):
            band = SphericalKuramotoLayer(
                num_channels=oscillators_per_band * components,
                components_per_oscillator=components,
                coupling_type="linear",
                use_omega=True,
                init_omega=freq.item() * 0.1,
                use_projection=True
            )
            self.bands.append(band)

        # Initialize states
        self.register_buffer(
            'states',
            torch.randn(1, num_bands, oscillators_per_band * components)
        )
        self._normalize_states()

    def _normalize_states(self):
        """Ensure states are on sphere."""
        n = self.bands[0].n
        for i in range(self.num_bands):
            self.states[0, i] = normalize_oscillators(
                self.states[0, i:i+1], n
            ).squeeze(0)

    def step(self, external_input: Optional[torch.Tensor] = None,
             num_steps: int = 10, step_size: float = 0.1) -> dict:
        """
        Step all bands forward.

        Args:
            external_input: Optional input [B, num_bands, C] or None
            num_steps: Integration steps
            step_size: Step size

        Returns:
            Dict with states, energies, order_parameters per band
        """
        batch_size = external_input.shape[0] if external_input is not None else 1

        # Expand states for batch
        states = self.states.expand(batch_size, -1, -1).clone()

        results = {
            'states': [],
            'energies': [],
            'order_parameters': []
        }

        for i, band in enumerate(self.bands):
            # Get input for this band
            if external_input is not None:
                inp = external_input[:, i]
            else:
                inp = torch.zeros_like(states[:, i])

            # Run band
            band_states, band_energies = band(
                states[:, i], inp, num_steps, step_size
            )

            # Store final state
            states[:, i] = band_states[-1]

            # Compute order parameter
            r = band.get_order_parameter(band_states[-1])

            results['states'].append(band_states[-1])
            results['energies'].append(band_energies[-1])
            results['order_parameters'].append(r)

        # Update internal state (use first batch item)
        with torch.no_grad():
            self.states.copy_(states[:1].detach())

        # Stack results
        results['states'] = torch.stack(results['states'], dim=1)
        results['energies'] = torch.stack(results['energies'], dim=1)
        results['order_parameters'] = torch.stack(results['order_parameters'], dim=1)
        results['global_order'] = results['order_parameters'].mean()

        return results
