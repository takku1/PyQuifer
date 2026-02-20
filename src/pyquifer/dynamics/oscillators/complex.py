"""Complex-valued oscillator backends for PyQuifer.

Represents oscillators as z = A * exp(i*phi) instead of real-valued phase
angles.  Working in the complex plane avoids repeated sin/cos decomposition
and enables direct complex-linear coupling (z_j * conj(z_k)) which is
algebraically equivalent to the real Kuramoto sin(phi_j - phi_k) coupling
but admits holomorphic gradient flow.

References:
    - Kuramoto Y (1984) Chemical Oscillations, Waves, and Turbulence
    - Trabelsi C et al. (2018) Deep Complex Networks, ICLR
    - Virtue P et al. (2017) Better than Real: Complex-valued Neural Nets
      for MRI Fingerprinting
    - torchcvnn library (https://github.com/jeremyfix/torchcvnn)
    - Ott E & Antonsen T (2008) Low dimensional behavior of large systems
      of globally coupled oscillators, Chaos 18(3)

Phase 11.2 of the PyQuifer enhancement plan.
"""

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Utility conversions
# ---------------------------------------------------------------------------

def to_complex(
    phases: torch.Tensor,
    amplitudes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert real-valued phases (and optional amplitudes) to complex z.

    z = A * exp(i * phi)

    Args:
        phases: Phase angles in radians, arbitrary shape.
        amplitudes: Optional magnitudes (same shape as *phases*).
            Defaults to unit amplitude.

    Returns:
        Complex tensor of the same shape as *phases*.
    """
    if amplitudes is None:
        return torch.polar(torch.ones_like(phases), phases)
    return torch.polar(amplitudes, phases)


def from_complex(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert complex z to (phases, amplitudes).

    Args:
        z: Complex tensor of arbitrary shape.

    Returns:
        Tuple of (phases, amplitudes), each real-valued with the same shape.
    """
    return torch.angle(z), torch.abs(z)


def complex_order_parameter(z: torch.Tensor) -> torch.Tensor:
    """Global order parameter from complex oscillator states.

    R * exp(i*Psi) = (1/N) * sum(z_k / |z_k|)

    Unlike the real-valued decomposition (mean of cos + i*mean of sin),
    this normalises each oscillator to the unit circle first so that
    amplitude differences do not bias the synchronisation measure.

    Args:
        z: Complex tensor of shape (N,) or (..., N).

    Returns:
        Scalar (or batch) complex order parameter.
    """
    # Normalise to unit circle to measure phase coherence only
    z_unit = z / (torch.abs(z).clamp(min=1e-12))
    return z_unit.mean(dim=-1)


# ---------------------------------------------------------------------------
# ComplexCoupling
# ---------------------------------------------------------------------------

class ComplexCoupling(nn.Module):
    """Coupling forces via complex arithmetic: z_j * conj(z_k).

    For unit-magnitude oscillators z_k = exp(i*phi_k):
        Im(z_j * conj(z_k)) = sin(phi_j - phi_k)

    which recovers the classical Kuramoto coupling.  In the complex
    formulation, however, amplitude information is naturally included:
    strong-amplitude oscillators exert stronger coupling.

    Supports the same topologies as :class:`LearnableKuramotoBank`:
    ``global``, ``ring``, ``small_world``, ``scale_free``, ``learnable``.

    Args:
        num_oscillators: Number of oscillators N.
        topology: Coupling topology string.
        topology_params: Topology-specific parameters dict (same keys as
            :class:`LearnableKuramotoBank`).
    """

    def __init__(
        self,
        num_oscillators: int,
        topology: Literal[
            "global", "small_world", "scale_free", "ring", "learnable"
        ] = "global",
        topology_params: Optional[dict] = None,
    ):
        super().__init__()
        self.num_oscillators = num_oscillators
        self.topology = topology
        self._topology_params = topology_params or {}
        self._global_topology = False

        self._init_adjacency(topology, self._topology_params)

    # ── Adjacency construction (mirrors LearnableKuramotoBank) ──

    def _init_adjacency(self, topology: str, params: dict) -> None:
        n = self.num_oscillators
        if topology == "global":
            self.register_buffer("adjacency", None)
            self._global_topology = True
        elif topology == "ring":
            k = params.get("k", 2)
            adj = self._create_ring(n, k)
            self.register_buffer("adjacency", adj)
            self._build_edge_list(adj)
        elif topology == "small_world":
            k = params.get("k", max(2, n // 10))
            p = params.get("p", 0.1)
            adj = self._create_small_world(n, k, p)
            self.register_buffer("adjacency", adj)
            self._build_edge_list(adj)
        elif topology == "scale_free":
            m = params.get("m", max(1, n // 20))
            adj = self._create_scale_free(n, m)
            self.register_buffer("adjacency", adj)
            self._build_edge_list(adj)
        elif topology == "learnable":
            sparsity = params.get("sparsity", 0.5)
            init_values = (
                torch.randn(n, n) * 0.5
                - torch.log(torch.tensor(1.0 / sparsity - 1.0))
            )
            self.adjacency_logits = nn.Parameter(init_values)
            self.register_buffer("self_mask", 1.0 - torch.eye(n))
        else:
            raise ValueError(f"Unknown topology: {topology}")

    def _build_edge_list(self, adj: torch.Tensor) -> None:
        src, dst = adj.nonzero(as_tuple=True)
        self.register_buffer("_edge_src", src.long())
        self.register_buffer("_edge_dst", dst.long())
        self.register_buffer("_edge_weight", adj[src, dst].float())
        self.register_buffer(
            "_degree", adj.abs().sum(dim=1).clamp(min=1.0)
        )

    @staticmethod
    def _create_ring(n: int, k: int) -> torch.Tensor:
        adj = torch.zeros(n, n)
        for i in range(n):
            for j in range(1, k + 1):
                adj[i, (i + j) % n] = 1
                adj[i, (i - j) % n] = 1
        return adj

    @staticmethod
    def _create_small_world(n: int, k: int, p: float) -> torch.Tensor:
        adj = torch.zeros(n, n)
        for i in range(n):
            for j in range(1, k + 1):
                adj[i, (i + j) % n] = 1
                adj[i, (i - j) % n] = 1
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1 and torch.rand(1).item() < p:
                    candidates = [
                        x for x in range(n) if x != i and adj[i, x] == 0
                    ]
                    if candidates:
                        new_j = candidates[
                            torch.randint(len(candidates), (1,)).item()
                        ]
                        adj[i, j] = 0
                        adj[j, i] = 0
                        adj[i, new_j] = 1
                        adj[new_j, i] = 1
        return adj

    @staticmethod
    def _create_scale_free(n: int, m: int) -> torch.Tensor:
        adj = torch.zeros(n, n)
        m0 = max(m, 2)
        for i in range(m0):
            for j in range(i + 1, m0):
                adj[i, j] = 1
                adj[j, i] = 1
        for i in range(m0, n):
            degrees = adj[:i, :i].sum(dim=1) + 1
            probs = degrees / degrees.sum()
            selected = torch.multinomial(probs, min(m, i), replacement=False)
            for j in selected:
                adj[i, j] = 1
                adj[j, i] = 1
        return adj

    def get_adjacency(self) -> Optional[torch.Tensor]:
        """Current adjacency matrix (handles learnable case)."""
        if self.topology == "learnable":
            return torch.sigmoid(self.adjacency_logits) * self.self_mask
        if self._global_topology:
            return None
        return self.adjacency

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute complex coupling forces for each oscillator.

        For oscillator *i*, the coupling force is:

            F_i = (1 / degree_i) * sum_j A_{ij} * (z_j - z_i)

        In the global case this simplifies to ``z_mean - z_i`` (Kuramoto
        mean-field form).

        Args:
            z: Complex tensor of shape ``(N,)`` with oscillator states.

        Returns:
            Complex tensor of shape ``(N,)`` with coupling forces.
        """
        adj = self.get_adjacency()

        if adj is None:
            # Global: mean-field coupling
            z_mean = z.mean()
            return z_mean.expand(self.num_oscillators) - z

        if hasattr(self, "_edge_src"):
            # Sparse COO path
            src = self._edge_src
            dst = self._edge_dst
            w = self._edge_weight.to(z.dtype)
            diffs = z[dst] - z[src]  # z_j - z_i for each edge (src=i, dst=j)
            forces = torch.zeros(
                self.num_oscillators, dtype=z.dtype, device=z.device
            )
            forces.scatter_add_(0, src, w * diffs)
            return forces / self._degree.to(z.dtype)

        # Dense (learnable topology)
        # z_diffs[i, j] = z_j - z_i
        z_diffs = z.unsqueeze(0) - z.unsqueeze(1)
        weighted = adj.to(z.dtype) * z_diffs
        force_sum = weighted.sum(dim=1)
        degree = adj.abs().sum(dim=1).clamp(min=1.0).to(z.dtype)
        return force_sum / degree


# ---------------------------------------------------------------------------
# ComplexKuramotoBank
# ---------------------------------------------------------------------------

class ComplexKuramotoBank(nn.Module):
    """Native complex Kuramoto oscillator bank: z = A * exp(i*phi).

    Instead of tracking real-valued phases and computing sin/cos at each
    step, all oscillator state lives in the complex plane.  The dynamics
    are:

        dz_i/dt = i*omega_i * z_i
                  + (K / N) * sum_j A_{ij} * (z_j - z_i)
                  + external

    where *i* is the imaginary unit (not an index).  For unit-magnitude
    oscillators on the unit circle, taking ``Im(z_j * conj(z_i))`` yields
    the classical Kuramoto ``sin(phi_j - phi_i)`` coupling exactly.

    Args:
        num_oscillators: Number of oscillators N.
        dt: Integration timestep.
        initial_frequency_range: ``(lo, hi)`` for uniform random natural
            frequencies (radians / time unit).
        initial_phase_range: ``(lo, hi)`` for uniform random initial phases.
        initial_amplitude: Initial oscillator amplitude (default 1.0 for
            unit circle, matching real-valued Kuramoto).
        topology: Coupling topology (see :class:`ComplexCoupling`).
        topology_params: Topology-specific parameter dict.
        integration_method: ``'euler'`` (default) or ``'rk4'``.
    """

    # Phase-accumulator buffers that must remain fp32 (see oscillators.py).
    _FP32_BUFFERS = frozenset({"z_real", "z_imag"})

    def __init__(
        self,
        num_oscillators: int,
        dt: float = 0.01,
        initial_frequency_range: Tuple[float, float] = (0.5, 1.5),
        initial_phase_range: Tuple[float, float] = (0.0, 2 * math.pi),
        initial_amplitude: float = 1.0,
        topology: Literal[
            "global", "small_world", "scale_free", "ring", "learnable"
        ] = "global",
        topology_params: Optional[dict] = None,
        integration_method: Literal["euler", "rk4"] = "euler",
    ):
        super().__init__()
        self.num_oscillators = num_oscillators
        self.dt = dt
        self.integration_method = integration_method

        # Learnable natural frequencies
        self.natural_frequencies = nn.Parameter(
            torch.rand(num_oscillators)
            * (initial_frequency_range[1] - initial_frequency_range[0])
            + initial_frequency_range[0]
        )

        # Learnable global coupling strength
        self.coupling_strength = nn.Parameter(torch.tensor(1.0))

        # Complex state stored as real/imag buffers (complex buffers cannot
        # be registered in older PyTorch versions).
        phases_init = (
            torch.rand(num_oscillators)
            * (initial_phase_range[1] - initial_phase_range[0])
            + initial_phase_range[0]
        )
        amp_init = torch.full((num_oscillators,), initial_amplitude)
        self.register_buffer("z_real", amp_init * torch.cos(phases_init))
        self.register_buffer("z_imag", amp_init * torch.sin(phases_init))

        # Coupling sub-module
        self.coupling = ComplexCoupling(
            num_oscillators,
            topology=topology,
            topology_params=topology_params,
        )

        # Cached order parameter (computed at end of forward())
        self._cached_R: Optional[torch.Tensor] = None
        self._cached_Psi: Optional[torch.Tensor] = None

    # ── Preserve fp32 for accumulators ──

    def _apply(self, fn):
        """Preserve fp32 for complex state buffers."""
        super()._apply(fn)
        for name in self._FP32_BUFFERS:
            buf = self._buffers.get(name)
            if buf is not None and buf.dtype != torch.float32:
                self._buffers[name] = buf.float()
        return self

    # ── Properties ──

    @property
    def z(self) -> torch.Tensor:
        """Complex oscillator states ``z = z_real + i * z_imag``."""
        return torch.complex(self.z_real, self.z_imag)

    def get_phases(self) -> torch.Tensor:
        """Current oscillator phases (radians), for interop with real-valued code."""
        return torch.atan2(self.z_imag, self.z_real) % (2 * math.pi)

    def get_amplitudes(self) -> torch.Tensor:
        """Current oscillator amplitudes, for interop with real-valued code."""
        return torch.sqrt(self.z_real ** 2 + self.z_imag ** 2)

    def get_order_parameter(self) -> torch.Tensor:
        """Global synchronisation R via direct complex mean.

        Uses the cached value from the last :meth:`forward` call when
        available.
        """
        if self._cached_R is not None:
            return self._cached_R
        op = complex_order_parameter(self.z)
        return torch.abs(op)

    # ── Dynamics ──

    def _rhs(
        self,
        z: torch.Tensor,
        external_input: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Right-hand side of the complex Kuramoto ODE.

        dz_i/dt = i*omega_i * z_i + K * F_i + external
        """
        # Free rotation: i * omega * z
        # i * z = (-z_imag, z_real) but we build it via multiplication
        i_omega = torch.complex(
            torch.zeros_like(self.natural_frequencies),
            self.natural_frequencies,
        )
        dz = i_omega * z

        # Coupling
        coupling_force = self.coupling(z)
        dz = dz + self.coupling_strength * coupling_force

        # External drive
        if external_input is not None:
            dz = dz + external_input

        return dz

    def forward(
        self,
        external_input: Optional[torch.Tensor] = None,
        steps: int = 1,
    ) -> torch.Tensor:
        """Evolve the complex Kuramoto dynamics for *steps* timesteps.

        Args:
            external_input: Optional complex driving force of shape
                ``(N,)`` or ``(N,)`` real (will be cast to complex).
                For compatibility with :class:`LearnableKuramotoBank`,
                a real tensor is interpreted as a phase-velocity
                perturbation ``i * input * z``.
            steps: Number of integration sub-steps.

        Returns:
            Complex tensor of shape ``(N,)`` with updated oscillator
            states.
        """
        z = self.z
        self._cached_R = None
        self._cached_Psi = None

        # Convert real external input to complex phase perturbation
        if external_input is not None and not external_input.is_complex():
            # Real input -> frequency perturbation: dz += i * ext * z
            external_input = torch.complex(
                torch.zeros_like(external_input),
                external_input,
            ) * z

        for _ in range(steps):
            if self.integration_method == "rk4":
                _ext = external_input

                def _rhs_fn(zz: torch.Tensor) -> torch.Tensor:
                    return self._rhs(zz, _ext)

                k1 = _rhs_fn(z)
                k2 = _rhs_fn(z + 0.5 * self.dt * k1)
                k3 = _rhs_fn(z + 0.5 * self.dt * k2)
                k4 = _rhs_fn(z + self.dt * k3)
                z = z + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                dz = self._rhs(z, external_input)
                z = z + dz * self.dt

        # ── Cache order parameter (avoids redundant computation) ──
        op = complex_order_parameter(z)
        self._cached_R = torch.abs(op)
        self._cached_Psi = torch.angle(op)

        # ── Store state ──
        with torch.no_grad():
            self.z_real.copy_(z.real.detach())
            self.z_imag.copy_(z.imag.detach())

        return z

    def reset(self, initial_amplitude: float = 1.0) -> None:
        """Reset oscillators to random phases with given amplitude."""
        phases = torch.rand(self.num_oscillators) * 2 * math.pi
        with torch.no_grad():
            self.z_real.copy_(initial_amplitude * torch.cos(phases))
            self.z_imag.copy_(initial_amplitude * torch.sin(phases))
        self._cached_R = None
        self._cached_Psi = None


# ---------------------------------------------------------------------------
# Complex-valued neural network layers
# ---------------------------------------------------------------------------

class ModReLU(nn.Module):
    """Magnitude-preserving ReLU for complex-valued networks.

    ModReLU(z) = (z / |z|) * ReLU(|z| + b)

    where *b* is a learnable bias.  This zeroes out complex values whose
    magnitude is below ``-b`` while preserving phase information.

    Reference:
        Arjovsky M, Shah A, Bengio Y (2016) Unitary Evolution Recurrent
        Neural Networks, ICML.

    Args:
        features: Number of channels / features.
        bias_init: Initial value for the learnable bias *b*.
    """

    def __init__(self, features: int, bias_init: float = -0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.full((features,), bias_init))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply ModReLU activation.

        Args:
            z: Complex tensor of shape ``(..., features)``.

        Returns:
            Complex tensor with the same shape.
        """
        mag = torch.abs(z).clamp(min=1e-12)
        phase = z / mag
        activated_mag = F.relu(mag + self.bias)
        return phase * activated_mag


class ComplexLinear(nn.Module):
    """Linear layer for complex-valued tensors.

    Implements the complex matrix multiplication:

        y = (W_r + i*W_i) @ x + (b_r + i*b_i)

    which expands to:

        y_r = W_r @ x_r - W_i @ x_i + b_r
        y_i = W_r @ x_i + W_i @ x_r + b_i

    Using separate real/imaginary weight matrices is more numerically
    stable (and torch.compile-friendly) than native complex Parameters.

    Reference:
        Trabelsi C et al. (2018) Deep Complex Networks, ICLR.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If ``True``, add a learnable complex bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Separate real and imaginary weight matrices
        self.weight_real = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.weight_imag = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Glorot / fan-avg initialisation adapted for complex.

        The effective fan is doubled compared to real-valued layers
        because both real and imaginary parts contribute to the
        variance (Trabelsi et al. 2018, Section 3.6).
        """
        fan_in = self.in_features
        fan_out = self.out_features
        # Variance = 2 / (fan_in + fan_out), split across Re and Im
        std = math.sqrt(1.0 / (fan_in + fan_out))
        nn.init.normal_(self.weight_real, 0.0, std)
        nn.init.normal_(self.weight_imag, 0.0, std)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply complex linear transformation.

        Args:
            z: Complex tensor of shape ``(..., in_features)``.

        Returns:
            Complex tensor of shape ``(..., out_features)``.
        """
        x_r = z.real
        x_i = z.imag

        # (W_r + iW_i)(x_r + ix_i) = (W_r*x_r - W_i*x_i) + i(W_r*x_i + W_i*x_r)
        out_r = F.linear(x_r, self.weight_real) - F.linear(x_i, self.weight_imag)
        out_i = F.linear(x_r, self.weight_imag) + F.linear(x_i, self.weight_real)

        if self.bias_real is not None:
            out_r = out_r + self.bias_real
            out_i = out_i + self.bias_imag

        return torch.complex(out_r, out_i)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias_real is not None}"
        )


class ComplexBatchNorm(nn.Module):
    """Batch normalisation for complex-valued tensors.

    Normalises the magnitude to unit mean while preserving phase:

        z_hat = (z / E[|z|]) * gamma + beta

    where ``gamma`` and ``beta`` are learnable real-valued scale and shift
    applied to the magnitude.  This is the "Type A" complex batch norm
    from Trabelsi et al. (2018), which is simpler and more stable than
    the full 2x2 covariance whitening variant.

    During training, running statistics of the mean magnitude are tracked
    with exponential moving average (momentum 0.1).

    Args:
        num_features: Number of features / channels.
        eps: Small constant for numerical stability.
        momentum: EMA momentum for running statistics.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable affine on magnitude
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Running mean of magnitudes
        self.register_buffer("running_mean_mag", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply complex batch normalisation.

        Args:
            z: Complex tensor of shape ``(B, num_features)`` or
                ``(B, num_features, *)``.

        Returns:
            Normalised complex tensor with the same shape.
        """
        mag = torch.abs(z).clamp(min=self.eps)

        if self.training:
            # Compute batch mean of magnitudes (over batch and spatial dims)
            reduce_dims = [0] + list(range(2, mag.dim()))
            mean_mag = mag.mean(dim=reduce_dims)

            with torch.no_grad():
                self.running_mean_mag.mul_(1 - self.momentum).add_(
                    mean_mag * self.momentum
                )
                self.num_batches_tracked.add_(1)
        else:
            mean_mag = self.running_mean_mag

        # Reshape for broadcasting: (1, C, 1, 1, ...)
        shape = [1, self.num_features] + [1] * (z.dim() - 2)
        mean_mag = mean_mag.view(shape)
        gamma = self.gamma.view(shape)
        beta = self.beta.view(shape)

        # Normalise magnitude, preserve phase
        phase = z / mag.clamp(min=self.eps)
        normalised_mag = mag / mean_mag.clamp(min=self.eps)
        scaled_mag = normalised_mag * gamma + beta

        # Ensure magnitude stays non-negative
        scaled_mag = F.relu(scaled_mag)

        return phase * scaled_mag

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}"


# ---------------------------------------------------------------------------
# Module-level demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Complex Kuramoto Bank ---")
    bank = ComplexKuramotoBank(
        num_oscillators=32,
        dt=0.01,
        topology="global",
        initial_frequency_range=(0.8, 1.2),
    )

    # Initial state
    R0 = bank.get_order_parameter()
    print(f"  Initial R: {R0.item():.4f}")

    # Simulate 500 steps
    z = bank(steps=500)
    R_final = bank.get_order_parameter()
    print(f"  Final R (500 steps): {R_final.item():.4f}")

    # Interop with real-valued code
    phases = bank.get_phases()
    amplitudes = bank.get_amplitudes()
    print(f"  Phases shape: {phases.shape}, Amplitudes shape: {amplitudes.shape}")
    print(f"  Mean amplitude: {amplitudes.mean().item():.4f}")

    # Round-trip conversion
    z_roundtrip = to_complex(phases, amplitudes)
    diff = (z - z_roundtrip).abs().max()
    print(f"  Round-trip error: {diff.item():.2e}")

    print("\n--- Complex Neural Net Layers ---")
    linear = ComplexLinear(32, 16)
    bn = ComplexBatchNorm(16)
    act = ModReLU(16)

    x = torch.randn(8, 32, dtype=torch.complex64)
    y = act(bn(linear(x)))
    print(f"  Input:  {x.shape} (complex64)")
    print(f"  Output: {y.shape} (complex64)")
    print(f"  Output magnitude range: [{y.abs().min().item():.3f}, {y.abs().max().item():.3f}]")

    print("\n--- Complex Coupling (ring topology) ---")
    coupling = ComplexCoupling(32, topology="ring", topology_params={"k": 3})
    z_test = to_complex(torch.rand(32) * 2 * math.pi)
    forces = coupling(z_test)
    print(f"  Forces shape: {forces.shape}")
    print(f"  Mean force magnitude: {forces.abs().mean().item():.4f}")

    print("\n--- Order Parameter Comparison ---")
    phases_test = torch.rand(64) * 2 * math.pi
    z_from_phases = to_complex(phases_test)
    R_complex = torch.abs(complex_order_parameter(z_from_phases))
    # Compare with real-valued computation
    R_real = torch.abs(torch.exp(1j * phases_test).mean())
    print(f"  R (complex): {R_complex.item():.6f}")
    print(f"  R (real):    {R_real.item():.6f}")
    print(f"  Difference:  {abs(R_complex.item() - R_real.item()):.2e}")
