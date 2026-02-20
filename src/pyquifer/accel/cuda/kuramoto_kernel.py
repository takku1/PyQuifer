"""
CUDA-accelerated Kuramoto coupling kernels with PyTorch fallbacks.

Provides:
- Fused all-to-all coupling computation
- Fused order parameter (no separate reduction pass)
- TensorDiagnostics for eliminating .item() CPU-GPU syncs
- Optional Triton JIT kernels when triton is available

References:
- Kuramoto (1984). Chemical Oscillations, Waves, and Turbulence.
- multilayer-cudamoto: GPU Kuramoto implementations
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ── Triton kernel (optional) ──────────────────────────────────────────────
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True

    @triton.jit
    def _kuramoto_coupling_kernel(
        phases_ptr, coupling_ptr, output_ptr,
        K: tl.constexpr, N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused sin(phi_j - phi_i) * A_ij coupling kernel."""
        pid = tl.program_id(0)
        i = pid

        if i >= N:
            return

        phi_i = tl.load(phases_ptr + i)
        acc = tl.zeros([], dtype=tl.float32)

        for j_start in range(0, N, BLOCK_SIZE):
            j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
            mask = j_offsets < N
            phi_j = tl.load(phases_ptr + j_offsets, mask=mask, other=0.0)
            a_ij = tl.load(coupling_ptr + i * N + j_offsets, mask=mask, other=0.0)
            diff = phi_j - phi_i
            coupling_val = a_ij * tl.sin(diff)
            acc += tl.sum(coupling_val)

        result = K * acc / N
        tl.store(output_ptr + i, result)

    @triton.jit
    def _order_parameter_kernel(
        phases_ptr, R_ptr, Psi_ptr,
        N: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    ):
        """Fused order parameter: R*exp(i*Psi) = (1/N) sum exp(i*phi)."""
        cos_sum = tl.zeros([], dtype=tl.float32)
        sin_sum = tl.zeros([], dtype=tl.float32)

        for start in range(0, N, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N
            phi = tl.load(phases_ptr + offsets, mask=mask, other=0.0)
            cos_sum += tl.sum(tl.cos(phi))
            sin_sum += tl.sum(tl.sin(phi))

        cos_mean = cos_sum / N
        sin_mean = sin_sum / N
        R = tl.sqrt(cos_mean * cos_mean + sin_mean * sin_mean)
        Psi = tl.libdevice.atan2(sin_mean, cos_mean)
        tl.store(R_ptr, R)
        tl.store(Psi_ptr, Psi)

except ImportError:
    pass


class KuramotoCUDAKernel:
    """CUDA-accelerated Kuramoto operations.

    Provides fused kernels for coupling computation and order parameter
    calculation, eliminating multiple kernel launches per tick.

    Falls back to optimized vectorized PyTorch when Triton is unavailable.
    """

    def __init__(self, use_triton: bool = True):
        self._use_triton = use_triton and _TRITON_AVAILABLE
        if self._use_triton:
            logger.info("KuramotoCUDAKernel: using Triton JIT kernels")
        else:
            logger.info("KuramotoCUDAKernel: using vectorized PyTorch fallback")

    @property
    def backend(self) -> str:
        return "triton" if self._use_triton else "pytorch"

    def all_to_all_coupling(
        self,
        phases: torch.Tensor,
        coupling_matrix: torch.Tensor,
        coupling_strength: float,
    ) -> torch.Tensor:
        """Compute all-to-all Kuramoto coupling forces.

        Args:
            phases: (N,) oscillator phases
            coupling_matrix: (N, N) adjacency/coupling weights
            coupling_strength: Global coupling K

        Returns:
            (N,) coupling forces for each oscillator
        """
        N = phases.shape[0]

        if self._use_triton and phases.is_cuda:
            output = torch.empty_like(phases)
            BLOCK_SIZE = min(128, N)
            _kuramoto_coupling_kernel[(N,)](
                phases, coupling_matrix, output,
                K=coupling_strength, N=N,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return output

        # Vectorized PyTorch fallback
        # (N,1) - (1,N) → (N,N) phase differences
        diff = phases.unsqueeze(0) - phases.unsqueeze(1)  # (N,N): diff[i,j] = phi_j - phi_i
        coupling_forces = (coupling_matrix * torch.sin(diff)).sum(dim=1)
        return coupling_strength * coupling_forces / N

    def fused_order_parameter(
        self, phases: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute order parameter R and mean phase Psi as tensors.

        No .item() calls — returns GPU tensors for zero-sync diagnostics.

        Args:
            phases: (N,) oscillator phases

        Returns:
            (R, Psi) both as scalar tensors on same device as phases
        """
        N = phases.shape[0]

        if self._use_triton and phases.is_cuda:
            R = torch.empty(1, device=phases.device, dtype=phases.dtype)
            Psi = torch.empty(1, device=phases.device, dtype=phases.dtype)
            BLOCK_SIZE = min(128, N)
            _order_parameter_kernel[(1,)](
                phases, R, Psi,
                N=N, BLOCK_SIZE=BLOCK_SIZE,
            )
            return R.squeeze(0), Psi.squeeze(0)

        # Vectorized PyTorch fallback
        cos_mean = torch.cos(phases).mean()
        sin_mean = torch.sin(phases).mean()
        R = torch.sqrt(cos_mean ** 2 + sin_mean ** 2)
        Psi = torch.atan2(sin_mean, cos_mean)
        return R, Psi

    def fused_step(
        self,
        phases: torch.Tensor,
        omega: torch.Tensor,
        coupling_matrix: torch.Tensor,
        K: float,
        dt: float,
        external_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fused Kuramoto step: compute coupling + Euler integrate.

        Args:
            phases: (N,) current phases
            omega: (N,) natural frequencies
            coupling_matrix: (N, N) coupling weights
            K: Global coupling strength
            dt: Integration timestep
            external_input: Optional (N,) external forcing

        Returns:
            (N,) updated phases
        """
        coupling = self.all_to_all_coupling(phases, coupling_matrix, K)
        d_phase = omega + coupling
        if external_input is not None:
            d_phase = d_phase + external_input
        new_phases = (phases + dt * d_phase) % (2 * math.pi)
        return new_phases


class TensorDiagnostics:
    """Convert between tensor and scalar diagnostic dicts.

    Eliminates ~30 CPU-GPU syncs per tick by keeping all values as
    tensors on GPU. Only converts to scalars when explicitly requested
    (e.g. for logging or JSON serialization).
    """

    @staticmethod
    def to_tensor_dict(
        diagnostics: Dict[str, Any],
        device: torch.device = None,
    ) -> Dict[str, Any]:
        """Convert mixed float/tensor dict to all-tensor dict.

        Args:
            diagnostics: Dict with mixed float/int/tensor/str values
            device: Target device for new tensors

        Returns:
            Dict where numeric values are tensors, others unchanged
        """
        result = {}
        for k, v in diagnostics.items():
            if isinstance(v, (int, float)):
                t = torch.tensor(v, dtype=torch.float32)
                if device is not None:
                    t = t.to(device)
                result[k] = t
            elif isinstance(v, torch.Tensor):
                result[k] = v
            elif isinstance(v, dict):
                result[k] = TensorDiagnostics.to_tensor_dict(v, device)
            else:
                # str, bool, list, None — keep as-is
                result[k] = v
        return result

    @staticmethod
    def to_scalar_dict(
        tensor_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Lazily convert tensor dict to scalar dict.

        Calls .item() on 0-dim tensors, .tolist() on 1-dim+ tensors.

        Args:
            tensor_dict: Dict with tensor values

        Returns:
            Dict with Python float/int/list values
        """
        result = {}
        for k, v in tensor_dict.items():
            if isinstance(v, torch.Tensor):
                if v.dim() == 0:
                    result[k] = v.item()
                elif v.numel() <= 64:
                    result[k] = v.detach().cpu().tolist()
                else:
                    # Large tensors: just report shape
                    result[k] = f"tensor({list(v.shape)})"
            elif isinstance(v, dict):
                result[k] = TensorDiagnostics.to_scalar_dict(v)
            else:
                result[k] = v
        return result

    @staticmethod
    def extract_scalars(
        tensor_dict: Dict[str, Any],
        keys: list,
    ) -> Dict[str, float]:
        """Extract specific keys as scalars (single batched .item() group).

        Args:
            tensor_dict: Dict with tensor values
            keys: List of keys to extract

        Returns:
            Dict mapping key → float
        """
        result = {}
        for k in keys:
            v = tensor_dict.get(k)
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                result[k] = v.item()
            elif isinstance(v, (int, float)):
                result[k] = float(v)
        return result


def try_load_cuda_kernels() -> Optional[KuramotoCUDAKernel]:
    """Try to load CUDA Kuramoto kernels.

    Returns:
        KuramotoCUDAKernel instance if CUDA available, None otherwise.
    """
    if not torch.cuda.is_available():
        logger.debug("CUDA not available — skipping Kuramoto CUDA kernels")
        return None

    kernel = KuramotoCUDAKernel(use_triton=_TRITON_AVAILABLE)
    logger.info(f"CUDA Kuramoto kernels loaded (backend={kernel.backend})")
    return kernel


def is_available() -> bool:
    """Check if CUDA kernels can be used."""
    return torch.cuda.is_available()
