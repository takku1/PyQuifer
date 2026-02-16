"""CUDA acceleration: Triton Kuramoto kernel and tensor diagnostics."""
from pyquifer.accel.cuda.kuramoto_kernel import KuramotoCUDAKernel, TensorDiagnostics

__all__ = [
    "KuramotoCUDAKernel",
    "TensorDiagnostics",
]
