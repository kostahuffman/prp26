"""GPU acceleration modules for path generation.

This package provides GPU-accelerated Monte Carlo path generation
using optional dependencies:

- CUDA (CuPy): Full GPU acceleration with custom CUDA kernels
- Numba: JIT compilation for CPU/GPU acceleration

Usage:
    # Automatically selects best available backend
    from prp26.gpu import get_gpu_backend
    
    backend = get_gpu_backend()  # Returns 'cuda', 'numba', or None
    
    # Or check availability explicitly
    from prp26.gpu import cuda_kernels, numba_kernels
    
    if cuda_kernels.is_available():
        generator = cuda_kernels.CUDAPathGenerator()
    elif numba_kernels.is_available():
        generator = numba_kernels.NumbaPathGenerator()

Installation:
    # For CUDA (NVIDIA GPU required)
    pip install cupy-cuda12x  # Replace 12x with your CUDA version
    
    # For Numba (works on CPU and CUDA)
    pip install numba
"""

from typing import Optional


def get_gpu_backend() -> Optional[str]:
    """Detect best available GPU backend.
    
    Returns:
        'cuda' if CuPy + CUDA available
        'numba' if Numba available
        None if no GPU backend available
    """
    try:
        from . import cuda_kernels
        if cuda_kernels.is_available():
            return 'cuda'
    except ImportError:
        pass
    
    try:
        from . import numba_kernels
        if numba_kernels.is_available():
            return 'numba'
    except ImportError:
        pass
    
    return None


def is_gpu_available() -> bool:
    """Check if any GPU acceleration is available."""
    return get_gpu_backend() is not None


__all__ = [
    'get_gpu_backend',
    'is_gpu_available',
]

