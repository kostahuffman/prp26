"""Quick test of GPU backend availability and basic functionality."""

import sys

# Test backend detection
from prp26.gpu import get_gpu_backend, is_gpu_available

sys.path.insert(0, "src")

import numpy as np

print("=" * 70)
print("GPU BACKEND TEST")
print("=" * 70)


backend = get_gpu_backend()
print("\n1. Backend Detection:")
print(f"   Available: {is_gpu_available()}")
print(f"   Backend: {backend or 'None (CPU only)'}")

# Test CUDA
print("\n2. CUDA (CuPy) Backend:")
try:
    from prp26.gpu import cuda_kernels

    available = cuda_kernels.is_available()
    print(f"   Available: {available}")
    if available:
        print("   ✅ CuPy installed with CUDA support")
        generator = cuda_kernels.CUDAPathGenerator()
        print("   ✅ CUDA kernels compiled successfully")
except ImportError as e:
    print(f"   ❌ Not available: {e}")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

# Test Numba
print("\n3. Numba Backend:")
try:
    from prp26.gpu import numba_kernels

    available = numba_kernels.is_available()
    cuda_available = numba_kernels.cuda_is_available()
    print(f"   Available: {available}")
    if available:
        print("   ✅ Numba installed")
        print(f"   CUDA support: {cuda_available}")
        generator = numba_kernels.NumbaPathGenerator()
        print("   ✅ Numba JIT ready")
except ImportError as e:
    print(f"   ❌ Not available: {e}")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

# Test path generation with available backend
if is_gpu_available():
    print("\n4. Quick Path Generation Test:")
    print(f"   Using backend: {backend}")

    from prp26.core.paths import PathGenerator
    from prp26.models.correlation.skew import CorrelationSkewModel
    from prp26.models.volatility.slv import HestonModel, SLVModel

    # Simple setup
    spots = np.array([100.0, 105.0])
    times = np.linspace(0, 1.0, 5)
    n_paths = 1000

    heston = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    vol_model = SLVModel(heston_model=heston)

    base_corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    corr_model = CorrelationSkewModel(base_corr=base_corr, skew_function=lambda m: 0.0)

    generator = PathGenerator(
        vol_model=vol_model,
        corr_model=corr_model,
        rate=0.03,
        dividend_yields=np.array([0.02, 0.02]),
        seed=42,
    )

    try:
        paths, variances = generator.generate_paths(
            spots=spots, times=times, n_paths=n_paths, backend="gpu"
        )

        print(f"   ✅ Generated {n_paths} paths")
        print(f"   Shape: {paths.shape}")
        print(f"   Mean final spot: {np.mean(paths[:, -1, :], axis=0)}")
        print("   Path generation successful!")

    except Exception as e:
        print(f"   ❌ Error generating paths: {e}")
else:
    print("\n4. No GPU Backend Available")
    print("   Install optional dependencies:")
    print("   - pip install cupy-cuda12x  (for CUDA)")
    print("   - pip install numba         (for Numba JIT)")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if backend == "cuda":
    print("✅ Optimal: CUDA backend active (fastest)")
    print("   GPU-accelerated path generation with custom kernels")
elif backend == "numba":
    print("✅ Good: Numba JIT backend active")
    print("   Accelerated path generation with JIT compilation")
    print("   Consider installing CuPy for even better performance")
else:
    print("⚠️  CPU only: No GPU acceleration")
    print("   Install cupy-cuda12x or numba for acceleration")

print()
