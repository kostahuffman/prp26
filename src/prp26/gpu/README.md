# GPU Acceleration

### 1. CUDA Backend (CuPy)
**File**: `src/prp26/gpu/cuda_kernels.py`

- Custom CUDA kernel for GBM step (geometric Brownian motion)
- Custom CUDA kernel for Heston QE scheme (variance simulation)
- Full GPU memory management (minimize CPU↔GPU transfers)
- Float32 optimization for memory efficiency
- Parallel execution across all paths
- Expected speedup: **15-50x** (GPU-dependent)

**Key Features:**
```python
class CUDAPathGenerator:
    - Custom CUDA kernels compiled via CuPy RawKernel
    - 256 threads per block (configurable)
    - All computation on GPU
    - Automatic CUDA availability detection
```

### 2. Numba Backend
**File**: `src/prp26/gpu/numba_kernels.py`

- JIT-compiled functions with `@jit(nopython=True, parallel=True)`
- Automatic parallelization with `prange`
- Works on CPU (multi-threaded) and CUDA GPUs
- LLVM optimization with `fastmath=True`
- Expected speedup: **3-10x**

**Key Features:**
```python
@jit(nopython=True, parallel=True, fastmath=True)
def gbm_step_numba(S_prev, vol, Z, rate, div_yield, dt, sqrt_dt):
    for i in prange(n_paths):  # Parallel across paths
        S_next[i] = S_prev[i] * np.exp(drift + diffusion)
```

### 3. Automatic Backend Selection
**File**: `src/prp26/gpu/__init__.py`

- Priority: CUDA (fastest) → Numba (good) → None
- `get_gpu_backend()` - detects best available backend
- `is_gpu_available()` - simple availability check
- Graceful fallback with user-friendly messages


## 🎯 Usage

### Simple Usage (Recommended)

```python
from prp26.core.paths import PathGenerator

# Create generator (same as before)
generator = PathGenerator(vol_model, corr_model, rate, div_yields)

# Generate paths - automatically uses GPU if available
paths, variances = generator.generate_paths(
    spots=spots,
    times=times,
    n_paths=100_000,
    backend="gpu"  # Will fallback to CPU if no GPU
)
```

### Check What's Available

```python
from prp26.gpu import get_gpu_backend, is_gpu_available

print(f"GPU Available: {is_gpu_available()}")
print(f"Backend: {get_gpu_backend()}")  # 'cuda', 'numba', or None
```

### Force Specific Backend

```python
# Force CPU (even if GPU available)
paths, _ = generator.generate_paths(..., backend="cpu")

# Use GPU or error if not available
paths, _ = generator.generate_paths(..., backend="gpu")
```

## 📦 Installation

**Optional Dependencies** (choose one or both):

See `pyproject.toml` for dependencies

## 🎓 Key Technical Details

### CUDA Implementation

**Custom Kernels:**
- Written in pure CUDA C
- Compiled via CuPy's `RawKernel`
- Launch config: 256 threads/block
- Memory: Float32 for efficiency

**Optimization:**
- All data stays on GPU
- Pre-generate all random numbers
- Minimize host↔device transfers
- Vectorized operations

### Numba Implementation

**JIT Compilation:**
- `@jit(nopython=True)` - no Python overhead
- `parallel=True` - automatic parallelization
- `fastmath=True` - aggressive optimizations
- `prange` - parallel loops

**Advantages:**
- Works on CPU and GPU
- No manual kernel writing
- Automatic optimization
- Easy to maintain
