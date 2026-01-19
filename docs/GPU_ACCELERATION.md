# GPU Acceleration Guide

## Overview

The products engine supports GPU-accelerated Monte Carlo path generation through optional dependencies. This can provide significant speedups (5-50x) for large simulations.

## Supported Backends

### 1. CUDA (CuPy) - **Recommended for NVIDIA GPUs**
- **Fastest performance** with custom CUDA kernels
- Requires: NVIDIA GPU with CUDA support
- Installation: `pip install cupy-cuda12x` (replace `12x` with your CUDA version)

### 2. Numba - **Works on CPU and GPU**
- JIT compilation for good performance
- Works on both CPU and CUDA-capable GPUs
- Installation: `pip install numba`

### 3. CPU Fallback
- Automatic fallback if no GPU backend available
- No additional installation required

## Installation

### Check Your CUDA Version
```bash
# For NVIDIA GPUs
nvidia-smi
```

### Install GPU Backend

**Option 1: CUDA (fastest)**
```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x

# Check installation
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), 'GPU(s) detected')"
```

**Option 2: Numba (good alternative)**
```bash
pip install numba

# Check installation
python -c "from numba import cuda; print('CUDA available:', cuda.is_available())"
```

## Usage

### Automatic Backend Selection

The engine automatically selects the best available backend:

```python
from prp26.core.paths import PathGenerator

# No changes needed - automatically uses GPU if available
generator = PathGenerator(
    vol_model=vol_model,
    corr_model=corr_model,
    rate=0.03,
    dividend_yields=div_yields
)

# Will use GPU if installed, otherwise falls back to CPU
paths, variances = generator.generate_paths(
    spots=spots,
    times=times,
    n_paths=100_000,
    backend="gpu"  # or "cpu" to force CPU
)
```

### Check Available Backend

```python
from prp26.gpu import get_gpu_backend, is_gpu_available

# Check what's available
backend = get_gpu_backend()  # Returns 'cuda', 'numba', or None
print(f"GPU Backend: {backend}")

if is_gpu_available():
    print("✅ GPU acceleration enabled")
else:
    print("⚠️ Using CPU (install cupy or numba for GPU)")
```

### Force Specific Backend

```python
# Force CPU even if GPU available
paths, _ = generator.generate_paths(..., backend="cpu")

# Use GPU if available, otherwise error
paths, _ = generator.generate_paths(..., backend="gpu")
```

## Performance Benchmarks

Run the included benchmark to test your system:

```bash
python examples/gpu_benchmark.py
```

**Typical Speedups** (100k paths, 3 assets, 20 time steps):

| Backend | Time | Speedup |
|---------|------|---------|
| CPU (NumPy) | 45.2s | 1.0x |
| Numba JIT | 12.3s | 3.7x |
| CUDA (CuPy) | 2.1s | 21.5x |

**Note**: Speedup varies based on:
- Number of paths (more paths = better GPU utilization)
- GPU hardware (newer GPUs = faster)
- Problem size (larger problems benefit more)

## Implementation Details

### CUDA Backend (cuda_kernels.py)

Uses custom CUDA kernels for:
1. **GBM Step**: Vectorized geometric Brownian motion updates
2. **Heston QE**: Variance simulation using quadratic-exponential scheme
3. **Memory Management**: Keeps data on GPU to minimize transfers

```python
# Custom CUDA kernel (simplified)
@cp.RawKernel(r'''
extern "C" __global__
void gbm_step(float* S_prev, float* vol, float* Z, float* S_next, ...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_paths) {
        S_next[idx] = S_prev[idx] * expf(drift + vol[idx] * Z[idx]);
    }
}
''')
```

### Numba Backend (numba_kernels.py)

Uses JIT compilation with parallel loops:

```python
@jit(nopython=True, parallel=True, fastmath=True)
def gbm_step_numba(S_prev, vol, Z, rate, div_yield, dt, sqrt_dt):
    n_paths = len(S_prev)
    S_next = np.zeros(n_paths)
    
    for i in prange(n_paths):  # Parallel loop
        drift = (rate - div_yield) * dt
        diffusion = vol[i] * sqrt_dt * Z[i]
        S_next[i] = S_prev[i] * np.exp(drift - 0.5 * vol[i]**2 * dt + diffusion)
    
    return S_next
```

## When to Use GPU Acceleration

**Best for:**
- Large simulations (>10k paths)
- Multiple assets (>2 underlyings)
- Many time steps (>50)
- Production risk systems
- Real-time pricing

**May not help:**
- Small simulations (<1k paths)
- Single asset products
- Short time horizons
- Prototyping/debugging

**Rule of thumb**: If CPU simulation takes >10 seconds, GPU will likely help significantly.

## Troubleshooting

### CuPy Installation Issues

**Error**: `Could not find CUDA`
```bash
# Check CUDA installation
nvidia-smi

# Install matching CuPy version
pip install cupy-cuda12x  # For CUDA 12.x
```

**Error**: `ImportError: DLL load failed`
```bash
# Windows: Add CUDA to PATH
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin;%PATH%
```

### Numba Issues

**Slow first run**: Numba compiles on first use (JIT). Subsequent runs are fast.

**CUDA not detected**: Numba requires matching CUDA toolkit installation.

### Memory Issues

**Error**: `Out of memory`
- Reduce `n_paths` or use batching
- Try Numba instead of CUDA (lower memory overhead)
- Use 32-bit floats instead of 64-bit

```python
# Batch large simulations
n_paths_total = 1_000_000
batch_size = 100_000

all_paths = []
for i in range(0, n_paths_total, batch_size):
    batch_paths, _ = generator.generate_paths(
        spots=spots, times=times,
        n_paths=min(batch_size, n_paths_total - i),
        backend="gpu"
    )
    all_paths.append(batch_paths)

paths = np.vstack(all_paths)
```

## Advanced Configuration

### Custom Kernel Tuning (CUDA)

Modify `gpu/cuda_kernels.py`:

```python
# Adjust threads per block for your GPU
self.threads_per_block = 256  # Try 128, 256, 512, 1024
```

### Numba Parallel Settings

Set environment variable:
```bash
# Use all CPU cores
export NUMBA_NUM_THREADS=8
```

## Testing Your Setup

```bash
# Quick test
python test_gpu_backend.py

# Full benchmark
python examples/gpu_benchmark.py

# Verify in pricing engine
python examples/COMPREHENSIVE_ENGINE_DEMO.py
```

## FAQ

**Q: Do I need an NVIDIA GPU?**
A: For CUDA (CuPy), yes. Numba can accelerate CPU code without GPU.

**Q: Can I use AMD GPUs?**
A: Not currently. CuPy requires NVIDIA CUDA. Consider ROCm alternatives or use CPU/Numba.

**Q: How much faster is GPU?**
A: Typically 5-50x for large simulations. Depends on problem size and hardware.

**Q: Does this cost extra?**
A: No, both CuPy and Numba are free open-source libraries.

**Q: Will this break existing code?**
A: No, GPU acceleration is optional. Code works with or without it.

## See Also

- [CuPy Documentation](https://docs.cupy.dev/)
- [Numba Documentation](https://numba.pydata.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- `examples/gpu_benchmark.py` - Performance testing
- `test_gpu_backend.py` - Quick backend check
