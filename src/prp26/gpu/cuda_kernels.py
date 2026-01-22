"""CUDA-accelerated path generation using CuPy.

Requires: pip install cupy-cuda12x (or appropriate CUDA version)

This module provides GPU-accelerated Monte Carlo path generation
using CuPy arrays and custom CUDA kernels for maximum performance.
"""

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


# Custom CUDA kernel for GBM step (vectorized across paths)
GBM_KERNEL = r"""
extern "C" __global__
void gbm_step(
    const float* S_prev,      // Previous spot prices (n_paths,)
    const float* vol,         // Volatilities (n_paths,)
    const float* Z,           // Random normals (n_paths,)
    float* S_next,            // Output spot prices (n_paths,)
    const float rate,         // Risk-free rate
    const float div_yield,    // Dividend yield
    const float dt,           // Time step
    const float sqrt_dt,      // sqrt(dt)
    const int n_paths
) {
    int path_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (path_idx < n_paths) {
        float S = S_prev[path_idx];
        float sigma = vol[path_idx];
        float z = Z[path_idx];

        float drift = (rate - div_yield) * dt;
        float diffusion = sigma * sqrt_dt * z;

        S_next[path_idx] = S * expf(drift - 0.5f * sigma * sigma * dt + diffusion);
    }
}
"""

# Custom CUDA kernel for Heston variance simulation (QE scheme)
HESTON_QE_KERNEL = r"""
extern "C" __global__
void heston_qe_step(
    const float* V_prev,      // Previous variance (n_paths,)
    const float* U,           // Uniform randoms (n_paths,)
    const float* Z,           // Normal randoms (n_paths,)
    float* V_next,            // Output variance (n_paths,)
    const float kappa,        // Mean reversion speed
    const float theta,        // Long-term variance
    const float sigma_v,      // Vol of vol
    const float dt,           // Time step
    const float psi_crit,     // Critical psi value (1.5)
    const int n_paths
) {
    int path_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (path_idx < n_paths) {
        float V = V_prev[path_idx];
        float u = U[path_idx];
        float z = Z[path_idx];

        // QE scheme parameters
        float exp_kappa_dt = expf(-kappa * dt);
        float m = theta + (V - theta) * exp_kappa_dt;

        float term1 = V * sigma_v * sigma_v * exp_kappa_dt / kappa * (1.0f - exp_kappa_dt);
        float term2 = theta * sigma_v * sigma_v / (2.0f * kappa) * powf(1.0f - exp_kappa_dt, 2.0f);
        float s2 = term1 + term2;
        float psi = s2 / (m * m + 1e-10f);

        float V_new;

        if (psi <= psi_crit) {
            // Case 1: Use inverse transform
            float b2 = 2.0f / psi - 1.0f + sqrtf(2.0f / psi) * sqrtf(2.0f / psi - 1.0f);
            float a = m / (1.0f + b2);
            float temp = sqrtf(b2) + z;
            V_new = a * temp * temp;
        } else {
            // Case 2: Use exponential
            float p = (psi - 1.0f) / (psi + 1.0f);
            float beta = (1.0f - p) / m;

            if (u <= p) {
                V_new = 0.0f;
            } else {
                V_new = logf((1.0f - p) / (1.0f - u)) / beta;
            }
        }

        V_next[path_idx] = fmaxf(V_new, 0.0f);
    }
}
"""


class CUDAPathGenerator:
    """GPU-accelerated path generator using CuPy and custom CUDA kernels."""

    def __init__(self):
        """Initialize CUDA kernels."""
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is not installed. Install with: pip install cupy-cuda12x")

        # Compile custom kernels
        self.gbm_kernel = cp.RawKernel(GBM_KERNEL, "gbm_step")
        self.heston_qe_kernel = cp.RawKernel(HESTON_QE_KERNEL, "heston_qe_step")

        # Kernel launch parameters
        self.threads_per_block = 256

    def generate_paths(
        self,
        spots: np.ndarray,
        times: np.ndarray,
        n_paths: int,
        vol_model,
        corr_model,
        rate: float,
        div_yields: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate paths on GPU using CUDA kernels.

        Args:
            spots: Initial spot prices (n_assets,)
            times: Time grid (n_steps,)
            n_paths: Number of Monte Carlo paths
            vol_model: Volatility model (must have Heston parameters)
            corr_model: Correlation model
            rate: Risk-free rate
            div_yields: Dividend yields (n_assets,)

        Returns:
            paths: Asset paths (n_paths, n_steps, n_assets) - on CPU
            variances: Variance paths (n_paths, n_steps, n_assets) - on CPU
        """
        n_assets = len(spots)
        n_steps = len(times)

        # Allocate GPU arrays
        paths_gpu = cp.zeros((n_paths, n_steps, n_assets), dtype=cp.float32)
        paths_gpu[:, 0, :] = cp.asarray(spots, dtype=cp.float32)

        variances_gpu = None
        is_heston = hasattr(vol_model, "v0")

        if is_heston:
            variances_gpu = cp.zeros((n_paths, n_steps, n_assets), dtype=cp.float32)
            variances_gpu[:, 0, :] = vol_model.v0

        # Convert inputs to GPU
        times_gpu = cp.asarray(times, dtype=cp.float32)
        spots_gpu = cp.asarray(spots, dtype=cp.float32)
        div_yields_gpu = cp.asarray(div_yields, dtype=cp.float32)

        # Kernel launch config
        blocks_per_grid = (n_paths + self.threads_per_block - 1) // self.threads_per_block

        # Generate all random numbers upfront (more efficient)
        Z_all = cp.random.standard_normal((n_steps - 1, n_paths, n_assets), dtype=cp.float32)

        if is_heston:
            U_all = cp.random.uniform(0, 1, (n_steps - 1, n_paths, n_assets), dtype=cp.float32)

        # Simulate step by step
        for step in range(1, n_steps):
            dt = float(times[step] - times[step - 1])
            sqrt_dt = np.sqrt(dt)

            # Get correlation matrix
            current_spots = cp.asnumpy(paths_gpu[:, step - 1, :])
            mean_spots = np.mean(current_spots, axis=0)
            moneyness = mean_spots / spots
            avg_moneyness = float(np.mean(moneyness))

            corr_matrix = corr_model.get_corr_matrix({"moneyness": avg_moneyness})

            # Cholesky decomposition (on CPU, then transfer)
            try:
                L = np.linalg.cholesky(corr_matrix)
            except np.linalg.LinAlgError:
                L = self._nearest_psd_cholesky(corr_matrix)

            L_gpu = cp.asarray(L, dtype=cp.float32)

            # Correlate random numbers: Z_corr = Z_indep @ L^T
            Z_independent = Z_all[step - 1]  # (n_paths, n_assets)
            Z_correlated = Z_independent @ L_gpu.T

            # Process each asset
            for asset in range(n_assets):
                S_prev = paths_gpu[:, step - 1, asset]

                if is_heston:
                    # Update variance using custom kernel
                    V_prev = variances_gpu[:, step - 1, asset]
                    V_next = cp.zeros(n_paths, dtype=cp.float32)

                    self.heston_qe_kernel(
                        (blocks_per_grid,),
                        (self.threads_per_block,),
                        (
                            V_prev,
                            U_all[step - 1, :, asset],
                            Z_all[step - 1, :, asset],
                            V_next,
                            cp.float32(vol_model.kappa),
                            cp.float32(vol_model.theta),
                            cp.float32(vol_model.xi),
                            cp.float32(dt),
                            cp.float32(1.5),
                            cp.int32(n_paths),
                        ),
                    )

                    variances_gpu[:, step, asset] = V_next
                    vol = cp.sqrt(V_next)

                    # Apply leverage if available
                    if hasattr(vol_model, "leverage") and vol_model.leverage is not None:
                        mean_spot = float(cp.mean(S_prev))
                        leverage = vol_model.leverage(mean_spot, times[step - 1])
                        vol = vol * leverage
                else:
                    # Constant or local vol
                    mean_spot = float(cp.mean(S_prev))
                    vol_scalar = vol_model.vol(mean_spot, times[step - 1])
                    vol = cp.full(n_paths, vol_scalar, dtype=cp.float32)

                # Update spot using custom kernel
                S_next = cp.zeros(n_paths, dtype=cp.float32)

                self.gbm_kernel(
                    (blocks_per_grid,),
                    (self.threads_per_block,),
                    (
                        S_prev,
                        vol,
                        Z_correlated[:, asset],
                        S_next,
                        cp.float32(rate),
                        cp.float32(div_yields[asset]),
                        cp.float32(dt),
                        cp.float32(sqrt_dt),
                        cp.int32(n_paths),
                    ),
                )

                paths_gpu[:, step, asset] = S_next

        # Transfer back to CPU
        paths = cp.asnumpy(paths_gpu)
        variances = cp.asnumpy(variances_gpu) if variances_gpu is not None else None

        return paths, variances

    def _nearest_psd_cholesky(self, matrix: np.ndarray) -> np.ndarray:
        """Get Cholesky of nearest positive semi-definite matrix."""
        from scipy import linalg

        A = (matrix + matrix.T) / 2
        eigvals, eigvecs = linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-8)
        A_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(A_psd)))
        A_psd = D_inv_sqrt @ A_psd @ D_inv_sqrt

        return np.linalg.cholesky(A_psd)


def is_available() -> bool:
    """Check if CUDA is available."""
    return CUPY_AVAILABLE and cp.cuda.is_available()
