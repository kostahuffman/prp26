"""Numba JIT-accelerated path generation.

Requires: pip install numba

Numba provides JIT compilation that works on both CPU and GPU (CUDA).
This implementation uses @jit decorators for automatic optimization.
"""

import numpy as np

try:
    from numba import cuda, jit, prange

    NUMBA_AVAILABLE = True
    NUMBA_CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    NUMBA_AVAILABLE = False
    NUMBA_CUDA_AVAILABLE = False
    jit = lambda *args, **kwargs: lambda f: f  # No-op decorator
    prange = range


@jit(nopython=True, parallel=True, fastmath=True)
def gbm_step_numba(
    S_prev: np.ndarray,
    vol: np.ndarray,
    Z: np.ndarray,
    rate: float,
    div_yield: float,
    dt: float,
    sqrt_dt: float,
) -> np.ndarray:
    """GBM step using Numba JIT.

    Args:
        S_prev: Previous spots (n_paths,)
        vol: Volatilities (n_paths,)
        Z: Random normals (n_paths,)
        rate: Risk-free rate
        div_yield: Dividend yield
        dt: Time step
        sqrt_dt: sqrt(dt)

    Returns:
        S_next: Updated spots (n_paths,)
    """
    n_paths = len(S_prev)
    S_next = np.zeros(n_paths, dtype=np.float64)

    for i in prange(n_paths):
        drift = (rate - div_yield) * dt
        diffusion = vol[i] * sqrt_dt * Z[i]
        S_next[i] = S_prev[i] * np.exp(drift - 0.5 * vol[i] ** 2 * dt + diffusion)

    return S_next


@jit(nopython=True, parallel=True, fastmath=True)
def heston_qe_step_numba(
    V_prev: np.ndarray,
    U: np.ndarray,
    Z: np.ndarray,
    kappa: float,
    theta: float,
    sigma_v: float,
    dt: float,
    psi_crit: float = 1.5,
) -> np.ndarray:
    """Heston variance QE scheme using Numba JIT.

    Args:
        V_prev: Previous variance (n_paths,)
        U: Uniform randoms (n_paths,)
        Z: Normal randoms (n_paths,)
        kappa: Mean reversion speed
        theta: Long-term variance
        sigma_v: Vol of vol
        dt: Time step
        psi_crit: Critical psi (default 1.5)

    Returns:
        V_next: Updated variance (n_paths,)
    """
    n_paths = len(V_prev)
    V_next = np.zeros(n_paths, dtype=np.float64)

    exp_kappa_dt = np.exp(-kappa * dt)

    for i in prange(n_paths):
        V = V_prev[i]

        # QE scheme parameters
        m = theta + (V - theta) * exp_kappa_dt

        term1 = V * sigma_v**2 * exp_kappa_dt / kappa * (1 - exp_kappa_dt)
        term2 = theta * sigma_v**2 / (2 * kappa) * (1 - exp_kappa_dt) ** 2
        s2 = term1 + term2
        psi = s2 / (m**2 + 1e-10)

        if psi <= psi_crit:
            # Case 1: Inverse transform
            b2 = 2 / psi - 1 + np.sqrt(2 / psi) * np.sqrt(2 / psi - 1)
            a = m / (1 + b2)
            temp = np.sqrt(b2) + Z[i]
            V_new = a * temp**2
        else:
            # Case 2: Exponential
            p = (psi - 1) / (psi + 1)
            beta = (1 - p) / m

            if U[i] <= p:
                V_new = 0.0
            else:
                V_new = np.log((1 - p) / (1 - U[i])) / beta

        V_next[i] = max(V_new, 0.0)

    return V_next


@jit(nopython=True, parallel=True)
def correlate_normals_numba(Z_independent: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Apply Cholesky correlation to independent normals.

    Z_correlated = Z_independent @ L^T

    Args:
        Z_independent: Independent normals (n_paths, n_assets)
        L: Cholesky factor (n_assets, n_assets)

    Returns:
        Z_correlated: Correlated normals (n_paths, n_assets)
    """
    n_paths, n_assets = Z_independent.shape
    Z_correlated = np.zeros((n_paths, n_assets), dtype=np.float64)

    for i in prange(n_paths):
        for j in range(n_assets):
            sum_val = 0.0
            for k in range(n_assets):
                sum_val += Z_independent[i, k] * L[k, j]
            Z_correlated[i, j] = sum_val

    return Z_correlated


class NumbaPathGenerator:
    """JIT-accelerated path generator using Numba."""

    def __init__(self):
        """Initialize Numba path generator."""
        if not NUMBA_AVAILABLE:
            raise ImportError("Numba is not installed. Install with: pip install numba")

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
        """Generate paths using Numba JIT compilation.

        Args:
            spots: Initial spot prices (n_assets,)
            times: Time grid (n_steps,)
            n_paths: Number of Monte Carlo paths
            vol_model: Volatility model
            corr_model: Correlation model
            rate: Risk-free rate
            div_yields: Dividend yields (n_assets,)

        Returns:
            paths: Asset paths (n_paths, n_steps, n_assets)
            variances: Variance paths (n_paths, n_steps, n_assets) or None
        """
        n_assets = len(spots)
        n_steps = len(times)

        # Allocate arrays
        paths = np.zeros((n_paths, n_steps, n_assets), dtype=np.float64)
        paths[:, 0, :] = spots

        variances = None
        is_heston = hasattr(vol_model, "v0")

        if is_heston:
            variances = np.zeros((n_paths, n_steps, n_assets), dtype=np.float64)
            variances[:, 0, :] = vol_model.v0

        # Pre-generate all random numbers
        Z_all = np.random.standard_normal((n_steps - 1, n_paths, n_assets))

        if is_heston:
            U_all = np.random.uniform(0, 1, (n_steps - 1, n_paths, n_assets))

        # Simulate step by step
        for step in range(1, n_steps):
            dt = times[step] - times[step - 1]
            sqrt_dt = np.sqrt(dt)

            # Get correlation matrix
            current_spots = paths[:, step - 1, :]
            mean_spots = np.mean(current_spots, axis=0)
            moneyness = mean_spots / spots
            avg_moneyness = float(np.mean(moneyness))

            corr_matrix = corr_model.get_corr_matrix({"moneyness": avg_moneyness})

            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(corr_matrix)
            except np.linalg.LinAlgError:
                L = self._nearest_psd_cholesky(corr_matrix)

            # Correlate random numbers using Numba
            Z_correlated = correlate_normals_numba(Z_all[step - 1], L)

            # Process each asset
            for asset in range(n_assets):
                S_prev = paths[:, step - 1, asset]

                if is_heston:
                    # Update variance
                    V_prev = variances[:, step - 1, asset]
                    V_next = heston_qe_step_numba(
                        V_prev,
                        U_all[step - 1, :, asset],
                        Z_all[step - 1, :, asset],
                        vol_model.kappa,
                        vol_model.theta,
                        vol_model.xi,
                        dt,
                    )

                    variances[:, step, asset] = V_next
                    vol = np.sqrt(V_next)

                    # Apply leverage if available
                    if hasattr(vol_model, "leverage") and vol_model.leverage is not None:
                        mean_spot = np.mean(S_prev)
                        leverage = vol_model.leverage(mean_spot, times[step - 1])
                        vol = vol * leverage
                else:
                    # Constant or local vol
                    mean_spot = np.mean(S_prev)
                    vol_scalar = vol_model.vol(mean_spot, times[step - 1])
                    vol = np.full(n_paths, vol_scalar)

                # Update spots using Numba
                S_next = gbm_step_numba(
                    S_prev, vol, Z_correlated[:, asset], rate, div_yields[asset], dt, sqrt_dt
                )

                paths[:, step, asset] = S_next

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
    """Check if Numba is available."""
    return NUMBA_AVAILABLE


def cuda_is_available() -> bool:
    """Check if Numba CUDA is available."""
    return NUMBA_CUDA_AVAILABLE
