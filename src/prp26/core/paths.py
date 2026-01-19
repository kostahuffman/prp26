"""Path generation for Monte Carlo simulation."""

import numpy as np

from ..models.correlation.base import CorrelationModel
from ..models.volatility.base import VolatilityModel


class PathGenerator:
    """Generates correlated asset paths using SLV dynamics.

    This is the core Monte Carlo path generator extracted from the
    proof-of-concept code. It handles:
    - SLV dynamics (Heston + leverage function)
    - Correlated Brownian motions (via Cholesky)
    - Time discretization
    - Optional GPU acceleration

    Institutional features:
    - Brownian bridge for barrier monitoring (future)
    - Quasi-random numbers (future)
    - Variance reduction (future)
    """

    def __init__(
        self,
        vol_model: VolatilityModel,
        corr_model: CorrelationModel,
        rate: float,
        dividend_yields: np.ndarray = None,
        seed: int | None = None,
    ):
        """Initialize path generator.

        Args:
            vol_model: Volatility model (SLV, Heston, Local Vol, etc.)
            corr_model: Correlation model (constant, skew, etc.)
            rate: Risk-free rate (continuous)
            dividend_yields: Dividend yield per asset (continuous)
            seed: Random seed for reproducibility
        """
        self.vol_model = vol_model
        self.corr_model = corr_model
        self.rate = rate
        self.dividend_yields = dividend_yields
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def generate_paths(
        self, spots: np.ndarray, times: np.ndarray, n_paths: int, backend: str = "cpu"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate correlated asset paths.

        Args:
            spots: Initial spot prices (n_assets,)
            times: Time grid for simulation (n_steps,)
            n_paths: Number of Monte Carlo paths
            backend: "cpu" or "gpu" (GPU requires CuPy)

        Returns:
            paths: Asset paths (n_paths, n_steps, n_assets)
            variances: Variance paths if SLV (n_paths, n_steps, n_assets) or None
        """
        n_assets = len(spots)
        n_steps = len(times)

        if backend == "gpu":
            try:
                return self._generate_paths_gpu(spots, times, n_paths, n_assets, n_steps)
            except ImportError:
                print("⚠️  CuPy not available, falling back to CPU")
                backend = "cpu"

        return self._generate_paths_cpu(spots, times, n_paths, n_assets, n_steps)

    def _generate_paths_cpu(
        self, spots: np.ndarray, times: np.ndarray, n_paths: int, n_assets: int, n_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """CPU implementation of path generation."""
        paths = np.zeros((n_paths, n_steps, n_assets))
        paths[:, 0, :] = spots

        # Initialize variances for SLV
        variances = None
        if hasattr(self.vol_model, "v0"):
            # This is a Heston-based model
            variances = np.zeros((n_paths, n_steps, n_assets))
            variances[:, 0, :] = self.vol_model.v0

        # Dividend yields (default to zero)
        if self.dividend_yields is None:
            div_yields = np.zeros(n_assets)
        else:
            div_yields = self.dividend_yields

        # Simulate paths step by step
        for step in range(1, n_steps):
            dt = times[step] - times[step - 1]
            sqrt_dt = np.sqrt(dt)

            # Get correlation matrix for current state
            current_spots = paths[:, step - 1, :]
            mean_spots = np.mean(current_spots, axis=0)
            moneyness = mean_spots / spots  # Relative to initial

            # Average moneyness for correlation
            avg_moneyness = np.mean(moneyness)
            corr_matrix = self.corr_model.get_corr_matrix({"moneyness": avg_moneyness})

            # Cholesky decomposition for correlated Brownian motions
            try:
                L = np.linalg.cholesky(corr_matrix)
            except np.linalg.LinAlgError:
                # If not positive definite, use nearest PSD matrix
                L = self._nearest_psd_cholesky(corr_matrix)

            # Generate independent standard normals
            Z_independent = np.random.standard_normal((n_paths, n_assets))

            # Correlate them: Z_correlated = Z_independent @ L^T
            Z_correlated = Z_independent @ L.T

            # Simulate each asset
            for asset in range(n_assets):
                S = paths[:, step - 1, asset]

                # Get volatility from model
                if variances is not None:
                    # SLV: use variance state and leverage function
                    V = variances[:, step - 1, asset]
                    vol = np.sqrt(V)

                    # Apply leverage function if available
                    if hasattr(self.vol_model, "leverage") and self.vol_model.leverage is not None:
                        # Use mean spot for leverage lookup
                        mean_spot = np.mean(S)
                        leverage = self.vol_model.leverage(mean_spot, times[step - 1])
                        vol = vol * leverage

                    # Update variance (Heston QE scheme)
                    V_next = self._simulate_variance_qe(V, dt, self.vol_model)
                    variances[:, step, asset] = V_next
                else:
                    # Local vol or constant vol
                    mean_spot = np.mean(S)
                    vol = self.vol_model.vol(mean_spot, times[step - 1])

                # GBM step with drift and diffusion
                drift = (self.rate - div_yields[asset]) * dt
                diffusion = vol * sqrt_dt * Z_correlated[:, asset]

                paths[:, step, asset] = S * np.exp(drift - 0.5 * (vol**2) * dt + diffusion)

        return paths, variances

    def _simulate_variance_qe(self, V: np.ndarray, dt: float, model) -> np.ndarray:
        """Simulate variance using QE scheme (Heston)."""
        kappa = model.kappa
        theta = model.theta
        sigma_v = model.xi  # vol of vol parameter

        # QE scheme parameters
        m = theta + (V - theta) * np.exp(-kappa * dt)
        s2 = (
            V * sigma_v**2 * np.exp(-kappa * dt) / kappa * (1 - np.exp(-kappa * dt))
            + theta * sigma_v**2 / (2 * kappa) * (1 - np.exp(-kappa * dt)) ** 2
        )
        psi = s2 / (m**2)

        # Generate uniform random numbers
        U = np.random.uniform(0, 1, size=V.shape)

        # QE scheme
        psi_crit = 1.5
        V_next = np.zeros_like(V)

        # Case 1: psi <= psi_crit (use inverse transform)
        mask1 = psi <= psi_crit
        if np.any(mask1):
            b2 = 2 / psi[mask1] - 1 + np.sqrt(2 / psi[mask1]) * np.sqrt(2 / psi[mask1] - 1)
            a = m[mask1] / (1 + b2)

            Z = np.random.standard_normal(size=U[mask1].shape)
            V_next[mask1] = a * (np.sqrt(b2) + Z) ** 2

        # Case 2: psi > psi_crit (use exponential)
        mask2 = ~mask1
        if np.any(mask2):
            p = (psi[mask2] - 1) / (psi[mask2] + 1)
            beta = (1 - p) / m[mask2]

            V_next[mask2] = np.where(U[mask2] <= p, 0.0, np.log((1 - p) / (1 - U[mask2])) / beta)

        return np.maximum(V_next, 0.0)  # Ensure non-negative

    def _generate_paths_gpu(
        self, spots: np.ndarray, times: np.ndarray, n_paths: int, n_assets: int, n_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """GPU implementation using available backend (CUDA or Numba).

        Tries backends in order of preference:
        1. CUDA (CuPy) - fastest, requires NVIDIA GPU
        2. Numba - JIT compilation, works on CPU and GPU
        3. CPU fallback - if no GPU backend available
        """
        # Prepare common parameters
        if self.dividend_yields is None:
            div_yields = np.zeros(n_assets)
        else:
            div_yields = self.dividend_yields

        # Try CUDA backend first (fastest)
        try:
            from ..gpu import cuda_kernels

            if cuda_kernels.is_available():
                generator = cuda_kernels.CUDAPathGenerator()
                return generator.generate_paths(
                    spots, times, n_paths, self.vol_model, self.corr_model, self.rate, div_yields
                )
        except ImportError:
            pass
        except Exception as e:
            print(f"⚠️  CUDA backend failed: {e}, trying Numba...")

        # Try Numba backend (good CPU/GPU performance)
        try:
            from ..gpu import numba_kernels

            if numba_kernels.is_available():
                generator = numba_kernels.NumbaPathGenerator()
                return generator.generate_paths(
                    spots, times, n_paths, self.vol_model, self.corr_model, self.rate, div_yields
                )
        except ImportError:
            pass
        except Exception as e:
            print(f"⚠️  Numba backend failed: {e}, falling back to CPU...")

        # Fallback to CPU
        print("ℹ️  No GPU backend available, using CPU (install cupy or numba for acceleration)")
        return self._generate_paths_cpu(spots, times, n_paths, n_assets, n_steps)

    def _nearest_psd_cholesky(self, matrix: np.ndarray) -> np.ndarray:
        """Get Cholesky of nearest positive semi-definite matrix.

        Uses Higham's algorithm.
        """
        from scipy import linalg

        # Symmetrize
        A = (matrix + matrix.T) / 2

        # Eigenvalue decomposition
        eigvals, eigvecs = linalg.eigh(A)

        # Clip negative eigenvalues
        eigvals = np.maximum(eigvals, 1e-8)

        # Reconstruct
        A_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Ensure unit diagonal
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(A_psd)))
        A_psd = D_inv_sqrt @ A_psd @ D_inv_sqrt

        return np.linalg.cholesky(A_psd)
