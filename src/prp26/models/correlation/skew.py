"""
Correlation skew model.
"""

from collections.abc import Callable

import numpy as np
from scipy.interpolate import interp1d


class CorrelationSkewModel:
    """
    Correlation skew surface model.

    Models correlation as function of joint moneyness:
        ρ(M) = ρ₀ + f(M)

    where M = min(S₁/S₁⁰, S₂/S₂⁰, ...) is worst-of moneyness.

    Empirical finding: correlation increases when assets move down together.
    """

    def __init__(
        self,
        base_corr: np.ndarray,
        skew_function: Callable[[float], float] | None = None,
        moneyness_grid: np.ndarray | None = None,
        correlation_adj_grid: np.ndarray | None = None,
    ):
        """
        Initialize correlation skew model.

        Args:
            base_corr: Base correlation matrix
            skew_function: Function mapping moneyness to correlation adjustment
            moneyness_grid: Grid of moneyness points
            correlation_adj_grid: Correlation adjustments at grid points
        """
        self.base_corr = base_corr
        self.n_assets = base_corr.shape[0]

        if skew_function is not None:
            self.skew_function = skew_function
        elif moneyness_grid is not None and correlation_adj_grid is not None:
            # Create interpolated skew function
            self._skew_interp = interp1d(
                moneyness_grid,
                correlation_adj_grid,
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False,
            )
            self.skew_function = lambda m: float(self._skew_interp(m))
        else:
            # Default: linear skew
            self.skew_function = lambda m: -0.2 * max(0, 1.0 - m)

    def calibrate(self, market_data) -> None:
        """
        Calibrate skew from market data.

        Uses:
        - Index option smile
        - Single-stock smiles
        - Dispersion trades
        - Correlation swaps
        """
        # Simplified: extract skew from implied correlations
        if hasattr(market_data, "get_correlation_surface"):
            corr_surface = market_data.get_correlation_surface()
            self._calibrate_from_surface(corr_surface)

    def _calibrate_from_surface(self, corr_surface):
        """Calibrate skew function from correlation surface."""
        moneyness = corr_surface["moneyness"]
        implied_corr = corr_surface["correlation"]

        # Base correlation from ATM
        atm_idx = np.argmin(np.abs(moneyness - 1.0))
        base_corr_scalar = implied_corr[atm_idx]

        # Compute adjustments
        adj = implied_corr - base_corr_scalar

        # Create interpolator
        self._skew_interp = interp1d(moneyness, adj, kind="linear", fill_value="extrapolate")
        self.skew_function = lambda m: float(self._skew_interp(m))

    def get_corr_matrix(self, state: dict | None = None) -> np.ndarray:
        """
        Get correlation matrix adjusted for current moneyness.

        Args:
            state: Must contain 'spots' and 'initial_spots'

        Returns:
            Adjusted correlation matrix
        """
        if state is None or "spots" not in state:
            return self.base_corr.copy()

        spots = state["spots"]
        initial_spots = state.get("initial_spots", spots)

        # Compute worst-of moneyness
        moneyness = spots / initial_spots
        worst_moneyness = np.min(moneyness)

        # Get correlation adjustment
        adj = self.skew_function(worst_moneyness)

        # Apply adjustment to off-diagonal elements
        adjusted_corr = self.base_corr.copy()
        n = self.n_assets

        for i in range(n):
            for j in range(i + 1, n):
                adjusted_corr[i, j] = np.clip(self.base_corr[i, j] + adj, -0.99, 0.99)
                adjusted_corr[j, i] = adjusted_corr[i, j]

        # Ensure positive definite
        adjusted_corr = self._nearest_positive_definite(adjusted_corr)

        return adjusted_corr

    def _nearest_positive_definite(self, A: np.ndarray) -> np.ndarray:
        """
        Find nearest positive definite matrix using Higham's algorithm.

        Ensures correlation matrix remains valid after adjustments.
        """
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)

        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if self._is_positive_definite(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self._is_positive_definite(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3

    def _is_positive_definite(self, A: np.ndarray) -> bool:
        """Check if matrix is positive definite."""
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False

    def plot_skew(self, moneyness_range=(0.5, 1.5), n_points=50):
        """
        Plot correlation skew function.

        Useful for visualization and validation.
        """
        import matplotlib.pyplot as plt

        moneyness = np.linspace(moneyness_range[0], moneyness_range[1], n_points)
        adj = [self.skew_function(m) for m in moneyness]
        base_corr_value = self.base_corr[0, 1]  # Assume uniform base
        total_corr = base_corr_value + np.array(adj)

        plt.figure(figsize=(10, 6))
        plt.plot(moneyness, total_corr, "b-", label="Total Correlation")
        plt.axhline(y=base_corr_value, color="r", linestyle="--", label="Base Correlation")
        plt.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5)
        plt.xlabel("Worst-of Moneyness")
        plt.ylabel("Correlation")
        plt.title("Correlation Skew Surface")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt.gcf()

    def to_dict(self) -> dict:
        """Serialize model."""
        return {
            "model_type": "skew",
            "base_corr": self.base_corr.tolist(),
            # Note: skew_function not serializable (would need parametric form)
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CorrelationSkewModel":
        """Deserialize model."""
        return cls(
            base_corr=np.array(data["base_corr"]),
        )
