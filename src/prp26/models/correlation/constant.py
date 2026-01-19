"""
Constant correlation model.
"""
import numpy as np


class ConstantCorrelationModel:
    """
    Static correlation matrix.

    Simplest model: correlations constant across time and market states.
    """

    def __init__(self, corr_matrix: np.ndarray):
        """
        Initialize with constant correlation matrix.

        Args:
            corr_matrix: Correlation matrix [n_assets, n_assets]
        """
        self.corr_matrix = corr_matrix
        self._validate()

    def _validate(self):
        """Validate correlation matrix properties."""
        if len(self.corr_matrix.shape) != 2:
            raise ValueError("Correlation matrix must be 2D")
        if self.corr_matrix.shape[0] != self.corr_matrix.shape[1]:
            raise ValueError("Correlation matrix must be square")
        if not np.allclose(self.corr_matrix, self.corr_matrix.T):
            raise ValueError("Correlation matrix must be symmetric")
        if not np.allclose(np.diag(self.corr_matrix), 1.0):
            raise ValueError("Diagonal of correlation matrix must be 1")

        # Check positive semi-definite
        eigenvalues = np.linalg.eigvalsh(self.corr_matrix)
        if np.any(eigenvalues < -1e-10):
            raise ValueError("Correlation matrix must be positive semi-definite")

    def calibrate(self, market_data) -> None:
        """No calibration needed for constant model."""
        pass

    def get_corr_matrix(self, state: dict | None = None) -> np.ndarray:
        """Return constant correlation matrix."""
        return self.corr_matrix.copy()

    def to_dict(self) -> dict:
        """Serialize model."""
        return {
            "model_type": "constant",
            "corr_matrix": self.corr_matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConstantCorrelationModel":
        """Deserialize model."""
        return cls(corr_matrix=np.array(data["corr_matrix"]))


def create_uniform_correlation(n_assets: int, rho: float) -> np.ndarray:
    """
    Create uniform correlation matrix.

    Args:
        n_assets: Number of assets
        rho: Pairwise correlation

    Returns:
        Correlation matrix with constant off-diagonal correlation
    """
    corr = rho * np.ones((n_assets, n_assets))
    np.fill_diagonal(corr, 1.0)
    return corr
