"""
Base correlation model interfaces.
"""

from typing import Protocol

import numpy as np


class CorrelationModel(Protocol):
    """
    Protocol for correlation models.

    All correlation models must implement these methods.
    """

    def calibrate(self, market_data) -> None:
        """Calibrate correlation model to market data."""
        ...

    def get_corr_matrix(self, state: dict | None = None) -> np.ndarray:
        """
        Get correlation matrix.

        Args:
            state: Optional state for state-dependent correlation

        Returns:
            Correlation matrix [n_assets, n_assets]
        """
        ...

    def to_dict(self) -> dict:
        """Serialize model to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict) -> "CorrelationModel":
        """Deserialize model from dictionary."""
        ...
