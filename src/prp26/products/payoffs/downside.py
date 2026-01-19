"""Downside protection components."""

from typing import Any

import numpy as np

from .base import PayoffComponent, PayoffState


class DownsideComponent(PayoffComponent):
    """Base class for downside payoff at maturity."""

    def __init__(self, strike: float, participation: float = 1.0):
        """Initialize downside component.

        Args:
            strike: Strike level (typically 1.0 = 100% of initial)
            participation: Participation in downside (1.0 = 100%, 0.0 = full protection)
        """
        self.strike = strike
        self.participation = participation

    def evaluate(
        self, spots: np.ndarray, initial_spots: np.ndarray, time: float, state: PayoffState
    ) -> dict[str, Any]:
        """Evaluate downside at maturity (only at final observation)."""
        # This is typically only evaluated at maturity
        # Base implementation does nothing
        return {
            "cashflow": np.zeros(spots.shape[0]),
            "terminated": np.zeros(spots.shape[0], dtype=bool),
            "continue": True,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"type": "downside", "strike": self.strike, "participation": self.participation}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DownsideComponent":
        """Deserialize from dictionary."""
        return cls(strike=data.get("strike", 1.0), participation=data.get("participation", 1.0))


class WorstOfPut(DownsideComponent):
    """Worst-of put at maturity.

    Payoff = 1 + participation * min(0, worst_performance - strike)

    Example:
        - strike = 1.0 (at-the-money)
        - participation = 1.0 (full downside)
        - If worst asset is at 80%, payoff = 1 + 1 * (0.8 - 1.0) = 0.8
    """

    def __init__(self, strike: float = 1.0, participation: float = 1.0):
        super().__init__(strike=strike, participation=participation)

    def evaluate(
        self, spots: np.ndarray, initial_spots: np.ndarray, time: float, state: PayoffState
    ) -> dict[str, Any]:
        """Evaluate worst-of put at maturity."""
        n_paths = spots.shape[0]

        # Calculate worst-of performance
        performances = spots / initial_spots
        worst_performance = np.min(performances, axis=1)

        # Put payoff: participation * max(strike - worst_performance, 0)
        put_payoff = self.participation * np.maximum(self.strike - worst_performance, 0.0)  # noqa: F841

        # Total payoff: strike - put_payoff = strike + participation * (worst_performance - strike) when ITM
        # Simplified: strike + participation * min(worst_performance - strike, 0)
        cashflow = self.strike + self.participation * np.minimum(
            worst_performance - self.strike, 0.0
        )

        return {
            "cashflow": cashflow,
            "terminated": np.ones(n_paths, dtype=bool),  # This ends the product
            "continue": False,  # Stop evaluating after this
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"type": "worst_of_put", "strike": self.strike, "participation": self.participation}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorstOfPut":
        """Deserialize from dictionary."""
        return cls(strike=data.get("strike", 1.0), participation=data.get("participation", 1.0))
