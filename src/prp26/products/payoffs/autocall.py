"""Autocall (early termination) components."""

from typing import Any

import numpy as np

from .base import PayoffComponent, PayoffState


class AutocallComponent(PayoffComponent):
    """Autocall feature for early termination.

    If worst-of performance is above barrier, the product terminates
    early and returns the notional (represented as 1.0).
    """

    def __init__(self, barrier: float, redemption: float = 1.0):
        """Initialize autocall component.

        Args:
            barrier: Autocall barrier level (e.g., 0.95 = 95% of initial)
            redemption: Redemption amount (typically 1.0 = 100% of notional)
        """
        self.barrier = barrier
        self.redemption = redemption

    def evaluate(
        self, spots: np.ndarray, initial_spots: np.ndarray, time: float, state: PayoffState
    ) -> dict[str, Any]:
        """Evaluate autocall trigger."""
        n_paths = spots.shape[0]

        # Calculate worst-of performance
        performances = spots / initial_spots
        worst_performance = np.min(performances, axis=1)

        # Check if autocall triggered
        autocall_triggered = worst_performance >= self.barrier

        # Pay redemption amount if triggered
        cashflow = np.where(autocall_triggered, self.redemption, 0.0)

        # Mark terminated paths
        terminated = autocall_triggered

        # Track autocall probability
        n_autocalled = state.get("n_autocalled", 0)
        state.set("n_autocalled", n_autocalled + np.sum(autocall_triggered))

        return {"cashflow": cashflow, "terminated": terminated, "continue": True}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"type": "autocall", "barrier": self.barrier, "redemption": self.redemption}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutocallComponent":
        """Deserialize from dictionary."""
        return cls(barrier=data["barrier"], redemption=data.get("redemption", 1.0))
