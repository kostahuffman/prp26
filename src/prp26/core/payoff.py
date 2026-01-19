"""Generic payoff evaluator interface."""

from typing import Any, Protocol

import numpy as np


class PayoffEvaluator(Protocol):
    """Protocol for payoff evaluation.

    This provides a unified interface for both:
    - Legacy hardcoded products (AutocallableProduct, etc.)
    - New compositional payoffs (ComposablePayoff)
    """

    def evaluate(
        self, paths: np.ndarray, times: np.ndarray, initial_spots: np.ndarray, notional: float
    ) -> dict[str, Any]:
        """Evaluate payoff along Monte Carlo paths.

        Args:
            paths: Asset paths (n_paths, n_steps, n_assets)
            times: Time grid (n_steps,)
            initial_spots: Initial spot prices (n_assets,)
            notional: Contract notional

        Returns:
            Dict with at least:
                - "payoffs": Final payoff per path (n_paths,)
                - "cashflows": Cashflows at each time (n_paths, n_steps)
        """
        ...
