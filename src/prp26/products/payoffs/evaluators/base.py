"""
Base interface for payoff evaluators.

Evaluators are optimized for Monte Carlo simulation in the pricing engine.
They process full path arrays and return payoff results.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class PayoffEvaluator(ABC):
    """Base class for all payoff evaluators.
    
    Evaluators are product-specific and optimized for the PricingEngine.
    They process Monte Carlo paths and compute:
    - Final payoffs per path
    - Cashflows at each observation
    - Early termination logic
    - Termination times
    """

    @abstractmethod
    def evaluate(self, paths: np.ndarray, initial_spots: np.ndarray) -> dict[str, Any]:
        """Evaluate payoff for all Monte Carlo paths.

        Args:
            paths: Asset paths (n_paths, n_steps, n_assets)
            initial_spots: Initial spot prices (n_assets,) - same for all paths

        Returns:
            Dict with:
                - payoffs: Final payoff per path (n_paths,)
                - cashflows: Cashflows at each observation (n_paths, n_obs)
                - terminated: Whether path terminated early (n_paths,)
                - termination_times: Time of termination (n_paths,)
                - termination_step: Which observation terminated (n_paths,)
        """
        ...

    def get_observation_times(self) -> list[float]:
        """Get observation times for this evaluator."""
        return getattr(self, 'observation_times', [])

    def get_product_type(self) -> str:
        """Get product type identifier."""
        return self.__class__.__name__.replace('PayoffEvaluator', '').lower()
