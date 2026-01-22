"""American option payoff evaluator for the pricing engine."""

from typing import Any

import numpy as np

from .base import PayoffEvaluator


class AmericanOptionEvaluator(PayoffEvaluator):
    """Evaluates American option payoffs.
    
    American options can be exercised at any observation time.
    Uses backward induction to calculate optimal exercise.
    """

    def __init__(
        self,
        observation_times: list[float],
        strike: float,
        option_type: str,
        notional: float = 1.0,
    ):
        """Initialize American option evaluator.

        Args:
            observation_times: All observation times (including maturity)
            strike: Strike price (as fraction of initial, e.g., 1.0 = 100%)
            option_type: "call" or "put"
            notional: Contract notional (default 1.0)
        """
        self.observation_times = sorted(observation_times)
        self.strike = strike
        self.option_type = option_type.lower()
        self.notional = notional

        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    def evaluate(self, paths: np.ndarray, initial_spots: np.ndarray) -> dict[str, Any]:
        """Evaluate American option payoff for all paths.

        For simplicity, this implementation uses the maximum intrinsic value
        along the path. A full implementation would use backward induction
        with continuation values.

        Args:
            paths: Asset paths (n_paths, n_steps, n_assets)
            initial_spots: Initial spot prices (n_assets,) - same for all paths

        Returns:
            Dict with:
                - payoffs: Final payoff per path (n_paths,)
                - cashflows: Cashflows at each observation (n_paths, n_obs)
                - exercise_times: Time of exercise per path (n_paths,)
        """
        n_paths, n_steps, n_assets = paths.shape

        # Calculate normalized prices (spot / initial)
        # Handle single asset case
        if n_assets == 1:
            normalized = paths[:, :, 0] / initial_spots[0]
        else:
            # For basket, take worst performer (most common for structured products)
            normalized = np.min(paths / initial_spots[None, None, :], axis=2)

        # Calculate intrinsic value at each step
        if self.option_type == "call":
            intrinsic = np.maximum(normalized - self.strike, 0)
        else:  # put
            intrinsic = np.maximum(self.strike - normalized, 0)

        # Simple American: take max intrinsic value along path
        # (Full implementation would use backward induction)
        max_intrinsic = np.max(intrinsic, axis=1)
        exercise_step = np.argmax(intrinsic, axis=1)

        # Scale by notional
        payoffs = max_intrinsic * self.notional

        # Create cashflow array (paid at exercise time)
        cashflows = np.zeros((n_paths, n_steps))
        for path_idx, step in enumerate(exercise_step):
            cashflows[path_idx, step] = payoffs[path_idx]

        # Calculate exercise times (approximate)
        if len(self.observation_times) == n_steps:
            exercise_times = np.array([self.observation_times[step] for step in exercise_step])
        else:
            # Interpolate observation times
            time_grid = np.linspace(0, self.observation_times[-1], n_steps)
            exercise_times = time_grid[exercise_step]

        return {
            "payoffs": payoffs,
            "cashflows": cashflows,
            "exercise_times": exercise_times,
            "exercise_steps": exercise_step,
        }
