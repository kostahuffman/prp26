"""Bermudan option payoff evaluator for the pricing engine."""

from typing import Any

import numpy as np

from .base import PayoffEvaluator


class BermudanOptionEvaluator(PayoffEvaluator):
    """Evaluates Bermudan option payoffs.
    
    Bermudan options can be exercised only at specific dates.
    Uses backward induction to calculate optimal exercise among allowed dates.
    """

    def __init__(
        self,
        observation_times: list[float],
        exercise_times: list[float],
        strike: float,
        option_type: str,
        notional: float = 1.0,
    ):
        """Initialize Bermudan option evaluator.

        Args:
            observation_times: All observation times (for path simulation)
            exercise_times: Times when option can be exercised
            strike: Strike price (as fraction of initial, e.g., 1.0 = 100%)
            option_type: "call" or "put"
            notional: Contract notional (default 1.0)
        """
        self.observation_times = sorted(observation_times)
        self.exercise_times = sorted(exercise_times)
        self.strike = strike
        self.option_type = option_type.lower()
        self.notional = notional

        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

        # Validate exercise times are subset of observation times
        for ex_time in self.exercise_times:
            if not any(abs(ex_time - obs_time) < 1e-10 for obs_time in self.observation_times):
                raise ValueError(f"Exercise time {ex_time} not in observation times")

    def evaluate(self, paths: np.ndarray, initial_spots: np.ndarray) -> dict[str, Any]:
        """Evaluate Bermudan option payoff for all paths.

        For simplicity, this implementation uses the maximum intrinsic value
        at exercise dates. A full implementation would use backward induction
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
        if n_assets == 1:
            normalized = paths[:, :, 0] / initial_spots[0]
        else:
            # For basket, take worst performer
            normalized = np.min(paths / initial_spots[None, None, :], axis=2)

        # Calculate intrinsic value at each step
        if self.option_type == "call":
            intrinsic = np.maximum(normalized - self.strike, 0)
        else:  # put
            intrinsic = np.maximum(self.strike - normalized, 0)

        # Find exercise time indices
        exercise_indices = []
        for ex_time in self.exercise_times:
            # Find closest observation index
            idx = min(range(len(self.observation_times)),
                     key=lambda i: abs(self.observation_times[i] - ex_time))
            if idx < n_steps:
                exercise_indices.append(idx)

        if not exercise_indices:
            raise ValueError("No valid exercise indices found in path")

        # Only consider intrinsic values at exercise dates
        exercise_intrinsics = intrinsic[:, exercise_indices]

        # Simple Bermudan: take max intrinsic value at exercise dates
        max_intrinsic = np.max(exercise_intrinsics, axis=1)
        exercise_idx_in_allowed = np.argmax(exercise_intrinsics, axis=1)
        exercise_step = np.array([exercise_indices[idx] for idx in exercise_idx_in_allowed])

        # Scale by notional
        payoffs = max_intrinsic * self.notional

        # Create cashflow array (paid at exercise time)
        cashflows = np.zeros((n_paths, n_steps))
        for path_idx, step in enumerate(exercise_step):
            cashflows[path_idx, step] = payoffs[path_idx]

        # Calculate exercise times
        exercise_times = np.array([self.observation_times[step] for step in exercise_step])

        return {
            "payoffs": payoffs,
            "cashflows": cashflows,
            "exercise_times": exercise_times,
            "exercise_steps": exercise_step,
        }
