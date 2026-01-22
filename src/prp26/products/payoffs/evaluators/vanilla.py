"""Vanilla option payoff evaluator for the pricing engine."""

from typing import Any

import numpy as np

from .base import PayoffEvaluator


class EuropeanOptionEvaluator(PayoffEvaluator):
    """Evaluates vanilla European option payoffs.

    Handles simple call/put options on a single underlying.
    """

    def __init__(
        self,
        maturity: float,
        strike: float,
        option_type: str,
        notional: float = 1.0,
    ):
        """Initialize vanilla option evaluator.

        Args:
            maturity: Time to maturity (in years)
            strike: Strike price (as fraction of initial, e.g., 1.0 = 100%)
            option_type: "call" or "put"
            notional: Contract notional (default 1.0)
        """
        self.observation_times = [maturity]
        self.maturity = maturity
        self.strike = strike
        self.option_type = option_type.lower()
        self.notional = notional

        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    def evaluate(self, paths: np.ndarray, initial_spots: np.ndarray) -> dict[str, Any]:
        """Evaluate vanilla option payoff for all paths.

        Args:
            paths: Asset paths (n_paths, n_steps, n_assets)
            initial_spots: Initial spot prices (n_assets,) - same for all paths

        Returns:
            Dict with:
                - payoffs: Final payoff per path (n_paths,)
                - cashflows: Cashflows at maturity (n_paths, 1)
                - terminated: All False (vanilla options don't terminate early)
                - termination_times: All set to maturity
                - termination_step: All set to 0 (final observation)
        """
        n_paths, n_steps, n_assets = paths.shape

        if n_assets != 1:
            raise ValueError(f"VanillaOption only supports single underlying, got {n_assets}")

        # Get final spot (at maturity, which should be the last step)
        final_spots = paths[:, -1, 0]  # (n_paths,)

        # Calculate performance relative to initial
        performance = final_spots / initial_spots[0]

        # Calculate intrinsic value
        if self.option_type == "call":
            intrinsic = np.maximum(performance - self.strike, 0.0)
        else:  # put
            intrinsic = np.maximum(self.strike - performance, 0.0)

        # Payoff per path
        payoffs = intrinsic * self.notional

        # Cashflows (single payment at maturity)
        cashflows = np.zeros((n_paths, 1))
        cashflows[:, 0] = payoffs

        # Vanilla options don't terminate early
        terminated = np.zeros(n_paths, dtype=bool)
        termination_times = np.full(n_paths, self.maturity)
        termination_step = np.zeros(n_paths, dtype=int)

        return {
            "payoffs": payoffs,
            "cashflows": cashflows,
            "terminated": terminated,
            "termination_times": termination_times,
            "termination_step": termination_step,
        }
