"""Asian option payoff evaluator for the pricing engine."""

from typing import Any

import numpy as np

from .base import PayoffEvaluator


class AsianOptionEvaluator(PayoffEvaluator):
    """Evaluates Asian option payoffs.

    Asian options pay based on the average price of the underlying over
    the observation period. Supports two variants:
    - Average price: payoff = max(average - strike, 0) for call
    - Average strike: payoff = max(spot_final - average, 0) for call
    """

    def __init__(
        self,
        observation_times: list[float],
        strike: float,
        option_type: str,
        averaging_type: str = "average_price",
        notional: float = 1.0,
    ):
        """Initialize Asian option evaluator.

        Args:
            observation_times: Times at which to observe prices for averaging
            strike: Strike price (as fraction of initial, e.g., 1.0 = 100%)
            option_type: "call" or "put"
            averaging_type: "average_price" or "average_strike"
            notional: Contract notional (default 1.0)
        """
        self.observation_times = sorted(observation_times)
        self.strike = strike
        self.option_type = option_type.lower()
        self.averaging_type = averaging_type.lower()
        self.notional = notional

        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

        if self.averaging_type not in ("average_price", "average_strike"):
            raise ValueError(
                f"averaging_type must be 'average_price' or 'average_strike', got {averaging_type}"
            )

    def evaluate(self, paths: np.ndarray, initial_spots: np.ndarray) -> dict[str, Any]:
        """Evaluate Asian option payoff for all paths.

        Args:
            paths: Asset paths (n_paths, n_steps, n_assets)
            initial_spots: Initial spot prices (n_assets,) - same for all paths

        Returns:
            Dict with:
                - payoffs: Final payoff per path (n_paths,)
                - cashflows: Cashflows at maturity (n_paths, 1)
                - average_prices: The computed averages per path (n_paths,)
        """
        n_paths, n_steps, n_assets = paths.shape

        # Calculate normalized prices (spot / initial)
        if n_assets == 1:
            normalized = paths[:, :, 0] / initial_spots[0]
        else:
            # For basket, take worst performer (most common for structured products)
            normalized = np.min(paths / initial_spots[None, None, :], axis=2)

        # Calculate average over all time steps
        average_normalized = np.mean(normalized, axis=1)

        # Get final spot (normalized)
        final_normalized = normalized[:, -1]

        # Calculate payoff based on averaging type
        if self.averaging_type == "average_price":
            # Payoff = max(average - strike, 0) for call
            if self.option_type == "call":
                intrinsic = np.maximum(average_normalized - self.strike, 0)
            else:  # put
                intrinsic = np.maximum(self.strike - average_normalized, 0)
        else:  # average_strike
            # Payoff = max(final - average, 0) for call
            if self.option_type == "call":
                intrinsic = np.maximum(final_normalized - average_normalized, 0)
            else:  # put
                intrinsic = np.maximum(average_normalized - final_normalized, 0)

        # Scale by notional
        payoffs = intrinsic * self.notional

        # Create cashflow array (paid at maturity)
        cashflows = np.zeros((n_paths, n_steps))
        cashflows[:, -1] = payoffs

        return {
            "payoffs": payoffs,
            "cashflows": cashflows,
            "average_prices": average_normalized,
        }
