"""
Reverse Convertible payoff evaluator.
"""

from typing import Any

import numpy as np

from .base import PayoffEvaluator


class ReverseConvertibleEvaluator(PayoffEvaluator):
    """Evaluates Reverse Convertible structured products.

    Features:
    - Regular coupon payments
    - Conversion to underlying if below strike at maturity
    - Optional barrier feature (Barrier Reverse Convertible)
    - Worst-of basket support
    """

    def __init__(
        self,
        observation_times: list[float],
        strike: float,
        coupon_rate: float,
        notional: float,
        barrier: float | None = None,
        barrier_type: str = "european",
    ):
        """Initialize Reverse Convertible evaluator.

        Args:
            observation_times: Coupon payment times (in years)
            strike: Strike level (typically 1.0 = 100% of initial)
            coupon_rate: Coupon rate per period
            notional: Contract notional
            barrier: Optional barrier level for conversion trigger
            barrier_type: "european", "american", or "bermudan"
        """
        self.observation_times = np.array(observation_times)
        self.strike = strike
        self.coupon_rate = coupon_rate
        self.notional = notional
        self.barrier = barrier
        self.barrier_type = barrier_type
        self.n_obs = len(observation_times)

    def evaluate(self, paths: np.ndarray, initial_spots: np.ndarray) -> dict[str, Any]:
        """Evaluate payoff for all paths.

        Args:
            paths: Asset paths (n_paths, n_steps, n_assets)
            initial_spots: Initial spot prices (n_assets,)

        Returns:
            Dict with:
                - payoffs: Final payoff per path (n_paths,)
                - cashflows: Cashflows at each observation (n_paths, n_obs)
                - terminated: Whether path terminated early (n_paths,) - always False
                - termination_times: Time of termination (n_paths,) - always maturity
        """
        n_paths, n_steps, n_assets = paths.shape

        # Initialize outputs
        payoffs = np.zeros(n_paths)
        cashflows = np.zeros((n_paths, self.n_obs))
        terminated = np.zeros(n_paths, dtype=bool)  # RC never terminates early
        termination_times = np.full(n_paths, self.observation_times[-1])

        # Pay coupons at each observation (except possibly last, handled differently)
        for obs_idx in range(self.n_obs - 1):
            cashflows[:, obs_idx] = self.coupon_rate * self.notional

        # Evaluate maturity
        final_spots = paths[:, -1, :]
        final_performances = final_spots / initial_spots
        worst_final_performance = np.min(final_performances, axis=1)

        # Check barrier breach (if barrier exists)
        if self.barrier is not None:
            barrier_breached = self._check_barrier_breach(
                paths, initial_spots, worst_final_performance
            )
        else:
            # No barrier - conversion triggered by strike
            barrier_breached = np.ones(n_paths, dtype=bool)

        # Conversion logic
        conversion_triggered = (worst_final_performance < self.strike) & barrier_breached

        # Final coupon
        final_coupon = self.coupon_rate * self.notional

        # Payoff at maturity
        # If not converted: notional + final coupon
        # If converted: notional * (worst_performance / strike) + final coupon
        payoffs = np.where(
            conversion_triggered,
            self.notional * (worst_final_performance / self.strike) + final_coupon,
            self.notional + final_coupon,
        )

        cashflows[:, -1] = payoffs

        return {
            "payoffs": payoffs,
            "cashflows": cashflows,
            "terminated": terminated,
            "termination_times": termination_times,
            "termination_step": np.full(n_paths, self.n_obs - 1),
        }

    def _check_barrier_breach(
        self, paths: np.ndarray, initial_spots: np.ndarray, final_performance: np.ndarray
    ) -> np.ndarray:
        """Check if barrier was breached based on barrier type.

        Args:
            paths: Asset paths (n_paths, n_steps, n_assets)
            initial_spots: Initial spots (n_assets,)
            final_performance: Final worst-of performance (n_paths,)

        Returns:
            Boolean array indicating barrier breach (n_paths,)
        """
        n_paths = paths.shape[0]

        if self.barrier_type == "european":
            # Only check at maturity
            return final_performance < self.barrier

        elif self.barrier_type == "american":
            # Check at any time during life
            # For each path, check if worst-of ever dropped below barrier
            breached = np.zeros(n_paths, dtype=bool)
            for step_idx in range(paths.shape[1]):
                step_spots = paths[:, step_idx, :]
                step_performances = step_spots / initial_spots
                step_worst = np.min(step_performances, axis=1)
                breached |= step_worst < self.barrier
            return breached

        elif self.barrier_type == "bermudan":
            # Check only at observation dates
            breached = np.zeros(n_paths, dtype=bool)
            for obs_idx in range(len(self.observation_times)):
                obs_spots = paths[:, obs_idx, :]
                obs_performances = obs_spots / initial_spots
                obs_worst = np.min(obs_performances, axis=1)
                breached |= obs_worst < self.barrier
            return breached

        else:
            raise ValueError(f"Unknown barrier type: {self.barrier_type}")
