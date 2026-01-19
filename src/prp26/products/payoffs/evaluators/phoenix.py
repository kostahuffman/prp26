"""Phoenix autocallable payoff evaluator for the pricing engine."""

from typing import Any

import numpy as np

from .base import PayoffEvaluator


class PhoenixPayoffEvaluator(PayoffEvaluator):
    """Evaluates Phoenix autocallable with memory coupons.

    This is designed to work with the PricingEngine and handles:
    - Memory coupons (accumulate unpaid coupons)
    - Autocall barriers
    - Worst-of basket
    - Capital at risk at maturity
    """

    def __init__(
        self,
        observation_times: list,
        autocall_barriers: list,
        coupon_barriers: list,
        coupon_rate: float,
        notional: float = 1.0,
    ):
        """Initialize Phoenix evaluator.

        Args:
            observation_times: Times to check autocall/coupon (in years)
            autocall_barriers: Autocall barrier at each observation (as fraction, e.g., 0.95)
            coupon_barriers: Coupon barrier at each observation (e.g., 0.70)
            coupon_rate: Coupon per period (e.g., 0.02 = 2%)
            notional: Contract notional (default 1.0)
        """
        self.observation_times = np.array(observation_times)
        self.autocall_barriers = np.array(autocall_barriers)
        self.coupon_barriers = np.array(coupon_barriers)
        self.coupon_rate = coupon_rate
        self.notional = notional
        self.n_obs = len(observation_times)

    def evaluate(self, paths: np.ndarray, initial_spots: np.ndarray) -> dict[str, Any]:
        """Evaluate payoff for all paths.

        Args:
            paths: Asset paths (n_paths, n_steps, n_assets)
            initial_spots: Initial spot prices (n_assets,) - same for all paths

        Returns:
            Dict with:
                - payoffs: Final payoff per path (n_paths,)
                - cashflows: Cashflows at each observation (n_paths, n_obs)
                - terminated: Whether path terminated early (n_paths,)
                - termination_step: Which observation terminated (n_paths,)
        """
        n_paths, n_steps, n_assets = paths.shape

        # Initialize outputs
        payoffs = np.zeros(n_paths)
        cashflows = np.zeros((n_paths, self.n_obs))
        terminated = np.zeros(n_paths, dtype=bool)
        termination_step = np.full(n_paths, self.n_obs - 1)  # Default to maturity

        # Track memory coupons for each path
        unpaid_coupons = np.zeros(n_paths)

        # Evaluate at each observation
        for obs_idx in range(self.n_obs):
            # Paths that haven't terminated yet
            active = ~terminated
            if not np.any(active):
                break

            # Get spots at this observation
            # Assume paths align with observation times (step obs_idx)
            current_spots = paths[active, obs_idx, :]  # (n_active, n_assets)

            # Calculate worst-of performance
            performances = current_spots / initial_spots  # (n_active, n_assets)
            worst_performance = np.min(performances, axis=1)  # (n_active,)

            # Check autocall barrier
            autocall_barrier = self.autocall_barriers[obs_idx]
            autocall_triggered = worst_performance >= autocall_barrier

            # Check coupon barrier
            coupon_barrier = self.coupon_barriers[obs_idx]
            coupon_earned = worst_performance >= coupon_barrier

            # Update memory coupons
            active_unpaid = unpaid_coupons[active]
            coupon_payment = np.where(
                coupon_earned,
                active_unpaid + self.coupon_rate,  # Pay accumulated + current
                0.0,
            )

            # Update unpaid for next period
            new_unpaid = np.where(
                coupon_earned,
                0.0,
                active_unpaid + self.coupon_rate,  # Accumulate
            )
            unpaid_coupons[active] = new_unpaid

            # Handle autocall
            autocalled_paths = np.where(active)[0][autocall_triggered]

            if len(autocalled_paths) > 0:
                # Autocalled paths receive: notional + coupon
                payoffs[autocalled_paths] = self.notional + coupon_payment[autocall_triggered]
                terminated[autocalled_paths] = True
                termination_step[autocalled_paths] = obs_idx
                cashflows[autocalled_paths, obs_idx] = payoffs[autocalled_paths]

            # Non-autocalled paths that earned coupon
            non_autocalled = np.where(active)[0][~autocall_triggered & coupon_earned]
            if len(non_autocalled) > 0:
                cashflows[non_autocalled, obs_idx] = coupon_payment[
                    ~autocall_triggered & coupon_earned
                ]

        # Handle maturity for paths that didn't autocall
        surviving_paths = ~terminated
        if np.any(surviving_paths):
            # Final performance
            final_spots = paths[surviving_paths, -1, :]
            final_performances = final_spots / initial_spots
            worst_final_performance = np.min(final_performances, axis=1)

            # Pay any accumulated coupons
            final_coupons = unpaid_coupons[surviving_paths]

            # Capital payoff: notional * worst_performance (capital at risk)
            capital_payoff = self.notional * worst_final_performance

            # Total maturity payoff
            maturity_payoff = capital_payoff + final_coupons

            payoffs[surviving_paths] = maturity_payoff
            cashflows[surviving_paths, -1] = maturity_payoff

        return {
            "payoffs": payoffs,
            "cashflows": cashflows,
            "terminated": terminated,
            "termination_step": termination_step,
            "termination_times": self.observation_times[termination_step],
        }
