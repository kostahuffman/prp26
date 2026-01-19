"""
Snowball product with accumulating coupon state.
"""

import numpy as np

from .base import PayoffEvaluator


class SnowballState:
    """
    Manages snowball coupon accumulation state.

    In a snowball:
    - Coupons accumulate if barrier not breached
    - Accumulated coupons may grow (compound)
    - All accrued coupons paid on first autocall or at maturity
    """

    def __init__(self, growth_rate: float = 0.0):
        """
        Initialize snowball state.

        Args:
            growth_rate: Rate at which accrued coupons grow (0 = simple accumulation)
        """
        self.accrued = 0.0
        self.growth_rate = growth_rate
        self.payment_history: list[tuple[float, float]] = []  # (time, amount)

    def update(self, paid: bool, coupon: float, time_step: float = 1.0) -> float:
        """
        Update snowball state and return payment amount.

        Args:
            paid: Whether coupon condition is met
            coupon: Current coupon amount
            time_step: Time since last observation (for growth calculation)

        Returns:
            Payment amount (0 if not paid, accrued + current if paid)
        """
        # Apply growth to accrued amount
        if self.growth_rate > 0 and self.accrued > 0:
            self.accrued *= 1 + self.growth_rate * time_step

        if paid:
            # Pay out all accrued plus current coupon
            payout = self.accrued + coupon
            self.payment_history.append((time_step, payout))
            self.accrued = 0.0
            return payout
        else:
            # Accumulate current coupon
            self.accrued += coupon
            return 0.0

    def get_total_accrued(self) -> float:
        """Get current accrued coupon amount."""
        return self.accrued

    def get_payment_history(self) -> list[tuple[float, float]]:
        """Get history of coupon payments."""
        return self.payment_history

    def reset(self):
        """Reset snowball state."""
        self.accrued = 0.0
        self.payment_history = []


class SnowballPayoffEvaluator(PayoffEvaluator):
    """
    Evaluates payoffs for snowball autocallable products.
    """

    def __init__(
        self,
        observation_times: list[float],
        barrier_levels: list[float],
        coupon_rate: float,
        notional: float,
        growth_rate: float = 0.0,
    ):
        self.observation_times = observation_times
        self.barrier_levels = barrier_levels
        self.coupon_rate = coupon_rate
        self.notional = notional
        self.growth_rate = growth_rate

    def evaluate(self, paths: np.ndarray, initial_levels: np.ndarray | None = None) -> np.ndarray:
        """
        Evaluate snowball payoff for simulated paths.

        Args:
            paths: Simulated paths [n_paths, n_steps, n_assets]
            initial_levels: Initial asset levels - can be 1D (n_assets,) or 2D (n_paths, n_assets)
                           Defaults to paths[:, 0, :]

        Returns:
            Payoff for each path
        """
        n_paths = paths.shape[0]

        if initial_levels is None:
            initial_levels = paths[:, 0, :]
        elif initial_levels.ndim == 1:
            # If 1D, use same initial levels for all paths
            initial_levels_1d = initial_levels
        else:
            # Already 2D
            initial_levels_1d = None

        payoffs = np.zeros(n_paths)

        for path_idx in range(n_paths):
            state = SnowballState(self.growth_rate)
            called = False

            for i, obs_time in enumerate(self.observation_times):
                if called:
                    break

                # Calculate worst-of performance
                current = paths[path_idx, i, :]
                if initial_levels_1d is not None:
                    # Use the 1D initial levels (same for all paths)
                    performance = (current / initial_levels_1d).min()
                else:
                    # Use 2D initial levels (per-path)
                    performance = (current / initial_levels[path_idx, :]).min()

                # Check if barrier breached
                barrier = self.barrier_levels[i]
                breached = performance >= barrier

                # Calculate time step
                prev_time = self.observation_times[i - 1] if i > 0 else 0.0
                time_step = obs_time - prev_time

                # Coupon amount for this period
                coupon = self.coupon_rate * time_step * self.notional

                # Update state
                payment = state.update(breached, coupon, time_step)

                if payment > 0:
                    # Autocall triggered
                    payoffs[path_idx] = self.notional + payment
                    called = True

            # If not called, evaluate maturity payoff
            if not called:
                final_performance = (paths[path_idx, -1, :] / initial_levels[path_idx, :]).min()

                # Final payment includes all accrued coupons
                final_coupon = state.get_total_accrued()

                if final_performance >= 1.0:
                    # Full capital + accrued coupons
                    payoffs[path_idx] = self.notional + final_coupon
                else:
                    # Capital loss + accrued coupons
                    payoffs[path_idx] = self.notional * final_performance + final_coupon

        return payoffs
