"""Coupon payoff components."""

from typing import Any

import numpy as np

from .base import PayoffComponent, PayoffState


class CouponComponent(PayoffComponent):
    """Base class for coupon payments."""

    def __init__(self, rate: float, barrier: float, memory: bool = False):
        """Initialize coupon component.

        Args:
            rate: Coupon rate (annualized)
            barrier: Barrier level as fraction of initial (e.g., 0.7 = 70%)
            memory: Whether coupon has memory feature
        """
        self.rate = rate
        self.barrier = barrier
        self.memory = memory

    def evaluate(
        self, spots: np.ndarray, initial_spots: np.ndarray, time: float, state: PayoffState
    ) -> dict[str, Any]:
        """Evaluate coupon at observation."""
        n_paths = spots.shape[0]

        # Calculate worst-of performance
        performances = spots / initial_spots
        worst_performance = np.min(performances, axis=1)

        # Check barrier
        barrier_hit = worst_performance >= self.barrier

        # Memory feature: accumulate unpaid coupons
        if self.memory:
            unpaid = state.get("unpaid_coupons", 0.0)

            # Pay accumulated + current if barrier hit
            cashflow = np.where(barrier_hit, self.rate + unpaid, 0.0)

            # Update unpaid coupons
            new_unpaid = np.where(barrier_hit, 0.0, unpaid + self.rate)
            state.set("unpaid_coupons", new_unpaid)
        else:
            # Simple coupon: pay only if barrier hit
            cashflow = np.where(barrier_hit, self.rate, 0.0)

        return {"cashflow": cashflow, "terminated": np.zeros(n_paths, dtype=bool), "continue": True}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "memory_coupon" if self.memory else "simple_coupon",
            "rate": self.rate,
            "barrier": self.barrier,
            "memory": self.memory,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CouponComponent":
        """Deserialize from dictionary."""
        return cls(rate=data["rate"], barrier=data["barrier"], memory=data.get("memory", False))


class MemoryCoupon(CouponComponent):
    """Memory coupon (Phoenix style)."""

    def __init__(self, rate: float, barrier: float):
        super().__init__(rate=rate, barrier=barrier, memory=True)


class SnowballCoupon(PayoffComponent):
    """Snowball coupon with accumulation and growth."""

    def __init__(self, rate: float, barrier: float, growth_rate: float = 0.0):
        """Initialize snowball coupon.

        Args:
            rate: Base coupon rate
            barrier: Barrier level for payment
            growth_rate: Growth rate for snowball (e.g., 0.01 = 1% growth per period)
        """
        self.rate = rate
        self.barrier = barrier
        self.growth_rate = growth_rate

    def evaluate(
        self, spots: np.ndarray, initial_spots: np.ndarray, time: float, state: PayoffState
    ) -> dict[str, Any]:
        """Evaluate snowball coupon."""
        n_paths = spots.shape[0]

        # Get accumulated snowball
        snowball = state.get("snowball_accumulated", np.zeros(n_paths))

        # Check barrier
        performances = spots / initial_spots
        worst_performance = np.min(performances, axis=1)
        barrier_hit = worst_performance >= self.barrier

        # Accumulate this period's coupon
        snowball += self.rate

        # Apply growth if configured
        if self.growth_rate > 0:
            snowball *= 1.0 + self.growth_rate

        # Pay accumulated snowball if barrier hit
        cashflow = np.where(barrier_hit, snowball, 0.0)

        # Reset snowball for paid paths
        snowball = np.where(barrier_hit, 0.0, snowball)
        state.set("snowball_accumulated", snowball)

        return {"cashflow": cashflow, "terminated": np.zeros(n_paths, dtype=bool), "continue": True}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "snowball_coupon",
            "rate": self.rate,
            "barrier": self.barrier,
            "growth_rate": self.growth_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SnowballCoupon":
        """Deserialize from dictionary."""
        return cls(
            rate=data["rate"], barrier=data["barrier"], growth_rate=data.get("growth_rate", 0.0)
        )


class StepUpCoupon(PayoffComponent):
    """Step-up coupon that increases over time.

    Coupon rate increases at each observation date according to a schedule.
    Commonly used in autocallables to compensate for longer holding periods.
    """

    def __init__(
        self,
        rate_schedule: dict[float, float] | list[float],
        barrier: float,
        memory: bool = False,
    ):
        """Initialize step-up coupon.

        Args:
            rate_schedule: Either dict mapping {time: rate} or list of rates by observation
            barrier: Barrier level for payment
            memory: Whether coupon has memory feature
        """
        self.rate_schedule = rate_schedule
        self.barrier = barrier
        self.memory = memory

    def _get_rate_at_time(self, time: float, observation_index: int) -> float:
        """Get coupon rate at given time.

        Args:
            time: Current observation time
            observation_index: Index of current observation

        Returns:
            Coupon rate for this period
        """
        if isinstance(self.rate_schedule, dict):
            # Find closest time in schedule
            times = sorted(self.rate_schedule.keys())
            for t in reversed(times):
                if time >= t:
                    return self.rate_schedule[t]
            return self.rate_schedule[times[0]]
        else:
            # Use list index
            idx = min(observation_index, len(self.rate_schedule) - 1)
            return self.rate_schedule[idx]

    def evaluate(
        self, spots: np.ndarray, initial_spots: np.ndarray, time: float, state: PayoffState
    ) -> dict[str, Any]:
        """Evaluate step-up coupon."""
        n_paths = spots.shape[0]

        # Get observation index
        obs_index = state.get("observation_index", 0)
        state.set("observation_index", obs_index + 1)

        # Get rate for this period
        current_rate = self._get_rate_at_time(time, obs_index)

        # Calculate worst-of performance
        performances = spots / initial_spots
        worst_performance = np.min(performances, axis=1)

        # Check barrier
        barrier_hit = worst_performance >= self.barrier

        # Memory feature: accumulate unpaid coupons
        if self.memory:
            unpaid = state.get("unpaid_coupons", 0.0)

            # Pay accumulated + current if barrier hit
            cashflow = np.where(barrier_hit, current_rate + unpaid, 0.0)

            # Update unpaid coupons
            new_unpaid = np.where(barrier_hit, 0.0, unpaid + current_rate)
            state.set("unpaid_coupons", new_unpaid)
        else:
            # Simple coupon: pay only if barrier hit
            cashflow = np.where(barrier_hit, current_rate, 0.0)

        return {"cashflow": cashflow, "terminated": np.zeros(n_paths, dtype=bool), "continue": True}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "step_up_coupon",
            "rate_schedule": self.rate_schedule,
            "barrier": self.barrier,
            "memory": self.memory,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepUpCoupon":
        """Deserialize from dictionary."""
        return cls(
            rate_schedule=data["rate_schedule"],
            barrier=data["barrier"],
            memory=data.get("memory", False),
        )


class StepDownCoupon(PayoffComponent):
    """Step-down coupon that decreases over time.

    Coupon rate decreases at each observation date. Less common but can be
    used in products where early exits are preferred.
    """

    def __init__(
        self,
        rate_schedule: dict[float, float] | list[float],
        barrier: float,
        memory: bool = False,
    ):
        """Initialize step-down coupon.

        Args:
            rate_schedule: Either dict mapping {time: rate} or list of rates by observation
            barrier: Barrier level for payment
            memory: Whether coupon has memory feature
        """
        self.rate_schedule = rate_schedule
        self.barrier = barrier
        self.memory = memory

    def _get_rate_at_time(self, time: float, observation_index: int) -> float:
        """Get coupon rate at given time."""
        if isinstance(self.rate_schedule, dict):
            times = sorted(self.rate_schedule.keys())
            for t in reversed(times):
                if time >= t:
                    return self.rate_schedule[t]
            return self.rate_schedule[times[0]]
        else:
            idx = min(observation_index, len(self.rate_schedule) - 1)
            return self.rate_schedule[idx]

    def evaluate(
        self, spots: np.ndarray, initial_spots: np.ndarray, time: float, state: PayoffState
    ) -> dict[str, Any]:
        """Evaluate step-down coupon."""
        n_paths = spots.shape[0]

        # Get observation index
        obs_index = state.get("observation_index", 0)
        state.set("observation_index", obs_index + 1)

        # Get rate for this period
        current_rate = self._get_rate_at_time(time, obs_index)

        # Calculate worst-of performance
        performances = spots / initial_spots
        worst_performance = np.min(performances, axis=1)

        # Check barrier
        barrier_hit = worst_performance >= self.barrier

        # Memory feature
        if self.memory:
            unpaid = state.get("unpaid_coupons", 0.0)
            cashflow = np.where(barrier_hit, current_rate + unpaid, 0.0)
            new_unpaid = np.where(barrier_hit, 0.0, unpaid + current_rate)
            state.set("unpaid_coupons", new_unpaid)
        else:
            cashflow = np.where(barrier_hit, current_rate, 0.0)

        return {"cashflow": cashflow, "terminated": np.zeros(n_paths, dtype=bool), "continue": True}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "step_down_coupon",
            "rate_schedule": self.rate_schedule,
            "barrier": self.barrier,
            "memory": self.memory,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepDownCoupon":
        """Deserialize from dictionary."""
        return cls(
            rate_schedule=data["rate_schedule"],
            barrier=data["barrier"],
            memory=data.get("memory", False),
        )
