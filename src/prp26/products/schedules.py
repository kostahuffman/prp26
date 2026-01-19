"""
Observation and barrier schedules for structured products.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class ObservationType(Enum):
    """Type of observation for autocall/barrier checking."""

    DAILY = "daily"
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class BarrierType(Enum):
    """Type of barrier."""

    AUTOCALL = "autocall"
    KNOCK_IN = "knock_in"
    KNOCK_OUT = "knock_out"
    PROTECTION = "protection"


@dataclass
class ObservationSchedule:
    """Schedule of observation dates/times for product events."""

    times: list[float]  # Years from valuation date
    observation_type: ObservationType = ObservationType.DISCRETE

    def __post_init__(self):
        if not self.times:
            raise ValueError("Observation schedule cannot be empty")
        if any(t < 0 for t in self.times):
            raise ValueError("Observation times must be non-negative")
        if self.times != sorted(self.times):
            raise ValueError("Observation times must be sorted")

    def to_dict(self):
        return {
            "times": self.times,
            "observation_type": self.observation_type.value,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            times=data["times"],
            observation_type=ObservationType(data["observation_type"]),
        )


@dataclass
class BarrierSchedule:
    """Schedule of barrier levels by observation time."""

    levels: dict[float, float]  # time -> barrier level (as % of initial)
    barrier_type: BarrierType = BarrierType.AUTOCALL

    def __post_init__(self):
        if not self.levels:
            raise ValueError("Barrier schedule cannot be empty")
        if any(level <= 0 for level in self.levels.values()):
            raise ValueError("Barrier levels must be positive")

    def get_level(self, time: float) -> float | None:
        """Get barrier level at specific time."""
        return self.levels.get(time)

    def to_dict(self):
        return {
            "levels": {str(k): v for k, v in self.levels.items()},
            "barrier_type": self.barrier_type.value,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            levels={float(k): v for k, v in data["levels"].items()},
            barrier_type=BarrierType(data["barrier_type"]),
        )


@dataclass
class CouponDefinition:
    """Definition of coupon payments."""

    rate: float  # Annual coupon rate
    memory: bool = False  # Memory feature (accumulate unpaid coupons)
    snowball: bool = False  # Snowball feature (accumulate and grow)
    conditional: bool = True  # Paid only if barrier condition met

    def __post_init__(self):
        if self.rate < 0:
            raise ValueError("Coupon rate cannot be negative")
        if self.snowball and not self.memory:
            raise ValueError("Snowball requires memory feature")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
