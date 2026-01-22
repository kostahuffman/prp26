"""Option payoff components (European, American, Asian, Bermudan).

These components provide standard option payoffs that can be composed
with other features like autocalls, coupons, etc.
"""

from typing import Any

import numpy as np

from .base import PayoffComponent, PayoffState


class EuropeanOption(PayoffComponent):
    """European option - exercises only at maturity.

    Payoff = max(phi * (S_T - K), 0)
    where phi = 1 for call, -1 for put
    """

    def __init__(
        self,
        strike: float,
        option_type: str = "call",
        participation: float = 1.0,
    ):
        """Initialize European option.

        Args:
            strike: Strike level (as fraction of initial, e.g., 1.0 = 100%)
            option_type: "call" or "put"
            participation: Participation rate (default 1.0)
        """
        self.strike = strike
        self.option_type = option_type.lower()
        self.participation = participation
        self.phi = 1.0 if self.option_type == "call" else -1.0

        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    def evaluate(
        self, spots: np.ndarray, initial_spots: np.ndarray, time: float, state: PayoffState
    ) -> dict[str, Any]:
        """Evaluate European option at observation.

        Only pays off at final observation (maturity).
        """
        n_paths = spots.shape[0]

        # Check if this is the final observation
        is_final = state.get("is_final_observation", False)

        if not is_final:
            # Not at maturity yet - no cashflow
            return {
                "cashflow": np.zeros(n_paths),
                "terminated": np.zeros(n_paths, dtype=bool),
                "continue": True,
            }

        # At maturity - calculate payoff
        # Calculate worst-of performance
        performances = spots / initial_spots
        worst_performance = np.min(performances, axis=1)

        # Intrinsic value
        intrinsic = np.maximum(self.phi * (worst_performance - self.strike), 0.0)
        cashflow = intrinsic * self.participation

        return {
            "cashflow": cashflow,
            "terminated": np.zeros(n_paths, dtype=bool),  # European doesn't terminate early
            "continue": True,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "european_option",
            "strike": self.strike,
            "option_type": self.option_type,
            "participation": self.participation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EuropeanOption":
        """Deserialize from dictionary."""
        return cls(
            strike=data["strike"],
            option_type=data["option_type"],
            participation=data.get("participation", 1.0),
        )


class AmericanOption(PayoffComponent):
    """American option - can exercise at any time.

    The component provides intrinsic value at each observation.
    The pricing engine must handle optimal exercise timing.

    Intrinsic(t) = max(phi * (S_t - K), 0)
    """

    def __init__(
        self,
        strike: float,
        option_type: str = "call",
        participation: float = 1.0,
    ):
        """Initialize American option.

        Args:
            strike: Strike level (as fraction of initial)
            option_type: "call" or "put"
            participation: Participation rate
        """
        self.strike = strike
        self.option_type = option_type.lower()
        self.participation = participation
        self.phi = 1.0 if self.option_type == "call" else -1.0

        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    def evaluate(
        self, spots: np.ndarray, initial_spots: np.ndarray, time: float, state: PayoffState
    ) -> dict[str, Any]:
        """Evaluate American option intrinsic value.

        Returns intrinsic value at current observation.
        The engine must decide whether to exercise based on continuation value.

        Note: This basic implementation does not include optimal exercise logic.
        For full American option pricing, use a dedicated American option evaluator
        with regression-based methods (e.g., Longstaff-Schwartz).
        """
        n_paths = spots.shape[0]

        # Calculate worst-of performance
        performances = spots / initial_spots
        worst_performance = np.min(performances, axis=1)

        # Intrinsic value (what we'd get if we exercise now)
        intrinsic = np.maximum(self.phi * (worst_performance - self.strike), 0.0)

        # Store intrinsic value in state for engine to use in exercise decision
        state.set("american_option_intrinsic", intrinsic)

        # For now, we don't exercise - just track intrinsic value
        # A full implementation would need continuation value estimation
        return {
            "cashflow": np.zeros(n_paths),  # No exercise yet
            "terminated": np.zeros(n_paths, dtype=bool),
            "continue": True,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "american_option",
            "strike": self.strike,
            "option_type": self.option_type,
            "participation": self.participation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AmericanOption":
        """Deserialize from dictionary."""
        return cls(
            strike=data["strike"],
            option_type=data["option_type"],
            participation=data.get("participation", 1.0),
        )


class AsianOption(PayoffComponent):
    """Asian option - payoff depends on average of spot prices.

    Average Price Asian (most common):
        Payoff = max(phi * (S_avg - K), 0)

    Average Strike Asian:
        Payoff = max(phi * (S_T - S_avg), 0)
    """

    def __init__(
        self,
        strike: float,
        option_type: str = "call",
        asian_type: str = "average_price",
        participation: float = 1.0,
    ):
        """Initialize Asian option.

        Args:
            strike: Strike level (for average_price) or ignored (for average_strike)
            option_type: "call" or "put"
            asian_type: "average_price" or "average_strike"
            participation: Participation rate
        """
        self.strike = strike
        self.option_type = option_type.lower()
        self.asian_type = asian_type.lower()
        self.participation = participation
        self.phi = 1.0 if self.option_type == "call" else -1.0

        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
        if self.asian_type not in ("average_price", "average_strike"):
            raise ValueError(
                f"asian_type must be 'average_price' or 'average_strike', got {asian_type}"
            )

    def evaluate(
        self, spots: np.ndarray, initial_spots: np.ndarray, time: float, state: PayoffState
    ) -> dict[str, Any]:
        """Evaluate Asian option.

        Accumulates average over all observations, pays off at maturity.
        """
        n_paths = spots.shape[0]

        # Calculate worst-of performance at this observation
        performances = spots / initial_spots
        worst_performance = np.min(performances, axis=1)

        # Get accumulated sum and count
        sum_performance = state.get("asian_sum", np.zeros(n_paths))
        count = state.get("asian_count", 0)

        # Update running average
        sum_performance = sum_performance + worst_performance
        count = count + 1
        avg_performance = sum_performance / count

        # Store updated values
        state.set("asian_sum", sum_performance)
        state.set("asian_count", count)
        state.set("asian_avg", avg_performance)

        # Check if this is the final observation
        is_final = state.get("is_final_observation", False)

        if not is_final:
            # Not at maturity - no cashflow yet
            return {
                "cashflow": np.zeros(n_paths),
                "terminated": np.zeros(n_paths, dtype=bool),
                "continue": True,
            }

        # At maturity - calculate payoff
        if self.asian_type == "average_price":
            # Payoff = max(phi * (S_avg - K), 0)
            intrinsic = np.maximum(self.phi * (avg_performance - self.strike), 0.0)
        else:  # average_strike
            # Payoff = max(phi * (S_T - S_avg), 0)
            intrinsic = np.maximum(self.phi * (worst_performance - avg_performance), 0.0)

        cashflow = intrinsic * self.participation

        return {
            "cashflow": cashflow,
            "terminated": np.zeros(n_paths, dtype=bool),
            "continue": True,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "asian_option",
            "strike": self.strike,
            "option_type": self.option_type,
            "asian_type": self.asian_type,
            "participation": self.participation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AsianOption":
        """Deserialize from dictionary."""
        return cls(
            strike=data["strike"],
            option_type=data["option_type"],
            asian_type=data.get("asian_type", "average_price"),
            participation=data.get("participation", 1.0),
        )


class BermudanOption(PayoffComponent):
    """Bermudan option - can exercise only at predefined dates.

    Exercise dates: E = {t_e1, t_e2, ..., t_em}
    Intrinsic(t) = max(phi * (S_t - K), 0) for t in E

    The engine must decide whether to exercise or continue at each exercise date.
    """

    def __init__(
        self,
        strike: float,
        exercise_times: list[float],
        option_type: str = "call",
        participation: float = 1.0,
    ):
        """Initialize Bermudan option.

        Args:
            strike: Strike level (as fraction of initial)
            exercise_times: List of times when exercise is allowed
            option_type: "call" or "put"
            participation: Participation rate
        """
        self.strike = strike
        self.exercise_times = sorted(exercise_times)
        self.option_type = option_type.lower()
        self.participation = participation
        self.phi = 1.0 if self.option_type == "call" else -1.0

        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    def evaluate(
        self, spots: np.ndarray, initial_spots: np.ndarray, time: float, state: PayoffState
    ) -> dict[str, Any]:
        """Evaluate Bermudan option.

        Provides intrinsic value at exercise dates.
        Engine must decide whether to exercise.

        Note: This basic implementation does not include optimal exercise logic.
        For full Bermudan option pricing, use regression-based methods.
        """
        n_paths = spots.shape[0]

        # Check if current time is an exercise date (with tolerance for floating point)
        is_exercise_date = any(abs(time - t) < 1e-6 for t in self.exercise_times)

        if not is_exercise_date:
            # Not an exercise date - no cashflow
            return {
                "cashflow": np.zeros(n_paths),
                "terminated": np.zeros(n_paths, dtype=bool),
                "continue": True,
            }

        # At an exercise date - calculate intrinsic value
        performances = spots / initial_spots
        worst_performance = np.min(performances, axis=1)

        # Intrinsic value
        intrinsic = np.maximum(self.phi * (worst_performance - self.strike), 0.0)

        # Store intrinsic value in state
        state.set("bermudan_option_intrinsic", intrinsic)

        # For now, don't exercise - just track intrinsic value
        # A full implementation would need continuation value estimation
        return {
            "cashflow": np.zeros(n_paths),
            "terminated": np.zeros(n_paths, dtype=bool),
            "continue": True,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "bermudan_option",
            "strike": self.strike,
            "exercise_times": self.exercise_times,
            "option_type": self.option_type,
            "participation": self.participation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BermudanOption":
        """Deserialize from dictionary."""
        return cls(
            strike=data["strike"],
            exercise_times=data["exercise_times"],
            option_type=data["option_type"],
            participation=data.get("participation", 1.0),
        )
