"""
Rainbow payoff components for multi-asset products.

Rainbow options depend on the relative performance of multiple assets:
- Best-of: Payoff based on best performing asset
- Worst-of: Payoff based on worst performing asset
- N-th best: Payoff based on N-th ranked asset
- Spread: Payoff based on difference between assets
"""

from typing import Any, Literal

import numpy as np

from .base import PayoffComponent, PayoffState


class RainbowComponent(PayoffComponent):
    """Base class for rainbow (multi-asset ranking) payoffs."""

    def __init__(
        self,
        selection: Literal["best", "worst", "median", "nth"] = "worst",
        rank: int = 1,
        strike: float = 1.0,
        participation: float = 1.0,
    ):
        """Initialize rainbow component.

        Args:
            selection: Which asset to select ("best", "worst", "median", "nth")
            rank: For "nth" selection, which rank (1=best, 2=second best, etc.)
            strike: Strike level
            participation: Participation rate
        """
        self.selection = selection
        self.rank = rank
        self.strike = strike
        self.participation = participation

    def _select_performance(self, performances: np.ndarray) -> np.ndarray:
        """Select performance based on ranking.

        Args:
            performances: Performance array (n_paths, n_assets)

        Returns:
            Selected performance (n_paths,)
        """
        if self.selection == "best":
            return np.max(performances, axis=1)
        elif self.selection == "worst":
            return np.min(performances, axis=1)
        elif self.selection == "median":
            return np.median(performances, axis=1)
        elif self.selection == "nth":
            # Sort in descending order and take the nth
            sorted_perfs = np.sort(performances, axis=1)[:, ::-1]
            rank_idx = min(self.rank - 1, sorted_perfs.shape[1] - 1)
            return sorted_perfs[:, rank_idx]
        else:
            raise ValueError(f"Unknown selection: {self.selection}")

    def evaluate(
        self, spots: np.ndarray, initial_spots: np.ndarray, time: float, state: PayoffState
    ) -> dict[str, Any]:
        """Evaluate rainbow component - to be implemented by subclasses."""
        raise NotImplementedError


class RainbowCall(RainbowComponent):
    """Rainbow call option (e.g., call on best-of basket)."""

    def evaluate(
        self, spots: np.ndarray, initial_spots: np.ndarray, time: float, state: PayoffState
    ) -> dict[str, Any]:
        """Evaluate rainbow call."""
        n_paths = spots.shape[0]

        # Calculate performances
        performances = spots / initial_spots

        # Select based on ranking
        selected_performance = self._select_performance(performances)

        # Call payoff: max(selected - strike, 0) * participation
        payoff = np.maximum(selected_performance - self.strike, 0) * self.participation

        return {"cashflow": payoff, "terminated": np.zeros(n_paths, dtype=bool), "continue": True}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "rainbow_call",
            "selection": self.selection,
            "rank": self.rank,
            "strike": self.strike,
            "participation": self.participation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RainbowCall":
        """Deserialize from dictionary."""
        return cls(
            selection=data.get("selection", "worst"),
            rank=data.get("rank", 1),
            strike=data.get("strike", 1.0),
            participation=data.get("participation", 1.0),
        )


class RainbowPut(RainbowComponent):
    """Rainbow put option (e.g., put on worst-of basket)."""

    def evaluate(
        self, spots: np.ndarray, initial_spots: np.ndarray, time: float, state: PayoffState
    ) -> dict[str, Any]:
        """Evaluate rainbow put."""
        n_paths = spots.shape[0]

        # Calculate performances
        performances = spots / initial_spots

        # Select based on ranking
        selected_performance = self._select_performance(performances)

        # Put payoff: max(strike - selected, 0) * participation
        payoff = np.maximum(self.strike - selected_performance, 0) * self.participation

        return {"cashflow": payoff, "terminated": np.zeros(n_paths, dtype=bool), "continue": True}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "rainbow_put",
            "selection": self.selection,
            "rank": self.rank,
            "strike": self.strike,
            "participation": self.participation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RainbowPut":
        """Deserialize from dictionary."""
        return cls(
            selection=data.get("selection", "worst"),
            rank=data.get("rank", 1),
            strike=data.get("strike", 1.0),
            participation=data.get("participation", 1.0),
        )


class RainbowDigital(PayoffComponent):
    """Digital payoff based on rainbow selection.

    Pays fixed amount if selected asset is above/below barrier.
    """

    def __init__(
        self,
        selection: Literal["best", "worst", "median", "nth"] = "worst",
        rank: int = 1,
        barrier: float = 1.0,
        payout: float = 1.0,
        trigger_type: Literal["above", "below"] = "above",
    ):
        """Initialize rainbow digital.

        Args:
            selection: Which asset to select
            rank: For "nth" selection, which rank
            barrier: Barrier level
            payout: Fixed payout if triggered
            trigger_type: "above" or "below" barrier
        """
        self.selection = selection
        self.rank = rank
        self.barrier = barrier
        self.payout = payout
        self.trigger_type = trigger_type

    def _select_performance(self, performances: np.ndarray) -> np.ndarray:
        """Select performance based on ranking."""
        if self.selection == "best":
            return np.max(performances, axis=1)
        elif self.selection == "worst":
            return np.min(performances, axis=1)
        elif self.selection == "median":
            return np.median(performances, axis=1)
        elif self.selection == "nth":
            sorted_perfs = np.sort(performances, axis=1)[:, ::-1]
            rank_idx = min(self.rank - 1, sorted_perfs.shape[1] - 1)
            return sorted_perfs[:, rank_idx]
        else:
            raise ValueError(f"Unknown selection: {self.selection}")

    def evaluate(
        self, spots: np.ndarray, initial_spots: np.ndarray, time: float, state: PayoffState
    ) -> dict[str, Any]:
        """Evaluate rainbow digital."""
        n_paths = spots.shape[0]

        # Calculate performances
        performances = spots / initial_spots

        # Select based on ranking
        selected_performance = self._select_performance(performances)

        # Digital payoff
        if self.trigger_type == "above":
            triggered = selected_performance >= self.barrier
        else:
            triggered = selected_performance <= self.barrier

        payoff = np.where(triggered, self.payout, 0.0)

        return {"cashflow": payoff, "terminated": np.zeros(n_paths, dtype=bool), "continue": True}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "rainbow_digital",
            "selection": self.selection,
            "rank": self.rank,
            "barrier": self.barrier,
            "payout": self.payout,
            "trigger_type": self.trigger_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RainbowDigital":
        """Deserialize from dictionary."""
        return cls(
            selection=data.get("selection", "worst"),
            rank=data.get("rank", 1),
            barrier=data.get("barrier", 1.0),
            payout=data.get("payout", 1.0),
            trigger_type=data.get("trigger_type", "above"),
        )


class SpreadOption(PayoffComponent):
    """Spread option between two assets.

    Payoff = max(asset1 - asset2 - strike, 0) * participation
    """

    def __init__(
        self,
        asset1_index: int = 0,
        asset2_index: int = 1,
        strike: float = 0.0,
        participation: float = 1.0,
        option_type: Literal["call", "put"] = "call",
    ):
        """Initialize spread option.

        Args:
            asset1_index: Index of first asset in basket
            asset2_index: Index of second asset in basket
            strike: Strike on the spread
            participation: Participation rate
            option_type: "call" or "put"
        """
        self.asset1_index = asset1_index
        self.asset2_index = asset2_index
        self.strike = strike
        self.participation = participation
        self.option_type = option_type

    def evaluate(
        self, spots: np.ndarray, initial_spots: np.ndarray, time: float, state: PayoffState
    ) -> dict[str, Any]:
        """Evaluate spread option."""
        n_paths = spots.shape[0]

        # Calculate performances
        perf1 = spots[:, self.asset1_index] / initial_spots[self.asset1_index]
        perf2 = spots[:, self.asset2_index] / initial_spots[self.asset2_index]

        # Spread
        spread = perf1 - perf2

        # Option payoff
        if self.option_type == "call":
            payoff = np.maximum(spread - self.strike, 0) * self.participation
        else:
            payoff = np.maximum(self.strike - spread, 0) * self.participation

        return {"cashflow": payoff, "terminated": np.zeros(n_paths, dtype=bool), "continue": True}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "spread_option",
            "asset1_index": self.asset1_index,
            "asset2_index": self.asset2_index,
            "strike": self.strike,
            "participation": self.participation,
            "option_type": self.option_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpreadOption":
        """Deserialize from dictionary."""
        return cls(
            asset1_index=data.get("asset1_index", 0),
            asset2_index=data.get("asset2_index", 1),
            strike=data.get("strike", 0.0),
            participation=data.get("participation", 1.0),
            option_type=data.get("option_type", "call"),
        )
