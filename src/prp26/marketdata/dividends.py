"""Dividend models for equity underlyings."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class DividendModel(ABC):
    """Abstract base class for dividend models."""

    @abstractmethod
    def yield_to_time(self, time: float) -> float:
        """Get equivalent continuous dividend yield to time T."""
        ...

    @abstractmethod
    def pv_dividends(self, spot: float, time: float, rate: float) -> float:
        """Present value of dividends paid up to time T."""
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "DividendModel":
        """Deserialize from dictionary."""
        ...


class ContinuousDividend(DividendModel):
    """Continuous dividend yield model (simple proportional yield).

    This is the standard Black-Scholes assumption: dS = (r - q) S dt + ...
    """

    def __init__(self, ticker: str, yield_rate: float):
        """Initialize continuous dividend model.

        Args:
            ticker: Stock ticker
            yield_rate: Continuous dividend yield (e.g., 0.02 = 2%)
        """
        self.ticker = ticker
        self.yield_rate = yield_rate

    def yield_to_time(self, time: float) -> float:
        """Return constant yield."""
        return self.yield_rate

    def pv_dividends(self, spot: float, time: float, rate: float) -> float:
        """PV of continuous dividends: S * (1 - exp(-q*T))."""
        return spot * (1.0 - np.exp(-self.yield_rate * time))

    def forward_factor(self, time: float) -> float:
        """Factor for forward price: F = S * exp(-q * T)."""
        return np.exp(-self.yield_rate * time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"type": "continuous", "ticker": self.ticker, "yield_rate": self.yield_rate}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContinuousDividend":
        """Deserialize from dictionary."""
        return cls(ticker=data["ticker"], yield_rate=data["yield_rate"])


class DiscreteDividend(DividendModel):
    """Discrete dividend payments (ex-dividend dates with amounts).

    More realistic for equity pricing - models actual dividend schedule.
    """

    def __init__(self, ticker: str, ex_dates: list[float], amounts: list[float], spot: float):
        """Initialize discrete dividend model.

        Args:
            ticker: Stock ticker
            ex_dates: Ex-dividend dates (in years from today)
            amounts: Dividend amounts at each ex-date
            spot: Current spot price (for yield calculation)
        """
        self.ticker = ticker
        self.ex_dates = np.array(ex_dates)
        self.amounts = np.array(amounts)
        self.spot = spot

        # Sort by ex-date
        sort_idx = np.argsort(self.ex_dates)
        self.ex_dates = self.ex_dates[sort_idx]
        self.amounts = self.amounts[sort_idx]

    def yield_to_time(self, time: float) -> float:
        """Calculate equivalent continuous yield to time T.

        This is approximate: q_eq = ln(S / (S - PV(divs))) / T
        """
        if time <= 0:
            return 0.0

        # Sum of dividends paid up to time T (undiscounted)
        divs_to_t = np.sum(self.amounts[self.ex_dates <= time])

        if divs_to_t == 0:
            return 0.0

        # Approximate equivalent yield
        # More accurate would discount dividends, but this is simple
        if divs_to_t >= self.spot:
            return 0.10  # Cap at 10% to avoid numerical issues

        return -np.log(1.0 - divs_to_t / self.spot) / time

    def pv_dividends(self, spot: float, time: float, rate: float) -> float:
        """Present value of dividends paid up to time T."""
        divs_to_t = self.amounts[self.ex_dates <= time]
        times_to_t = self.ex_dates[self.ex_dates <= time]

        if len(divs_to_t) == 0:
            return 0.0

        # Discount each dividend
        discount_factors = np.exp(-rate * times_to_t)
        return np.sum(divs_to_t * discount_factors)

    def forward_factor(self, time: float, rate: float) -> float:
        """Factor for forward price with discrete dividends.

        F(T) = (S - PV(divs)) * exp(r*T)
        So forward_factor = (S - PV(divs)) / S
        """
        pv_divs = self.pv_dividends(self.spot, time, rate)
        return (self.spot - pv_divs) / self.spot if self.spot > 0 else 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "discrete",
            "ticker": self.ticker,
            "ex_dates": self.ex_dates.tolist(),
            "amounts": self.amounts.tolist(),
            "spot": self.spot,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscreteDividend":
        """Deserialize from dictionary."""
        return cls(
            ticker=data["ticker"],
            ex_dates=data["ex_dates"],
            amounts=data["amounts"],
            spot=data["spot"],
        )


# Factory function for creating dividend models from dict
def dividend_model_from_dict(data: dict[str, Any]) -> DividendModel:
    """Factory to deserialize dividend models."""
    div_type = data.get("type", "continuous")

    if div_type == "continuous":
        return ContinuousDividend.from_dict(data)
    elif div_type == "discrete":
        return DiscreteDividend.from_dict(data)
    else:
        raise ValueError(f"Unknown dividend model type: {div_type}")


# Patch the from_dict to use factory
DividendModel.from_dict = staticmethod(dividend_model_from_dict)
