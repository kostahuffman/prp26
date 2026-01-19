"""Base classes and protocols for market data."""

from datetime import datetime
from typing import Any, Protocol

import numpy as np


class MarketData(Protocol):
    """Protocol for market data providers."""

    def get_spot(self, ticker: str) -> float:
        """Get current spot price for a ticker."""
        ...

    def get_forward(self, ticker: str, time: float) -> float:
        """Get forward price at time T."""
        ...

    def get_discount_factor(self, currency: str, time: float) -> float:
        """Get discount factor for currency at time T."""
        ...

    def get_dividend_yield(self, ticker: str, time: float) -> float:
        """Get dividend yield for ticker at time T."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarketData":
        """Deserialize from dictionary."""
        ...


class MarketDataSnapshot:
    """Point-in-time snapshot of market data for pricing and repricing.

    This is the institutional "market data as of" pattern - captures all
    market inputs at a specific valuation date for reproducibility.

    Attributes:
        valuation_date: The as-of date for this snapshot
        spots: Dict[ticker, spot_price]
        rates: Dict[currency, RateCurve]
        dividends: Dict[ticker, DividendModel]
        implied_vols: Optional dict of vol surfaces
        correlations: Optional correlation matrix
        fx_rates: Optional dict of FX rates for quanto products (e.g., {"EUR/USD": 1.10})
        metadata: Additional information (source, time, version)
    """

    def __init__(
        self,
        valuation_date: datetime,
        spots: dict[str, float],
        rates: dict[str, Any],  # RateCurve objects
        dividends: dict[str, Any],  # DividendModel objects
        implied_vols: dict[str, Any] = None,
        correlations: np.ndarray = None,
        fx_rates: dict[str, float] = None,
        metadata: dict[str, Any] = None,
    ):
        self.valuation_date = valuation_date
        self.spots = spots
        self.rates = rates
        self.dividends = dividends
        self.implied_vols = implied_vols or {}
        self.correlations = correlations
        self.fx_rates = fx_rates or {}  # FX rates for quanto products
        self.metadata = metadata or {}

    def get_spot(self, ticker: str) -> float:
        """Get spot price for ticker."""
        if ticker not in self.spots:
            raise ValueError(f"Ticker {ticker} not found in market data snapshot")
        return self.spots[ticker]

    def get_forward(self, ticker: str, time: float) -> float:
        """Calculate forward price: F(T) = S * exp((r - q) * T)."""
        spot = self.get_spot(ticker)

        # Get rate and dividend yield (assume continuous for simplicity)
        # In production, would need to handle currency mapping
        rate = self.get_discount_rate(time)
        div_yield = self.get_dividend_yield(ticker, time)

        return spot * np.exp((rate - div_yield) * time)

    def get_discount_factor(self, currency: str, time: float) -> float:
        """Get discount factor: DF(T) = exp(-r * T)."""
        if currency not in self.rates:
            raise ValueError(f"Currency {currency} not found in rate curves")

        rate_curve = self.rates[currency]
        return rate_curve.discount_factor(time)

    def get_discount_rate(self, time: float, currency: str = "USD") -> float:
        """Get continuously compounded rate for time T."""
        df = self.get_discount_factor(currency, time)
        return -np.log(df) / time if time > 0 else 0.0

    def get_dividend_yield(self, ticker: str, time: float) -> float:
        """Get dividend yield (continuous equivalent) for ticker."""
        if ticker not in self.dividends:
            return 0.0

        div_model = self.dividends[ticker]
        return div_model.yield_to_time(time)

    def get_fx_rate(self, currency_pair: str) -> float:
        """Get FX rate for quanto products.
        
        Args:
            currency_pair: Format "CCY1/CCY2" (e.g., "EUR/USD")
            
        Returns:
            FX rate (how much CCY2 per unit of CCY1)
        """
        if currency_pair in self.fx_rates:
            return self.fx_rates[currency_pair]
        
        # Try inverse
        ccy1, ccy2 = currency_pair.split("/")
        inverse_pair = f"{ccy2}/{ccy1}"
        if inverse_pair in self.fx_rates:
            return 1.0 / self.fx_rates[inverse_pair]
        
        # If same currency, rate is 1
        if ccy1 == ccy2:
            return 1.0
        
        raise ValueError(f"FX rate for {currency_pair} not found in market data")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "valuation_date": self.valuation_date.isoformat(),
            "spots": self.spots,
            "rates": {ccy: curve.to_dict() for ccy, curve in self.rates.items()},
            "dividends": {ticker: div.to_dict() for ticker, div in self.dividends.items()},
            "implied_vols": self.implied_vols,
            "correlations": self.correlations.tolist() if self.correlations is not None else None,
            "fx_rates": self.fx_rates,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarketDataSnapshot":
        """Deserialize from dictionary."""
        from datetime import datetime

        from .dividends import DividendModel
        from .rates import RateCurve

        valuation_date = datetime.fromisoformat(data["valuation_date"])

        # Reconstruct rate curves
        rates = {ccy: RateCurve.from_dict(curve_data) for ccy, curve_data in data["rates"].items()}

        # Reconstruct dividend models
        dividends = {
            ticker: DividendModel.from_dict(div_data)
            for ticker, div_data in data["dividends"].items()
        }

        # Reconstruct correlation matrix
        corr = np.array(data["correlations"]) if data.get("correlations") else None

        return cls(
            valuation_date=valuation_date,
            spots=data["spots"],
            rates=rates,
            dividends=dividends,
            implied_vols=data.get("implied_vols", {}),
            correlations=corr,
            fx_rates=data.get("fx_rates", {}),
            metadata=data.get("metadata", {}),
        )
