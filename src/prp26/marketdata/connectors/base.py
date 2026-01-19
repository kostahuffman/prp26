"""
Market data connectors for real-time pricing.

Provides interfaces to:
- Yahoo Finance (equities, indices, ETFs)
- Kraken (cryptocurrency spot and options)
- Extensible for other data providers

All connectors implement a common interface for authentication and data retrieval.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd


class MarketDataConnector(ABC):
    """Abstract base class for market data connectors.

    All connectors must implement:
    - Authentication/connection
    - Spot price retrieval
    - Volatility surface retrieval (if available)
    - Historical data for calibration
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize connector with optional configuration.

        Args:
            config: Configuration dict (API keys, endpoints, etc.)
        """
        self.config = config or {}
        self.connected = False

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to data provider.

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def get_spot_price(self, ticker: str) -> float:
        """Get current spot price for ticker.

        Args:
            ticker: Symbol/ticker to query

        Returns:
            Current spot price
        """
        pass

    @abstractmethod
    def get_spot_prices(self, tickers: list[str]) -> dict[str, float]:
        """Get spot prices for multiple tickers.

        Args:
            tickers: List of symbols

        Returns:
            Dict mapping ticker to spot price
        """
        pass

    @abstractmethod
    def get_historical_prices(
        self, ticker: str, start_date: datetime, end_date: datetime, frequency: str = "1d"
    ) -> pd.DataFrame:
        """Get historical price data.

        Args:
            ticker: Symbol to query
            start_date: Start date
            end_date: End date
            frequency: Data frequency (1d, 1h, etc.)

        Returns:
            DataFrame with OHLCV data
        """
        pass

    @abstractmethod
    def get_implied_volatility(self, ticker: str, maturity: float) -> float:
        """Get implied volatility (ATM).

        Args:
            ticker: Symbol
            maturity: Time to maturity in years

        Returns:
            Implied volatility (annualized)
        """
        pass

    def disconnect(self):
        """Close connection to data provider."""
        self.connected = False
