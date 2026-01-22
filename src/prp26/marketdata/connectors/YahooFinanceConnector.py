from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from prp26.marketdata.connectors import MarketDataConnector


class YahooFinanceConnector(MarketDataConnector):
    """Yahoo Finance connector using yfinance library.

    Provides access to:
    - Real-time spot prices
    - Historical data
    - Dividend yields
    - Approximate implied volatility from historical data

    No API key required.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize Yahoo Finance connector.

        Args:
            config: Optional config dict (not required for Yahoo)
        """
        super().__init__(config)
        try:
            import yfinance as yf

            self.yf = yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

    def connect(self) -> bool:
        """Establish connection (no-op for Yahoo Finance).

        Returns:
            True
        """
        self.connected = True
        return True

    def get_spot_price(self, ticker: str) -> float:
        """Get current spot price from Yahoo Finance.

        Args:
            ticker: Yahoo Finance ticker symbol (e.g., 'AAPL', '^GSPC')

        Returns:
            Current spot price
        """
        if not self.connected:
            self.connect()

        stock = self.yf.Ticker(ticker)

        # Try to get real-time price
        try:
            info = stock.info
            # Try multiple fields (Yahoo Finance API varies)
            for field in ["regularMarketPrice", "currentPrice", "price", "previousClose"]:
                if field in info and info[field] is not None:
                    return float(info[field])
        except:
            pass

        # Fall back to latest close
        hist = stock.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])

        raise ValueError(f"Could not retrieve spot price for {ticker}")

    def get_spot_prices(self, tickers: list[str]) -> dict[str, float]:
        """Get spot prices for multiple tickers.

        Args:
            tickers: List of Yahoo Finance symbols

        Returns:
            Dict mapping ticker to spot price
        """
        if not self.connected:
            self.connect()

        prices = {}
        for ticker in tickers:
            try:
                prices[ticker] = self.get_spot_price(ticker)
            except Exception as e:
                print(f"Warning: Could not get price for {ticker}: {e}")
                prices[ticker] = None

        return prices

    def get_historical_prices(
        self, ticker: str, start_date: datetime, end_date: datetime, frequency: str = "1d"
    ) -> pd.DataFrame:
        """Get historical OHLCV data from Yahoo Finance.

        Args:
            ticker: Yahoo Finance symbol
            start_date: Start date
            end_date: End date
            frequency: Data frequency ('1d', '1h', etc.)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        if not self.connected:
            self.connect()

        stock = self.yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date, interval=frequency)

        return hist

    def get_implied_volatility(self, ticker: str, maturity: float = 1.0) -> float:
        """Estimate implied volatility from historical data.

        Args:
            ticker: Yahoo Finance symbol
            maturity: Time horizon in years (for scaling)

        Returns:
            Annualized volatility estimate
        """
        if not self.connected:
            self.connect()

        # Get historical data (last 1 year for calibration)
        from datetime import timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        hist = self.get_historical_prices(ticker, start_date, end_date, frequency="1d")

        # Calculate returns
        returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()

        # Annualized volatility
        vol = returns.std() * np.sqrt(252)  # 252 trading days

        return float(vol)

    def get_dividend_yield(self, ticker: str) -> float:
        """Get dividend yield.

        Args:
            ticker: Yahoo Finance symbol

        Returns:
            Annual dividend yield (as decimal)
        """
        if not self.connected:
            self.connect()

        stock = self.yf.Ticker(ticker)
        info = stock.info

        # Try multiple fields
        for field in ["dividendYield", "trailingAnnualDividendYield"]:
            if field in info and info[field] is not None:
                return float(info[field])

        return 0.0  # Default to 0 if not available
