from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests

from prp26.marketdata.connectors import MarketDataConnector


class KrakenConnector(MarketDataConnector):
    """Kraken cryptocurrency exchange connector.

    Provides access to:
    - Real-time crypto spot prices
    - Historical OHLC data
    - No API key required for public endpoints

    Supports major cryptocurrencies: BTC, ETH, SOL, etc.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize Kraken connector.

        Args:
            config: Optional config dict with API credentials (not required for public data)
        """
        super().__init__(config)
        self.base_url = "https://api.kraken.com/0/public"
        self.api_key = config.get("api_key") if config else None
        self.api_secret = config.get("api_secret") if config else None

    def connect(self) -> bool:
        """Test connection to Kraken API.

        Returns:
            True if successful
        """

        try:
            # Test with server time endpoint
            response = requests.get(f"{self.base_url}/Time", timeout=5)
            response.raise_for_status()
            self.connected = True
            return True
        except Exception as e:
            print(f"Kraken connection failed: {e}")
            return False

    def _normalize_pair(self, ticker: str) -> str:
        """Normalize crypto pair to Kraken format.

        Args:
            ticker: Crypto symbol (e.g., 'BTC/USD', 'ETH/USD')

        Returns:
            Kraken pair format (e.g., 'XXBTZUSD', 'XETHZUSD')
        """
        # Handle common formats
        ticker = ticker.upper().replace("-", "/")

        # Map common symbols to Kraken format
        symbol_map = {
            "BTC/USD": "XXBTZUSD",
            "ETH/USD": "XETHZUSD",
            "SOL/USD": "SOLUSD",
            "XRP/USD": "XXRPZUSD",
            "ADA/USD": "ADAUSD",
            "DOT/USD": "DOTUSD",
            "MATIC/USD": "MATICUSD",
            "BTC/EUR": "XXBTZEUR",
            "ETH/EUR": "XETHZEUR",
        }

        return symbol_map.get(ticker, ticker.replace("/", ""))

    def get_spot_price(self, ticker: str) -> float:
        """Get current spot price from Kraken.

        Args:
            ticker: Crypto pair (e.g., 'BTC/USD', 'ETH/USD')

        Returns:
            Current spot price
        """
        if not self.connected:
            self.connect()

        import requests

        pair = self._normalize_pair(ticker)

        url = f"{self.base_url}/Ticker"
        params = {"pair": pair}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            raise ValueError(f"Kraken API error: {data['error']}")

        # Get the result for this pair
        result = data["result"]
        # Kraken returns different key formats, get first result
        pair_data = list(result.values())[0]

        # Use last trade price
        price = float(pair_data["c"][0])  # 'c' is current/close price

        return price

    def get_spot_prices(self, tickers: list[str]) -> dict[str, float]:
        """Get spot prices for multiple crypto pairs.

        Args:
            tickers: List of crypto pairs

        Returns:
            Dict mapping ticker to spot price
        """
        if not self.connected:
            self.connect()

        import requests

        # Kraken supports batch requests
        pairs = [self._normalize_pair(t) for t in tickers]

        url = f"{self.base_url}/Ticker"
        params = {"pair": ",".join(pairs)}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("error") and len(data["error"]) > 0:
                # Fall back to individual requests
                return {ticker: self.get_spot_price(ticker) for ticker in tickers}

            result = data["result"]
            prices = {}

            for ticker, pair in zip(tickers, pairs):  # noqa: B905
                if pair in result:
                    prices[ticker] = float(result[pair]["c"][0])
                else:
                    # Try to find by original ticker
                    for key, value in result.items():
                        if ticker.replace("/", "").upper() in key.upper():
                            prices[ticker] = float(value["c"][0])
                            break

            return prices

        except Exception as e:
            print(f"Batch request failed: {e}, falling back to individual requests")
            return {ticker: self.get_spot_price(ticker) for ticker in tickers}

    def get_historical_prices(
        self, ticker: str, start_date: datetime, end_date: datetime, frequency: str = "1d"
    ) -> pd.DataFrame:
        """Get historical OHLC data from Kraken.

        Args:
            ticker: Crypto pair
            start_date: Start date
            end_date: End date
            frequency: Interval (1, 5, 15, 30, 60, 240, 1440 minutes)

        Returns:
            DataFrame with OHLCV data
        """
        if not self.connected:
            self.connect()

        import requests

        pair = self._normalize_pair(ticker)

        # Map frequency to Kraken interval (in minutes)
        interval_map = {
            "1d": 1440,
            "1h": 60,
            "15m": 15,
            "5m": 5,
        }
        interval = interval_map.get(frequency, 1440)

        url = f"{self.base_url}/OHLC"
        params = {
            "pair": pair,
            "interval": interval,
            "since": int(start_date.timestamp()),
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("error") and len(data["error"]) > 0:
            raise ValueError(f"Kraken API error: {data['error']}")

        # Parse OHLC data
        result = data["result"]
        ohlc_data = list(result.values())[0]  # Get first result

        # Convert to DataFrame
        df = pd.DataFrame(
            ohlc_data,
            columns=["Time", "Open", "High", "Low", "Close", "VWAP", "Volume", "Count"],
        )

        # Convert time to datetime
        df["Time"] = pd.to_datetime(df["Time"], unit="s")
        df.set_index("Time", inplace=True)

        # Convert to float
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)

        # Filter by end date
        df = df[df.index <= end_date]

        return df[["Open", "High", "Low", "Close", "Volume"]]

    def get_implied_volatility(self, ticker: str, maturity: float = 1.0) -> float:
        """Estimate volatility from historical data.

        Args:
            ticker: Crypto pair
            maturity: Time horizon (not used for crypto)

        Returns:
            Annualized volatility
        """
        if not self.connected:
            self.connect()

        # Get last 90 days of data
        from datetime import timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

        hist = self.get_historical_prices(ticker, start_date, end_date, frequency="1d")

        # Calculate returns
        returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()

        # Annualized volatility (crypto trades 24/7, so use 365 days)
        vol = returns.std() * np.sqrt(365)

        return float(vol)
