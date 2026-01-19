# Market Data Connectors

Real-time market data connectors for products pricing.

## Overview

The market data connector framework provides a unified interface for fetching real-time and historical market data from multiple providers. All connectors implement the `MarketDataConnector` abstract base class, ensuring consistent API across different data sources.

## Supported Connectors

### 1. Yahoo Finance (`YahooFinanceConnector`)

**Data Source**: Yahoo Finance via `yfinance` library  
**API Key**: Not required  
**Asset Classes**: Equities, ETFs, Indices  
**Supported Operations**:
- Real-time spot prices
- Historical OHLCV data
- Dividend yields
- Implied volatility (estimated from historical data)

**Example Usage**:
```python
from prp26.marketdata import YahooFinanceConnector

# Initialize connector
connector = YahooFinanceConnector()
connector.connect()

# Get current spot price
spot = connector.get_spot_price("AAPL")
print(f"AAPL: ${spot:.2f}")

# Get multiple spots
spots = connector.get_spot_prices(["AAPL", "MSFT", "GOOGL"])

# Get historical data
from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
hist = connector.get_historical_prices("AAPL", start_date, end_date)

# Get implied volatility
vol = connector.get_implied_volatility("AAPL", maturity=1.0)
print(f"AAPL 1Y Vol: {vol*100:.1f}%")

# Get dividend yield
div_yield = connector.get_dividend_yield("AAPL")
print(f"AAPL Dividend Yield: {div_yield*100:.2f}%")
```

### 2. Kraken (`KrakenConnector`)

**Data Source**: Kraken cryptocurrency exchange API  
**API Key**: Not required for public endpoints  
**Asset Classes**: Cryptocurrencies (BTC, ETH, SOL, etc.)  
**Supported Operations**:
- Real-time crypto spot prices
- Historical OHLC data
- Implied volatility (estimated from historical data)

**Example Usage**:
```python
from prp26.marketdata import KrakenConnector

# Initialize connector
connector = KrakenConnector()
connector.connect()

# Get crypto spot price
btc_price = connector.get_spot_price("BTC/USD")
print(f"BTC: ${btc_price:,.2f}")

# Get multiple crypto prices
prices = connector.get_spot_prices(["BTC/USD", "ETH/USD", "SOL/USD"])

# Get historical data
from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=90)
hist = connector.get_historical_prices("BTC/USD", start_date, end_date)

# Get implied volatility
vol = connector.get_implied_volatility("BTC/USD", maturity=1.0)
print(f"BTC 1Y Vol: {vol*100:.1f}%")
```

**Supported Crypto Pairs**:
- BTC/USD, ETH/USD, SOL/USD
- BTC/EUR, ETH/EUR
- XRP/USD, ADA/USD, DOT/USD, MATIC/USD
- See Kraken API documentation for full list

## Connector Interface

All connectors implement the following methods:

### Core Methods

```python
class MarketDataConnector(ABC):
    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize connector with optional configuration."""
        
    def connect(self) -> bool:
        """Establish connection to data provider."""
        
    def get_spot_price(self, ticker: str) -> float:
        """Get current spot price for ticker."""
        
    def get_spot_prices(self, tickers: list[str]) -> dict[str, float]:
        """Get spot prices for multiple tickers."""
        
    def get_historical_prices(
        self, ticker: str, start_date: datetime, end_date: datetime, frequency: str = "1d"
    ) -> pd.DataFrame:
        """Get historical OHLCV data."""
        
    def get_implied_volatility(self, ticker: str, maturity: float) -> float:
        """Get implied volatility (ATM)."""
        
    def disconnect(self):
        """Close connection to data provider."""
```

## Integration with Pricing Engine

### Building a MarketDataSnapshot

The connectors are designed to easily populate `MarketDataSnapshot` objects for pricing:

```python
from datetime import datetime
from prp26.marketdata import YahooFinanceConnector
from prp26.marketdata.base import MarketDataSnapshot
from prp26.marketdata.rates import RateCurve
from prp26.marketdata.dividends import ContinuousDividend

# Connect to data source
connector = YahooFinanceConnector()
connector.connect()

# Fetch market data
tickers = ["AAPL", "MSFT", "GOOGL"]
spots = connector.get_spot_prices(tickers)
div_yields = {t: connector.get_dividend_yield(t) for t in tickers}
vols = {t: connector.get_implied_volatility(t, 1.0) for t in tickers}

# Create rate curve
usd_curve = RateCurve(currency="USD", rates=[0.045], tenors=[30.0])

# Create dividend models
dividends = {
    t: ContinuousDividend(ticker=t, yield_rate=div_yields[t])
    for t in tickers
}

# Build market data snapshot
market_data = MarketDataSnapshot(
    valuation_date=datetime.now(),
    spots=spots,
    rates={"USD": usd_curve},
    dividends=dividends,
    metadata={"source": "YahooFinance"}
)

# Use in pricing engine
from prp26.engine import PricingEngine

engine = PricingEngine(
    product=my_product,
    market_data=market_data,
    model_bundle=my_models,
)
result = engine.price()
```

## Quanto Support

Quanto products allow payoffs to be denominated in a different currency than the underlyings. This is particularly useful for crypto products denominated in USD.

### Example: USD-denominated Crypto Autocallable

```python
from prp26.products.autocallable import PhoenixAutocallable

# Create product with quanto feature
product = PhoenixAutocallable(
    product_id="CRYPTO_QUANTO_001",
    currency="USD",
    notional=1_000_000.0,
    basket=crypto_basket,
    coupon_rate=0.10,
    autocall_barrier=0.75,
    coupon_barrier=0.65,
    put_barrier=0.50,
    observation_dates=observation_dates,
    quanto_currency="USD",  # Payoff in USD, underlyings are crypto
)

# Check if product is quanto
if product.is_quanto():
    print("This product has quanto feature enabled")

# Add FX rates to market data
market_data = MarketDataSnapshot(
    valuation_date=datetime.now(),
    spots=crypto_spots,
    rates={"USD": usd_curve},
    dividends=dividends,
    fx_rates={"USD/USD": 1.0},  # Identity for USD-denominated crypto
)
```

**Benefits of Quanto**:
- Eliminates FX risk for investors in the payoff currency
- Direct exposure to underlying performance without currency hedging
- Common for cross-border products
- Particularly useful for crypto products targeting fiat currency investors

## Demo Examples

### 1. Equity Autocallable (Yahoo Finance)

See [`examples/yahoo_finance_autocallable.py`](../examples/yahoo_finance_autocallable.py)

Demonstrates:
- Real-time pricing of Phoenix autocallable on US tech stocks (AAPL, MSFT, GOOGL)
- Fetching spots, dividends, and volatilities from Yahoo Finance
- Monte Carlo pricing with live market data
- Quarterly observations over 1 year

Run:
```bash
cd snippets/python/prp26
python examples/yahoo_finance_autocallable.py
```

### 2. Crypto Autocallable with Quanto (Kraken)

See [`examples/kraken_crypto_quanto_autocallable.py`](../examples/kraken_crypto_quanto_autocallable.py)

Demonstrates:
- Real-time crypto pricing (BTC, ETH, SOL)
- USD-denominated autocallable (quanto feature)
- Higher volatility parameters for crypto
- Monthly observations over 6 months
- Historical data fetching

Run:
```bash
cd snippets/python/prp26
python examples/kraken_crypto_quanto_autocallable.py
```

## Configuration

### Yahoo Finance

No configuration required. The `yfinance` library uses Yahoo Finance's free public API.

### Kraken

Public endpoints do not require API credentials. For private endpoints (trading, account data), configure with:

```python
config = {
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
}
connector = KrakenConnector(config=config)
```

## Error Handling

All connectors implement robust error handling:

```python
try:
    spot = connector.get_spot_price("INVALID_TICKER")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

Common errors:
- `ValueError`: Invalid ticker or missing data
- `ConnectionError`: Network issues
- `TimeoutError`: API request timeout

## Rate Limits

### Yahoo Finance
- No official rate limits for `yfinance`
- Recommended: < 2000 requests/hour per IP
- Add delays between batch requests if needed

### Kraken
- Public API: 1 request/second (soft limit)
- Batch requests are supported for multiple tickers
- No authentication required for public data

## Best Practices

1. **Connection Management**
   ```python
   connector = YahooFinanceConnector()
   connector.connect()
   try:
       # Use connector
       spots = connector.get_spot_prices(tickers)
   finally:
       connector.disconnect()
   ```

2. **Batch Requests**
   ```python
   # Efficient: Single batch request
   spots = connector.get_spot_prices(["AAPL", "MSFT", "GOOGL"])
   
   # Inefficient: Multiple individual requests
   spots = {t: connector.get_spot_price(t) for t in tickers}
   ```

3. **Caching**
   ```python
   from datetime import datetime, timedelta
   
   # Cache market data snapshots
   cache = {}
   cache_key = datetime.now().strftime("%Y-%m-%d")
   
   if cache_key not in cache:
       spots = connector.get_spot_prices(tickers)
       cache[cache_key] = spots
   else:
       spots = cache[cache_key]
   ```

4. **Fallback Values**
   ```python
   try:
       vol = connector.get_implied_volatility("AAPL", 1.0)
   except Exception:
       vol = 0.25  # Fallback to 25% default
   ```

## Future Enhancements

Planned connector additions:
- Bloomberg API (DAPI/SAPI)
- Refinitiv (Eikon/Workspace)
- Interactive Brokers
- Coinbase (crypto)
- CME DataMine (derivatives)

Feature roadmap:
- Options data (implied vol surfaces)
- Futures curves
- Credit spreads
- Real-time streaming (WebSocket)
- Advanced caching layer

## Dependencies

```toml
dependencies = [
    "yfinance>=0.2.0",
    "requests>=2.31.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
]
```

Install with:
```bash
pip install yfinance requests
```

Or using uv:
```bash
uv pip install yfinance requests
```

## Troubleshooting

### Yahoo Finance Connection Issues

```python
# Test connection
connector = YahooFinanceConnector()
try:
    spot = connector.get_spot_price("AAPL")
    print(f"Connection OK: AAPL = ${spot}")
except Exception as e:
    print(f"Connection failed: {e}")
```

**Common issues**:
- Yahoo Finance API changes (update `yfinance` library)
- Network/proxy issues
- Invalid ticker symbols

### Kraken Connection Issues

```python
# Test Kraken connection
connector = KrakenConnector()
if connector.connect():
    print("Kraken connection OK")
else:
    print("Kraken connection failed")
```

**Common issues**:
- Rate limiting (add delays between requests)
- Invalid pair format (use BTC/USD not BTCUSD)
- Network/firewall blocking API requests

## License

MIT License - See [LICENSE](../LICENSE) for details.
