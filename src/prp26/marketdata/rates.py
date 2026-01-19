"""Interest rate curves for discounting."""

from typing import Dict, Any, Optional
import numpy as np
from scipy.interpolate import interp1d


class RateCurve:
    """Interest rate curve for discounting cash flows.
    
    Supports:
    - Flat rate (single continuously compounded rate)
    - Piecewise linear interpolation on rates
    - Piecewise linear interpolation on discount factors
    
    Institutional curves would use spline interpolation with proper
    extrapolation rules, day count conventions, etc.
    """
    
    def __init__(
        self,
        currency: str,
        times: np.ndarray = None,
        rates: np.ndarray = None,
        discount_factors: np.ndarray = None,
        interpolation: str = "rate_linear"
    ):
        """Initialize rate curve.
        
        Args:
            currency: Currency code (e.g., "USD", "EUR")
            times: Array of times (in years) for curve points
            rates: Continuously compounded rates at each time
            discount_factors: Discount factors at each time (alternative to rates)
            interpolation: "rate_linear" or "df_linear"
        """
        self.currency = currency
        self.interpolation = interpolation
        
        if times is None:
            # Flat rate curve at 0%
            self.times = np.array([0.0, 30.0])
            self.rates = np.array([0.0, 0.0])
            self.discount_factors = np.ones(2)
        elif rates is not None:
            self.times = np.asarray(times)
            self.rates = np.asarray(rates)
            self.discount_factors = np.exp(-self.rates * self.times)
        elif discount_factors is not None:
            self.times = np.asarray(times)
            self.discount_factors = np.asarray(discount_factors)
            # Back out rates: r = -ln(DF) / T
            self.rates = -np.log(self.discount_factors) / self.times
            self.rates[0] = self.rates[1]  # Handle t=0
        else:
            raise ValueError("Must provide either rates or discount_factors")
        
        # Create interpolators
        if interpolation == "rate_linear":
            self._rate_interp = interp1d(
                self.times, self.rates,
                kind='linear',
                fill_value=(self.rates[0], self.rates[-1]),
                bounds_error=False
            )
        elif interpolation == "df_linear":
            self._df_interp = interp1d(
                self.times, self.discount_factors,
                kind='linear',
                fill_value=(self.discount_factors[0], self.discount_factors[-1]),
                bounds_error=False
            )
    
    def rate(self, time: float) -> float:
        """Get continuously compounded rate at time T."""
        if self.interpolation == "rate_linear":
            return float(self._rate_interp(time))
        else:
            # Back out from discount factor
            df = self.discount_factor(time)
            return -np.log(df) / time if time > 0 else self.rates[0]
    
    def discount_factor(self, time: float) -> float:
        """Get discount factor at time T: DF(T) = exp(-r(T) * T)."""
        if self.interpolation == "df_linear":
            return float(self._df_interp(time))
        else:
            r = self.rate(time)
            return np.exp(-r * time)
    
    def forward_rate(self, t1: float, t2: float) -> float:
        """Get forward rate between t1 and t2.
        
        F(t1, t2) = [ln(DF(t1)) - ln(DF(t2))] / (t2 - t1)
        """
        if t2 <= t1:
            return self.rate(t1)
        
        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        return (np.log(df1) - np.log(df2)) / (t2 - t1)
    
    @classmethod
    def flat_curve(cls, currency: str, rate: float) -> "RateCurve":
        """Create a flat rate curve.
        
        Args:
            currency: Currency code
            rate: Flat continuously compounded rate
        """
        times = np.array([0.0, 30.0])
        rates = np.array([rate, rate])
        return cls(currency=currency, times=times, rates=rates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "currency": self.currency,
            "times": self.times.tolist(),
            "rates": self.rates.tolist(),
            "discount_factors": self.discount_factors.tolist(),
            "interpolation": self.interpolation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RateCurve":
        """Deserialize from dictionary."""
        return cls(
            currency=data["currency"],
            times=np.array(data["times"]),
            rates=np.array(data["rates"]),
            interpolation=data.get("interpolation", "rate_linear")
        )


class DiscountCurve:
    """Wrapper around RateCurve for discounting operations.
    
    This is a convenience class that could have additional methods
    for present value calculations, annuity factors, etc.
    """
    
    def __init__(self, rate_curve: RateCurve):
        self.rate_curve = rate_curve
    
    def pv(self, cashflow: float, time: float) -> float:
        """Present value of a single cash flow."""
        df = self.rate_curve.discount_factor(time)
        return cashflow * df
    
    def pv_series(self, cashflows: np.ndarray, times: np.ndarray) -> float:
        """Present value of a series of cash flows."""
        dfs = np.array([self.rate_curve.discount_factor(t) for t in times])
        return np.sum(cashflows * dfs)
    
    def annuity_factor(self, times: np.ndarray) -> float:
        """Sum of discount factors at given times."""
        return sum(self.rate_curve.discount_factor(t) for t in times)
