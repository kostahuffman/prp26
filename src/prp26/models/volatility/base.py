"""
Base volatility model interfaces.
"""

from typing import Protocol

import numpy as np


class VolatilityModel(Protocol):
    """
    Protocol for volatility models.

    All volatility models must implement these methods to be compatible
    with the pricing engine.
    """

    def calibrate(self, market_data) -> None:
        """
        Calibrate model to market data.

        Args:
            market_data: Market data provider with implied vol surface
        """
        ...

    def vol(self, t: float, asset_idx: int, state: dict | None = None) -> float:
        """
        Get volatility at time t for asset.

        Args:
            t: Time in years
            asset_idx: Asset index in basket
            state: Optional state dictionary for stochastic vol models

        Returns:
            Volatility (annualized)
        """
        ...

    def to_dict(self) -> dict:
        """Serialize model parameters to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict) -> "VolatilityModel":
        """Deserialize model from dictionary."""
        ...


class VolSurface:
    """
    Volatility surface representation.

    Provides implied volatility as function of time and strike.
    """

    def __init__(
        self,
        times: np.ndarray,
        strikes: np.ndarray,
        vols: np.ndarray,
        interpolation: str = "linear",
    ):
        """
        Initialize vol surface.

        Args:
            times: Expiry times (years)
            strikes: Strike levels (absolute or moneyness)
            vols: Implied volatilities [n_times, n_strikes]
            interpolation: Interpolation method (linear, cubic, etc.)
        """
        self.times = times
        self.strikes = strikes
        self.vols = vols
        self.interpolation = interpolation

        self._validate()

    def _validate(self):
        """Validate surface data."""
        if len(self.times) != self.vols.shape[0]:
            raise ValueError("Times dimension must match vol rows")
        if len(self.strikes) != self.vols.shape[1]:
            raise ValueError("Strikes dimension must match vol columns")
        if np.any(self.vols <= 0):
            raise ValueError("Volatilities must be positive")

    def get_vol(self, t: float, k: float) -> float:
        """
        Get interpolated volatility at (t, k).

        Args:
            t: Time to expiry
            k: Strike level

        Returns:
            Interpolated implied volatility
        """
        from scipy.interpolate import RegularGridInterpolator

        if not hasattr(self, "_interpolator"):
            self._interpolator = RegularGridInterpolator(
                (self.times, self.strikes),
                self.vols,
                method=self.interpolation,
                bounds_error=False,
                fill_value=None,
            )

        return float(self._interpolator([t, k])[0])

    def get_atm_vol(self, t: float) -> float:
        """Get ATM volatility at time t."""
        # Assume ATM is at strike = 1.0 for normalized surface
        return self.get_vol(t, 1.0)

    def to_dict(self) -> dict:
        """Serialize surface to dictionary."""
        return {
            "times": self.times.tolist(),
            "strikes": self.strikes.tolist(),
            "vols": self.vols.tolist(),
            "interpolation": self.interpolation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VolSurface":
        """Deserialize surface from dictionary."""
        return cls(
            times=np.array(data["times"]),
            strikes=np.array(data["strikes"]),
            vols=np.array(data["vols"]),
            interpolation=data.get("interpolation", "linear"),
        )
