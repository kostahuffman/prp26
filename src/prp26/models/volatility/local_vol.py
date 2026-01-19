"""
Local volatility model implementation.
"""

import numpy as np

from .base import VolSurface


class LocalVolModel:
    """
    Local volatility model.

    Implements Dupire local volatility:
        dS = r*S*dt + σ_LV(t, S)*S*dW

    The local vol surface σ_LV(t, K) is computed from implied vol surface
    using Dupire's formula.
    """

    def __init__(
        self,
        vol_surface: VolSurface | None = None,
        flat_vol: float | None = None,
    ):
        """
        Initialize local vol model.

        Args:
            vol_surface: Calibrated local vol surface
            flat_vol: Flat volatility (for testing/simple cases)
        """
        if vol_surface is None and flat_vol is None:
            raise ValueError("Must provide either vol_surface or flat_vol")

        self.vol_surface = vol_surface
        self.flat_vol = flat_vol
        self.is_flat = flat_vol is not None

    def calibrate(self, market_data) -> None:
        """
        Calibrate local vol surface from implied vols using Dupire's formula.

        Dupire: σ_LV²(T, K) = (∂C/∂T + rK*∂C/∂K) / (½K²*∂²C/∂K²)

        Args:
            market_data: Provider with get_implied_vol_surface()
        """
        if self.is_flat:
            return  # No calibration needed for flat vol

        implied_surface = market_data.get_implied_vol_surface()

        # Compute local vol from implied vol via Dupire
        self.vol_surface = self._dupire_transform(implied_surface, market_data)

    def _dupire_transform(self, implied_surface: VolSurface, market_data) -> VolSurface:
        """
        Transform implied vol surface to local vol using Dupire's formula.

        Simplified implementation using finite differences.
        """
        times = implied_surface.times
        strikes = implied_surface.strikes
        impl_vols = implied_surface.vols

        # Get discount rate
        r = market_data.get_rate()

        # Compute local vol using finite differences
        local_vols = np.zeros_like(impl_vols)

        for i, T in enumerate(times):
            for j, K in enumerate(strikes):
                # Get implied vol and derivatives
                sigma = impl_vols[i, j]

                # Approximate derivatives using finite differences
                if i < len(times) - 1:
                    dC_dT = self._call_price_derivative_time(
                        T, K, sigma, impl_vols[i + 1, j], times[i + 1] - T
                    )
                else:
                    dC_dT = 0.0

                if j > 0 and j < len(strikes) - 1:
                    d2C_dK2 = self._call_price_second_derivative_strike(
                        T,
                        K,
                        sigma,
                        strikes[j - 1],
                        strikes[j + 1],
                        impl_vols[i, j - 1],
                        impl_vols[i, j + 1],
                    )
                else:
                    d2C_dK2 = 1.0  # Fallback

                # Dupire formula
                numerator = dC_dT + r * K * 0.0  # Simplified: ignore first derivative
                denominator = 0.5 * K * K * d2C_dK2

                if denominator > 1e-10:
                    local_vol_squared = numerator / denominator
                    local_vols[i, j] = np.sqrt(max(local_vol_squared, sigma * sigma))
                else:
                    local_vols[i, j] = sigma

        return VolSurface(times, strikes, local_vols, implied_surface.interpolation)

    def _call_price_derivative_time(self, T, K, sigma, sigma_next, dT) -> float:
        """Approximate ∂C/∂T using Black-Scholes."""
        from scipy.stats import norm

        S = 100.0  # Assume normalized
        d1 = (np.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * np.sqrt(T))

        # Simplified: use theta approximation
        vega = S * norm.pdf(d1) * np.sqrt(T)
        return vega * (sigma_next - sigma) / dT

    def _call_price_second_derivative_strike(
        self, T, K, sigma, K_minus, K_plus, sigma_minus, sigma_plus
    ) -> float:
        """Approximate ∂²C/∂K² using finite differences."""
        from scipy.stats import norm

        S = 100.0

        # Digital option approximation
        d2 = (np.log(S / K) - 0.5 * sigma * sigma * T) / (sigma * np.sqrt(T))
        return norm.pdf(d2) / (K * sigma * np.sqrt(T))

    def vol(self, t: float, asset_idx: int, state: dict | None = None) -> float:
        """
        Get local volatility at time t.

        Args:
            t: Time in years
            asset_idx: Asset index (for multi-asset support)
            state: Optional state dict with 'spot' for spot-dependent vol

        Returns:
            Local volatility
        """
        if self.is_flat:
            return self.flat_vol

        # Get spot from state, default to ATM
        spot = state.get("spot", 1.0) if state else 1.0

        return self.vol_surface.get_vol(t, spot)

    def to_dict(self) -> dict:
        """Serialize model to dictionary."""
        return {
            "model_type": "local_vol",
            "vol_surface": self.vol_surface.to_dict() if self.vol_surface else None,
            "flat_vol": self.flat_vol,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LocalVolModel":
        """Deserialize model from dictionary."""
        vol_surface = None
        if data.get("vol_surface"):
            vol_surface = VolSurface.from_dict(data["vol_surface"])

        return cls(
            vol_surface=vol_surface,
            flat_vol=data.get("flat_vol"),
        )
