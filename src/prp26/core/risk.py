"""Risk engine for computing Greeks and sensitivities."""

from copy import deepcopy
from typing import Any, Callable  # noqa: UP035

import numpy as np


class RiskEngine:
    """Computes risk sensitivities (Greeks) for structured products.

    Supports multiple computation methods:
    - Bump-and-reprice (finite differences)
    - Pathwise method (future)
    - Adjoint/AAD (future - full implementation)

    Institutional Greeks:
    - Delta: dPV/dS (spot sensitivity)
    - Gamma: d²PV/dS² (convexity)
    - Vega: dPV/dσ (vol sensitivity)
    - Rho: dPV/dr (rate sensitivity)
    - Correlation sensitivity: dPV/dρ
    """

    def __init__(
        self,
        pricing_function: Callable,
        market_data: Any,
        bump_sizes: dict[str, float] | None = None,
    ):
        """Initialize risk engine.

        Args:
            pricing_function: Function that takes market_data and returns price
            market_data: MarketDataSnapshot instance
            bump_sizes: Dict of bump sizes for each Greek
                        Default: {"delta": 0.01, "gamma": 0.01, "vega": 0.01, "rho": 0.0001}
        """
        self.pricing_function = pricing_function
        self.market_data = market_data

        if bump_sizes is None:
            self.bump_sizes = {
                "delta": 0.01,  # 1% spot bump
                "gamma": 0.01,  # 1% spot bump
                "vega": 0.01,  # 1 vol point bump
                "rho": 0.0001,  # 1bp rate bump
                "correlation": 0.01,  # 1% correlation bump
            }
        else:
            self.bump_sizes = bump_sizes

    def compute_greeks(
        self, tickers: list, base_price: float | None = None, method: str = "bump_reprice"
    ) -> dict[str, Any]:
        """Compute all Greeks for given tickers.

        Args:
            tickers: List of asset tickers to compute Greeks for
            base_price: Base price (if None, will compute)
            method: "bump_reprice" or "adjoint"

        Returns:
            Dict with Greeks:
                {
                    "base_price": float,
                    "delta": {ticker: value, ...},
                    "gamma": {ticker: value, ...},
                    "vega": {ticker: value, ...},
                    "rho": float,
                    "correlation": {(ticker1, ticker2): value, ...}
                }
        """
        if method == "bump_reprice":
            return self._compute_greeks_bump_reprice(tickers, base_price)
        elif method == "adjoint":
            # Future: full AAD implementation
            return self._compute_greeks_adjoint(tickers, base_price)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _compute_greeks_bump_reprice(
        self, tickers: list, base_price: float | None
    ) -> dict[str, Any]:
        """Compute Greeks using bump-and-reprice (finite differences)."""

        # Compute base price if not provided
        if base_price is None:
            base_price = self.pricing_function(self.market_data)

        greeks = {
            "base_price": base_price,
            "delta": {},
            "gamma": {},
            "vega": {},
            "rho": None,
            "correlation": {},
        }

        # Delta and Gamma (spot sensitivities)
        for ticker in tickers:
            delta, gamma = self._compute_delta_gamma(ticker, base_price)
            greeks["delta"][ticker] = delta
            greeks["gamma"][ticker] = gamma

        # Vega (vol sensitivities)
        for ticker in tickers:
            vega = self._compute_vega(ticker, base_price)
            greeks["vega"][ticker] = vega

        # Rho (rate sensitivity)
        rho = self._compute_rho(base_price)
        greeks["rho"] = rho

        # Correlation sensitivities (pairwise)
        if len(tickers) > 1:
            for i, ticker1 in enumerate(tickers):
                for ticker2 in tickers[i + 1 :]:
                    corr_sens = self._compute_correlation_sensitivity(ticker1, ticker2, base_price)
                    greeks["correlation"][(ticker1, ticker2)] = corr_sens

        return greeks

    def _compute_delta_gamma(self, ticker: str, base_price: float) -> tuple:
        """Compute delta and gamma for a ticker using central differences.

        Delta = (P(S+h) - P(S-h)) / (2h)
        Gamma = (P(S+h) - 2*P(S) + P(S-h)) / h²
        """
        bump_pct = self.bump_sizes["delta"]

        # Get original spot
        original_spot = self.market_data.get_spot(ticker)

        # Bump up - need to modify spots dict in-place
        market_data_up = deepcopy(self.market_data)
        new_spot_up = original_spot * (1 + bump_pct)
        market_data_up.spots[ticker] = new_spot_up
        price_up = self.pricing_function(market_data_up)

        # Bump down
        market_data_down = deepcopy(self.market_data)
        new_spot_down = original_spot * (1 - bump_pct)
        market_data_down.spots[ticker] = new_spot_down
        price_down = self.pricing_function(market_data_down)

        # Compute delta (first derivative) - per unit change in spot
        # Delta = (P_up - P_down) / (S_up - S_down)
        spot_change = new_spot_up - new_spot_down
        delta = (price_up - price_down) / spot_change

        # Compute gamma (second derivative)
        # Gamma = (P_up - 2*P_base + P_down) / h²
        # where h is the bump amount
        h = original_spot * bump_pct
        gamma = (price_up - 2 * base_price + price_down) / (h**2)

        return delta, gamma

    def _compute_vega(self, ticker: str, base_price: float) -> float:
        """Compute vega for a ticker.

        Vega = dP/dσ (sensitivity to 1% vol change)

        Bumps implied volatility in market data snapshot.
        """
        bump_vol = self.bump_sizes["vega"]  # Typically 0.01 (1%)

        # Check if implied_vols exist in market data
        if not hasattr(self.market_data, "implied_vols") or not self.market_data.implied_vols:
            # No vol data to bump - return 0
            return 0.0

        # Check if this ticker has vol data
        if ticker not in self.market_data.implied_vols:
            return 0.0

        # Get original vol
        original_vol = self.market_data.implied_vols.get(ticker)
        if original_vol is None:
            return 0.0

        # Bump vol up
        market_data_up = deepcopy(self.market_data)
        market_data_up.implied_vols[ticker] = original_vol + bump_vol
        price_up = self.pricing_function(market_data_up)

        # Bump vol down
        market_data_down = deepcopy(self.market_data)
        market_data_down.implied_vols[ticker] = original_vol - bump_vol
        price_down = self.pricing_function(market_data_down)

        # Central difference: Vega = (P_up - P_down) / (2 * bump)
        vega = (price_up - price_down) / (2 * bump_vol)

        return vega

    def _compute_rho(self, base_price: float) -> float:
        """Compute rho (rate sensitivity).

        Rho = dP/dr (sensitivity to 1bp rate change)
        """
        bump_rate = self.bump_sizes["rho"]

        # Bump all rate curves up
        market_data_up = deepcopy(self.market_data)
        for _currency, rate_curve in market_data_up.rates.items():
            # Bump rates uniformly
            rate_curve.rates = rate_curve.rates + bump_rate
            rate_curve.discount_factors = np.exp(-rate_curve.rates * rate_curve.times)

        price_up = self.pricing_function(market_data_up)

        # Rho per 1bp
        rho = (price_up - base_price) / bump_rate

        return rho

    def _compute_correlation_sensitivity(
        self, ticker1: str, ticker2: str, base_price: float
    ) -> float:
        """Compute correlation sensitivity.

        dP/dρ (sensitivity to correlation change)
        """
        bump_corr = self.bump_sizes["correlation"]  # Typically 0.01 (1%)

        # Check if correlation matrix exists in market data
        if not hasattr(self.market_data, "correlations") or self.market_data.correlations is None:
            return 0.0

        # Get ticker indices (need to map tickers to indices)
        # This assumes tickers are ordered in the same way as correlation matrix
        # In production, would need proper ticker-to-index mapping
        try:
            # Try to find indices - this is a simplified approach
            # In practice, market_data should have a ticker_map
            spots_list = list(self.market_data.spots.keys())
            idx1 = spots_list.index(ticker1)
            idx2 = spots_list.index(ticker2)
        except (ValueError, AttributeError):
            # Can't find ticker indices
            return 0.0

        # Get original correlation
        original_corr = self.market_data.correlations[idx1, idx2]

        # Bump correlation up (respecting bounds [-1, 1])
        new_corr_up = min(original_corr + bump_corr, 0.999)  # Cap at 0.999 to avoid singularity
        market_data_up = deepcopy(self.market_data)
        market_data_up.correlations[idx1, idx2] = new_corr_up
        market_data_up.correlations[idx2, idx1] = new_corr_up  # Symmetric

        # Ensure matrix remains positive semi-definite (simplified - just price)
        try:
            price_up = self.pricing_function(market_data_up)
        except:
            # If correlation bump breaks positive definiteness, return 0
            return 0.0

        # Bump correlation down
        new_corr_down = max(original_corr - bump_corr, -0.999)  # Cap at -0.999
        market_data_down = deepcopy(self.market_data)
        market_data_down.correlations[idx1, idx2] = new_corr_down
        market_data_down.correlations[idx2, idx1] = new_corr_down  # Symmetric

        try:
            price_down = self.pricing_function(market_data_down)
        except:
            # If correlation bump breaks positive definiteness, return 0
            return 0.0

        # Central difference: dP/dρ = (P_up - P_down) / (2 * bump)
        corr_sens = (price_up - price_down) / (2 * bump_corr)

        return corr_sens

    def _compute_greeks_adjoint(self, tickers: list, base_price: float | None) -> dict[str, Any]:
        """Compute Greeks using adjoint/AAD (future implementation).

        This would use tape-based automatic differentiation for O(1) Greeks.
        Requires integration with AD frameworks (JAX, PyTorch, etc.)
        """
        # Placeholder - full AAD implementation would go here
        print("⚠️  Adjoint Greeks not yet fully implemented, using bump-reprice")
        return self._compute_greeks_bump_reprice(tickers, base_price)

    def compute_pnl_explain(self, new_market_data: Any, greeks: dict[str, Any]) -> dict[str, float]:
        """Explain P&L using Taylor expansion with Greeks.

        ΔPV ≈ Σ(Δᵢ * ΔSᵢ) + 0.5 * Σ(Γᵢ * ΔSᵢ²) + ...

        Args:
            new_market_data: New market data snapshot
            greeks: Greeks dict from compute_greeks()

        Returns:
            Dict with P&L attribution by risk factor
        """
        pnl_explain = {
            "delta_pnl": 0.0,
            "gamma_pnl": 0.0,
            "vega_pnl": 0.0,
            "rho_pnl": 0.0,
            "unexplained": 0.0,
        }

        # Delta P&L
        for ticker, delta in greeks["delta"].items():
            old_spot = self.market_data.get_spot(ticker)
            new_spot = new_market_data.get_spot(ticker)
            spot_change = new_spot - old_spot
            pnl_explain["delta_pnl"] += delta * spot_change

        # Gamma P&L
        for ticker, gamma in greeks["gamma"].items():
            old_spot = self.market_data.get_spot(ticker)
            new_spot = new_market_data.get_spot(ticker)
            spot_change = new_spot - old_spot
            pnl_explain["gamma_pnl"] += 0.5 * gamma * (spot_change**2)

        # Actual P&L
        new_price = self.pricing_function(new_market_data)
        actual_pnl = new_price - greeks["base_price"]

        # Unexplained
        explained = (
            pnl_explain["delta_pnl"]
            + pnl_explain["gamma_pnl"]
            + pnl_explain["vega_pnl"]
            + pnl_explain["rho_pnl"]
        )
        pnl_explain["unexplained"] = actual_pnl - explained
        pnl_explain["actual_pnl"] = actual_pnl

        return pnl_explain
