"""Master pricing engine that orchestrates all components.

This is the institutional-grade pricing engine that brings together:
- Market Data Layer
- Model Bundle (vol, correlation, rates, dividends)
- Path Generator
- Payoff Evaluator
- Risk Engine
- Persistence Layer
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..marketdata.base import MarketDataSnapshot
from ..models.correlation.base import CorrelationModel
from ..models.volatility.base import VolatilityModel
from ..products.base import StructuredProduct
from ..products.payoffs.base import ComposablePayoff
from .paths import PathGenerator
from .payoff import PayoffEvaluator  # noqa: F401
from .risk import RiskEngine
from .snapshot import EngineSnapshot


@dataclass
class ModelBundle:
    """Bundle of models for pricing.

    This contains all the stochastic models needed:
    - Volatility model (Local Vol, SLV, Heston, etc.)
    - Correlation model (constant, skew, etc.)
    - Rate model (implicit in market data)
    - Dividend model (implicit in market data)
    """

    vol_model: VolatilityModel
    corr_model: CorrelationModel

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"vol_model": self.vol_model.to_dict(), "corr_model": self.corr_model.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelBundle":
        """Deserialize from dictionary."""
        # Would need model registry for full implementation
        raise NotImplementedError("Model deserialization requires registry")


class PricingEngine:
    """Master pricing engine following institutional architecture.

    Usage:
        # Setup engine
        engine = PricingEngine(
            product=my_product,
            market_data=market_snapshot,
            models=ModelBundle(vol_model=slv, corr_model=corr_skew)
        )

        # Price
        result = engine.price(n_paths=100_000)

        # Compute Greeks
        greeks = engine.compute_greeks(["AAPL", "MSFT", "GOOGL"])

        # Save snapshot
        snapshot = engine.create_snapshot("snapshot_20260117")
        snapshot.save("snapshots/snapshot_20260117.json")

        # Reprice later
        snapshot = EngineSnapshot.load("snapshots/snapshot_20260117.json")
        new_result = engine.reprice_from_snapshot(snapshot, new_market_data)
    """

    def __init__(
        self,
        product: StructuredProduct | ComposablePayoff,
        market_data: MarketDataSnapshot,
        models: ModelBundle,
        pricing_config: dict[str, Any] | None = None,
    ):
        """Initialize pricing engine.

        Args:
            product: StructuredProduct or ComposablePayoff
            market_data: MarketDataSnapshot with spots, rates, dividends
            models: ModelBundle with vol and correlation models
            pricing_config: Optional pricing configuration
                {
                    "n_paths": 100000,
                    "seed": 42,
                    "backend": "cpu",  # or "gpu"
                    "antithetic": False
                }
        """
        self.product = product
        self.market_data = market_data
        self.models = models

        # Default pricing config
        if pricing_config is None:
            pricing_config = {"n_paths": 100_000, "seed": 42, "backend": "cpu", "antithetic": False}
        self.pricing_config = pricing_config

        # Extract product information
        self._setup_product_info()

        # Initialize path generator
        self._setup_path_generator()

        # Initialize risk engine
        self.risk_engine = None  # Created on demand

    def _setup_product_info(self) -> None:
        """Extract information from product."""
        # Get tickers from product
        if hasattr(self.product, "basket"):
            self.tickers = [u.symbol for u in self.product.basket.underlyings]
            self.notional = self.product.notional
            self.times = np.array(self.product.observation_schedule.times)
        else:
            # For ComposablePayoff, would need different extraction
            raise NotImplementedError("ComposablePayoff product info extraction")

        # Get initial spots
        self.initial_spots = np.array(
            [self.market_data.get_spot(ticker) for ticker in self.tickers]
        )

        # Get dividend yields
        self.dividend_yields = np.array(
            [self.market_data.get_dividend_yield(ticker, self.times[-1]) for ticker in self.tickers]
        )

    def _setup_path_generator(self) -> None:
        """Initialize path generator with models and market data."""
        # Get risk-free rate (use first currency in rates)
        currency = list(self.market_data.rates.keys())[0]
        rate = self.market_data.get_discount_rate(self.times[-1], currency)

        self.path_generator = PathGenerator(
            vol_model=self.models.vol_model,
            corr_model=self.models.corr_model,
            rate=rate,
            dividend_yields=self.dividend_yields,
            seed=self.pricing_config.get("seed"),
        )

    def price(
        self,
        n_paths: int | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        """Price the product using Monte Carlo simulation.

        Args:
            n_paths: Number of Monte Carlo paths (overrides config)
            backend: "cpu" or "gpu" (overrides config)

        Returns:
            Dict with:
                - "price": Present value
                - "price_std": Standard error
                - "payoffs": Individual payoffs (n_paths,)
                - "cashflows": Cashflows by time (n_paths, n_steps)
                - "termination_times": When paths terminated
                - "diagnostics": Additional info
        """
        n_paths = n_paths or self.pricing_config["n_paths"]
        backend = backend or self.pricing_config["backend"]

        # Get current spots from market data (not cached initial_spots)
        # This allows Greeks to work correctly when market data is bumped
        current_spots = np.array([self.market_data.get_spot(t) for t in self.tickers])

        # Generate paths with current spots
        paths, variances = self.path_generator.generate_paths(
            spots=current_spots, times=self.times, n_paths=n_paths, backend=backend
        )

        # Evaluate payoff (also needs current spots for initial level comparison)
        payoff_result = self._evaluate_payoff(paths, initial_spots=current_spots)

        # Discount cash flows
        discounted_payoffs = self._discount_payoffs(payoff_result)

        # Compute price and standard error
        price = np.mean(discounted_payoffs)
        price_std = np.std(discounted_payoffs) / np.sqrt(n_paths)

        # Diagnostics
        diagnostics = {
            "n_paths": n_paths,
            "backend": backend,
            "termination_rate": np.mean(payoff_result.get("terminated", [])),
            "mean_termination_time": np.mean(payoff_result.get("termination_times", [])),
        }

        return {
            "price": price,
            "price_std": price_std,
            "payoffs": discounted_payoffs,
            "cashflows": payoff_result["cashflows"],
            "termination_times": payoff_result.get("termination_times"),
            "diagnostics": diagnostics,
        }

    def _evaluate_payoff(self, paths: np.ndarray, initial_spots: np.ndarray) -> dict[str, Any]:
        """Evaluate payoff for paths.

        Args:
            paths: Asset paths (n_paths, n_steps, n_assets)
            initial_spots: Initial spot prices (n_assets,) - taken from current market data
        """
        if isinstance(self.product, ComposablePayoff):
            # Use compositional payoff system
            return self.product.evaluate_path(
                paths=paths, times=self.times, initial_spots=initial_spots, notional=self.notional
            )
        
        # All StructuredProduct subclasses must implement get_evaluator()
        evaluator = self.product.get_evaluator()
        return evaluator.evaluate(paths, initial_spots=initial_spots)

    def _discount_payoffs(self, payoff_result: dict[str, Any]) -> np.ndarray:
        """Discount cashflows to present value."""
        # Check if we have pre-computed payoffs or need to discount cashflows
        if "payoffs" in payoff_result and payoff_result["payoffs"] is not None:
            # Payoffs are already computed (undiscounted total payoffs)
            # We need to discount them to their payment time
            payoffs = payoff_result["payoffs"]
            termination_times = payoff_result.get(
                "termination_times", np.full(len(payoffs), self.times[-1])
            )

            # Get discount factor for each path's termination time
            currency = list(self.market_data.rates.keys())[0]
            discount_factors = np.array(
                [self.market_data.get_discount_factor(currency, t) for t in termination_times]
            )

            return payoffs * discount_factors
        else:
            # Fall back to cashflow discounting
            cashflows = payoff_result["cashflows"]
            n_paths, n_steps = cashflows.shape

            # Get discount factors
            currency = list(self.market_data.rates.keys())[0]
            discount_factors = np.array(
                [self.market_data.get_discount_factor(currency, t) for t in self.times]
            )

            # Discount each cashflow
            discounted = cashflows * discount_factors

            # Sum to get present value per path
            pv_per_path = np.sum(discounted, axis=1)

            return pv_per_path

    def compute_greeks(self, tickers: list = None, method: str = "bump_reprice") -> dict[str, Any]:
        """Compute Greeks (delta, gamma, vega, rho).

        Args:
            tickers: List of tickers (default: all in product)
            method: "bump_reprice" or "adjoint"

        Returns:
            Dict with Greeks by ticker
        """
        if tickers is None:
            tickers = self.tickers

        # Create risk engine if not exists
        if self.risk_engine is None:
            # Wrap pricing function
            def pricing_fn(market_data):
                # Temporarily swap market data
                old_market_data = self.market_data
                self.market_data = market_data
                self._setup_path_generator()  # Rebuild with new data

                result = self.price()

                # Restore
                self.market_data = old_market_data
                self._setup_path_generator()

                return result["price"]

            self.risk_engine = RiskEngine(pricing_function=pricing_fn, market_data=self.market_data)

        # Compute Greeks
        greeks = self.risk_engine.compute_greeks(tickers=tickers, method=method)

        return greeks

    def create_snapshot(self, snapshot_id: str) -> EngineSnapshot:
        """Create a snapshot of current engine state.

        Args:
            snapshot_id: Unique identifier for snapshot

        Returns:
            EngineSnapshot that can be saved and reloaded
        """
        # Price if not already done
        pricing_result = self.price()

        # Create snapshot
        snapshot = EngineSnapshot(
            snapshot_id=snapshot_id,
            product=self.product,
            market_data=self.market_data,
            model_config=self.models.to_dict(),
            pricing_config=self.pricing_config,
            results=pricing_result,
            metadata={"tickers": self.tickers, "notional": self.notional},
        )

        return snapshot

    @classmethod
    def from_snapshot(
        cls, snapshot: EngineSnapshot, market_data: MarketDataSnapshot | None = None
    ) -> "PricingEngine":
        """Create engine from snapshot (for repricing).

        Args:
            snapshot: EngineSnapshot to load
            market_data: Optional new market data (for bumps)

        Returns:
            PricingEngine instance
        """
        # Use snapshot's market data if not provided
        if market_data is None:
            market_data = snapshot.market_data

        # Reconstruct models (would need registry)
        # For now, this is a placeholder
        raise NotImplementedError("Snapshot reconstruction requires model registry")

    def price_from_snapshot(
        self,
        snapshot: EngineSnapshot,
        market_data: MarketDataSnapshot | None = None,
        pricing_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Price using snapshot configuration with optional overrides.

        This is for the repricing workflow.
        """
        # Update market data if provided
        if market_data is not None:
            self.market_data = market_data
            self._setup_path_generator()

        # Update pricing config if provided
        if pricing_config is not None:
            self.pricing_config.update(pricing_config)

        # Price
        return self.price()
