"""Engine state persistence for repricing workflows."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np


class EngineSnapshot:
    """Captures complete engine state for save/load/reprice workflows.

    This is the institutional "snapshot" pattern that enables:
    - Repricing: Load snapshot, bump market, reprice
    - Amendments: Load snapshot, change product, reprice
    - Audit trail: Save snapshots with metadata
    - What-if analysis: Load base, create scenarios

    Snapshot contains:
    - Product definition
    - Market data (as of date)
    - Model configuration (vol model, correlation, etc.)
    - Pricing configuration (MC params, etc.)
    - Results (price, Greeks, etc.)
    - Metadata (timestamp, user, version)
    """

    def __init__(
        self,
        snapshot_id: str,
        product: Any,
        market_data: Any,
        model_config: dict[str, Any],
        pricing_config: dict[str, Any],
        results: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize engine snapshot.

        Args:
            snapshot_id: Unique identifier for this snapshot
            product: StructuredProduct instance (or ComposablePayoff)
            market_data: MarketDataSnapshot instance
            model_config: Model configuration (vol model, corr model, etc.)
            pricing_config: Pricing configuration (n_paths, seed, etc.)
            results: Pricing results (price, Greeks, etc.)
            metadata: Additional metadata (user, timestamp, version)
        """
        self.snapshot_id = snapshot_id
        self.product = product
        self.market_data = market_data
        self.model_config = model_config
        self.pricing_config = pricing_config
        self.results = results or {}
        self.metadata = metadata or {}

        # Auto-populate metadata
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now().isoformat()
        if "version" not in self.metadata:
            self.metadata["version"] = "1.0"

    def save(self, filepath: str) -> None:
        """Save snapshot to JSON file.

        Args:
            filepath: Path to save snapshot (e.g., "snapshots/snapshot_20260117.json")
        """
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Serialize to dict
        snapshot_dict = self.to_dict()

        # Write to file
        with open(filepath, "w") as f:
            json.dump(snapshot_dict, f, indent=2)

        print(f"✅ Snapshot saved: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "EngineSnapshot":
        """Load snapshot from JSON file.

        Args:
            filepath: Path to snapshot file

        Returns:
            EngineSnapshot instance
        """
        with open(filepath, "r") as f:  # noqa: UP015
            snapshot_dict = json.load(f)

        return cls.from_dict(snapshot_dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "product": self._serialize_product(),
            "market_data": self.market_data.to_dict(),
            "model_config": self.model_config,
            "pricing_config": self.pricing_config,
            "results": self._serialize_results(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineSnapshot":
        """Deserialize from dictionary."""
        # Reconstruct market data
        from ..marketdata.base import MarketDataSnapshot

        market_data = MarketDataSnapshot.from_dict(data["market_data"])

        # Reconstruct product (requires registry)
        product = cls._deserialize_product(data["product"])

        # Deserialize results (handle numpy arrays)
        results = cls._deserialize_results(data.get("results", {}))

        return cls(
            snapshot_id=data["snapshot_id"],
            product=product,
            market_data=market_data,
            model_config=data["model_config"],
            pricing_config=data["pricing_config"],
            results=results,
            metadata=data.get("metadata", {}),
        )

    def _serialize_product(self) -> dict[str, Any]:
        """Serialize product to dict."""
        if hasattr(self.product, "to_dict"):
            return self.product.to_dict()
        elif hasattr(self.product, "to_json"):
            return json.loads(self.product.to_json())
        else:
            raise ValueError("Product must implement to_dict() or to_json()")

    @staticmethod
    def _deserialize_product(data: dict[str, Any]) -> Any:
        """Deserialize product from dict."""
        product_type = data.get("type", "unknown")

        if product_type == "composable":
            from ..products.payoffs.base import ComposablePayoff

            return ComposablePayoff.from_dict(data)
        elif "product_id" in data:
            # Legacy AutocallableProduct
            from ..products.autocallable import AutocallableProduct

            return AutocallableProduct.from_json(json.dumps(data))
        else:
            raise ValueError(f"Unknown product type: {product_type}")

    def _serialize_results(self) -> dict[str, Any]:
        """Serialize results (handle numpy arrays)."""
        serialized = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                serialized[key] = {
                    "_type": "ndarray",
                    "data": value.tolist(),
                    "dtype": str(value.dtype),
                    "shape": value.shape,
                }
            elif isinstance(value, (int, float, str, bool, type(None))):
                serialized[key] = value
            elif isinstance(value, dict):
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized

    @staticmethod
    def _deserialize_results(data: dict[str, Any]) -> dict[str, Any]:
        """Deserialize results (reconstruct numpy arrays)."""
        deserialized = {}
        for key, value in data.items():
            if isinstance(value, dict) and value.get("_type") == "ndarray":
                deserialized[key] = np.array(value["data"], dtype=value["dtype"])
            else:
                deserialized[key] = value
        return deserialized

    def reprice(
        self,
        pricing_engine: Any,
        new_market_data: Any | None = None,
        new_pricing_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reprice using this snapshot with optional market/config changes.

        This is the key repricing workflow:
        1. Load snapshot
        2. Optionally bump market data
        3. Reprice with same product/models
        4. Return new results

        Args:
            pricing_engine: PricingEngine instance
            new_market_data: Optional new market data (for bumps)
            new_pricing_config: Optional new pricing config (e.g., more paths)

        Returns:
            New pricing results dict
        """
        market_data = new_market_data if new_market_data is not None else self.market_data
        pricing_config = (
            new_pricing_config if new_pricing_config is not None else self.pricing_config
        )

        # Reconstruct engine from snapshot
        # (This requires PricingEngine to accept snapshot)
        results = pricing_engine.price_from_snapshot(
            snapshot=self, market_data=market_data, pricing_config=pricing_config
        )

        return results

    def create_scenario(
        self, scenario_name: str, market_data_bumps: dict[str, Any]
    ) -> "EngineSnapshot":
        """Create a scenario snapshot with market data bumps.

        Args:
            scenario_name: Name for scenario
            market_data_bumps: Dict of bumps, e.g.:
                {"spots": {"AAPL": 110.0}, "rates": {"USD": 0.035}}

        Returns:
            New EngineSnapshot with bumped market data
        """
        from copy import deepcopy

        # Copy market data
        new_market_data = deepcopy(self.market_data)

        # Apply bumps
        if "spots" in market_data_bumps:
            for ticker, new_spot in market_data_bumps["spots"].items():
                new_market_data.spots[ticker] = new_spot

        # (Other bump types would go here)

        # Create new snapshot
        scenario_id = f"{self.snapshot_id}_scenario_{scenario_name}"
        metadata = self.metadata.copy()
        metadata["scenario"] = scenario_name
        metadata["base_snapshot"] = self.snapshot_id

        return EngineSnapshot(
            snapshot_id=scenario_id,
            product=self.product,
            market_data=new_market_data,
            model_config=self.model_config,
            pricing_config=self.pricing_config,
            metadata=metadata,
        )
