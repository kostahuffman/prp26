"""
Base product definitions for structured products.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import date
import json  # noqa: F401
from typing import TYPE_CHECKING

from .taxonomy import ProductTaxonomy

if TYPE_CHECKING:
    from .payoffs.evaluators.base import PayoffEvaluator


@dataclass
class Underlying:
    """Represents a single underlying asset."""

    symbol: str
    asset_class: str = "equity"  # equity, fx, rates, commodity
    spot: float | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Basket:
    """Basket of underlyings with weights."""

    underlyings: list[Underlying]
    weights: list[float]
    worst_of: bool = True
    best_of: bool = False

    def __post_init__(self):
        if len(self.underlyings) != len(self.weights):
            raise ValueError("Number of underlyings must match number of weights")
        if abs(sum(self.weights) - 1.0) > 1e-6 and not self.worst_of:
            raise ValueError("Weights must sum to 1.0 for non worst-of baskets")

    def to_dict(self):
        return {
            "underlyings": [u.to_dict() for u in self.underlyings],
            "weights": self.weights,
            "worst_of": self.worst_of,
            "best_of": self.best_of,
        }


class StructuredProduct(ABC):
    """Base class for all structured products.

    All concrete product classes must implement get_evaluator() to provide
    the PayoffEvaluator that the PricingEngine will use to evaluate payoffs.
    """

    def __init__(
        self,
        product_id: str,
        currency: str,
        notional: float,
        issue_date: date | None = None,
        maturity_date: date | None = None,
        taxonomy: ProductTaxonomy | None = None,
        quanto_currency: str | None = None,
    ):
        self.product_id = product_id
        self.currency = currency
        self.notional = notional
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self.taxonomy = taxonomy
        self.quanto_currency = quanto_currency  # If set, payoff is in different currency

    def is_quanto(self) -> bool:
        """Check if product is quanto (payoff in different currency than underlying).

        Returns:
            True if quanto feature is enabled
        """
        return self.quanto_currency is not None and self.quanto_currency != self.currency

    @abstractmethod
    def get_evaluator(self) -> PayoffEvaluator:
        """Get the PayoffEvaluator for this product.

        This method must be implemented by all concrete product classes.
        The evaluator is used by PricingEngine to evaluate payoffs along Monte Carlo paths.

        Returns:
            PayoffEvaluator: The evaluator configured for this product
        """
        ...

    def to_json(self) -> str:
        """Serialize product to JSON."""
        raise NotImplementedError

    @classmethod
    def from_json(cls, json_str: str) -> StructuredProduct:
        """Deserialize product from JSON."""
        raise NotImplementedError

    def validate(self) -> bool:
        """Validate product definition."""
        if self.notional <= 0:
            raise ValueError("Notional must be positive")
        return True
