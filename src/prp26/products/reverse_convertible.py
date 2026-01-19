"""
Reverse Convertible product definitions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date

from .base import Basket, StructuredProduct, Underlying
from .taxonomy import ProductTaxonomy, TAXONOMY_REVERSE_CONVERTIBLE, TAXONOMY_BARRIER_REVERSE_CONVERTIBLE


@dataclass
class ReverseConvertiblePayoffDefinition:
    """Definition of reverse convertible payoff structure.
    
    This is a data class for serialization/deserialization,
    not a PayoffComponent or PayoffEvaluator.
    """

    payoff_type: str = "reverse_convertible"
    strike: float = 1.0  # Strike level (typically 100% of initial)
    coupon_rate: float = 0.10  # Annual coupon rate (e.g., 10%)
    coupon_frequency: str = "quarterly"  # "quarterly", "semi-annual", "annual", "at_maturity"
    barrier: float | None = None  # For Barrier Reverse Convertibles
    barrier_type: str = "european"  # "european", "american", "bermudan"

    def to_dict(self):
        return asdict(self)


class ReverseConvertible(StructuredProduct):
    """
    Reverse Convertible structured product.

    Features:
    - High coupon payments (e.g., 10% p.a.)
    - If underlying(s) below strike at maturity, physical delivery or cash settlement
    - Can be single-asset or worst-of basket
    - Optional barrier feature (only convert if barrier is breached)
    
    Payoff at Maturity:
        If worst-of >= strike:
            Notional + Final Coupon
        Else:
            Notional * (worst-of / strike) + Accrued Coupons
    """

    def __init__(
        self,
        product_id: str,
        currency: str,
        notional: float,
        basket: Basket,
        payoff: ReverseConvertiblePayoffDefinition,
        maturity: float,
        issue_date: date | None = None,
        maturity_date: date | None = None,
        taxonomy: ProductTaxonomy | None = None,
    ):
        if taxonomy is None:
            # Default taxonomy based on whether barrier exists
            taxonomy = (
                TAXONOMY_BARRIER_REVERSE_CONVERTIBLE
                if payoff.barrier is not None
                else TAXONOMY_REVERSE_CONVERTIBLE
            )
        
        super().__init__(product_id, currency, notional, issue_date, maturity_date, taxonomy)
        self.basket = basket
        self.payoff = payoff
        self.maturity = maturity

        self.validate()

    def validate(self) -> bool:
        """Validate product definition."""
        super().validate()

        if self.maturity <= 0:
            raise ValueError("Maturity must be positive")
        
        if self.payoff.coupon_rate <= 0:
            raise ValueError("Coupon rate must be positive")
        
        if self.payoff.barrier is not None:
            if not (0 < self.payoff.barrier <= 1):
                raise ValueError("Barrier must be between 0 and 1")

        return True

    def to_json(self) -> str:
        """Serialize to JSON."""
        data = {
            "product_id": self.product_id,
            "currency": self.currency,
            "notional": self.notional,
            "taxonomy": self.taxonomy.to_dict() if self.taxonomy else None,
            "basket": self.basket.to_dict(),
            "payoff": self.payoff.to_dict(),
            "maturity": self.maturity,
            "issue_date": self.issue_date.isoformat() if self.issue_date else None,
            "maturity_date": self.maturity_date.isoformat() if self.maturity_date else None,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> ReverseConvertible:
        """Deserialize from JSON."""
        data = json.loads(json_str)

        basket = Basket(
            underlyings=[Underlying(**u) for u in data["basket"]["underlyings"]],
            weights=data["basket"]["weights"],
            worst_of=data["basket"]["worst_of"],
            best_of=data["basket"].get("best_of", False),
        )

        payoff = ReverseConvertiblePayoffDefinition(**data["payoff"])
        
        issue_date = date.fromisoformat(data["issue_date"]) if data.get("issue_date") else None
        maturity_date = (
            date.fromisoformat(data["maturity_date"]) if data.get("maturity_date") else None
        )
        taxonomy = ProductTaxonomy.from_dict(data["taxonomy"]) if data.get("taxonomy") else None

        return cls(
            product_id=data["product_id"],
            currency=data["currency"],
            notional=data["notional"],
            basket=basket,
            payoff=payoff,
            maturity=data["maturity"],
            issue_date=issue_date,
            maturity_date=maturity_date,
            taxonomy=taxonomy,
        )

    def has_barrier(self) -> bool:
        """Check if product has barrier feature."""
        return self.payoff.barrier is not None
    
    def get_coupon_payment_times(self) -> list[float]:
        """Get list of coupon payment times."""
        freq_map = {
            "quarterly": 4,
            "semi-annual": 2,
            "annual": 1,
            "at_maturity": 0,
        }
        
        payments_per_year = freq_map.get(self.payoff.coupon_frequency, 4)
        
        if payments_per_year == 0:
            return [self.maturity]
        
        n_payments = int(self.maturity * payments_per_year)
        return [i / payments_per_year for i in range(1, n_payments + 1)]

    def get_evaluator(self):
        """Get payoff evaluator for pricing engine.
        
        Returns:
            ReverseConvertibleEvaluator configured for this product
        """
        from .payoffs.evaluators import ReverseConvertibleEvaluator
        
        return ReverseConvertibleEvaluator(
            observation_times=self.get_coupon_payment_times(),
            strike=self.payoff.strike,
            coupon_rate=self.payoff.coupon_rate / self._get_payments_per_year(),
            notional=self.notional,
            barrier=self.payoff.barrier,
            barrier_type=self.payoff.barrier_type,
        )
    
    def _get_payments_per_year(self) -> int:
        """Get number of coupon payments per year."""
        freq_map = {
            "quarterly": 4,
            "semi-annual": 2,
            "annual": 1,
            "at_maturity": 1,
        }
        return freq_map.get(self.payoff.coupon_frequency, 4)
