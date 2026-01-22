"""
Bonus Certificate and other participation products.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import json

from .base import Basket, StructuredProduct, Underlying
from .taxonomy import TAXONOMY_BONUS_CERTIFICATE, TAXONOMY_TRACKER, ProductTaxonomy


@dataclass
class BonusCertificatePayoff:
    """Definition of bonus certificate payoff structure."""

    payoff_type: str = "bonus_certificate"
    participation: float = 1.0  # Typically 1.0 for 1:1 participation
    bonus_level: float = 1.20  # Bonus level (e.g., 120% of initial)
    barrier: float = 0.70  # Downside barrier (e.g., 70% of initial)
    barrier_type: str = "american"  # "european", "american", "bermudan"
    cap: float | None = None  # Optional cap on upside

    def to_dict(self):
        return asdict(self)


class BonusCertificate(StructuredProduct):
    """

    Features:
    - 1:1 participation in underlying performance
    - Bonus minimum return if barrier never breached
    - Full downside if barrier is breached
    - Optional cap on upside

    Payoff at Maturity:
        If barrier never breached:
            Notional * max(Final Price / Initial, Bonus Level)
        Else (barrier breached):
            Notional * (Final Price / Initial)

    Example:
        - Initial: 100
        - Bonus: 120%
        - Barrier: 70%
        - Final: 95

        If barrier never breached: 120 (get the bonus)
        If barrier breached: 95 (normal participation)
    """

    def __init__(
        self,
        product_id: str,
        currency: str,
        notional: float,
        basket: Basket,
        payoff: BonusCertificatePayoff,
        maturity: float,
        issue_date: date | None = None,
        maturity_date: date | None = None,
        taxonomy: ProductTaxonomy | None = None,
    ):
        if taxonomy is None:
            taxonomy = TAXONOMY_BONUS_CERTIFICATE

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

        if not (0 < self.payoff.barrier <= 1):
            raise ValueError("Barrier must be between 0 and 1")

        if self.payoff.bonus_level <= 1.0:
            raise ValueError("Bonus level should be > 1.0 (e.g., 1.20 for 120%)")

        if self.payoff.cap is not None and self.payoff.cap <= self.payoff.bonus_level:
            raise ValueError("Cap should be greater than bonus level")

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
    def from_json(cls, json_str: str) -> BonusCertificate:
        """Deserialize from JSON."""
        data = json.loads(json_str)

        basket = Basket(
            underlyings=[Underlying(**u) for u in data["basket"]["underlyings"]],
            weights=data["basket"]["weights"],
            worst_of=data["basket"]["worst_of"],
            best_of=data["basket"].get("best_of", False),
        )

        payoff = BonusCertificatePayoff(**data["payoff"])

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

    def has_cap(self) -> bool:
        """Check if product has upside cap."""
        return self.payoff.cap is not None


@dataclass
class TrackerPayoff:
    """Definition of tracker certificate payoff."""

    payoff_type: str = "tracker"
    participation: float = 1.0  # Typically 1.0 for pure tracking
    fees: float = 0.0  # Annual management fee

    def to_dict(self):
        return asdict(self)


class TrackerCertificate(StructuredProduct):
    """
    Tracker Certificate (Delta-1 Product).

    Features:
    - Pure 1:1 exposure to underlying basket
    - No barriers, no caps
    - Used for thematic/sector exposure
    - Low fees

    Payoff at Maturity:
        Notional * (Final Price / Initial) * (1 - fees * maturity)
    """

    def __init__(
        self,
        product_id: str,
        currency: str,
        notional: float,
        basket: Basket,
        payoff: TrackerPayoff,
        maturity: float,
        issue_date: date | None = None,
        maturity_date: date | None = None,
        taxonomy: ProductTaxonomy | None = None,
    ):
        if taxonomy is None:
            taxonomy = TAXONOMY_TRACKER

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

        if self.payoff.fees < 0:
            raise ValueError("Fees cannot be negative")

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
    def from_json(cls, json_str: str) -> TrackerCertificate:
        """Deserialize from JSON."""
        data = json.loads(json_str)

        basket = Basket(
            underlyings=[Underlying(**u) for u in data["basket"]["underlyings"]],
            weights=data["basket"]["weights"],
            worst_of=data["basket"]["worst_of"],
            best_of=data["basket"].get("best_of", False),
        )

        payoff = TrackerPayoff(**data["payoff"])

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
