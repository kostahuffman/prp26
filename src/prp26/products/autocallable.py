"""
Autocallable product definitions (Phoenix, etc.).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import json

from .base import Basket, StructuredProduct, Underlying
from .schedules import BarrierSchedule, CouponDefinition, ObservationSchedule
from .taxonomy import ProductTaxonomy


@dataclass
class PayoffDefinition:
    """Definition of product payoff structure."""

    payoff_type: str = "autocallable"  # autocallable, barrier_reverse_convertible, etc.
    participation: float = 1.0
    downside_protection: float | None = None  # Level of capital protection

    def to_dict(self):
        return asdict(self)


class AutocallableProduct(StructuredProduct):
    """
    Autocallable product (Phoenix, Memory Phoenix, etc.).

    Features:
    - Autocall on specified observation dates if basket above barrier
    - Memory coupons (optional)
    - Downside protection (optional)
    - Worst-of basket support
    """

    def __init__(
        self,
        product_id: str,
        currency: str,
        notional: float,
        basket: Basket,
        observation_schedule: ObservationSchedule,
        barrier_schedule: BarrierSchedule,
        coupon: CouponDefinition,
        maturity: float,
        payoff_definition: PayoffDefinition | None = None,
        issue_date: date | None = None,
        maturity_date: date | None = None,
        taxonomy: ProductTaxonomy | None = None,
    ):
        super().__init__(product_id, currency, notional, issue_date, maturity_date, taxonomy)
        self.basket = basket
        self.observation_schedule = observation_schedule
        self.barrier_schedule = barrier_schedule
        self.coupon = coupon
        self.maturity = maturity
        self.payoff_definition = payoff_definition or PayoffDefinition()

        self.validate()

    def validate(self) -> bool:
        """Validate product definition."""
        super().validate()

        if self.maturity <= 0:
            raise ValueError("Maturity must be positive")

        # Check that barrier schedule aligns with observation schedule
        for obs_time in self.observation_schedule.times:
            if obs_time not in self.barrier_schedule.levels:
                raise ValueError(f"No barrier level defined for observation time {obs_time}")

        return True

    def to_json(self) -> str:
        """Serialize to JSON."""
        data = {
            "product_id": self.product_id,
            "currency": self.currency,
            "notional": self.notional,
            "taxonomy": self.taxonomy.to_dict() if self.taxonomy else None,
            "basket": self.basket.to_dict(),
            "observation_schedule": self.observation_schedule.to_dict(),
            "barrier_schedule": self.barrier_schedule.to_dict(),
            "coupon": self.coupon.to_dict(),
            "maturity": self.maturity,
            "payoff_definition": self.payoff_definition.to_dict(),
            "issue_date": self.issue_date.isoformat() if self.issue_date else None,
            "maturity_date": self.maturity_date.isoformat() if self.maturity_date else None,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> AutocallableProduct:
        """Deserialize from JSON."""
        data = json.loads(json_str)

        basket = Basket(
            underlyings=[Underlying(**u) for u in data["basket"]["underlyings"]],
            weights=data["basket"]["weights"],
            worst_of=data["basket"]["worst_of"],
            best_of=data["basket"].get("best_of", False),
        )

        observation_schedule = ObservationSchedule.from_dict(data["observation_schedule"])
        barrier_schedule = BarrierSchedule.from_dict(data["barrier_schedule"])
        coupon = CouponDefinition.from_dict(data["coupon"])
        payoff_definition = PayoffDefinition(**data["payoff_definition"])

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
            observation_schedule=observation_schedule,
            barrier_schedule=barrier_schedule,
            coupon=coupon,
            maturity=data["maturity"],
            payoff_definition=payoff_definition,
            issue_date=issue_date,
            maturity_date=maturity_date,
            taxonomy=taxonomy,
        )

    def get_callability_dates(self) -> list[float]:
        """Get list of autocall observation dates."""
        return self.observation_schedule.times

    def is_memory(self) -> bool:
        """Check if product has memory feature."""
        return self.coupon.memory

    def is_snowball(self) -> bool:
        """Check if product has snowball feature."""
        return self.coupon.snowball

    def get_evaluator(self):
        """Get payoff evaluator for pricing engine.

        Returns:
            PhoenixPayoffEvaluator or SnowballPayoffEvaluator based on product features
        """
        observation_times = self.observation_schedule.times

        # Extract autocall barriers (use barrier schedule)
        autocall_barriers = [self.barrier_schedule.levels[t] for t in observation_times]

        # Coupon barriers - use 70% if memory coupon, else same as autocall
        if self.coupon.memory:
            coupon_barriers = [0.70] * len(observation_times)  # Phoenix typically 70%
        else:
            coupon_barriers = autocall_barriers

        coupon_rate = self.coupon.rate

        # Use Snowball evaluator if snowball feature is enabled
        if self.is_snowball():
            from .payoffs.evaluators import SnowballPayoffEvaluator

            return SnowballPayoffEvaluator(
                observation_times=observation_times,
                barrier_levels=autocall_barriers,
                coupon_rate=coupon_rate,
                notional=self.notional,
                growth_rate=0.0,  # Can be parameterized if needed
            )
        else:
            # Use Phoenix evaluator for memory/standard autocallables
            from .payoffs.evaluators import PhoenixPayoffEvaluator

            return PhoenixPayoffEvaluator(
                observation_times=observation_times,
                autocall_barriers=autocall_barriers,
                coupon_barriers=coupon_barriers,
                coupon_rate=coupon_rate,
                notional=self.notional,
            )
