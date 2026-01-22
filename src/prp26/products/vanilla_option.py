import json
from datetime import datetime

from prp26.products import Basket, StructuredProduct, Underlying
from prp26.products.schedules import BarrierSchedule, CouponDefinition, ObservationSchedule


class VanillaOption(StructuredProduct):
    """Simple vanilla European option.

    This is a minimal implementation to demonstrate the engine
    can price simple products, not just complex autocallables.
    """

    def __init__(
        self,
        product_id: str,
        currency: str,
        notional: float,
        underlying: Underlying,
        strike: float,
        maturity: float,
        option_type: str = "call",  # "call" or "put"
        issue_date: datetime = None,
        maturity_date: datetime = None,
    ):
        super().__init__(
            product_id=product_id,
            currency=currency,
            notional=notional,
            issue_date=issue_date,
            maturity_date=maturity_date,
        )
        self.underlying = underlying
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type.lower()

        # Create basket with single underlying
        self.basket = Basket(
            underlyings=[underlying],
            weights=[1.0],
            worst_of=False,
        )

        # Create observation schedule (just maturity)
        self.observation_schedule = ObservationSchedule(times=[maturity])

        # Barrier schedule (impossible barrier so it never autocalls)
        self.barrier_schedule = BarrierSchedule(levels={maturity: 10.0})

        # Coupon (dummy - not used for vanilla)
        self.coupon = CouponDefinition(rate=0.0, memory=False)

    def get_evaluator(self):
        """Get payoff evaluator for pricing engine.

        Returns:
            VanillaOptionEvaluator configured for this product
        """
        from .payoffs.evaluators import VanillaOptionEvaluator

        return VanillaOptionEvaluator(
            maturity=self.maturity,
            strike=self.strike,
            option_type=self.option_type,
            notional=self.notional,
        )

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(
            {
                "product_id": self.product_id,
                "currency": self.currency,
                "notional": self.notional,
                "underlying": self.underlying.symbol,
                "strike": self.strike,
                "maturity": self.maturity,
                "option_type": self.option_type,
            }
        )
