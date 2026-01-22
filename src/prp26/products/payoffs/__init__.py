"""Compositional payoff system for flexible product construction."""

from .autocall import AutocallComponent
from .base import ComposablePayoff, PayoffComponent
from .coupon import (
    CouponComponent,
    MemoryCoupon,
    SnowballCoupon,
    StepDownCoupon,
    StepUpCoupon,
)
from .downside import DownsideComponent, WorstOfPut

# Import evaluators from subfolder
from .evaluators import (
    PayoffEvaluator,
    PhoenixPayoffEvaluator,
    ReverseConvertibleEvaluator,
    SnowballPayoffEvaluator,
    SnowballState,
)
from .options import AmericanOption, AsianOption, BermudanOption, EuropeanOption
from .rainbow import (
    RainbowCall,
    RainbowComponent,
    RainbowDigital,
    RainbowPut,
    SpreadOption,
)
from .registry import PayoffRegistry

__all__ = [
    # Components
    "PayoffComponent",
    "ComposablePayoff",
    "CouponComponent",
    "MemoryCoupon",
    "SnowballCoupon",
    "StepUpCoupon",
    "StepDownCoupon",
    "AutocallComponent",
    "DownsideComponent",
    "WorstOfPut",
    "RainbowComponent",
    "RainbowCall",
    "RainbowPut",
    "RainbowDigital",
    "SpreadOption",
    # Option components
    "EuropeanOption",
    "AmericanOption",
    "AsianOption",
    "BermudanOption",
    "PayoffRegistry",
    # Evaluators
    "PayoffEvaluator",
    "PhoenixPayoffEvaluator",
    "ReverseConvertibleEvaluator",
    "SnowballPayoffEvaluator",
    "SnowballState",
]
