"""Compositional payoff system for flexible product construction."""

from .base import PayoffComponent, ComposablePayoff
from .coupon import (
    CouponComponent,
    MemoryCoupon,
    SnowballCoupon,
    StepUpCoupon,
    StepDownCoupon,
)
from .autocall import AutocallComponent
from .downside import DownsideComponent, WorstOfPut
from .rainbow import (
    RainbowComponent,
    RainbowCall,
    RainbowPut,
    RainbowDigital,
    SpreadOption,
)
from .registry import PayoffRegistry

# Import evaluators from subfolder
from .evaluators import (
    PayoffEvaluator,
    PhoenixPayoffEvaluator,
    ReverseConvertibleEvaluator,
    SnowballPayoffEvaluator,
    SnowballState,
)

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
    "PayoffRegistry",
    # Evaluators
    "PayoffEvaluator",
    "PhoenixPayoffEvaluator",
    "ReverseConvertibleEvaluator",
    "SnowballPayoffEvaluator",
    "SnowballState",
]
