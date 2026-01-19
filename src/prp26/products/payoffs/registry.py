"""Registry for payoff component types.

This enables JSON-driven product construction by mapping
component type strings to their classes.
"""

from typing import Any

from .base import PayoffComponent


class PayoffRegistry:
    """Registry for payoff component types."""

    _registry: dict[str, type[PayoffComponent]] = {}

    @classmethod
    def register(cls, component_type: str, component_class: type[PayoffComponent]) -> None:
        """Register a payoff component type.

        Args:
            component_type: Type string (e.g., "memory_coupon")
            component_class: Component class
        """
        cls._registry[component_type] = component_class

    @classmethod
    def create(cls, component_type: str, data: dict[str, Any]) -> PayoffComponent:
        """Create payoff component from type and data.

        Args:
            component_type: Type string
            data: Component configuration

        Returns:
            PayoffComponent instance

        Raises:
            ValueError: If component type not registered
        """
        if component_type not in cls._registry:
            raise ValueError(f"Unknown payoff component type: {component_type}")

        component_class = cls._registry[component_type]
        return component_class.from_dict(data)

    @classmethod
    def list_types(cls) -> list:
        """List all registered component types."""
        return list(cls._registry.keys())


# Register built-in components
def _register_builtin_components():
    """Register all built-in payoff components."""
    from .autocall import AutocallComponent
    from .coupon import (
        CouponComponent,
        MemoryCoupon,
        SnowballCoupon,
        StepUpCoupon,
        StepDownCoupon,
    )
    from .downside import DownsideComponent, WorstOfPut
    from .rainbow import RainbowCall, RainbowPut, RainbowDigital, SpreadOption

    # Coupon components
    PayoffRegistry.register("simple_coupon", CouponComponent)
    PayoffRegistry.register("memory_coupon", MemoryCoupon)
    PayoffRegistry.register("snowball_coupon", SnowballCoupon)
    PayoffRegistry.register("step_up_coupon", StepUpCoupon)
    PayoffRegistry.register("step_down_coupon", StepDownCoupon)

    # Autocall components
    PayoffRegistry.register("autocall", AutocallComponent)

    # Downside components
    PayoffRegistry.register("downside", DownsideComponent)
    PayoffRegistry.register("worst_of_put", WorstOfPut)

    # Rainbow components
    PayoffRegistry.register("rainbow_call", RainbowCall)
    PayoffRegistry.register("rainbow_put", RainbowPut)
    PayoffRegistry.register("rainbow_digital", RainbowDigital)
    PayoffRegistry.register("spread_option", SpreadOption)


# Auto-register on module import
_register_builtin_components()
