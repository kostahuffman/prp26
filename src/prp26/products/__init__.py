"""product definitions."""

from .base import StructuredProduct, Basket, Underlying
from .autocallable import AutocallableProduct
from .reverse_convertible import ReverseConvertible, ReverseConvertiblePayoffDefinition
from .participation import (
    BonusCertificate,
    BonusCertificatePayoff,
    TrackerCertificate,
    TrackerPayoff,
)
from .taxonomy import ProductTaxonomy
from .schedules import ObservationSchedule, BarrierSchedule, CouponDefinition
from .payoffs import SnowballState, SnowballPayoffEvaluator, PhoenixPayoffEvaluator

__all__ = [
    "StructuredProduct",
    "Basket",
    "Underlying",
    "AutocallableProduct",
    "ReverseConvertible",
    "ReverseConvertiblePayoffDefinition",
    "BonusCertificate",
    "BonusCertificatePayoff",
    "TrackerCertificate",
    "TrackerPayoff",
    "ProductTaxonomy",
    "ObservationSchedule",
    "BarrierSchedule",
    "CouponDefinition",
    "SnowballState",
    "SnowballPayoffEvaluator",
    "PhoenixPayoffEvaluator",
]
