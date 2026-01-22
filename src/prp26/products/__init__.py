"""product definitions."""

from .autocallable import AutocallableProduct
from .base import Basket, StructuredProduct, Underlying
from .participation import (
    BonusCertificate,
    BonusCertificatePayoff,
    TrackerCertificate,
    TrackerPayoff,
)
from .payoffs import PhoenixPayoffEvaluator, SnowballPayoffEvaluator, SnowballState
from .reverse_convertible import ReverseConvertible, ReverseConvertiblePayoffDefinition
from .schedules import BarrierSchedule, CouponDefinition, ObservationSchedule
from .taxonomy import ProductTaxonomy
from .vanilla_option import VanillaOption

__all__ = [
    "StructuredProduct",
    "Basket",
    "Underlying",
    "AutocallableProduct",
    "ReverseConvertible",
    "ReverseConvertiblePayoffDefinition",
    "VanillaOption",
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
