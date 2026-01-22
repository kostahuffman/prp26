"""
Product taxonomy following EUSIPA/SSPA standards.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProductTaxonomy:
    """EUSIPA/SSPA-style product classification.

    This provides a hierarchical classification system similar to GICS
    for s. products.

    Levels:
        - Level 1: Investment Class (Investment vs Leverage)
        - Level 2: Category (Yield Enhancement, Participation, Capital Protection)
        - Level 3: Product Type (Autocallable, Reverse Convertible, etc.)
        - Level 4: Subtype/Features (Phoenix, Memory, Step-Down, Worst-Of, etc.)
    """

    investment_class: str  # "Investment" or "Leverage"
    category: str  # "Yield Enhancement", "Participation", "Capital Protection"
    product_type: str  # "Autocallable", "Reverse Convertible", "Bonus Certificate", etc.
    subtype: str  # "Phoenix", "Memory", "Step-Down", "Worst-Of", etc.

    def to_dict(self):
        """Serialize to dictionary."""
        return {
            "class": self.investment_class,
            "category": self.category,
            "type": self.product_type,
            "subtype": self.subtype,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ProductTaxonomy:
        """Deserialize from dictionary."""
        return cls(
            investment_class=data["class"],
            category=data["category"],
            product_type=data["type"],
            subtype=data["subtype"],
        )


# Common taxonomy definitions
TAXONOMY_PHOENIX_AUTOCALLABLE = ProductTaxonomy(
    investment_class="Investment",
    category="Yield Enhancement",
    product_type="Autocallable",
    subtype="Phoenix Memory",
)

TAXONOMY_REVERSE_CONVERTIBLE = ProductTaxonomy(
    investment_class="Investment",
    category="Yield Enhancement",
    product_type="Reverse Convertible",
    subtype="Standard",
)

TAXONOMY_BARRIER_REVERSE_CONVERTIBLE = ProductTaxonomy(
    investment_class="Investment",
    category="Yield Enhancement",
    product_type="Reverse Convertible",
    subtype="Barrier",
)

TAXONOMY_BONUS_CERTIFICATE = ProductTaxonomy(
    investment_class="Investment",
    category="Participation",
    product_type="Bonus Certificate",
    subtype="Standard",
)

TAXONOMY_TRACKER = ProductTaxonomy(
    investment_class="Investment",
    category="Participation",
    product_type="Tracker",
    subtype="1:1 Delta",
)
