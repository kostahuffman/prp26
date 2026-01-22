"""Spot price providers."""

from typing import Any


class SpotProvider:
    """Simple spot price provider.

    In production, this would connect to market data feeds (Bloomberg, Refinitiv, etc.).
    For now, it's a simple dictionary-based implementation.
    """

    def __init__(self, spots: dict[str, float]):
        """Initialize with spot prices.

        Args:
            spots: Dict mapping ticker -> spot price
        """
        self.spots = spots

    def get_spot(self, ticker: str) -> float:
        """Get spot price for ticker."""
        if ticker not in self.spots:
            raise ValueError(f"Spot price not available for ticker: {ticker}")
        return self.spots[ticker]

    def update_spot(self, ticker: str, price: float) -> None:
        """Update spot price (for scenario analysis)."""
        self.spots[ticker] = price

    def bump_spot(self, ticker: str, bump_pct: float) -> float:
        """Bump spot price by percentage and return new value.

        Args:
            ticker: The ticker to bump
            bump_pct: Percentage bump (0.01 = 1%)

        Returns:
            New spot price after bump
        """
        old_price = self.get_spot(ticker)
        new_price = old_price * (1.0 + bump_pct)
        self.spots[ticker] = new_price
        return new_price

    def restore_spot(self, ticker: str, original_price: float) -> None:
        """Restore spot to original value (after bump)."""
        self.spots[ticker] = original_price

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"spots": self.spots}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpotProvider":
        """Deserialize from dictionary."""
        return cls(spots=data["spots"])
