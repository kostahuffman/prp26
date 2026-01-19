"""Base classes for compositional payoff system.

This enables JSON-driven product construction where payoffs are assembled
from modular components rather than hardcoded product classes.

Example:
    {
      "payoff_components": [
        {"type": "memory_coupon", "rate": 0.08, "barrier": 0.7},
        {"type": "autocall", "barrier": 0.95},
        {"type": "worst_of_put", "strike": 1.0}
      ]
    }
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class PayoffState:
    """Mutable state for path-dependent payoffs."""

    def __init__(self):
        self.data: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get state variable."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set state variable."""
        self.data[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Serialize state."""
        return self.data.copy()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PayoffState":
        """Deserialize state."""
        state = cls()
        state.data = data.copy()
        return state


class PayoffComponent(ABC):
    """Abstract base class for payoff components.

    Each component represents a piece of the total payoff:
    - Coupon payments
    - Early termination (autocall)
    - Downside protection/participation
    - Barriers
    - etc.

    Components are evaluated sequentially at each observation date.
    """

    @abstractmethod
    def evaluate(
        self, spots: np.ndarray, initial_spots: np.ndarray, time: float, state: PayoffState
    ) -> dict[str, Any]:
        """Evaluate component at observation time.

        Args:
            spots: Current spot prices (n_paths, n_assets)
            initial_spots: Initial spot prices (n_assets,)
            time: Current time
            state: Mutable payoff state

        Returns:
            Dict with keys:
                - "cashflow": Payment at this time (n_paths,)
                - "terminated": Boolean array of terminated paths (n_paths,)
                - "continue": Whether to continue evaluating (scalar bool)
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize component to dictionary."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "PayoffComponent":
        """Deserialize component from dictionary."""
        ...


class ComposablePayoff:
    """Container for multiple payoff components.

    This is the main orchestrator that:
    1. Evaluates components in sequence at each observation
    2. Aggregates cash flows
    3. Tracks early termination
    4. Manages state for path-dependent features

    Usage:
        payoff = ComposablePayoff([
            MemoryCoupon(rate=0.08, barrier=0.7),
            AutocallComponent(barrier=0.95),
            WorstOfPut(strike=1.0)
        ])

        result = payoff.evaluate_path(paths, times, initial_spots)
    """

    def __init__(self, components: list[PayoffComponent]):
        """Initialize with list of components.

        Args:
            components: Ordered list of payoff components
        """
        self.components = components

    def evaluate_path(
        self,
        paths: np.ndarray,
        times: np.ndarray,
        initial_spots: np.ndarray,
        notional: float = 1.0,
    ) -> dict[str, Any]:
        """Evaluate payoff along Monte Carlo paths.

        Args:
            paths: Asset paths (n_paths, n_steps, n_assets)
            times: Time grid (n_steps,)
            initial_spots: Initial spots (n_assets,)
            notional: Contract notional

        Returns:
            Dict with:
                - "payoffs": Final payoff per path (n_paths,)
                - "cashflows": Cashflows at each time (n_paths, n_steps)
                - "termination_times": When each path terminated (n_paths,)
                - "state": Final payoff state
        """
        n_paths, n_steps, n_assets = paths.shape

        # Initialize
        cashflows = np.zeros((n_paths, n_steps))
        terminated = np.zeros(n_paths, dtype=bool)
        termination_times = np.full(n_paths, times[-1])

        # State for path-dependent features
        state = PayoffState()

        # Evaluate at each observation date
        for step in range(n_steps):
            if np.all(terminated):
                break

            current_spots = paths[:, step, :]
            current_time = times[step]

            # Evaluate each component
            for component in self.components:
                result = component.evaluate(
                    spots=current_spots, initial_spots=initial_spots, time=current_time, state=state
                )

                # Add cash flows (only for non-terminated paths)
                cashflow = result.get("cashflow", np.zeros(n_paths))
                cashflows[~terminated, step] += cashflow[~terminated]

                # Mark newly terminated paths
                newly_terminated = result.get("terminated", np.zeros(n_paths, dtype=bool))
                new_terminations = newly_terminated & ~terminated
                if np.any(new_terminations):
                    terminated[new_terminations] = True
                    termination_times[new_terminations] = current_time

                # Check if we should stop evaluating
                if not result.get("continue", True):
                    break

        # Final payoffs are sum of discounted cashflows (discounting done externally)
        final_payoffs = np.sum(cashflows, axis=1) * notional

        return {
            "payoffs": final_payoffs,
            "cashflows": cashflows * notional,
            "termination_times": termination_times,
            "terminated": terminated,
            "state": state,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"components": [comp.to_dict() for comp in self.components]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComposablePayoff":
        """Deserialize from dictionary.

        Uses PayoffRegistry to construct components by type.
        """
        from .registry import PayoffRegistry

        components = []
        for comp_data in data["components"]:
            comp_type = comp_data["type"]
            component = PayoffRegistry.create(comp_type, comp_data)
            components.append(component)

        return cls(components=components)
