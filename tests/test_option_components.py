"""Tests for option payoff components (European, American, Asian, Bermudan)."""

import numpy as np
import pytest

from prp26.products.payoffs import (
    AmericanOption,
    AsianOption,
    AutocallComponent,
    BermudanOption,
    ComposablePayoff,
    EuropeanOption,
)
from prp26.products.payoffs.base import PayoffState


def test_european_option_call():
    """Test European call option component."""
    option = EuropeanOption(strike=1.0, option_type="call", participation=1.0)

    # Setup
    n_paths = 100
    initial_spots = np.array([100.0])
    state = PayoffState()

    # Before maturity - should return zero cashflow
    spots_before = np.full((n_paths, 1), 110.0)  # 10% above strike
    state.set("is_final_observation", False)

    result = option.evaluate(spots_before, initial_spots, time=0.5, state=state)

    assert np.all(result["cashflow"] == 0.0), "Should have no cashflow before maturity"
    assert not np.any(result["terminated"]), "Should not terminate"

    # At maturity - should pay off
    spots_final = np.full((n_paths, 1), 110.0)  # 10% above strike
    state.set("is_final_observation", True)

    result = option.evaluate(spots_final, initial_spots, time=1.0, state=state)

    # Expected: max(110/100 - 1.0, 0) = 0.1
    expected_payoff = 0.1
    assert np.allclose(result["cashflow"], expected_payoff), f"Expected {expected_payoff}, got {result['cashflow']}"


def test_european_option_put():
    """Test European put option component."""
    option = EuropeanOption(strike=1.0, option_type="put", participation=1.0)

    # Setup
    n_paths = 100
    initial_spots = np.array([100.0])
    state = PayoffState()

    # At maturity - spots below strike
    spots_final = np.full((n_paths, 1), 90.0)  # 10% below strike
    state.set("is_final_observation", True)

    result = option.evaluate(spots_final, initial_spots, time=1.0, state=state)

    # Expected: max(1.0 - 90/100, 0) = 0.1
    expected_payoff = 0.1
    assert np.allclose(result["cashflow"], expected_payoff), f"Expected {expected_payoff}, got {result['cashflow']}"


def test_asian_option_average_price():
    """Test Asian option with average price."""
    option = AsianOption(
        strike=1.0,
        option_type="call",
        asian_type="average_price",
        participation=1.0,
    )

    # Setup
    n_paths = 100
    initial_spots = np.array([100.0])
    state = PayoffState()

    # Observations at t=0.25, 0.5, 0.75, 1.0
    # Spots: 105, 110, 100, 105
    # Average: (105 + 110 + 100 + 105) / 4 = 105 / 100 = 1.05

    observations = [
        (0.25, 105.0, False),
        (0.5, 110.0, False),
        (0.75, 100.0, False),
        (1.0, 105.0, True),
    ]

    for time, spot_value, is_final in observations:
        spots = np.full((n_paths, 1), spot_value)
        state.set("is_final_observation", is_final)

        result = option.evaluate(spots, initial_spots, time, state)

        if not is_final:
            assert np.all(result["cashflow"] == 0.0), f"Should have no cashflow at t={time}"

    # At final observation, should pay based on average
    # Average performance = 1.05, strike = 1.0
    # Payoff = max(1.05 - 1.0, 0) = 0.05
    expected_payoff = 0.05
    assert np.allclose(
        result["cashflow"], expected_payoff, atol=1e-6
    ), f"Expected {expected_payoff}, got {result['cashflow']}"


def test_asian_option_average_strike():
    """Test Asian option with average strike."""
    option = AsianOption(
        strike=1.0,  # Not used for average_strike
        option_type="call",
        asian_type="average_strike",
        participation=1.0,
    )

    # Setup
    n_paths = 100
    initial_spots = np.array([100.0])
    state = PayoffState()

    # Observations: 105, 100, 95, 110
    # Average: 102.5 / 100 = 1.025
    # Final: 110 / 100 = 1.1
    # Payoff: max(1.1 - 1.025, 0) = 0.075

    observations = [
        (0.25, 105.0, False),
        (0.5, 100.0, False),
        (0.75, 95.0, False),
        (1.0, 110.0, True),
    ]

    for time, spot_value, is_final in observations:
        spots = np.full((n_paths, 1), spot_value)
        state.set("is_final_observation", is_final)

        result = option.evaluate(spots, initial_spots, time, state)

    expected_payoff = 0.075
    assert np.allclose(
        result["cashflow"], expected_payoff, atol=1e-6
    ), f"Expected {expected_payoff}, got {result['cashflow']}"


def test_bermudan_option():
    """Test Bermudan option with specific exercise dates."""
    # Exercise allowed at t=0.5 and t=1.0
    option = BermudanOption(
        strike=1.0,
        exercise_times=[0.5, 1.0],
        option_type="call",
        participation=1.0,
    )

    # Setup
    n_paths = 100
    initial_spots = np.array([100.0])
    state = PayoffState()

    # Not an exercise date - should return zero
    spots_t025 = np.full((n_paths, 1), 110.0)
    result = option.evaluate(spots_t025, initial_spots, time=0.25, state=state)
    assert np.all(result["cashflow"] == 0.0), "Should have no cashflow at non-exercise date"

    # Exercise date - should calculate intrinsic value
    spots_t05 = np.full((n_paths, 1), 110.0)
    result = option.evaluate(spots_t05, initial_spots, time=0.5, state=state)

    # Should store intrinsic value in state
    intrinsic = state.get("bermudan_option_intrinsic")
    expected_intrinsic = 0.1  # max(1.1 - 1.0, 0)
    assert intrinsic is not None, "Intrinsic value should be stored in state"
    assert np.allclose(intrinsic, expected_intrinsic), f"Expected {expected_intrinsic}, got {intrinsic}"


def test_american_option():
    """Test American option component."""
    option = AmericanOption(strike=1.0, option_type="call", participation=1.0)

    # Setup
    n_paths = 100
    initial_spots = np.array([100.0])
    state = PayoffState()

    # At any observation, should store intrinsic value
    spots = np.full((n_paths, 1), 110.0)
    result = option.evaluate(spots, initial_spots, time=0.5, state=state)

    # Should store intrinsic value in state
    intrinsic = state.get("american_option_intrinsic")
    expected_intrinsic = 0.1  # max(1.1 - 1.0, 0)
    assert intrinsic is not None, "Intrinsic value should be stored in state"
    assert np.allclose(intrinsic, expected_intrinsic), f"Expected {expected_intrinsic}, got {intrinsic}"


def test_option_with_autocall():
    """Test that options respect autocall termination."""
    # Create a payoff with autocall + European option
    payoff = ComposablePayoff(
        components=[
            AutocallComponent(barrier=1.05, redemption=1.0),
            EuropeanOption(strike=1.0, option_type="call", participation=1.0),
        ]
    )

    # Setup paths that trigger autocall
    n_paths = 10
    n_steps = 2
    n_assets = 1

    # Paths: first observation above 1.05, second at 1.1
    paths = np.zeros((n_paths, n_steps, n_assets))
    paths[:, 0, 0] = 106.0  # Above autocall barrier
    paths[:, 1, 0] = 110.0

    times = np.array([0.5, 1.0])
    initial_spots = np.array([100.0])

    result = payoff.evaluate_path(paths, times, initial_spots, notional=1.0)

    # All paths should terminate at first observation
    assert np.all(result["terminated"]), "All paths should have terminated from autocall"
    assert np.all(result["termination_times"] == 0.5), "Should terminate at first observation"

    # Cashflow should be from autocall (redemption = 1.0), not from option
    # Option should not pay because paths are terminated
    assert np.allclose(result["cashflows"][:, 0], 1.0), "Should get autocall redemption"
    assert np.allclose(result["cashflows"][:, 1], 0.0), "Should have no cashflow at second observation"


def test_composable_payoff_with_multiple_options():
    """Test composable payoff with multiple option components."""
    # Create a custom structure: European call + Asian put
    payoff = ComposablePayoff(
        components=[
            EuropeanOption(strike=1.0, option_type="call", participation=1.0),
            AsianOption(strike=1.0, option_type="put", asian_type="average_price", participation=0.5),
        ]
    )

    # Setup paths
    n_paths = 10
    n_steps = 2
    n_assets = 1

    # Paths that are up at maturity but down on average
    paths = np.zeros((n_paths, n_steps, n_assets))
    paths[:, 0, 0] = 95.0  # Down
    paths[:, 1, 0] = 110.0  # Up

    times = np.array([0.5, 1.0])
    initial_spots = np.array([100.0])

    result = payoff.evaluate_path(paths, times, initial_spots, notional=1.0)

    # European call should pay: max(1.1 - 1.0, 0) = 0.1
    # Asian put: average = (0.95 + 1.1) / 2 = 1.025, payoff = max(1.0 - 1.025, 0) * 0.5 = 0
    # Total = 0.1
    expected_payoff = 0.1

    assert np.allclose(
        result["payoffs"], expected_payoff, atol=1e-6
    ), f"Expected {expected_payoff}, got {result['payoffs']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
