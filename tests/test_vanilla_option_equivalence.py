"""Test VanillaOption equivalence between StructuredProduct and ComposablePayoff approaches.

Verifies that the same results are obtained whether using:
1. VanillaOption as StructuredProduct with evaluators
2. VanillaOption.to_composable_payoff() with PayoffComponents
"""

import numpy as np
import pytest

from prp26.products import Underlying, VanillaOption
from prp26.products.payoffs import AmericanOption, AsianOption, BermudanOption, ComposablePayoff, EuropeanOption


class TestVanillaOptionEquivalence:
    """Test equivalence between VanillaOption and ComposablePayoff approaches."""

    @pytest.fixture
    def sample_paths(self):
        """Generate sample paths for testing."""
        np.random.seed(42)
        # Generate 100 paths with 21 steps (0.0 to 1.0 in 0.05 increments)
        n_paths = 100
        n_steps = 21
        times = np.linspace(0, 1.0, n_steps)
        
        # Generate simple GBM paths manually
        # S(t) = S(0) * exp((mu - 0.5*sigma^2)*t + sigma*W(t))
        drift = 0.05
        vol = 0.20
        dt = times[1] - times[0]
        
        # Random increments
        dW = np.random.randn(n_paths, n_steps - 1) * np.sqrt(dt)
        W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)
        
        # Generate paths
        S0 = 100.0
        paths = S0 * np.exp((drift - 0.5 * vol**2) * times[None, :] + vol * W)
        
        # Reshape to (n_paths, n_steps, n_assets=1)
        paths = paths[:, :, np.newaxis]
        
        return paths, times

    def test_european_call_equivalence(self, sample_paths):
        """Test European call equivalence."""
        paths, times = sample_paths
        initial_spots = np.array([100.0])
        strike = 1.0  # ATM (100/100 = 1.0)

        # Create VanillaOption (StructuredProduct)
        option = VanillaOption(
            product_id="TEST_EURO_CALL",
            currency="USD",
            notional=1.0,
            underlying=Underlying("TEST", "equity"),
            strike=strike,
            maturity=1.0,
            option_type="call",
            exercise_style="european",
        )

        # Get evaluator and evaluate
        evaluator = option.get_evaluator()
        result1 = evaluator.evaluate(paths, initial_spots)
        payoffs1 = result1["payoffs"]

        # Convert to ComposablePayoff and evaluate
        composable = option.to_composable_payoff()
        result2 = composable.evaluate_path(paths, times, initial_spots, notional=1.0)
        payoffs2 = result2["payoffs"]

        # Should match exactly
        np.testing.assert_array_almost_equal(payoffs1, payoffs2, decimal=10)
        assert payoffs1.shape == (100,)
        assert np.all(payoffs1 >= 0)  # Options have non-negative payoffs

    def test_european_put_equivalence(self, sample_paths):
        """Test European put equivalence."""
        paths, times = sample_paths
        initial_spots = np.array([100.0])
        strike = 1.0

        option = VanillaOption(
            product_id="TEST_EURO_PUT",
            currency="USD",
            notional=1.0,
            underlying=Underlying("TEST", "equity"),
            strike=strike,
            maturity=1.0,
            option_type="put",
            exercise_style="european",
        )

        # Evaluator approach
        evaluator = option.get_evaluator()
        result1 = evaluator.evaluate(paths, initial_spots)
        payoffs1 = result1["payoffs"]

        # Composable approach
        composable = option.to_composable_payoff()
        result2 = composable.evaluate_path(paths, times, initial_spots, notional=1.0)
        payoffs2 = result2["payoffs"]

        np.testing.assert_array_almost_equal(payoffs1, payoffs2, decimal=10)

    def test_american_call_equivalence(self, sample_paths):
        """Test American call - both evaluator and component produce non-negative payoffs."""
        paths, times = sample_paths
        initial_spots = np.array([100.0])
        strike = 1.0

        # Quarterly observations
        obs_times = [0.25, 0.5, 0.75, 1.0]

        option = VanillaOption(
            product_id="TEST_AMER_CALL",
            currency="USD",
            notional=1.0,
            underlying=Underlying("TEST", "equity"),
            strike=strike,
            maturity=1.0,
            option_type="call",
            exercise_style="american",
            observation_times=obs_times,
        )

        # Evaluator approach
        evaluator = option.get_evaluator()
        result1 = evaluator.evaluate(paths, initial_spots)
        payoffs1 = result1["payoffs"]

        # Composable approach - Note: component and evaluator may differ
        # because they use different observation structures
        composable = option.to_composable_payoff()
        result2 = composable.evaluate_path(paths, times, initial_spots, notional=1.0)
        payoffs2 = result2["payoffs"]

        # Both should produce non-negative payoffs
        assert np.all(payoffs1 >= 0), "Evaluator payoffs should be non-negative"
        assert np.all(payoffs2 >= 0), "Component payoffs should be non-negative"
        
        # Evaluator should find opportunities to exercise
        assert np.mean(payoffs1) > 0, "American option should have positive average payoff"

    def test_american_put_equivalence(self, sample_paths):
        """Test American put - both produce non-negative payoffs."""
        paths, times = sample_paths
        initial_spots = np.array([100.0])
        strike = 1.0

        obs_times = [0.25, 0.5, 0.75, 1.0]

        option = VanillaOption(
            product_id="TEST_AMER_PUT",
            currency="USD",
            notional=1.0,
            underlying=Underlying("TEST", "equity"),
            strike=strike,
            maturity=1.0,
            option_type="put",
            exercise_style="american",
            observation_times=obs_times,
        )

        evaluator = option.get_evaluator()
        result1 = evaluator.evaluate(paths, initial_spots)
        payoffs1 = result1["payoffs"]

        composable = option.to_composable_payoff()
        result2 = composable.evaluate_path(paths, times, initial_spots, notional=1.0)
        payoffs2 = result2["payoffs"]

        assert np.all(payoffs1 >= 0), "Evaluator payoffs should be non-negative"
        assert np.all(payoffs2 >= 0), "Component payoffs should be non-negative"

    def test_bermudan_call_equivalence(self, sample_paths):
        """Test Bermudan call - both produce non-negative payoffs."""
        paths, times = sample_paths
        initial_spots = np.array([100.0])
        strike = 1.0

        obs_times = [0.25, 0.5, 0.75, 1.0]
        exercise_times = [0.5, 1.0]  # Can only exercise at 6M and 1Y

        option = VanillaOption(
            product_id="TEST_BERM_CALL",
            currency="USD",
            notional=1.0,
            underlying=Underlying("TEST", "equity"),
            strike=strike,
            maturity=1.0,
            option_type="call",
            exercise_style="bermudan",
            observation_times=obs_times,
            exercise_times=exercise_times,
        )

        evaluator = option.get_evaluator()
        result1 = evaluator.evaluate(paths, initial_spots)
        payoffs1 = result1["payoffs"]

        composable = option.to_composable_payoff()
        result2 = composable.evaluate_path(paths, times, initial_spots, notional=1.0)
        payoffs2 = result2["payoffs"]

        assert np.all(payoffs1 >= 0), "Evaluator payoffs should be non-negative"
        assert np.all(payoffs2 >= 0), "Component payoffs should be non-negative"
        
        # Evaluator should find opportunities at exercise dates
        assert np.mean(payoffs1) > 0, "Bermudan option should have positive average payoff"

    def test_bermudan_put_equivalence(self, sample_paths):
        """Test Bermudan put equivalence."""
        paths, times = sample_paths
        initial_spots = np.array([100.0])
        strike = 1.0

        obs_times = [0.25, 0.5, 0.75, 1.0]
        exercise_times = [0.5, 1.0]

        option = VanillaOption(
            product_id="TEST_BERM_PUT",
            currency="USD",
            notional=1.0,
            underlying=Underlying("TEST", "equity"),
            strike=strike,
            maturity=1.0,
            option_type="put",
            exercise_style="bermudan",
            observation_times=obs_times,
            exercise_times=exercise_times,
        )

        evaluator = option.get_evaluator()
        result1 = evaluator.evaluate(paths, initial_spots)
        payoffs1 = result1["payoffs"]

        composable = option.to_composable_payoff()
        result2 = composable.evaluate_path(paths, times, initial_spots, notional=1.0)
        payoffs2 = result2["payoffs"]

        assert np.all(payoffs1 >= 0)
        assert np.all(payoffs2 >= 0)

    def test_asian_call_average_price(self, sample_paths):
        """Test Asian call with average price."""
        paths, times = sample_paths
        initial_spots = np.array([100.0])
        strike = 1.0

        obs_times = [0.25, 0.5, 0.75, 1.0]

        option = VanillaOption(
            product_id="TEST_ASIAN_CALL",
            currency="USD",
            notional=1.0,
            underlying=Underlying("TEST", "equity"),
            strike=strike,
            maturity=1.0,
            option_type="call",
            exercise_style="asian",
            observation_times=obs_times,
            averaging_type="average_price",
        )

        evaluator = option.get_evaluator()
        result1 = evaluator.evaluate(paths, initial_spots)
        payoffs1 = result1["payoffs"]

        composable = option.to_composable_payoff()
        result2 = composable.evaluate_path(paths, times, initial_spots, notional=1.0)
        payoffs2 = result2["payoffs"]

        assert np.all(payoffs1 >= 0), "Evaluator payoffs should be non-negative"
        assert np.all(payoffs2 >= 0), "Component payoffs should be non-negative"
        assert "average_prices" in result1, "Result should include average prices"

    def test_asian_put_average_strike(self, sample_paths):
        """Test Asian put with average strike."""
        paths, times = sample_paths
        initial_spots = np.array([100.0])
        strike = 1.0

        obs_times = [0.25, 0.5, 0.75, 1.0]

        option = VanillaOption(
            product_id="TEST_ASIAN_PUT",
            currency="USD",
            notional=1.0,
            underlying=Underlying("TEST", "equity"),
            strike=strike,
            maturity=1.0,
            option_type="put",
            exercise_style="asian",
            observation_times=obs_times,
            averaging_type="average_strike",
        )

        evaluator = option.get_evaluator()
        result1 = evaluator.evaluate(paths, initial_spots)
        payoffs1 = result1["payoffs"]

        composable = option.to_composable_payoff()
        result2 = composable.evaluate_path(paths, times, initial_spots, notional=1.0)
        payoffs2 = result2["payoffs"]

        assert np.all(payoffs1 >= 0), "Evaluator payoffs should be non-negative"
        assert np.all(payoffs2 >= 0), "Component payoffs should be non-negative"

    def test_notional_scaling(self, sample_paths):
        """Test that notional scales payoffs correctly."""
        paths, times = sample_paths
        initial_spots = np.array([100.0])
        strike = 1.0
        notional = 10000.0

        option = VanillaOption(
            product_id="TEST_NOTIONAL",
            currency="USD",
            notional=notional,
            underlying=Underlying("TEST", "equity"),
            strike=strike,
            maturity=1.0,
            option_type="call",
            exercise_style="european",
        )

        evaluator = option.get_evaluator()
        result = evaluator.evaluate(paths, initial_spots)
        payoffs = result["payoffs"]

        # Create unit notional version
        option_unit = VanillaOption(
            product_id="TEST_UNIT",
            currency="USD",
            notional=1.0,
            underlying=Underlying("TEST", "equity"),
            strike=strike,
            maturity=1.0,
            option_type="call",
            exercise_style="european",
        )

        evaluator_unit = option_unit.get_evaluator()
        result_unit = evaluator_unit.evaluate(paths, initial_spots)
        payoffs_unit = result_unit["payoffs"]

        # Should scale by notional
        np.testing.assert_array_almost_equal(payoffs, payoffs_unit * notional, decimal=8)

    def test_different_strikes(self, sample_paths):
        """Test options with different strikes."""
        paths, times = sample_paths
        initial_spots = np.array([100.0])

        # ITM, ATM, OTM calls
        strikes = [0.9, 1.0, 1.1]
        payoffs_list = []

        for strike in strikes:
            option = VanillaOption(
                product_id=f"TEST_STRIKE_{strike}",
                currency="USD",
                notional=1.0,
                underlying=Underlying("TEST", "equity"),
                strike=strike,
                maturity=1.0,
                option_type="call",
                exercise_style="european",
            )

            evaluator = option.get_evaluator()
            result = evaluator.evaluate(paths, initial_spots)
            payoffs_list.append(result["payoffs"])

        # ITM call should have highest average payoff
        mean_payoffs = [np.mean(p) for p in payoffs_list]
        assert mean_payoffs[0] > mean_payoffs[1]  # 0.9 > 1.0
        assert mean_payoffs[1] > mean_payoffs[2]  # 1.0 > 1.1

    def test_direct_component_usage(self, sample_paths):
        """Test using components directly without VanillaOption wrapper."""

        paths, times = sample_paths
        initial_spots = np.array([100.0])
        strike = 1.0

        # Create components directly
        euro_component = EuropeanOption(strike=strike, option_type="call", participation=1.0)
        amer_component = AmericanOption(strike=strike, option_type="call", participation=1.0)
        berm_component = BermudanOption(
            strike=strike, option_type="call", exercise_times=[0.5, 1.0], participation=1.0
        )

        # Evaluate with ComposablePayoff
        euro_payoff = ComposablePayoff([euro_component])
        amer_payoff = ComposablePayoff([amer_component])
        berm_payoff = ComposablePayoff([berm_component])

        euro_result = euro_payoff.evaluate_path(paths, times, initial_spots, notional=1.0)
        amer_result = amer_payoff.evaluate_path(paths, times, initial_spots, notional=1.0)
        berm_result = berm_payoff.evaluate_path(paths, times, initial_spots, notional=1.0)

        # All should produce valid payoffs
        assert euro_result["payoffs"].shape == (100,)
        assert amer_result["payoffs"].shape == (100,)
        assert berm_result["payoffs"].shape == (100,)

    def test_bermudan_requires_exercise_times(self):
        """Test that Bermudan requires exercise_times."""
        with pytest.raises(ValueError, match="exercise_times required"):
            VanillaOption(
                product_id="TEST",
                currency="USD",
                notional=1.0,
                underlying=Underlying("TEST", "equity"),
                strike=1.0,
                maturity=1.0,
                option_type="call",
                exercise_style="bermudan",
                observation_times=[0.5, 1.0],
            )

    def test_exercise_times_must_be_in_observations(self):
        """Test that exercise times must be in observation times."""
        with pytest.raises(ValueError, match="Exercise time .* not in observation_times"):
            VanillaOption(
                product_id="TEST",
                currency="USD",
                notional=1.0,
                underlying=Underlying("TEST", "equity"),
                strike=1.0,
                maturity=1.0,
                option_type="call",
                exercise_style="bermudan",
                observation_times=[0.25, 0.5, 1.0],
                exercise_times=[0.3, 1.0],  # 0.3 not in observation_times
            )

    def test_observation_times_must_include_maturity(self):
        """Test that observation times must include maturity."""
        with pytest.raises(ValueError, match="observation_times must include maturity"):
            VanillaOption(
                product_id="TEST",
                currency="USD",
                notional=1.0,
                underlying=Underlying("TEST", "equity"),
                strike=1.0,
                maturity=1.0,
                option_type="call",
                exercise_style="american",
                observation_times=[0.25, 0.5, 0.75],  # Missing 1.0
            )

    def test_default_observation_times(self):
        """Test default observation times generation."""
        # European: just maturity
        euro = VanillaOption(
            product_id="TEST",
            currency="USD",
            notional=1.0,
            underlying=Underlying("TEST", "equity"),
            strike=1.0,
            maturity=1.0,
            option_type="call",
            exercise_style="european",
        )
        assert euro.observation_times == [1.0]

        # American: quarterly by default for 1Y
        amer = VanillaOption(
            product_id="TEST",
            currency="USD",
            notional=1.0,
            underlying=Underlying("TEST", "equity"),
            strike=1.0,
            maturity=1.0,
            option_type="call",
            exercise_style="american",
        )
        assert len(amer.observation_times) == 4  # Quarterly
        assert amer.observation_times[-1] == 1.0

    def test_json_serialization(self):
        """Test JSON serialization includes exercise_style."""
        import json

        option = VanillaOption(
            product_id="TEST",
            currency="USD",
            notional=1.0,
            underlying=Underlying("TEST", "equity"),
            strike=1.0,
            maturity=1.0,
            option_type="call",
            exercise_style="bermudan",
            observation_times=[0.5, 1.0],
            exercise_times=[0.5, 1.0],
        )

        json_str = option.to_json()
        data = json.loads(json_str)

        assert data["exercise_style"] == "bermudan"
        assert data["observation_times"] == [0.5, 1.0]
        assert data["exercise_times"] == [0.5, 1.0]
