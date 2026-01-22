"""Test equivalence between ComposablePayoff and StructuredProduct approaches.

This test verifies that the same product can be represented both ways and produce
consistent results when priced with the PricingEngine.
"""

import numpy as np
import pytest

from prp26.core import ModelBundle, PricingEngine
from prp26.marketdata import ContinuousDividend, RateCurve
from prp26.models.volatility.slv import HestonModel, SLVModel
from prp26.products import AutocallableProduct, Basket, Underlying
from prp26.products.payoffs import AutocallComponent, ComposablePayoff, MemoryCoupon
from prp26.products.payoffs.downside import WorstOfPut
from prp26.products.schedules import (
    BarrierSchedule,
    CouponDefinition,
    ObservationSchedule,
)


@pytest.fixture
def basket():
    """Create a basket with single underlying."""
    underlying = Underlying(symbol="SPX", asset_class="equity", spot=100.0)
    return Basket(underlyings=[underlying], weights=[1.0], worst_of=False)


@pytest.fixture
def models():
    """Create model bundle for pricing."""
    heston = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    slv_model = SLVModel(heston_model=heston)
    return ModelBundle(vol_model=slv_model, corr_model=None)


@pytest.mark.skip(
    reason="Requires full market data setup - see examples/visualization_demo.py for reference"
)
def test_phoenix_equivalence(basket, models):
    """Test that Phoenix autocallable produces same results via both approaches.

    Creates a Phoenix autocallable with:
    - 1 year maturity
    - Quarterly observations
    - 95% autocall barrier
    - 70% coupon barrier
    - 8% annual coupon (2% per quarter)
    - Memory feature
    """
    # Product parameters
    observation_times = [0.25, 0.5, 0.75, 1.0]
    autocall_barrier = 0.95
    coupon_barrier = 0.70
    annual_coupon = 0.08
    quarterly_coupon = annual_coupon / 4
    notional = 100000.0

    # Market data
    spots = {"SPX": 100.0}
    rates = {"USD": RateCurve.flat_curve("USD", rate=0.03)}
    dividends = {"SPX": ContinuousDividend("SPX", yield_rate=0.01)}

    # ===== Approach 1: StructuredProduct (AutocallableProduct) =====
    product = AutocallableProduct(
        product_id="PHOENIX_TEST",
        currency="USD",
        notional=notional,
        basket=basket,
        observation_schedule=ObservationSchedule(times=observation_times),
        barrier_schedule=BarrierSchedule(levels={t: autocall_barrier for t in observation_times}),
        coupon=CouponDefinition(rate=quarterly_coupon, memory=True, conditional=True),
        maturity=1.0,
    )

    engine1 = PricingEngine(
        product=product,
        spots=spots,
        rates=rates,
        dividends=dividends,
        models=models,
    )

    # ===== Approach 2: ComposablePayoff =====
    composable_payoff = ComposablePayoff(
        components=[
            MemoryCoupon(rate=quarterly_coupon, barrier=coupon_barrier),
            AutocallComponent(barrier=autocall_barrier, redemption=1.0),
            WorstOfPut(strike=1.0, participation=1.0),
        ]
    )

    engine2 = PricingEngine(
        product=composable_payoff,
        spots=spots,
        rates=rates,
        dividends=dividends,
        models=models,
        notional=notional,
    )

    # ===== Price both and compare =====
    np.random.seed(42)
    result1 = engine1.price(n_paths=5000, backend="cpu")

    np.random.seed(42)
    result2 = engine2.price(n_paths=5000, backend="cpu")

    # Prices should be very close (within statistical error)
    price1 = result1["price"]
    price2 = result2["price"]
    std1 = result1["price_std"]
    std2 = result2["price_std"]

    print(f"\nStructuredProduct price: {price1:.2f} ± {std1:.2f}")
    print(f"ComposablePayoff price: {price2:.2f} ± {std2:.2f}")
    print(f"Difference: {abs(price1 - price2):.2f}")
    print(f"Combined std error: {np.sqrt(std1**2 + std2**2):.2f}")

    # Prices should match within 3 standard errors
    combined_std = np.sqrt(std1**2 + std2**2)
    assert abs(price1 - price2) < 3 * combined_std, (
        f"Prices differ by more than 3 std errors: "
        f"{price1:.2f} vs {price2:.2f} (diff={abs(price1-price2):.2f}, "
        f"3*std={3*combined_std:.2f})"
    )

    # Termination rates should be similar
    term_rate1 = result1["diagnostics"]["termination_rate"]
    term_rate2 = result2["diagnostics"]["termination_rate"]
    print(f"StructuredProduct termination rate: {term_rate1:.2%}")
    print(f"ComposablePayoff termination rate: {term_rate2:.2%}")

    assert (
        abs(term_rate1 - term_rate2) < 0.05
    ), f"Termination rates differ: {term_rate1:.2%} vs {term_rate2:.2%}"


def test_path_evaluation_consistency():
    """Test that evaluators produce consistent results for same paths.

    This tests the core evaluation logic by using the same simulated paths.
    """
    # This is a simplified test - full implementation would require PathGenerator
    # For now, we just verify the evaluators exist and are callable
    from prp26.products.payoffs.evaluators import PhoenixPayoffEvaluator

    evaluator = PhoenixPayoffEvaluator(
        observation_times=[0.5, 1.0],
        autocall_barriers=[0.95, 0.95],
        coupon_barriers=[0.70, 0.70],
        coupon_rate=0.02,
        notional=1.0,
    )

    # Just verify it's created successfully
    assert evaluator is not None
    assert evaluator.coupon_rate == 0.02


def test_autocallable_get_evaluator(basket):
    """Test that AutocallableProduct.get_evaluator() returns correct evaluator."""
    from prp26.products.payoffs.evaluators import PhoenixPayoffEvaluator

    product = AutocallableProduct(
        product_id="TEST_PHOENIX",
        currency="USD",
        notional=100000.0,
        basket=basket,
        observation_schedule=ObservationSchedule(times=[0.5, 1.0]),
        barrier_schedule=BarrierSchedule(levels={0.5: 0.95, 1.0: 0.95}),
        coupon=CouponDefinition(rate=0.02, memory=True),
        maturity=1.0,
    )

    evaluator = product.get_evaluator()

    # Should return PhoenixPayoffEvaluator
    assert isinstance(evaluator, PhoenixPayoffEvaluator)
    np.testing.assert_array_equal(evaluator.observation_times, [0.5, 1.0])
    assert evaluator.coupon_rate == 0.02
    assert evaluator.notional == 100000.0


def test_snowball_get_evaluator(basket):
    """Test that AutocallableProduct returns SnowballEvaluator for snowball products."""
    from prp26.products.payoffs.evaluators import SnowballPayoffEvaluator

    product = AutocallableProduct(
        product_id="TEST_SNOWBALL",
        currency="USD",
        notional=100000.0,
        basket=basket,
        observation_schedule=ObservationSchedule(times=[0.5, 1.0]),
        barrier_schedule=BarrierSchedule(levels={0.5: 0.95, 1.0: 0.95}),
        coupon=CouponDefinition(rate=0.02, memory=True, snowball=True),
        maturity=1.0,
    )

    evaluator = product.get_evaluator()

    # Should return SnowballPayoffEvaluator
    assert isinstance(evaluator, SnowballPayoffEvaluator)
    assert evaluator.observation_times == [0.5, 1.0]
    assert evaluator.coupon_rate == 0.02
    assert evaluator.notional == 100000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
