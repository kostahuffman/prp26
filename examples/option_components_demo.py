"""Demo of new option payoff components.

This script demonstrates how to use the European, American, Asian, 
and Bermudan option components both standalone and in combination
with other payoff features like autocalls.
"""

import numpy as np

from prp26.products import Underlying, VanillaOption
from prp26.products.payoffs import (
    AmericanOption,
    AsianOption,
    AutocallComponent,
    BermudanOption,
    ComposablePayoff,
    EuropeanOption,
    MemoryCoupon,
)

print("=" * 80)
print("OPTION PAYOFF COMPONENTS DEMO")
print("=" * 80)
print()

# ========== Demo 1: VanillaOption Product ==========
print("[1] VanillaOption Product")
print("-" * 40)

vanilla = VanillaOption(
    product_id="SPX_CALL_1Y",
    currency="USD",
    notional=100_000,
    underlying=Underlying("SPX", "equity"),
    strike=1.0,
    maturity=1.0,
    option_type="call",
)

print(f"Product: {vanilla.product_id}")
print(f"Type: {vanilla.option_type} option")
print(f"Strike: {vanilla.strike}")
print(f"Maturity: {vanilla.maturity} years")

# Can get evaluator for pricing
evaluator = vanilla.get_evaluator()
print(f"Evaluator: {type(evaluator).__name__}")

# Can also convert to composable payoff
composable = vanilla.to_composable_payoff()
print(f"Composable: {len(composable.components)} component(s)")
print()

# ========== Demo 2: European Option as Component ==========
print("[2] European Option Component (standalone)")
print("-" * 40)

european_call = EuropeanOption(strike=1.0, option_type="call", participation=1.0)

# Simulate a simple path
n_paths = 5
initial_spots = np.array([100.0])
times = np.array([0.5, 1.0])

# Create paths: first at 105, final at 110
paths = np.zeros((n_paths, 2, 1))
paths[:, 0, 0] = 105.0
paths[:, 1, 0] = 110.0

payoff = ComposablePayoff([european_call])
result = payoff.evaluate_path(paths, times, initial_spots, notional=1.0)

print(f"Paths: T=0.5 @ 105, T=1.0 @ 110")
print(f"Payoff: {result['payoffs'][0]:.4f}")
print(f"Expected: max(1.10 - 1.00, 0) = 0.10")
print()

# ========== Demo 3: Asian Option ==========
print("[3] Asian Option (Average Price)")
print("-" * 40)

asian_call = AsianOption(
    strike=1.0, option_type="call", asian_type="average_price", participation=1.0
)

# Paths with varying spot values
times_asian = np.array([0.25, 0.5, 0.75, 1.0])
paths_asian = np.zeros((n_paths, 4, 1))
paths_asian[:, 0, 0] = 105.0  # +5%
paths_asian[:, 1, 0] = 110.0  # +10%
paths_asian[:, 2, 0] = 100.0  # 0%
paths_asian[:, 3, 0] = 105.0  # +5%

payoff_asian = ComposablePayoff([asian_call])
result_asian = payoff_asian.evaluate_path(paths_asian, times_asian, initial_spots, notional=1.0)

avg_perf = (1.05 + 1.10 + 1.00 + 1.05) / 4  # = 1.05
print(f"Spot observations: 105, 110, 100, 105")
print(f"Average performance: {avg_perf:.4f}")
print(f"Payoff: {result_asian['payoffs'][0]:.4f}")
print(f"Expected: max({avg_perf:.2f} - 1.00, 0) = {max(avg_perf - 1.0, 0):.2f}")
print()

# ========== Demo 4: Bermudan Option ==========
print("[4] Bermudan Option (Exercise at T=0.5 or T=1.0)")
print("-" * 40)

bermudan_call = BermudanOption(
    strike=1.0, exercise_times=[0.5, 1.0], option_type="call", participation=1.0
)

payoff_bermudan = ComposablePayoff([bermudan_call])
result_bermudan = payoff_bermudan.evaluate_path(paths, times, initial_spots, notional=1.0)

print(f"Exercise dates: T=0.5, T=1.0")
print(f"Intrinsic at T=0.5: max(1.05 - 1.00, 0) = 0.05")
print(f"Intrinsic at T=1.0: max(1.10 - 1.00, 0) = 0.10")
print("Note: Full optimal exercise requires regression methods")
print()

# ========== Demo 5: Combined Payoff (Autocall + Option) ==========
print("[5] Autocallable with European Option")
print("-" * 40)

combined_payoff = ComposablePayoff(
    components=[
        AutocallComponent(barrier=1.08, redemption=1.0),
        EuropeanOption(strike=1.0, option_type="call", participation=1.0),
    ]
)

# Paths that trigger autocall at first observation
paths_autocall = np.zeros((n_paths, 2, 1))
paths_autocall[:, 0, 0] = 109.0  # Above autocall barrier (1.09 > 1.08)
paths_autocall[:, 1, 0] = 110.0  # Would be even higher

result_combined = combined_payoff.evaluate_path(
    paths_autocall, times, initial_spots, notional=1.0
)

print(f"Autocall barrier: 1.08 (108%)")
print(f"Spot at T=0.5: 109 (triggers autocall)")
print(f"All paths terminated: {np.all(result_combined['terminated'])}")
print(f"Cashflow at T=0.5: {result_combined['cashflows'][0, 0]:.2f} (from autocall)")
print(f"Cashflow at T=1.0: {result_combined['cashflows'][0, 1]:.2f} (option not exercised)")
print()

# ========== Demo 6: Complex Structure ==========
print("[6] Complex Structure: Memory Coupon + European Call")
print("-" * 40)

complex_payoff = ComposablePayoff(
    components=[
        MemoryCoupon(rate=0.02, barrier=0.70),  # 2% coupon if above 70%
        AutocallComponent(barrier=1.05, redemption=1.0),
        EuropeanOption(strike=1.0, option_type="call", participation=1.0),
    ]
)

# Paths: first below coupon barrier, second above autocall
paths_complex = np.zeros((n_paths, 2, 1))
paths_complex[:, 0, 0] = 65.0  # Below coupon barrier (no coupon paid)
paths_complex[:, 1, 0] = 107.0  # Above autocall barrier

result_complex = complex_payoff.evaluate_path(
    paths_complex, times, initial_spots, notional=1.0
)

print(f"T=0.5: Spot=65 (below coupon barrier 70%)")
print(f"  → No coupon paid, memory accumulates 0.02")
print(f"T=1.0: Spot=107 (above autocall barrier 105%)")
print(f"  → Autocall triggered, pays 1.0 redemption")
print(f"  → European option not evaluated (path terminated)")
print(f"Payoff: {result_complex['payoffs'][0]:.4f}")
print()

print("=" * 80)
print("Demo complete! Option components work seamlessly with other payoffs.")
print("=" * 80)
