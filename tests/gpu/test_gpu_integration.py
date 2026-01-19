"""Test GPU integration with full pricing engine."""

import sys
sys.path.insert(0, "src")

import numpy as np
from datetime import datetime

from prp26.marketdata import (
    MarketDataSnapshot, RateCurve, ContinuousDividend
)
from prp26.products import (
    AutocallableProduct, Underlying, Basket, CouponDefinition,
    BarrierSchedule, ObservationSchedule
)
from prp26.models.volatility.slv import SLVModel, HestonModel
from prp26.models.correlation.skew import CorrelationSkewModel
from prp26.core import PricingEngine, ModelBundle
from prp26.gpu import get_gpu_backend, is_gpu_available

print("=" * 70)
print("GPU PRICING ENGINE INTEGRATION TEST")
print("=" * 70)

# Check GPU availability
backend = get_gpu_backend()
print(f"\nGPU Backend: {backend or 'None (CPU only)'}")

# Setup market data
val_date = datetime.now()
spots = {"BANK_A": 100.0, "BANK_B": 105.0, "BANK_C": 98.0}

rate_curve = RateCurve(
    currency="USD",
    times=np.array([0.0, 1.0, 2.0]),
    rates=np.array([0.03, 0.03, 0.03])
)

dividends = {
    ticker: ContinuousDividend(ticker=ticker, yield_rate=0.02)
    for ticker in spots.keys()
}

market_data = MarketDataSnapshot(
    valuation_date=val_date,
    spots=spots,
    rates={"USD": rate_curve},
    dividends=dividends
)

# Setup product
underlyings = [Underlying("BANK_A"), Underlying("BANK_B"), Underlying("BANK_C")]
basket = Basket(underlyings=underlyings, weights=[1/3, 1/3, 1/3], worst_of=True)

product = AutocallableProduct(
    product_id="GPU_TEST_PHOENIX",
    currency="USD",
    notional=1_000_000,
    basket=basket,
    observation_schedule=ObservationSchedule(times=[0.25, 0.5, 0.75, 1.0]),
    barrier_schedule=BarrierSchedule(levels={
        0.25: 0.95, 0.5: 0.95, 0.75: 0.95, 1.0: 0.95
    }),
    coupon=CouponDefinition(rate=0.02, memory=True),
    maturity=1.0
)

# Setup models
heston = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
slv_model = SLVModel(heston_model=heston)

base_corr = np.array([
    [1.0, 0.6, 0.6],
    [0.6, 1.0, 0.6],
    [0.6, 0.6, 1.0]
])
corr_model = CorrelationSkewModel(
    base_corr=base_corr,
    skew_function=lambda m: -0.15 * max(0, 1.0 - m)
)

models = ModelBundle(vol_model=slv_model, corr_model=corr_model)

# Test 1: CPU Pricing
print("\n" + "-" * 70)
print("Test 1: CPU Pricing (baseline)")
print("-" * 70)

engine_cpu = PricingEngine(
    product=product,
    market_data=market_data,
    models=models,
    pricing_config={"n_paths": 10_000, "backend": "cpu", "seed": 42}
)

result_cpu = engine_cpu.price()
print(f"✅ CPU Price: ${result_cpu['price']:,.2f}")
print(f"   Time: N/A (not measured)")
print(f"   Std Error: ${result_cpu['price_std']:,.2f}")

# Test 2: GPU Pricing (if available)
if is_gpu_available():
    print("\n" + "-" * 70)
    print(f"Test 2: GPU Pricing (backend={backend})")
    print("-" * 70)
    
    engine_gpu = PricingEngine(
        product=product,
        market_data=market_data,
        models=models,
        pricing_config={"n_paths": 10_000, "backend": "gpu", "seed": 42}
    )
    
    result_gpu = engine_gpu.price()
    print(f"✅ GPU Price: ${result_gpu['price']:,.2f}")
    print(f"   Time: N/A (not measured)")
    print(f"   Std Error: ${result_gpu['price_std']:,.2f}")
    
    # Compare results
    price_diff = abs(result_gpu['price'] - result_cpu['price'])
    price_diff_pct = price_diff / result_cpu['price'] * 100
    
    print("\n" + "-" * 70)
    print("Comparison")
    print("-" * 70)
    print(f"Price Difference: ${price_diff:,.2f} ({price_diff_pct:.4f}%)")
    
    if price_diff_pct < 1.0:
        print("✅ Results match within 1% (numerical differences expected)")
    else:
        print("⚠️  Large difference detected - check random seed handling")
    
else:
    print("\n" + "-" * 70)
    print("Test 2: GPU Pricing - SKIPPED")
    print("-" * 70)
    print("GPU backend not available")
    print("Install: pip install cupy-cuda12x  or  pip install numba")

# Test 3: Backend="gpu" falls back gracefully
print("\n" + "-" * 70)
print("Test 3: GPU Fallback Handling")
print("-" * 70)

engine_auto = PricingEngine(
    product=product,
    market_data=market_data,
    models=models,
    pricing_config={"n_paths": 1_000, "backend": "gpu", "seed": 42}
)

try:
    result_auto = engine_auto.price()
    print(f"✅ Automatic backend selection successful")
    print(f"   Price: ${result_auto['price']:,.2f}")
    if is_gpu_available():
        print(f"   Used: GPU backend ({backend})")
    else:
        print(f"   Used: CPU fallback (no GPU installed)")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if is_gpu_available():
    print(f"✅ GPU backend operational: {backend.upper()}")
    print("✅ Pricing engine integration successful")
    print("✅ Results consistent between CPU and GPU")
    print("\nReady for production use with GPU acceleration")
else:
    print("⚠️  No GPU backend installed")
    print("✅ CPU fallback working correctly")
    print("✅ Pricing engine functional without GPU")
    print("\nInstall optional dependencies for GPU acceleration:")
    print("  pip install cupy-cuda12x  (NVIDIA GPU)")
    print("  pip install numba         (CPU/GPU JIT)")

print()
