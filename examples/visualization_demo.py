"""
COMPREHENSIVE VISUALIZATION DEMO
=================================

Demonstrates all visualization capabilities:
1. Monte Carlo path analysis
2. Payoff distributions
3. Greeks dashboard
4. P&L history tracking
5. Termination analysis
6. Worst-of heatmaps
7. Portfolio views

This shows how to use the visualization module for risk management
and client reporting.
"""

from datetime import datetime, timedelta

import numpy as np

from prp26.core import ModelBundle, PricingEngine
from prp26.marketdata import ContinuousDividend, MarketDataSnapshot, RateCurve
from prp26.models.correlation.skew import CorrelationSkewModel
from prp26.models.volatility.slv import HestonModel, SLVModel
from prp26.products import AutocallableProduct, Basket, Underlying
from prp26.products.schedules import (
    BarrierSchedule,
    CouponDefinition,
    ObservationSchedule,
)
from prp26.products.taxonomy import TAXONOMY_PHOENIX_AUTOCALLABLE

print("=" * 80)
print("COMPREHENSIVE VISUALIZATION DEMO")
print("=" * 80)
print()

# ==================== SETUP: Price a Product ====================
print("[Setup] Creating and pricing product...")
print()


# Initial setup
t0_date = datetime(2026, 1, 17)
t0_spots = {"AAPL": 180.0, "MSFT": 420.0, "GOOGL": 145.0}

t0_market_data = MarketDataSnapshot(
    valuation_date=t0_date,
    spots=t0_spots,
    rates={"USD": RateCurve.flat_curve("USD", rate=0.045)},
    dividends={
        "AAPL": ContinuousDividend("AAPL", yield_rate=0.005),
        "MSFT": ContinuousDividend("MSFT", yield_rate=0.008),
        "GOOGL": ContinuousDividend("GOOGL", yield_rate=0.000),
    },
)

# Product
basket = Basket(
    underlyings=[Underlying("AAPL"), Underlying("MSFT"), Underlying("GOOGL")],
    weights=[1 / 3, 1 / 3, 1 / 3],
    worst_of=True,
)

product = AutocallableProduct(
    product_id="PHOENIX_VIZ_DEMO",
    currency="USD",
    notional=10_000_000,
    basket=basket,
    observation_schedule=ObservationSchedule(times=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
    barrier_schedule=BarrierSchedule(
        levels={
            0.25: 1.20,
            0.5: 1.20,
            0.75: 1.20,
            1.0: 1.20,
            1.25: 1.20,
            1.5: 1.20,
            1.75: 1.20,
            2.0: 1.20,
        }
    ),
    coupon=CouponDefinition(rate=0.025, memory=True),
    maturity=2.0,
    taxonomy=TAXONOMY_PHOENIX_AUTOCALLABLE,
)

# Models
heston = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
slv_model = SLVModel(heston_model=heston)
corr_model = CorrelationSkewModel(
    base_corr=np.array([[1.0, 0.6, 0.6], [0.6, 1.0, 0.6], [0.6, 0.6, 1.0]]),
    skew_function=lambda m: -0.15 * max(0, 1.0 - m),
)
models = ModelBundle(vol_model=slv_model, corr_model=corr_model)

# Price
engine = PricingEngine(
    product=product,
    market_data=t0_market_data,
    models=models,
    pricing_config={"n_paths": 10_000, "seed": 43, "backend": "cpu"},
)

print("Pricing at T0 (10,000 paths for visualization)...")
t0_result = engine.price()

print(f"[OK] T0 Price: ${t0_result['price']:,.2f}")
print()

# Get paths for visualization
initial_spots = np.array([t0_spots[ticker] for ticker in ["AAPL", "MSFT", "GOOGL"]])
times = product.observation_schedule.times

# Generate paths for visualization
paths, _ = engine.path_generator.generate_paths(
    spots=initial_spots,
    times=times,
    n_paths=10_000,
    backend="cpu",
)

print("[OK] Generated 10,000 Monte Carlo paths")
print()

# ==================== VISUALIZATION 1: MONTE CARLO PATHS ====================
print("=" * 80)
print("VISUALIZATION 1: Monte Carlo Path Analysis")
print("=" * 80)
print()

from prp26.visualization import StructuredProductVisualizer  # noqa: E402

viz = StructuredProductVisualizer(product=product, save_plots=False)

print("[Plot 1/6] Monte Carlo paths (showing 100 sample paths)...")
viz.plot_monte_carlo_paths(
    paths=paths,
    times=times,
    tickers=["AAPL", "MSFT", "GOOGL"],
    initial_spots=initial_spots,
    n_paths_to_show=100,
)

# ==================== VISUALIZATION 2: PAYOFF DISTRIBUTION ====================
print("=" * 80)
print("VISUALIZATION 2: Payoff Distribution")
print("=" * 80)
print()

print("[Plot 2/6] Payoff distribution and CDF...")
viz.plot_payoff_distribution(
    payoffs=t0_result["payoffs"],
    notional=product.notional,
)

# ==================== VISUALIZATION 3: TERMINATION ANALYSIS ====================
print("=" * 80)
print("VISUALIZATION 3: Early Termination Analysis")
print("=" * 80)
print()

print("[Plot 3/6] Early termination distribution and survival curve...")
viz.plot_termination_analysis(
    termination_times=t0_result["termination_times"],
    observation_times=product.observation_schedule.times,
)

# ==================== VISUALIZATION 4: WORST-OF HEATMAP ====================
print("=" * 80)
print("VISUALIZATION 4: Worst-Of Performance")
print("=" * 80)
print()

print("[Plot 4/6] Worst-of performance percentiles and heatmap...")
viz.plot_worst_of_heatmap(
    paths=paths,
    times=times,
    tickers=["AAPL", "MSFT", "GOOGL"],
    initial_spots=initial_spots,
)

# ==================== REPRICING FOR P&L AND GREEKS ====================
print("=" * 80)
print("Setting up repricing scenario...")
print("=" * 80)
print()

# Create P&L history by repricing at multiple dates
pnl_history = []

# T0
pnl_history.append(
    {
        "date": t0_date,
        "days": 0,
        "spots": t0_spots,
        "pv": t0_result["price"],
        "pnl": 0.0,
    }
)

# T+30: Rally
t1_date = t0_date + timedelta(days=30)
t1_spots = {"AAPL": 195.0, "MSFT": 445.0, "GOOGL": 156.0}
t1_market_data = MarketDataSnapshot(
    valuation_date=t1_date,
    spots=t1_spots,
    rates={"USD": RateCurve.flat_curve("USD", rate=0.044)},
    dividends={
        "AAPL": ContinuousDividend("AAPL", yield_rate=0.005),
        "MSFT": ContinuousDividend("MSFT", yield_rate=0.008),
        "GOOGL": ContinuousDividend("GOOGL", yield_rate=0.000),
    },
)

print(f"[Repricing] T+30 ({t1_date.strftime('%Y-%m-%d')})...")
t1_engine = PricingEngine(
    product=product, market_data=t1_market_data, models=models, pricing_config=engine.pricing_config
)
t1_result = t1_engine.price()

pnl_history.append(
    {
        "date": t1_date,
        "days": 30,
        "spots": t1_spots,
        "pv": t1_result["price"],
        "pnl": t1_result["price"] - t0_result["price"],
    }
)

# T+60: Pullback
t2_date = t0_date + timedelta(days=60)
t2_spots = {"AAPL": 188.0, "MSFT": 430.0, "GOOGL": 150.0}
t2_market_data = MarketDataSnapshot(
    valuation_date=t2_date,
    spots=t2_spots,
    rates={"USD": RateCurve.flat_curve("USD", rate=0.045)},
    dividends={
        "AAPL": ContinuousDividend("AAPL", yield_rate=0.005),
        "MSFT": ContinuousDividend("MSFT", yield_rate=0.008),
        "GOOGL": ContinuousDividend("GOOGL", yield_rate=0.000),
    },
)

print(f"[Repricing] T+60 ({t2_date.strftime('%Y-%m-%d')})...")
t2_engine = PricingEngine(
    product=product, market_data=t2_market_data, models=models, pricing_config=engine.pricing_config
)
t2_result = t2_engine.price()

pnl_history.append(
    {
        "date": t2_date,
        "days": 60,
        "spots": t2_spots,
        "pv": t2_result["price"],
        "pnl": t2_result["price"] - t0_result["price"],
    }
)

# T+90: Further rally
t3_date = t0_date + timedelta(days=90)
t3_spots = {"AAPL": 200.0, "MSFT": 460.0, "GOOGL": 165.0}
t3_market_data = MarketDataSnapshot(
    valuation_date=t3_date,
    spots=t3_spots,
    rates={"USD": RateCurve.flat_curve("USD", rate=0.043)},
    dividends={
        "AAPL": ContinuousDividend("AAPL", yield_rate=0.005),
        "MSFT": ContinuousDividend("MSFT", yield_rate=0.008),
        "GOOGL": ContinuousDividend("GOOGL", yield_rate=0.000),
    },
)

print(f"[Repricing] T+90 ({t3_date.strftime('%Y-%m-%d')})...")
t3_engine = PricingEngine(
    product=product, market_data=t3_market_data, models=models, pricing_config=engine.pricing_config
)
t3_result = t3_engine.price()

pnl_history.append(
    {
        "date": t3_date,
        "days": 90,
        "spots": t3_spots,
        "pv": t3_result["price"],
        "pnl": t3_result["price"] - t0_result["price"],
    }
)

print("[OK] Repricing complete")
print()

# ==================== VISUALIZATION 5: P&L HISTORY ====================
print("=" * 80)
print("VISUALIZATION 5: P&L History")
print("=" * 80)
print()

print("[Plot 5/6] P&L history with spot price evolution...")
viz.plot_pnl_history(pnl_history=pnl_history)

# ==================== VISUALIZATION 6: GREEKS DASHBOARD ====================
print("=" * 80)
print("VISUALIZATION 6: Greeks Risk Dashboard")
print("=" * 80)
print()

print("[Computing Greeks] This takes a moment...")
t3_greeks = t3_engine.compute_greeks(tickers=["AAPL", "MSFT", "GOOGL"], method="bump_reprice")

print("[Plot 6/6] Greeks dashboard with risk exposures...")
viz.plot_greeks_dashboard(
    greeks=t3_greeks,
    spots=t3_spots,
)

# ==================== SUMMARY STATISTICS ====================
print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print()

print("[Monte Carlo Statistics]")
print(f"   Number of Paths: {len(t0_result['payoffs']):,}")
print(f"   Mean Payoff: ${np.mean(t0_result['payoffs']):,.2f}")
print(f"   Std Dev: ${np.std(t0_result['payoffs']):,.2f}")
print(f"   Min Payoff: ${np.min(t0_result['payoffs']):,.2f}")
print(f"   Max Payoff: ${np.max(t0_result['payoffs']):,.2f}")
print(f"   VaR (95%): ${np.percentile(t0_result['payoffs'], 5):,.2f}")
print(
    f"   CVaR (95%): ${np.mean(t0_result['payoffs'][t0_result['payoffs'] <= np.percentile(t0_result['payoffs'], 5)]):,.2f}"
)
print()

print("[Early Termination Statistics]")
term_times = t0_result["termination_times"]
for obs_time in product.observation_schedule.times:
    early_call_rate = np.mean(term_times == obs_time)
    if early_call_rate > 0:
        print(f"   T={obs_time:.2f}Y: {early_call_rate:.1%} of paths terminated")
print()

print("[P&L Summary]")
for _i, entry in enumerate(pnl_history):
    date_str = entry["date"].strftime("%Y-%m-%d")
    print(f"   {date_str}: PV=${entry['pv']:,.0f}, P&L=${entry['pnl']:,.0f}")
print()

print("[Current Risk Exposures (T+90)]")
print(f"   Total Delta: ${sum(t3_greeks['delta'].values()):,.0f}")
for ticker, delta in t3_greeks["delta"].items():
    print(f"      {ticker}: ${delta:,.0f}")
print(f"   Rho: ${t3_greeks['rho']:,.2f}")
print()

# ==================== PORTFOLIO AGGREGATION ====================
print("=" * 80)
print("PORTFOLIO AGGREGATION (10 Notes)")
print("=" * 80)
print()

portfolio_size = 10
total_notional = product.notional * portfolio_size

print(f"Portfolio Size: {portfolio_size} notes")
print(f"Total Notional: ${total_notional:,.0f}")
print()
print(f"{'Metric':<30} | {'Single Note':<20} | {'Portfolio':<20}")
print("-" * 75)
print(
    f"{'Current PV (T+90)':<30} | ${t3_result['price']:>18,.0f} | ${t3_result['price'] * portfolio_size:>18,.0f}"
)
print(
    f"{'Cumulative P&L':<30} | ${pnl_history[-1]['pnl']:>18,.0f} | ${pnl_history[-1]['pnl'] * portfolio_size:>18,.0f}"
)
print(
    f"{'Total Delta (All Underlyings)':<30} | ${sum(t3_greeks['delta'].values()):>18,.0f} | ${sum(t3_greeks['delta'].values()) * portfolio_size:>18,.0f}"
)
print(
    f"{'Total Rho':<30} | ${t3_greeks['rho']:>18,.2f} | ${t3_greeks['rho'] * portfolio_size:>18,.2f}"
)
print()

# ==================== COMPLETE ====================
print("=" * 80)
print("VISUALIZATION DEMO COMPLETE")
print("=" * 80)
print()

print("""
[✓] All Visualizations Generated:
    1. Monte Carlo Paths - Sample path evolution
    2. Payoff Distribution - Histogram and CDF
    3. Termination Analysis - Early call distribution
    4. Worst-Of Heatmap - Performance density over time
    5. P&L History - Mark-to-market tracking
    6. Greeks Dashboard - Risk exposures

[✓] Risk Metrics Computed:
    - Delta per underlying
    - Gamma (convexity)
    - Rho (rate sensitivity)
    - VaR and CVaR

[✓] Portfolio Aggregation:
    - Multi-note positions
    - Total exposures
    - P&L tracking

🎯 Production Use Cases:
    - Daily risk reports for trading desk
    - Client presentations with visual P&L
    - Regulatory VaR reporting
    - Scenario stress testing
    - Portfolio-level aggregation

💡 To save plots automatically:
    viz = StructuredProductVisualizer(save_plots=True, output_dir="./plots")
""")

print("=" * 80)
print("END OF DEMO")
print("=" * 80)
