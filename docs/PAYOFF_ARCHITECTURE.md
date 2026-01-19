# Payoff Architecture Guide

## Overview

This document clarifies the architecture for product pricing, specifically the relationship between **StructuredProduct**, **ComposablePayoff**, **PayoffEvaluators**, and **PayoffComponents**.

## Architecture Layers

### 1. **StructuredProduct** (Legacy Product Definitions)

**Location**: `products/autocallable.py`, `products/reverse_convertible.py`

**Purpose**: Traditional product classes that encapsulate all product details including:
- Product metadata (ID, currency, notional, dates)
- Basket/underlying definitions
- Observation schedules
- Barrier schedules
- Coupon definitions
- Taxonomy

**Use Case**: 
- When you need a complete product object with validation
- For serialization/deserialization (JSON)
- For product catalog and trade booking systems
- When product is well-defined and standardized

**Example**:
```python
product = AutocallableProduct(
    product_id="PHOENIX_3Y",
    currency="USD",
    notional=1_000_000,
    basket=basket,
    observation_schedule=ObservationSchedule(times=[0.25, 0.5, 0.75, 1.0]),
    barrier_schedule=BarrierSchedule(...),
    coupon=CouponDefinition(rate=0.08, memory=True),
    maturity=1.0
)
```

**Limitations**:
- Tightly coupled to specific product types
- Hard to mix and match features
- Requires new class for each product variant

---

### 2. **ComposablePayoff** (Modern Compositional System)

**Location**: `products/payoffs/base.py`

**Purpose**: Build payoffs by composing reusable **PayoffComponents** - enables:
- JSON-driven product construction
- Mix-and-match payoff features
- No need for new product classes
- Flexible feature combinations

**Use Case**:
- Dynamic product construction (e.g., from UI/API)
- Custom/exotic products not in standard catalog
- Rapid prototyping of new structures
- When flexibility > standardization

**Example**:
```python
payoff = ComposablePayoff(components=[
    MemoryCoupon(rate=0.02, barrier=0.70),
    AutocallComponent(barrier=0.95),
    WorstOfPut(strike=1.0, participation=1.0)
])

# Can be serialized to/from JSON
payoff_json = payoff.to_dict()
```

**Advantages**:
- Extremely flexible
- Easy to extend with new components
- Natural JSON representation
- Component reuse across products

---

### 3. **PayoffComponent** (Building Blocks)

**Location**: `products/payoffs/coupon.py`, `products/payoffs/autocall.py`, `products/payoffs/downside.py`

**Purpose**: Individual payoff features that can be combined via `ComposablePayoff`

**Interface**:
```python
class PayoffComponent(ABC):
    @abstractmethod
    def evaluate(self, spots, initial_spots, time, state) -> dict:
        """
        Returns:
            - cashflow: Payment at this observation
            - terminated: Whether path terminated early
            - continue: Whether to continue evaluation
        """
        ...
```

**Available Components**:
- **MemoryCoupon**: Phoenix-style accumulating coupon
- **SnowballCoupon**: Growing accumulated coupon
- **AutocallComponent**: Early termination trigger
- **WorstOfPut**: Downside participation

**When to Create**:
- Reusable payoff feature across multiple products
- Stateful behavior (memory, accumulation)
- Conditional logic (barriers, triggers)

---

### 4. **PayoffEvaluator** (Engine Integration)

**Location**: `products/payoffs/phoenix.py`, `products/payoffs/snowball_evaluator.py`

**Purpose**: Complete payoff evaluation for specific product types - optimized for Monte Carlo simulation in `PricingEngine`

**Interface**:
```python
class PayoffEvaluator:
    def evaluate(self, paths: np.ndarray, initial_spots: np.ndarray) -> dict:
        """
        Args:
            paths: (n_paths, n_steps, n_assets)
            initial_spots: (n_assets,)
        
        Returns:
            - payoffs: (n_paths,)
            - cashflows: (n_paths, n_obs)
            - terminated: (n_paths,)
            - termination_times: (n_paths,)
        """
        ...
```

**Available Evaluators**:
- **PhoenixPayoffEvaluator**: Memory coupon autocallables
- **SnowballPayoffEvaluator**: Snowball accumulation products
- **ReverseConvertibleEvaluator**: (To be created)

**When to Create**:
- Product-specific optimizations needed
- Complex state management across observations
- Performance-critical calculations
- Legacy products not using ComposablePayoff

---

## Design Patterns

### Pattern 1: Product Definition → PayoffEvaluator (Legacy)

```python
# 1. Define product with StructuredProduct class
product = AutocallableProduct(...)

# 2. Engine extracts parameters and creates PayoffEvaluator
evaluator = PhoenixPayoffEvaluator(
    observation_times=product.observation_schedule.times,
    autocall_barriers=...,
    coupon_barriers=...,
    coupon_rate=product.coupon.rate,
    notional=product.notional
)

# 3. Evaluator processes Monte Carlo paths
result = evaluator.evaluate(paths, initial_spots)
```

**Usage**: Established product types (Phoenix, Reverse Convertible)

---

### Pattern 2: ComposablePayoff (Modern)

```python
# 1. Build payoff from components
payoff = ComposablePayoff(components=[
    MemoryCoupon(rate=0.02, barrier=0.70),
    AutocallComponent(barrier=0.95),
    WorstOfPut(strike=1.0)
])

# 2. Engine uses ComposablePayoff directly
result = payoff.evaluate_path(paths, times, initial_spots, notional)
```

**Usage**: New/custom products, JSON-driven construction

---

### Pattern 3: Hybrid (Bridge)

```python
# 1. Product with composable payoff reference
class ReverseConvertible(StructuredProduct):
    def get_evaluator(self) -> PayoffEvaluator:
        """Return appropriate evaluator for this product."""
        return ReverseConvertibleEvaluator(
            observation_times=self.get_coupon_payment_times(),
            strike=self.payoff.strike,
            coupon_rate=self.payoff.coupon_rate,
            barrier=self.payoff.barrier
        )
```

**Usage**: Gradual migration from legacy to composable

---

## Organizational Rules

### File Organization

```
products/
├── base.py                    # StructuredProduct base class, Basket, Underlying
├── autocallable.py            # AutocallableProduct (StructuredProduct subclass)
├── reverse_convertible.py    # ReverseConvertible (StructuredProduct subclass)
├── participation.py           # Participation products
└── payoffs/
    ├── base.py                # PayoffComponent, ComposablePayoff
    ├── coupon.py              # Coupon components (MemoryCoupon, SnowballCoupon)
    ├── autocall.py            # Autocall component
    ├── downside.py            # Downside components (WorstOfPut)
    ├── rainbow.py             # Rainbow components (NEW)
    └── evaluators/
        ├── phoenix.py         # PhoenixPayoffEvaluator
        ├── snowball.py        # SnowballPayoffEvaluator
        ├── reverse_convertible.py  # ReverseConvertibleEvaluator (NEW)
        └── rainbow.py         # RainbowPayoffEvaluator (NEW)
```

### Naming Conventions

1. **PayoffComponent**: `<Feature>Component` or `<Feature>Coupon`
   - Examples: `AutocallComponent`, `MemoryCoupon`, `RainbowComponent`
   
2. **PayoffEvaluator**: `<ProductType>PayoffEvaluator`
   - Examples: `PhoenixPayoffEvaluator`, `SnowballPayoffEvaluator`
   
3. **StructuredProduct**: `<ProductType>Product` or `<ProductType>`
   - Examples: `AutocallableProduct`, `ReverseConvertible`

4. **Product-specific payoff data**: `<ProductType>PayoffDefinition` (dataclass)
   - Examples: `ReverseConvertiblePayoffDefinition`, `ParticipationPayoffDefinition`
   - Purpose: Serialization/deserialization of product-specific payoff parameters

---

## When to Use Which Approach

| Scenario | Use This | Reason |
|----------|----------|--------|
| Standard Phoenix autocallable | `AutocallableProduct` + `PhoenixPayoffEvaluator` | Well-tested, optimized |
| Custom multi-feature product | `ComposablePayoff` | Flexible composition |
| JSON-driven product UI | `ComposablePayoff` | Natural JSON mapping |
| Product catalog/booking | `StructuredProduct` subclass | Complete product definition |
| Performance-critical calc | `PayoffEvaluator` | Optimized for Monte Carlo |
| Reusable payoff feature | `PayoffComponent` | Component reuse |

---

## Migration Strategy

### Phase 1: Maintain Both Systems
- Keep existing `StructuredProduct` classes
- Support `ComposablePayoff` alongside them
- `PricingEngine` handles both via `isinstance()` check

### Phase 2: Create Bridges
- Add `get_evaluator()` methods to `StructuredProduct` classes
- Standardize evaluator interface

### Phase 3: Consolidate (Future)
- Migrate common products to `ComposablePayoff`
- Keep `StructuredProduct` for product catalog only
- All pricing uses evaluators

---

## Examples

### Example 1: Phoenix Memory Autocallable

**Using StructuredProduct (Traditional)**:
```python
product = AutocallableProduct(
    product_id="PHOENIX_001",
    currency="USD",
    notional=1_000_000,
    basket=Basket(...),
    observation_schedule=ObservationSchedule(times=[0.25, 0.5, 0.75, 1.0]),
    barrier_schedule=BarrierSchedule(levels={...}),
    coupon=CouponDefinition(rate=0.08, memory=True),
    maturity=1.0
)

engine = PricingEngine(product=product, market_data=market, models=models)
result = engine.price()
```

**Using ComposablePayoff (Modern)**:
```python
payoff = ComposablePayoff(components=[
    MemoryCoupon(rate=0.02, barrier=0.70),
    AutocallComponent(barrier=0.95),
    WorstOfPut(strike=1.0, participation=1.0)
])

engine = PricingEngine(product=payoff, market_data=market, models=models)
result = engine.price()
```

### Example 2: Reverse Convertible with Barrier

**Using StructuredProduct**:
```python
product = ReverseConvertible(
    product_id="RC_001",
    currency="USD",
    notional=1_000_000,
    basket=Basket(...),
    payoff=ReverseConvertiblePayoffDefinition(
        strike=1.0,
        coupon_rate=0.10,
        coupon_frequency="quarterly",
        barrier=0.70,
        barrier_type="european"
    ),
    maturity=1.0
)
```

**Using ComposablePayoff** (After refactoring):
```python
payoff = ComposablePayoff(components=[
    SimpleCoupon(rate=0.025, frequency="quarterly"),  # 10% annual = 2.5% quarterly
    EuropeanBarrier(barrier=0.70, knock_in=True),
    WorstOfPut(strike=1.0, participation=1.0)
])
```

---

## Summary

- **StructuredProduct**: Complete product definition with metadata and validation
- **ComposablePayoff**: Flexible payoff construction from components
- **PayoffComponent**: Reusable building blocks (coupons, barriers, autocalls)
- **PayoffEvaluator**: Optimized Monte Carlo evaluation for specific products

The system supports both approaches, allowing legacy products to coexist with modern compositional designs.
