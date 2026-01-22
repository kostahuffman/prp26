# Payoff Architecture Guide

## Overview

This document clarifies the architecture for product pricing, specifically the relationship between **StructuredProduct**, **ComposablePayoff**, **PayoffEvaluators**, and **PayoffComponents**.

## Architecture Layers

### 1. **StructuredProduct** (Product Definitions with Evaluators)

**Location**: `products/autocallable.py`, `products/reverse_convertible.py`, `products/vanilla_option.py`

**Purpose**: Product classes that encapsulate all product details including:
- Product metadata (ID, currency, notional, dates)
- Basket/underlying definitions
- Observation schedules
- Barrier schedules
- Coupon definitions
- Taxonomy
- **Required**: Must implement `get_evaluator()` method

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

# Product must provide its evaluator
evaluator = product.get_evaluator()  # Returns PhoenixPayoffEvaluator
```

**Key Requirement**:
- All `StructuredProduct` subclasses **must** implement `get_evaluator()`
- This is enforced via abstract method in base class
- The evaluator is used by `PricingEngine` for Monte Carlo simulation

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

# Can be passed directly to PricingEngine
engine = PricingEngine(product=payoff, ...)
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

**Location**: `products/payoffs/evaluators/phoenix.py`, `products/payoffs/evaluators/snowball.py`, etc.

**Purpose**: Complete payoff evaluation for specific product types - optimized for Monte Carlo simulation in `PricingEngine`

**Interface**:
```python
class PayoffEvaluator(ABC):
    @abstractmethod
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
- **ReverseConvertibleEvaluator**: Reverse convertible notes
- **VanillaOptionEvaluator**: Simple European options

**When to Create**:
- New product type with standardized structure
- Product-specific optimizations needed
- Complex state management across observations
- Performance-critical calculations

**Connection to StructuredProduct**:
- Every `StructuredProduct` subclass returns an evaluator via `get_evaluator()`
- The evaluator encapsulates the pricing logic
- Clear separation: product definition vs. pricing implementation

---

## Design Patterns

### Pattern 1: StructuredProduct → PayoffEvaluator (Mandatory)

```python
# 1. Define product with StructuredProduct class
product = AutocallableProduct(
    product_id="PHOENIX_001",
    currency="USD",
    notional=1_000_000,
    basket=basket,
    observation_schedule=ObservationSchedule(times=[0.25, 0.5, 0.75, 1.0]),
    barrier_schedule=BarrierSchedule(...),
    coupon=CouponDefinition(rate=0.08, memory=True),
    maturity=1.0
)

# 2. Product provides its evaluator (required by abstract method)
evaluator = product.get_evaluator()  # Returns PhoenixPayoffEvaluator

# 3. PricingEngine calls get_evaluator() automatically
engine = PricingEngine(product=product, ...)
result = engine.price()  # Engine uses evaluator internally
```

**Usage**: All structured products - Phoenix, Snowball, Reverse Convertible, Vanilla Options

**Key Points**:
- `get_evaluator()` is abstract in `StructuredProduct` base class
- Must be implemented by all concrete product classes
- Returns configured evaluator ready for Monte Carlo simulation
- Clean separation between product definition and pricing logic

---

### Pattern 2: ComposablePayoff (Direct Evaluation)

```python
# 1. Build payoff from components
payoff = ComposablePayoff(components=[
    MemoryCoupon(rate=0.02, barrier=0.70),
    AutocallComponent(barrier=0.95),
    WorstOfPut(strike=1.0)
])

# 2. PricingEngine uses ComposablePayoff directly
engine = PricingEngine(product=payoff, ...)
result = engine.price()  # Engine calls payoff.evaluate_path()
```

**Usage**: Custom/exotic products, JSON-driven construction, rapid prototyping

**Key Points**:
- `ComposablePayoff` has its own `evaluate_path()` method
- No separate evaluator needed
- Components handle the evaluation logic
- Extremely flexible for custom products

---

## PricingEngine Integration

The `PricingEngine` handles both approaches seamlessly:

```python
def _evaluate_payoff(self, paths: np.ndarray, initial_spots: np.ndarray) -> dict:
    """Evaluate payoff for paths."""
    if isinstance(self.product, ComposablePayoff):
        # Use compositional payoff system
        return self.product.evaluate_path(
            paths=paths, times=self.times, initial_spots=initial_spots, notional=self.notional
        )
    
    # All StructuredProduct subclasses must implement get_evaluator()
    evaluator = self.product.get_evaluator()
    return evaluator.evaluate(paths, initial_spots=initial_spots)
```

**Key Points**:
- No `hasattr()` checks or fallbacks needed
- Clean interface via abstract method
- `ComposablePayoff` and `StructuredProduct` are both supported
- Type safety enforced at class level

---

## Organizational Rules

### File Organization

```
products/
├── base.py                    # StructuredProduct base class (with abstract get_evaluator)
├── autocallable.py            # AutocallableProduct → Phoenix/Snowball evaluators
├── reverse_convertible.py    # ReverseConvertible → ReverseConvertibleEvaluator
├── vanilla_option.py          # VanillaOption → VanillaOptionEvaluator
├── participation.py           # Participation products
└── payoffs/
    ├── base.py                # PayoffComponent, ComposablePayoff, PayoffState
    ├── coupon.py              # Coupon components (MemoryCoupon, SnowballCoupon)
    ├── autocall.py            # AutocallComponent
    ├── downside.py            # Downside components (WorstOfPut)
    ├── rainbow.py             # Rainbow components
    └── evaluators/
        ├── base.py            # PayoffEvaluator abstract base class
        ├── phoenix.py         # PhoenixPayoffEvaluator
        ├── snowball.py        # SnowballPayoffEvaluator
        ├── reverse_convertible.py  # ReverseConvertibleEvaluator
        └── vanilla.py         # VanillaOptionEvaluator
```

### Naming Conventions

1. **PayoffComponent**: `<Feature>Component` or `<Feature>Coupon`
   - Examples: `AutocallComponent`, `MemoryCoupon`, `RainbowComponent`
   
2. **PayoffEvaluator**: `<ProductType>PayoffEvaluator` or `<ProductType>Evaluator`
   - Examples: `PhoenixPayoffEvaluator`, `VanillaOptionEvaluator`
   
3. **StructuredProduct**: `<ProductType>Product` or `<ProductType>`
   - Examples: `AutocallableProduct`, `ReverseConvertible`, `VanillaOption`
   - **Must implement**: `get_evaluator()` method

4. **Product-specific payoff data**: `<ProductType>PayoffDefinition` (dataclass)
   - Examples: `ReverseConvertiblePayoffDefinition`, `ParticipationPayoffDefinition`
   - Purpose: Serialization/deserialization of product-specific payoff parameters

---

## When to Use Which Approach

| Scenario | Use This | Reason |
|----------|----------|--------|
| Standard Phoenix autocallable | `AutocallableProduct` | Well-tested, validated product class with evaluator |
| Standard Reverse Convertible | `ReverseConvertible` | Complete product definition with coupon logic |
| Vanilla option | `VanillaOption` | Simple option with evaluator |
| Custom multi-feature product | `ComposablePayoff` | Flexible composition without new classes |
| JSON-driven product UI | `ComposablePayoff` | Natural JSON mapping and dynamic construction |
| Product catalog/booking | `StructuredProduct` subclass | Complete metadata, validation, serialization |
| Rapid prototyping | `ComposablePayoff` | Quick iteration without boilerplate |
| Performance-critical | `PayoffEvaluator` | Optimized for Monte Carlo paths |

---

## Creating New Products

### Option 1: Create StructuredProduct + Evaluator

**When**: Standard product with fixed structure and validation requirements

**Steps**:
1. Create `<ProductType>` class extending `StructuredProduct`
2. Implement all required fields and validation
3. Create `<ProductType>Evaluator` extending `PayoffEvaluator`
4. Implement `get_evaluator()` in product class
5. Add tests

**Example**:
```python
# products/barrier_note.py
class BarrierNote(StructuredProduct):
    def __init__(self, product_id, currency, notional, barrier, ...):
        super().__init__(product_id, currency, notional)
        self.barrier = barrier
        # ... other fields
        
    def get_evaluator(self):
        from .payoffs.evaluators import BarrierNoteEvaluator
        return BarrierNoteEvaluator(
            barrier=self.barrier,
            notional=self.notional,
            # ... other params
        )

# products/payoffs/evaluators/barrier_note.py
class BarrierNoteEvaluator(PayoffEvaluator):
    def __init__(self, barrier, notional, ...):
        self.barrier = barrier
        self.notional = notional
        
    def evaluate(self, paths, initial_spots):
        # Pricing logic here
        ...
```

### Option 2: Use ComposablePayoff with Existing Components

**When**: Product can be built from existing components, custom/one-off products

**Steps**:
1. Identify required components (coupons, barriers, etc.)
2. Compose using `ComposablePayoff`
3. Pass directly to `PricingEngine`

**Example**:
```python
# Custom product via composition
payoff = ComposablePayoff(components=[
    MemoryCoupon(rate=0.02, barrier=0.65),
    AutocallComponent(barrier=0.90),
    WorstOfPut(strike=0.80, participation=1.5)
])

engine = PricingEngine(product=payoff, ...)
result = engine.price()
```

---

## Migration Notes

### Previous Architecture (Removed)
- ❌ `hasattr()` checks for `get_evaluator()`
- ❌ Legacy fallback in engine for products without evaluators
- ❌ Hybrid/Bridge pattern (Pattern 3)

### Current Architecture (Enforced)
- ✅ Abstract `get_evaluator()` in `StructuredProduct` base class
- ✅ All products must implement evaluator
- ✅ Clean type checking via `isinstance()`
- ✅ Two clear patterns: StructuredProduct or ComposablePayoff

### Benefits
- Type safety at compile time
- No runtime attribute checks
- Clear contract for all products
- Easier to understand and maintain
- Better IDE support and autocomplete

---

## Examples

### Example 1: Phoenix Memory Autocallable

**Using StructuredProduct**:
```python
# Create product definition
product = AutocallableProduct(
    product_id="PHOENIX_001",
    currency="USD",
    notional=1_000_000,
    basket=Basket(
        underlyings=[Underlying(symbol="SPX", asset_class="equity")],
        weights=[1.0],
        worst_of=False
    ),
    observation_schedule=ObservationSchedule(times=[0.25, 0.5, 0.75, 1.0]),
    barrier_schedule=BarrierSchedule(levels={
        0.25: 0.95, 0.5: 0.95, 0.75: 0.95, 1.0: 0.95
    }),
    coupon=CouponDefinition(rate=0.02, memory=True),  # 2% per quarter
    maturity=1.0
)

# Product automatically provides evaluator
engine = PricingEngine(product=product, spot_data=spot, volatility=vol, ...)
result = engine.price()  # Engine calls product.get_evaluator() internally
```

**Using ComposablePayoff**:
```python
# Compose payoff from components
payoff = ComposablePayoff(components=[
    MemoryCoupon(rate=0.02, barrier=0.70),
    AutocallComponent(barrier=0.95),
    WorstOfPut(strike=1.0, participation=1.0)
])

# Pass directly to engine
engine = PricingEngine(product=payoff, spot_data=spot, volatility=vol, ...)
result = engine.price()  # Engine calls payoff.evaluate_path() internally
```

**Key Differences**:
- `StructuredProduct`: Full product metadata, validation, JSON serialization
- `ComposablePayoff`: Lightweight, flexible, direct evaluation

### Example 2: Vanilla European Option

**Using StructuredProduct**:
```python
product = VanillaOption(
    product_id="CALL_SPX_1Y",
    currency="USD",
    notional=100_000,
    underlying=Underlying(symbol="SPX", asset_class="equity"),
    strike=1.0,  # At-the-money (100%)
    maturity=1.0,
    option_type="call"
)

# Product provides VanillaOptionEvaluator
engine = PricingEngine(product=product, ...)
result = engine.price()
```

### Example 3: Testing Equivalence

See `tests/test_payoff_equivalence.py` for comprehensive tests that verify:
- Same input parameters produce same outputs
- Both approaches handle early termination correctly
- Memory coupon accumulation matches
- Pricing results converge within statistical error

```python
def test_phoenix_equivalence():
    # Create both versions with identical parameters
    product = AutocallableProduct(...)  # Phoenix via StructuredProduct
    payoff = ComposablePayoff([...])   # Phoenix via components
    
    # Price with same random seed
    np.random.seed(42)
    result1 = engine1.price(n_paths=10000)
    
    np.random.seed(42)
    result2 = engine2.price(n_paths=10000)
    
    # Verify equivalence
    assert abs(result1["price"] - result2["price"]) < 3 * combined_std
```

---

## Summary

- **StructuredProduct**: Complete product definition with **required** `get_evaluator()` method
  - Returns configured `PayoffEvaluator` for pricing
  - Use for standard products with validation and metadata
  
- **ComposablePayoff**: Flexible payoff construction from `PayoffComponent` blocks
  - Has its own `evaluate_path()` method
  - Use for custom/exotic products and rapid prototyping
  
- **PayoffEvaluator**: Optimized Monte Carlo evaluation
  - Returned by `StructuredProduct.get_evaluator()`
  - Performance-critical path evaluation logic
  
- **PayoffComponent**: Reusable building blocks for `ComposablePayoff`
  - Coupons, barriers, autocalls, downside participation
  - Mix and match for custom products

**The architecture enforces clean separation**:
- Product definition → `StructuredProduct` or `ComposablePayoff`
- Pricing logic → `PayoffEvaluator` or `PayoffComponent.evaluate()`
- No hybrid patterns, no runtime checks, clear interfaces
