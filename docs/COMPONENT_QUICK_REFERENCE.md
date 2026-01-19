# Quick Reference: Payoff Components

## Component Types

### Coupon Components

| Component | Description | Key Parameters | Example |
|-----------|-------------|----------------|---------|
| `MemoryCoupon` | Accumulates unpaid coupons | `rate`, `barrier` | Phoenix autocallables |
| `SnowballCoupon` | Accumulating with growth | `rate`, `barrier`, `growth_rate` | Snowball products |
| `StepUpCoupon` | Increasing coupon rates | `rate_schedule`, `barrier`, `memory` | Long-term autocallables |
| `StepDownCoupon` | Decreasing coupon rates | `rate_schedule`, `barrier`, `memory` | Early-exit incentive products |
| `CouponComponent` | Simple fixed coupon | `rate`, `barrier`, `memory` | Basic coupons |

### Rainbow Components

| Component | Description | Key Parameters | Example |
|-----------|-------------|----------------|---------|
| `RainbowCall` | Call on ranked asset | `selection`, `rank`, `strike`, `participation` | Best-of call |
| `RainbowPut` | Put on ranked asset | `selection`, `rank`, `strike`, `participation` | Worst-of put |
| `RainbowDigital` | Digital on ranked asset | `selection`, `barrier`, `payout`, `trigger_type` | Best-of digital coupon |
| `SpreadOption` | Option on asset spread | `asset1_index`, `asset2_index`, `strike`, `option_type` | Outperformance options |

**Selection Options**: `"best"`, `"worst"`, `"median"`, `"nth"`

### Autocall Components

| Component | Description | Key Parameters | Example |
|-----------|-------------|----------------|---------|
| `AutocallComponent` | Early termination trigger | `barrier` | Standard autocallables |

### Downside Components

| Component | Description | Key Parameters | Example |
|-----------|-------------|----------------|---------|
| `WorstOfPut` | Put on worst performing asset | `strike`, `participation` | Capital protection |
| `DownsideComponent` | Generic downside participation | `participation`, `protection_level` | Custom downside |

## Usage Patterns

### Pattern 1: Standard Phoenix
```python
from prp26.products.payoffs import (
    ComposablePayoff, MemoryCoupon, AutocallComponent, WorstOfPut
)

phoenix = ComposablePayoff(components=[
    MemoryCoupon(rate=0.02, barrier=0.70),
    AutocallComponent(barrier=0.95),
    WorstOfPut(strike=1.0, participation=1.0),
])
```

### Pattern 2: Best-of with Step-up Coupons
```python
from prp26.products.payoffs import (
    ComposablePayoff, StepUpCoupon, AutocallComponent, RainbowCall
)

best_of_autocall = ComposablePayoff(components=[
    StepUpCoupon(
        rate_schedule=[0.06, 0.08, 0.10, 0.12],
        barrier=0.70,
        memory=True
    ),
    AutocallComponent(barrier=0.95),
    RainbowCall(selection="best", strike=1.0, participation=1.0),
])
```

### Pattern 3: Spread Autocallable
```python
from prp26.products.payoffs import (
    ComposablePayoff, RainbowDigital, AutocallComponent, SpreadOption
)

spread_product = ComposablePayoff(components=[
    RainbowDigital(selection="worst", barrier=0.70, payout=0.08),
    AutocallComponent(barrier=1.0),
    SpreadOption(asset1_index=0, asset2_index=1, strike=0.0, option_type="call"),
])
```

### Pattern 4: Reverse Convertible with Step-down
```python
from prp26.products.payoffs import (
    ComposablePayoff, StepDownCoupon, RainbowPut
)

reverse_conv = ComposablePayoff(components=[
    StepDownCoupon(
        rate_schedule={0.0: 0.12, 0.25: 0.10, 0.50: 0.08, 0.75: 0.06},
        barrier=1.0,
        memory=False
    ),
    RainbowPut(selection="worst", strike=1.0, participation=1.0),
])
```

## JSON Serialization

### Component to JSON
```python
component = StepUpCoupon(
    rate_schedule=[0.06, 0.08, 0.10],
    barrier=0.70,
    memory=True
)

json_data = component.to_dict()
# {
#     "type": "step_up_coupon",
#     "rate_schedule": [0.06, 0.08, 0.10],
#     "barrier": 0.70,
#     "memory": true
# }
```

### JSON to Component
```python
from prp26.products.payoffs import PayoffRegistry

json_data = {
    "type": "rainbow_call",
    "selection": "best",
    "rank": 1,
    "strike": 1.0,
    "participation": 1.0
}

component = PayoffRegistry.create(json_data["type"], json_data)
```

### Full Payoff Serialization
```python
payoff = ComposablePayoff(components=[...])

# Serialize
json_dict = payoff.to_dict()

# Deserialize
restored = ComposablePayoff.from_dict(json_dict)
```

## Working with Evaluators

### Get Evaluator from Product
```python
from prp26.products import ReverseConvertible

product = ReverseConvertible(...)
evaluator = product.get_evaluator()
```

### Direct Evaluator Usage
```python
from prp26.products.payoffs import PhoenixPayoffEvaluator
import numpy as np

evaluator = PhoenixPayoffEvaluator(
    observation_times=[0.25, 0.5, 0.75, 1.0],
    autocall_barriers=[0.95, 0.95, 0.95, 0.95],
    coupon_barriers=[0.70, 0.70, 0.70, 0.70],
    coupon_rate=0.02,
    notional=1.0,
)

# Mock paths: (n_paths, n_steps, n_assets)
paths = np.random.randn(10000, 4, 3)
initial_spots = np.array([100.0, 150.0, 200.0])

result = evaluator.evaluate(paths, initial_spots)
# Returns: {
#     "payoffs": np.ndarray,
#     "cashflows": np.ndarray,
#     "terminated": np.ndarray,
#     "termination_times": np.ndarray,
# }
```

## Registry Usage

### List Available Components
```python
from prp26.products.payoffs import PayoffRegistry

component_types = PayoffRegistry.list_types()
print(component_types)
# ['simple_coupon', 'memory_coupon', 'snowball_coupon', 'step_up_coupon',
#  'step_down_coupon', 'autocall', 'downside', 'worst_of_put',
#  'rainbow_call', 'rainbow_put', 'rainbow_digital', 'spread_option']
```

### Register Custom Component
```python
from prp26.products.payoffs import PayoffRegistry, PayoffComponent

class CustomCoupon(PayoffComponent):
    def __init__(self, custom_param: float):
        self.custom_param = custom_param
    
    def evaluate(self, spots, initial_spots, time, state):
        # Custom logic
        ...
    
    def to_dict(self):
        return {"type": "custom_coupon", "custom_param": self.custom_param}
    
    @classmethod
    def from_dict(cls, data):
        return cls(custom_param=data["custom_param"])

PayoffRegistry.register("custom_coupon", CustomCoupon)
```

## Integration with PricingEngine

### Using ComposablePayoff
```python
from prp26.core import PricingEngine, ModelBundle
from prp26.marketdata import MarketDataSnapshot

payoff = ComposablePayoff(components=[...])

engine = PricingEngine(
    product=payoff,  # Can pass ComposablePayoff directly
    market_data=market_data,
    models=models,
    pricing_config={"n_paths": 100000}
)

result = engine.price()
```

### Using StructuredProduct (To Be Deprecated)
```python
product = ReverseConvertible(...)  # or any StructuredProduct

engine = PricingEngine(
    product=product,  # Engine calls product.get_evaluator()
    market_data=market_data,
    models=models,
)

result = engine.price()
```

## Common Recipes

### 1. Phoenix with Step-up Memory Coupons
```python
phoenix = ComposablePayoff(components=[
    StepUpCoupon(rate_schedule=[0.02, 0.025, 0.03, 0.035], barrier=0.70, memory=True),
    AutocallComponent(barrier=0.95),
    WorstOfPut(strike=1.0, participation=1.0),
])
```

### 2. Best-of Call with Fixed Coupon
```python
best_of = ComposablePayoff(components=[
    MemoryCoupon(rate=0.08, barrier=0.75),
    AutocallComponent(barrier=1.0),
    RainbowCall(selection="best", strike=1.0, participation=1.0),
])
```

### 3. Spread Option with Digital Coupons
```python
spread = ComposablePayoff(components=[
    RainbowDigital(selection="worst", barrier=0.80, payout=0.05),
    AutocallComponent(barrier=1.05),
    SpreadOption(asset1_index=0, asset2_index=1, strike=0.1, option_type="call"),
])
```

### 4. Snowball with Rainbow Downside
```python
snowball = ComposablePayoff(components=[
    SnowballCoupon(rate=0.02, barrier=0.70, growth_rate=0.01),
    AutocallComponent(barrier=1.0),
    RainbowPut(selection="worst", strike=1.0, participation=1.0),
])
```

## See Also

- [PAYOFF_ARCHITECTURE.md](PAYOFF_ARCHITECTURE.md) - Full architecture guide
- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - Change summary
- `examples/rainbow_and_step_coupons_demo.py` - Working examples
