# Architecture

The engine follows a modular, institutional design:

```
┌─────────────────────────────────────────┐
│         StructuredProduct                │
│  (Product Definition + JSON Schema)      │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│          PricingEngine                   │
│  ┌────────────────────────────────────┐ │
│  │  ModelBundle                        │ │
│  │   • VolatilityModel (LV/SLV)      │ │
│  │   • CorrelationModel (Skew)        │ │
│  │   • RateModel                       │ │
│  │   • DividendModel                   │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │  CalibrationEngine                  │ │
│  │   • SLV two-stage calibration      │ │
│  │   • Correlation from market        │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │  PathGenerator                      │ │
│  │   • SLV dynamics                    │ │
│  │   • Correlated Brownian motions    │ │
│  │   • GPU-accelerated (optional)     │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │  PayoffEvaluator                    │ │
│  │   • Snowball coupon state          │ │
│  │   • Early termination              │ │
│  │   • Callability probabilities      │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │  RiskEngine                         │ │
│  │   • Adjoint Greeks (AAD)           │ │
│  │   • Bump-and-reprice               │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│       PersistenceLayer                   │
│  • EngineSnapshot (save/load)            │
│  • Audit trail                           │
│  • Model versioning                      │
└─────────────────────────────────────────┘
```

## Requirements

- Python 3.9+
- NumPy, SciPy
- Optional: CuPy (for GPU), Matplotlib (for visualization)

## License

MIT License - see [LICENSE](../LICENSE)