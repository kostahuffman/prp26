# Products Taxonomy

Products are can be categorized using a hierarchy similar to the **GICS (Global Industry Classification Standard)**. The industry standard for this is maintained by **EUSIPA** (European Structured Investment Products Association) and the **SSPA** (Swiss Structured Products Association).

Below is a robust, institutional-grade taxonomy designed for a professional "Product Engine" (like the one in your `prp26` project).

---

## 1. The SP-Taxonomy (The "GICS" of S. Products)

This hierarchy organizes products by their risk-return profile and the behavior of the principal.

| Level | Name | Example / Description |
| --- | --- | --- |
| **Level 1** | **Investment Class** | Primary Split: **Investment** (Capital Preservation) vs. **Leverage** (Trading/Spec). |
| **Level 2** | **Category** | **Yield Enhancement**, **Participation**, or **Capital Protection**. |
| **Level 3** | **Product Type** | The core payoff logic (e.g., **Autocallable**, **Reverse Convertible**, **Tracker**). |
| **Level 4** | **Subtype/Feature** | Specific modifiers (e.g., **Phoenix**, **Step-down**, **Worst-of**, **Memory**). |

---

## 2. Institutional Product Catalog

### A. Yield Enhancement (The "Income" Desks)

These are the most popular products in private banking and institutional "search for yield." They typically trade sideways-to-bullish.

1. **Reverse Convertibles (RC)**
   * *Standard:* High coupon, but you might be "converted" into the stock if it's below the strike at maturity.
   * *Barrier Reverse Convertible (BRC):* Only converted if a downside barrier is touched.

2. **Autocallables (The "Express" Family)**
   * *Classic Autocallable:* Redeems early if the underlying is above a trigger level.
   * *Phoenix Autocallable:* Pays a periodic coupon even if not autocalled, as long as a "coupon barrier" isn't breached.
   * *Step-down Autocall:* The trigger level for the early redemption decreases over time (making it easier to call).
   * *Memory/Snowball:* If a coupon is missed because of a price dip, it "accumulates" and is paid later if the price recovers.

### B. Participation (The "Growth" Desks)

These allow for 1:1 or leveraged exposure to an underlying, often with a "cushion" or "bonus" feature.

1. **Tracker Certificates:** Pure 1:1 delta exposure to an index or basket (often used for niche themes).
2. **Bonus Certificates:** Provides 1:1 participation, but offers a "Bonus" minimum return if a downside barrier is never touched.
3. **Outperformance/Sprint:** Provides >1x or <1x participation in a specific upside range, capped at a certain level.
4. **Twin-Win:** If the stock goes up, you get the gain. If it goes down (but stays above the barrier), the loss is turned into a positive gain.

### C. Capital Protected (The "Conservative" Desks)

Designed for risk-averse institutions (pension funds, insurance) where the principal is (nominally) guaranteed.

1. **Equity Linked Notes (ELN):** Zero-coupon bond + Call option. 100% protection with some upside participation.
2. **Shark Fin (Barrier Digital):** Full protection, but if the underlying rises *too* much and hits a "knock-out" level, the investor only gets a small rebate instead of the full gain.
3. **Range Accrual:** Pays a coupon for every day the underlying stays within a specific "corridor."

---

## 3. The "Institutional Size" Modifier List

When building an engine, you don't just define "an Autocallable." You apply these institutional modifiers that define the pricing complexity:

* **Underlying Type:** Single Stock, Index, Basket, or **Worst-Of** (Performance is tied to the "losing" stock in a set).
* **Quanto:** Removes FX risk (e.g., a USD-denominated note on the Nikkei 225 index).
* **Barrier Style:**
  * *European:* Barrier only checked at the very end (Maturity).
  * *American:* Barrier checked every second (Continuous).
  * *Bermudan:* Barrier checked on specific dates (Monthly/Quarterly).

* **Mountain Range (Exotics):**
  * *Altiplano:* High coupon paid if none of the stocks in a basket ever hit a barrier.
  * *Himalaya:* Periodically "locks in" the best-performing stock and removes it from the basket.

---

## 4. Schema

Since you are a developer, here is how you might represent this hierarchy in a Python/JSON schema for your engine:

```json
{
  "product_id": "SP-7742",
  "taxonomy": {
    "class": "Investment",
    "category": "Yield Enhancement",
    "type": "Autocallable",
    "subtype": "Phoenix Memory Step-Down"
  },
  "structure": {
    "underlyings": ["AAPL.OQ", "MSFT.OQ", "GOOGL.OQ"],
    "payoff_logic": "Worst-Of",
    "barriers": {
      "autocall": [1.0, 0.95, 0.90, 0.85],
      "coupon": 0.70,
      "protection": 0.60
    },
    "currency_logic": "Quanto"
  }
}
```

## 5. Implementation in Python

Example taxonomy dataclass:

```python
from dataclasses import dataclass

@dataclass
class ProductTaxonomy:
    """EUSIPA/SSPA-style product classification."""
    
    investment_class: str  # "Investment" or "Leverage"
    category: str  # "Yield Enhancement", "Participation", "Capital Protection"
    product_type: str  # "Autocallable", "Reverse Convertible", "Bonus Certificate", etc.
    subtype: str  # "Phoenix", "Memory", "Step-Down", "Worst-Of", etc.
    
    def to_dict(self):
        return {
            "class": self.investment_class,
            "category": self.category,
            "type": self.product_type,
            "subtype": self.subtype
        }
```

## 6. Common Product Mappings

| Product | Class | Category | Type | Subtype |
|---------|-------|----------|------|---------|
| Phoenix Autocallable | Investment | Yield Enhancement | Autocallable | Phoenix Memory |
| Reverse Convertible | Investment | Yield Enhancement | Reverse Convertible | Standard |
| Barrier Reverse Convertible | Investment | Yield Enhancement | Reverse Convertible | Barrier |
| Bonus Certificate | Investment | Participation | Bonus Certificate | Standard |
| Tracker Certificate | Investment | Participation | Tracker | 1:1 Delta |
| Capital Protected Note | Investment | Capital Protection | ELN | Zero-Strike Call |
| Twin-Win | Investment | Participation | Twin-Win | Standard |

---

## References

- **EUSIPA:** European Structured Investment Products Association
- **SSPA:** Swiss Structured Products Association
- **GICS:** Global Industry Classification Standard (for equities)
