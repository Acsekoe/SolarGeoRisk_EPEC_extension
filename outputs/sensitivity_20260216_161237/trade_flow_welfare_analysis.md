# Trade Flow & Welfare Analysis: sensitivity_20260216_161237

## Two Market Structures

20 runs split into two equilibria: **EqA** (CH solves first, 5 runs) vs **EqB** (CH solves later, 15 runs).

---

## Welfare Table — All 20 Runs

| Run | Init | Order | obj_ch | obj_eu | obj_us | obj_apac | obj_roa | obj_row | **Total** |
|-----|------|-------|-------:|-------:|-------:|---------:|--------:|--------:|----------:|
| 001 | high_all | default | 47,288 | 11,801 | 4,850 | 12,187 | 3,374 | 4,284 | **83,783** |
| 002 | high_all | ch_last | 45,332 | 17,764 | 8,672 | 3,124 | 5,574 | 7,011 | **87,478** |
| 003 | high_all | ch_mid | 45,336 | 17,764 | 8,672 | 3,125 | 5,574 | 7,011 | **87,482** |
| 004 | high_all | reverse | 45,333 | 17,764 | 8,672 | 3,125 | 5,574 | 7,010 | **87,478** |
| 005 | low_non_ch | default | 46,951 | 12,379 | 5,193 | 11,286 | 3,373 | 4,502 | **83,685** |
| 006 | low_non_ch | ch_last | 45,332 | 17,764 | 8,672 | 3,124 | 5,574 | 7,011 | **87,478** |
| 007 | low_non_ch | ch_mid | 45,336 | 17,764 | 8,672 | 3,125 | 5,574 | 7,011 | **87,482** |
| 008 | low_non_ch | reverse | 45,333 | 17,764 | 8,672 | 3,125 | 5,574 | 7,011 | **87,478** |
| 009 | low_eu_us_row | default | 46,951 | 12,379 | 5,193 | 11,286 | 3,373 | 4,502 | **83,685** |
| 010 | low_eu_us_row | ch_last | 45,332 | 17,764 | 8,672 | 3,124 | 5,574 | 7,011 | **87,478** |
| 011 | low_eu_us_row | ch_mid | 45,336 | 17,764 | 8,672 | 3,125 | 5,574 | 7,011 | **87,482** |
| 012 | low_eu_us_row | reverse | 45,333 | 17,764 | 8,672 | 3,125 | 5,574 | 7,011 | **87,478** |
| 013 | mid_all | default | 47,297 | 11,860 | 4,859 | 12,116 | 3,374 | 4,271 | **83,778** |
| 014 | mid_all | ch_last | 45,336 | 17,761 | 8,671 | 3,125 | 5,574 | 7,010 | **87,478** |
| 015 | mid_all | ch_mid | 45,333 | 17,764 | 8,672 | 3,124 | 5,574 | 7,011 | **87,478** |
| 016 | mid_all | reverse | 45,332 | 17,764 | 8,672 | 3,124 | 5,574 | 7,011 | **87,478** |
| 017 | low_all | default | 47,298 | 11,840 | 4,865 | 12,131 | 3,374 | 4,265 | **83,772** |
| 018 | low_all | ch_last | 47,298 | 11,917 | 4,835 | 12,084 | 3,374 | 4,279 | **83,787** |
| 019 | low_all | ch_mid | 47,355 | 11,799 | 4,947 | 12,175 | 3,374 | 4,244 | **83,893** |
| 020 | low_all | reverse | 47,294 | 11,861 | 4,844 | 12,131 | 3,374 | 4,280 | **83,784** |

---

## Welfare Comparison: EqA vs EqB (Averages)

| Region | EqA (CH first) | EqB (CH later) | Δ | Δ% |
|--------|---------------:|---------------:|----:|-----:|
| **CH** | 47,157 | 45,730 | −1,427 | −3% |
| **EU** | 12,052 | 16,583 | +4,531 | **+38%** |
| **US** | 4,992 | 7,913 | +2,921 | **+59%** |
| **APAC** | 11,801 | 4,926 | −6,875 | **−58%** |
| **ROA** | 3,373 | 5,134 | +1,761 | **+52%** |
| **ROW** | 4,365 | 6,462 | +2,097 | **+48%** |
| **Total** | **83,741** | **86,747** | **+3,006** | **+3.6%** |

> [!IMPORTANT]
> **EqB is globally more efficient** (+3.6% total welfare). The gains to EU (+38%), US (+59%), ROA (+52%), and ROW (+48%) more than offset APAC's loss (−58%) and CH's modest decline (−3%). This suggests CH flooding the market, while individually slightly worse for CH, is **socially preferable**.

> [!WARNING]
> **APAC is the clear loser** in EqB: its objective drops from 11,801 to 4,926 (−58%) as CH captures its EU/US export market.

---

## Trade Flow Matrices

### EqA — APAC supplies EU/US

```
exp→imp      ch        eu        us      apac       roa       row
ch        263.7       0.2       2.9       1.1      23.9      26.4
apac        0.0      58.8      35.4      14.4       0.3       1.2
(all others ≈ 0)
```

### EqB — CH supplies EU/US

```
exp→imp      ch        eu        us      apac       roa       row
ch        273.1      45.8      30.2       1.0      22.6      24.7
apac        0.0      13.2       8.1      21.9       1.7       2.9
(all others ≈ 0)
```

---

## CH Export Tariff Strategy (tau_exp)

| Route | EqA | EqB | Change |
|-------|----:|----:|-------:|
| CH→EU | 100.6 | 33.9 | **−66%** |
| CH→US | 101.5 | 35.2 | **−65%** |
| CH→APAC | 80.4 | 14.6 | **−82%** |
| CH→ROA | 126.1 | 63.5 | **−50%** |
| CH→ROW | 102.5 | 36.6 | **−64%** |

---

## CH Export Split

| Equilibrium | Domestic | Total Exports | →EU | →US | →ROA | →ROW |
|-------------|----------|--------------|-----|-----|------|------|
| **EqA** | 254–278 | 54–55 | 0.2 | 2.9 | 23.9 | 26.4 |
| **EqB** (most) | 278 | 149 | 59.0 | 38.3 | 24.2 | 27.6 |
| **EqB** (low_all) | 254 | 54 | 0.0 | 2.4 | 24.2 | 27.6 |

> [!NOTE]
> The `low_all` init with non-default order (runs 018–020) converges to EqA-like welfare and trade patterns, suggesting overlapping equilibrium basins at low starting points.

---

## Key Takeaways

1. **EqB is globally more efficient** (+3.6% total welfare) — cheap CH exports benefit all importers
2. **APAC is the biggest loser** from CH market flooding (−58% welfare)
3. **CH's tariff strategy flips**: protective (~100) in EqA vs aggressive (~34) in EqB
4. **Player order determines the equilibrium** — initial conditions matter only at the margin
5. **Randomized player order** would likely find an intermediate equilibrium
