# Sprint 6 - Task 1.3: Comprehensive Error Analysis Report

**Generated**: 2025-11-11T20:29:03.151361

---

## 1. Worst-Performing Samples

**Threshold**: 200 m
**Samples exceeding threshold**: 2 (0.21%)
**Mean error (flagged samples)**: 216.2 m
**Maximum error**: 218.7 m

### Top Worst Samples

| Sample ID | Flight ID | Error (m) |
|-----------|-----------|-----------|
| 351 | 0 | 218.7 |
| 675 | 1 | 213.8 |
| 854 | 3 | 174.1 |
| 856 | 3 | 140.9 |
| 662 | 1 | 137.2 |
| 210 | 0 | 129.9 |
| 725 | 2 | 126.9 |
| 852 | 3 | 119.3 |
| 259 | 0 | 108.9 |
| 641 | 1 | 107.5 |

---

## 2. Correlation Analysis

Analysis of error correlations with input features:

| Feature | Correlation | P-value | Significant |
|---------|-------------|---------|-------------|
| sza | 0.0227 | 4.8888e-01 | No |
| saa | -0.0029 | 9.3000e-01 | No |
| blh | 0.0079 | 8.0857e-01 | No |
| lcl | 0.0200 | 5.4181e-01 | No |
| t2m | 0.0424 | 1.9558e-01 | No |
| d2m | 0.0159 | 6.2764e-01 | No |

### Interpretation

No significant correlations found (p < 0.05).

---

## 3. Per-Flight Error Analysis

| Flight | N Samples | Mean Error (m) | Std Error (m) | Median Error (m) |
|--------|-----------|----------------|---------------|------------------|

---

## 4. Statistical Significance Tests

### ANOVA Across Flights

**F-statistic**: 0.0000
**P-value**: 1.0000e+00
**Conclusion**: Insufficient flights for ANOVA

**Flights tested**: 

---

## 5. Summary and Recommendations

- **Flight consistency**: No significant differences between flights. Model generalizes well across flights.

---

**Report generated**: 2025-11-11T20:29:03.151414
