# Cloud Base Height Retrieval Project: One-Page Executive Summary

**Date:** November 2024  
**Status:** Diagnostic phase complete; pivoting to physics-constrained approach  
**Target:** Manuscript submission in 10-12 weeks

---

## What We're Trying to Do

Retrieve cloud base height (CBH) from ER-2 aircraft camera imagery to spatially extend sparse lidar measurements. Goal: automated system that generalizes across different flights and meteorological conditions.

**Dataset:** 933 labeled samples (CPL-aligned) across 5 flights; 61,946 unlabeled images available.

---

## What We've Learned (The Hard Way)

### ❌ What Doesn't Work

1. **Image-only features fail cross-flight validation**
   - Self-supervised learning (masked autoencoder): R² < 0 on leave-one-flight-out CV
   - Reconstruction-based training learns texture, not geometry
   - Random embeddings actually outperform trained ones (!)

2. **Solar angles alone don't generalize**
   - Strong within-flight: R² = 0.70
   - Zero cross-flight: R² = -4.46 ± 7.09
   - Reason: temporal confounding (angles correlate with time-of-day cloud evolution, not physics)

3. **Spatial feature preservation insufficient**
   - Tested pooling, convolution, attention mechanisms
   - All variants: mean R² < 0 on cross-flight validation
   - Root cause: missing physical constraints

### ✅ What We Know Works

1. **Rigorous validation protocol**
   - Leave-one-flight-out cross-validation reveals true generalization
   - Random train/test split gave false confidence (inflated metrics)
   - Per-flight diagnostics essential

2. **Diagnostic framework**
   - Comprehensive ablation studies
   - Systematic failure mode analysis
   - Publishable negative results

---

## Path Forward: Physics-Constrained Hybrid

**Core insight:** Image features must be grounded in physical constraints to generalize.

### Proposed Approach

**Physical Features (flight-invariant):**
- **Shadow geometry:** CBH = shadow_length × tan(90° - SZA) — direct geometric constraint
- **Atmospheric profiles:** Boundary layer height, lifting condensation level from ERA5 reanalysis
- **Thermodynamic constraints:** Inversion height, moisture gradients, stability indices

**Hybrid Model:**
```
[Shadow + Atmospheric + Image features] → Gradient Boosted Trees → CBH estimate
```

### Implementation Plan (Next 4 Weeks)

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Shadow geometry extraction | Feature module + quick test |
| 2 | ERA5 integration | Atmospheric profiles for all flights |
| 3 | Physical baseline | LOO CV with physics-only features |
| 4 | Hybrid model | Combined physical + learned features |

**Success criterion:** R² > 0.3 on leave-one-flight-out CV (meaningful cross-flight generalization)

---

## Paper Framing

**Title concept:** *"Physics-Constrained Machine Learning for Cross-Flight Cloud Base Height Retrieval from Airborne Imagery"*

**Key contributions:**
1. **Methodological:** Demonstration that reconstruction-based SSL fails for geometric retrieval
2. **Diagnostic:** Why naive ML doesn't generalize (temporal confounding, missing physics)
3. **Practical:** Hybrid framework combining shadow geometry + atmospheric state + learned features
4. **Validation:** Rigorous cross-flight evaluation protocol

**Story arc:**
- CBH is important → Limited measurements → ML opportunity → **But naive approaches fail** → Physics constraints necessary → Hybrid solution → Lessons for ML in geophysics

**Target venues:**
- Tier 1: *Geophysical Research Letters* (if results strong)
- Tier 2: *Artificial Intelligence for the Earth Systems*
- Tier 3: Methods/negative results journals

---

## Bottom Line

We've learned **why image-only approaches fail** for this problem. The next 2-3 weeks will determine if **physics-constrained methods succeed**. Either outcome is publishable:

- **If physics works:** Novel hybrid framework for CBH retrieval (positive contribution)
- **If physics fails:** Rigorous analysis of ML generalization limits (methodological contribution)

Both paths lead to a manuscript. The diagnostic work already completed has scientific value.

---

## Discussion Points

1. Shadow detection over ocean—feasible or problematic?
2. ERA5 spatial resolution (25 km)—adequate or need higher resolution?
3. Should we build cloud regime-specific models or universal model?
4. Data release policy—can we publish the CPL-aligned dataset?

**Full technical report:** See `docs/project_status_report.pdf` (16 pages, physics-focused)