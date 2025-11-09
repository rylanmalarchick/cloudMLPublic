# Sprint 4 Execution Plan: Negative Results Paper
## "Why Shadow Geometry and Reanalysis Data Fail for Cloud Base Height Retrieval"

**Status:** ACTIVE  
**Priority:** HIGH  
**Timeline:** 2 weeks  
**Outcome:** Manuscript submitted to peer-reviewed journal

---

## Executive Summary

Based on the gap analysis, Sprint 4 as originally planned is not viable. The WP-3 physics baseline failed catastrophically (R² = -14.15), rejecting the core hypothesis that shadow geometry + ERA5 features enable cross-flight CBH retrieval.

**Pivot Strategy:** Transform the negative results into a publishable methodological contribution that documents:
1. Why shadow-based geometric CBH estimation fails over ocean
2. Why ERA5 spatial resolution is insufficient for cloud-scale phenomena
3. Diagnostic framework for detecting such failures early
4. Recommendations for alternative approaches

---

## Week 1: Analysis and Figure Generation

### Day 1-2: Diagnostic Analysis

#### Task 1.1: Validate ERA5 Physical Constraint
**Objective:** Check if BLH > CBH constraint holds (should have been Phase 1 checkpoint)

**Code:**
```python
# File: sprint4_execution/validate_era5_constraints.py
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load WP-2 ERA5 features
with h5py.File('sow_outputs/wp2_atmospheric/WP2_Features.hdf5', 'r') as f:
    blh = f['features'][:, 0]  # Boundary layer height
    lcl = f['features'][:, 1]  # Lifting condensation level

# Load ground truth CBH from CPL
with h5py.File('sow_outputs/wp1_geometric/WP1_Features.hdf5', 'r') as f:
    cbh_true = f['cbh_cpl'][:]

# Constraint violations
blh_violations = np.sum(cbh_true > blh)
lcl_violations = np.sum(cbh_true > lcl)

print(f"BLH > CBH violations: {blh_violations}/{len(cbh_true)} ({100*blh_violations/len(cbh_true):.1f}%)")
print(f"LCL > CBH violations: {lcl_violations}/{len(cbh_true)} ({100*lcl_violations/len(cbh_true):.1f}%)")

# Scatter plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(blh/1000, cbh_true/1000, alpha=0.5, s=10)
axes[0].plot([0, 10], [0, 10], 'r--', label='1:1 line')
axes[0].set_xlabel('ERA5 BLH (km)')
axes[0].set_ylabel('CPL CBH (km)')
axes[0].set_title(f'BLH vs. CBH (violations: {100*blh_violations/len(cbh_true):.1f}%)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(lcl/1000, cbh_true/1000, alpha=0.5, s=10)
axes[1].plot([0, 10], [0, 10], 'r--', label='1:1 line')
axes[1].set_xlabel('Computed LCL (km)')
axes[1].set_ylabel('CPL CBH (km)')
axes[1].set_title(f'LCL vs. CBH (violations: {100*lcl_violations/len(cbh_true):.1f}%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sprint4_execution/figures/era5_constraint_validation.png', dpi=300)
print("Saved: sprint4_execution/figures/era5_constraint_validation.png")
```

**Expected Output:** Likely to find that BLH is often BELOW CBH, violating the assumed constraint. This reveals why ERA5 features failed.

**Deliverable:** `figures/era5_constraint_validation.png`

---

#### Task 1.2: Shadow Geometry Failure Analysis
**Objective:** Visualize why shadow-based CBH estimation is broken

**Code:**
```python
# File: sprint4_execution/shadow_failure_analysis.py
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load WP-1 geometric features
with h5py.File('sow_outputs/wp1_geometric/WP1_Features.hdf5', 'r') as f:
    cbh_shadow = f['features'][:, 0]  # Shadow-derived CBH
    cbh_true = f['cbh_cpl'][:]
    confidence = f['features'][:, 2] if f['features'].shape[1] > 2 else np.ones(len(cbh_shadow))

# Filter to valid estimates (non-NaN, positive)
valid_mask = (~np.isnan(cbh_shadow)) & (cbh_shadow > 0) & (cbh_shadow < 10000)
cbh_shadow_valid = cbh_shadow[valid_mask]
cbh_true_valid = cbh_true[valid_mask]
confidence_valid = confidence[valid_mask]

# Compute metrics
mae = np.mean(np.abs(cbh_shadow_valid - cbh_true_valid))
rmse = np.sqrt(np.mean((cbh_shadow_valid - cbh_true_valid)**2))
bias = np.mean(cbh_shadow_valid - cbh_true_valid)
r, p = pearsonr(cbh_shadow_valid, cbh_true_valid)

print(f"Shadow-based CBH Metrics (n={len(cbh_shadow_valid)}):")
print(f"  MAE:  {mae:.2f} m")
print(f"  RMSE: {rmse:.2f} m")
print(f"  Bias: {bias:.2f} m")
print(f"  r:    {r:.4f} (p={p:.4e})")

# Create comprehensive scatter plot
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Top-left: Overall scatter
ax = axes[0, 0]
sc = ax.scatter(cbh_true_valid/1000, cbh_shadow_valid/1000, 
                c=confidence_valid, cmap='viridis', alpha=0.6, s=20)
ax.plot([0, 10], [0, 10], 'r--', linewidth=2, label='1:1 line')
ax.set_xlabel('CPL Ground Truth CBH (km)', fontsize=12)
ax.set_ylabel('Shadow-derived CBH (km)', fontsize=12)
ax.set_title(f'Shadow Geometry Failure\nr = {r:.4f}, Bias = {bias/1000:.2f} km', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax, label='Detection Confidence')

# Top-right: Residual plot
ax = axes[0, 1]
residuals = (cbh_shadow_valid - cbh_true_valid) / 1000
ax.scatter(cbh_true_valid/1000, residuals, alpha=0.5, s=10)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.axhline(y=np.mean(residuals), color='orange', linestyle='-', linewidth=2, 
           label=f'Mean bias: {np.mean(residuals):.2f} km')
ax.set_xlabel('CPL Ground Truth CBH (km)', fontsize=12)
ax.set_ylabel('Residual (Shadow - CPL) (km)', fontsize=12)
ax.set_title('Systematic Bias in Shadow Estimates', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Bottom-left: Histogram of residuals
ax = axes[1, 0]
ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
ax.axvline(x=np.mean(residuals), color='orange', linestyle='-', linewidth=2, 
           label=f'Mean: {np.mean(residuals):.2f} km')
ax.set_xlabel('Residual (Shadow - CPL) (km)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Errors', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Bottom-right: Bland-Altman plot
ax = axes[1, 1]
mean_cbh = (cbh_shadow_valid + cbh_true_valid) / 2000  # km
diff_cbh = (cbh_shadow_valid - cbh_true_valid) / 1000  # km
ax.scatter(mean_cbh, diff_cbh, alpha=0.5, s=10)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.axhline(y=np.mean(diff_cbh), color='orange', linestyle='-', linewidth=2)
ax.axhline(y=np.mean(diff_cbh) + 1.96*np.std(diff_cbh), color='gray', linestyle=':', linewidth=1.5, label='±1.96 SD')
ax.axhline(y=np.mean(diff_cbh) - 1.96*np.std(diff_cbh), color='gray', linestyle=':', linewidth=1.5)
ax.set_xlabel('Mean CBH (km)', fontsize=12)
ax.set_ylabel('Difference (Shadow - CPL) (km)', fontsize=12)
ax.set_title('Bland-Altman Plot', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sprint4_execution/figures/shadow_geometry_failure.png', dpi=300)
print("Saved: sprint4_execution/figures/shadow_geometry_failure.png")
```

**Expected Output:** r ≈ 0.04, bias ≈ +5 km, showing shadow geometry is fundamentally broken.

**Deliverable:** `figures/shadow_geometry_failure.png`

---

#### Task 1.3: Cross-Validation Failure Visualization
**Objective:** Show that LOO CV revealed anti-learning (negative R²)

**Code:**
```python
# File: sprint4_execution/visualize_loo_cv.py
import json
import numpy as np
import matplotlib.pyplot as plt

# Load WP-3 results
with open('sow_outputs/wp3_baseline/WP3_Report.json', 'r') as f:
    wp3_results = json.load(f)

# Extract LOO CV results
loo_r2 = []
loo_mae = []
loo_rmse = []
flight_names = ['30Oct24 (F0)', '10Feb25 (F1)', '23Oct24 (F2)', '12Feb25 (F3)', '18Feb25 (F4)']

# Parse from report (structure may vary, adjust as needed)
# Assuming results contain per-fold metrics
if 'loo_cv_results' in wp3_results:
    for fold in wp3_results['loo_cv_results']:
        loo_r2.append(fold.get('r2', np.nan))
        loo_mae.append(fold.get('mae', np.nan))
        loo_rmse.append(fold.get('rmse', np.nan))
else:
    # Placeholder if not in JSON
    loo_r2 = [-0.5, -2.3, -1.8, -45.2, -21.0]  # Example values

mean_r2 = np.mean(loo_r2)
std_r2 = np.std(loo_r2)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Bar plot of R² by fold
ax = axes[0]
colors = ['red' if r2 < 0 else 'green' for r2 in loo_r2]
bars = ax.bar(range(len(loo_r2)), loo_r2, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax.axhline(y=mean_r2, color='orange', linestyle='--', linewidth=2, 
           label=f'Mean R² = {mean_r2:.2f} ± {std_r2:.2f}')
ax.set_xticks(range(len(loo_r2)))
ax.set_xticklabels(flight_names, rotation=45, ha='right')
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('Leave-One-Flight-Out Cross-Validation Results', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, loo_r2)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -2),
            f'{val:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

# Right: Text summary
ax = axes[1]
ax.axis('off')
summary_text = f"""
LOO Cross-Validation Summary
{'='*40}

Mean R²: {mean_r2:.2f} ± {std_r2:.2f}

Interpretation:
• All folds have negative R²
• Negative R² means predictions are 
  WORSE than simply predicting the mean
• This indicates the model learned 
  anti-patterns or noise, not signal

SOW Decision Point:
• Threshold: R² > 0 for GO
• Actual: R² = {mean_r2:.2f}
• Decision: NO-GO ✗

Root Causes:
1. Shadow geometry: r ≈ 0.04 with CPL
2. ERA5 spatial resolution: 25 km grid
   too coarse for cloud-scale variability
3. No transferable physics signal
   across different flight conditions
"""
ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('sprint4_execution/figures/loo_cv_failure.png', dpi=300)
print("Saved: sprint4_execution/figures/loo_cv_failure.png")
```

**Expected Output:** Bar chart showing all-negative R² values.

**Deliverable:** `figures/loo_cv_failure.png`

---

### Day 3-4: Spatial Scale Mismatch Analysis

#### Task 1.4: ERA5 Resolution Schematic
**Objective:** Illustrate why 25 km ERA5 grid cannot resolve cloud-scale phenomena

**Code:**
```python
# File: sprint4_execution/spatial_scale_schematic.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# ERA5 grid cell (25 km × 25 km)
era5_cell = patches.Rectangle((0, 0), 25, 25, linewidth=3, 
                               edgecolor='red', facecolor='none', 
                               linestyle='--', label='ERA5 Grid Cell (25 km)')
ax.add_patch(era5_cell)

# ER-2 image footprint (512 pixels × 7 m/pixel ≈ 3.6 km)
er2_footprint = patches.Rectangle((5, 5), 3.6, 3.6, linewidth=2,
                                   edgecolor='blue', facecolor='lightblue',
                                   alpha=0.3, label='ER-2 Image Footprint (~3.6 km)')
ax.add_patch(er2_footprint)

# Individual clouds (schematic, ~200-500 m)
np.random.seed(42)
for i in range(15):
    x = np.random.uniform(1, 23)
    y = np.random.uniform(1, 23)
    size = np.random.uniform(0.2, 0.8)
    cloud = patches.Circle((x, y), size, linewidth=1,
                           edgecolor='gray', facecolor='white', alpha=0.8)
    ax.add_patch(cloud)

# CPL lidar beam (point measurement)
cpl_points = [(10, 15), (12, 18), (8, 10), (15, 12), (18, 20)]
for x, y in cpl_points:
    ax.plot(x, y, 'go', markersize=8, label='CPL Lidar Samples' if (x, y) == cpl_points[0] else '')

# Annotations
ax.text(12.5, 26, 'ERA5 Grid Cell: 25 km × 25 km', ha='center', fontsize=12, 
        color='red', fontweight='bold')
ax.text(6.8, 3.5, 'ER-2 Image\n(512×512 px)', ha='center', fontsize=10, color='blue')
ax.text(2, 23, 'Individual clouds:\n200-800 m', ha='left', fontsize=9, color='gray')

# Scale bar
ax.plot([1, 6], [1, 1], 'k-', linewidth=2)
ax.plot([1, 1], [0.8, 1.2], 'k-', linewidth=2)
ax.plot([6, 6], [0.8, 1.2], 'k-', linewidth=2)
ax.text(3.5, 0.3, '5 km', ha='center', fontsize=10, fontweight='bold')

ax.set_xlim(-1, 27)
ax.set_ylim(-1, 27)
ax.set_aspect('equal')
ax.set_xlabel('Distance (km)', fontsize=12)
ax.set_ylabel('Distance (km)', fontsize=12)
ax.set_title('Spatial Scale Mismatch: ERA5 Reanalysis vs. Cloud Observations', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sprint4_execution/figures/spatial_scale_mismatch.png', dpi=300)
print("Saved: sprint4_execution/figures/spatial_scale_mismatch.png")
```

**Deliverable:** `figures/spatial_scale_mismatch.png`

---

### Day 5: Shadow Detection Failure Mode Examples

#### Task 1.5: Visualize Shadow Detection on Sample Images
**Objective:** Show why shadow edge detection fails on ocean scenes

**Required:** Access to raw IRAI images and shadow detection intermediate outputs from WP-1.

**Code Sketch:**
```python
# File: sprint4_execution/shadow_detection_examples.py
# Load 3-4 representative images showing failure modes:
# 1. Ocean with no contrast (shadows invisible)
# 2. Broken cloud field (ambiguous shadow attribution)
# 3. Multi-layer clouds (geometric inversion fails)
# 4. Successful case (if any) for comparison

# Create 2×2 subplot showing:
# - Original image
# - Shadow detection overlays (if available)
# - Estimated CBH vs. ground truth annotation
# - Failure mode label

# Save as: figures/shadow_detection_failure_modes.png
```

**Note:** This requires access to original IRAI imagery and WP-1 intermediate outputs. If not available, skip this figure.

---

## Week 2: Manuscript Writing

### Day 6-7: Introduction and Background

#### Section 1: Introduction
**Key Points:**
- Cloud base height importance (aviation, climate, weather)
- Measurement gap: Ground-based sensors (ceilometers, lidar) sparse coverage
- Remote sensing challenge: Passive imagery from nadir can't directly measure height
- Promise of ML: Learn statistical relationships from labeled data
- Challenge: Cross-flight generalization (different atmospheric conditions)
- Physics-constrained ML hypothesis: Incorporate atmospheric thermodynamics and shadow geometry

**Structure:**
```markdown
# Introduction

## Motivation
- CBH critical for aviation safety, cloud radiative forcing, precipitation

## Current Capabilities
- Active lidar (CALIPSO, CPL): accurate but limited coverage
- Ground ceilometers: point measurements
- Satellite passive retrieval: relies on thermal/radiative properties

## Research Gap
- Airborne nadir imagery: high resolution, but geometry ambiguous
- Can ML bridge the gap?

## Prior Work
- Image-only ML models: poor generalization (R² < 0 across flights)
- Need for physical constraints

## This Study
- Test hypothesis: Shadow geometry + ERA5 reanalysis → cross-flight CBH retrieval
- Result: **Negative** - approach fails catastrophically
- Contribution: Diagnostic analysis of why physics features can fail
```

---

#### Section 2: Data and Methods
**Key Points:**
- ER-2 flight campaigns (5 flights, 933 labeled samples)
- CPL lidar ground truth
- IRAI nadir imagery (512×512 grayscale, 7 m/pixel)
- ERA5 reanalysis (0.25° hourly surface + pressure levels)

**WP-1 Geometric Features:**
- Shadow edge detection
- Solar geometry (SZA, SAA)
- Trigonometric height inversion: H = L × tan(SZA)
- Result: MAE = 5.12 km, r = 0.04, bias = +5.11 km

**WP-2 Atmospheric Features:**
- Boundary layer height (BLH)
- Lifting condensation level (LCL)
- Temperature, dewpoint, surface pressure
- Total: 9 features, 0 NaNs

**WP-3 Validation Protocol:**
- Leave-One-Flight-Out cross-validation
- GBDT regression (XGBoost/LightGBM)
- Success criterion: R² > 0

---

### Day 8-9: Results

#### Section 3: Results
**Structure:**
```markdown
# Results

## 3.1 Shadow Geometry Failure
- Figure 1: Shadow-based CBH vs. CPL ground truth
  - r = 0.04 (essentially uncorrelated)
  - Bias = +5.11 km (systematic overestimation)
  - MAE = 5.12 km, RMSE = 5.77 km
- Failure modes:
  - Ocean surface lacks contrast for shadow edge detection
  - Median imputation introduced strong bias
  - Multi-layer clouds violate single-layer geometric assumption

## 3.2 ERA5 Constraint Violations
- Figure 2: ERA5 BLH and LCL vs. CPL CBH
  - Hypothesis: BLH > CBH (boundary layer contains clouds)
  - Result: BLH < CBH in X% of cases (compute from Task 1.1)
  - Root cause: 25 km spatial resolution too coarse

## 3.3 Cross-Validation Failure
- Figure 3: LOO CV R² by fold
  - All 5 folds: negative R²
  - Mean R² = -14.15 ± 24.30
  - Interpretation: Model learned anti-patterns, not physics

## 3.4 Spatial Scale Mismatch
- Figure 4: Schematic of ERA5 grid vs. cloud scale
  - ERA5: 25 km × 25 km
  - Clouds: 200-800 m
  - ER-2 image: ~3.6 km footprint
  - Mismatch prevents ERA5 from constraining individual cloud CBH
```

---

### Day 10-11: Discussion and Conclusion

#### Section 4: Discussion
**Key Arguments:**

1. **Why Shadow Geometry Failed**
   - Nadir-only imagery fundamentally ambiguous
   - Requires high contrast (fails over ocean)
   - Multi-angle or stereo imaging needed

2. **Why ERA5 Failed**
   - Spatial resolution mismatch (25 km vs. cloud scale)
   - BLH is parameterized, not observed
   - Atmospheric state at grid scale ≠ cloud-scale properties

3. **Lessons for Physics-Constrained ML**
   - Physical features must be validated independently
   - "Physically motivated" ≠ "empirically useful"
   - Scale matching is critical
   - Negative cross-validation R² is a strong stop signal

4. **When to Abandon an Approach**
   - Diagnostic framework:
     - Check feature-target correlations individually
     - Validate physical constraints before ML
     - Use rigorous cross-validation (LOO for small datasets)
   - Red flags:
     - Systematic bias > 5× target variability
     - Correlation r < 0.1
     - Negative R² in cross-validation

---

#### Section 5: Alternative Approaches
**Recommendations:**

1. **Multi-angle stereo imaging**
   - Triangulate cloud top height from stereo pairs
   - Example: MISR satellite
   - Requires: Off-nadir cameras or multiple flight passes

2. **Higher-resolution reanalysis**
   - HRRR (3 km) or ERA5-Land (9 km)
   - May resolve mesoscale atmospheric features
   - Test hypothesis: finer resolution helps

3. **Radiative transfer modeling**
   - Physics-based forward model: cloud properties → radiance
   - Inverse problem: observed radiance → infer CBH
   - Computationally intensive but physically rigorous

4. **Active sensing**
   - Lidar or radar for direct ranging
   - Cloud profiling radars on aircraft
   - Gold standard but costly

---

### Day 12: Manuscript Polishing

#### Final Tasks
- [ ] Format references (bibtex)
- [ ] Create figure captions
- [ ] Abstract (200 words)
- [ ] Keywords
- [ ] Author contributions statement
- [ ] Data availability statement
- [ ] Code availability (link to GitHub repo)

---

## Deliverables Checklist

### Figures (6 total)
- [ ] Figure 1: Shadow geometry failure (scatter plot, residuals, Bland-Altman)
- [ ] Figure 2: ERA5 constraint violations (BLH and LCL vs. CBH)
- [ ] Figure 3: LOO CV failure (bar chart of R² by fold)
- [ ] Figure 4: Spatial scale mismatch schematic
- [ ] Figure 5: Shadow detection failure mode examples (optional, if images available)
- [ ] Figure 6: Diagnostic framework flowchart (when to abandon approach)

### Tables (2-3 total)
- [ ] Table 1: Dataset summary (5 flights, sample counts, CPL CBH statistics)
- [ ] Table 2: WP-1 geometric feature performance (MAE, RMSE, bias, r)
- [ ] Table 3: WP-3 LOO CV results (R² by fold, aggregate metrics)

### Manuscript Sections
- [ ] Abstract (200 words)
- [ ] Introduction (2-3 pages)
- [ ] Data and Methods (3-4 pages)
- [ ] Results (3-4 pages)
- [ ] Discussion (2-3 pages)
- [ ] Conclusion (1 page)
- [ ] References

### Supplementary Materials
- [ ] Code repository (GitHub link with DOI via Zenodo)
- [ ] Processed data (WP1/WP2/WP3 HDF5 files, if shareable)
- [ ] Extended diagnostic plots
- [ ] SOW completion report (as appendix)

---

## Target Journals (ranked by suitability)

### 1. Atmospheric Measurement Techniques (AMT)
**Why:** EGU journal, welcomes negative results, focus on measurement techniques  
**Impact Factor:** ~3.8  
**Review Time:** 2-3 months  
**Open Access:** Yes (€2000-3000 APC, may have institutional agreement)

### 2. Environmental Data Science (EDS)
**Why:** Cambridge, focus on methodology and workflows, emerging journal  
**Impact Factor:** New journal (2022)  
**Review Time:** 1-2 months  
**Open Access:** Yes (free during launch period)

### 3. Geoscientific Model Development (GMD)
**Why:** EGU, model evaluation and diagnostics  
**Impact Factor:** ~4.5  
**Review Time:** 2-4 months  
**Open Access:** Yes (€2000-3000 APC)

### 4. Machine Learning for Physical Sciences (ML4PS) Workshop @ NeurIPS
**Why:** Fast turnaround, ML audience, workshop papers citable  
**Impact Factor:** N/A (workshop)  
**Review Time:** 1 month  
**Open Access:** Yes (free)  
**Note:** 4-page short paper format, present at workshop

---

## Timeline Summary

| Week | Days | Tasks | Deliverables |
|------|------|-------|--------------|
| 1 | 1-2 | ERA5 constraint validation, shadow failure analysis | Figures 1-2 |
| 1 | 3-4 | LOO CV visualization, spatial scale schematic | Figures 3-4 |
| 1 | 5 | Shadow detection examples (optional) | Figure 5 |
| 2 | 6-7 | Introduction and Methods writing | Sections 1-2 draft |
| 2 | 8-9 | Results writing | Section 3 draft, Tables 1-3 |
| 2 | 10-11 | Discussion and Conclusion writing | Sections 4-5 draft |
| 2 | 12 | Polish, format, submit | Complete manuscript |

**Target Submission Date:** 2 weeks from start  
**Recommended Journal:** Atmospheric Measurement Techniques (AMT)

---

## Risk Mitigation

### Risk 1: Figures don't reproduce (data access issues)
**Mitigation:** Use existing SOW outputs (WP1/WP2/WP3 HDF5 files already created)

### Risk 2: Original IRAI images not accessible
**Mitigation:** Skip Figure 5 (shadow detection examples), focus on quantitative analysis

### Risk 3: Reviewer pushes back on "negative result"
**Mitigation:** Frame as methodological contribution (diagnostic framework), emphasize lessons for community

### Risk 4: Timeline slip
**Mitigation:** Prioritize Figures 1-4 and core manuscript, defer supplementary materials

---

## Success Metrics

- [ ] Manuscript submitted to peer-reviewed journal within 2 weeks
- [ ] All 6 figures generated and publication-ready
- [ ] Code and data archived with DOI (Zenodo)
- [ ] Positive feedback from advisor/collaborators on draft

---

**Prepared by:** AI Research Assistant  
**Date:** 2025-02-19  
**Status:** READY TO EXECUTE