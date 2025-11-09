# Sprint 4 Execution: Negative Results Analysis and Paper Preparation

**Status:** ACTIVE - Week 1 in progress  
**Date Started:** 2025-02-19  
**Expected Completion:** 2025-03-05 (2 weeks)  
**Primary Deliverable:** Manuscript submitted to peer-reviewed journal

---

## Executive Summary

Sprint 4 has been **fundamentally redesigned** based on the catastrophic failure of WP-3 (physics baseline). The original Sprint 4 plan assumed that physics-constrained features (shadow geometry + ERA5 reanalysis) would achieve R² > 0 in cross-validation, enabling hybrid image+physics models in later phases.

**Reality:** WP-3 achieved mean LOO R² = **-14.15 ± 24.30**, rejecting the core hypothesis.

**Pivot:** Transform negative results into a publishable methodological contribution documenting why shadow geometry and ERA5 fail for cloud base height retrieval from nadir imagery.

---

## Documents in This Directory

### Strategic Planning
- **`gap_analysis.md`**: Comprehensive comparison of Sprint 4 plan vs. actual WP1-WP4 results
  - Phase-by-phase assessment of what failed and why
  - Root cause analysis of planning failures
  - Recommended paths forward (Options A-D)

- **`action_plan.md`**: Detailed 2-week execution plan for negative results paper
  - Week 1: Diagnostic analysis and figure generation (6 figures)
  - Week 2: Manuscript writing (Introduction → Conclusion)
  - Target journal: Atmospheric Measurement Techniques (AMT)
  - Timeline, tasks, deliverables, and success metrics

### Analysis Scripts
- **`validate_era5_constraints.py`**: Validate ERA5 physical constraints
  - Check if BLH > CBH holds in practice
  - Compute correlations between ERA5 features and CPL CBH
  - **Output:** `figures/era5_constraint_validation.png`

- **`shadow_failure_analysis.py`**: Analyze shadow geometry failure
  - Scatter plots, residual analysis, Bland-Altman plots
  - Quantify bias, MAE, RMSE, correlation
  - **Output:** `figures/shadow_geometry_failure.png`

- **`visualize_loo_cv.py`**: (TODO) Visualize LOO cross-validation failure
  - Bar chart of R² by fold (all negative)
  - Summary statistics and interpretation
  - **Output:** `figures/loo_cv_failure.png`

- **`spatial_scale_schematic.py`**: (TODO) Create spatial scale mismatch diagram
  - ERA5 25 km grid vs. cloud scale (200-800 m)
  - ER-2 image footprint (~3.6 km)
  - CPL lidar point measurements
  - **Output:** `figures/spatial_scale_mismatch.png`

### Outputs
- **`figures/`**: Publication-ready diagnostic figures
  - Target: 6 figures for manuscript
  - DPI: 300 (publication quality)
  - Format: PNG (convert to EPS/PDF for journal submission)

---

## Progress Tracking

### Week 1: Analysis and Figure Generation ✅ In Progress

#### Day 1-2: Diagnostic Analysis
- [x] Script created: `validate_era5_constraints.py`
- [x] Script created: `shadow_failure_analysis.py`
- [ ] Run ERA5 constraint validation
- [ ] Run shadow failure analysis
- [ ] Generate Figures 1-2

#### Day 3-4: Spatial Scale Analysis
- [ ] Create LOO CV visualization script
- [ ] Create spatial scale schematic script
- [ ] Generate Figures 3-4

#### Day 5: Optional Image Examples
- [ ] Assess availability of raw IRAI images
- [ ] If available: Create shadow detection failure mode examples (Figure 5)
- [ ] If not available: Skip and focus on quantitative analysis

### Week 2: Manuscript Writing ⏳ Not Started

#### Day 6-7: Introduction and Methods
- [ ] Draft Introduction section
- [ ] Draft Data and Methods section
- [ ] Create Table 1: Dataset summary
- [ ] Create Table 2: WP-1 performance metrics

#### Day 8-9: Results
- [ ] Draft Results section (4 subsections)
- [ ] Create Table 3: LOO CV results by fold
- [ ] Insert all figures with captions

#### Day 10-11: Discussion and Conclusion
- [ ] Draft Discussion section
- [ ] Draft Conclusion section
- [ ] Create Figure 6: Diagnostic framework flowchart

#### Day 12: Polish and Submit
- [ ] Abstract (200 words)
- [ ] Format references
- [ ] Final proofreading
- [ ] Submit to Atmospheric Measurement Techniques

---

## Key Findings (From WP1-WP4 Execution)

### Shadow Geometry (WP-1)
- **Performance:** MAE = 5.12 km, RMSE = 5.77 km, r = 0.04
- **Bias:** +5.11 km (predicted 5.94 km vs. actual 0.83 km mean CBH)
- **Validity:** 813/933 samples (87.1%) had valid estimates
- **Conclusion:** Shadow-based geometric inversion from nadir imagery over ocean is **INFEASIBLE**

### ERA5 Features (WP-2)
- **Success:** 933 samples × 9 atmospheric features, 0 NaNs
- **Features:** blh, lcl, inversion_height, moisture_gradient, stability_index, t2m, d2m, sp, tcwv
- **Issue:** 25 km spatial resolution too coarse for cloud-scale variability
- **Hypothesis to test:** Do BLH and LCL actually constrain CBH in practice?

### Physics Baseline (WP-3)
- **Performance:** Mean LOO R² = -14.15 ± 24.30
- **All folds:** Negative R² (worse than predicting the mean)
- **Features used:** 3 geometric + 9 atmospheric = 12 total
- **Conclusion:** Physics-constrained ML hypothesis **REJECTED**

---

## Research Questions for Paper

### Primary Questions
1. **Why does shadow geometry fail?**
   - Ocean surface lacks contrast
   - Broken cloud fields create ambiguous attribution
   - Multi-layer clouds violate geometric assumptions
   - Median imputation introduced systematic bias

2. **Why does ERA5 fail?**
   - Spatial scale mismatch (25 km vs. 200-800 m clouds)
   - BLH is parameterized, not observed
   - Does BLH > CBH constraint even hold?

3. **Why did cross-validation fail so badly?**
   - No transferable physics signal across flights
   - Model learned anti-patterns (negative R²)
   - Each flight has different atmospheric conditions

### Methodological Contributions
1. **When to abandon an approach?**
   - Diagnostic framework for early detection of failures
   - Red flags: r < 0.1, bias > 5× target variability, negative R² in CV

2. **Lessons for physics-constrained ML**
   - "Physically motivated" ≠ "empirically useful"
   - Validate each component independently before integration
   - Scale matching is critical (25 km reanalysis ≠ cloud scale)

---

## Alternative Approaches (Future Work)

### Option A: Multi-Angle Stereo (Long-term, 3+ months)
- Triangulate cloud top height from stereo pairs
- Requires: Off-nadir cameras or multiple flight passes
- Example: MISR satellite approach
- **Prerequisite:** Confirm ER-2 has multi-angle capability

### Option B: Higher-Resolution Reanalysis (Medium-term, 1 month)
- HRRR (3 km) or ERA5-Land (9 km)
- Test hypothesis: finer resolution helps
- **Next step:** Download HRRR for 5 flight dates, re-run WP-3

### Option C: Negative Results Paper (Immediate, 2 weeks)
- **CURRENT PLAN**
- Methodological contribution
- Diagnostic analysis of failures
- Recommendations for future work

### Option D: Pure ML, Ignore Physics (Medium-term, 1-2 months)
- Fine-tune MAE on cloud imagery
- Vision transformers with attention
- Ensemble of domain-specific models
- **Caveat:** Likely still fails due to nadir-only limitation

---

## Target Journals (Ranked)

### 1. Atmospheric Measurement Techniques (AMT) ⭐ RECOMMENDED
- **Publisher:** EGU (European Geosciences Union)
- **Focus:** Measurement techniques, instrument validation, negative results welcome
- **Impact Factor:** ~3.8
- **Review Time:** 2-3 months
- **Open Access:** Yes (€2000-3000 APC, check institutional agreement)
- **Why:** Perfect fit for negative results, methodological focus

### 2. Environmental Data Science (EDS)
- **Publisher:** Cambridge University Press
- **Focus:** Data science workflows, methodology
- **Impact Factor:** New journal (2022)
- **Review Time:** 1-2 months
- **Open Access:** Yes (free during launch period)

### 3. Machine Learning for Physical Sciences (ML4PS) @ NeurIPS
- **Type:** Workshop paper
- **Format:** 4-page short paper
- **Review Time:** 1 month
- **Open Access:** Yes (free)
- **Why:** Fast publication, ML audience

---

## Success Metrics

### Minimum Viable Product (MVP)
- [ ] 4-6 publication-quality figures generated
- [ ] Manuscript draft complete (all sections)
- [ ] Submitted to peer-reviewed journal
- [ ] Code and data archived with DOI (Zenodo)

### Stretch Goals
- [ ] Manuscript accepted after first round of reviews
- [ ] Code cited by other researchers
- [ ] Invited to present at AGU/AMS conference

---

## Running the Analysis Scripts

### Prerequisites
```bash
# Ensure WP1-WP4 outputs exist
ls sow_outputs/wp1_geometric/WP1_Features.hdf5
ls sow_outputs/wp2_atmospheric/WP2_Features.hdf5
ls sow_outputs/wp3_baseline/WP3_Report.json

# Install dependencies
pip install h5py numpy matplotlib scipy
```

### Execute Analysis
```bash
# From repository root
cd cloudMLPublic

# Run ERA5 constraint validation
python sprint4_execution/validate_era5_constraints.py

# Run shadow failure analysis
python sprint4_execution/shadow_failure_analysis.py

# (TODO) Run LOO CV visualization
# python sprint4_execution/visualize_loo_cv.py

# (TODO) Create spatial scale schematic
# python sprint4_execution/spatial_scale_schematic.py
```

### Outputs
All figures saved to: `sprint4_execution/figures/`
- `era5_constraint_validation.png` (Figure 2)
- `shadow_geometry_failure.png` (Figure 1)
- `loo_cv_failure.png` (Figure 3)
- `spatial_scale_mismatch.png` (Figure 4)

---

## Contact and Collaboration

**Research Lead:** Rylan Malarchick  
**Institution:** Embry-Riddle Aeronautical University  
**Project:** NASA Cloud Base Height Retrieval from ER-2 Imagery

**Questions or collaboration inquiries:** Contact via repository issues or institutional email.

---

## References

### Key Papers to Cite
1. **ERA5 Reanalysis:** Hersbach et al. (2020), QJRMS
2. **Cloud Base Height from Satellites:** Lin et al. (2022), Remote Sensing of Environment
3. **LCL Computation:** Romps (2017), Journal of the Atmospheric Sciences
4. **Physics-Informed Neural Networks:** Raissi et al. (2019), Journal of Computational Physics

### Related Work
- MISR multi-angle stereo cloud height retrieval
- CALIPSO/CPL lidar cloud profiling
- Machine learning for atmospheric retrievals (review papers)

---

**Last Updated:** 2025-02-19  
**Version:** 1.0  
**Status:** Week 1 in progress - diagnostic analysis phase