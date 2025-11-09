# Cloud Base Height Retrieval Project: Sprint 4 Research Plan
## Physics-Constrained Machine Learning with ERA5 Integration

**Project:** Cloud Base Height Retrieval from ER-2 Imagery  
**Research Lead:** Rylan Malarchick  
**Date:** November 5, 2025  
**Version:** 4.0 (Post-Sprint 3 Analysis)

---

## Executive Summary

Sprint 3 attempted to implement ERA5 atmospheric reanalysis integration and shadow geometry detection for physics-constrained cloud base height (CBH) retrieval. The workflow labeled **wl-3** failed during ERA5 data integration and preprocessing, revealing fundamental challenges in:

1. **Data Pipeline Complexity:** ERA5 spatiotemporal collocation with ER-2 flight data at 0.25° resolution vs. 200m imagery resolution
2. **Physical Feature Engineering:** Shadow geometry extraction over ocean surfaces proved infeasible with existing camera setup
3. **Computational Bottlenecks:** ERA5 download, storage, and preprocessing workflows exceeded available infrastructure

Sprint 4 pivots to a **hybrid incremental approach** that decouples ERA5 integration from shadow geometry, implements robust data pipelines, and leverages physics-informed loss functions instead of explicit geometric features.

---

## Sprint 3 Failure Analysis

### What Went Wrong (wl-3 Failures)

Based on the GitHub repository analysis and project reports:

**Primary Failure Points:**
1. **ERA5 API Integration Issues**
   - CDS API authentication and quota limits
   - Incomplete atmospheric profile downloads for all flight times/locations
   - Temporal mismatch: ERA5 hourly data vs. CPL 1Hz measurements
   - Missing pressure level interpolation for aircraft altitude

2. **Spatial Collocation Problems**
   - 0.25° ERA5 grid (~25km) vs. 200m ER-2 pixel resolution mismatch
   - Nearest-neighbor interpolation introduced artifacts
   - No consideration of atmospheric advection during flight

3. **Computational Resource Constraints**
   - ERA5 data volume exceeded local storage (5 flights × hourly × 37 pressure levels)
   - Processing pipeline not optimized for cloud computing
   - No incremental processing or caching strategy

4. **Shadow Detection Infeasibility**
   - Ocean surface lacks contrast for shadow edge detection
   - Broken cloud fields create ambiguous shadow attribution
   - Multi-layer clouds complicate geometric inversion
   - Existing camera setup (grayscale, 512×512) insufficient for texture analysis

### Key Lessons Learned

1. **Physical features must be feasible before implementation** - Shadow geometry sounded promising in theory but was impractical given data constraints
2. **Data pipeline robustness is critical** - ERA5 integration needs error handling, incremental downloads, and validation checks
3. **Scalability matters** - Solutions must work within computational budget (no 5 PB ERA5 downloads)
4. **Physics constraints can be encoded differently** - Instead of explicit geometric features, use physics-informed loss functions and auxiliary predictions

---

## Sprint 4: Revised Research Strategy

### Core Insight

Rather than engineering explicit physical features (shadow length, BLH, LCL) that require complex data pipelines, **embed physical constraints directly into the learning objective** through:

1. **Physics-Informed Loss Functions:** Penalize predictions that violate atmospheric thermodynamics
2. **Auxiliary Task Learning:** Train model to predict related variables (cloud top height, optical depth) that share physical relationships with CBH
3. **Simplified ERA5 Integration:** Use only surface-level and boundary layer variables, not full atmospheric profiles
4. **Incremental Validation:** Build and validate each component independently before integration

### Proposed Hybrid Framework

```
Input: ER-2 Image (512×512) + Solar Geometry (SZA, SAA) + ERA5 Surface State

                    ↓

        ┌───────────────────────────┐
        │   Vision Encoder (MAE)    │  ← Pre-trained, frozen
        │   • Extract spatial       │
        │     features from image   │
        └───────────────────────────┘
                    ↓
        ┌───────────────────────────┐
        │  Atmospheric Context      │
        │  • ERA5 surface variables │
        │  • Solar geometry         │
        │  • Location, time         │
        └───────────────────────────┘
                    ↓
        ┌───────────────────────────┐
        │   Hybrid Fusion Network   │
        │   • Concatenate features  │
        │   • GBDT or shallow MLP   │
        └───────────────────────────┘
                    ↓
        ┌───────────────────────────┐
        │  Multi-Task Predictions   │
        │  • Primary: CBH           │
        │  • Auxiliary: CTH, COD    │
        │  • Uncertainty estimate   │
        └───────────────────────────┘
                    ↓
        ┌───────────────────────────┐
        │ Physics-Informed Loss     │
        │ • CBH prediction error    │
        │ • CBH < CTH constraint    │
        │ • Geometric thickness     │
        │ • Thermodynamic bounds    │
        └───────────────────────────┘
```

---

## Research Plan: Phased Implementation

### Phase 1: Minimal ERA5 Integration (Weeks 1-2)

**Objective:** Establish robust ERA5 data pipeline with minimal subset of variables

**Tasks:**
1. **ERA5 Variable Selection**
   - Surface skin temperature (skt)
   - 2m temperature (t2m) and 2m dewpoint (d2m)
   - Boundary layer height (blh) - direct ERA5 diagnostic
   - Surface pressure (sp)
   - Total cloud cover (tcc) - for validation

2. **Data Pipeline Development**
   - Implement CDS API wrapper with retry logic and quota management
   - Download ERA5 for 5 flight dates (Oct 30, Feb 10, Oct 23, Feb 12, Feb 18) at hourly resolution
   - Store as lightweight NetCDF or Zarr format (~2GB total)
   - Create collocation module: nearest-neighbor for each CPL sample

3. **Derived Feature Engineering**
   - Compute Lifting Condensation Level (LCL):  
     `LCL = (T_2m - T_dewpoint) / 8 K/km`
   - Compute lapse rate: `Γ = (T_surface - T_2m) / altitude_diff`
   - Flag atmospheric stability: stable (Γ < 6.5 K/km) vs. unstable (Γ > 9.8 K/km)

4. **Validation Checkpoints**
   - Verify ERA5 temporal coverage: no missing hours
   - Check spatial coverage: all flight lat/lon within ERA5 grid
   - Sanity check: BLH from ERA5 vs. CBH from CPL (BLH should be upper bound)

**Success Criteria:**
- ERA5 data successfully downloaded and collocated for all 933 CPL samples
- LCL computation completes without errors
- BLH > CBH for >90% of samples (physical consistency check)

**Deliverables:**
- `era5_downloader.py` - Robust ERA5 API client
- `era5_collocation.py` - Spatiotemporal matching module
- `era5_features_933samples.csv` - Processed ERA5 features for training set

---

### Phase 2: Physics-Informed Baseline (Weeks 2-3)

**Objective:** Train physically-constrained model without image features

**Tasks:**
1. **Feature Set Construction**
   - Input features: [SZA, SAA, BLH_ERA5, LCL, T_2m, T_dewpoint, sp, tcc, lat, lon, time_of_day]
   - Target: CBH from CPL

2. **Model Development**
   - Train Gradient Boosted Regression Trees (GBDT) with XGBoost or LightGBM
   - Hyperparameter tuning: tree depth, learning rate, n_estimators
   - Cross-validation: Leave-One-Flight-Out (LOFO)

3. **Physics-Informed Loss Function**
   - Standard regression loss: `L_regression = MSE(CBH_pred, CBH_true)`
   - Physical constraint penalties:
     - `L_upper_bound = max(0, CBH_pred - BLH_ERA5)^2` (CBH must be below BLH)
     - `L_lower_bound = max(0, 0 - CBH_pred)^2` (CBH must be non-negative)
     - `L_lcl_deviation = |CBH_pred - LCL|^2` for convective clouds (weight by instability flag)
   - Total loss: `L_total = L_regression + λ1*L_upper_bound + λ2*L_lower_bound + λ3*L_lcl_deviation`
   - Tune λ weights on validation set

4. **Interpretability Analysis**
   - SHAP (SHapley Additive exPlanations) values for feature importance
   - Partial dependence plots: How does CBH vary with BLH, LCL, solar geometry?
   - Physical plausibility: Are learned relationships consistent with atmospheric physics?

**Success Criteria:**
- LOFO cross-validation: R² > 0.0 (better than image-only models)
- Physical constraint violations < 5% of predictions
- Feature importance: BLH and LCL among top 3 features

**Deliverables:**
- `physics_baseline.py` - GBDT model with physics-informed loss
- `shap_analysis.ipynb` - Feature importance and interpretability
- `baseline_results.csv` - LOFO CV metrics by flight

---

### Phase 3: Hybrid Image + ERA5 Integration (Weeks 3-4)

**Objective:** Combine learned image features with physical constraints

**Tasks:**
1. **Feature Fusion Architecture**
   - Image features: Use pre-trained MAE encoder (frozen weights) → extract CLS token embedding (192-dim)
   - ERA5 features: [BLH, LCL, T_2m, T_dewpoint, lapse_rate] (5-dim)
   - Solar geometry: [SZA, SAA, cos(SZA), hour_of_day] (4-dim)
   - Concatenate: 201-dim input vector

2. **Model Variants**
   - **Variant A (GBDT):** Feed 201-dim vector to XGBoost
   - **Variant B (Shallow MLP):** 2-layer neural network (201 → 128 → 64 → 1)
   - **Variant C (Multi-Task MLP):** Predict CBH + auxiliary tasks

3. **Multi-Task Learning Setup**
   - Primary task: CBH regression
   - Auxiliary tasks (if data available):
     - Cloud top height (CTH) - estimate as CBH + 1km for single-layer clouds
     - Cloud optical depth (COD) - proxy from CPL backscatter
   - Shared encoder → task-specific heads
   - Multi-task loss: `L = L_CBH + α*L_CTH + β*L_COD + physics_penalties`

4. **Uncertainty Quantification**
   - Ensemble method: Train 10 models with different random seeds
   - Prediction = mean of ensemble
   - Uncertainty = standard deviation of ensemble
   - Flag high-uncertainty predictions (σ > 500m) for manual review

**Success Criteria:**
- Hybrid model (image + ERA5) outperforms ERA5-only baseline by >10% MAE reduction
- R² > 0.3 on LOFO CV (stretch goal: R² > 0.5)
- Physical constraint violations < 2%

**Deliverables:**
- `hybrid_model.py` - Fusion architecture and training loop
- `multitask_learning.py` - MTL variant with auxiliary tasks
- `ensemble_uncertainty.py` - Uncertainty quantification module
- `sprint4_results.pdf` - Performance comparison table

---

### Phase 4: Evaluation and Paper Preparation (Week 5)

**Objective:** Comprehensive evaluation and manuscript drafting

**Tasks:**
1. **Ablation Studies**
   - Quantify contribution of each feature group:
     - Angles only
     - ERA5 only
     - Images only
     - Images + Angles
     - Images + ERA5
     - Full model (Images + Angles + ERA5)
   - Statistical significance testing (paired t-test on LOFO CV folds)

2. **Failure Mode Analysis**
   - Identify systematic errors:
     - Overestimation/underestimation by cloud regime (stratocumulus vs. cumulus)
     - Performance degradation by flight
     - Sensitivity to ERA5 spatial resolution
   - Error correlation with physical variables (time of day, atmospheric stability, CBH range)

3. **Comparison with State-of-the-Art**
   - Benchmark against:
     - LCL-based climatology
     - Satellite CBH retrievals (if available for validation region)
     - Pure ML baselines (angles-only, MAE-only)

4. **Manuscript Outline**
   - **Title:** "Physics-Informed Machine Learning for Cross-Flight Cloud Base Height Retrieval: Integrating ERA5 Reanalysis with Airborne Imagery"
   - **Story Arc:**
     1. CBH importance and measurement gap
     2. Naive ML approaches fail (negative results from Sprint 1-2)
     3. Physical constraints are necessary for generalization
     4. Hybrid ERA5 + image approach achieves robust cross-flight performance
     5. Lessons for ML in geophysical retrieval problems
   - **Target Venues:** AIES, Environmental Data Science, Remote Sensing of Environment

**Success Criteria:**
- Complete ablation study table with ≥6 model variants
- Manuscript draft (Introduction, Methods, Results) ready for advisor review
- Negative results from Sprint 1-3 framed as valuable methodological contribution

**Deliverables:**
- `ablation_study_results.csv` - Systematic feature group comparison
- `error_analysis.ipynb` - Failure mode visualization and analysis
- `manuscript_draft_v1.pdf` - Manuscript draft with figures
- `codebase_release/` - Cleaned, documented code for reproducibility

---

## Risk Management

### High-Priority Risks

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|-----------|--------|---------------------|
| **ERA5 API quota exceeded** | High | High | Implement retry logic, download incrementally, cache locally, use Google ARCO ERA5 if needed |
| **ERA5 resolution too coarse** | Medium | High | Test sensitivity to spatial interpolation, consider using ERA5-Land (9km resolution) for surface variables |
| **Physical constraints too weak** | Medium | Medium | Iterate on loss function weights (λ), add more constraints (e.g., CTH > CBH), use hard constraints in post-processing |
| **Hybrid model still fails LOFO CV** | Medium | High | Frame as negative result, emphasize methodological contribution, explore domain adaptation techniques |
| **Timeline slip due to data issues** | Medium | Medium | Allocate buffer time in Phase 1, parallelize tasks, defer non-critical experiments to post-sprint |

### Low-Priority Risks

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|-----------|--------|---------------------|
| **MAE features unhelpful again** | Low | Low | Test with alternative pre-trained encoders (ResNet, ViT) or fine-tune MAE on unlabeled cloud images |
| **Multi-task learning degrades CBH** | Low | Low | Treat as optional extension, focus on single-task CBH if MTL doesn't improve |
| **Computational budget exceeded** | Low | Medium | Use lightweight models (GBDT preferred over deep networks), leverage ERAU HPC if available |

---

## Open Research Questions

### Scientific Questions
1. **Can ERA5 surface variables (BLH, LCL) provide sufficient physical constraints for CBH retrieval, or is full atmospheric profiling necessary?**
   - Hypothesis: Surface variables sufficient for boundary-layer clouds (CBH < 3km)
   - Test: Compare ERA5 surface-only vs. ERA5 with 37 pressure levels (if feasible)

2. **How does atmospheric advection affect ERA5-imagery collocation accuracy?**
   - Hypothesis: Clouds advect 10-50 km during ER-2 flight, ERA5 hourly snapshots miss transient events
   - Test: Compute advection correction using ERA5 wind fields, assess impact on CBH error

3. **Do physics-informed loss functions improve generalization, or just reduce constraint violations?**
   - Hypothesis: Physics penalties improve cross-flight transfer by regularizing against spurious correlations
   - Test: Compare standard MSE loss vs. physics-informed loss on held-out flights

### Methodological Questions
1. **Is GBDT or neural network better for fusion of heterogeneous features (image embeddings, ERA5 scalars)?**
   - Hypothesis: GBDT handles mixed feature types better, neural networks better for learning interactions
   - Test: Ablation study comparing GBDT, shallow MLP, attention-based fusion

2. **Can self-supervised learning on unlabeled images be improved by physics-based data augmentation?**
   - Hypothesis: Augment images with simulated atmospheric conditions (time-of-day, cloud regime) to learn invariant features
   - Test: Retrain MAE with physics-aware augmentations, evaluate downstream CBH performance

3. **Should we enforce hard constraints (CBH < BLH) or soft penalties in loss function?**
   - Hypothesis: Soft penalties during training + hard constraints at inference balances flexibility and physical consistency
   - Test: Compare 3 strategies: (1) soft only, (2) hard only, (3) soft training + hard clipping

---

## Timeline and Milestones

### Week 1: ERA5 Integration
- **Mon-Tue:** Set up CDS API, download test subset (1 flight)
- **Wed-Thu:** Implement collocation and LCL computation
- **Fri:** Validate data quality, checkpoint review

**Milestone:** ERA5 data pipeline operational, 933 samples processed

---

### Week 2: Physics Baseline
- **Mon-Tue:** Train GBDT with ERA5 features, implement physics-informed loss
- **Wed-Thu:** SHAP analysis and feature importance
- **Fri:** LOFO CV evaluation, checkpoint review

**Milestone:** Physics-only baseline achieves R² > 0 on LOFO CV

---

### Week 3: Hybrid Model Development
- **Mon-Tue:** Implement feature fusion, train Variant A (GBDT)
- **Wed-Thu:** Train Variant B (shallow MLP) and Variant C (multi-task)
- **Fri:** Uncertainty quantification, checkpoint review

**Milestone:** Hybrid model outperforms baseline by >10% MAE

---

### Week 4: Evaluation and Analysis
- **Mon-Tue:** Ablation studies (6 model variants)
- **Wed-Thu:** Failure mode analysis, error correlation
- **Fri:** Prepare results summary, checkpoint review

**Milestone:** Complete ablation study and error analysis

---

### Week 5: Paper Preparation
- **Mon-Tue:** Draft Introduction and Methods sections
- **Wed-Thu:** Create figures (scatter plots, feature importance, ablation table)
- **Fri:** Manuscript draft v1, code cleanup and documentation

**Milestone:** Manuscript draft ready for advisor review

---

## Success Metrics

### Quantitative Metrics
- **Primary:** R² > 0.3 on LOFO cross-validation (vs. R² < 0 for image-only models)
- **Physical Consistency:** <2% of predictions violate CBH < BLH constraint
- **Generalization:** MAE variance across LOFO folds < 0.15 km (consistent performance across flights)
- **Uncertainty Calibration:** High-uncertainty predictions (σ > 500m) have 2× higher MAE than low-uncertainty predictions

### Qualitative Metrics
- **Interpretability:** SHAP values align with domain knowledge (e.g., BLH and LCL are top features)
- **Reproducibility:** Code runs on fresh Python environment with `requirements.txt`
- **Publication Readiness:** Manuscript passes internal review (advisor feedback), targets AIES or Environmental Data Science

---

## Resources and Infrastructure

### Computational Resources
- **Local Machine:** Initial development and testing
- **ERAU HPC (if available):** Hyperparameter tuning, ensemble training
- **Google Colab Pro (backup):** GPU for MAE feature extraction if needed

### Data Storage
- **ERA5 Data:** ~2GB (surface variables, hourly, 5 flights)
- **Processed Features:** ~50MB (933 samples, 201-dim vectors)
- **Model Checkpoints:** ~500MB (ensemble of 10 models)

### Software Stack
- **Python 3.11** (per user preference)
- **Core Libraries:** `xarray`, `netCDF4`, `cdsapi` (ERA5), `xgboost`, `lightgbm` (GBDT), `torch` (MAE encoder), `shap` (interpretability)
- **Environment Management:** Conda (per user workflow)

---

## Expected Outcomes

### Optimistic Scenario (90% confidence)
- ERA5 integration successful, robust data pipeline
- Physics-informed baseline achieves R² > 0.2 on LOFO CV
- Hybrid model achieves R² > 0.4, <5% constraint violations
- Manuscript draft complete, targeting AIES journal

### Realistic Scenario (70% confidence)
- ERA5 integration successful with minor issues (1-2 missing hours)
- Physics-informed baseline achieves R² > 0.1 on LOFO CV
- Hybrid model achieves R² > 0.3, <10% constraint violations
- Manuscript draft complete, negative results framed as methodological contribution

### Pessimistic Scenario (10% confidence)
- ERA5 integration faces major issues (API failures, missing data)
- Physics-informed baseline fails to beat R² = 0
- Hybrid model marginally better than baseline (R² ~0.1)
- Pivot to negative result paper: "Why Physical Constraints Alone Are Insufficient for Cross-Flight CBH Retrieval"

**In all scenarios:** Valuable lessons learned, publishable contribution, foundation for future work

---

## Next Steps (Post-Sprint 4)

If Sprint 4 succeeds:
1. **Expand to HRRR reanalysis** (3km resolution) for higher-fidelity atmospheric state
2. **Transfer to new datasets** (GOES-16 imagery, MODIS cloud products)
3. **Real-time deployment** (operational CBH product for NASA ER-2 flights)
4. **Domain adaptation** (few-shot learning for new cloud regimes)

If Sprint 4 struggles:
1. **Deep dive into ERA5 resolution limitations** (sensitivity analysis, super-resolution techniques)
2. **Alternative physical constraints** (radiative transfer modeling, cloud microphysics)
3. **Collaborate with atmospheric scientists** (validate assumptions, identify missing physics)
4. **Reframe as methods paper** (diagnostic framework for ML in geophysical retrieval)

---

## References and Resources

### Key Papers
1. **ERA5 Reanalysis:** Hersbach et al. (2020) - ERA5 global reanalysis, QJRMS
2. **Physics-Informed Neural Networks:** Raissi et al. (2019) - PINNs for PDEs, JCP
3. **Cloud Base Height from Satellites:** Lin et al. (2022) - GBRT for CBH from ABI+ERA5, RSE
4. **LCL Computation:** Romps (2017) - Exact expression for lifting condensation level, JAS

### External Tools and Datasets
- **CDS API:** https://cds.climate.copernicus.eu/api-how-to
- **Google ARCO ERA5:** https://github.com/google-research/arco-era5 (cloud-optimized ERA5)
- **XGBoost Documentation:** https://xgboost.readthedocs.io/
- **SHAP Library:** https://github.com/slundberg/shap

---

## Approval and Sign-Off

**Prepared by:** AI Research Assistant  
**Date:** November 5, 2025  
**Status:** DRAFT - Pending approval by Rylan Malarchick

**Action Required:**
- [ ] Review research plan and timeline
- [ ] Approve/modify Phase 1-4 tasks
- [ ] Confirm computational resources (ERAU HPC access?)
- [ ] Approve risk mitigation strategies
- [ ] Sign off to proceed with Sprint 4

**Notes for Modification:**
Please provide feedback on:
1. ERA5 variable selection - any additions/removals?
2. Timeline feasibility given semester end (Dec 10)
3. Priority: Should we focus on getting baseline working (Phases 1-2) or push for hybrid model?
4. Publication strategy: AIES vs. NeurIPS workshop vs. GRL?

---

**End of Sprint 4 Research Plan**
