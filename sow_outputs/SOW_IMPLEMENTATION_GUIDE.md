# SOW Sprint 3 Implementation Guide

**Document ID:** SOW-IMPLEMENTATION-GUIDE  
**Version:** 1.0  
**Date:** 2025  
**Status:** IN PROGRESS

---

## Executive Summary

This document tracks the implementation of the Physics-Constrained CBH Model Validation project as specified in `ScopeWorkSprint3.md`. The work is organized into 4 Work Packages (WP-1 through WP-4), with clear deliverables and success criteria.

### Project Objective
Validate the hypothesis that physics-constrained features (shadow geometry + atmospheric thermodynamics) are essential for cross-flight generalization in Cloud Base Height (CBH) retrieval.

### Success Criteria
- **Minimum Success:** Physical Features Baseline (WP-3) achieves LOO CV R² > 0
- **Target Success:** Final Hybrid Model (WP-4) achieves LOO CV R² > 0.3

---

## Implementation Status

### ✅ Work Package 1: Geometric Feature Engineering
**Status:** IMPLEMENTATION COMPLETE - READY FOR TESTING

**Script:** `sow_outputs/wp1_geometric_features.py`

**Functionality Implemented:**
1. **Shadow Detection Algorithm**
   - Gradient-based edge detection using Sobel operators
   - Adaptive thresholding for shadow regions
   - Bright cloud region detection
   - Morphological cleanup to reduce noise

2. **Cloud-Shadow Pair Identification**
   - Projects along solar azimuth direction
   - Validates alignment with expected shadow direction
   - Computes confidence based on geometric and photometric consistency
   - Handles multiple cloud-shadow candidates, selects best match

3. **Geometric CBH Derivation**
   - Formula: `H = L × tan(90° - SZA) × scale_factor`
   - Converts shadow length from pixels to meters using ground sampling distance
   - Accounts for solar elevation angle

4. **Confidence Scoring**
   - Multi-factor assessment (5 factors):
     - Geometric alignment with solar azimuth (25%)
     - Shadow contrast relative to surroundings (25%)
     - Cloud/shadow size consistency (15%)
     - Solar angle quality (20%)
     - Shape coherence (solidity) (15%)
   - Flags low-confidence detections as NaN

5. **Robust Handling of Challenges**
   - Low-contrast surfaces: Checks shadow contrast before accepting
   - Multi-layer clouds: Uses shape coherence metrics
   - Broken cloud fields: Validates cloud/shadow pair alignment

**Deliverable:** `WP1_Features.hdf5`

**Features Extracted:**
- `derived_geometric_H` (km)
- `shadow_length_pixels`
- `shadow_detection_confidence` (0-1)
- `cloud_edge_x`, `cloud_edge_y`
- `shadow_edge_x`, `shadow_edge_y`
- `shadow_angle_deg`
- `sza_deg`, `saa_deg`
- `true_cbh_km` (ground truth)

**Usage:**
```bash
cd cloudMLPublic
python sow_outputs/wp1_geometric_features.py \
    --config configs/bestComboConfig.yaml \
    --output sow_outputs/wp1_geometric/WP1_Features.hdf5 \
    --scale 50.0 \
    --verbose
```

**Parameters:**
- `--scale`: Ground sampling distance in meters/pixel (default: 50.0)
  - Depends on aircraft altitude and camera FOV
  - For ER-2 at ~20 km altitude: approximately 20-100 m/pixel
  - May need calibration based on actual imaging geometry

**Next Steps:**
1. Run on full dataset to validate shadow detection performance
2. Analyze success rate and confidence distribution
3. Tune parameters if needed (thresholds, scale factor)
4. Generate diagnostic plots to visualize detections

---

### ✅ Work Package 2: Atmospheric Feature Engineering
**Status:** IMPLEMENTATION COMPLETE - SYNTHETIC MODE

**Script:** `sow_outputs/wp2_atmospheric_features.py`

**Functionality Implemented:**
1. **ERA5 Data Manager**
   - CDS API integration for ERA5 reanalysis download
   - Handles both single-level and pressure-level variables
   - Automatic caching to avoid redundant downloads
   - Configurable spatial resolution (default: 0.25° ≈ 25 km)

2. **Thermodynamic Variable Derivation**
   - **BLH (Boundary Layer Height):** Direct from ERA5
   - **LCL (Lifting Condensation Level):** Computed from surface T and Td
     - Formula: `LCL = 125 × (T - Td)` meters
   - **Inversion Height:** Finds strongest temperature gradient (dT/dz > 0)
   - **Moisture Gradient:** Vertical gradient of specific humidity (dq/dz)
   - **Stability Index:** Mean lapse rate in lower troposphere (0-3 km)

3. **Spatio-Temporal Interpolation**
   - 4D interpolation (lat, lon, time, pressure level)
   - Nearest-neighbor method (can be upgraded to linear)
   - Handles 25 km ERA5 → 200 m imagery resolution mismatch

4. **Profile Extraction**
   - Extracts full vertical profile for each sample location/time
   - Quality assessment and confidence scoring
   - Robust error handling for missing data

**Deliverable:** `WP2_Features.hdf5`

**Features Extracted:**
- `blh_m` (Boundary Layer Height, meters)
- `lcl_m` (Lifting Condensation Level, meters)
- `inversion_height_m` (Temperature inversion height, meters)
- `moisture_gradient` (kg/kg/m)
- `stability_index` (K/km - atmospheric lapse rate)
- `surface_temp_k` (Surface temperature, Kelvin)
- `surface_dewpoint_k` (Surface dewpoint, Kelvin)
- `surface_pressure_pa` (Surface pressure, Pascals)
- `lapse_rate_k_per_km` (Temperature lapse rate)
- `profile_confidence` (0-1)
- `latitude`, `longitude`

**Current Limitation:**
The script currently generates **synthetic atmospheric features** because:
1. Navigation file parsing is not yet implemented
2. ERA5 API credentials need to be configured
3. Actual download would require ~hours for all flight dates

**Usage (Synthetic Mode):**
```bash
cd cloudMLPublic
python sow_outputs/wp2_atmospheric_features.py \
    --config configs/bestComboConfig.yaml \
    --output sow_outputs/wp2_atmospheric/WP2_Features.hdf5 \
    --verbose
```

**Next Steps for Production:**
1. **Parse Navigation Files:**
   - Read `nFileName` HDF files from config
   - Extract lat/lon/time for each sample
   - Match with image timestamps

2. **Configure ERA5 API:**
   - Install: `pip install cdsapi`
   - Register at: https://cds.climate.copernicus.eu
   - Create `~/.cdsapirc` with API credentials

3. **Download ERA5 Data:**
   - Determine spatial/temporal extent from all flights
   - Download single-level variables (BLH, T2m, Td2m, SP)
   - Download pressure-level variables (T, Q, Z at 1000-700 hPa)
   - Store in NetCDF format

4. **Run Full Extraction:**
   - Load ERA5 datasets with xarray
   - Interpolate to each sample's location/time
   - Derive thermodynamic features
   - Save to WP2_Features.hdf5

**Alternative Approach (if ERA5 download is problematic):**
- Use local radiosonde data if available
- Use MERRA-2 or NCEP reanalysis as alternative
- Use climatological profiles as baseline

---

### ⏳ Work Package 3: Physical Baseline Model Validation
**Status:** NOT STARTED

**Purpose:** 
Train and validate a GBDT model using ONLY physical features (geometric + atmospheric) to test the core hypothesis. This is the **go/no-go gate** for the entire project.

**Requirements:**

1. **Input Features:**
   - Load WP1_Features.hdf5 (geometric features)
   - Load WP2_Features.hdf5 (atmospheric features)
   - Combine into single feature matrix
   - **EXCLUDE** image embeddings and raw solar angles

2. **Feature Vector Composition:**
   ```
   [Geometric Features]
   - derived_geometric_H
   - shadow_length_pixels
   - shadow_detection_confidence
   - (Optional: cloud/shadow positions, angles)
   
   [Atmospheric Features]
   - blh_m
   - lcl_m
   - inversion_height_m
   - moisture_gradient
   - stability_index
   - surface_temp_k
   - surface_dewpoint_k
   - lapse_rate_k_per_km
   ```

3. **Model Training:**
   - Algorithm: Gradient-Boosted Decision Trees (GBDT)
   - Library: XGBoost or LightGBM (consistent with prior experiments)
   - Hyperparameters: Use defaults or grid search
   - 5-fold Leave-One-Flight-Out Cross-Validation

4. **LOO CV Protocol (MANDATORY):**
   ```
   Fold 0: Train [F1, F2, F3, F4] → Test F0 (30Oct24, n=501)
   Fold 1: Train [F0, F2, F3, F4] → Test F1 (10Feb25, n=191)
   Fold 2: Train [F0, F1, F3, F4] → Test F2 (23Oct24, n=105)
   Fold 3: Train [F0, F1, F2, F4] → Test F3 (12Feb25, n=92)
   Fold 4: Train [F0, F1, F2, F3] → Test F4 (18Feb25, n=44)
   ```

5. **Evaluation Metrics (per fold and aggregated):**
   - R² (Coefficient of Determination)
   - MAE (Mean Absolute Error, km)
   - RMSE (Root Mean Squared Error, km)

6. **Success Criterion:**
   - **PASS:** Mean LOO CV R² > 0
   - **FAIL:** Mean LOO CV R² ≤ 0 → HALT and report hypothesis failure

**Deliverable:** `WP3_Report.json`

**Report Schema:**
```json
{
  "model": "Physical_Baseline_GBDT",
  "features": ["geometric", "atmospheric"],
  "n_samples": 933,
  "n_folds": 5,
  "folds": [
    {
      "fold_id": 0,
      "test_flight": "30Oct24",
      "n_train": 432,
      "n_test": 501,
      "r2": 0.XX,
      "mae_km": 0.XX,
      "rmse_km": 0.XX
    },
    // ... folds 1-4
  ],
  "aggregate_metrics": {
    "mean_r2": 0.XX,
    "std_r2": 0.XX,
    "mean_mae_km": 0.XX,
    "std_mae_km": 0.XX,
    "mean_rmse_km": 0.XX,
    "std_rmse_km": 0.XX
  },
  "pass_threshold": 0.0,
  "status": "PASS" or "FAIL"
}
```

**Implementation Plan:**

**Script:** `sow_outputs/wp3_physical_baseline.py`

```python
# Key components needed:
1. Load WP1 and WP2 features
2. Merge into single dataset indexed by sample_id
3. Split into 5 LOO folds based on flight_id
4. Train GBDT for each fold
5. Evaluate on held-out flight
6. Aggregate results
7. Generate WP3_Report.json
8. Check if mean R² > 0
```

**Next Steps:**
1. Create `wp3_physical_baseline.py` script
2. Implement LOO CV framework (can adapt from `validate_hybrid_loo.py`)
3. Run validation and generate report
4. If PASS → proceed to WP-4
5. If FAIL → analyze failure modes and revise hypothesis

---

### ⏳ Work Package 4: Hybrid Model Integration
**Status:** NOT STARTED

**Purpose:** 
Combine physical features with MAE image embeddings to achieve target performance (R² > 0.3).

**Requirements:**

1. **MAE Feature Extraction:**
   - Load pretrained MAE encoder: `outputs/mae_pretrain/mae_encoder_pretrained.pth`
   - **CRITICAL:** Use spatial features, NOT CLS token
   - Method: Global average pooling of patch tokens
   - Extract embeddings for all 933 samples
   - Dimension: Typically 256-512 (depends on MAE architecture)

2. **Feature Integration:**
   - Geometric features (from WP1)
   - Atmospheric features (from WP2)
   - MAE spatial embeddings (extracted here)
   - Solar angles (SZA, SAA) from metadata

3. **Model Variants (Ablation Study):**
   ```
   M_PHYSICAL_ONLY:     [Geometric + Atmospheric]
   M_PHYSICAL_ANGLES:   [Geometric + Atmospheric + SZA + SAA]
   M_PHYSICAL_MAE:      [Geometric + Atmospheric + MAE_Embeddings]
   M_HYBRID_FULL:       [Geometric + Atmospheric + MAE_Embeddings + SZA + SAA]
   ```

4. **Training and Validation:**
   - Train each variant using 5-fold LOO CV
   - Use same GBDT algorithm and protocol as WP-3
   - Evaluate with same metrics (R², MAE, RMSE)

5. **Feature Importance Analysis:**
   - Run permutation importance on best model (likely M_HYBRID_FULL)
   - Rank top 20 features
   - Categorize as: Geometric, Atmospheric, Angle, or MAE
   - Validate that physical features are driving predictions

**Deliverables:**

1. **final_features.hdf5:**
   - All 933 samples with all feature types
   - Indexed by sample_id
   - Fields:
     - Geometric_Features (from WP1)
     - Atmospheric_Features (from WP2)
     - MAE_Spatial_Embeddings (from WP4)
     - Metadata_Angles (SZA, SAA)
     - Target (CBH)

2. **WP4_Report.json:**
   - Similar to WP3_Report.json but with all 4 model variants
   - Comparison table showing ablation results
   - Identifies best-performing model

3. **WP4_Feature_Importance.json:**
   - Ranked list of top 20 features
   - Category labels for each feature
   - Importance scores

4. **models/final_gbdt_models/:**
   - 5 trained GBDT models (one per fold) for best variant
   - Saved as pickle files

5. **SOW_Validation_Summary.json:**
   - Final comparison table (Table 7.3a from SOW)
   - Compares new models (M4, M5) against failed baselines (M1-M3)

**Implementation Plan:**

**Script:** `sow_outputs/wp4_hybrid_models.py`

```python
# Key components:
1. Load pretrained MAE encoder
2. Extract spatial embeddings (global pooling of patch tokens)
3. Load WP1 and WP2 features
4. Combine all features into final_features.hdf5
5. Train 4 model variants with LOO CV
6. Compare performance across variants
7. Run feature importance on best model
8. Generate all deliverables
```

**Next Steps:**
1. Verify MAE encoder checkpoint exists and is loadable
2. Implement spatial embedding extraction (avoid CLS token)
3. Create feature integration pipeline
4. Run ablation studies
5. Generate final reports and models

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT DATA                                                  │
│  - 933 labeled cloud images (5 flights)                     │
│  - Metadata: SZA, SAA, timestamps                           │
│  - Navigation: lat, lon, altitude                           │
│  - Ground truth: CPL CBH measurements                       │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌───────────────┐  ┌──────────────────┐
│   WP-1        │  │   WP-2           │
│   Geometric   │  │   Atmospheric    │
│   Features    │  │   Features       │
│               │  │                  │
│ - Shadow      │  │ - ERA5 download  │
│   detection   │  │ - BLH, LCL       │
│ - Cloud-      │  │ - Inversion      │
│   shadow      │  │ - Stability      │
│   pairing     │  │ - Interpolation  │
│ - Geometric   │  │                  │
│   CBH         │  │                  │
└───────┬───────┘  └────────┬─────────┘
        │                   │
        │  WP1_Features     │  WP2_Features
        │  .hdf5            │  .hdf5
        │                   │
        └─────────┬─────────┘
                  │
                  ▼
        ┌─────────────────┐
        │   WP-3          │
        │   Physical      │
        │   Baseline      │
        │                 │
        │ - Combine       │
        │   WP1 + WP2     │
        │ - Train GBDT    │
        │ - LOO CV        │
        │ - Validate R²>0 │
        └────────┬────────┘
                 │
                 │ WP3_Report.json
                 │ PASS/FAIL decision
                 │
                 ▼
        ┌─────────────────┐
        │   WP-4          │
        │   Hybrid        │
        │   Models        │
        │                 │
        │ - MAE spatial   │
        │   embeddings    │
        │ - Ablation      │
        │   studies       │
        │ - Feature       │
        │   importance    │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────────────────┐
        │  FINAL DELIVERABLES         │
        │  - final_features.hdf5      │
        │  - WP4_Report.json          │
        │  - Feature_Importance.json  │
        │  - Trained models           │
        │  - Validation_Summary.json  │
        └─────────────────────────────┘
```

---

## Expected Outcomes

### Hypothesis Validation Scenarios

**Scenario A: Success (Physical features work)**
- WP-3 Physical Baseline: R² > 0 ✓ (PASS)
- WP-4 Hybrid models show incremental improvement
- Feature importance shows physical features dominate
- **Conclusion:** Shadow geometry + atmospheric profiles are essential

**Scenario B: Partial Success (Physical + MAE synergy)**
- WP-3 Physical Baseline: R² slightly > 0 ✓ (PASS)
- WP-4 Hybrid with MAE: Significant boost (R² > 0.3)
- Feature importance shows balanced contribution
- **Conclusion:** Physical features ground the MAE embeddings

**Scenario C: Failure (Hypothesis incorrect)**
- WP-3 Physical Baseline: R² ≤ 0 ✗ (FAIL)
- Project halts at WP-3
- **Action:** Revise hypothesis, explore alternative features

### Comparison to Failed Baselines

From project history (to be documented in final report):

| Model | Features | LOO CV R² | Status |
|-------|----------|-----------|--------|
| M1: Angles-Only | [SZA, SAA] | -4.46 ± 7.09 | FAILED |
| M2: MAE CLS Hybrid | [MAE_CLS, Angles] | TBD | FAILED |
| M3: Spatial MAE (Attn) | [MAE_Spatial, Angles] | -3.92 | FAILED |
| **M4: Physical Baseline** | **[Geometric, Atmospheric]** | **TBD** | **WP-3** |
| **M5: Hybrid Full** | **[Geo, Atm, MAE, Angles]** | **TBD** | **WP-4** |

Expected result: M4 > 0, M5 > 0.3

---

## Technical Implementation Notes

### 1. LOO CV Implementation
- Use flight_id to split data (not random split)
- Ensure no data leakage between folds
- Fit scalers on training data only
- Track per-fold metrics for variance analysis

### 2. Feature Scaling
- Standardize all features (zero mean, unit variance)
- Fit scalers on training folds only
- Apply same scaling to test folds
- Store scaler objects for inference

### 3. GBDT Hyperparameters
- Start with defaults from prior experiments
- Consider grid search if performance is borderline
- Key params: n_estimators, max_depth, learning_rate
- Use early stopping with validation set

### 4. MAE Embedding Extraction
```python
# Correct approach (spatial features):
encoder.eval()
with torch.no_grad():
    embeddings = encoder(images)  # (B, N_patches, D)
    spatial_features = embeddings.mean(dim=1)  # Global avg pooling

# Incorrect approach (DO NOT USE):
# cls_token = embeddings[:, 0, :]  # CLS token - proven ineffective
```

### 5. Missing Data Handling
- Shadow detection: Use confidence score, set low-confidence to NaN
- Atmospheric features: Interpolate or use nearest neighbor
- MAE embeddings: Should always be available
- During training: Impute NaN with median or use tree-based methods

### 6. Reproducibility
- Set random seeds for all experiments
- Save hyperparameters in reports
- Version control all scripts
- Document data preprocessing steps

---

## Dependencies

### Python Packages Required

**Core ML:**
```
numpy
scipy
scikit-learn
xgboost  # or lightgbm
torch
torchvision
h5py
```

**Image Processing:**
```
opencv-python
scikit-image
```

**Atmospheric Data:**
```
cdsapi  # ERA5 API
xarray
netcdf4
pandas
```

**Utilities:**
```
pyyaml
matplotlib  # for diagnostics
tqdm  # progress bars
```

### Installation
```bash
pip install numpy scipy scikit-learn xgboost torch torchvision h5py
pip install opencv-python scikit-image
pip install cdsapi xarray netcdf4 pandas
pip install pyyaml matplotlib tqdm
```

---

## Testing and Validation Checklist

### WP-1 Testing
- [ ] Run on small subset (10 samples) to verify pipeline
- [ ] Check shadow detection visualizations
- [ ] Validate confidence scores make sense
- [ ] Test edge cases (no shadows, multiple shadows, low sun angle)
- [ ] Run on full dataset
- [ ] Analyze success rate distribution across flights
- [ ] Tune scale factor if derived CBH is systematically off

### WP-2 Testing
- [ ] Parse navigation files successfully
- [ ] Configure ERA5 API credentials
- [ ] Download sample ERA5 data for one flight
- [ ] Verify interpolation works correctly
- [ ] Check derived features are physically reasonable
- [ ] Run on full dataset
- [ ] Validate temporal and spatial alignment

### WP-3 Testing
- [ ] Verify LOO CV splits are correct
- [ ] Check no data leakage between folds
- [ ] Validate metrics computation
- [ ] Test with synthetic features first
- [ ] Run full validation with real features
- [ ] Analyze per-fold variance
- [ ] Generate diagnostic plots (predicted vs actual)

### WP-4 Testing
- [ ] Verify MAE encoder loads correctly
- [ ] Check embedding dimensions are correct
- [ ] Validate spatial pooling (not CLS token)
- [ ] Test feature integration pipeline
- [ ] Run ablation studies systematically
- [ ] Verify feature importance makes sense
- [ ] Generate final comparison tables

---

## Timeline Estimate

**Assuming access to compute resources and data:**

- **WP-1 Execution:** 2-4 hours (full dataset processing)
- **WP-2 Setup:** 1-2 hours (navigation parsing, ERA5 config)
- **WP-2 ERA5 Download:** 4-8 hours (depends on CDS queue)
- **WP-2 Processing:** 1-2 hours
- **WP-3 Training:** 2-4 hours (5 folds × GBDT training)
- **WP-4 MAE Extraction:** 1-2 hours
- **WP-4 Training:** 4-6 hours (4 variants × 5 folds)
- **Analysis & Reporting:** 2-4 hours

**Total:** ~20-35 hours of compute + human time

---

## Known Limitations and Risks

### WP-1 Risks
1. **Shadow detection may fail in many cases:**
   - Ocean scenes have low contrast
   - Multi-layer clouds create ambiguous shadows
   - Broken clouds have no clear shadows
   - **Mitigation:** Use confidence scoring, accept partial success

2. **Scale factor uncertainty:**
   - Ground sampling distance varies with altitude
   - May need per-flight calibration
   - **Mitigation:** Use derived CBH as feature, not direct prediction

### WP-2 Risks
1. **ERA5 spatial resolution mismatch:**
   - 25 km grid vs 200 m imagery
   - Atmospheric features may be too coarse
   - **Mitigation:** Use as general context, not precise location

2. **Navigation data parsing:**
   - HDF format may be non-standard
   - Timestamp alignment may be tricky
   - **Mitigation:** Validate with known flight paths

3. **ERA5 API reliability:**
   - CDS may be slow or unavailable
   - Download limits exist
   - **Mitigation:** Cache all downloads, use offline mode

### WP-3 Risks
1. **Physical features may not generalize:**
   - Core hypothesis could be wrong
   - R² may still be ≤ 0
   - **Outcome:** Project halts, hypothesis revised

2. **Small sample size for some flights:**
   - F4 has only 44 samples
   - May lead to high variance
   - **Mitigation:** Report per-fold metrics and variance

### WP-4 Risks
1. **MAE embeddings may not help:**
   - Already proven ineffective alone
   - May not synergize with physical features
   - **Mitigation:** Ablation studies will reveal this

2. **Feature importance may be unclear:**
   - GBDT feature importance can be unstable
   - Correlated features confound interpretation
   - **Mitigation:** Use permutation importance, multiple runs

---

## Contact and Support

**Project Lead:** Principal Investigator (Atmospheric Research)  
**Autonomous Agent:** SOW-AGENT-CBH-WP-001  
**Documentation:** `ScopeWorkSprint3.md`

**Key Reference Documents:**
- Project status: `docs/project_status_report.pdf`
- One-page summary: `docs/ONE_PAGE_SUMMARY.md`
- Critical findings: Various docs in project archive

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025 | Initial implementation guide | Agent |

---

## Appendix: File Structure

```
cloudMLPublic/
├── sow_outputs/
│   ├── SOW_IMPLEMENTATION_GUIDE.md  (this file)
│   │
│   ├── wp1_geometric/
│   │   ├── WP1_Features.hdf5        (deliverable)
│   │   └── diagnostics/             (optional)
│   │
│   ├── wp2_atmospheric/
│   │   ├── WP2_Features.hdf5        (deliverable)
│   │   └── era5_data/               (cached downloads)
│   │
│   ├── wp3_baseline/
│   │   ├── WP3_Report.json          (deliverable)
│   │   └── diagnostics/             (optional)
│   │
│   ├── wp4_hybrid/
│   │   ├── WP4_Report.json          (deliverable)
│   │   ├── WP4_Feature_Importance.json
│   │   └── final_features.hdf5      (deliverable)
│   │
│   ├── models/
│   │   └── final_gbdt_models/       (5 trained models)
│   │
│   ├── SOW_Validation_Summary.json  (final deliverable)
│   │
│   ├── wp1_geometric_features.py    (WP-1 script)
│   ├── wp2_atmospheric_features.py  (WP-2 script)
│   ├── wp3_physical_baseline.py     (WP-3 script - TBD)
│   └── wp4_hybrid_models.py         (WP-4 script - TBD)
│
├── configs/
│   └── bestComboConfig.yaml         (configuration)
│
├── src/
│   └── ...                          (existing codebase)
│
└── ScopeWorkSprint3.md              (requirements)
```

---

**END OF IMPLEMENTATION GUIDE**