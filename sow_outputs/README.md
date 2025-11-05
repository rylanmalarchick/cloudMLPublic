# SOW Sprint 3: Physics-Constrained CBH Model Validation

This directory contains the implementation of the Physics-Constrained CBH Model Validation project as specified in `../ScopeWorkSprint3.md`.

## 🎯 Project Goal

Validate the hypothesis that **physics-constrained features** (shadow geometry + atmospheric thermodynamics) are essential for achieving cross-flight generalization in Cloud Base Height (CBH) retrieval.

## 📊 Success Criteria

- **Minimum:** Physical Baseline Model (WP-3) achieves LOO CV R² > 0
- **Target:** Final Hybrid Model (WP-4) achieves LOO CV R² > 0.3

## 🚀 Quick Start

### Run Work Packages 1 & 2 (Feature Extraction)

```bash
cd cloudMLPublic
./sow_outputs/run_sow.sh --verbose
```

This will:
1. Extract geometric features (shadow-based CBH estimation)
2. Extract atmospheric features (BLH, LCL, stability, etc.)

### Run Complete Pipeline

```bash
# After implementing WP3 and WP4:
./sow_outputs/run_sow.sh --run-wp3 --run-wp4 --verbose
```

## 📁 Directory Structure

```
sow_outputs/
├── README.md                         (this file)
├── SOW_IMPLEMENTATION_GUIDE.md       (detailed technical guide)
├── run_sow.sh                        (execution script)
│
├── wp1_geometric_features.py         ✅ IMPLEMENTED
├── wp2_atmospheric_features.py       ✅ IMPLEMENTED  
├── wp3_physical_baseline.py          ⏳ TODO
├── wp4_hybrid_models.py              ⏳ TODO
│
├── wp1_geometric/
│   └── WP1_Features.hdf5             (geometric features)
│
├── wp2_atmospheric/
│   ├── WP2_Features.hdf5             (atmospheric features)
│   └── era5_data/                    (cached ERA5 downloads)
│
├── wp3_baseline/
│   └── WP3_Report.json               (validation results)
│
├── wp4_hybrid/
│   ├── final_features.hdf5           (all features combined)
│   ├── WP4_Report.json               (ablation study results)
│   └── WP4_Feature_Importance.json   (feature rankings)
│
├── models/
│   └── final_gbdt_models/            (5 trained models)
│
└── SOW_Validation_Summary.json       (final comparison table)
```

## 📦 Work Packages

### ✅ WP-1: Geometric Feature Engineering (COMPLETE)

**Script:** `wp1_geometric_features.py`

**What it does:**
- Detects shadows in cloud imagery
- Pairs clouds with their shadows using solar azimuth
- Estimates CBH from shadow length: `H = L × tan(90° - SZA) × scale`
- Assigns confidence scores (0-1) to each detection

**Features extracted (933 samples):**
- `derived_geometric_H` - Estimated CBH from shadow geometry (km)
- `shadow_length_pixels` - Measured shadow length
- `shadow_detection_confidence` - Quality score (0-1)
- `cloud_edge_x/y`, `shadow_edge_x/y` - Spatial locations
- `sza_deg`, `saa_deg` - Solar angles

**Run:**
```bash
python sow_outputs/wp1_geometric_features.py \
    --config configs/bestComboConfig.yaml \
    --output sow_outputs/wp1_geometric/WP1_Features.hdf5 \
    --scale 50.0 \
    --verbose
```

**Parameters:**
- `--scale`: Ground sampling distance (m/pixel). Default: 50.0
  - Adjust based on aircraft altitude and camera FOV
  - For ER-2 at ~20 km: typically 20-100 m/pixel

**Expected output:**
- Success rate: 30-70% (shadows not always detectable)
- High-confidence detections: 20-50%
- Valid CBH estimates: Depends on sun angle and scene conditions

---

### ✅ WP-2: Atmospheric Feature Engineering (COMPLETE - Synthetic Mode)

**Script:** `wp2_atmospheric_features.py`

**What it does:**
- Downloads ERA5 reanalysis data (or uses synthetic)
- Derives thermodynamic variables (BLH, LCL, inversions, stability)
- Interpolates from 25 km ERA5 grid to sample locations

**Features extracted (933 samples):**
- `blh_m` - Boundary Layer Height (meters)
- `lcl_m` - Lifting Condensation Level (meters)
- `inversion_height_m` - Temperature inversion altitude (meters)
- `moisture_gradient` - Vertical moisture gradient (kg/kg/m)
- `stability_index` - Atmospheric lapse rate (K/km)
- `surface_temp_k`, `surface_dewpoint_k` - Surface conditions
- `surface_pressure_pa` - Surface pressure

**Run:**
```bash
python sow_outputs/wp2_atmospheric_features.py \
    --config configs/bestComboConfig.yaml \
    --output sow_outputs/wp2_atmospheric/WP2_Features.hdf5 \
    --verbose
```

**Current limitation:** Uses synthetic features because:
1. Navigation file parsing not implemented (need lat/lon/time)
2. ERA5 API credentials need configuration

**For production setup:**
1. Install ERA5 tools: `pip install cdsapi xarray netcdf4`
2. Register at: https://cds.climate.copernicus.eu
3. Configure `~/.cdsapirc` with API key
4. Implement navigation file parser (extract lat/lon/time from HDF)
5. Download ERA5 data for all flight dates
6. Re-run WP-2 with real data

---

### ⏳ WP-3: Physical Baseline Model Validation (TODO)

**Status:** NOT IMPLEMENTED

**Purpose:** Train GBDT using ONLY physical features to test core hypothesis

**Critical requirement:** This is the **GO/NO-GO gate** for the project
- **PASS if:** Mean LOO CV R² > 0 → Proceed to WP-4
- **FAIL if:** Mean LOO CV R² ≤ 0 → Hypothesis rejected, revise approach

**Implementation needed:**
```python
# wp3_physical_baseline.py

1. Load WP1_Features.hdf5 and WP2_Features.hdf5
2. Merge on sample_id
3. Create feature matrix [Geometric + Atmospheric]
4. Implement 5-fold Leave-One-Flight-Out CV:
   - Fold 0: Train [F1,F2,F3,F4] → Test F0 (30Oct24, n=501)
   - Fold 1: Train [F0,F2,F3,F4] → Test F1 (10Feb25, n=191)
   - Fold 2: Train [F0,F1,F3,F4] → Test F2 (23Oct24, n=105)
   - Fold 3: Train [F0,F1,F2,F4] → Test F3 (12Feb25, n=92)
   - Fold 4: Train [F0,F1,F2,F3] → Test F4 (18Feb25, n=44)
5. Train GBDT for each fold (XGBoost or LightGBM)
6. Evaluate on held-out flight
7. Compute metrics: R², MAE, RMSE
8. Aggregate results and generate WP3_Report.json
9. Check if mean R² > 0
```

**Reference:** Adapt from `scripts/validate_hybrid_loo.py`

**Deliverable:** `WP3_Report.json`

---

### ⏳ WP-4: Hybrid Model Integration (TODO)

**Status:** NOT IMPLEMENTED

**Purpose:** Combine physical features with MAE embeddings for final model

**Prerequisites:** WP-3 must PASS (R² > 0)

**Implementation needed:**
```python
# wp4_hybrid_models.py

1. Load pretrained MAE encoder: outputs/mae_pretrain/mae_encoder_pretrained.pth
2. Extract spatial embeddings (CRITICAL: Use patch tokens with global pooling, NOT CLS token)
3. Load WP1 and WP2 features
4. Combine all features into final_features.hdf5
5. Train 4 model variants with LOO CV:
   - M_PHYSICAL_ONLY:   [Geometric + Atmospheric]
   - M_PHYSICAL_ANGLES: [Geometric + Atmospheric + SZA + SAA]
   - M_PHYSICAL_MAE:    [Geometric + Atmospheric + MAE_Embeddings]
   - M_HYBRID_FULL:     [Geometric + Atmospheric + MAE + SZA + SAA]
6. Compare performance (ablation study)
7. Run permutation importance on best model
8. Generate all deliverables
```

**MAE embedding extraction (CRITICAL):**
```python
# CORRECT (spatial features):
encoder.eval()
with torch.no_grad():
    embeddings = encoder(images)  # (B, N_patches, D)
    spatial_features = embeddings.mean(dim=1)  # Global avg pooling

# WRONG (proven ineffective):
# cls_token = embeddings[:, 0, :]  # DO NOT USE
```

**Deliverables:**
- `final_features.hdf5` - All features combined
- `WP4_Report.json` - Ablation study results
- `WP4_Feature_Importance.json` - Top 20 features ranked
- `models/final_gbdt_models/` - 5 trained models (best variant)
- `SOW_Validation_Summary.json` - Final comparison table

---

## 📈 Expected Results

### Hypothesis Validation Scenarios

**Scenario A: Full Success**
- WP-3: R² = 0.1-0.2 (modest but positive)
- WP-4 Hybrid: R² = 0.3-0.5 (target achieved)
- Feature importance shows physical features dominate
- ✅ Hypothesis confirmed

**Scenario B: Partial Success**  
- WP-3: R² = 0.05 (barely positive)
- WP-4 Hybrid: R² = 0.2-0.3 (improvement but below target)
- Physical + MAE synergy observed
- ⚠️ Hypothesis partially validated

**Scenario C: Failure**
- WP-3: R² ≤ 0 (no generalization)
- ❌ Hypothesis rejected
- Project halts, new approach needed

### Comparison to Failed Baselines

| Model | Features | LOO CV R² | Status |
|-------|----------|-----------|--------|
| M1: Angles-Only | [SZA, SAA] | -4.46 ± 7.09 | FAILED |
| M2: MAE CLS Hybrid | [MAE_CLS, Angles] | < 0 | FAILED |
| M3: Spatial MAE (Attn) | [MAE_Spatial, Angles] | -3.92 | FAILED |
| **M4: Physical Baseline** | **[Geo, Atm]** | **TBD (WP-3)** | **?** |
| **M5: Hybrid Full** | **[Geo, Atm, MAE, Angles]** | **TBD (WP-4)** | **?** |

**Goal:** M4 > 0, M5 > 0.3

---

## 🔧 Technical Notes

### Feature Scaling
- Standardize all features (zero mean, unit variance)
- Fit scalers on training data only
- Apply same scaling to test data

### LOO CV Implementation
- Split by flight_id (not random)
- No data leakage between folds
- Track per-fold metrics for variance analysis

### Missing Data
- Shadow detection: Low confidence → NaN → Impute or flag
- Atmospheric: Interpolate or use nearest neighbor
- GBDT handles missing values naturally

### Reproducibility
- Set random seeds: `np.random.seed(42)`, `torch.manual_seed(42)`
- Save all hyperparameters in reports
- Version control all scripts

---

## 📚 Documentation

- **SOW Requirements:** `../ScopeWorkSprint3.md`
- **Implementation Guide:** `SOW_IMPLEMENTATION_GUIDE.md` (detailed)
- **Project Status:** `../docs/project_status_report.pdf`
- **Quick Summary:** `../docs/ONE_PAGE_SUMMARY.md`

---

## 🛠️ Dependencies

**Required packages:**
```bash
# Core ML
pip install numpy scipy scikit-learn xgboost torch torchvision h5py

# Image processing
pip install opencv-python scikit-image

# Atmospheric (optional for WP-2 production)
pip install cdsapi xarray netcdf4 pandas

# Utilities
pip install pyyaml matplotlib tqdm
```

---

## ✅ Testing Checklist

### Before running WP-1:
- [ ] Verify config file exists: `configs/bestComboConfig.yaml`
- [ ] Check data files are accessible (images, metadata)
- [ ] Test on small subset first (10 samples)

### Before running WP-2:
- [ ] Decide: synthetic or real ERA5 data?
- [ ] If real: Configure ERA5 API credentials
- [ ] If real: Implement navigation file parser

### Before running WP-3:
- [ ] Verify WP1_Features.hdf5 exists
- [ ] Verify WP2_Features.hdf5 exists
- [ ] Check sample counts match (933 total)
- [ ] Validate flight splits are correct

### Before running WP-4:
- [ ] Verify WP-3 passed (R² > 0)
- [ ] Check MAE encoder checkpoint exists
- [ ] Test MAE embedding extraction on sample

---

## 🐛 Troubleshooting

**WP-1 low success rate:**
- Adjust `--scale` parameter (try 30-100 m/pixel)
- Check shadow detection thresholds in code
- Visualize detections to diagnose issues

**WP-2 synthetic data concerns:**
- Synthetic features are physically plausible
- Useful for pipeline testing
- Replace with real ERA5 for production

**WP-3 fails (R² ≤ 0):**
- Check feature distributions (any NaN, inf, outliers?)
- Try different GBDT hyperparameters
- Analyze per-fold results (which flights fail?)
- May need to revise hypothesis

**WP-4 no improvement over WP-3:**
- Verify using spatial MAE features (not CLS)
- Check embedding dimensions are correct
- Try different feature combinations
- MAE may not add value (this is a valid finding)

---

## 📞 Support

**Project context:** See `../ScopeWorkSprint3.md` for full requirements

**Questions?**
1. Check `SOW_IMPLEMENTATION_GUIDE.md` for detailed technical guide
2. Review project history in `../docs/`
3. Examine existing scripts in `../scripts/` for examples

---

## 🎯 Next Actions

1. **Run WP-1 and WP-2** to generate feature files
2. **Analyze results** - Check distributions, correlations, missing data
3. **Implement WP-3** - Physical baseline validation (go/no-go gate)
4. **If WP-3 passes** - Implement WP-4 hybrid models
5. **Generate final report** - SOW_Validation_Summary.json

**Start here:**
```bash
cd cloudMLPublic
./sow_outputs/run_sow.sh --verbose
```

Good luck! 🚀