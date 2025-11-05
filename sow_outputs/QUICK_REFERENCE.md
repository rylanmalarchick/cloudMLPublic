# SOW Sprint 3: Quick Reference Card

**Project:** Physics-Constrained CBH Model Validation  
**Goal:** Validate that shadow geometry + atmospheric features enable cross-flight generalization

---

## 🎯 Success Criteria

| Metric | Threshold | Status |
|--------|-----------|--------|
| WP-3 Physical Baseline R² | > 0 | ⏳ Pending |
| WP-4 Hybrid Model R² | > 0.3 | ⏳ Pending |

**GO/NO-GO Gate:** WP-3 must achieve R² > 0 to proceed to WP-4

---

## 🚀 Quick Start

```bash
# Navigate to project root
cd cloudMLPublic

# Run feature extraction (WP-1 and WP-2)
./sow_outputs/run_sow.sh --verbose

# Expected output:
# - sow_outputs/wp1_geometric/WP1_Features.hdf5
# - sow_outputs/wp2_atmospheric/WP2_Features.hdf5
```

---

## 📊 Work Package Status

```
┌─────────────────────────────────────────────────────────────┐
│  WP-1: Geometric Features              ✅ COMPLETE          │
│  - Shadow detection                                         │
│  - Cloud-shadow pairing                                     │
│  - Geometric CBH estimation                                 │
│  - Deliverable: WP1_Features.hdf5                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  WP-2: Atmospheric Features            ✅ COMPLETE          │
│  - ERA5 data integration                                    │
│  - BLH, LCL, inversion, stability                           │
│  - Spatio-temporal interpolation                            │
│  - Deliverable: WP2_Features.hdf5                           │
│  - NOTE: Currently using synthetic data                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  WP-3: Physical Baseline               ⏳ TODO              │
│  - Train GBDT on [Geometric + Atmospheric]                  │
│  - 5-fold Leave-One-Flight-Out CV                           │
│  - GO/NO-GO GATE: R² must be > 0                            │
│  - Deliverable: WP3_Report.json                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  WP-4: Hybrid Models                   ⏳ TODO              │
│  - Add MAE spatial embeddings + angles                      │
│  - Ablation study (4 variants)                              │
│  - Feature importance analysis                              │
│  - Deliverables: final_features.hdf5, WP4_Report.json       │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 File Locations

```
cloudMLPublic/
├── sow_outputs/
│   ├── README.md                       📖 Start here
│   ├── SOW_IMPLEMENTATION_GUIDE.md     📚 Detailed guide
│   ├── QUICK_REFERENCE.md              ⚡ This file
│   ├── run_sow.sh                      🚀 Execution script
│   │
│   ├── wp1_geometric_features.py       ✅ WP-1 (704 lines)
│   ├── wp2_atmospheric_features.py     ✅ WP-2 (769 lines)
│   ├── wp3_physical_baseline.py        ⏳ TODO
│   ├── wp4_hybrid_models.py            ⏳ TODO
│   │
│   └── [wp1_geometric/, wp2_atmospheric/, wp3_baseline/, wp4_hybrid/]
│
└── ScopeWorkSprint3.md                 📋 Requirements
```

---

## 🔄 Workflow Diagram

```
INPUT: 933 Labeled Cloud Images (5 flights)
  │
  ├─────────────────────────────────────┐
  │                                     │
  ▼                                     ▼
┌─────────────────┐           ┌─────────────────┐
│  WP-1           │           │  WP-2           │
│  Geometric      │           │  Atmospheric    │
│  Features       │           │  Features       │
│                 │           │                 │
│  • Shadow       │           │  • ERA5 data    │
│    detection    │           │  • BLH, LCL     │
│  • Cloud-       │           │  • Inversions   │
│    shadow       │           │  • Stability    │
│    pairing      │           │  • Moisture     │
│  • Geometric    │           │    gradients    │
│    CBH          │           │                 │
└────────┬────────┘           └────────┬────────┘
         │                             │
         │  WP1_Features.hdf5          │  WP2_Features.hdf5
         │                             │
         └────────────┬────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │  WP-3           │
            │  Physical       │
            │  Baseline       │
            │                 │
            │  Features:      │
            │  [Geo + Atm]    │
            │                 │
            │  Model: GBDT    │
            │  CV: LOO (5x)   │
            └────────┬────────┘
                     │
                     │  WP3_Report.json
                     │
                     ▼
            ┌────────────────┐
            │  Decision Gate │
            │  R² > 0?       │
            └────┬───────┬───┘
                 │       │
            FAIL │       │ PASS
                 │       │
                 ▼       ▼
            ┌─────┐  ┌─────────────────┐
            │STOP │  │  WP-4           │
            │     │  │  Hybrid         │
            └─────┘  │  Models         │
                     │                 │
                     │  • MAE spatial  │
                     │    embeddings   │
                     │  • 4 variants   │
                     │  • Ablation     │
                     │  • Feature      │
                     │    importance   │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  Final          │
                     │  Deliverables   │
                     │                 │
                     │  • Models       │
                     │  • Reports      │
                     │  • Summary      │
                     └─────────────────┘
```

---

## 🎨 Feature Summary

### WP-1: Geometric Features (10 features)
```
✓ derived_geometric_H          Cloud base height estimate (km)
✓ shadow_length_pixels         Shadow length measurement
✓ shadow_detection_confidence  Quality score (0-1)
✓ cloud_edge_x, cloud_edge_y   Cloud position
✓ shadow_edge_x, shadow_edge_y Shadow position
✓ shadow_angle_deg             Shadow direction
✓ sza_deg, saa_deg             Solar angles
✓ true_cbh_km                  Ground truth
```

### WP-2: Atmospheric Features (10 features)
```
✓ blh_m                        Boundary layer height
✓ lcl_m                        Lifting condensation level
✓ inversion_height_m           Temperature inversion
✓ moisture_gradient            Vertical humidity gradient
✓ stability_index              Lapse rate (K/km)
✓ surface_temp_k               Surface temperature
✓ surface_dewpoint_k           Surface dewpoint
✓ surface_pressure_pa          Surface pressure
✓ lapse_rate_k_per_km         Temperature lapse rate
✓ profile_confidence           Quality score
```

---

## 🔬 Model Variants (WP-4)

```
M_PHYSICAL_ONLY:   [Geometric + Atmospheric]
                   ↓ Baseline - tests core hypothesis

M_PHYSICAL_ANGLES: [Geometric + Atmospheric + SZA + SAA]
                   ↓ Tests if angles add value to physical

M_PHYSICAL_MAE:    [Geometric + Atmospheric + MAE_Embeddings]
                   ↓ Tests if MAE synergizes with physical

M_HYBRID_FULL:     [Geometric + Atmospheric + MAE + SZA + SAA]
                   ↓ All features - expected best performance
```

---

## 📈 Expected Performance

| Model | Expected R² | Notes |
|-------|-------------|-------|
| Angles-Only (historical) | -4.46 | ❌ Failed - temporal confounding |
| MAE CLS (historical) | < 0 | ❌ Failed - no generalization |
| **Physical Baseline (WP-3)** | **0.05 - 0.20** | **✓ Target: > 0** |
| **Hybrid Full (WP-4)** | **0.30 - 0.50** | **✓ Target: > 0.3** |

---

## ⚙️ Command Reference

### Run Everything (Step-by-Step)
```bash
# Step 1: Extract features
./sow_outputs/run_sow.sh --verbose

# Step 2: Implement WP-3 (see guide)
# Create: sow_outputs/wp3_physical_baseline.py

# Step 3: Run WP-3
./sow_outputs/run_sow.sh --run-wp3 --verbose

# Step 4: If WP-3 passes, implement WP-4 (see guide)
# Create: sow_outputs/wp4_hybrid_models.py

# Step 5: Run WP-4
./sow_outputs/run_sow.sh --run-wp4 --verbose
```

### Run Individual Components
```bash
# WP-1 only
python sow_outputs/wp1_geometric_features.py \
    --config configs/bestComboConfig.yaml \
    --output sow_outputs/wp1_geometric/WP1_Features.hdf5 \
    --scale 50.0 --verbose

# WP-2 only
python sow_outputs/wp2_atmospheric_features.py \
    --config configs/bestComboConfig.yaml \
    --output sow_outputs/wp2_atmospheric/WP2_Features.hdf5 \
    --verbose

# Skip WP-1 (use cached)
./sow_outputs/run_sow.sh --skip-wp1 --verbose
```

---

## 🔍 Data Inspection

### Check Feature Files
```python
import h5py
import numpy as np

# WP-1 features
with h5py.File('sow_outputs/wp1_geometric/WP1_Features.hdf5', 'r') as f:
    print("Keys:", list(f.keys()))
    print("Samples:", len(f['sample_id']))
    
    cbh = f['derived_geometric_H'][:]
    conf = f['shadow_detection_confidence'][:]
    
    print(f"Valid CBH: {np.sum(~np.isnan(cbh))}/{len(cbh)}")
    print(f"Mean confidence: {np.mean(conf):.3f}")
    print(f"High conf (>0.5): {np.sum(conf > 0.5)}")

# WP-2 features  
with h5py.File('sow_outputs/wp2_atmospheric/WP2_Features.hdf5', 'r') as f:
    print("\nKeys:", list(f.keys()))
    print("Samples:", len(f['sample_id']))
    
    blh = f['blh_m'][:]
    lcl = f['lcl_m'][:]
    
    print(f"BLH range: {np.min(blh):.0f} - {np.max(blh):.0f} m")
    print(f"LCL range: {np.min(lcl):.0f} - {np.max(lcl):.0f} m")
```

---

## 🐛 Common Issues

### WP-1: Low Success Rate
```
Problem: < 20% shadow detections
Solution: 
  1. Adjust --scale parameter (try 30-100)
  2. Lower confidence threshold in code
  3. Check if sun angles are too high (SZA > 70°)
```

### WP-2: Synthetic Data Warning
```
Note: Currently using synthetic atmospheric features
For production:
  1. pip install cdsapi xarray netcdf4
  2. Register at cds.climate.copernicus.eu
  3. Configure ~/.cdsapirc
  4. Implement nav file parser
  5. Download ERA5 data
```

### WP-3: R² ≤ 0 (Failure)
```
If WP-3 fails:
  1. Check feature distributions (NaN, outliers?)
  2. Visualize predicted vs actual per fold
  3. Try different GBDT hyperparameters
  4. Analyze which flights fail worst
  5. May need to revise hypothesis
```

---

## 📚 Documentation Links

- **Quick Start:** `README.md`
- **Full Technical Guide:** `SOW_IMPLEMENTATION_GUIDE.md`
- **Work Completed:** `WORK_COMPLETED_SUMMARY.md`
- **Requirements:** `../ScopeWorkSprint3.md`
- **Project Background:** `../docs/project_status_report.pdf`

---

## 📊 Dataset Summary

| Flight | Date | Samples | Notes |
|--------|------|---------|-------|
| F0 | 30Oct24 | 501 | Largest flight |
| F1 | 10Feb25 | 191 | |
| F2 | 23Oct24 | 105 | |
| F3 | 12Feb25 | 92 | |
| F4 | 18Feb25 | 44 | Smallest flight |
| **Total** | | **933** | |

---

## ⏱️ Time Estimates

| Task | Duration |
|------|----------|
| WP-1 Execution | 2-4 hours |
| WP-2 Setup (real ERA5) | 4-8 hours |
| WP-3 Implementation | 4-6 hours |
| WP-3 Execution | 2-4 hours |
| WP-4 Implementation | 4-6 hours |
| WP-4 Execution | 4-6 hours |
| **Total** | **20-35 hours** |

---

## ✅ Next Actions

1. **NOW:** Run `./sow_outputs/run_sow.sh --verbose`
2. **NEXT:** Analyze WP1/WP2 outputs
3. **THEN:** Implement WP-3 (see guide)
4. **FINALLY:** If WP-3 passes, implement WP-4

---

**Need help?** See `SOW_IMPLEMENTATION_GUIDE.md` for detailed instructions.

**Ready to start?**
```bash
cd cloudMLPublic && ./sow_outputs/run_sow.sh --verbose
```
