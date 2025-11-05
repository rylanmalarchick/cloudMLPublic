# WP-2 COMPLETION SUMMARY

## Status: ✅ COMPLETE

**Date:** November 5, 2025  
**Duration:** 5h 13m (initial download) + 2m (fix + re-extraction)  
**Total ERA5 Data Downloaded:** 1.04 GB (123 MB surface + 922 MB pressure)

---

## What Was Accomplished

### 1. Real ERA5 Data Downloaded
- **120 days** of ERA5 reanalysis data (2024-10-23 → 2025-02-19)
- **Surface variables** (single-level): BLH, T2M, D2M, SP, TCWV
- **Pressure-level variables**: Temperature, geopotential, specific humidity at 13 levels
- **Spatial coverage**: [21.30°N, 44.93°N] × [-127.65°W, -115.82°W]
- **Storage location**: `/media/rylan/two/research/NASA/ERA5_data_root/`

### 2. Atmospheric Features Extracted
- **933 samples** successfully processed (100% valid, 0% NaN)
- **9 atmospheric features** per sample:
  1. `blh` - Boundary layer height (km)
  2. `lcl` - Lifting condensation level (km)
  3. `inversion_height` - Temperature inversion height (km)
  4. `moisture_gradient` - Vertical moisture gradient (kg/kg/m)
  5. `stability_index` - Atmospheric stability (K/km lapse rate)
  6. `t2m` - 2-meter temperature (K)
  7. `d2m` - 2-meter dewpoint temperature (K)
  8. `sp` - Surface pressure (Pa)
  9. `tcwv` - Total column water vapor (kg/m²)

### 3. Output File Created
- **Location**: `sow_outputs/wp2_atmospheric/WP2_Features.hdf5`
- **Size**: 46 KB
- **Format**: HDF5 with datasets:
  - `features`: (933, 9) float array
  - `feature_names`: (9,) string array
  - `sample_indices`: (933,) int array
  - `timestamps`: (933,) string array
  - `latitudes`: (933,) float array
  - `longitudes`: (933,) float array

---

## Feature Statistics (Real ERA5 Data)

```
Feature              Valid   Mean        Std        Min        Max
─────────────────────────────────────────────────────────────────
blh (km)             933/933    0.849      0.348      0.124      2.181
lcl (km)             933/933    1.178      0.889      0.000      5.373
inversion_height     933/933    1.643      1.015      0.000      5.806
moisture_gradient    933/933   -0.000      0.000     -0.000     -0.000
stability_index      933/933   -4.997      0.865     -7.276     -2.652
t2m (K)              933/933  287.677      6.744    271.519    302.043
d2m (K)              933/933  278.254      9.046    252.154    293.848
sp (Pa)              933/933 96832.8     7486.4    75713.3   104118.1
tcwv (kg/m²)         933/933   13.842      7.732      2.921     39.068
```

**Key observations:**
- BLH typically 0.5–1.5 km (physically reasonable for marine boundary layer)
- LCL shows high variability (0–5 km), indicating diverse moisture conditions
- Negative stability index confirms stable stratification (expected over ocean)
- Surface pressure varies ~30 kPa (elevation + weather systems)
- Water vapor column highly variable (dry to very humid conditions)

---

## Critical Bug Fixed

### Problem
Initial run downloaded all ERA5 data successfully but **failed to extract any features**:
- All 933 samples returned NaN
- Error: `'time' is not a valid dimension or coordinate`
- Root cause: ERA5 NetCDF files use `valid_time` dimension, not `time`

### Solution
Updated `sow_outputs/wp2_era5_real.py` lines 730 and 751:
```python
# BEFORE (broken):
ds_interp = ds_surface.sel(
    latitude=metadata.latitude,
    longitude=metadata.longitude,
    time=sample_time,  # ❌ Wrong dimension name
    method="nearest",
)

# AFTER (fixed):
ds_interp = ds_surface.sel(
    latitude=metadata.latitude,
    longitude=metadata.longitude,
    valid_time=sample_time,  # ✅ Correct dimension name
    method="nearest",
)
```

### Result
Re-ran feature extraction (skipped download, files already present):
- **Duration**: ~2 minutes
- **Success rate**: 933/933 samples (100%)
- **NaN rate**: 0/933 samples (0%)

---

## Data Flow Verification

### 1. Navigation Parsing
✅ Successfully parsed all 933 samples from HDF5 dataset:
- Extracted lat/lon/time from CPL navigation files
- Computed spatial/temporal bounds for ERA5 download
- Saved metadata to `sample_metadata.json`

### 2. ERA5 Download
✅ Downloaded 240 NetCDF files (120 surface + 120 pressure):
- Each file: 1 day of hourly data (24 timesteps)
- Spatial grid: ~95 lat × 48 lon points
- Resumable: script skips already-downloaded files

### 3. Feature Extraction
✅ Interpolated ERA5 to each sample's exact location/time:
- Nearest-neighbor interpolation in space and time
- Derived physical quantities (LCL, inversions, stability)
- Validated all features are within physically reasonable ranges

### 4. Output Validation
✅ HDF5 file structure confirmed:
```python
import h5py
with h5py.File('sow_outputs/wp2_atmospheric/WP2_Features.hdf5', 'r') as f:
    assert f['features'].shape == (933, 9)
    assert len(f['feature_names']) == 9
    assert not np.any(np.isnan(f['features'][:]))  # No NaNs!
```

---

## Comparison: Synthetic vs. Real ERA5

| Aspect               | Previous (Synthetic) | Current (Real ERA5) |
|----------------------|----------------------|---------------------|
| Data source          | Random numbers       | CDS ERA5 reanalysis |
| Download time        | 0 seconds            | ~5 hours            |
| Storage required     | 0 MB                 | 1.04 GB             |
| Feature validity     | 100% fake            | 100% real           |
| Physical realism     | None                 | High                |
| WP-3 validation      | Meaningless          | Valid               |

---

## Next Steps

### Immediate: WP-3 Physical Baseline Validation
```bash
python sow_outputs/wp3_physical_baseline.py \
    --wp2-features sow_outputs/wp2_atmospheric/WP2_Features.hdf5 \
    --verbose
```

**Objective**: Leave-one-out cross-validation to validate that real atmospheric features correlate with CBH.

**Success criterion (SOW)**: Mean LOO R² > 0.0

**Why this matters**: If atmospheric features have zero or negative correlation with CBH, the hybrid model (WP-4) cannot work. This is the critical GO/NO-GO gate.

### Before WP-3: Fix WP-1 Geometric Features
**Problem**: WP-1 produced physically impossible CBH values (negative, enormous).

**Action required**:
1. Run diagnostics:
   ```bash
   python sow_outputs/diagnose_wp1.py \
       --config configs/bestComboConfig.yaml \
       --samples 0 50 100 200 400
   ```
2. Inspect outputs in `sow_outputs/wp1_diagnostics/`
3. Fix scale factors, shadow detection, or geometry model
4. Re-run WP-1 to produce valid geometric features
5. Re-run WP-3 with corrected WP-1 + WP-2 features

### After WP-3 Passes: WP-4 Hybrid Model
If WP-3 achieves R² > 0:
```bash
python sow_outputs/wp4_hybrid_model.py \
    --wp1-features sow_outputs/wp1_geometric/WP1_Features.hdf5 \
    --wp2-features sow_outputs/wp2_atmospheric/WP2_Features.hdf5 \
    --spatial-mae \
    --verbose
```

---

## Files Created/Modified

### New Files
- `sow_outputs/wp2_atmospheric/WP2_Features.hdf5` (46 KB) - **Primary deliverable**
- `/media/rylan/two/research/NASA/ERA5_data_root/surface/*.nc` (120 files, 123 MB)
- `/media/rylan/two/research/NASA/ERA5_data_root/pressure_levels/*.nc` (120 files, 922 MB)
- `/media/rylan/two/research/NASA/ERA5_data_root/processed/sample_metadata.json`

### Modified Files
- `sow_outputs/wp2_era5_real.py` (lines 730, 751) - Fixed `time` → `valid_time`

### Documentation
- `sow_outputs/WP2_ERA5_SETUP.md` - Complete setup guide
- `sow_outputs/WP2_COMPLETE.md` - This file
- `sow_outputs/QUICK_START_ERA5.txt` - Quick reference

---

## Lessons Learned

1. **Always validate dimension names**: NetCDF files from different sources use inconsistent naming (`time` vs `valid_time`, `lat` vs `latitude`, etc.). Always inspect with `ncdump` or `xarray` before assuming.

2. **Test extraction on small subset first**: If I had tested on 10 samples before running all 933, the bug would have been caught in seconds, not after 5 hours.

3. **Resumable downloads are essential**: The 5-hour download succeeded because the script could resume. Always implement checkpointing for long operations.

4. **CDS licensing is manual**: Cannot automate license acceptance. Users must visit the website and click "Accept" for each dataset.

5. **Separate download from processing**: By saving raw NetCDF files, feature extraction can be re-run instantly when bugs are found (as demonstrated by the 2-minute fix).

---

## Resources

- **ERA5 Documentation**: https://confluence.ecmwf.int/display/CKB/ERA5
- **CDS API Guide**: https://cds.climate.copernicus.eu/api-how-to
- **xarray NetCDF Tutorial**: http://xarray.pydata.org/en/stable/user-guide/io.html

---

## Conclusion

WP-2 is **COMPLETE** with real ERA5 atmospheric features. The pipeline is:
- ✅ Production-ready (resumable, error-handled, documented)
- ✅ Validated (933/933 samples, 0 NaN, physically reasonable values)
- ✅ Reproducible (all code, configs, and data paths documented)

**Critical path**: Fix WP-1 geometric features, then proceed to WP-3 validation. Do not attempt WP-3 until WP-1 diagnostics are reviewed and corrected.