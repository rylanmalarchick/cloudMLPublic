# WP-2: Real ERA5 Data Download and Processing

**Status:** ✅ Ready to Execute  
**Data Location:** `/media/rylan/two/research/NASA/ERA5_data_root/`  
**Expected Download:** 30-70 GB  
**Expected Duration:** 2-8 hours (depends on CDS queue)

---

## 🚀 Quick Start

### Prerequisites

1. **Install Required Packages:**
```bash
pip install cdsapi xarray netCDF4 cfgrib
```

2. **Set Up CDS API Credentials:**

- Register at: https://cds.climate.copernicus.eu/user/register
- Login and go to: https://cds.climate.copernicus.eu/user
- Copy your UID and API key
- Create `~/.cdsapirc` with:

```
url: https://cds.climate.copernicus.eu/api
key: YOUR_UID:YOUR_API_KEY
```

Replace `YOUR_UID:YOUR_API_KEY` with your actual credentials (e.g., `12345:abcdef12-3456-7890-abcd-ef1234567890`)

3. **Verify Disk Space:**
```bash
df -h /media/rylan/two/research/NASA/ERA5_data_root/
```

Ensure you have **at least 80 GB free** on that drive.

### Run the Pipeline

```bash
cd cloudMLPublic

# Full download (surface + pressure levels)
./sow_outputs/run_wp2_real_era5.sh

# Or run Python script directly
python sow_outputs/wp2_era5_real.py \
    --config configs/bestComboConfig.yaml \
    --output sow_outputs/wp2_atmospheric/WP2_Features.hdf5 \
    --era5-dir /media/rylan/two/research/NASA/ERA5_data_root \
    --verbose
```

---

## 📊 What Gets Downloaded

### Flight Data Coverage

The 933 labeled samples span:
- **Flight 0:** 30 Oct 2024 (501 samples)
- **Flight 1:** 10 Feb 2025 (163 samples)  
- **Flight 2:** 23 Oct 2024 (101 samples)
- **Flight 3:** 12 Feb 2025 (144 samples)
- **Flight 4:** 18 Feb 2025 (24 samples)

**Date range:** October 23, 2024 - February 18, 2025 (~120 days)

### ERA5 Data Files

#### Surface Data (`surface/`)
**File pattern:** `era5_surface_YYYYMMDD.nc`  
**Size:** ~50-100 MB per day × 120 days = **6-12 GB total**

**Variables:**
- `blh` - Boundary Layer Height (m)
- `t2m` - 2-meter temperature (K)
- `d2m` - 2-meter dewpoint temperature (K)
- `sp` - Surface pressure (Pa)
- `tcwv` - Total column water vapor (kg/m²)

#### Pressure Level Data (`pressure_levels/`)
**File pattern:** `era5_pressure_YYYYMMDD.nc`  
**Size:** ~200-500 MB per day × 120 days = **24-60 GB total**

**Variables:**
- `t` - Temperature (K)
- `q` - Specific humidity (kg/kg)
- `z` - Geopotential (m²/s²)

**Levels:** 1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500 hPa

**Resolution:** 0.25° × 0.25° (~25 km)  
**Temporal resolution:** Hourly

---

## 🔧 Usage Options

### Full Download (Recommended)
```bash
./sow_outputs/run_wp2_real_era5.sh
```

Downloads both surface and pressure level data. Enables all atmospheric features.

### Surface Only (Faster)
```bash
./sow_outputs/run_wp2_real_era5.sh --surface-only
```

Downloads only surface data (~6-12 GB, 30-60 min).  
⚠️ Warning: `inversion_height` and `stability_index` will be NaN.

### Use Existing Data (No Download)
```bash
./sow_outputs/run_wp2_real_era5.sh --skip-download
```

Use if ERA5 data is already downloaded. Only extracts features.

### Force Re-download
```bash
./sow_outputs/run_wp2_real_era5.sh --force-download
```

Re-downloads all files even if they exist.

### Dry Run (Preview)
```bash
./sow_outputs/run_wp2_real_era5.sh --dry-run
```

Shows what would be done without executing.

---

## 📁 Directory Structure

```
/media/rylan/two/research/NASA/ERA5_data_root/
├── surface/
│   ├── era5_surface_20241023.nc
│   ├── era5_surface_20241024.nc
│   ├── ...
│   └── era5_surface_20250218.nc
│
├── pressure_levels/
│   ├── era5_pressure_20241023.nc
│   ├── era5_pressure_20241024.nc
│   ├── ...
│   └── era5_pressure_20250218.nc
│
└── processed/
    └── sample_metadata.json  (navigation data for 933 samples)
```

---

## 🎯 Derived Features

The pipeline extracts/derives **9 atmospheric features** for each sample:

### Direct from ERA5:
1. **BLH** - Boundary Layer Height (km)
2. **t2m** - 2-meter temperature (K)
3. **d2m** - 2-meter dewpoint temperature (K)
4. **sp** - Surface pressure (Pa)
5. **tcwv** - Total column water vapor (kg/m²)

### Derived Thermodynamic Variables:
6. **LCL** - Lifting Condensation Level (km)
   - Formula: `LCL = (T - Td) / 8 K/km`
   
7. **Inversion Height** - Altitude of strongest temperature gradient (km)
   - Found by analyzing vertical temperature profile
   
8. **Moisture Gradient** - Vertical humidity gradient (kg/kg/m)
   - `dq/dz` in the lower troposphere
   
9. **Stability Index** - Mean lapse rate (K/km)
   - Negative = stable, Positive = unstable

---

## 🔍 Verification

### Check Downloaded Data

```bash
# Count surface files
ls -1 /media/rylan/two/research/NASA/ERA5_data_root/surface/ | wc -l

# Count pressure files
ls -1 /media/rylan/two/research/NASA/ERA5_data_root/pressure_levels/ | wc -l

# Check total size
du -sh /media/rylan/two/research/NASA/ERA5_data_root/
```

### Inspect Output Features

```python
import h5py
import numpy as np

# Open the output file
f = h5py.File('sow_outputs/wp2_atmospheric/WP2_Features.hdf5', 'r')

# Check contents
print("Keys:", list(f.keys()))
print("Features shape:", f['features'].shape)  # Should be (933, 9)
print("Feature names:", f['feature_names'][:])

# Check a sample
sample_idx = 0
features = f['features'][sample_idx, :]
names = f['feature_names'][:]

print(f"\nSample {sample_idx} features:")
for name, value in zip(names, features):
    print(f"  {name.decode():20s}: {value:.3f}")

# Check for NaN values
n_nan = np.sum(np.isnan(f['features'][:]))
total = f['features'].size
print(f"\nNaN values: {n_nan}/{total} ({n_nan/total*100:.1f}%)")

f.close()
```

### Validate Navigation Data

```bash
# Check sample metadata
cat /media/rylan/two/research/NASA/ERA5_data_root/processed/sample_metadata.json | head -50
```

---

## ⚠️ Troubleshooting

### CDS API Issues

**Error:** `"Invalid API key"`
- Solution: Check `~/.cdsapirc` credentials
- Verify at: https://cds.climate.copernicus.eu/user

**Error:** `"Request queued"`
- Normal! CDS processes requests in queue
- May take 10-60 minutes per file during peak times
- Script will wait automatically

**Error:** `"Maximum request limit exceeded"`
- CDS has daily limits (check your account)
- Try again tomorrow or split into smaller date ranges

### Download Failures

**Error:** Network timeout or connection reset
- CDS can be unstable - just re-run with `--skip-download`
- Already-downloaded files will be skipped automatically

**Partial download:**
```bash
# Check which files are missing
ls /media/rylan/two/research/NASA/ERA5_data_root/surface/

# Re-run to download only missing files
./sow_outputs/run_wp2_real_era5.sh
```

### Disk Space Issues

**Error:** `"No space left on device"`
- Check space: `df -h /media/rylan/two/`
- Delete old/unused files or use a larger drive
- Try `--surface-only` mode (uses less space)

### Feature Extraction Issues

**Many NaN values in features:**
- Check if ERA5 files actually downloaded
- Verify navigation data was parsed correctly
- Check lat/lon bounds in metadata

---

## 🚦 Pipeline Stages

The script executes in 4 stages:

### Stage 1: Parse Navigation Data
- Reads all 5 navigation HDF files
- Extracts lat/lon/timestamp for 933 samples
- Calculates spatial/temporal bounds for ERA5 download
- **Duration:** 1-2 minutes

### Stage 2: Download ERA5 Surface Data
- Downloads ~120 daily NetCDF files
- One file per day from Oct 23, 2024 to Feb 18, 2025
- **Duration:** 30-90 minutes (depends on CDS queue)

### Stage 3: Download ERA5 Pressure Level Data
- Downloads ~120 daily NetCDF files
- Pressure level data is much larger (13 levels × 3 variables)
- **Duration:** 1-6 hours (depends on CDS queue)
- **Skip this:** Use `--surface-only` flag (faster but fewer features)

### Stage 4: Extract Features
- Opens each ERA5 file
- Interpolates to sample locations
- Derives thermodynamic variables
- **Duration:** 5-10 minutes

---

## 📈 Expected Timeline

**Optimistic (low CDS queue):** 2-3 hours  
**Typical (moderate queue):** 4-6 hours  
**Worst case (high queue):** 8-12 hours

The script can be interrupted and resumed. Already-downloaded files are skipped.

---

## 🔗 Next Steps

After WP-2 completes:

### 1. Run WP-1 Diagnostics
```bash
python sow_outputs/diagnose_wp1.py \
    --config configs/bestComboConfig.yaml \
    --samples 0 50 100 200 400 600 800
```

View diagnostic plots in `sow_outputs/wp1_diagnostics/`

### 2. Run WP-3 (Physical Baseline)
```bash
python sow_outputs/wp3_physical_baseline.py \
    --config configs/bestComboConfig.yaml \
    --wp1-features sow_outputs/wp1_geometric/WP1_Features.hdf5 \
    --wp2-features sow_outputs/wp2_atmospheric/WP2_Features.hdf5 \
    --output sow_outputs/wp3_baseline/WP3_Report.json \
    --verbose
```

This is the **GO/NO-GO gate**: Must achieve R² > 0

### 3. If WP-3 Passes: Run WP-4 (Hybrid Models)
```bash
python sow_outputs/wp4_hybrid_models.py \
    --config configs/bestComboConfig.yaml \
    --wp1-features sow_outputs/wp1_geometric/WP1_Features.hdf5 \
    --wp2-features sow_outputs/wp2_atmospheric/WP2_Features.hdf5 \
    --output sow_outputs/wp4_hybrid/WP4_Report.json \
    --verbose
```

Target: R² > 0.3

---

## 📚 References

- **ERA5 Documentation:** https://confluence.ecmwf.int/display/CKB/ERA5
- **CDS API Guide:** https://cds.climate.copernicus.eu/api-how-to
- **SOW Document:** `../ScopeWorkSprint3.md` Section 4
- **Implementation Guide:** `SOW_IMPLEMENTATION_GUIDE.md` Section 4

---

## 💾 Storage Management

### Clean Up Old Downloads (if needed)

```bash
# Remove all ERA5 data (if you need to start over)
rm -rf /media/rylan/two/research/NASA/ERA5_data_root/surface/*
rm -rf /media/rylan/two/research/NASA/ERA5_data_root/pressure_levels/*

# Keep processed features, delete raw ERA5 (saves ~60 GB)
# Only do this AFTER features are extracted!
rm -rf /media/rylan/two/research/NASA/ERA5_data_root/surface/*
rm -rf /media/rylan/two/research/NASA/ERA5_data_root/pressure_levels/*
# Features HDF5 is only ~1 MB, can copy to project directory
```

### Backup Features Only

```bash
# The features file is tiny, back it up!
cp sow_outputs/wp2_atmospheric/WP2_Features.hdf5 ~/backup/
```

---

## ✅ Success Criteria

- [ ] CDS API credentials configured
- [ ] Disk space verified (80+ GB free)
- [ ] Python packages installed
- [ ] Surface data downloaded (~120 files)
- [ ] Pressure level data downloaded (~120 files)
- [ ] Features extracted (933 samples × 9 features)
- [ ] Output HDF5 created: `WP2_Features.hdf5`
- [ ] <10% NaN values in features

---

**Ready to begin?**

```bash
./sow_outputs/run_wp2_real_era5.sh
```

Monitor progress with `htop` or check logs. The script is resumable - you can Ctrl+C and restart.