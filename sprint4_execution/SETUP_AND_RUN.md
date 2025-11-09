# Quick Setup and Run Guide
## Install h5py and Run Diagnostic Scripts

**Date:** 2025-02-19  
**Purpose:** Install dependencies and investigate WP-3 results

---

## Option 1: Quick Install (Recommended)

```bash
# Navigate to project root
cd ~/Documents/research/NASA/programDirectory/cloudMLPublic

# Install h5py system-wide (requires sudo)
sudo apt-get update
sudo apt-get install -y python3-h5py python3-scipy

# Verify installation
python3 -c "import h5py; print('h5py installed:', h5py.__version__)"
```

---

## Option 2: Virtual Environment (If you don't have sudo)

```bash
# Navigate to project root
cd ~/Documents/research/NASA/programDirectory/cloudMLPublic

# Create virtual environment
python3 -m venv venv_sprint4

# Activate it
source venv_sprint4/bin/activate

# Install dependencies
pip install h5py numpy scipy matplotlib

# Verify
python -c "import h5py; print('h5py installed:', h5py.__version__)"
```

**Note:** Remember to activate the venv each time:
```bash
source ~/Documents/research/NASA/programDirectory/cloudMLPublic/venv_sprint4/bin/activate
```

---

## Option 3: Use --break-system-packages (Not recommended but works)

```bash
pip3 install --user --break-system-packages h5py scipy matplotlib
```

---

## Run the Diagnostic Scripts

Once h5py is installed, run:

```bash
cd ~/Documents/research/NASA/programDirectory/cloudMLPublic

# Investigation 1: Imputation bug (CRITICAL)
python3 sprint4_execution/investigate_imputation_bug.py

# Investigation 2: ERA5 constraints validation
python3 sprint4_execution/validate_era5_constraints.py

# Investigation 3: Shadow geometry failure
python3 sprint4_execution/shadow_failure_analysis.py
```

---

## What Each Script Does

### 1. `investigate_imputation_bug.py` ⚠️ CRITICAL
**Purpose:** Find if there's a bug in how NaN values were imputed

**Checks:**
- How many NaN values in geometric CBH feature?
- What value was used to fill them? (Reported: 6.166 km)
- Compare to true mean CBH (Reported: 0.83 km)
- If imputation value >> true mean → MAJOR BUG FOUND
- Per-flight CBH distributions (do they differ?)
- Feature-target correlations (are they really r < 0.1?)
- Fold 4 catastrophe analysis (why R² = -62.66?)

**Expected output:**
```
SMOKING GUN: IMPUTATION BUG FOUND
Imputation value: 6.166 km (median of shadow estimates)
True CBH mean:    0.830 km (from CPL lidar)
Ratio:            7.4× TOO HIGH
```

### 2. `validate_era5_constraints.py`
**Purpose:** Check if ERA5 physical constraints hold

**Checks:**
- Does BLH > CBH? (Boundary layer should contain clouds)
- Correlation between ERA5 features and true CBH
- Are violations common?

**Expected output:**
- Scatter plots showing BLH vs CBH, LCL vs CBH
- Violation rates
- Correlation coefficients

### 3. `shadow_failure_analysis.py`
**Purpose:** Visualize why shadow geometry failed

**Checks:**
- Scatter plot: shadow CBH vs true CBH
- Residual analysis
- Bland-Altman plot

**Expected output:**
- 4-panel diagnostic figure
- Confirmation of r ≈ 0.04, bias ≈ +5 km

---

## Interpreting Results

### If imputation bug is found (ratio > 2×):
```
✓ R² = -14.15 is explained by the bug
✓ Fix: Re-run WP-3 without geometric H feature
✓ Or: Drop NaN samples instead of imputing
✓ Expected: R² improves significantly
```

### If no imputation bug (ratio < 1.5×):
```
✓ R² = -14.15 is real (not a bug)
✓ Features genuinely have no predictive signal
✓ Cross-flight generalization failed legitimately
✓ Sprint 4 negative results paper is appropriate
```

### If per-flight distributions differ significantly:
```
✓ Explains why LOO CV fails
✓ Train on one distribution, test on another
✓ Model predicts training mean, doesn't match test
✓ Result: Negative R²
```

---

## Quick Verification Without Installing Anything

If you can't install h5py, you can still check file sizes and structure:

```bash
cd ~/Documents/research/NASA/programDirectory/cloudMLPublic

# Check file sizes
ls -lh sow_outputs/wp1_geometric/WP1_Features.hdf5
ls -lh sow_outputs/wp2_atmospheric/WP2_Features.hdf5

# Check WP3 JSON (no dependencies needed)
cat sow_outputs/wp3_baseline/WP3_Report.json | python3 -m json.tool

# Look for imputation mention in logs
grep -i "impute\|median\|nan" sow_outputs/wp3_baseline/*.txt
```

---

## Next Steps After Running Diagnostics

### If imputation bug confirmed:
1. Create fix: `sow_outputs/wp3_physical_baseline_FIXED.py`
2. Re-run WP-3 with:
   - Option A: Drop geometric H entirely (use only 9 ERA5 features)
   - Option B: Drop samples with NaN geometric H
   - Option C: Impute with ground truth mean (0.83 km) not shadow median
3. Compare new R² to original R² = -14.15
4. Document the fix

### If no bug (results are legitimate):
1. Proceed with Sprint 4 negative results paper
2. Document why physics features failed
3. Use diagnostic figures in paper
4. Publish methodology contribution

---

## Troubleshooting

### "externally-managed-environment" error
→ Use Option 2 (virtual environment) or Option 3 (--break-system-packages)

### "Permission denied" when installing
→ Use `sudo` or create virtual environment

### "h5py still not found" after install
→ Check Python version: `python3 --version`
→ Try: `python3.12 -m pip install h5py` (adjust version)

### Scripts crash with import errors
→ Install all requirements:
```bash
pip3 install --user h5py numpy scipy matplotlib
```

---

## Contact

If you find the imputation bug, document it in:
- `sprint4_execution/IMPUTATION_BUG_REPORT.md`
- Include screenshots of diagnostic output
- Note what the fix should be

---

**Estimated time:** 5-10 minutes to install and run all diagnostics