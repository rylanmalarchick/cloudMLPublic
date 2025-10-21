# CloudML Diagnostics

**Purpose:** Determine if cloud optical depth prediction is fundamentally learnable from single-wavelength IR images + sun angles.

**Status:** Ready to run  
**Priority:** HIGH - Run these BEFORE more neural network experiments

---

## Why Run Diagnostics?

After 4 training runs, all neural networks achieved **negative R²** (worse than predicting the mean). We need to know:

1. **Is the task learnable at all?** (Correlation analysis)
2. **Can simple models work?** (Baseline comparison)
3. **Is our architecture the problem?** (Ablation study)

**Time investment:** 2-3 hours to get clear answers vs weeks of guessing.

---

## Scripts Overview

### 1. Correlation Analysis (`1_correlation_analysis.py`)
**Runtime:** ~30 minutes  
**Goal:** Check if ANY feature correlates with optical depth

**What it does:**
- Extracts 28 hand-crafted features (intensity stats, gradients, metadata)
- Computes Pearson & Spearman correlations with target
- Identifies strongest predictors

**Success criteria:**
- If max r² > 0.1 → Signal exists, proceed to step 2
- If max r² = 0.05-0.1 → Weak signal, proceed with caution
- If max r² < 0.05 → Task likely not learnable from these features

**Output:**
- `results/correlation_results.csv` - Full correlation table
- `results/correlation_summary.json` - Key findings

---

### 2. Simple Baselines (`2_simple_baselines.py`)
**Runtime:** ~1 hour  
**Goal:** Can simple models beat mean baseline (R² > 0)?

**What it does:**
- Tests 7 models: Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting
- Trains on hand-crafted features
- Evaluates on held-out test set

**Success criteria:**
- If best R² > 0.3 → Strongly learnable, deep learning should work
- If best R² = 0.0-0.3 → Weakly learnable, deep learning might help
- If best R² < 0 → Data doesn't contain learnable signal

**Output:**
- `results/baseline_results.csv` - All model results
- `results/baseline_summary.json` - Best model & decision

**Critical comparison:**
- If simple models get R² > 0 but your neural nets get R² < 0
- → Neural network has training issues (NOT a data problem!)

---

## How to Run

### Option 1: Run All Diagnostics (Recommended)
```bash
cd /path/to/cloudMLPublic

# Run correlation analysis
python diagnostics/1_correlation_analysis.py

# Run simple baselines
python diagnostics/2_simple_baselines.py
```

### Option 2: Run in Colab
```python
# In Colab notebook
%cd /content/repo

# Correlation analysis
!python diagnostics/1_correlation_analysis.py

# Simple baselines (wait for D1 results first)
!python diagnostics/2_simple_baselines.py
```

---

## Expected Results & Decisions

### Scenario A: Task Not Learnable
**Evidence:**
- Correlation r² < 0.05
- All baselines R² < 0

**Decision:** STOP neural network experiments  
**Next steps:**
- Consult atmospheric scientists
- Need multi-wavelength IR (CO2 absorption bands)
- Consider different target (cloud type vs optical depth)

---

### Scenario B: Weak Signal
**Evidence:**
- Correlation r² = 0.05-0.1
- Best baseline R² = 0.0-0.1

**Decision:** Proceed with LOW expectations  
**Next steps:**
- Run 5 might achieve R² = 0.05-0.15
- Consider if this performance is useful
- May need different approach

---

### Scenario C: Moderate Signal
**Evidence:**
- Correlation r² > 0.1
- Best baseline R² = 0.1-0.3

**Decision:** Deep learning should work!  
**Next steps:**
- Run 5 should achieve R² = 0.2-0.4
- Your previous failures were training issues
- Variance collapse + initialization were the problem

---

## Literature Context

From the papers you provided:

1. **Himawari-8 Cloud Top Height (Yu et al.)**
   - Used 16 channels (visible + IR + near-IR)
   - ConvLSTM for spatiotemporal prediction
   - Success required multi-spectral data

2. **VIIRS/CrIS Cloud Height (Heidinger et al.)**
   - Single-wavelength IR **struggles with thin clouds**
   - Needed CO2 absorption bands (13.3-14.2 µm) for accuracy
   - Quote: "imagers [...] lack the IR channels in H2O and CO2 absorption bands needed for accurate thin ice cloud height estimation"

3. **Your Task:**
   - Single-wavelength IR only (no CO2 bands)
   - Predict optical depth (related to cloud thickness)
   - **This is harder than what the papers solved!**

**Key insight:** Literature shows single-wavelength IR has limited information content for cloud properties. Multi-spectral is typically required.

---

## What If Diagnostics Show "Not Learnable"?

Don't panic! This is valuable information. It tells you:

1. **It's not your fault** - The features don't contain enough information
2. **You need different data** - Multi-wavelength, polarization, terrain, etc.
3. **Reformulate the problem** - Classification (low/med/high) instead of regression?

**This is still a successful internship!** Scientific research means:
- Testing hypotheses
- Identifying limitations
- Recommending what's needed

You've systematically diagnosed the problem and can now make informed recommendations.

---

## Files Generated

```
diagnostics/
 README.md                        (this file)
 1_correlation_analysis.py        (30 min runtime)
 2_simple_baselines.py           (1 hour runtime)
 results/
     correlation_results.csv      (all features)
     correlation_summary.json     (top findings)
     baseline_results.csv         (all models)
     baseline_summary.json        (best model)
```

---

## Quick Reference

**Before running:**
- Ensure data is available in Drive/CloudML/data/
- Config: `configs/colab_optimized_full_tuned.yaml`
- Test set used for unbiased evaluation

**After running:**
1. Check `correlation_summary.json` for max r²
2. Check `baseline_summary.json` for best model R²
3. Compare with your neural network results (Run 1-4)
4. Make informed decision on next steps

---

**Questions?** Check output files or review DIAGNOSTIC_PLAN.md for full details.

**Ready to proceed?** Start with `1_correlation_analysis.py` now!