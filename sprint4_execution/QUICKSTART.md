# Sprint 4 Quick Start Guide
## Get Running in 5 Minutes

**Last Updated:** 2025-02-19  
**Purpose:** Fast-track execution of Sprint 4 negative results analysis

---

## TL;DR - Execute Now

```bash
# Navigate to repository
cd ~/Documents/research/NASA/programDirectory/cloudMLPublic

# Install dependencies (choose one method)
pip install h5py numpy scipy matplotlib  # Minimal
# OR
pip install -r sprint4_execution/requirements.txt  # Complete

# Run diagnostic analyses
python3 sprint4_execution/validate_era5_constraints.py
python3 sprint4_execution/shadow_failure_analysis.py

# View results
ls -lh sprint4_execution/figures/
```

**Expected outputs:**
- `figures/era5_constraint_validation.png` - Does BLH > CBH hold?
- `figures/shadow_geometry_failure.png` - Why shadow CBH failed (r=0.04, bias=+5km)

---

## What You're Doing

You're converting the **catastrophic failure** of WP-3 physics baseline (R² = -14.15) into a **publishable methodological paper** documenting:

1. Why shadow geometry from nadir imagery doesn't work
2. Why ERA5 25km resolution can't constrain cloud-scale CBH
3. How to detect such failures early
4. What to try instead

**Target journal:** Atmospheric Measurement Techniques (AMT)  
**Timeline:** 2 weeks to manuscript submission  
**Status:** Week 1 - diagnostic analysis phase

---

## Quick Data Check

```bash
# Verify WP1-WP4 outputs exist
python3 sprint4_execution/inspect_data.py
```

**Should show:**
- ✓ WP1 Geometric Features (81.7 KB)
- ✓ WP2 Atmospheric Features (45.9 KB)
- ✓ WP3 Baseline Report (JSON readable)
- Mean R² = -14.15 (confirmed failure)

---

## File Roadmap

```
sprint4_execution/
├── QUICKSTART.md           ← YOU ARE HERE
├── README.md               ← Full overview
├── gap_analysis.md         ← Sprint 4 plan vs reality
├── action_plan.md          ← Detailed 2-week schedule
├── SPRINT4_KICKOFF.md      ← Comprehensive context
│
├── requirements.txt        ← Python dependencies
├── inspect_data.py         ← Quick data check (no deps)
│
├── validate_era5_constraints.py  ← Figure 2: BLH/LCL vs CBH
├── shadow_failure_analysis.py    ← Figure 1: Shadow geometry failure
├── visualize_loo_cv.py           ← TODO: Figure 3
├── spatial_scale_schematic.py    ← TODO: Figure 4
│
└── figures/                ← Output directory
    ├── era5_constraint_validation.png
    ├── shadow_geometry_failure.png
    └── (more to come...)
```

---

## 2-Week Plan at a Glance

### Week 1: Generate Figures
- **Days 1-2:** Run existing scripts → Figures 1-2
- **Days 3-4:** Create new scripts → Figures 3-4
- **Day 5:** Optional image examples → Figure 5

### Week 2: Write Paper
- **Days 6-7:** Introduction + Methods
- **Days 8-9:** Results (4 subsections)
- **Days 10-11:** Discussion + Conclusion
- **Day 12:** Polish + Submit to AMT

---

## Key Numbers to Remember

From WP1-WP4 execution (Nov 4-5, 2025):

**Shadow Geometry (WP-1):**
- Correlation: r = 0.04 (essentially zero)
- Bias: +5.11 km
- MAE: 5.12 km

**ERA5 Features (WP-2):**
- 933 samples, 9 features
- Resolution: 25 km grid
- Question: Does BLH > CBH?

**Physics Baseline (WP-3):**
- Mean LOO R²: -14.15 ± 24.30
- All 5 folds: NEGATIVE R²
- Fold 4: R² = -62.66 (catastrophic)

**SOW Decision:** HALT at WP-3 ✓ (correct)

---

## Next Steps (After Figure Generation)

1. **Read** gap_analysis.md for full context
2. **Review** action_plan.md for detailed tasks
3. **Execute** Week 2 writing plan
4. **Submit** to Atmospheric Measurement Techniques

---

## Questions?

- **What failed?** Shadow geometry + ERA5 for cross-flight CBH retrieval
- **Why did it fail?** Nadir ambiguity, 25km resolution too coarse, no generalizable signal
- **Is this bad?** No! Negative results are valuable science
- **What's the paper?** "Why shadow geometry and ERA5 fail for CBH retrieval"
- **Will it publish?** Yes - AMT welcomes methodological/negative results

---

## One-Liner Summary

> We tried physics-constrained ML for cloud base height retrieval. It failed spectacularly (R² = -14). Now we're documenting WHY it failed so others don't waste time on the same approach.

---

**Ready to start?** Run the first analysis script:

```bash
python3 sprint4_execution/validate_era5_constraints.py
```

🚀 **Let's turn failure into science!**