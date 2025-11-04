# Executive Summary: Stratified Splitting Implementation & Critical Findings

**Date:** November 1, 2024  
**To:** Rylan  
**Re:** Stratified splitting implementation complete + critical model performance issues discovered

---

## TL;DR

✅ **Implementation Complete:** Stratified splitting now ensures balanced flight representation across train/val/test splits.

❌ **Critical Finding:** The hybrid MAE+GBDT model **does not generalize across flights**. LOO CV shows R² = -4.46 (negative = worse than predicting the mean).

🎯 **Action Required:** Decide whether to fix, pivot, or reframe the research direction.

---

## What Was Implemented

### Stratified Splitting System

**New Module:** `src/split_utils.py`
- `stratified_split_by_flight()` - Main splitting function
- `check_split_leakage()` - Validation utility
- `analyze_split_balance()` - Diagnostic tool

**Scripts Updated (all use stratified splits now):**
- ✅ `scripts/hybrid_mae_gbdt.py`
- ✅ `scripts/ablation_studies.py`
- ✅ `scripts/analyze_results.py`
- ✅ `scripts/finetune_cbh.py`
- ✅ `check_test_set.py`

**Documentation Created:**
- `docs/STRATIFIED_SPLITTING.md` - Full technical docs
- `docs/STRATIFIED_SPLIT_SUMMARY.md` - Detailed summary
- `STRATIFIED_SPLIT_QUICKSTART.md` - Quick reference
- `docs/STRATIFIED_RESULTS_ANALYSIS.md` - Results analysis
- `CHANGELOG_STRATIFIED_SPLIT.md` - Implementation log
- `scripts/compare_splits.py` - Comparison tool

### Problem Fixed

**Before:** Random splitting allowed test sets to be dominated by single flights
- 30Oct24 had 52.5% of test samples
- Led to inflated performance metrics (angles-only R² ≈ 0.83)

**After:** Stratified splitting ensures proportional representation
- Each flight contributes to splits proportionally to its dataset size
- More honest metrics (angles-only R² = 0.71 on stratified split)

---

## Critical Findings from Results

### 1. MAE Embeddings Are Harmful

| Method | R² | MAE (m) | Conclusion |
|--------|-----|---------|------------|
| **Angles_only** | **0.71** | 120 | 🏆 Best performer |
| MAE+Angles | 0.49 | 188 | ⚠️ MAE hurts performance |
| Random_MAE+Angles | 0.58 | 173 | 🚨 Random > Trained! |
| MAE_only | 0.09 | 258 | ❌ No signal |

**Interpretation:** Adding trained MAE embeddings **reduces R² from 0.71 to 0.49** (31% drop). Random embeddings actually perform better than trained ones.

### 2. No Cross-Flight Generalization

**Stratified Single Split:**
- MAE+Angles: R² = 0.49, MAE = 188 m

**LOO Cross-Validation (Testing on Completely Unseen Flights):**
- MAE+Angles: R² = **-4.46 ± 7.09**, MAE = 348 m
- Negative R² = worse than predicting the mean
- Worst fold (18Feb25): R² = -18.4

**Conclusion:** The model learns within-flight patterns, not generalizable CBH prediction.

### 3. Why Angles Appear to Work

**Within-flight (stratified split):** R² = 0.71  
**Cross-flight (LOO CV):** R² = negative

**Explanation:** 
- During a single flight, time progresses → angles change → clouds evolve
- Model learns: "In flight X at angle Y, CBH ≈ Z"
- This is **temporal correlation**, not physical prediction
- Different flights have different angle↔CBH mappings
- Doesn't transfer across flights

---

## What This Means for Your Research

### What You Thought You Had

- SSL MAE pretraining learns useful cloud representations
- MAE embeddings + GBDT → strong CBH prediction
- Competitive or better than supervised approaches

### What You Actually Have

- MAE learns features **uncorrelated** with CBH
- Adding MAE **hurts** performance vs. angles alone
- Random embeddings work better than trained ones
- **Zero cross-flight generalization**
- Current approach fundamentally flawed

### Scientific Value

This is **publishable** as:
- Analysis of why SSL fails for this task
- Lessons learned in multi-source CBH prediction
- Importance of proper validation (LOO CV vs single split)
- Challenges in cross-flight generalization

**NOT publishable as:**
- A working CBH prediction system
- Evidence that MAE+GBDT works for this task

---

## Decision Points

You have three options:

### Option A: Try to Fix the Approach (Risky)

**Potential fixes:**
- End-to-end fine-tuning (unfreeze MAE encoder)
- 2D spatial features instead of 1D flattened embeddings
- Different fusion strategy (MLP instead of GBDT)
- Multi-task SSL (reconstruct + predict CBH simultaneously)
- Supervised contrastive learning with CBH labels

**Pros:** Could salvage the hybrid approach  
**Cons:** May not work; already spent significant effort  
**Timeline:** 2-4 weeks of experiments

### Option B: Pivot to Different Approach (Safe)

**Alternative approaches:**
- Pure supervised CNN (train on 933 labeled samples)
- Transfer learning from ImageNet/cloud classification
- Physics-based features (radiative transfer models)
- Multi-modal fusion (combine GOES + other sensors)

**Pros:** Fresh start with known methods  
**Cons:** Abandons SSL work already done  
**Timeline:** 3-6 weeks to implement and validate

### Option C: Reframe as Failure Analysis (Pragmatic)

**Focus on:**
- Why SSL fails for this task
- Importance of LOO CV for multi-source data
- Analysis of what MAE actually learns
- Recommendations for future work

**Pros:** Publishable now; valuable to community  
**Cons:** Not a "positive" result  
**Timeline:** 1-2 weeks to write up

---

## Immediate Action Items

### Must Do (This Week)

1. **Run embedding visualization**
   ```bash
   # Create t-SNE/UMAP plots colored by CBH, flight, SZA
   # Determine what MAE actually learned
   ```

2. **Analyze per-dimension correlations**
   ```python
   # For each of 192 MAE dimensions:
   # - Correlation with CBH
   # - Correlation with flight ID
   # - Correlation with angles
   ```

3. **Decide on research direction**
   - Option A (fix), B (pivot), or C (reframe)?
   - Discuss with advisor
   - Update research plan accordingly

### Should Do (Next Week)

4. **If pursuing Option A (fix):**
   - Implement end-to-end fine-tuning
   - Try 2D spatial representations
   - Test alternative fusion strategies

5. **If pursuing Option B (pivot):**
   - Implement supervised baseline CNN
   - Explore transfer learning options
   - Investigate multi-modal approaches

6. **If pursuing Option C (reframe):**
   - Write failure analysis section
   - Create comprehensive ablation figures
   - Draft lessons-learned document

### Documentation Updates

7. **Update any existing drafts/presentations:**
   - Remove claims about cross-flight generalization
   - Report LOO CV as primary metric
   - Include stratified split as secondary
   - Show per-flight performance breakdown

8. **Prepare figures for publication:**
   - LOO CV results with error bars
   - Ablation comparison bar chart
   - Per-flight scatter plots (predicted vs. actual)
   - Embedding visualization (t-SNE/UMAP)

---

## Key Numbers to Remember

### Single Stratified Split (Development Metric)
- Angles_only: R² = 0.71, MAE = 120 m
- MAE+Angles: R² = 0.49, MAE = 188 m
- Random_MAE+Angles: R² = 0.58, MAE = 173 m

### LOO Cross-Validation (Publication Metric)
- MAE+Angles: R² = -4.46 ± 7.09, MAE = 348 ± 78 m
- Per-flight range: R² from -18.4 to -0.02

### Dataset Composition
- Total labeled samples: 933
- Flights: 5 (10Feb25, 30Oct24, 23Oct24, 18Feb25, 12Feb25)
- Dominant flight: 30Oct24 (53.7% of data)
- Stratified split: 70% train, 15% val, 15% test

---

## Commands Reference

```bash
# Check current split balance
./venv/bin/python check_test_set.py

# Compare old vs new splitting
./venv/bin/python scripts/compare_splits.py

# Run ablations (uses stratified splits)
./scripts/run_ablation_studies.sh

# Run LOO validation (gold standard)
./scripts/run_loo_validation.sh

# Run full hybrid pipeline
./venv/bin/python scripts/hybrid_mae_gbdt.py \
    --config configs/ssl_finetune_cbh.yaml \
    --encoder outputs/mae_pretrain/mae_encoder_pretrained.pth
```

---

## Documentation Structure

```
cloudMLPublic/
├── EXECUTIVE_SUMMARY.md (this file)
├── STRATIFIED_SPLIT_QUICKSTART.md (quick reference)
├── CHANGELOG_STRATIFIED_SPLIT.md (implementation details)
├── docs/
│   ├── STRATIFIED_SPLITTING.md (full technical docs)
│   ├── STRATIFIED_SPLIT_SUMMARY.md (detailed summary)
│   └── STRATIFIED_RESULTS_ANALYSIS.md (results + interpretation)
├── src/
│   └── split_utils.py (implementation)
└── scripts/
    └── compare_splits.py (old vs new comparison)
```

---

## Bottom Line

### The Good News

✅ Stratified splitting is implemented and working perfectly  
✅ All scripts updated and validated  
✅ Comprehensive documentation created  
✅ Discovered issues before publication (scientific integrity intact)  
✅ LOO CV provides honest assessment of generalization

### The Bad News

❌ Model does not generalize across flights at all  
❌ MAE embeddings provide no value (actually hurt performance)  
❌ Random embeddings outperform trained ones  
❌ Current approach unsuitable for deployment  
❌ Major rework or pivot needed

### The Path Forward

You need to make a strategic decision:
- **Fix:** Try to salvage hybrid approach (risky, time-consuming)
- **Pivot:** Switch to different approach (safe, fresh start)
- **Reframe:** Publish as failure analysis (pragmatic, quick)

**My recommendation:** Start with Option C (reframe) while exploring Option A (fix) in parallel. If fixes don't work quickly (1-2 weeks), pivot to Option B.

---

## Questions to Discuss with Advisor

1. Is negative LOO CV R² publishable in your field?
2. How important is cross-flight generalization vs. within-flight performance?
3. Should we focus on why MAE fails or try alternative approaches?
4. What's the publication timeline? (affects whether we have time to pivot)
5. Are there domain-specific insights that might help fix the approach?

---

## Final Thoughts

The stratified splitting implementation was a **success** - it revealed the truth about model performance. The model failure is **not a failure of validation**, it's a failure of the modeling approach.

Good science requires honest metrics. You now have them. The next step is deciding how to proceed with this knowledge.

I'm ready to help with whichever direction you choose.

---

**Status:** Implementation complete ✅  
**Model status:** Does not generalize ❌  
**Next steps:** Your decision 🎯  
**Timeline:** Pending discussion 📅