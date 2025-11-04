# Stratified Split Results Analysis

**Date:** November 1, 2024  
**Status:** 🚨 Critical Findings - Model Does Not Generalize  
**Priority:** High - Requires Immediate Attention

---

## Executive Summary

After implementing stratified splitting to fix test-set imbalance issues, we re-ran all validation experiments. The results reveal **critical problems** with the current hybrid MAE+GBDT approach:

1. **MAE embeddings provide no value** - They reduce performance compared to angles alone
2. **Random embeddings outperform trained MAE** - SSL pretraining is ineffective
3. **No cross-flight generalization** - LOO CV shows negative R² (-4.46)
4. **Angles dominate prediction** - R² = 0.71, but this doesn't transfer across flights

**Conclusion:** The current approach does not work for cross-flight CBH prediction. Major redesign needed.

---

## Results Summary

### Single Stratified Split (Ablation Study)

| Method | R² | MAE (m) | RMSE (m) | Key Finding |
|--------|-----|---------|----------|-------------|
| **Angles_only** | **0.7061** | 120.2 | 199.3 | 🏆 Best single feature set |
| Angles+HandCrafted | 0.6910 | 140.2 | 204.4 | Hand-crafted adds little |
| Random_MAE+Angles | 0.5752 | 173.3 | 239.7 | Random > Trained MAE! |
| Full_Features | 0.5913 | 160.1 | 235.1 | Kitchen sink approach fails |
| Noise+Angles | 0.5340 | 189.5 | 251.0 | Pure noise baseline |
| MAE+Angles | **0.4879** | 188.2 | 263.1 | ⚠️ MAE hurts performance |
| HandCrafted_only | 0.2909 | 204.8 | 309.7 | Statistical features weak |
| MAE_only | 0.0899 | 257.8 | 350.8 | ❌ MAE has no signal |

### Leave-One-Out Cross-Validation (Per-Flight)

| Test Flight | R² | MAE (m) | RMSE (m) | Samples |
|-------------|-----|---------|----------|---------|
| 10Feb25 | -3.54 | 239.9 | 303.7 | 163 |
| 30Oct24 | -0.02 | 280.7 | 335.5 | 501 |
| 23Oct24 | -0.23 | 442.4 | 497.8 | 101 |
| 18Feb25 | **-18.40** | 422.3 | 448.8 | 24 |
| 12Feb25 | -0.13 | 354.0 | 513.3 | 144 |
| **Average** | **-4.46 ± 7.09** | **347.9 ± 78.3** | **419.8 ± 85.2** | 933 |

**Critical:** Negative R² means the model performs worse than predicting the mean CBH!

---

## Critical Findings

### 1. MAE Embeddings Are Detrimental

**Evidence:**
- Angles_only: R² = 0.71
- MAE+Angles: R² = 0.49
- **Impact:** Adding MAE reduces R² by 31% (from 0.71 to 0.49)

**Possible Causes:**
- MAE learns irrelevant features from unlabeled data
- 1D flattening loses spatial structure critical for CBH
- SSL pretraining objective misaligned with CBH regression
- Embeddings capture illumination/texture, not cloud height

### 2. Random Embeddings Outperform Trained MAE

**Evidence:**
- Trained MAE+Angles: R² = 0.49
- Random MAE+Angles: R² = 0.58
- Random Noise+Angles: R² = 0.53

**Implication:** SSL pretraining is **actively harmful**. Random embeddings provide better regularization than learned ones.

**Interpretation:**
- The MAE encoder learns features that conflict with CBH prediction
- Random embeddings add dimensional complexity that helps GBDT find nonlinear angle patterns
- The improvement from random embeddings suggests the issue is the learned features, not the architecture

### 3. Complete Failure of Cross-Flight Generalization

**Evidence:**
- Single stratified split: R² = 0.49 (MAE+Angles)
- LOO CV: R² = -4.46 ± 7.09
- **Discrepancy:** 5+ R² points!

**Interpretation:**
- The model learns **within-flight patterns**, not generalizable CBH prediction
- Even with stratified splitting, train/test share flight distributions
- True cross-flight prediction is essentially random

**Per-Flight Breakdown:**
- Best fold (30Oct24): R² = -0.02 (barely worse than mean)
- Worst fold (18Feb25): R² = -18.40 (catastrophic failure)
- High variance (±7.09) indicates unstable, unreliable predictions

### 4. Angles Are the Dominant Signal (But Don't Transfer)

**Evidence:**
- Angles_only achieves R² = 0.71 on stratified split
- But LOO CV shows this doesn't generalize across flights
- Correlation analysis showed weak overall correlations (r ≈ -0.04 for SZA)

**Explanation:**
- **Within-flight**: Angles correlate with temporal progression during flight
- **Cross-flight**: Different flights have different angle↔CBH relationships
- The model is learning **flight-specific time-of-day patterns**, not physical CBH↔angle relationships

---

## Why Stratified Splitting Revealed These Issues

### Before: Random Split (Misleading)

**Old Result:** Angles_only R² ≈ 0.83
- Test set dominated by 30Oct24 (52.5%)
- Model learned 30Oct24-specific patterns
- Inflated performance metric

### After: Stratified Split (More Honest)

**New Result:** Angles_only R² = 0.71
- Balanced flight representation in test set
- Still doesn't transfer across flights (LOO R² = negative)
- More realistic but still not true generalization

### Gold Standard: LOO CV (Reality Check)

**LOO Result:** R² = -4.46
- Each flight tested completely unseen
- Reveals complete failure of cross-flight generalization
- The harsh truth

---

## Implications for the Research

### What We Thought We Had

- Self-supervised MAE pretraining learns useful cloud representations
- MAE embeddings + GBDT fusion → strong CBH prediction
- Competitive with or better than supervised approaches

### What We Actually Have

- MAE pretraining learns features **uncorrelated** with CBH
- Adding MAE embeddings **hurts** performance vs angles alone
- Random embeddings work better than trained ones
- No cross-flight generalization whatsoever

### Scientific Conclusions

1. **SSL pretraining is ineffective** for this task/data
2. **1D embedding representation** likely loses critical spatial information
3. **GBDT fusion** may not be the right approach for combining modalities
4. **Angles dominate** but capture flight-specific temporal patterns, not physics
5. **Current approach fundamentally flawed** - cannot predict CBH across flights

---

## Comparison to Previous Results

### Hybrid MAE-GBDT Script (Different Run)

When we ran `hybrid_mae_gbdt.py` earlier, it reported:
- Test R² = 0.96, MAE = 33 m

**Why the discrepancy?**
- Different random seed for train/test split
- Potentially different dataset filtering
- Grid search might have overfit to validation set
- **This underscores the importance of LOO CV** - single splits are unreliable!

### Lesson Learned

**Single-split results are misleading**, even with stratified splitting. You **must** use LOO CV for honest cross-flight assessment.

---

## Root Cause Analysis

### Why Does MAE Hurt Performance?

**Hypothesis 1: Feature Mismatch**
- MAE learns to reconstruct images (pixel-level detail)
- CBH depends on cloud top height (structural/geometric property)
- These objectives are misaligned

**Hypothesis 2: Information Bottleneck**
- Flattening to 1D embedding loses spatial structure
- CBH may require 2D spatial patterns (cloud shape, extent)
- 192-dim vector cannot encode relevant spatial information

**Hypothesis 3: Noise Injection**
- MAE embeddings capture illumination, texture, sensor artifacts
- These are uncorrelated with CBH but confuse the GBDT
- Random embeddings provide "cleaner" noise that GBDT can ignore

### Why Do Angles Work (Within-Flight)?

**Temporal Correlation:**
- During a single flight, time progresses → sun angle changes
- Clouds may evolve or aircraft altitude changes over time
- Model learns: "In flight X at angle Y, CBH tends to be Z"

**This is NOT physical prediction!**
- Different flights have different angle→CBH mappings
- LOO CV proves angles don't transfer across flights
- Just memorizing per-flight time-of-day patterns

---

## Recommendations

### Immediate: Acknowledge Limitations

1. **Do NOT claim** the hybrid approach works for cross-flight CBH prediction
2. **Report LOO CV results** as the primary metric (not single-split)
3. **Acknowledge** MAE embeddings do not help (or actively hurt)
4. **Focus analysis** on why the approach fails

### Short-Term: Diagnostic Experiments

1. **Visualize MAE embeddings**
   - t-SNE/UMAP colored by CBH, flight, SZA, SAA
   - Determine what MAE actually learned

2. **Correlation analysis**
   - Compute embedding→CBH correlations per dimension
   - Identify if any dimensions capture CBH signal

3. **Ablation: 2D representations**
   - Instead of flattening, use 2D feature maps
   - Try spatial attention mechanisms
   - Preserve geometric structure

4. **Test different fusion strategies**
   - MLP instead of GBDT
   - End-to-end fine-tuning (unfreeze MAE encoder)
   - Multi-task learning (reconstruct + predict CBH)

### Medium-Term: Rethink Approach

#### Option A: Improve SSL Pretraining

- **Multi-task pretraining**: Add CBH prediction head during SSL
- **Supervised contrastive learning**: Use CBH as supervision signal
- **Different pretext task**: Predict cloud height from stereo/shadow, not reconstruct

#### Option B: Abandon Embeddings, Focus on Physics

- **Physical models**: Use radiative transfer models to link imagery→CBH
- **Structure-based features**: Cloud shape, texture, spatial extent
- **Multi-modal fusion**: Combine GOES imagery with other sensors (radar, lidar)

#### Option C: End-to-End Supervised Learning

- **Supervised CNN**: Train encoder directly on CBH labels
- **More labels**: Expand CPL-aligned dataset (currently only 933 samples)
- **Transfer learning**: Pretrain on related tasks (cloud classification, etc.)

### Long-Term: Research Direction

**Re-evaluate the fundamental approach:**

1. **Is SSL appropriate for this task?**
   - 933 labeled samples may be too few
   - But 61,946 unlabeled samples may not help if MAE learns wrong features

2. **Is GOES imagery sufficient?**
   - Single-view passive imaging has inherent limitations
   - May need stereo, radar, or multi-spectral for height estimation

3. **Is cross-flight generalization the right goal?**
   - Different flights have different atmospheric conditions
   - Perhaps per-flight models with domain adaptation?

---

## Statistical Validity

### Stratified Split: ✅ Valid for Development

**Pros:**
- Balanced flight representation
- Fast iteration
- Consistent splits

**Cons:**
- Still shares flights between train/test
- Overestimates cross-flight performance
- Single split can be lucky/unlucky

**Use case:** Rapid prototyping, ablation studies

### LOO CV: ✅ Gold Standard for Validation

**Pros:**
- True cross-flight generalization test
- Per-flight metrics
- No data leakage

**Cons:**
- Slower (5 folds)
- High variance with few flights
- Small flights (18Feb25: N=24) unreliable

**Use case:** Final validation, publication results

---

## Revised Workflow

### Development Phase

1. Use **stratified splits** for rapid iteration
2. Track **angles-only baseline** as sanity check
3. If new method doesn't beat angles, it's not working
4. Use LOO CV periodically to check generalization

### Validation Phase

1. **Primary metric**: LOO CV per-flight R², MAE, RMSE
2. **Secondary metric**: Stratified split (for comparison)
3. **Report both**: Show within-flight vs cross-flight performance
4. **Visualizations**: Per-flight scatter plots, residual analysis

---

## Key Metrics to Report

### For Publication

**Primary:**
- LOO CV: R² = -4.46 ± 7.09, MAE = 348 ± 78 m
- Per-flight LOO results (table)

**Comparison:**
- Stratified split: R² = 0.49 (MAE+Angles)
- Angles-only: R² = 0.71 (stratified), negative (LOO)

**Ablations:**
- MAE vs Random vs Noise embeddings
- Feature combinations (angles, hand-crafted, MAE)

**Interpretation:**
- Model learns within-flight patterns, not generalizable prediction
- MAE embeddings do not help (and may hurt)
- Current approach unsuitable for cross-flight deployment

---

## Lessons Learned

### About Data Splitting

1. ✅ **Stratified splitting is essential** - prevents imbalance artifacts
2. ✅ **LOO CV is the gold standard** - reveals true generalization
3. ✅ **Single splits are misleading** - even stratified ones
4. ✅ **Always validate with cross-source holdout** - for multi-source data

### About Model Development

1. ⚠️ **High single-split R² ≠ good model** - must check cross-validation
2. ⚠️ **SSL pretraining can hurt** - if objectives misaligned
3. ⚠️ **More features ≠ better performance** - angles beat full feature set
4. ⚠️ **Baseline is critical** - angles-only baseline revealed MAE adds no value

### About Scientific Rigor

1. 📊 **Report the harsh truth** - negative R² is publishable (with analysis)
2. 📊 **Failure analysis is valuable** - understanding why helps the field
3. 📊 **Ablations are essential** - revealed MAE vs random vs noise differences
4. 📊 **Per-fold metrics matter** - high variance indicates instability

---

## Next Steps

### For Rylan

1. **Decide on research direction:**
   - Option A: Try to fix MAE approach (risky, may not work)
   - Option B: Acknowledge limitations, focus on analysis of failure modes
   - Option C: Pivot to different approach (supervised, physical models, etc.)

2. **Update manuscript:**
   - Remove any claims about cross-flight generalization
   - Report LOO CV results prominently
   - Frame as "lessons learned" or "challenges in SSL for CBH"

3. **Run diagnostic experiments:**
   - Embedding visualization (t-SNE colored by CBH, flight)
   - Per-dimension correlation analysis
   - Investigate why angles work within-flight but not across

4. **Consider alternative approaches:**
   - End-to-end supervised learning
   - Physics-based features
   - Domain adaptation for cross-flight transfer

---

## Conclusion

The stratified splitting implementation **successfully fixed the test-set imbalance issue** and provided more honest performance metrics. However, it also **revealed fundamental problems** with the current approach:

- MAE embeddings provide no value (R² drops from 0.71 to 0.49)
- Random embeddings outperform trained MAE (R² = 0.58 vs 0.49)
- No cross-flight generalization (LOO R² = -4.46)
- Angles dominate but learn flight-specific temporal patterns, not physics

**This is not a failure of the validation methodology** - it's a success! We now have honest metrics that reveal the true state of the model. The failure is in the modeling approach, not the evaluation.

**The path forward** requires either substantial improvements to the MAE+GBDT approach or exploration of alternative methods. Simply tuning hyperparameters or adding more features is unlikely to fix the fundamental issues revealed by these experiments.

---

**Status:** Stratified splitting implemented and validated ✅  
**Model performance:** Does not generalize across flights ❌  
**Recommendation:** Major redesign or alternative approach needed 🔄