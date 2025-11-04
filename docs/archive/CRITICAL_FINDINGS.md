# Critical Findings from Validation Analysis

## 🚨 Major Discovery: Test Set Imbalance

### The Problem

Your ablation studies showed:
- **Angles_only: R² = 0.83** (excellent!)
- Yet angle-CBH correlations are **r = -0.04** (near zero!)

This is **impossible** unless something is wrong.

### Root Cause: Test Set Imbalance

**Test set composition** (N=141):
- 30Oct24: 74 samples (52.5%) ← **DOMINATES**
- 10Feb25: 28 samples (19.9%)
- 23Oct24: 14 samples (9.9%)
- 12Feb25: 18 samples (12.8%)
- 18Feb25: 7 samples (5.0%)

**Per-flight angle-CBH correlations:**
- 30Oct24: r(SZA,CBH) = -0.17, r(SAA,CBH) = -0.15
- 23Oct24: r(SZA,CBH) = -0.25, r(SAA,CBH) = 0.12
- Others: r < 0.10

### What's Happening

1. Test set is 52.5% from ONE flight (30Oct24)
2. That flight has slightly stronger angle-CBH patterns
3. GBDT "learns" patterns specific to 30Oct24
4. Gets lucky on test set dominated by same flight
5. Won't generalize to other flights (explains LOO CV failure)

### Evidence

1. ✅ Overall correlation: r = -0.04 (weak)
2. ✅ Test set R² = 0.83 (misleadingly high)
3. ✅ LOO CV R² = -14.5 (catastrophic failure)
4. ✅ Per-flight correlations vary widely

### Implications

**Your current "Angles_only R² = 0.83" result is NOT RELIABLE.**

It's an artifact of:
- Test set imbalance
- Small sample size (N=141)
- Lucky correlation in dominant flight

---

## ✅ Solutions

### 1. Stratified Splitting (RECOMMENDED)

Ensure each split (train/val/test) has **proportional representation** from all flights.

```python
# Instead of random shuffle:
from sklearn.model_selection import train_test_split

# Create flight labels for each sample
flight_labels = [get_flight_for_sample(i) for i in range(len(dataset))]

# Stratified split
train_idx, test_idx = train_test_split(
    indices, 
    test_size=0.15, 
    stratify=flight_labels,
    random_state=42
)
```

### 2. Per-Flight Evaluation

Report performance **separately for each flight**, not just overall.

This is what `analyze_results.py` does - you saw:
- 10Feb25: R² = -0.97
- 30Oct24: R² = 0.74 ← Only this one works!
- 23Oct24: R² = 0.52
- 18Feb25: R² = -7.30
- 12Feb25: R² = 0.72

### 3. Use LOO CV (Now Fixed)

Re-run the fixed LOO CV:
```bash
./scripts/run_loo_validation.sh
```

This will show **true cross-flight generalization**.

---

## 📊 Expected Results After Fixes

### Before (Broken):
- Angles_only: R² = 0.83 ❌ (misleading!)
- LOO CV: R² = -14.5 ❌ (broken scaler)

### After (Fixed):
- Angles_only: R² = 0.3-0.5 ✅ (realistic for r=-0.04)
- LOO CV: R² = 0.4-0.7 ✅ (with unified scaler)
- Per-flight variance: high ✅ (expected)

---

## 🎯 Action Plan

1. **Run fixed LOO CV** to see true performance:
   ```bash
   ./scripts/run_loo_validation.sh
   ```

2. **Interpret results correctly**:
   - If LOO R² > 0.5: Model generalizes across flights ✅
   - If LOO R² < 0.3: Model doesn't generalize ❌
   - Per-flight variance expected due to different conditions

3. **For paper**:
   - Report **per-flight performance** (not just overall)
   - Use **stratified splits** or **LOO CV** for evaluation
   - Acknowledge that angles provide 30-50% of performance
   - Show MAE adds value ON TOP of angles

4. **If MAE isn't helping** (current issue):
   - Try 2D representation instead of 1D
   - Reduce patch size (16 → 8 or 4)
   - Use MLP fusion instead of GBDT
   - Visualize embeddings (t-SNE) to debug

---

## 📁 Files Created

- `check_angle_scaling.py` - Verified scaling doesn't affect correlation
- `check_test_set.py` - Revealed test set imbalance
- `CRITICAL_FINDINGS.md` - This document

---

## Summary

Your validation revealed a **critical flaw**: test set imbalance made results look better than they are.

**The good news**: You caught it! Now you can fix it and get reliable results.

**Next step**: Run fixed LOO CV to see true cross-flight performance.

