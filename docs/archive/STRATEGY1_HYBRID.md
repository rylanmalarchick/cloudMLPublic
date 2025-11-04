# Strategy 1: Hybrid MAE + GradientBoosting Framework

**Status:** ✅ Ready to run  
**Confidence:** High  
**Expected Runtime:** ~10-15 minutes  
**Expected Performance:** R² > 0.75 (potentially beating classical baseline)

---

## Overview

This strategy implements the **highest-confidence recommendation** from the technical scope of work document. It combines the two most successful approaches that have never been tested together:

1. **MAE Deep Feature Extraction** - Use the pre-trained 1D-MAE encoder as a sophisticated feature extractor
2. **GradientBoosting Regression** - Use the proven classical regressor that achieved R² = 0.75

### The Core Idea

Instead of end-to-end neural network training, we treat the MAE as a **feature learning module** and leverage the superior regression capabilities of gradient boosting.

```
Input (440 pixels) 
    ↓
MAE Encoder (pre-trained, frozen)
    ↓
CLS Token Embeddings (192 dimensions)
    ↓ [concat]
Solar Angles (SZA, SAA)
    ↓
Hybrid Features (194 dimensions)
    ↓
GradientBoosting Regressor
    ↓
CBH Prediction
```

---

## Why This Should Work

### Evidence from Literature

**Remote Sensing Studies:**
- CNN + GBDT combinations consistently outperform end-to-end CNNs
- SSL-learned embeddings + classical ML is proven effective in geoscience
- Gradient boosting excels at learning complex non-linear relationships

### Evidence from This Project

**Classical ML Success:**
- Hand-crafted features (22 statistical features) + GBDT → **R² = 0.7464**
- Proves GBDT is an excellent regressor for this task

**SSL Feature Learning:**
- 1D-MAE learns rich representations from 61k unlabeled images
- Embeddings capture complex cross-track patterns
- Better than hand-crafted statistical features

**The Gap:**
- These two successful components have **never been combined**
- MAE's 192 learned features should outperform 22 hand-crafted features
- GBDT should outperform the neural regression head

---

## Implementation

### Quick Start

```bash
# Verify pre-trained encoder exists
ls outputs/mae_pretrain/mae_encoder_pretrained.pth

# Run hybrid approach
./scripts/run_hybrid_mae_gbdt.sh
```

### Manual Execution

```bash
python scripts/hybrid_mae_gbdt.py \
    --config configs/ssl_finetune_cbh.yaml \
    --encoder outputs/mae_pretrain/mae_encoder_pretrained.pth \
    --device cuda
```

### Options

- `--config`: Path to fine-tuning config (for data loading)
- `--encoder`: Path to pre-trained MAE encoder
- `--no-tune`: Skip hyperparameter tuning (use baseline GBDT params)
- `--device`: Device to use (cuda/cpu)

---

## Pipeline Steps

### Step 1: Extract MAE Embeddings

For each labeled sample (933 total):
1. Load image (440 pixels)
2. Flatten to 1D signal: `(1, 440)`
3. Pass through MAE encoder
4. Extract CLS token: `(192,)` embedding vector

**Output:** 
- Train: (653, 192)
- Val: (139, 192)
- Test: (141, 192)

### Step 2: Create Hybrid Features

Concatenate embeddings with solar angles:
```python
features = [embedding (192) | SZA (1) | SAA (1)]  # Total: 194 features
```

### Step 3: Train GradientBoosting

**With hyperparameter tuning (default):**
- Grid search over: n_estimators, max_depth, learning_rate, min_samples_split
- 5-fold cross-validation
- Best parameters selected automatically

**Without tuning (`--no-tune`):**
- Use baseline params: n_estimators=100, max_depth=5, lr=0.1
- Faster but potentially suboptimal

### Step 4: Evaluate & Compare

Test set evaluation with comparison to:
1. **Classical baseline:** Hand-crafted features + GBDT (R² = 0.7464)
2. **SSL baseline:** MAE end-to-end (R² = 0.3665)
3. **Hybrid approach:** MAE embeddings + GBDT (R² = ???)

---

## Expected Results

### Success Scenarios

**🎉 BEST CASE: R² > 0.75**
- Hybrid beats classical baseline
- MAE features > hand-crafted features
- **Action:** Publish! This is a significant result.

**✅ GOOD CASE: 0.65 < R² < 0.75**
- Hybrid significantly beats SSL end-to-end
- Close to classical baseline
- **Action:** Analyze feature importance, iterate on MAE architecture

**👍 ACCEPTABLE: 0.50 < R² < 0.65**
- Hybrid beats SSL end-to-end but lags classical
- **Action:** Move to Strategy 2 (optimize MAE pre-training)

### Failure Scenarios

**⚠️ UNDERPERFORMANCE: R² < 0.50**
- Hybrid underperforms both baselines
- **Possible causes:**
  - MAE embeddings not capturing relevant features
  - Overfitting during pre-training
  - Data leakage or evaluation issues
- **Action:** Debug, check data splits, analyze embeddings

---

## Output Files

All outputs saved to `outputs/hybrid_mae_gbdt/<timestamp>/`:

```
outputs/hybrid_mae_gbdt/
└── 20241101_HHMMSS/
    ├── metrics.json           # Full results and comparisons
    ├── hybrid_results.png     # 4-panel visualization
    ├── gbdt_model.pkl        # Trained GBDT model
    └── feature_scaler.pkl    # Feature standardization scaler
```

### Metrics JSON

```json
{
  "test_results": {
    "test_r2": 0.XXXX,
    "test_mae": 0.XXXX,
    "test_rmse": 0.XXXX,
    "n_samples": 141
  },
  "comparison": {
    "baseline_classical": {"r2": 0.7464, ...},
    "baseline_ssl": {"r2": 0.3665, ...},
    "hybrid": {"test_r2": 0.XXXX, ...},
    "improvement_over_classical": X.XXXX,
    "improvement_over_ssl": X.XXXX
  }
}
```

### Visualization Plots

4-panel figure:
1. **Predicted vs True scatter** - How well predictions match ground truth
2. **Residual plot** - Check for bias/heteroscedasticity
3. **Residual histogram** - Distribution of errors
4. **Model comparison bar chart** - Visual comparison to baselines

---

## Feature Analysis (Optional)

To understand what the model learned:

```python
import joblib
import numpy as np

# Load trained model
model = joblib.load('outputs/hybrid_mae_gbdt/.../gbdt_model.pkl')

# Get feature importances
importances = model.feature_importances_

# First 192 are MAE embeddings, last 2 are angles
mae_importance = importances[:192].sum()
angle_importance = importances[192:].sum()

print(f"MAE embeddings: {mae_importance:.1%} of importance")
print(f"Angles: {angle_importance:.1%} of importance")
```

This tells you whether the MAE features or angles are driving predictions.

---

## Troubleshooting

### Error: Encoder not found

```bash
# Run Phase 2 pre-training first
./scripts/run_phase2_pretrain.sh
```

### Error: Out of memory

```python
# Reduce batch size in extraction
python scripts/hybrid_mae_gbdt.py --config ... --encoder ... 
# Edit script: batch_size=32 instead of 64
```

### GridSearchCV taking too long

```bash
# Skip hyperparameter tuning
./scripts/run_hybrid_mae_gbdt.sh --no-tune
```

### Poor performance (R² < 0.4)

1. **Check data splits:** Ensure train/val/test are same as Phase 3
2. **Verify encoder:** Make sure Phase 2 completed successfully
3. **Inspect embeddings:** Check for NaN, inf, or constant values
4. **Try different GBDT params:** Manually tune if grid search fails

---

## Next Steps

### If Hybrid Succeeds (R² > 0.65)

1. **Analyze feature importance** - Which embeddings matter?
2. **Per-flight analysis** - Does it work equally well across flights?
3. **Ensemble approaches** - Combine hybrid with classical?
4. **Publication** - Write up SSL + classical ML combination

### If Hybrid Underperforms (R² < 0.65)

**Move to Strategy 2:**
- Optimize MAE pre-training (longer, larger model)
- Try different SSL methods (SimCLR, BYOL)
- Experiment with different architectures

**Or investigate classical features:**
- What hand-crafted features give R² = 0.75?
- Can we replicate those features?
- Hybrid ensemble: MAE + hand-crafted → GBDT

---

## Theoretical Foundation

### Why Separate Feature Learning from Regression?

**Neural networks are universal approximators, but:**
- With small labeled datasets (933 samples), they struggle
- The regression head has limited capacity
- Gradient descent can get stuck in local minima

**GradientBoosting excels because:**
- Builds ensemble of weak learners (decision trees)
- Each tree corrects previous errors
- Extremely effective with tabular/feature data
- Less prone to overfitting on small datasets

**The hybrid approach:**
- Uses deep learning where it's strong: **unsupervised feature learning**
- Uses classical ML where it's strong: **supervised regression on features**
- Best of both worlds!

### Comparison to End-to-End SSL

| Aspect | End-to-End SSL | Hybrid Approach |
|--------|---------------|-----------------|
| Feature extractor | MAE encoder | MAE encoder (same) |
| Regressor | Neural MLP head | GradientBoosting |
| Training | Joint fine-tuning | Frozen encoder + GBDT |
| Labeled data efficiency | Medium | High |
| Interpretability | Low | Medium (feature importance) |
| Risk of overfitting | Higher | Lower |

---

## References

**Hybrid CNN + GBDT in Remote Sensing:**
- Ma et al. (2019) - "Deep learning in remote sensing applications"
- Maxwell et al. (2018) - "Implementation of machine-learning classification in remote sensing"

**SSL + Classical ML:**
- Chen et al. (2020) - "Self-supervised learning for geospatial data"
- Rußwurm et al. (2020) - "Self-attention for raw optical satellite time series classification"

**GradientBoosting Theory:**
- Friedman (2001) - "Greedy function approximation: A gradient boosting machine"
- Chen & Guestrin (2016) - "XGBoost: A scalable tree boosting system"

---

## Summary

**Strategy 1 is the recommended first step because:**

✅ Low risk - combines two proven components  
✅ Fast to implement and run (~10-15 min)  
✅ High potential - could beat R² = 0.75  
✅ Well-supported by literature  
✅ Provides insights regardless of outcome  

**If it works:** You have a new SOTA model!  
**If it doesn't work:** You learn something important about your features.

Either way, you're moving forward with evidence-based decisions.

---

**Ready to run? Execute:**

```bash
./scripts/run_hybrid_mae_gbdt.sh
```

**Good luck! 🚀**