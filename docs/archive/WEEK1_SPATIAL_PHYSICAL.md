# Week 1: Spatial Features + Physical Priors
## Cloud Base Height Estimation - Development Sprint

**Duration:** 7 days  
**Status:** In Progress  
**Date Started:** 2024

---

## Overview

Week 1 focuses on addressing the fundamental limitations identified in the MAE CLS token approach by:

1. **Spatial Feature Extraction** (Task 1.1, 2 days): Replace 1D CLS token with spatial-aware features
2. **Physical Priors Integration** (Task 1.2, 3 days): Incorporate shadow geometry and atmospheric physics
3. **LOO Cross-Validation** (Task 1.3, 2 days): Rigorous evaluation of all Week 1 approaches

---

## Background: Why Spatial Features?

### The CLS Token Problem

Current MAE implementation extracts a **single 192-dimensional vector** (CLS token) to represent the entire image:

```
Image [440 pixels] → MAE Encoder → 27 patch tokens [27, 192]
                                  → CLS token [192]  ← Used for prediction
                                  (Spatial info discarded)
```

**Issues:**
- Cloud height is inherently **spatial** (shadows, parallax, texture gradients)
- CLS token is a **1D bottleneck** that loses spatial structure
- Analysis shows weak correlation: max |r| = 0.384 with CBH

### The Spatial Solution

Preserve spatial information by using **patch tokens** instead:

```
Image [440 pixels] → MAE Encoder → 27 patch tokens [27, 192]
                                  → Spatial head (pooling/CNN/attention)
                                  → CBH prediction
                                  (Preserves spatial relationships)
```

---

## Task 1.1: Spatial Feature Extraction (Days 1-2)

### Objective

Implement and evaluate three spatial feature extraction variants to determine if spatial features outperform the CLS token.

### Three Variants

#### **Variant A: Global Pooling + MLP**

**Concept:** Simple aggregation of spatial features

```
Patch tokens [27, 192]
  → Global Average Pooling → [192]
  → MLP [192 → 256 → 128 → 1]
  → CBH prediction
```

**Pros:** Simple, interpretable, fast  
**Cons:** Still loses fine spatial structure

**Implementation:** `src/models/spatial_mae.py::SpatialPoolingHead`

---

#### **Variant B: Lightweight CNN**

**Concept:** Convolutional processing of spatial features

```
Patch tokens [27, 192]
  → Transpose to [192, 27]
  → Conv1D(192 → 128, k=3) + ReLU + Dropout
  → Conv1D(128 → 64, k=3) + ReLU + Dropout
  → Global Average Pooling → [64]
  → Linear(64 → 1)
  → CBH prediction
```

**Pros:** Learns spatial patterns, more expressive  
**Cons:** More parameters, slower

**Implementation:** `src/models/spatial_mae.py::CNNSpatialHead`

---

#### **Variant C: Attention Pooling**

**Concept:** Learn which spatial regions matter for CBH

```
Patch tokens [27, 192]
  + Learnable query vector [1, 192]
  → Multi-head Attention (query attends to patches)
  → Attention weights [1, 27]  ← Shows which patches are important
  → Weighted pooling → [192]
  → Linear(192 → 1)
  → CBH prediction
```

**Pros:** Interpretable (attention weights), adaptive  
**Cons:** More complex, potential overfitting

**Implementation:** `src/models/spatial_mae.py::AttentionPoolingHead`

---

### Running the Experiments

#### Quick Start (All Variants)

```bash
./scripts/run_week1_task1.sh
```

This will:
1. Train all 3 variants with LOO cross-validation
2. Generate results for each variant
3. Create comparison plots
4. Save summary to `logs/week1_task1_<timestamp>/summary.txt`

#### Individual Variants

```bash
# Variant A: Pooling
python scripts/train_spatial_mae.py \
    --variant pooling \
    --config configs/ssl_finetune_cbh.yaml \
    --encoder models/mae_pretrained.pt \
    --epochs 50

# Variant B: CNN
python scripts/train_spatial_mae.py \
    --variant cnn \
    --config configs/ssl_finetune_cbh.yaml \
    --encoder models/mae_pretrained.pt \
    --epochs 50

# Variant C: Attention
python scripts/train_spatial_mae.py \
    --variant attention \
    --config configs/ssl_finetune_cbh.yaml \
    --encoder models/mae_pretrained.pt \
    --epochs 50
```

### Expected Outputs

For each variant:

```
outputs/spatial_mae/<variant>_<timestamp>/
├── loo_results.json              # LOO CV metrics (R², MAE, RMSE per fold)
├── loo_results.png               # Visualization plots
├── best_model_fold0.pt           # Trained model (fold 0)
├── best_model_fold1.pt           # Trained model (fold 1)
├── ...
└── best_model_fold4.pt           # Trained model (fold 4)
```

**loo_results.json structure:**
```json
{
  "variant": "pooling",
  "timestamp": "2024-01-15T10:30:00",
  "folds": [
    {
      "fold": 0,
      "test_flight": "30Oct24",
      "r2": 0.45,
      "mae": 150.2,
      "rmse": 185.3
    },
    ...
  ],
  "aggregate": {
    "mean_r2": 0.32,
    "std_r2": 0.15,
    "mean_mae": 175.5,
    "std_mae": 45.2,
    "mean_rmse": 210.8,
    "std_rmse": 52.1
  }
}
```

### Success Criteria

**Minimum Success:**
- ✅ All 3 variants train successfully
- ✅ LOO CV R² > 0 (better than mean baseline) for at least one variant
- ✅ Clear comparison of variants

**Target Success:**
- ✅ LOO CV R² > 0.3 (positive predictive value)
- ✅ MAE < 200m (better than random/CLS baseline)
- ✅ Attention variant produces interpretable spatial weights

**Stretch Goal:**
- ✅ LOO CV R² > 0.5 (approaching angles-only within-split performance)
- ✅ MAE < 150m

### Decision Point (End of Day 2)

Based on results, determine next steps:

**If R² > 0.3:** ✅ Spatial features work! Proceed to Task 1.2 (Physical Priors)

**If 0 < R² < 0.3:** ⚠️ Marginal improvement. Proceed to Task 1.2, but manage expectations

**If R² < 0:** ❌ Spatial features don't help. Pivot to:
- Investigate failure modes (embedding visualization, error analysis)
- Consider supervised learning from scratch (no MAE pretraining)
- Focus on physical priors only (Task 1.2)

---

## Task 1.2: Physical Priors Integration (Days 3-5)

### Objective

Incorporate atmospheric physics and shadow geometry to provide domain knowledge that pure data-driven approaches lack.

### Physical Prior #1: Shadow Geometry

**Theory:** Classical photogrammetry

```
Sun elevation angle θ + Shadow length L → Cloud height H

H = L × tan(θ)

where:
  θ = 90° - SZA (solar zenith angle)
  L = estimated from image features
```

**Implementation Plan:**

1. **Shadow Detection**
   - Edge detection on image
   - Texture analysis (variance, gradients)
   - Identify dark regions (potential shadows)

2. **Shadow Length Estimation**
   - Spatial extent of shadow regions
   - Direction aligned with sun azimuth angle (SAA)
   - Scale from pixel distance to meters (using image resolution)

3. **Geometric CBH Estimate**
   - Compute H = L × tan(90° - SZA)
   - Use as feature or auxiliary target

**File:** `src/features/physical_priors.py::ShadowGeometry`

---

### Physical Prior #2: Sun Angle Geometry

**Beyond raw SZA/SAA:** Compute derived geometric features

```python
# Incidence angle on horizontal cloud surface
incidence_angle = SZA

# Azimuth difference to flight direction
azimuth_diff = SAA - flight_heading

# Shadow direction components
shadow_x = sin(SAA * π/180)
shadow_y = cos(SAA * π/180)

# Illumination factor (affects brightness)
illumination = cos(SZA * π/180)
```

**File:** `src/features/physical_priors.py::SunAngleFeatures`

---

### Physical Prior #3: Multi-Task Learning

**Concept:** Train model to predict multiple related tasks simultaneously

```
MAE Encoder → Spatial Features → Shared Representation
                                       ├→ Task 1: CBH (primary) [MSE loss]
                                       ├→ Task 2: Shadow length [MSE loss]
                                       └→ Task 3: Cloud presence [BCE loss]

Total Loss = α × L_CBH + β × L_shadow + γ × L_presence
```

**Rationale:**
- Forces model to learn features useful for multiple cloud-related tasks
- Shadow length task encourages learning of shadow geometry
- Cloud presence task improves general cloud detection

**File:** `src/models/multitask_model.py::MultiTaskSpatialMAE`

---

### Running Physical Priors Experiments

```bash
# Train with shadow geometry features
python scripts/train_with_priors.py \
    --variant pooling \
    --use-shadow-geometry \
    --config configs/ssl_finetune_cbh.yaml

# Train multi-task model
python scripts/train_with_priors.py \
    --variant pooling \
    --multitask \
    --config configs/ssl_finetune_cbh.yaml

# Ablation: Test individual priors
python scripts/train_with_priors.py \
    --variant pooling \
    --ablation \
    --config configs/ssl_finetune_cbh.yaml
```

### Expected Outputs

```
outputs/physical_priors/<timestamp>/
├── shadow_geometry_results.json      # Results with shadow features
├── multitask_results.json            # Multi-task learning results
├── ablation_results.json             # Impact of each prior
├── shadow_visualizations/            # Detected shadows, geometric estimates
└── feature_importance/               # Which priors contribute most
```

### Success Criteria

**Minimum Success:**
- ✅ Shadow geometry features extracted successfully
- ✅ Multi-task model trains without errors
- ✅ At least one physical prior improves R² by > 0.05

**Target Success:**
- ✅ Physical priors improve LOO CV R² by > 0.1
- ✅ Shadow geometry shows correlation with CBH (r > 0.3)
- ✅ Multi-task learning outperforms single-task

**Stretch Goal:**
- ✅ Combined (spatial + physical priors) achieves R² > 0.5
- ✅ Shadow geometry alone provides interpretable CBH estimates

---

## Task 1.3: LOO Cross-Validation (Days 6-7)

### Objective

Comprehensive evaluation of all Week 1 approaches with consistent protocol and statistical testing.

### Models to Evaluate

| ID | Model | Features |
|----|-------|----------|
| M0 | Baseline | Angles-only GBDT (from diagnostics) |
| M1 | Spatial-A | Pooling + MLP |
| M2 | Spatial-B | CNN head |
| M3 | Spatial-C | Attention pooling |
| M4 | Physical-Shadow | Spatial-A + shadow geometry |
| M5 | Physical-Angles | Spatial-A + derived sun angles |
| M6 | Multi-Task | Spatial-A + multi-task learning |
| M7 | Best-Combined | Best spatial + best physical priors |

### Evaluation Protocol

**5-Fold LOO Cross-Validation:**

```python
for test_flight in ['30Oct24', '10Feb25', '12Feb25', '23Oct24', '18Feb25']:
    # Hold out test flight
    train_flights = [f for f in all_flights if f != test_flight]
    
    # Fit scalers on train only
    scaler = fit_scaler(train_data)
    
    # Train model
    model = train(train_data)
    
    # Evaluate on test flight
    metrics = evaluate(model, test_data)
```

**Metrics (per-fold and aggregate):**
- R² (coefficient of determination)
- MAE (mean absolute error, meters)
- RMSE (root mean squared error, meters)
- MAPE (mean absolute percentage error)
- 90th percentile error

### Statistical Testing

**Paired t-test:** Each model vs baseline (M0)

```python
from scipy.stats import ttest_rel

# Null hypothesis: mean(model_errors) = mean(baseline_errors)
# Alternative: model is better (lower errors)

t_stat, p_value = ttest_rel(baseline_errors, model_errors, alternative='greater')

if p_value < 0.05:
    print("Model is significantly better than baseline")
```

**Friedman test:** Overall ranking across all models

```python
from scipy.stats import friedmanchisquare

# Ranks models across all folds
ranks = friedmanchisquare(model1_r2, model2_r2, ..., model7_r2)
```

### Running Comprehensive Evaluation

```bash
# Run all models with LOO CV + statistical tests
python scripts/week1_comprehensive_eval.py \
    --models all \
    --output outputs/week1_evaluation

# Generate comparison report
python scripts/generate_week1_report.py \
    --results outputs/week1_evaluation \
    --output docs/WEEK1_RESULTS.md
```

### Expected Outputs

```
outputs/week1_evaluation/
├── loo_results_all_models.json       # Full results
├── statistical_tests.json            # p-values, effect sizes
├── comparison_table.csv              # Summary table
├── plots/
│   ├── r2_comparison.png             # Bar plot: R² per model
│   ├── mae_comparison.png            # Bar plot: MAE per model
│   ├── per_fold_breakdown.png        # Heatmap: model × fold
│   ├── predictions_vs_targets.png    # Scatter: all models
│   └── error_distributions.png       # Box plots: error per model
└── week1_report.md                   # Auto-generated summary
```

### Success Criteria

**Minimum Success:**
- ✅ All models evaluated with consistent protocol
- ✅ Statistical tests completed
- ✅ Clear recommendation: best Week 1 model

**Target Success:**
- ✅ At least one model achieves LOO CV R² > 0.3 (p < 0.05 vs baseline)
- ✅ Best model improves over CLS token baseline by > 0.2 R²
- ✅ Interpretable results (which features matter, why some folds fail)

**Stretch Goal:**
- ✅ Best model achieves LOO CV R² > 0.5
- ✅ All folds achieve R² > 0 (consistent cross-flight generalization)

---

## Week 1 Decision Tree

```
Day 2 Decision:
├─ Spatial R² > 0.3  → ✅ Continue to physical priors (Task 1.2)
├─ Spatial 0 < R² < 0.3 → ⚠️ Continue but manage expectations
└─ Spatial R² < 0 → ❌ Pivot: focus on physical priors or supervised learning

Day 5 Decision:
├─ Physical priors improve R² by > 0.1 → ✅ Excellent! Combine with best spatial
├─ Physical priors improve R² by 0.05-0.1 → ⚠️ Marginal, but proceed
└─ Physical priors don't help → ❌ Week 1 approach insufficient

Day 7 Decision:
├─ Best model R² > 0.5 → ✅ Success! Proceed to Week 2 (ERA5 + SSL)
├─ Best model R² > 0.3 → ✅ Moderate success. Week 2 may improve further
├─ Best model R² > 0 → ⚠️ Weak. Consider alternative approaches in Week 2
└─ Best model R² < 0 → ❌ Week 1 failed. Pivot to supervised/ensemble/reanalysis-only
```

---

## Troubleshooting

### Issue: Encoder not found

```bash
ERROR: Pretrained encoder not found at models/mae_pretrained.pt
```

**Solution:** Train MAE encoder first

```bash
python scripts/pretrain_mae.py --config configs/ssl_pretrain_mae.yaml
```

Or use random initialization (not recommended):
```bash
python scripts/train_spatial_mae.py --variant pooling --encoder none
```

---

### Issue: Out of memory during training

```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size in config
2. Use gradient accumulation
3. Freeze more encoder layers
4. Use smaller model variant

```yaml
# In config file
training:
  batch_size: 8  # Reduce from 16
  gradient_accumulation_steps: 2
```

---

### Issue: Training diverges (loss → NaN)

**Possible causes:**
- Learning rate too high
- Gradient explosion
- Data normalization issues

**Solutions:**
1. Lower learning rate: `lr: 1e-4` → `lr: 1e-5`
2. Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
3. Check data preprocessing (ensure targets are normalized)

---

### Issue: All folds have R² < 0

**This is a real possibility!** It means the model is worse than predicting the mean.

**Analysis steps:**
1. Check embedding quality: `python scripts/visualize_embeddings.py`
2. Verify data preprocessing (are scalers fitted correctly?)
3. Try angles-only baseline to confirm task is learnable
4. Consider that cross-flight generalization may be fundamentally hard

**Next steps:**
- Focus on per-flight models (accept that cross-flight is hard)
- Investigate domain adaptation techniques
- Incorporate more context (ERA5 in Week 2)

---

## Files Created/Modified

### New Files

```
src/models/
├── __init__.py
└── spatial_mae.py                    # Spatial feature extraction variants

src/features/
├── __init__.py
└── physical_priors.py                # Shadow geometry, sun angles

src/models/
└── multitask_model.py                # Multi-task learning architecture

scripts/
├── train_spatial_mae.py              # Training script for spatial variants
├── train_with_priors.py              # Training with physical priors
├── week1_comprehensive_eval.py       # Full LOO CV evaluation
├── generate_week1_report.py          # Auto-generate results report
└── run_week1_task1.sh                # Master runner for Task 1.1

docs/
├── WEEK1_SPATIAL_PHYSICAL.md         # This file
└── WEEK1_RESULTS.md                  # Auto-generated results (after Day 7)
```

### Modified Files

```
configs/
└── ssl_finetune_cbh.yaml             # May need batch size adjustments

src/
└── hdf5_dataset.py                   # May need to support auxiliary targets
```

---

## Timeline

| Day | Tasks | Deliverables |
|-----|-------|--------------|
| 1 | Implement 3 spatial variants | `src/models/spatial_mae.py` |
| 1 | Create training script | `scripts/train_spatial_mae.py` |
| 2 | Train all variants with LOO CV | `outputs/spatial_mae/*/loo_results.json` |
| 2 | **Decision Point 1** | Determine if spatial features work |
| 3 | Implement shadow geometry | `src/features/physical_priors.py` |
| 4 | Implement multi-task model | `src/models/multitask_model.py` |
| 5 | Train models with physical priors | `outputs/physical_priors/` |
| 5 | **Decision Point 2** | Assess impact of physical priors |
| 6 | Comprehensive LOO CV (all models) | `outputs/week1_evaluation/` |
| 7 | Statistical analysis + reporting | `docs/WEEK1_RESULTS.md` |
| 7 | **Decision Point 3** | Week 1 success? Proceed to Week 2? |

---

## Next Steps (Week 2)

If Week 1 is successful (R² > 0.3):

**Week 2 Tasks:**
1. ERA5 reanalysis integration (atmospheric profiles)
2. Semi-supervised learning (MAE on ~10k unlabeled images)
3. Improved fusion strategies

If Week 1 is unsuccessful (R² < 0):

**Pivot Options:**
1. Supervised learning from scratch (no MAE)
2. Per-flight models + domain adaptation
3. Focus on physical models only (ERA5-based)
4. Publish negative results (valuable lessons learned)

---

## References

**Spatial Feature Extraction:**
- He et al. (2022): "Masked Autoencoders Are Scalable Vision Learners"
- Dosovitskiy et al. (2021): "An Image is Worth 16x16 Words" (ViT)

**Physical Priors:**
- Seethala & Horváth (2010): "Global assessment of cloud retrievals"
- Shadow geometry in photogrammetry: Classical computer vision

**Multi-Task Learning:**
- Ruder (2017): "An Overview of Multi-Task Learning"
- Kendall et al. (2018): "Multi-Task Learning Using Uncertainty"

---

## Contact

Questions or issues? Check:
1. This documentation
2. Code comments in `src/models/spatial_mae.py`
3. GitHub issues (if repository is public)
4. Project lead

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Status:** Week 1 in progress