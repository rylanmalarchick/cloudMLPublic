# Technical Analysis and Scope of Work Recommendation
# Cloud Base Height (CBH) Estimation from Multi-Spectral Imagery

**Document Version:** 2.0  
**Date:** 2024  
**Project:** Hybrid MAE + GBDT Cloud Base Height Estimation  
**Status:** Development Phase - 3-Week Sprint Planned

---

## Executive Summary

This document outlines the technical analysis and recommended scope of work for improving cloud base height (CBH) estimation from multi-spectral satellite imagery using deep learning approaches. Based on comprehensive diagnostic analysis, we have identified specific limitations in the current MAE (Masked Autoencoder) + GBDT (Gradient Boosted Decision Tree) hybrid approach and propose a structured 3-week development plan to address them.

**Current Status:**
- Dataset: 933 labeled samples across 5 flights
- Current best performance: R² ≈ 0.70 (angles-only, within-split)
- Cross-flight generalization: Failed (LOO CV R² = -4.46)
- Root causes identified: CLS token information loss, objective mismatch, lack of physical priors

**Goal:** Develop robust CBH estimation that generalizes across flights and leverages deep learning effectively.

---

## 1. Problem Statement

### 1.1 Dataset Characteristics
- **Total labeled samples:** 933 CPL-aligned multi-spectral images
- **Flight distribution:**
  - 30Oct24: 501 samples (54%)
  - 10Feb25: 163 samples (17%)
  - 12Feb25: 144 samples (15%)
  - 23Oct24: 101 samples (11%)
  - 18Feb25: 24 samples (3%)
- **Target variable:** Cloud base height (CBH) from CPL lidar
- **Input modalities:** Multi-spectral imagery (VIS/NIR/SWIR channels), sun angles (SZA/SAA)

### 1.2 Current Performance Summary

| Approach | Within-Split R² | LOO CV R² | Status |
|----------|----------------|-----------|---------|
| Angles-only | 0.70-0.71 | -4.46 | Overfits to flight |
| MAE-only | 0.09-0.19 | Not tested | Poor |
| MAE + Angles | 0.49-0.51 | Not tested | Degraded |
| Random + Angles | ~0.50 | Not tested | Similar to MAE |

**Key Finding:** MAE embeddings currently degrade performance compared to angles-only baseline.

### 1.3 Root Causes Identified

1. **Information Bottleneck (CLS Token)**
   - CLS token is 1D (192-dim vector)
   - Discards spatial structure needed for cloud height estimation
   - Cloud height depends on shadows, parallax, texture gradients (inherently spatial)

2. **Objective Mismatch**
   - MAE pretraining: pixel-level reconstruction
   - Target task: geometric property estimation (CBH)
   - No guarantee that reconstruction-optimal features align with CBH-relevant features

3. **Lack of Physical Constraints**
   - Current model ignores atmospheric physics
   - No use of shadow geometry, temperature profiles, or reanalysis data
   - Purely data-driven approach struggles with limited samples

4. **Cross-Flight Variability**
   - Different atmospheric conditions per flight
   - Different cloud types, seasons, locations
   - Model learns flight-specific patterns that don't transfer

---

## 2. Technical Analysis: Why Current Approach Fails

### 2.1 MAE Embedding Analysis Results

**Embedding characteristics (192 dimensions):**
- Max correlation with CBH: |r| = 0.384
- Mean correlation with CBH: |r| = 0.195
- Dimensions with |r| > 0.3: 29/192 (15%)
- Dimensions with |r| > 0.5: 0/192

**Interpretation:** 
- Weak linear relationship between embeddings and CBH
- Most embedding dimensions encode information unrelated to cloud height
- Some dimensions correlate more with sun angles than CBH

### 2.2 Feature Importance Analysis

**GBDT built-in importance (MAE + Angles model):**
- MAE dimensions dominate numerically (high Gini importance)
- However, permutation importance shows MAE contributes little to actual predictions
- Model relies on angle features for predictive power

**Interpretation:**
- GBDT overfits to high-dimensional MAE embeddings
- Numerical "importance" ≠ useful generalizable signal
- MAE adds noise/overfitting capacity, not signal

### 2.3 Cross-Flight Generalization Failure

**Leave-One-Out (LOO) Cross-Validation Results:**

| Held-Out Flight | R² | MAE (m) | RMSE (m) | Interpretation |
|----------------|-----|---------|----------|----------------|
| 30Oct24 | -1.47 | 291 | 335 | Worse than mean |
| 10Feb25 | -3.08 | 371 | 417 | Severe overprediction |
| 12Feb25 | -0.85 | 250 | 295 | Poor generalization |
| 23Oct24 | -10.12 | 614 | 720 | Complete failure |
| 18Feb25 | -18.40 | 814 | 912 | Catastrophic failure |
| **Aggregate** | **-4.46 ± 7.09** | **348 ± 78** | **420 ± 85** | **No generalization** |

**Interpretation:**
- Negative R² indicates predictions worse than using mean CBH
- Model learns flight-specific correlations that don't transfer
- Small flights (18Feb25: 24 samples) are completely unpredictable

---

## 3. Proposed Solutions: 3-Week Development Plan

### Week 1: Spatial Representations + Physical Priors

#### Task 1.1: Spatial Feature Extraction (2 days)

**Objective:** Replace 1D CLS token with spatial-aware features that preserve cloud structure.

**Technical Approach:**

1. **Extract spatial feature maps from MAE encoder**
   - Use patch tokens instead of CLS token
   - Reshape to 2D spatial grid (e.g., 14×14 grid for ViT-Base with patch_size=16)
   - Preserve spatial relationships between image regions

2. **Implement spatial-aware heads (3 variants to test)**

   **Variant A: Spatial Pooling + MLP**
   ```
   Input: Patch tokens [N_patches, embed_dim] → Reshape to [H, W, embed_dim]
   → Global Average Pooling → [embed_dim]
   → MLP [embed_dim → 256 → 128 → 1]
   → CBH prediction
   ```

   **Variant B: Lightweight CNN**
   ```
   Input: Spatial feature map [H, W, embed_dim]
   → Conv2D(embed_dim, 128, 3×3) + ReLU
   → Conv2D(128, 64, 3×3) + ReLU
   → Global Average Pooling → [64]
   → Linear(64 → 1)
   → CBH prediction
   ```

   **Variant C: Attention Pooling**
   ```
   Input: Patch tokens [N_patches, embed_dim]
   → Learn attention weights per patch (which regions matter for CBH)
   → Weighted pooling → [embed_dim]
   → Linear(embed_dim → 1)
   → CBH prediction
   ```

3. **Hybrid variants (combine with angles/hand-crafted features)**
   - Concatenate spatial features with SZA/SAA
   - Feed to GBDT or MLP for fusion

**Implementation Files:**
- `src/models/spatial_mae.py` - Spatial feature extraction module
- `scripts/train_spatial_mae.py` - Training script for spatial variants
- `scripts/eval_spatial_mae.py` - LOO CV evaluation

**Success Criteria:**
- Spatial features show higher correlation with CBH than CLS token
- LOO CV R² > 0 (better than mean prediction)
- Hybrid spatial+angles outperforms angles-only

**Deliverables:**
- 3 trained spatial models (Variants A, B, C)
- LOO CV results for each
- Embedding visualization comparing CLS vs spatial features

---

#### Task 1.2: Physical Priors Integration (3 days)

**Objective:** Incorporate atmospheric physics and shadow geometry to constrain CBH estimates.

**Physical Priors to Implement:**

1. **Shadow Geometry (Classical Photogrammetry)**
   - **Theory:** Shadow length L, sun elevation θ → Cloud height H = L × tan(θ)
   - **Implementation:**
     - Extract shadow features from imagery (edge detection, texture analysis)
     - Estimate shadow length from spatial features
     - Compute geometric CBH estimate
   - **Use case:** As additional feature or auxiliary prediction target

2. **Thermal Constraints (if IR channels available)**
   - **Theory:** Cloud-top brightness temperature + lapse rate → height estimate
   - **Implementation:**
     - Extract brightness temperature from thermal IR bands
     - Use standard atmosphere lapse rate (6.5 K/km)
     - Estimate cloud top height, constrain CBH below it
   - **Use case:** As valid range constraint or feature

3. **Sun Angle Geometry**
   - **Beyond raw SZA/SAA:** Compute derived geometric features
     - Azimuth difference to flight direction
     - Incidence angle on hypothetical horizontal cloud surface
     - Shadow direction vector components
   - **Use case:** More informative than raw angles

**Multi-Task Learning Implementation:**
```
MAE Encoder → Spatial Features → Shared Representation
                                       ├→ Task 1: CBH regression (primary)
                                       ├→ Task 2: Shadow length estimation (auxiliary)
                                       └→ Task 3: Cloud presence classification (auxiliary)

Loss = α × L_CBH + β × L_shadow + γ × L_presence
```

**Implementation Files:**
- `src/features/physical_priors.py` - Shadow geometry, thermal constraints
- `src/models/multitask_model.py` - Multi-task architecture
- `scripts/train_with_priors.py` - Training with physical constraints

**Success Criteria:**
- Physical features show meaningful correlation with CBH
- Multi-task learning improves primary CBH prediction
- Physical constraints reduce unrealistic predictions (e.g., negative CBH)

**Deliverables:**
- Shadow geometry feature extractor
- Multi-task model with 2-3 auxiliary tasks
- Ablation study: impact of each physical prior
- LOO CV results with physical priors

---

#### Task 1.3: LOO Cross-Validation on Week 1 Models (2 days)

**Objective:** Rigorous evaluation of spatial and physics-based approaches.

**Models to Evaluate:**
1. Spatial Variant A (pooling + MLP)
2. Spatial Variant B (CNN head)
3. Spatial Variant C (attention pooling)
4. Spatial + Physical priors
5. Multi-task model
6. Baseline (angles-only) for comparison

**Evaluation Protocol:**
- 5-fold LOO CV (hold out each flight)
- Metrics: R², MAE, RMSE, per-fold breakdown
- Statistical significance testing (paired t-test vs baseline)
- Uncertainty quantification (prediction intervals)

**Analysis:**
- Per-flight performance breakdown
- Failure case analysis (which flights/conditions fail?)
- Embedding visualization (t-SNE, correlation heatmaps)
- Feature importance (which spatial regions or priors matter?)

**Deliverables:**
- Comprehensive LOO CV results table
- Statistical analysis of improvements
- Diagnostic plots and failure analysis
- Recommendation: best Week 1 approach to build on

---

### Week 2: Reanalysis Integration + Semi-Supervised Learning

#### Task 2.1: ERA5 Reanalysis Integration (4 days)

**Objective:** Incorporate atmospheric profile data to provide physical context for CBH estimation.

**ERA5 Variables to Extract:**

1. **Boundary Layer Diagnostics**
   - Boundary layer height (BLH) - direct constraint on low cloud CBH
   - Lifting condensation level (LCL) - theoretical cloud base height
   - Convective available potential energy (CAPE)

2. **Vertical Profiles (1000-500 hPa)**
   - Temperature (K)
   - Specific humidity (kg/kg)
   - Relative humidity (%)
   - Geopotential height (m)

3. **Derived Quantities**
   - Temperature inversion layers (likely cloud bases)
   - Humidity gradients (cloud layer detection)
   - Static stability (lapse rate)

**Implementation Steps:**

**Day 1: Data Acquisition**
- Set up ERA5 API access (CDS API)
- Extract reanalysis data at flight times/locations
- Spatial resolution: 0.25° × 0.25° (nearest grid point to flight)
- Temporal resolution: Hourly interpolation

```python
# Pseudo-code
import cdsapi

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': ['temperature', 'specific_humidity', 'geopotential'],
        'pressure_level': [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500],
        'year': flight_years,
        'month': flight_months,
        'day': flight_days,
        'time': flight_times,
        'area': [max_lat, min_lon, min_lat, max_lon],  # Flight bounding box
        'format': 'netcdf'
    }
)
```

**Day 2: Feature Engineering**
- Compute derived quantities:
  - LCL from surface T, Td: LCL = 125 × (T - Td) meters
  - Lapse rate: dT/dz
  - Inversion detection: layers where dT/dz > 0
  - Humidity jump detection: large d(RH)/dz
- Create profile-based features:
  - Average T, RH in BL
  - Strongest inversion height
  - Max humidity gradient height

**Day 3: Integration with Model**

**Approach A: Direct Feature Integration**
```
Spatial MAE Features [dim_spatial]
        +
ERA5 Features [dim_era5]
        ↓
Concatenate → [dim_spatial + dim_era5]
        ↓
GBDT or MLP Fusion → CBH prediction
```

**Approach B: Physics-Guided Prediction**
```
ERA5 → Compute LCL, BLH (physics-based CBH estimate)
        +
MAE → Data-driven CBH residual
        ↓
Final CBH = LCL + learned_residual
```

**Approach C: Multi-Modal Architecture**
```
Image Branch: MAE → Spatial Features
Profile Branch: ERA5 → 1D CNN or LSTM → Profile Encoding
        ↓
Cross-Attention or Concatenation → Fusion
        ↓
CBH prediction
```

**Day 4: Evaluation and Analysis**
- LOO CV with ERA5 features
- Ablation: Which ERA5 variables contribute most?
- Analysis: Does ERA5 help cross-flight generalization?
- Visualization: ERA5 profile examples from each flight

**Implementation Files:**
- `src/data/era5_downloader.py` - ERA5 API interface
- `src/features/atmospheric_profiles.py` - Profile feature engineering
- `src/models/multimodal_fusion.py` - Image + ERA5 fusion
- `scripts/prepare_era5_data.py` - Batch data preparation
- `scripts/train_with_era5.py` - Training script

**Success Criteria:**
- ERA5 features improve LOO CV R² by > 0.1
- Physical consistency: predictions respect BLH, LCL constraints
- Reduced per-flight variance in predictions

**Deliverables:**
- ERA5 dataset aligned with flight data
- Feature importance analysis for ERA5 variables
- LOO CV results with ERA5 integration
- Profile visualization and analysis

---

#### Task 2.2: Semi-Supervised Learning with Unlabeled Data (3 days)

**Objective:** Leverage unlabeled CPL-aligned imagery to improve representation learning.

**Unlabeled Data Inventory:**
- Estimate: ~10,000+ unlabeled CPL-aligned images available
- Covers more flights, atmospheric conditions, cloud types
- No ground-truth CBH labels

**Semi-Supervised Strategies:**

**Strategy 1: Improved MAE Pretraining (Days 1-2)**

**Current limitation:** MAE trained only on 933 labeled images
**Solution:** Pretrain MAE on full unlabeled dataset

```
Phase 1: MAE Pretraining
    Input: ~10,000 unlabeled images
    Objective: Masked image reconstruction
    Output: Pretrained encoder with better representations

Phase 2: Supervised Fine-Tuning
    Input: 933 labeled images + pretrained encoder
    Objective: CBH regression
    Output: Task-specific model
```

**Implementation:**
- Data loading pipeline for unlabeled images
- MAE pretraining on full dataset (2-3 epochs, ~6 hours on GPU)
- Evaluate: Do embeddings from larger pretraining show better CBH correlation?

**Strategy 2: Pseudo-Labeling (Day 2-3)**

**Approach:** Use best current model to generate noisy labels for unlabeled data

```
Step 1: Train teacher model on labeled data
    Model: Best Week 1 approach (e.g., Spatial MAE + ERA5)
    
Step 2: Generate pseudo-labels
    Predict CBH for unlabeled images
    Keep high-confidence predictions (e.g., top 25% by uncertainty)
    
Step 3: Train student model
    Combined dataset: labeled (933) + pseudo-labeled (2,000-3,000)
    Weight loss: α × L_labeled + β × L_pseudo (β < α)
    
Step 4: Iterate (optional)
    Student becomes new teacher
    Repeat 2-3 times (self-training)
```

**Confidence Filtering:**
- Ensemble uncertainty: Train 5 models, keep predictions with low variance
- ERA5 consistency: Keep predictions within ±500m of LCL/BLH
- Physical constraints: Reject negative CBH, CBH > 5km

**Strategy 3: Consistency Regularization (Day 3)**

**Approach:** Unsupervised loss on unlabeled data using augmentation consistency

```
For each unlabeled image:
    x_1 = weak_augmentation(x)    # e.g., small intensity jitter
    x_2 = strong_augmentation(x)  # e.g., large intensity + spectral mixing
    
    pred_1 = model(x_1)
    pred_2 = model(x_2)
    
    L_consistency = MSE(pred_1, pred_2)  # Predictions should match

Total loss = L_supervised(labeled) + λ × L_consistency(unlabeled)
```

**Implementation Files:**
- `src/data/unlabeled_dataset.py` - Unlabeled data loader
- `src/ssl/mae_pretrain_large.py` - MAE on full dataset
- `src/ssl/pseudo_labeling.py` - Pseudo-label generation and filtering
- `src/ssl/consistency_loss.py` - Consistency regularization
- `scripts/pretrain_mae_unlabeled.py` - Pretraining script
- `scripts/train_semi_supervised.py` - Semi-supervised training

**Success Criteria:**
- MAE pretrained on unlabeled data shows better embeddings (higher CBH correlation)
- Pseudo-labeling increases effective training set by 2-3x
- LOO CV improves with semi-supervised learning

**Deliverables:**
- MAE model pretrained on ~10k unlabeled images
- Pseudo-labeled dataset with confidence scores
- Semi-supervised training results (LOO CV)
- Ablation: Impact of unlabeled data scale

---

### Week 3: Integration, Comparison, and Documentation

#### Task 3.1: Comprehensive Model Comparison (2 days)

**Objective:** Rigorous comparison of all developed approaches under consistent evaluation.

**Models to Compare:**

| ID | Model Description | Key Features |
|----|-------------------|--------------|
| M1 | Baseline: Angles-only GBDT | SZA, SAA |
| M2 | MAE CLS + GBDT (original) | 192-dim CLS token + angles |
| M3 | Spatial MAE (Variant A) | Global pooled spatial features |
| M4 | Spatial MAE (Variant B) | CNN on spatial features |
| M5 | Spatial MAE (Variant C) | Attention-pooled features |
| M6 | Spatial MAE + Physical Priors | Shadow geometry, multi-task |
| M7 | Spatial MAE + ERA5 | Atmospheric profiles |
| M8 | Spatial MAE + ERA5 + SSL | Pretrained on unlabeled data |
| M9 | Ensemble (Top 3 models) | Uncertainty-weighted average |

**Evaluation Protocol:**

1. **5-Fold LOO Cross-Validation**
   - Consistent splits across all models
   - Same preprocessing, same scalers
   - Same random seeds for reproducibility

2. **Metrics (per-fold and aggregate):**
   - R² (coefficient of determination)
   - MAE (mean absolute error, meters)
   - RMSE (root mean squared error, meters)
   - MAPE (mean absolute percentage error)
   - 90th percentile error
   - Per-flight R², MAE breakdown

3. **Statistical Testing:**
   - Paired t-test: each model vs baseline (M1)
   - Friedman test: overall ranking across folds
   - Effect size (Cohen's d)
   - Confidence intervals (bootstrap, 1000 samples)

4. **Uncertainty Quantification:**
   - Prediction intervals (90%, 95%)
   - Per-prediction uncertainty estimates
   - Calibration analysis (reliability diagrams)

**Analysis Dimensions:**

- **Generalization:** LOO CV performance (primary metric)
- **Data Efficiency:** Learning curves (performance vs training set size)
- **Robustness:** Performance across different cloud types, seasons, SZA ranges
- **Computational Cost:** Training time, inference time, model size
- **Interpretability:** Feature importance, spatial attention maps

**Implementation Files:**
- `scripts/compare_all_models.py` - Unified comparison framework
- `scripts/statistical_tests.py` - Significance testing
- `scripts/generate_comparison_plots.py` - Visualization

**Deliverables:**
- Comprehensive results table (all models × all metrics × all folds)
- Statistical significance matrix (which models are significantly better?)
- Visualization: box plots, error distribution, per-flight breakdown
- Recommendation: best single model and best ensemble

---

#### Task 3.2: Best Model Implementation + Uncertainty Quantification (3 days)

**Objective:** Productionize the best-performing approach with robust uncertainty estimates.

**Day 1: Model Selection and Refinement**

Based on Task 3.1 results, select best model(s). Likely candidates:
- If spatial features work: Spatial MAE + ERA5 + SSL
- If ensemble wins: Uncertainty-weighted ensemble of top-3

**Refinements:**
- Hyperparameter tuning on best architecture (small grid search via nested CV)
- Optimal feature selection (remove low-importance features)
- Calibration: Apply isotonic regression or temperature scaling to uncertainty estimates

**Day 2: Uncertainty Quantification Implementation**

**Approach A: Ensemble-Based Uncertainty**
```python
# Train K models with different initializations
models = [train_model(seed=i) for i in range(K)]

# Prediction + uncertainty
predictions = [model.predict(x) for model in models]
mean_pred = np.mean(predictions, axis=0)
std_pred = np.std(predictions, axis=0)  # Epistemic uncertainty

# Prediction interval
lower_bound = mean_pred - 1.96 * std_pred  # 95% CI
upper_bound = mean_pred + 1.96 * std_pred
```

**Approach B: Quantile Regression**
```python
# Train model to predict quantiles instead of mean
model_lower = train_model(loss=quantile_loss(α=0.05))  # 5th percentile
model_median = train_model(loss=quantile_loss(α=0.50))  # Median
model_upper = train_model(loss=quantile_loss(α=0.95))  # 95th percentile

# Prediction interval directly from model
pred_interval = (model_lower.predict(x), model_median.predict(x), model_upper.predict(x))
```

**Approach C: Monte Carlo Dropout (if using neural network)**
```python
# Enable dropout at inference time
model.eval()  # But keep dropout active
predictions = [model.predict(x) for _ in range(100)]  # 100 stochastic forward passes
mean_pred = np.mean(predictions, axis=0)
std_pred = np.std(predictions, axis=0)
```

**Calibration Check:**
- Expected: 90% of true CBH should fall within 90% prediction interval
- Actual: Compute empirical coverage on validation set
- If miscalibrated: Apply post-hoc calibration (isotonic regression)

**Day 3: Production Pipeline**

**Components:**
1. **Preprocessing module**
   - Image normalization
   - Angle encoding
   - ERA5 feature extraction

2. **Inference module**
   - Load trained model(s)
   - Predict CBH + uncertainty
   - Apply physical constraints (CBH ≥ 0, CBH ≤ BLH + margin)

3. **Out-of-Distribution Detection**
   - Flag predictions with high uncertainty (> threshold)
   - Flag images dissimilar to training set (Mahalanobis distance in embedding space)
   - Recommend caution for OOD predictions

4. **API/Interface**
```python
def predict_cbh(image, metadata):
    """
    Predict cloud base height with uncertainty.
    
    Args:
        image: Multi-spectral image array [H, W, C]
        metadata: Dict with 'sza', 'saa', 'time', 'lat', 'lon'
    
    Returns:
        {
            'cbh_mean': float,           # meters
            'cbh_lower': float,          # 5th percentile
            'cbh_upper': float,          # 95th percentile
            'uncertainty': float,        # std dev
            'confidence': float,         # [0, 1], inverse of uncertainty
            'is_ood': bool,              # out-of-distribution flag
            'era5_lcl': float,           # ERA5-based LCL for reference
        }
    """
```

**Implementation Files:**
- `src/models/best_model.py` - Best model architecture
- `src/uncertainty/quantification.py` - Uncertainty estimation
- `src/inference/pipeline.py` - End-to-end inference
- `scripts/calibrate_uncertainty.py` - Uncertainty calibration
- `api/predict.py` - Prediction API

**Deliverables:**
- Production-ready model checkpoint
- Uncertainty-calibrated predictions
- Inference API with OOD detection
- User guide for model deployment

---

#### Task 3.3: Documentation and Write-Up (2 days)

**Objective:** Comprehensive documentation for publication and future development.

**Day 1: Technical Documentation**

**1. Methods Documentation**
- `docs/METHODS.md`:
  - Dataset description and preprocessing
  - Architecture details (diagrams, equations)
  - Training procedure (hyperparameters, optimization)
  - Evaluation protocol (LOO CV, metrics)
  - Uncertainty quantification approach

**2. Results Documentation**
- `docs/RESULTS.md`:
  - Performance summary table
  - LOO CV results breakdown
  - Ablation studies (what contributed most?)
  - Failure analysis (where and why it fails)
  - Comparison to baselines and prior work

**3. Code Documentation**
- Docstrings for all modules
- README updates with usage examples
- Tutorial notebook: `notebooks/CBH_Prediction_Tutorial.ipynb`

**Day 2: Publication Draft**

**Manuscript Outline (Depending on Outcome):**

**Option A: Success Story**
*"Spatial-Aware Deep Learning for Cross-Flight Cloud Base Height Estimation"*

1. Introduction
   - Cloud base height importance
   - Challenges: small labeled datasets, cross-flight variability
   - Contribution: Spatial MAE + physical priors + semi-supervised learning

2. Related Work
   - Cloud property retrieval (traditional, DL)
   - Self-supervised learning for remote sensing
   - Transfer learning with limited labels

3. Methods
   - Dataset and preprocessing
   - Spatial MAE architecture
   - Physical priors integration
   - ERA5 reanalysis fusion
   - Semi-supervised learning strategy
   - Uncertainty quantification

4. Experiments
   - Evaluation protocol (LOO CV)
   - Ablation studies
   - Comparison to baselines

5. Results
   - Performance metrics
   - Generalization analysis
   - Uncertainty calibration
   - Feature importance and interpretability

6. Discussion
   - Why spatial features matter
   - Role of physical priors
   - Limitations and future work

7. Conclusion

**Option B: Negative Results / Lessons Learned**
*"Challenges in Self-Supervised Learning for Geometric Property Estimation: A Case Study in Cloud Heights"*

1. Introduction
   - Promise of SSL for remote sensing
   - Our hypothesis: MAE → GBDT for CBH
   - Spoiler: It didn't work, here's why

2. Methods (same as above)

3. Results: What Worked and What Didn't
   - MAE CLS embeddings fail
   - Spatial features partially help
   - Physical priors essential
   - Cross-flight generalization remains challenging

4. Analysis: Root Causes
   - Objective mismatch (reconstruction ≠ geometry)
   - Information bottleneck (CLS token)
   - Dataset scale vs variability
   - Flight-specific confounders

5. Lessons Learned
   - Importance of proper validation (LOO CV)
   - Domain knowledge > pure data-driven
   - SSL objectives must align with downstream tasks
   - Small-data regime requires different approaches

6. Recommendations
   - When to use SSL vs supervised
   - Architecture choices for geometric tasks
   - Incorporating physical priors
   - Validation strategies for limited data

7. Conclusion

**Figures to Prepare:**
1. Dataset overview (flight map, CBH distribution)
2. Architecture diagram (spatial MAE + fusion)
3. LOO CV results (box plots, per-flight breakdown)
4. Embedding analysis (t-SNE, correlation heatmap)
5. Ablation study (bar chart of R² by approach)
6. Feature importance (SHAP or permutation)
7. Prediction examples (good and bad cases)
8. Uncertainty calibration (reliability diagram)

**Implementation Files:**
- `docs/METHODS.md`
- `docs/RESULTS.md`
- `docs/PUBLICATION_DRAFT.md`
- `notebooks/CBH_Prediction_Tutorial.ipynb`
- `scripts/generate_publication_figures.py`

**Deliverables:**
- Complete technical documentation
- Publication draft (4,000-6,000 words)
- 8-10 publication-quality figures
- Tutorial notebook for reproducibility
- Code release preparation (clean repo, license, citation)

---

## 4. Timeline and Milestones

### Week 1: Spatial + Physical (7 days)

| Day | Tasks | Deliverables |
|-----|-------|--------------|
| 1 | Implement spatial feature extraction (3 variants) | `src/models/spatial_mae.py` |
| 2 | Train and evaluate spatial models | LOO CV results for variants A, B, C |
| 3 | Implement shadow geometry features | `src/features/physical_priors.py` |
| 4 | Implement multi-task learning | `src/models/multitask_model.py` |
| 5 | Train models with physical priors | Trained models with priors |
| 6 | Comprehensive LOO CV evaluation | Week 1 results summary |
| 7 | Analysis and diagnostics | Embedding visualizations, ablations |

**Milestone 1:** Determine if spatial features outperform CLS token (decision point for Week 2)

---

### Week 2: Reanalysis + Semi-Supervised (7 days)

| Day | Tasks | Deliverables |
|-----|-------|--------------|
| 8 | ERA5 data download and preprocessing | ERA5 dataset aligned with flights |
| 9 | ERA5 feature engineering | Atmospheric profile features |
| 10 | Implement ERA5 fusion models | `src/models/multimodal_fusion.py` |
| 11 | Train and evaluate with ERA5 | LOO CV results with ERA5 |
| 12 | MAE pretraining on unlabeled data | Pretrained MAE checkpoint |
| 13 | Pseudo-labeling and semi-supervised training | Semi-supervised models |
| 14 | Evaluation and comparison | Week 2 results summary |

**Milestone 2:** Assess impact of ERA5 and unlabeled data (decision point for Week 3)

---

### Week 3: Integration + Documentation (7 days)

| Day | Tasks | Deliverables |
|-----|-------|--------------|
| 15 | Comprehensive model comparison | All models × LOO CV results |
| 16 | Statistical testing and analysis | Significance tests, rankings |
| 17 | Best model selection and refinement | Final production model |
| 18 | Uncertainty quantification implementation | Calibrated uncertainty estimates |
| 19 | Production pipeline development | Inference API |
| 20 | Technical documentation | `docs/METHODS.md`, `docs/RESULTS.md` |
| 21 | Publication draft and figures | Draft manuscript + figures |

**Milestone 3:** Final model ready for deployment + publication draft complete

---

## 5. Risk Assessment and Mitigation

### Risk 1: Spatial Features Don't Improve Performance
**Likelihood:** Medium  
**Impact:** High  
**Mitigation:**
- Have fallback: Physical priors may work independently
- Pivot to supervised learning if SSL fails
- Negative result is still publishable

### Risk 2: ERA5 Data Download Issues
**Likelihood:** Low-Medium  
**Impact:** Medium  
**Mitigation:**
- Start download early (Day 8)
- Have backup: MERRA-2 reanalysis as alternative
- Can proceed with other tasks while downloading

### Risk 3: Unlabeled Data Not Available/Insufficient
**Likelihood:** Low  
**Impact:** Low  
**Mitigation:**
- Semi-supervised learning is bonus, not critical
- Can skip Task 2.2 if needed
- Focus on ERA5 and physical priors instead

### Risk 4: No Approach Achieves Good Cross-Flight Generalization
**Likelihood:** Medium  
**Impact:** Medium (for deployment), Low (for publication)  
**Mitigation:**
- Per-flight models + uncertainty quantification as fallback
- Few-shot adaptation approach (use first N samples from new flight)
- Publish negative results + lessons learned (valuable contribution)

### Risk 5: Computational Resources Insufficient
**Likelihood:** Low  
**Impact:** Medium  
**Mitigation:**
- Use smaller model variants if needed
- Cloud GPU rental (AWS, Google Cloud) if local GPU insufficient
- Optimize training (mixed precision, gradient checkpointing)

---

## 6. Resource Requirements

### Computational Resources
- **GPU:** 1× NVIDIA GPU with ≥16GB VRAM (RTX 4090, A5000, or better)
- **RAM:** ≥32GB for data loading
- **Storage:** ~100GB for datasets, models, results
- **Estimated compute time:**
  - Week 1: ~40 GPU-hours
  - Week 2: ~60 GPU-hours (MAE pretraining intensive)
  - Week 3: ~20 GPU-hours
  - **Total:** ~120 GPU-hours over 3 weeks

### Data Requirements
- **Labeled dataset:** 933 samples (already available)
- **Unlabeled dataset:** ~10,000 images (to be confirmed)
- **ERA5 reanalysis:** ~5GB download (free via CDS API)

### Human Resources
- **Primary researcher:** Full-time, 3 weeks
- **Domain expert consultation:** ~2 hours/week (atmospheric physics, shadow geometry validation)
- **Code review (optional):** ~2 hours at Week 3

---

## 7. Success Criteria

### Minimum Viable Success (Must Achieve)
1. ✓ Implement and evaluate spatial feature extraction
2. ✓ Integrate at least one physical prior (shadow geometry or ERA5)
3. ✓ Comprehensive LOO CV evaluation of all approaches
4. ✓ Identify which components contribute to performance
5. ✓ Complete technical documentation

### Target Success (Goal)
1. ✓ LOO CV R² > 0.3 (better than baseline, positive predictive value)
2. ✓ All folds achieve R² > 0 (better than mean prediction)
3. ✓ Uncertainty-calibrated predictions (90% interval coverage ≈ 90%)
4. ✓ Publication-ready manuscript draft
5. ✓ Production-ready inference pipeline

### Stretch Goals (Aspirational)
1. ◯ LOO CV R² > 0.5 (strong cross-flight generalization)
2. ◯ MAE < 100m on all flights
3. ◯ Interpretable spatial attention maps (which image regions determine CBH)
4. ◯ Real-time inference capability (<100ms per image)

---

## 8. Decision Points

### End of Week 1 (Day 7)
**Question:** Do spatial features outperform CLS token?
- **If YES:** Proceed with Week 2 as planned (ERA5 + SSL)
- **If NO:** Pivot focus to physical priors only, consider supervised learning alternative

### End of Week 2 (Day 14)
**Question:** Has any approach achieved LOO CV R² > 0?
- **If YES:** Week 3 focuses on refinement, ensemble, publication of success
- **If NO:** Week 3 focuses on failure analysis, lessons learned, diagnostic publication

### Mid-Week 3 (Day 17)
**Question:** Is best model good enough for deployment?
- **If YES:** Complete production pipeline, deployment documentation
- **IF NO:** Focus on research publication, recommendations for future work

---

## 9. Expected Outcomes

### Technical Outcomes
1. **Deep Learning Model:** Production-ready CBH estimation model with uncertainty quantification
2. **Diagnostic Framework:** Tools for embedding analysis, feature importance, LOO CV evaluation
3. **Code Repository:** Clean, documented, reproducible codebase
4. **Dataset:** ERA5-augmented labeled dataset for future experiments

### Scientific Outcomes
1. **Understanding:** Why MAE CLS embeddings fail for CBH (objective mismatch, information loss)
2. **Solutions:** Spatial features, physical priors, reanalysis integration as fixes
3. **Validation:** Importance of LOO CV for small, flight-based datasets
4. **Generalization:** Limits of cross-flight generalization, per-flight adaptation strategies

### Publication Outcomes
1. **Manuscript:** 4,000-6,000 word paper (conference or journal)
2. **Target Venues (if positive results):**
   - IEEE Transactions on Geoscience and Remote Sensing (TGRS)
   - Remote Sensing (MDPI)
   - NeurIPS/ICLR Workshop on AI for Earth Sciences
3. **Target Venues (if negative results):**
   - NeurIPS/ICML Workshop on Negative Results
   - ICLR Blogposts Track
   - Remote Sensing Special Issue on ML Challenges

---

## 10. References and Prior Work

### Self-Supervised Learning for Remote Sensing
- He et al. (2022): "Masked Autoencoders Are Scalable Vision Learners"
- Caron et al. (2021): "Emerging Properties in Self-Supervised Vision Transformers" (DINO)
- Reed et al. (2022): "Self-Supervised Pretraining Improves Self-Supervised Pretraining" (multi-task SSL)

### Cloud Property Retrieval
- Seethala & Horváth (2010): "Global assessment of AMSR-E and MODIS cloud liquid water path retrievals"
- Stubenrauch et al. (2013): "Assessment of Global Cloud Datasets from Satellites"
- Desmons et al. (2017): "A global multilayer cloud identification with CALIPSO"

### Physical Constraints in DL
- Beucler et al. (2021): "Enforcing analytic constraints in neural networks emulating physical systems"
- Karpatne et al. (2017): "Theory-Guided Data Science: A New Paradigm for Scientific Discovery"

### Semi-Supervised Learning
- Tarvainen & Valpola (2017): "Mean teachers are better role models" (consistency regularization)
- Sohn et al. (2020): "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"

---

## 11. Appendices

### Appendix A: Current Codebase Structure
```
cloudMLPublic/
├── src/
│   ├── models/
│   │   ├── mae.py                    # Current MAE implementation
│   │   ├── spatial_mae.py            # [TO CREATE] Spatial feature extraction
│   │   ├── multitask_model.py        # [TO CREATE] Multi-task architecture
│   │   └── multimodal_fusion.py      # [TO CREATE] Image + ERA5 fusion
│   ├── features/
│   │   ├── physical_priors.py        # [TO CREATE] Shadow geometry, etc.
│   │   └── atmospheric_profiles.py   # [TO CREATE] ERA5 feature engineering
│   ├── data/
│   │   ├── era5_downloader.py        # [TO CREATE] ERA5 API interface
│   │   └── unlabeled_dataset.py      # [TO CREATE] Unlabeled data loader
│   ├── ssl/
│   │   ├── mae_pretrain_large.py     # [TO CREATE] MAE on unlabeled data
│   │   └── pseudo_labeling.py        # [TO CREATE] Pseudo-labeling
│   ├── uncertainty/
│   │   └── quantification.py         # [TO CREATE] Uncertainty estimation
│   └── split_utils.py                # [EXISTS] Stratified splitting
├── scripts/
│   ├── train_spatial_mae.py          # [TO CREATE]
│   ├── train_with_priors.py          # [TO CREATE]
│   ├── train_with_era5.py            # [TO CREATE]
│   ├── train_semi_supervised.py      # [TO CREATE]
│   ├── compare_all_models.py         # [TO CREATE]
│   └── run_loo_validation.sh         # [EXISTS] LOO CV runner
└── docs/
    ├── METHODS.md                    # [TO CREATE]
    ├── RESULTS.md                    # [TO CREATE]
    └── PUBLICATION_DRAFT.md          # [TO CREATE]
```

### Appendix B: LOO CV Protocol Details

**Stratified Leave-One-Flight-Out Cross-Validation:**

```python
flights = ['10Feb25', '30Oct24', '23Oct24', '18Feb25', '12Feb25']

for test_flight in flights:
    # Split data
    train_flights = [f for f in flights if f != test_flight]
    train_data = df[df['flight'].isin(train_flights)]
    test_data = df[df['flight'] == test_flight]
    
    # Further split train into train/val (stratified by flight)
    val_flight = train_flights[0]  # Or use 15% of each remaining flight
    train_final = train_data[train_data['flight'] != val_flight]
    val_data = train_data[train_data['flight'] == val_flight]
    
    # Fit scalers on train_final only
    scaler = fit_scaler(train_final)
    train_scaled = scaler.transform(train_final)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    
    # Train model
    model = train(train_scaled, val_scaled)
    
    # Evaluate on held-out test flight
    predictions = model.predict(test_scaled)
    metrics = compute_metrics(test_data['cbh'], predictions)
    
    results[test_flight] = metrics
```

**Key Principles:**
- Test flight completely held out (no data leakage)
- Scalers fitted only on training data
- Validation split for hyperparameter tuning (nested CV)
- Report per-fold and aggregate metrics

### Appendix C: Computational Cost Estimates

| Task | GPU Hours | Wall Time |
|------|-----------|-----------|
| Spatial MAE training (3 variants) | 12 | 4 hours (parallel) |
| Multi-task model training | 8 | 8 hours |
| ERA5 data download | 0 (CPU) | 4-6 hours |
| ERA5 fusion model training | 10 | 10 hours |
| MAE pretraining on 10k unlabeled | 30 | 12-15 hours |
| Semi-supervised training | 10 | 10 hours |
| LOO CV (5 folds × 10 models) | 50 | 20-30 hours |
| **Total** | **~120** | **~80 hours** |

**Parallelization opportunities:**
- Train multiple model variants in parallel (if multi-GPU available)
- LOO CV folds can run in parallel
- ERA5 download concurrent with other tasks

---

## 12. Contact and Collaboration

**Primary Researcher:** [Name]  
**Institution:** NASA/[University]  
**Project Repository:** [GitHub URL]  
**Questions/Issues:** [GitHub Issues or email]

---

## Document Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | [Date] | Initial document | [Name] |
| 2.0 | [Date] | Updated with 3-week development plan | [Name] |

---

**END OF DOCUMENT**