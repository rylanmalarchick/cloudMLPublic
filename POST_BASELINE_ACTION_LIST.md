# Post-Baseline Action List

**Status:** Baseline training in progress (memory_optimized=true, batch_size=16)  
**Goal:** Make this a standout undergraduate research paper  
**Timeline:** Implement in phases after baseline completes

---

## Phase 1: Performance Optimization (1-2 days)
*Goal: Faster training + potentially better model*

### 1.1 Add torch.compile() Optimization
- [ ] Edit `src/pipeline.py` - add compile wrapper after model creation
- [ ] Test with `mode='reduce-overhead'` (best for small batches)
- [ ] Expected: 20-30% speedup, 10-15% memory reduction
- [ ] Try full model (`memory_optimized=false`) with `batch_size=20`
- [ ] Document performance comparison in logs

**Code location:** `src/pipeline.py` lines ~80-85, ~200-205, ~405-410

**Expected impact:** Faster training, possibly better results with larger model

---

### 1.2 Review Baseline Results
- [ ] Extract final metrics: MAE, RMSE, R², per-flight breakdown
- [ ] Plot training curves (train/val loss over epochs)
- [ ] Check GPU memory peak usage (should be ~7-8GB)
- [ ] Record actual training time
- [ ] Review attention maps and predictions
- [ ] Identify patterns in errors

**Deliverable:** Summary document with baseline performance

---

## Phase 2: Statistical Rigor (1 day)
*Goal: Proper statistical testing for paper*

### 2.1 Add Statistical Significance Testing
- [ ] Implement paired t-tests for model comparisons
- [ ] Add bootstrap confidence intervals (95% CI)
- [ ] Bonferroni correction for multiple ablation comparisons
- [ ] Per-flight statistical tests
- [ ] Document p-values in results tables

**New file:** `src/statistical_tests.py`

**Code snippet:**
```python
from scipy.stats import ttest_rel
from sklearn.utils import resample

def compare_models(errors_a, errors_b):
    t_stat, p_value = ttest_rel(errors_a, errors_b)
    return p_value

def bootstrap_ci(errors, n_bootstrap=1000, ci=0.95):
    means = [np.mean(resample(errors)) for _ in range(n_bootstrap)]
    return np.percentile(means, [(1-ci)/2*100, (1+ci)/2*100])
```

---

### 2.2 Add Uncertainty Quantification
- [ ] Implement MC Dropout (easiest, 10 min implementation)
- [ ] Generate prediction intervals for all predictions
- [ ] Calculate coverage (% of true values in intervals)
- [ ] Visualize uncertainty vs error correlation
- [ ] Add to final plots (prediction ± uncertainty bars)

**New file:** `src/uncertainty.py`

**Code snippet:**
```python
def mc_dropout_predict(model, x, n_samples=20):
    model.train()  # Keep dropout active
    predictions = [model(x).detach().cpu().numpy() for _ in range(n_samples)]
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    return mean, std
```

**Expected impact:** "We provide 90% prediction intervals with X% coverage"

---

## Phase 3: Error Analysis (1-2 days)
*Goal: Understand when/why model fails*

### 3.1 Stratified Error Analysis
- [ ] Error distribution by cloud type (if available in metadata)
- [ ] Error vs altitude (CBH range stratification)
- [ ] Error vs sun angle (SZA/SAA bins)
- [ ] Error vs optical depth (if available)
- [ ] Temporal patterns (time of day, season)
- [ ] Identify worst 10% predictions - qualitative analysis

**New file:** `src/error_analysis.py`

**Plots to generate:**
- Error heatmap: SZA vs SAA
- Error distribution by altitude bins
- Scatter: predicted vs actual (colored by error magnitude)
- Worst predictions: show images + attention maps

---

### 3.2 Attention Pattern Analysis
- [ ] Visualize attention maps for correct vs incorrect predictions
- [ ] Calculate attention entropy (focused vs diffuse)
- [ ] Compare attention patterns across cloud types
- [ ] Statistical test: attention entropy vs prediction error
- [ ] Qualitative examples in paper

**Add to:** `src/plot_saved_results.py`

---

## Phase 4: Self-Supervised Pretraining (3-5 days)
*Goal: Better representations from all unlabeled data*

### 4.1 Implement Contrastive Learning
- [ ] Create augmentation pairs (same image, different augs)
- [ ] Implement contrastive loss (SimCLR or MoCo)
- [ ] Pretrain encoder on ALL flights (no labels needed)
- [ ] Fine-tune on supervised CBH task
- [ ] Compare: scratch vs pretrained vs current approach

**New file:** `src/self_supervised.py`

**Alternative (simpler):**
- [ ] Masked autoencoding (mask 30% of patches, reconstruct)
- [ ] Pretraining objective: reconstruction loss
- [ ] Then load encoder weights for CBH task

**Expected impact:** Better feature representations, especially for limited labeled data

---

### 4.2 Learning Curves
- [ ] Train on 10%, 25%, 50%, 75%, 100% of data
- [ ] Plot performance vs data size
- [ ] Answer: Are we data-limited or model-limited?
- [ ] Document sample efficiency in paper

**Add to:** `src/pipeline.py` (new function `run_data_efficiency_study`)

---

## Phase 5: Baselines & Robustness (2-3 days)
*Goal: Show DL is necessary, test generalization*

### 5.1 Simple Baseline Comparisons
- [ ] Linear regression on hand-crafted features
- [ ] Random Forest on engineered features
- [ ] XGBoost baseline
- [ ] Simple CNN (already have this in ablations)
- [ ] Document in results table

**Feature engineering for baselines:**
- Mean/std/skew of pixel intensities
- Histogram features
- Texture features (GLCM)
- SZA, SAA raw values

**New file:** `src/baseline_models.py`

---

### 5.2 Robustness Tests
- [ ] Temporal split: train on early flights, test on late flights
- [ ] Noise injection: add Gaussian noise to images, measure degradation
- [ ] Per-flight generalization: LOO results (already have framework)
- [ ] Distribution shift analysis
- [ ] Document robustness in paper

**Add to:** `src/pipeline.py` or new `src/robustness_tests.py`

---

## Phase 6: Interpretability (1-2 days)
*Goal: Understand what model learns*

### 6.1 SHAP Values for Scalar Features
- [ ] Install shap: `pip install shap`
- [ ] Calculate SHAP values for SZA, SAA features
- [ ] Visualize feature importance
- [ ] Compare across different flights/cloud types

**New file:** `src/interpretability.py`

**Code snippet:**
```python
import shap
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(test_data)
shap.summary_plot(shap_values, test_data)
```

---

### 6.2 GradCAM for Spatial Attention
- [ ] Implement GradCAM for CNN layers
- [ ] Visualize what spatial regions drive predictions
- [ ] Compare to attention maps
- [ ] Qualitative examples in paper

---

## Phase 7: Paper-Specific Enhancements (ongoing)

### 7.1 Connect to Future Work (Quantum Computing)
- [ ] Add signal processing perspective to error analysis
  - FFT of errors: white noise vs systematic?
  - Power spectral density
  - Autocorrelation analysis
- [ ] Frame as control problem in discussion
  - Feedback loop: observation → prediction → decision
  - Stability analysis
- [ ] Emphasize uncertainty quantification (critical for quantum)
- [ ] Discuss decision-making under uncertainty

**Paper sections to add:**
- "Signal Processing Analysis" subsection in Results
- "Connections to Quantum Control" in Discussion/Future Work

---

### 7.2 Comprehensive Documentation
- [ ] Update README with all new features
- [ ] Document statistical tests in methodology
- [ ] Add uncertainty quantification to methods
- [ ] Create supplementary materials document
- [ ] Prepare code release (clean, documented)

---

## Priority Rankings

### 🔥 CRITICAL (Will get asked in interviews/reviews)
1. Statistical significance testing (p-values, CI)
2. Uncertainty quantification (MC dropout)
3. Error analysis (when/why fails)
4. Simple baseline comparisons

### ⭐ HIGH VALUE (Shows sophistication)
5. torch.compile() optimization
6. Learning curves (data efficiency)
7. Self-supervised pretraining
8. Attention pattern analysis

### 💡 NICE TO HAVE (Bonus points)
9. SHAP interpretability
10. Robustness tests
11. GradCAM visualization
12. Signal processing analysis (for quantum connection)

---

## Timeline Estimate

- **Week 1:** Phase 1 (optimization) + Phase 2 (statistics)
- **Week 2:** Phase 3 (error analysis) + Phase 5 (baselines)
- **Week 3:** Phase 4 (self-supervised) + Phase 6 (interpretability)
- **Week 4:** Phase 7 (paper writing + polish)

**Total:** 4 weeks to publication-ready

---

## Success Metrics

After implementing these:
- [ ] Can answer: "Is improvement statistically significant?" (p-values)
- [ ] Can answer: "How confident are predictions?" (uncertainty intervals)
- [ ] Can answer: "When does it fail?" (error analysis)
- [ ] Can answer: "Why deep learning?" (baseline comparisons)
- [ ] Can answer: "How much data needed?" (learning curves)
- [ ] Can demonstrate: Understanding beyond "it works"

---

## Deliverables for Paper

1. **Main Results Table:**
   - Model performance with CI and p-values
   - Comparison to baselines
   - Ablation studies with statistical tests

2. **Figures:**
   - Training curves
   - Prediction scatter (with uncertainty bars)
   - Error analysis heatmaps
   - Learning curves
   - Attention visualizations
   - SHAP feature importance

3. **Supplementary:**
   - Per-flight detailed results
   - Robustness test results
   - Hyperparameter sensitivity
   - Additional visualizations

---

## Notes

- Start with Phase 1 & 2 (quick wins, foundational)
- Phase 4 (self-supervised) can run in parallel (long training)
- Keep baseline model for comparisons
- Document everything as you go
- Git commit after each major feature
- Test each addition on small subset first

---

## Quick Reference: File Modifications

**New files to create:**
- `src/statistical_tests.py`
- `src/uncertainty.py`
- `src/error_analysis.py`
- `src/self_supervised.py`
- `src/baseline_models.py`
- `src/robustness_tests.py`
- `src/interpretability.py`

**Files to modify:**
- `src/pipeline.py` (add torch.compile, learning curves)
- `src/plot_saved_results.py` (add error analysis plots)
- `src/train_model.py` (add uncertainty collection)
- `colab_training.ipynb` (add new experiment cells)

**Configs to add:**
- `configs/self_supervised_pretrain.yaml`
- `configs/baseline_comparison.yaml`

---

## Contact for Help

- torch.compile issues: PyTorch forums
- Statistical tests: Scipy documentation
- Self-supervised: SimCLR paper, PyTorch Lightning examples
- Interpretability: SHAP documentation, Captum library

**This is your roadmap to a standout paper. Let's go! 🚀**