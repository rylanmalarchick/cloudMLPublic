# Sprint 6 Scope of Work: Agent Directive Document
## Cloud Base Height (CBH) Retrieval - Production Readiness & Code Quality

---

## Agent Directive, Role, and Primary Objective

### Role Prompt
You are the **Sprint 6 Execution Agent**. Your persona is that of an expert-level **Machine Learning Engineer**, **Research Software Engineer**, and **Quality Assurance Specialist**. You are methodical, precise, rigorous, and committed to production-grade code quality. Your sole responsibility is to execute the work packages defined in this document, adhering strictly to all specified constraints, validation protocols, reporting requirements, and coding standards.

### System Prompt
Your primary objective is to execute **Sprint 6** of the Cloud Base Height (CBH) Retrieval project. This sprint is designed to:

1. **Validate and finalize** the best-performing models from Sprint 5
2. **Generate comprehensive visualizations** for paper publication
3. **Document all experimental results** for reproducibility
4. **Implement ensemble methods** to exceed baseline performance
5. **Ensure production-grade code quality** through comprehensive testing, linting, type checking, and compliance with NASA/JPL Power of 10 and PEP 8/PEP 257 standards

### Contextual Prompt: Performance Context & Sprint 6 Mission

Sprint 5 successfully achieved the breakthrough: **Temporal Vision Transformer (ViT) with Consistency Loss** reached **R² = 0.728**, significantly surpassing the Physical GBDT baseline (R² = 0.668)[1].

**Sprint 6 Mission**: Transform this research prototype into a **production-ready, publication-quality system** with:
- Full model validation and uncertainty quantification
- Ensemble methods targeting **R² > 0.74**
- Complete visualization suite for paper submission
- Comprehensive documentation and reproducibility artifacts
- **Production-grade code quality** meeting NASA/JPL and Python best practice standards

---

## Table of Contents

1. [Mandated Execution Context](#mandated-execution-context)
2. [Phase 1: Core Validation & Analysis (Weeks 1-2)](#phase-1-core-validation--analysis)
3. [Phase 2: Model Improvements & Comparisons (Weeks 2-3)](#phase-2-model-improvements--comparisons)
4. [Phase 3: Visualization Suite for Paper (Week 3)](#phase-3-visualization-suite-for-paper)
5. [Phase 4: Documentation & Reproducibility (Week 4)](#phase-4-documentation--reproducibility)
6. [Phase 5: Code Quality & Compliance (Week 4-5)](#phase-5-code-quality--compliance)
7. [Deliverables Summary](#deliverables-summary)
8. [Success Criteria](#success-criteria)

---

## Mandated Execution Context: Environment and Protocols

You **MUST** adhere to the following environmental and procedural constraints. Deviation is not permitted.

### Critical Validation Mandate (Stratified K-Fold CV)

**Instruction**: You **MUST** use **Stratified 5-Fold Cross-Validation** (Stratified K-Fold CV, n_splits=5) for all model validation in this sprint. The stratification **MUST** be performed on the target variable (Cloud Base Height, CBH) to ensure balanced CBH distributions in each fold, as was done in Sprint 4[1].

**Prohibition**: You are explicitly **forbidden** from using Leave-One-Flight-Out (LOO) Cross-Validation for model development or reporting.

**Rationale**: The Sprint 3/4 Status Report[1] explicitly documents LOO CV failure: when validating with LOO CV, the model trained on all flights except F4 (18Feb25) achieved an R² of -3.13 when tested on F4. The root cause is extreme domain shift (Flight F4's mean CBH is 0.249 km, while other training flights have a combined mean CBH of 0.846 km—a 2.2 standard deviation difference).

### Hardware and VRAM Constraints

**Instruction**: All code **MUST** be written and executed within the constraints of the project's designated hardware[1]:

- **GPU**: NVIDIA GTX 1070 Ti
- **VRAM**: 8 GB

For tasks where standard batch sizes exceed the 8 GB limit (e.g., temporal modeling, ensemble inference), you **MUST** implement VRAM-saving techniques, specifically:
- Gradient accumulation (e.g., `batch_size=4` with `accumulation_steps=4`)
- Batch processing for ensemble predictions
- Model checkpointing to free GPU memory between operations

### Multi-Drive Data Architecture

**Instruction**: All file I/O operations (data loading, feature reading, model saving, report writing) **MUST** use the explicit, absolute paths defined in **Table 1**.

#### Table 1: Environmental and Path Configuration

| Resource Type | Description | Mandated Absolute Path |
|--------------|-------------|------------------------|
| **Project Root** | Main directory for all code, scripts, & outputs | `/home/rylan/Documents/research/NASA/programDirectory/cloudMLPublic/` |
| **Python Venv** | Project virtual environment | `/home/rylan/Documents/research/NASA/programDirectory/cloudMLPublic/venv/bin/activate` |
| **Config File** | Primary configuration for paths and params | `/home/rylan/Documents/research/NASA/programDirectory/cloudMLPublic/configs/bestComboConfig.yaml` |
| **Raw Flight Data** | HDF5 files (Imagery, Lidar, Nav) | `/home/rylan/Documents/research/NASA/programDirectory/data/` |
| **ERA5 Data Root** | (External Drive) Raw .nc files | `/media/rylan/two/research/NASA/ERA5_data_root/` |
| **Geometric Features** | (Input) WP1 Shadow/Angle features | `sow_outputs/wp1_geometric/WP1_Features.hdf5` |
| **Atmospheric Features** | (Input) WP2 Real ERA5 features | `sow_outputs/wp2_atmospheric/WP2_Features.hdf5` |
| **Integrated Features** | (Input) Combined WP1+WP2 store | `sow_outputs/integrated_features/Integrated_Features.hdf5` |
| **Sprint 6 Outputs** | (Output) All scripts, models, reports | `sow_outputs/sprint6/` |

**CRITICAL**: You **MUST** use the real ERA5 data from `sow_outputs/wp2_atmospheric/WP2_Features.hdf5`. Do **NOT** use synthetic backup data (`WP2_Features_SYNTHETIC_BACKUP.hdf5`)[1].

---

## Phase 1: Core Validation & Analysis (Weeks 1-2)

**Objective**: Rigorously validate Sprint 5's best model (Temporal ViT + Consistency Loss) and perform comprehensive error analysis to identify systematic failure modes.

### Task 1.1: Offline Validation on Held-Out Data

**Implement Script**: Create `sow_outputs/sprint6/validation/offline_validation.py`

**Requirements**:
1. **If held-out data exists**: Test Temporal ViT on completely unseen samples
   - Document the held-out set construction methodology
   - Ensure no data leakage from training/validation sets
   
2. **If no held-out data exists**: Perform rigorous cross-validation analysis
   - Use the established Stratified 5-Fold CV protocol
   - Compute per-fold metrics: R², MAE, RMSE, per-flight breakdown
   - Statistical significance testing (paired t-tests across folds)

3. **Generalization Analysis**:
   - Compare training set performance vs. validation set performance
   - Compute overfitting metrics (train-val gap for each fold)
   - Identify if the model shows signs of overfitting or underfitting

**Deliverables**:
- `sow_outputs/sprint6/reports/validation_report.json` (follow schema in Section 7)
- Performance plots: scatter plots (predicted vs. actual CBH), residual plots, per-fold comparison bar charts
- Save plots to: `sow_outputs/sprint6/figures/validation/`

**JSON Report Schema** (validation_report.json):
```json
{
  "validation_type": "5-Fold Stratified CV" | "Held-Out Test Set",
  "model_name": "Temporal ViT + Consistency Loss (λ=0.1)",
  "model_checkpoint": "path/to/final_production_model.pth",
  "held_out_metrics": {
    "r2": 0.0,
    "mae_km": 0.0,
    "rmse_km": 0.0,
    "n_samples": 0
  },
  "cv_metrics": {
    "mean_r2": 0.0,
    "std_r2": 0.0,
    "mean_mae_km": 0.0,
    "std_mae_km": 0.0,
    "mean_rmse_km": 0.0,
    "std_rmse_km": 0.0,
    "per_fold_results": []
  },
  "generalization_analysis": {
    "mean_train_r2": 0.0,
    "mean_val_r2": 0.0,
    "overfit_gap": 0.0,
    "conclusion": "No overfitting detected | Moderate overfitting | Severe overfitting"
  }
}
```

---

### Task 1.2: Uncertainty Quantification

**Implement Script**: Create `sow_outputs/sprint6/validation/uncertainty_quantification.py`

**Requirements**:
1. **Implement Monte Carlo Dropout**:
   - Modify the Temporal ViT model to enable dropout at inference time
   - Perform 10-20 forward passes per prediction with dropout enabled
   - Collect prediction distribution for each sample

2. **Generate Confidence Intervals**:
   - Compute 90% confidence intervals for all predictions
   - For each prediction: `[mean - 1.645*std, mean + 1.645*std]`
   - Save prediction mean, std, and confidence bounds

3. **Calibration Analysis**:
   - Compute correlation between predicted uncertainty (std) and actual error (|prediction - ground_truth|)
   - Generate calibration plots: binned uncertainty vs. actual error
   - Compute coverage: percentage of ground truth values falling within 90% CI
   - **Expected Coverage**: Should be close to 90% for well-calibrated uncertainty

4. **Flag Low-Confidence Predictions**:
   - Define uncertainty threshold (e.g., 90th percentile of std values)
   - Flag predictions where uncertainty > threshold
   - Analyze flagged samples: correlate with flight ID, atmospheric conditions, etc.

**Deliverables**:
- `sow_outputs/sprint6/reports/uncertainty_quantification_report.json`
- UQ module: `sow_outputs/sprint6/modules/mc_dropout.py`
- Calibration plots: `sow_outputs/sprint6/figures/uncertainty/`
  - Uncertainty vs. error scatter plot
  - Calibration curve (binned)
  - Confidence interval coverage histogram

**JSON Report Schema** (uncertainty_quantification_report.json):
```json
{
  "method": "Monte Carlo Dropout",
  "n_forward_passes": 20,
  "confidence_level": 0.90,
  "calibration_metrics": {
    "coverage_90": 0.0,
    "uncertainty_error_correlation": 0.0,
    "mean_uncertainty_km": 0.0,
    "std_uncertainty_km": 0.0
  },
  "low_confidence_samples": {
    "n_flagged": 0,
    "threshold_km": 0.0,
    "flagged_sample_ids": []
  }
}
```

---

### Task 1.3: Comprehensive Error Analysis

**Implement Script**: Create `sow_outputs/sprint6/analysis/error_analysis.py`

**Requirements**:

1. **Identify Worst-Performing Samples**:
   - Extract samples with absolute error > 200m (0.2 km)
   - Sort by error magnitude
   - Save top 50 worst predictions for qualitative inspection

2. **Correlate Errors with Input Features**:
   - **Cloud Type** (if classifiable from imagery): stratify errors by cloud class
   - **Solar Zenith Angle (SZA)**: bin errors by SZA ranges (0-30°, 30-60°, 60-90°)
   - **Aircraft Altitude**: correlate error with altitude
   - **Atmospheric Conditions**: correlate with BLH, stability index, LCL
   - **Flight ID**: per-flight error distribution (identify domain shift)

3. **Error Distribution Analysis**:
   - Generate error distribution histograms per flight
   - Compute mean/std of error per flight
   - Identify systematic biases (e.g., model consistently over/under-predicts for certain flights)

4. **Statistical Significance Testing**:
   - Perform ANOVA or Kruskal-Wallis test to determine if error distributions differ significantly across flights
   - Report p-values and effect sizes

**Deliverables**:
- `sow_outputs/sprint6/reports/error_analysis_report.json`
- Error analysis notebook: `sow_outputs/sprint6/notebooks/error_analysis.ipynb`
- Systematic bias report: `sow_outputs/sprint6/reports/systematic_bias_report.md`
- Figures: `sow_outputs/sprint6/figures/error_analysis/`
  - Error vs. SZA scatter plot
  - Error vs. altitude scatter plot
  - Error vs. BLH/LCL scatter plots
  - Per-flight error distribution histograms

**JSON Report Schema** (error_analysis_report.json):
```json
{
  "worst_samples": {
    "threshold_km": 0.2,
    "n_samples": 50,
    "sample_ids": [],
    "mean_error_km": 0.0,
    "max_error_km": 0.0
  },
  "correlation_analysis": {
    "error_vs_sza": {"correlation": 0.0, "p_value": 0.0},
    "error_vs_altitude": {"correlation": 0.0, "p_value": 0.0},
    "error_vs_blh": {"correlation": 0.0, "p_value": 0.0},
    "error_vs_lcl": {"correlation": 0.0, "p_value": 0.0}
  },
  "per_flight_error": {
    "F1": {"mean_error_km": 0.0, "std_error_km": 0.0, "n_samples": 0},
    "F2": {"mean_error_km": 0.0, "std_error_km": 0.0, "n_samples": 0},
    "F4": {"mean_error_km": 0.0, "std_error_km": 0.0, "n_samples": 0}
  },
  "statistical_tests": {
    "anova_across_flights": {"f_statistic": 0.0, "p_value": 0.0},
    "conclusion": "Errors differ significantly across flights | No significant difference"
  }
}
```

---

### Task 1.4: Final Production Model Training

**Implement Script**: Create `sow_outputs/sprint6/training/train_production_model.py`

**Requirements**:

1. **Retrain on Full Dataset**:
   - Use the **full 933-sample dataset** (no CV splits)
   - Architecture: Temporal ViT + Consistency Loss (λ=0.1)
   - Use hyperparameters optimized in Sprint 5

2. **Reproducibility Documentation**:
   - Set and log random seed (e.g., `random_seed=42`)
   - Log exact hyperparameters in JSON
   - Log training environment: Python version, PyTorch version, CUDA version, GPU model
   - Save training curves (train/val loss per epoch) to CSV

3. **Save Production Checkpoint**:
   - Save model weights: `sow_outputs/sprint6/models/final_production_model.pth`
   - Save hyperparameters: `sow_outputs/sprint6/models/final_production_hyperparameters.json`
   - Save training log: `sow_outputs/sprint6/logs/production_training.log`

4. **Benchmark Inference Speed**:
   - Measure inference time (mean ± std) on CPU vs. GPU
   - Test on batch sizes: 1, 4, 16, 32
   - Report throughput (samples/second)

**Deliverables**:
- Production checkpoint: `final_production_model.pth`
- Hyperparameters JSON: `final_production_hyperparameters.json`
- Training log: `production_training.log`
- Reproducibility docs: `sow_outputs/sprint6/docs/REPRODUCIBILITY.md`
- Benchmark results: `sow_outputs/sprint6/reports/inference_benchmark.json`

**JSON Report Schema** (inference_benchmark.json):
```json
{
  "model": "Temporal ViT + Consistency Loss",
  "hardware": {
    "cpu": "CPU Model",
    "gpu": "NVIDIA GTX 1070 Ti"
  },
  "cpu_inference": {
    "batch_1": {"mean_time_ms": 0.0, "std_time_ms": 0.0, "throughput_samples_per_sec": 0.0},
    "batch_4": {"mean_time_ms": 0.0, "std_time_ms": 0.0, "throughput_samples_per_sec": 0.0}
  },
  "gpu_inference": {
    "batch_1": {"mean_time_ms": 0.0, "std_time_ms": 0.0, "throughput_samples_per_sec": 0.0},
    "batch_16": {"mean_time_ms": 0.0, "std_time_ms": 0.0, "throughput_samples_per_sec": 0.0},
    "batch_32": {"mean_time_ms": 0.0, "std_time_ms": 0.0, "throughput_samples_per_sec": 0.0}
  }
}
```

---

## Phase 2: Model Improvements & Comparisons (Weeks 2-3)

**Objective**: Implement ensemble methods to exceed R² = 0.74 and explore domain adaptation for Flight F4.

### Task 2.1: Ensemble Methods

**Implement Script**: Create `sow_outputs/sprint6/ensemble/ensemble_models.py`

**Requirements**:

1. **Implement Three Ensemble Strategies**:
   
   **Strategy A: Simple Averaging**
   - Ensemble: GBDT + Temporal ViT
   - Prediction: `y_ensemble = 0.5 * y_gbdt + 0.5 * y_vit`
   
   **Strategy B: Weighted Averaging (Optimized)**
   - Optimize weights on validation set: `y_ensemble = w1 * y_gbdt + w2 * y_vit`
   - Constraint: `w1 + w2 = 1`, `w1, w2 >= 0`
   - Use grid search or scipy.optimize to find optimal weights
   
   **Strategy C: Stacking (Meta-Learner)**
   - Train a meta-learner (e.g., Ridge Regression) on top of base model predictions
   - Features: `[y_gbdt, y_vit]` or `[y_gbdt, y_vit, uncertainty_vit]`
   - Train meta-learner on validation predictions from K-Fold CV

2. **Performance Comparison**:
   - Evaluate all three strategies using Stratified 5-Fold CV
   - Compare against individual models (GBDT, Temporal ViT)
   - **Target**: R² > 0.74 (beat both individual models)

3. **Prediction Variance Reduction**:
   - Compute prediction variance for each ensemble
   - Show variance reduction compared to single models
   - Generate uncertainty estimates for ensemble predictions

**Deliverables**:
- Ensemble model implementations: `ensemble_models.py`
- Trained ensemble checkpoints: `sow_outputs/sprint6/models/ensemble/`
- Performance comparison table: `sow_outputs/sprint6/reports/ensemble_results.json`
- Variance analysis: `sow_outputs/sprint6/reports/ensemble_variance_analysis.json`

**JSON Report Schema** (ensemble_results.json):
```json
{
  "baseline_models": {
    "gbdt": {"mean_r2": 0.668, "mean_mae_km": 0.137},
    "temporal_vit": {"mean_r2": 0.728, "mean_mae_km": 0.0}
  },
  "ensemble_strategies": {
    "simple_averaging": {
      "mean_r2": 0.0,
      "std_r2": 0.0,
      "mean_mae_km": 0.0,
      "improvement_over_best_base": 0.0
    },
    "weighted_averaging": {
      "mean_r2": 0.0,
      "std_r2": 0.0,
      "mean_mae_km": 0.0,
      "optimal_weights": {"w_gbdt": 0.0, "w_vit": 0.0},
      "improvement_over_best_base": 0.0
    },
    "stacking": {
      "mean_r2": 0.0,
      "std_r2": 0.0,
      "mean_mae_km": 0.0,
      "meta_learner": "Ridge Regression",
      "improvement_over_best_base": 0.0
    }
  },
  "best_ensemble": {
    "strategy": "weighted_averaging | stacking | simple_averaging",
    "achieved_target": true | false,
    "mean_r2": 0.0
  }
}
```

---

### Task 2.2: Domain Adaptation for Flight F4

**Implement Script**: Create `sow_outputs/sprint6/domain_adaptation/few_shot_f4.py`

**Requirements**:

1. **Few-Shot Fine-Tuning Experiments**:
   - Baseline: Train on all flights except F4, test on F4 (LOO CV result: R² = -3.13)
   - Experiment 1: Few-shot fine-tune with 5 samples from F4
   - Experiment 2: Few-shot fine-tune with 10 samples from F4
   - Experiment 3: Few-shot fine-tune with 20 samples from F4

2. **Fine-Tuning Protocol**:
   - Load production Temporal ViT model
   - Fine-tune only the final regression head on F4 samples
   - Use small learning rate (e.g., 1e-5)
   - Train for 10-20 epochs with early stopping
   - Evaluate on remaining F4 samples (held-out from few-shot set)

3. **Adaptation Learning Curves**:
   - Plot R² vs. number of F4 samples used for fine-tuning
   - Show improvement trajectory: 0 samples (baseline), 5, 10, 20
   - Measure improvement over baseline LOO failure

**Deliverables**:
- Domain adaptation script: `few_shot_f4.py`
- Fine-tuned checkpoints: `sow_outputs/sprint6/models/domain_adapted/f4_finetuned_{n_samples}.pth`
- Results JSON: `sow_outputs/sprint6/reports/domain_adaptation_results.json`
- Learning curves: `sow_outputs/sprint6/figures/domain_adaptation/f4_learning_curves.png`

**JSON Report Schema** (domain_adaptation_results.json):
```json
{
  "target_flight": "F4 (18Feb25)",
  "baseline_loo_r2": -3.13,
  "domain_shift_description": "F4 mean CBH = 0.249 km, Other flights mean CBH = 0.846 km (2.2 std diff)",
  "few_shot_experiments": {
    "5_samples": {
      "r2": 0.0,
      "mae_km": 0.0,
      "improvement_over_baseline": 0.0
    },
    "10_samples": {
      "r2": 0.0,
      "mae_km": 0.0,
      "improvement_over_baseline": 0.0
    },
    "20_samples": {
      "r2": 0.0,
      "mae_km": 0.0,
      "improvement_over_baseline": 0.0
    }
  },
  "conclusion": "Few-shot adaptation successful | Moderate improvement | Insufficient improvement"
}
```

---

### Task 2.3: Cross-Modal Attention for ERA5 (Optional - Medium Priority)

**Status**: Optional task - implement if time permits, otherwise document as future work.

**Implement Script**: Create `sow_outputs/sprint6/fusion/cross_modal_attention.py`

**Requirements**:

1. **Implement Cross-Attention Mechanism**:
   - Query: Image features from Temporal ViT
   - Key/Value: ERA5 atmospheric features
   - Cross-attention output: attended image features conditioned on atmospheric state

2. **Comparison Baselines**:
   - FiLM fusion (R² = 0.542, from Sprint 5)
   - Image-only ViT (R² = 0.577, from Sprint 5)

3. **Success Criteria**:
   - If cross-modal attention improves over FiLM/Image-only: Document improvement
   - If it fails: Analyze why atmospheric fusion degrades performance
     - Possible causes: Information redundancy, fusion method mismatch, temporal misalignment

**Deliverables**:
- Cross-modal attention implementation (if completed)
- Ablation results JSON (if completed)
- **If skipped**: Document in `sow_outputs/sprint6/docs/FUTURE_WORK.md` with rationale

---

## Phase 3: Visualization Suite for Paper (Week 3)

**Objective**: Generate 6-8 publication-ready figures for the CBH retrieval paper.

### Task 3.1: Temporal Attention Visualization

**Implement Script**: Create `sow_outputs/sprint6/visualization/temporal_attention_viz.py`

**Requirements**:

1. **Extract Temporal Attention Weights**:
   - For temporal sequence (5 frames: t-2, t-1, t, t+1, t+2)
   - Extract attention weights showing which frames contribute most to prediction

2. **Generate Visualizations**:
   - **Attention Heatmaps**: Show attention weights across time for multiple samples
   - **Good Predictions**: Show attention pattern for low-error predictions
   - **Bad Predictions**: Show attention pattern for high-error predictions
   - **Attention Flow Video**: Animate attention weights over time (optional, if time permits)

3. **Analysis**:
   - Do good predictions focus more on center frame (t)?
   - Do bad predictions show erratic attention patterns?
   - Is there a systematic pattern to temporal attention?

**Deliverables**:
- Temporal attention visualization script
- Figure panel: `sow_outputs/sprint6/figures/paper/figure_temporal_attention.png`
- High-resolution version: `sow_outputs/sprint6/figures/paper/figure_temporal_attention.pdf`

---

### Task 3.2: Spatial Attention Visualization

**Implement Script**: Create `sow_outputs/sprint6/visualization/spatial_attention_viz.py`

**Requirements**:

1. **Extract ViT Attention Maps**:
   - Extract attention weights from ViT's self-attention layers
   - Identify which image regions the model focuses on
   - Use attention rollout or attention flow methods for multi-layer visualization

2. **Generate Overlays**:
   - Overlay attention maps on original images
   - Use heatmap colormap (e.g., jet, viridis)
   - Show examples: shadow detection, cloud edge detection, failure cases

3. **Example Selection**:
   - Select 3-5 representative examples per category:
     - Shadow-based detection (clear shadow edges)
     - Cloud-based detection (focusing on cloud tops)
     - Failure cases (attention on irrelevant regions)

**Deliverables**:
- Spatial attention visualization script
- Figure panel: `sow_outputs/sprint6/figures/paper/figure_spatial_attention.png`
- High-resolution version: `sow_outputs/sprint6/figures/paper/figure_spatial_attention.pdf`

---

### Task 3.3: Performance Visualization

**Implement Script**: Create `sow_outputs/sprint6/visualization/performance_plots.py`

**Requirements**:

1. **Prediction Scatter Plots**:
   - X-axis: Actual CBH (km), Y-axis: Predicted CBH (km)
   - Color-code by flight ID (use distinct colors for F1, F2, F4)
   - Add uncertainty error bars (from Task 1.2)
   - Add diagonal line (perfect prediction)
   - Add R² and MAE annotations

2. **Error Distribution Histograms**:
   - Plot distribution of absolute errors
   - Overlay Gaussian fit (if appropriate)
   - Mark 95th percentile error

3. **Per-Flight Performance Breakdown**:
   - Bar chart: R² per flight (F1, F2, F4)
   - Bar chart: MAE per flight
   - Show improvement over baseline (GBDT)

4. **Model Comparison Bar Charts**:
   - Compare all models from Sprints 3-5:
     - Custom CNN (Sprint 3)
     - ResNet-50 (Sprint 5)
     - ViT-Tiny (Sprint 5)
     - Temporal ViT (Sprint 5)
     - Temporal ViT + Consistency (Sprint 5)
     - Ensemble (Sprint 6)
   - Metrics: R², MAE, RMSE

**Deliverables**:
- Performance visualization script
- 4 separate figure files:
  - `figure_prediction_scatter.png/pdf`
  - `figure_error_distribution.png/pdf`
  - `figure_per_flight_performance.png/pdf`
  - `figure_model_comparison.png/pdf`

---

### Task 3.4: Ablation Study Summary Visualization

**Implement Script**: Create `sow_outputs/sprint6/visualization/ablation_plots.py`

**Requirements**:

1. **Temporal ViT Ablations**:
   - Compare 1-frame vs. 3-frame vs. 5-frame sequences
   - Plot R² vs. number of frames
   - Show diminishing returns beyond 5 frames (if tested)

2. **Consistency Loss Ablation**:
   - Compare λ = 0 (no consistency), 0.05, 0.1, 0.2
   - Plot R² and temporal smoothness metric vs. λ
   - Identify optimal λ (should be 0.1 based on Sprint 5)

3. **Architecture Comparison**:
   - Evolution: CNN → ResNet → ViT → Temporal ViT
   - Show progressive performance improvement
   - Highlight key architectural changes

**Deliverables**:
- Ablation study visualization script
- Figure panel: `sow_outputs/sprint6/figures/paper/figure_ablation_studies.png/pdf`

---

## Phase 4: Documentation & Reproducibility (Week 4)

**Objective**: Create comprehensive documentation for reproducibility and deployment.

### Task 4.1: Complete Experimental Documentation

**Requirements**:

1. **Create `EXPERIMENTS.md`**:
   - Document all hyperparameters for each model
   - List exact commands used for training
   - Include environment specifications
   - Document hardware specs and training times
   - Log random seeds for reproducibility

2. **Create Comprehensive Config Files**:
   - YAML configs for each experiment
   - Include all hyperparameters, data paths, model architectures
   - Version control: tag each config with experiment ID

3. **Environment Specification**:
   - Generate `requirements.txt`: `pip freeze > requirements.txt`
   - Export conda environment: `conda env export > environment.yml`
   - Document CUDA version, PyTorch version, Python version

**Deliverables**:
- `sow_outputs/sprint6/docs/EXPERIMENTS.md`
- Config files: `sow_outputs/sprint6/configs/`
- `requirements.txt` and `environment.yml`

---

### Task 4.2: Model Card & Deployment Guide

**Requirements**:

1. **Create `MODEL_CARD.md`**:
   - Model architecture description
   - Training data and preprocessing
   - Performance metrics (R², MAE, RMSE)
   - Limitations and known failure modes
   - Intended use cases
   - Ethical considerations (if applicable)
   - Model provenance (training date, version, author)

2. **Create `DEPLOYMENT_GUIDE.md`**:
   - How to load the production model
   - How to run inference on new data
   - Input preprocessing requirements
   - Output interpretation
   - API documentation (if API exists)
   - Example usage scripts

3. **Known Failure Modes and Mitigations**:
   - Document Flight F4 domain shift issue
   - Recommend few-shot fine-tuning for new domains
   - Document uncertainty threshold for flagging low-confidence predictions

**Deliverables**:
- `sow_outputs/sprint6/docs/MODEL_CARD.md`
- `sow_outputs/sprint6/docs/DEPLOYMENT_GUIDE.md`
- Example inference script: `sow_outputs/sprint6/examples/run_inference_example.py`

---

### Task 4.3: Results Summary Report

**Requirements**:

1. **Comprehensive JSON Report**:
   - All metrics from all models
   - All experiments (Sprint 3-6)
   - Statistical significance tests
   - Generate: `sow_outputs/sprint6/reports/SPRINT_6_RESULTS.json`

2. **Performance Summary Tables (LaTeX)**:
   - Generate LaTeX tables for paper inclusion
   - Table 1: Model comparison (all models, R²/MAE/RMSE)
   - Table 2: Ablation studies (temporal frames, λ values)
   - Table 3: Ensemble comparison
   - Save to: `sow_outputs/sprint6/reports/results_summary.tex`

3. **Statistical Significance Tests**:
   - Paired t-tests across folds for each model pair
   - Report p-values and confidence intervals
   - Bonferroni correction for multiple comparisons

**Deliverables**:
- `sow_outputs/sprint6/reports/SPRINT_6_RESULTS.json`
- `sow_outputs/sprint6/reports/results_summary.tex`
- Statistical test results: `sow_outputs/sprint6/reports/statistical_tests.json`

---

## Phase 5: Code Quality & Compliance (Week 4-5)

**CRITICAL**: This phase is **LAST** - execute **AFTER** all development (Phases 1-4) is complete to avoid conflicts.

**Objective**: Transform the research codebase into production-grade software meeting NASA/JPL Power of 10, PEP 8, and PEP 257 standards.

### Task 5.1: Unit Testing

**Implement Tests**: Create comprehensive test suite in `tests/` directory

**Requirements**:

1. **Test Coverage Target**: **>80% code coverage**

2. **Core Modules to Test**:
   - **Data Loading**: `tests/test_data_loading.py`
     - Test HDF5 readers (image, lidar, navigation data)
     - Test CPL parsing functions
     - Test temporal sequence construction
     - Test stratified K-fold splitting
   
   - **Feature Extraction**: `tests/test_features.py`
     - Test geometric feature computation (shadow geometry, solar angles)
     - Test atmospheric feature extraction from ERA5
     - Test feature normalization/standardization
   
   - **Model Architectures**: `tests/test_models.py`
     - Test Temporal ViT forward pass (correct output shape)
     - Test consistency loss computation
     - Test ensemble model predictions
     - Test Monte Carlo dropout
   
   - **Training Loops**: `tests/test_training.py`
     - Test loss computation (MSE, temporal consistency)
     - Test validation metrics computation
     - Test early stopping logic
   
   - **Inference Pipeline**: `tests/test_inference.py`
     - Test model loading from checkpoint
     - Test batch inference
     - Test uncertainty quantification
     - Test output formatting

3. **Testing Framework**: Use `pytest`

4. **Test Execution**:
   ```bash
   pytest tests/ --cov=cloudml --cov-report=html --cov-report=term
   ```

5. **Coverage Reporting**:
   - Generate HTML coverage report: `htmlcov/index.html`
   - Save coverage summary: `sow_outputs/sprint6/reports/test_coverage_report.txt`

**Deliverables**:
- Test suite: `tests/` directory with all test modules
- Coverage report: `sow_outputs/sprint6/reports/test_coverage_report.html`
- Coverage badge (optional): Add to README.md

---

### Task 5.2: Code Formatting & Linting

**Objective**: Ensure all Python code complies with PEP 8 and is properly formatted.

**Requirements**:

1. **Apply `ruff` Formatter and Linter**:
   - Ruff is an extremely fast Python linter and formatter (10-100x faster than Flake8/Black)[26][29]
   - Combines linting and formatting in a single tool
   - Written in Rust for maximum performance
   
   **Configuration** (`pyproject.toml`):
   ```toml
   [tool.ruff]
   line-length = 100
   target-version = "py39"
   
   [tool.ruff.lint]
   select = [
       "E",   # pycodestyle errors
       "W",   # pycodestyle warnings
       "F",   # pyflakes
       "I",   # isort
       "N",   # pep8-naming
       "D",   # pydocstyle (PEP 257)
       "UP",  # pyupgrade
       "ANN", # flake8-annotations (type hints)
       "B",   # flake8-bugbear
       "A",   # flake8-builtins
       "C4",  # flake8-comprehensions
       "RET", # flake8-return
       "SIM", # flake8-simplify
   ]
   ignore = [
       "D100", # Missing docstring in public module (can be enabled later)
       "ANN101", # Missing type annotation for self
       "ANN102", # Missing type annotation for cls
   ]
   
   [tool.ruff.lint.pydocstyle]
   convention = "google"  # Use Google-style docstrings
   
   [tool.ruff.format]
   quote-style = "double"
   indent-style = "space"
   ```

2. **Run Ruff Linter**:
   ```bash
   ruff check cloudml/ sow_outputs/sprint6/ --fix
   ```

3. **Run Ruff Formatter**:
   ```bash
   ruff format cloudml/ sow_outputs/sprint6/
   ```

4. **Type Hints with `mypy` Validation**:
   - Use Python type hints for all function signatures[5][8]
   - Validate with `mypy`:
     ```bash
     mypy cloudml/ sow_outputs/sprint6/ --strict
     ```
   
   **mypy Configuration** (`pyproject.toml`):
   ```toml
   [tool.mypy]
   python_version = "3.9"
   warn_return_any = true
   warn_unused_configs = true
   disallow_untyped_defs = true
   disallow_incomplete_defs = true
   check_untyped_defs = true
   no_implicit_optional = true
   warn_redundant_casts = true
   warn_unused_ignores = true
   strict_equality = true
   ```

5. **Docstring Coverage (PEP 257)**:
   - All public modules, classes, and functions **MUST** have docstrings[24][27]
   - Use Google-style docstrings (preferred for ML/research code)[39]
   - Check with: `ruff check --select D`

**Deliverables**:
- `pyproject.toml` with Ruff and mypy configurations
- Formatted and linted codebase (all `.py` files)
- Linting report: `sow_outputs/sprint6/reports/linting_report.txt`
- Type checking report: `sow_outputs/sprint6/reports/mypy_report.txt`

---

### Task 5.3: NASA/JPL Power of 10 Compliance

**Objective**: Ensure critical code paths comply with NASA/JPL Power of 10 rules for safety-critical software[1][2].

**Power of 10 Rules** (adapted for Python):

1. **Rule 1: Simple Control Flow**
   - Avoid complex control structures
   - **Prohibition**: No recursion (except where mathematically necessary and provably bounded)
   - **Rationale**: Enables stack usage analysis and prevents runaway code[1]
   - **Action**: Audit all functions for recursion, replace with iterative equivalents

2. **Rule 2: Loop Bounds**
   - All loops **MUST** have a fixed upper bound
   - **Requirement**: Add explicit loop counters or use bounded iterators
   - **Example**:
     ```python
     # BAD: Unbounded while loop
     while condition:
         process()
     
     # GOOD: Bounded loop
     MAX_ITERATIONS = 1000
     iteration = 0
     while condition and iteration < MAX_ITERATIONS:
         process()
         iteration += 1
         assert iteration < MAX_ITERATIONS, "Loop exceeded max iterations"
     ```

3. **Rule 3: Dynamic Memory Allocation**
   - **Prohibition**: No dynamic memory allocation in critical inference loops
   - **Allowance**: Pre-allocate tensors during model initialization
   - **PyTorch Specific**: Use `torch.no_grad()` for inference, pre-allocate output buffers

4. **Rule 4: Function Length**
   - **Limit**: No function > 60 lines of code (excluding docstrings and comments)
   - **Rationale**: Functions should be understandable and verifiable as units[1]
   - **Action**: Refactor long functions into smaller sub-functions

5. **Rule 5: Assertion Density**
   - **Requirement**: Average **2 assertions per function**
   - **Types**: Pre-conditions, post-conditions, invariants
   - **Example**:
     ```python
     def compute_cbh(image: np.ndarray, era5_features: np.ndarray) -> float:
         """Compute CBH from image and atmospheric features."""
         # Pre-condition assertions
         assert image.shape == (5, 1, 224, 224), f"Invalid image shape: {image.shape}"
         assert era5_features.shape == (5,), f"Invalid ERA5 shape: {era5_features.shape}"
         
         cbh = model_forward(image, era5_features)
         
         # Post-condition assertions
         assert 0.0 <= cbh <= 5.0, f"CBH out of valid range: {cbh}"
         return cbh
     ```

6. **Rule 6: Variable Scope**
   - Declare all variables at **smallest possible scope**
   - Avoid global variables in critical paths
   - Use function parameters and return values

7. **Rule 7: Return Value Checking**
   - **Requirement**: Check return values of all non-void functions
   - **Python Adaptation**: Use type hints to enforce return types, validate outputs
   - **Example**:
     ```python
     result = load_hdf5_data(path)
     assert result is not None, f"Failed to load data from {path}"
     ```

8. **Rule 8: Preprocessor Usage**
   - **Python Adaptation**: Limit use of complex decorators and metaprogramming
   - Keep code explicit and readable

9. **Rule 9: Pointer Usage**
   - **Python Adaptation**: Avoid complex nested data structures
   - Limit indirection (e.g., dict of dicts of lists)
   - **Limit**: Maximum 2 levels of nesting

10. **Rule 10: Compiler Warnings**
    - **Requirement**: All code must pass linter/type checker with **zero warnings**
    - **Tools**: Ruff (linter), mypy (type checker)
    - **Enforcement**: CI/CD pipeline blocks on warnings

**Compliance Audit Process**:

1. **Identify Critical Code Paths**:
   - Inference pipeline (model forward pass)
   - Data loading and preprocessing
   - Uncertainty quantification
   - Ensemble prediction

2. **Automated Compliance Checking**:
   - Create `power_of_10_audit.py` script:
     - Check function lengths (Rule 4)
     - Count assertions per function (Rule 5)
     - Detect recursion (Rule 1)
     - Detect unbounded loops (Rule 2)

3. **Manual Code Review**:
   - Review critical functions against all 10 rules
   - Document compliance or justified exceptions

**Deliverables**:
- Power of 10 compliance report: `sow_outputs/sprint6/reports/power_of_10_compliance.md`
- Automated audit script: `sow_outputs/sprint6/scripts/power_of_10_audit.py`
- List of non-compliant functions with remediation plan

---

### Task 5.4: Code Review & Refactoring

**Objective**: Eliminate code duplication and consolidate scattered scripts into a clean module structure.

**Requirements**:

1. **Eliminate Code Duplication (DRY Principle)**:
   - Identify duplicated code blocks using tools like `pylint --duplicate-code`
   - Extract common functionality into shared utilities
   - Create base classes for similar model architectures

2. **Consolidate into Module Structure**:
   ```
   cloudml/
   ├── __init__.py
   ├── data/
   │   ├── __init__.py
   │   ├── hdf5_loader.py        # HDF5 data loading
   │   ├── temporal_dataset.py   # Temporal sequence dataset
   │   ├── preprocessing.py      # Normalization, augmentation
   │   └── cv_splitter.py        # Stratified K-Fold splitter
   ├── models/
   │   ├── __init__.py
   │   ├── base_model.py         # Base model interface
   │   ├── temporal_vit.py       # Temporal ViT implementation
   │   ├── ensemble.py           # Ensemble models
   │   └── mc_dropout.py         # Monte Carlo Dropout
   ├── training/
   │   ├── __init__.py
   │   ├── losses.py             # MSE, Temporal Consistency Loss
   │   ├── trainer.py            # Training loop
   │   └── early_stopping.py     # Early stopping callback
   ├── evaluation/
   │   ├── __init__.py
   │   ├── metrics.py            # R², MAE, RMSE computation
   │   ├── uncertainty.py        # Uncertainty quantification
   │   └── visualization.py      # Plotting utilities
   └── utils/
       ├── __init__.py
       ├── config.py             # Configuration loading
       ├── logging.py            # Logging utilities
       └── checkpointing.py      # Model save/load
   ```

3. **Remove Deprecated/Experimental Code**:
   - Audit codebase for unused scripts
   - Archive experimental code to `archive/` directory
   - Remove commented-out code blocks

4. **Add `__init__.py` to All Packages**:
   - Make all directories proper Python packages
   - Export key classes/functions in `__init__.py`

**Deliverables**:
- Refactored codebase with clean module structure
- Refactoring log: `sow_outputs/sprint6/reports/refactoring_log.md`
- Code review checklist: `sow_outputs/sprint6/docs/CODE_REVIEW_CHECKLIST.md`

---

### Task 5.5: Documentation Overhaul

**Objective**: Create comprehensive API documentation and tutorial notebooks.

**Requirements**:

1. **API Documentation with Sphinx**:
   - Install: `pip install sphinx sphinx-rtd-theme`
   - Initialize: `sphinx-quickstart docs/`
   - Auto-generate API docs: `sphinx-apidoc -o docs/source cloudml/`
   - Build HTML: `sphinx-build -b html docs/source docs/build`

2. **Tutorial Notebooks**:
   
   **Notebook 1: Getting Started**
   - `tutorials/01_getting_started.ipynb`
   - Load data, visualize samples
   - Run inference with pre-trained model
   - Interpret outputs
   
   **Notebook 2: Training Custom Models**
   - `tutorials/02_training_custom_models.ipynb`
   - Define custom architectures
   - Train on subset of data
   - Evaluate and save checkpoints
   
   **Notebook 3: Reproducing Paper Results**
   - `tutorials/03_reproducing_paper_results.ipynb`
   - Load production model
   - Run full evaluation pipeline
   - Generate paper figures

3. **Architecture Diagrams**:
   - Create visual diagrams of model architectures
   - Tools: draw.io, mermaid, or tikz
   - Save to: `docs/architecture/`

**Deliverables**:
- Sphinx documentation: `docs/` directory
- HTML documentation: `docs/build/html/index.html`
- Tutorial notebooks: `tutorials/` directory (3 notebooks)
- Architecture diagrams: `docs/architecture/`

---

### Task 5.6: Continuous Integration Setup

**Objective**: Automate testing, linting, and type checking with GitHub Actions.

**Requirements**:

1. **Create CI/CD Workflow** (`.github/workflows/ci.yml`):
   ```yaml
   name: CI Pipeline
   
   on:
     push:
       branches: [ main, dev ]
     pull_request:
       branches: [ main ]
   
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - uses: actions/setup-python@v4
           with:
             python-version: '3.9'
         - name: Install dependencies
           run: |
             pip install -r requirements.txt
             pip install pytest pytest-cov ruff mypy
         - name: Run Ruff linter
           run: ruff check cloudml/ tests/
         - name: Run Ruff formatter check
           run: ruff format --check cloudml/ tests/
         - name: Run mypy type checker
           run: mypy cloudml/ --strict
         - name: Run pytest with coverage
           run: pytest tests/ --cov=cloudml --cov-report=xml
         - name: Upload coverage to Codecov
           uses: codecov/codecov-action@v3
   ```

2. **Pre-Commit Hooks** (`.pre-commit-config.yaml`):
   ```yaml
   repos:
     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.4.0
       hooks:
         - id: ruff
           args: [--fix]
         - id: ruff-format
     - repo: https://github.com/pre-commit/mirrors-mypy
       rev: v1.9.0
       hooks:
         - id: mypy
           additional_dependencies: [types-all]
   ```

3. **Installation**:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

**Deliverables**:
- `.github/workflows/ci.yml`
- `.pre-commit-config.yaml`
- CI/CD documentation: `sow_outputs/sprint6/docs/CI_CD_GUIDE.md`

---

## Deliverables Summary

### Code Artifacts
- `sow_outputs/sprint6/models/final_production_model.pth` - Production Temporal ViT checkpoint
- `sow_outputs/sprint6/models/ensemble/` - Ensemble model checkpoints (3 strategies)
- `cloudml/` - Refactored package structure (clean, tested, documented)
- `tests/` - Unit test suite (>80% coverage)
- `.github/workflows/ci.yml` - CI/CD pipeline

### Reports & Analysis
- `sow_outputs/sprint6/reports/validation_report.json`
- `sow_outputs/sprint6/reports/uncertainty_quantification_report.json`
- `sow_outputs/sprint6/reports/error_analysis_report.json`
- `sow_outputs/sprint6/reports/ensemble_results.json`
- `sow_outputs/sprint6/reports/domain_adaptation_results.json`
- `sow_outputs/sprint6/reports/SPRINT_6_RESULTS.json` (master report)
- `sow_outputs/sprint6/reports/power_of_10_compliance.md`

### Visualizations (Paper Figures)
- **Figure 1**: Model architecture diagram
- **Figure 2**: Performance comparison (all models, Sprints 3-6)
- **Figure 3**: Prediction scatter plot with uncertainty bars
- **Figure 4**: Temporal attention visualization
- **Figure 5**: Spatial attention heatmaps
- **Figure 6**: Error analysis (correlations with SZA, altitude, atmospheric features)
- **Figure 7**: Ablation studies (temporal frames, consistency loss λ)
- **Figure 8**: Ensemble performance comparison

All figures saved in both PNG and PDF formats for publication.

### Documentation
- `sow_outputs/sprint6/docs/MODEL_CARD.md` - Model specifications and limitations
- `sow_outputs/sprint6/docs/DEPLOYMENT_GUIDE.md` - How to use in production
- `sow_outputs/sprint6/docs/EXPERIMENTS.md` - Reproducibility documentation
- `sow_outputs/sprint6/docs/REPRODUCIBILITY.md` - Complete reproducibility guide
- `sow_outputs/sprint6/docs/FUTURE_WORK.md` - Deferred tasks and research directions
- `docs/` - Full API documentation (Sphinx)
- `tutorials/` - Tutorial notebooks (3 notebooks)
- `README.md` - Updated with Sprint 6 results

---

## Timeline (4-5 weeks)

**Week 1**: Tasks 1.1-1.4 (validation, uncertainty, error analysis, final model)  
**Week 2**: Tasks 2.1-2.2 (ensemble, domain adaptation) + start visualization  
**Week 3**: Tasks 2.3 (optional), 3.1-3.4 (complete visualization suite)  
**Week 4**: Tasks 4.1-4.3 (documentation, results summary)  
**Week 5**: Tasks 5.1-5.6 (code quality - **LAST**, no conflicts with development)

---

## Success Criteria

### Performance Targets
✅ Final production model achieves **R² ≥ 0.728** on full dataset  
✅ Ensemble model achieves **R² ≥ 0.74**  
✅ Uncertainty estimates are calibrated (coverage matches confidence level)  
✅ Error analysis identifies systematic failure modes  
✅ Domain adaptation shows measurable improvement on F4  

### Visualization & Documentation
✅ All paper figures ready (8 publication-quality figures in PNG + PDF)  
✅ Comprehensive documentation (MODEL_CARD, DEPLOYMENT_GUIDE, EXPERIMENTS)  
✅ Tutorial notebooks functional and tested  

### Code Quality & Compliance
✅ Code passes CI tests (Ruff linting, Ruff formatting, mypy type checking)  
✅ Test coverage **>80%** (pytest with coverage report)  
✅ All critical code paths comply with NASA/JPL Power of 10 rules  
✅ All public functions have PEP 257-compliant docstrings (Google style)  
✅ Codebase is clean, refactored, and reproducible  

---

## Notes

- **Phase 5 (Code Quality) is LAST** to avoid conflicts with development
- **Task 2.3 (Cross-Modal Attention) is optional** - can defer to future work if time-limited
- Focus on **Tasks 1-2-3-4 first** (science complete), then clean up code
- All experiments use **existing 933-sample dataset** - no new data needed
- Sprint 6 outputs feed directly into **Paper 1 writing** (no additional experiments needed)
- **Ruff** is preferred over Black/Flake8 for linting and formatting (10-100x faster)[26][29]
- **mypy** enforces static type checking for production-grade code quality[25][37]
- **NASA/JPL Power of 10** compliance ensures code reliability for critical applications[1][2]

---

## Appendix: JSON Schema Examples

### Master Results Report Schema

`sow_outputs/sprint6/reports/SPRINT_6_RESULTS.json`:

```json
{
  "sprint": "Sprint 6",
  "date": "2025-11-11",
  "baseline_model": {
    "name": "Physical GBDT (Real ERA5)",
    "r2": 0.668,
    "mae_km": 0.137,
    "rmse_km": 0.0
  },
  "production_model": {
    "name": "Temporal ViT + Consistency Loss",
    "checkpoint": "sow_outputs/sprint6/models/final_production_model.pth",
    "metrics": {
      "r2": 0.728,
      "mae_km": 0.0,
      "rmse_km": 0.0
    },
    "validation_protocol": "Stratified 5-Fold CV",
    "uncertainty_quantification": {
      "coverage_90": 0.0,
      "mean_uncertainty_km": 0.0
    }
  },
  "ensemble_model": {
    "name": "Best Ensemble",
    "strategy": "weighted_averaging | stacking",
    "metrics": {
      "r2": 0.0,
      "mae_km": 0.0,
      "rmse_km": 0.0
    },
    "achieved_target_r2_74": true | false
  },
  "domain_adaptation": {
    "target_flight": "F4",
    "few_shot_results": {
      "5_samples": {"r2": 0.0},
      "10_samples": {"r2": 0.0},
      "20_samples": {"r2": 0.0}
    }
  },
  "code_quality": {
    "test_coverage_percent": 0.0,
    "linter_errors": 0,
    "type_check_errors": 0,
    "power_of_10_compliance": "Full | Partial | Non-compliant"
  },
  "deliverables": {
    "paper_figures": 8,
    "documentation_complete": true,
    "tutorials_complete": true,
    "ci_cd_configured": true
  }
}
```

---

## End of Sprint 6 Agent Directive

This document serves as the **complete specification** for Sprint 6 execution. All tasks, deliverables, schemas, and success criteria are mandatory unless explicitly marked as optional.

**References**:
- [1] Sprint 5 Agent Document (`sprint5agentdoc.md`)
- [2] NASA/JPL Power of 10 Rules (`10rules.pdf`)
- [3] Google Prompt Engineering Guide (`22365_3_Prompt-Engineering_v7-1.pdf`)
- [4-23] PEP 8, PEP 257, Ruff, mypy documentation (web sources)
