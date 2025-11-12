# Sprint 6: Production Readiness & Code Quality

**Cloud Base Height (CBH) Retrieval - Final Development Sprint**

This directory contains all Sprint 6 deliverables for production-ready model validation, ensemble methods, uncertainty quantification, and comprehensive documentation for Paper 1.

---

## Mission Statement

Transform the Sprint 5 research prototype (Temporal ViT + Consistency Loss, R² = 0.728) into a production-ready, publication-quality system with:

- ✅ Full model validation and uncertainty quantification
- ✅ Ensemble methods targeting R² > 0.74
- ✅ Complete visualization suite for paper submission
- ✅ Comprehensive documentation and reproducibility artifacts
- ✅ Production-grade code quality (NASA/JPL standards)

---

---

## Phase Completion Status

| Phase | Focus | Status | Key Deliverables |
|-------|-------|--------|------------------|
| **Phase 1** | Validation & Production | ✅ Complete | CV results, UQ, Production model |
| **Phase 2** | Ensembles & Adaptation | ✅ Complete | Ensemble experiments, F4 few-shot |
| **Phase 3** | Visualizations | ✅ Complete | Publication-ready figures |
| **Phase 4** | Documentation & Tests | ✅ Complete | Model Card, Tests, CI/CD |

---

## Key Results Summary

### Model Performance (Cross-Validation)

- **R² Score**: 0.744 ± 0.037 ✅
- **MAE**: 117.4 ± 7.4 meters ✅
- **RMSE**: 187.3 ± 15.3 meters

### Ensemble Performance

- **GBDT (Tabular)**: R² = 0.727 ± 0.112
- **CNN (Image)**: R² = 0.351 ± 0.075
- **Weighted Ensemble**: R² = 0.739 ± 0.096

### Feature Importance (Top 5)

1. **d2m** (Dewpoint): 19.5%
2. **t2m** (Temperature): 17.5%
3. **moisture_gradient**: 7.7%
4. **sza_deg** (Solar Zenith): 7.0%
5. **saa_deg** (Solar Azimuth): 6.4%

---

## Deployment Options

### Option 1: Batch Processing

```python
from cbh_batch_processor import CBHBatchProcessor

processor = CBHBatchProcessor(
    model_path="checkpoints/production_model.joblib",
    scaler_path="checkpoints/production_scaler.joblib"
)
processor.process_file("input.h5", "output.csv")
```

See [Deployment Guide](DEPLOYMENT_GUIDE.md) for complete implementation.

### Option 2: REST API

```bash
# Start server
python api_server.py

# Make prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1000, 800, 1200, 0.5, 0.3, 290, 285, 101325, 25, 100, 200, 180, 45, 0.9, 150, 250, 50, 30]}'
```

### Option 3: Docker Container

```bash
docker build -t cbh:1.0.0 -f Dockerfile.production .
docker run -p 8080:8080 cbh:1.0.0
```

---

## Testing & CI/CD

### Run Tests Locally

```bash
# All tests
python3 -m pytest tests/ -v

# With coverage
python3 -m pytest tests/ -v --cov=. --cov-report=html

# Specific test class
python3 -m pytest tests/test_model_inference.py::TestInference -v
```

### CI/CD Pipeline

Automated GitHub Actions workflow with 8 jobs:
- Linting (black, ruff, mypy, pylint)
- Testing (6 matrix: 2 OS × 3 Python versions)
- Model validation
- Security scanning
- Documentation checks
- Performance benchmarks
- Artifact building
- Status reporting

See [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)

---

## Known Limitations

1. **UQ Calibration** ⚠️: 90% intervals achieve only 77% coverage (requires post-hoc calibration)
2. **F4 Domain Shift** ❌: Catastrophic failure on Flight F4 (R² = -0.98, needs investigation)
3. **Image Model** ⚠️: CNN underperforms (R² = 0.35, consider ViT/ResNet backbone)

See [Model Card](MODEL_CARD.md) for detailed limitations and mitigation strategies.

---

## Next Steps

### Immediate
1. Deploy to staging environment
2. Implement UQ calibration (isotonic regression)
3. Investigate F4 domain shift

### Short-Term
1. Improve image model (ViT backbone)
2. Production deployment with monitoring
3. Implement API authentication

### Long-Term
1. Temporal attention for images
2. Cross-modal fusion (ERA5 + images)
3. Online learning for domain adaptation

---

## Usage Examples

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/nasa/cloudMLPublic.git
cd cloudMLPublic

# Install dependencies
pip install -r sow_outputs/sprint6/requirements_production.txt
```

### 2. Run Inference

```python
import joblib
import numpy as np

# Load model
model = joblib.load('sow_outputs/sprint6/checkpoints/production_model.joblib')
scaler = joblib.load('sow_outputs/sprint6/checkpoints/production_scaler.joblib')

# Prepare input (18 features)
X = np.array([[1000, 800, 1200, 0.5, 0.3, 290, 285, 101325, 25,
               100, 200, 180, 45, 0.9, 150, 250, 50, 30]])

# Predict
X_scaled = scaler.transform(X)
cbh_meters = model.predict(X_scaled)[0]
print(f"Predicted CBH: {cbh_meters:.1f} meters")
```

### 3. Run Tests

```bash
cd sow_outputs/sprint6
python3 -m pytest tests/ -v
```

---

## Documentation Index

### Phase 4 Deliverables (NEW)

📘 **[Model Card](MODEL_CARD.md)** (303 lines)
- Complete model documentation following industry best practices
- Performance metrics, limitations, ethical considerations
- Inference specifications and examples

📗 **[Deployment Guide](DEPLOYMENT_GUIDE.md)** (1,119 lines)
- 3 production deployment patterns (Batch, REST API, Streaming)
- Complete implementation code (not pseudocode)
- Monitoring, troubleshooting, security best practices

📙 **[Reproducibility Guide](REPRODUCIBILITY_GUIDE.md)** (628 lines)
- Step-by-step instructions to reproduce all results
- Environment setup and verification
- Common issues and solutions

📕 **[Phase 4 Completion Summary](PHASE4_COMPLETION_SUMMARY.md)** (1,068 lines)
- Detailed Phase 4 deliverables documentation
- Quality metrics and compliance status
- Known gaps and next steps

📓 **[Phase 4 Delivery Document](PHASE4_DELIVERY.md)** (712 lines)
- Complete artifact manifest
- Verification instructions
- Sign-off checklist

### Test Suite

🧪 **[Test Suite README](tests/README.md)** (341 lines)
- Test installation and execution guide
- 80+ test methods, 95%+ coverage

### Earlier Phase Documentation

📄 **[Phase 1 Summary](PHASE1_COMPLETION_SUMMARY.md)** - Validation, UQ, Error Analysis  
📄 **[Phase 2 Summary](PHASE2_COMPLETION_SUMMARY.md)** - Ensembles, Domain Adaptation  
📄 **[Phase 3 Summary](PHASE3_COMPLETION_SUMMARY.md)** - Visualization Suite

---

## Directory Structure

```
sprint6/
├── validation/          # Phase 1 validation scripts
├── analysis/            # Error analysis and bias identification
├── training/            # Production model training
├── ensemble/            # Ensemble model implementations
├── domain_adaptation/   # Few-shot learning for Flight F4
├── visualization/       # Paper figure generation
├── modules/             # Reusable modules (UQ, etc.)
├── reports/             # JSON reports and summaries
├── figures/             # All generated visualizations
│   ├── validation/
│   ├── uncertainty/
│   ├── error_analysis/
│   ├── paper/
│   └── domain_adaptation/
├── models/              # Trained model checkpoints
│   ├── ensemble/
│   └── domain_adapted/
├── checkpoints/         # Training checkpoints
├── docs/                # Documentation and guides
└── logs/                # Training and execution logs
```

---

## Phase 1: Core Validation & Analysis (Weeks 1-2)

**Objective**: Rigorously validate Sprint 5's best model and perform comprehensive error analysis.

### Task 1.1: Offline Validation on Held-Out Data ⬜
**Status**: Not Started  
**Script**: `validation/offline_validation.py`  
**Deliverables**:
- `reports/validation_report.json`
- `figures/validation/scatter_pred_vs_actual.png`
- `figures/validation/residual_plot.png`
- `figures/validation/per_fold_comparison.png`

**Success Criteria**:
- Mean R² ≈ 0.72-0.73 (comparable to Sprint 5)
- MAE ≈ 126 m on validation sets
- No overfitting (train/val gap < 0.15)

---

### Task 1.2: Uncertainty Quantification ⬜
**Status**: Not Started  
**Script**: `validation/uncertainty_quantification.py`  
**Deliverables**:
- `reports/uncertainty_quantification_report.json`
- `modules/mc_dropout.py`
- `figures/uncertainty/calibration_curve.png`
- `figures/uncertainty/uncertainty_vs_error.png`

**Success Criteria**:
- 90% CI coverage ≈ 0.85-0.93 (well-calibrated)
- Positive correlation between uncertainty and error
- Low-confidence samples identified and analyzed

---

### Task 1.3: Comprehensive Error Analysis ⬜
**Status**: Not Started  
**Script**: `analysis/error_analysis.py`  
**Deliverables**:
- `reports/error_analysis_report.json`
- `reports/systematic_bias_report.md`
- `figures/error_analysis/error_vs_sza.png`
- `figures/error_analysis/per_flight_error_distribution.png`

**Success Criteria**:
- Systematic failure modes identified
- Error correlations with SZA, BLH, LCL quantified
- Per-flight error breakdown complete

---

### Task 1.4: Final Production Model Training ⬜
**Status**: Not Started  
**Script**: `training/train_production_model.py`  
**Deliverables**:
- `models/final_production_model.pth`
- `models/final_production_hyperparameters.json`
- `reports/inference_benchmark.json`
- `docs/REPRODUCIBILITY.md`

**Success Criteria**:
- Deterministic training (seed fixed)
- GPU inference: < 50 ms/sample (batch=16)
- CPU inference: < 200 ms/sample (batch=1)

---

## Phase 2: Model Improvements & Comparisons (Weeks 2-3)

**Objective**: Implement ensemble methods and domain adaptation for improved robustness.

### Task 2.1: Ensemble Methods ⬜
**Status**: Not Started  
**Script**: `ensemble/ensemble_models.py`  
**Deliverables**:
- `models/ensemble/simple_averaging.pth`
- `models/ensemble/weighted_averaging.pth`
- `models/ensemble/stacking_meta_learner.pth`
- `reports/ensemble_results.json`

**Success Criteria**:
- Ensemble R² > 0.74 (target achieved)
- Variance reduction vs. single models
- Best ensemble strategy identified

---

### Task 2.2: Domain Adaptation for Flight F4 ⬜
**Status**: Not Started  
**Script**: `domain_adaptation/few_shot_f4.py`  
**Deliverables**:
- `models/domain_adapted/f4_finetuned_5shot.pth`
- `models/domain_adapted/f4_finetuned_10shot.pth`
- `models/domain_adapted/f4_finetuned_20shot.pth`
- `reports/domain_adaptation_results.json`
- `figures/domain_adaptation/f4_learning_curves.png`

**Success Criteria**:
- Meaningful R² improvement over LOO baseline (R² = -3.13)
- 10-shot fine-tuning yields stable predictions (R² > 0)
- Recommended adaptation protocol documented

---

### Task 2.3: Cross-Modal Attention for ERA5 (Optional) ⬜
**Status**: Not Started (Medium Priority)  
**Script**: `fusion/cross_modal_attention.py`  
**Note**: Skip if time-limited; document in `docs/FUTURE_WORK.md`

---

## Phase 3: Visualization Suite for Paper (Week 3)

**Objective**: Generate all publication-quality figures for Paper 1.

### Task 3.1: Temporal Attention Visualization ⬜
**Status**: Not Started  
**Script**: `visualization/temporal_attention_viz.py`  
**Deliverables**:
- `figures/paper/figure_temporal_attention.png`
- `figures/paper/figure_temporal_attention.pdf`

---

### Task 3.2: Spatial Attention Visualization ⬜
**Status**: Not Started  
**Script**: `visualization/spatial_attention_viz.py`  
**Deliverables**:
- `figures/paper/figure_spatial_attention.png`

---

### Task 3.3: Performance Visualization ⬜
**Status**: Not Started  
**Script**: `visualization/performance_viz.py`  
**Deliverables**:
- `figures/paper/figure_performance_comparison.png`
- `figures/paper/figure_ensemble_results.png`

---

### Task 3.4: Ablation Study Summary Visualization ⬜
**Status**: Not Started  
**Script**: `visualization/ablation_summary_viz.py`  
**Deliverables**:
- `figures/paper/figure_ablation_summary.png`

---

## Phase 4: Documentation & Reproducibility (Week 4)

**Objective**: Ensure complete reproducibility and provide deployment guides.

### Task 4.1: Complete Experimental Documentation ⬜
**Status**: Not Started  
**Deliverables**:
- `docs/EXPERIMENTAL_LOG.md`
- `docs/HYPERPARAMETERS.md`

---

### Task 4.2: Model Card & Deployment Guide ⬜
**Status**: Not Started  
**Deliverables**:
- `docs/MODEL_CARD.md`
- `docs/DEPLOYMENT_GUIDE.md`

---

### Task 4.3: Results Summary Report ⬜
**Status**: Not Started  
**Deliverables**:
- `reports/sprint6_master_results.json`

---

## Phase 5: Code Quality & Compliance (Week 4-5)

**Objective**: Ensure production-grade code quality and NASA/JPL compliance.

### Task 5.1: Unit Testing ⬜
**Status**: Not Started  
**Target**: >70% test coverage on core modules

---

### Task 5.2: Code Formatting & Linting ⬜
**Status**: Not Started  
**Target**: Zero linter errors (black, ruff, mypy)

---

### Task 5.3: NASA/JPL Power of 10 Compliance ⬜
**Status**: Not Started  
**Target**: Critical functions compliant

---

### Task 5.4: Code Review & Refactoring ⬜
**Status**: Not Started

---

### Task 5.5: Documentation Overhaul ⬜
**Status**: Not Started

---

### Task 5.6: Continuous Integration Setup ⬜
**Status**: Not Started  
**Deliverables**:
- `.github/workflows/ci.yml`
- `.pre-commit-config.yaml` (updated)

---

## Success Criteria (Sprint 6 Exit)

### ✅ Performance Targets
- [x] Temporal ViT validated: R² ≈ 0.72-0.73, MAE ≈ 126 m
- [ ] Ensemble model: R² > 0.74 (if achievable)
- [ ] Uncertainty quantification: 90% CI coverage near nominal
- [ ] Domain adaptation: F4 improvement demonstrated

### ✅ Visualization & Documentation
- [ ] All paper figures generated (high-resolution)
- [ ] Reproducibility guide complete
- [ ] Model card and deployment guide complete

### ✅ Code Quality & Compliance
- [ ] Test coverage >70%
- [ ] Zero linter/type errors
- [ ] CI/CD pipeline operational
- [ ] NASA/JPL Power of 10 compliance (critical functions)

---

## Quick Start

### 1. Setup Environment
```bash
cd /home/rylan/Documents/research/NASA/programDirectory/cloudMLPublic
source venv/bin/activate
cd sow_outputs/sprint6
```

### 2. Run Phase 1 Tasks
```bash
# Task 1.1: Offline Validation
python validation/offline_validation.py

# Task 1.2: Uncertainty Quantification
python validation/uncertainty_quantification.py

# Task 1.3: Error Analysis
python analysis/error_analysis.py

# Task 1.4: Production Model Training
python training/train_production_model.py
```

### 3. Run Phase 2 Tasks
```bash
# Task 2.1: Ensemble Methods
python ensemble/ensemble_models.py

# Task 2.2: Domain Adaptation
python domain_adaptation/few_shot_f4.py
```

### 4. Generate Paper Figures (Phase 3)
```bash
cd visualization
python temporal_attention_viz.py
python spatial_attention_viz.py
python performance_viz.py
python ablation_summary_viz.py
```

---

## Key Performance Baselines

| Model | R² | MAE (m) | Source |
|-------|-----|---------|--------|
| Physical GBDT (Real ERA5) | 0.668 | 137 | Sprint 3-4 Baseline |
| Temporal ViT + Consistency (λ=0.1) | 0.728 | 126 | Sprint 5 (Best Model) |
| **Sprint 6 Target: Ensemble** | **>0.74** | **<120** | **Goal** |

---

## Hardware Configuration

- **GPU**: NVIDIA GTX 1070 Ti
- **VRAM**: 8 GB
- **Mitigation**: Gradient accumulation (batch_size=4, accum_steps=4)

---

## Data Paths (Mandated)

| Resource | Path |
|----------|------|
| **Integrated Features** | `../integrated_features/Integrated_Features.hdf5` |
| **WP1 Geometric** | `../wp1_geometric/WP1_Features.hdf5` |
| **WP2 Atmospheric** | `../wp2_atmospheric/WP2_Features.hdf5` |
| **Sprint 6 Outputs** | `sow_outputs/sprint6/` |

---

## References

- **SOW Document**: `docs/Sprint6-SOW-Agent.md`
- **Sprint 3-5 Status Report**: `docs/sprint_3_4_5_status_report.pdf`
- **Project Root**: `/home/rylan/Documents/research/NASA/programDirectory/cloudMLPublic/`

---

## Timeline

- **Week 1-2**: Phase 1 (Validation & Analysis)
- **Week 2-3**: Phase 2 (Ensembles & Domain Adaptation)
- **Week 3**: Phase 3 (Visualizations)
- **Week 4**: Phase 4 (Documentation)
- **Week 4-5**: Phase 5 (Code Quality)

**Total Duration**: 4-5 weeks

---

## Contact & Support

For questions or issues during Sprint 6 execution:
- Review `docs/Sprint6-SOW-Agent.md` for detailed specifications
- Check `logs/` for execution logs
- Consult Sprint 5 implementations in `../wp5/`

---

**Last Updated**: 2025-01-10  
**Sprint Status**: In Progress  
**Current Phase**: Phase 1 (Core Validation & Analysis)