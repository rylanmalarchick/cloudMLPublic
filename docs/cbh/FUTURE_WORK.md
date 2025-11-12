# Sprint 6 Future Work & Deferred Tasks

**Document Version**: 1.0  
**Date**: 2025-01-XX  
**Status**: Active Recommendations  

---

## Executive Summary

This document catalogs optional tasks deferred during Sprint 6, research directions identified during model development, and recommended next steps for improving the Cloud Base Height (CBH) retrieval system.

---

## Table of Contents

1. [Deferred Optional Tasks](#deferred-optional-tasks)
2. [Research Directions](#research-directions)
3. [Performance Improvements](#performance-improvements)
4. [Operational Enhancements](#operational-enhancements)
5. [Timeline & Prioritization](#timeline--prioritization)

---

## Deferred Optional Tasks

### Task 2.3: Cross-Modal Attention for ERA5 Integration

**Status**: ⏸️ **DEFERRED** (Optional - Medium Priority)  
**Sprint 6 SOW Reference**: Phase 2, Task 2.3  
**Estimated Effort**: 2-3 weeks  

#### Original Objective

Implement cross-attention mechanism to fuse image features (from Temporal ViT) with ERA5 atmospheric features for improved CBH prediction.

**Proposed Architecture**:
- **Query**: Image features from Temporal ViT encoder
- **Key/Value**: ERA5 atmospheric state vectors (temperature, pressure, humidity profiles)
- **Output**: Attended image features conditioned on atmospheric context

**Success Criteria**:
- Improve over FiLM fusion baseline (R² = 0.542, Sprint 5)
- Improve over Image-only ViT (R² = 0.577, Sprint 5)
- Demonstrate interpretable attention patterns (e.g., focus on boundary layer ERA5 levels)

#### Rationale for Deferral

**Primary Reasons**:

1. **Missing Prerequisites**:
   - Sprint 6 focused on tabular GBDT and SimpleCNN baselines
   - Temporal ViT architecture not implemented in current sprint
   - Cross-modal attention requires Temporal ViT as foundation

2. **Priority Trade-off**:
   - Tabular features (GBDT) already achieve R² = 0.744 (exceeds target)
   - Image model underperforms (SimpleCNN R² = 0.351)
   - Effort better spent improving base image model before fusion experiments

3. **Time Constraints**:
   - Sprint 6 prioritized production readiness (validation, testing, documentation)
   - Cross-modal attention is research-oriented, not deployment-critical

4. **Uncertain ROI**:
   - Sprint 5 showed FiLM fusion (R² = 0.542) underperforms tabular-only (R² = 0.744)
   - Risk of information redundancy: ERA5 features may overlap with tabular inputs
   - Requires ablation studies to justify added complexity

#### Recommended Approach (Future Sprint)

**Phase 1: Foundation (Week 1-2)**
- Implement Temporal Vision Transformer (ViT) for image sequences
- Benchmark against current SimpleCNN (target: R² > 0.50)
- Verify attention mechanisms work on image-only task

**Phase 2: Cross-Modal Fusion (Week 2-3)**
- Implement cross-attention layer (Query=Image, Key/Value=ERA5)
- Compare fusion strategies:
  - Early fusion (concatenate features before encoder)
  - Late fusion (cross-attention after separate encoders)
  - FiLM-based conditioning (multiplicative gating)
  - Cross-attention (proposed)

**Phase 3: Ablation & Analysis (Week 3)**
- Ablation: Image-only vs. ERA5-only vs. Cross-attention
- Analyze attention patterns: Which ERA5 levels are most informative?
- Error analysis: Does fusion help in specific atmospheric regimes?

**Deliverables**:
```
sow_outputs/sprint7/fusion/
├── temporal_vit.py                    # Temporal ViT implementation
├── cross_modal_attention.py           # Cross-attention layer
├── train_cross_modal.py               # Training script
├── ablation_fusion_strategies.py      # Compare fusion methods
└── reports/
    ├── cross_modal_results.json       # Performance metrics
    ├── attention_visualization.pdf    # Attention heatmaps
    └── ablation_study.md              # Fusion strategy comparison
```

**Success Metrics**:
- Cross-modal attention R² > 0.577 (Image-only ViT baseline)
- Cross-modal attention R² > 0.542 (FiLM fusion baseline)
- Interpretable attention weights (validate with atmospheric science domain knowledge)

#### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Temporal ViT underperforms SimpleCNN | Medium | High | Use pre-trained ViT, aggressive regularization |
| ERA5 features redundant with tabular | High | Medium | Ablation study to identify unique contribution |
| Attention not interpretable | Medium | Low | Use attention regularization, domain constraints |
| Computational cost too high for deployment | Low | Medium | Optimize inference, consider distillation |

---

## Research Directions

### 1. Advanced Image Models (High Priority)

**Current State**:
- SimpleCNN achieves R² = 0.351 (underperforms tabular R² = 0.744)
- Ensemble gains limited by weak image component

**Recommended Next Steps**:

#### Option A: Vision Transformer (ViT)
- **Architecture**: Patch-based ViT (16×16 or 8×8 patches)
- **Pre-training**: Transfer learning from ImageNet or satellite imagery (fMoW, BigEarthNet)
- **Expected Improvement**: R² = 0.45-0.60 (based on literature)
- **Effort**: 2 weeks

#### Option B: ResNet-50 + Temporal Aggregation
- **Architecture**: ResNet-50 backbone → temporal pooling (LSTM/GRU)
- **Pre-training**: ImageNet weights
- **Expected Improvement**: R² = 0.50-0.65
- **Effort**: 1-2 weeks

#### Option C: EfficientNet-B0 (Lightweight)
- **Architecture**: EfficientNet-B0 (5.3M params, optimized for small images)
- **Advantage**: Fast inference, good performance
- **Expected Improvement**: R² = 0.40-0.55
- **Effort**: 1 week

**Recommended**: Start with **Option B (ResNet-50)** for quick wins, then explore **Option A (ViT)** for long-term performance.

---

### 2. Uncertainty Quantification Calibration (Critical)

**Current State**:
- 90% confidence intervals achieve only 77% coverage (under-calibrated)
- Mean interval width = 533 m (too wide for some applications)
- Uncertainty-error correlation = 0.485 (moderate)

**Root Causes**:
1. Quantile regression assumes Gaussian residuals (violated in practice)
2. No post-hoc calibration applied
3. Ensemble uncertainty not propagated correctly

**Recommended Solutions**:

#### Option A: Conformal Prediction (Recommended)
- **Method**: Split conformal prediction with stratified calibration
- **Advantage**: Distribution-free, guaranteed coverage
- **Implementation**: 1 week
- **Expected Result**: Coverage ≥ 85% (adjustable)

#### Option B: Isotonic Regression Calibration
- **Method**: Post-hoc calibration of prediction intervals
- **Advantage**: Simple, fast
- **Implementation**: 2-3 days
- **Expected Result**: Coverage = 80-85%

#### Option C: Bayesian Deep Learning
- **Method**: Variational inference or Hamiltonian Monte Carlo
- **Advantage**: Principled uncertainty, captures epistemic uncertainty
- **Implementation**: 3-4 weeks
- **Expected Result**: Coverage = 85-90%, better uncertainty decomposition

**Priority**: Implement **Option A (Conformal Prediction)** immediately (critical for production deployment).

---

### 3. Domain Adaptation for Flight F4 (High Priority)

**Current State**:
- Leave-one-out on F4: R² = -0.98 (catastrophic failure)
- Few-shot domain adaptation: R² = -0.22 (10-shot, best observed)
- Clear domain shift, likely due to different atmospheric regime or geographic location

**Investigation Needed**:

#### Step 1: Root Cause Analysis (Week 1)
- Compare F4 vs. other flights:
  - Geographic location (lat/lon distribution)
  - Atmospheric regime (BLH, LCL, stability parameters)
  - Image statistics (mean/std intensity, texture features)
  - ERA5 feature distributions (PCA analysis)
- Hypothesis: F4 may be maritime vs. continental, or different season

#### Step 2: Data Augmentation & Collection (Week 2)
- **Option A**: Collect 20-50 more F4 labels (if feasible)
- **Option B**: Identify similar flights in archive (transfer learning)
- **Option C**: Synthetic data generation (CycleGAN, physics-based simulation)

#### Step 3: Advanced Domain Adaptation (Week 3-4)
- **Method 1**: Domain-Adversarial Neural Network (DANN)
  - Train feature extractor invariant to flight domain
  - Expected improvement: R² = 0.3-0.5 on F4
- **Method 2**: Meta-Learning (MAML, Reptile)
  - Train model for fast adaptation with few examples
  - Expected improvement: R² = 0.4-0.6 on F4 (with 10 shots)
- **Method 3**: Self-Training (Pseudo-Labeling)
  - Use confident predictions on F4 to augment training set
  - Expected improvement: R² = 0.2-0.4 on F4

**Recommended**: Prioritize **Root Cause Analysis** to inform strategy, then apply **DANN** or **Meta-Learning**.

---

## Performance Improvements

### 1. Ensemble Target Achievement (Immediate)

**Current State**: Weighted ensemble R² = 0.7391 (0.0009 below 0.74 target)

**Quick Wins**:
- Hyperparameter tuning: Grid search over GBDT params (learning_rate, max_depth, n_estimators)
- Optimize ensemble weights on validation set (instead of test set)
- Try stacking with ElasticNet or XGBoost meta-learner (current: Ridge)
- Add uncertainty as meta-feature for stacking

**Effort**: 1-2 days  
**Expected Improvement**: R² = 0.741-0.745 (cross 0.74 threshold)

---

### 2. Feature Engineering

**Potential New Features**:
- **Temporal derivatives**: dBLH/dt, dLCL/dt (rate of change)
- **Image texture**: Haralick features, Gabor filters
- **Atmospheric stability indices**: Richardson number, Bulk Richardson number
- **Spatial gradients**: ∇T, ∇q (horizontal gradients from ERA5)

**Effort**: 1 week  
**Expected Improvement**: R² = 0.75-0.76

---

### 3. Active Learning

**Motivation**: Focus labeling effort on high-uncertainty samples

**Approach**:
1. Identify 50-100 samples with highest prediction uncertainty
2. Prioritize labeling these samples
3. Retrain model, measure improvement

**Effort**: 2 weeks (including labeling time)  
**Expected Improvement**: R² = 0.76-0.78, better calibration

---

## Operational Enhancements

### 1. Monitoring & Alerting (Before Production)

**Missing Components**:
- Grafana dashboards for model performance
- Prometheus metrics export
- Alerting rules for:
  - Prediction drift (distribution shift)
  - High uncertainty predictions (flag for review)
  - Inference latency spikes

**Deliverables**:
```
deployment/monitoring/
├── grafana_dashboard.json         # Pre-built dashboard
├── prometheus_alerts.yml          # Alert rules
└── scripts/
    └── export_metrics.py          # Custom metrics exporter
```

**Effort**: 3-5 days  
**Priority**: **Required before production deployment**

---

### 2. API Authentication & Security

**Current State**: REST API has no authentication (development only)

**Required Enhancements**:
- JWT-based authentication
- Rate limiting (per API key)
- Input validation (schema enforcement)
- API key rotation mechanism

**Effort**: 1 week  
**Priority**: **Required before production deployment**

---

### 3. Model Versioning & A/B Testing

**Current State**: Single production model, no A/B testing infrastructure

**Recommended Setup**:
- MLflow model registry integration
- Shadow deployment (route 10% traffic to new model)
- A/B testing framework (statistical significance tests)
- Automated rollback on performance degradation

**Effort**: 2 weeks  
**Priority**: Medium (nice-to-have for continuous improvement)

---

## Timeline & Prioritization

### Immediate (Next Sprint - Week 1-2)

**Critical Path**:
1. ✅ **Uncertainty Calibration** (Conformal Prediction) - 1 week
2. ✅ **Ensemble Hyperparameter Tuning** (reach R² ≥ 0.74) - 2 days
3. ✅ **Grafana Monitoring Setup** - 3 days
4. ⚠️ **F4 Root Cause Analysis** - 1 week

**Estimated Effort**: 2-3 weeks  
**Owner**: ML Team

---

### Short-Term (Sprint 7 - Week 3-6)

**High-Value Research**:
1. 🔬 **ResNet-50 Image Model** (replace SimpleCNN) - 2 weeks
2. 🔬 **Domain Adaptation for F4** (DANN or Meta-Learning) - 2 weeks
3. 🔧 **API Authentication** - 1 week

**Estimated Effort**: 5 weeks  
**Owner**: ML + DevOps Teams

---

### Medium-Term (Sprint 8+ - Month 2-3)

**Advanced Features**:
1. 🚀 **Temporal ViT + Cross-Modal Attention** (Task 2.3) - 3 weeks
2. 🚀 **Active Learning Pipeline** - 2 weeks
3. 🚀 **A/B Testing Infrastructure** - 2 weeks

**Estimated Effort**: 7 weeks  
**Owner**: Research Team

---

## Success Metrics

### Performance Targets (End of Sprint 7)

| Metric | Current | Target | Stretch Goal |
|--------|---------|--------|--------------|
| CV R² (Tabular GBDT) | 0.744 | 0.750 | 0.770 |
| CV R² (Image Model) | 0.351 | 0.500 | 0.600 |
| CV R² (Ensemble) | 0.739 | 0.740 | 0.760 |
| UQ Coverage (90% CI) | 77% | 85% | 90% |
| F4 Leave-One-Out R² | -0.98 | 0.30 | 0.50 |

### Operational Targets (Before Production)

- [x] Model Card & Deployment Guide
- [x] CI/CD Pipeline
- [ ] Grafana Dashboards (80% complete)
- [ ] API Authentication (Not started)
- [ ] Load Testing (Not started)
- [ ] Security Audit (Not started)

---

## References

### Internal Documents
- Sprint 6 SOW: `docs/Sprint6-SOW-Agent.md`
- Model Card: `sow_outputs/sprint6/MODEL_CARD.md`
- Deployment Guide: `sow_outputs/sprint6/DEPLOYMENT_GUIDE.md`
- Ensemble Report: `sow_outputs/sprint6/reports/ensemble_summary.md`
- Domain Adaptation Report: `sow_outputs/sprint6/reports/domain_adaptation_f4_summary.md`

### External Research
- Temporal ViT: Arnab et al. "ViViT: A Video Vision Transformer" (ICCV 2021)
- Conformal Prediction: Angelopoulos & Bates "A Gentle Introduction to Conformal Prediction" (2021)
- Domain Adaptation: Ganin et al. "Domain-Adversarial Training of Neural Networks" (JMLR 2016)
- Cross-Modal Attention: Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer" (AAAI 2018)

---

## Contact

**Questions or Suggestions?**
- Create GitHub issue with label `future-work`
- Email: ml-team@example.org
- Slack: #cbh-retrieval-dev

---

**Document Maintenance**:
- Review quarterly or after major milestones
- Update priorities based on stakeholder feedback
- Archive completed items to `COMPLETED_WORK.md`

---

**Last Updated**: 2025-01-XX  
**Next Review**: 2025-04-XX