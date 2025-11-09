# Sprint 5: Scope of Work and Feature Requirements

**Project:** Cloud Base Height Retrieval from Airborne Imagery  
**Document Version:** 1.0  
**Date:** January 2025  
**Duration:** 8 weeks  
**Previous Sprint:** Sprint 3/4 (Feature Engineering & Hybrid Models)

---

## Executive Summary

Sprint 5 focuses on advancing the CNN architectures and temporal modeling capabilities to improve upon the current baseline performance. With real ERA5 atmospheric data now fully integrated (R²=0.668, MAE=137m), the primary bottleneck is image feature extraction. This sprint will implement pre-trained vision models, temporal sequence processing, and advanced fusion strategies.

**Key Deliverables:**
1. Pre-trained CNN backbone (ResNet-50/ViT) achieving R² > 0.45
2. Temporal modeling framework (3-5 frame sequences)
3. Advanced feature fusion with real ERA5 data
4. Cross-flight validation assessment
5. Publication-ready results and documentation

---

## Current Baseline Performance

### Physical GBDT (Shadow Geometry + Real ERA5)
- **R² = 0.6681 ± 0.0345**
- **MAE = 0.1366 ± 0.0046 km (137 meters)**
- **RMSE = 0.2134 ± 0.0105 km**
- **Status:** Production-ready baseline

### CNN Models (Current Architecture)
- Image-only CNN: R² = 0.279, MAE = 233m
- Attention fusion: R² = 0.326, MAE = 222m
- **Gap to close:** 2.4× improvement needed to match physical baseline

### Critical Finding from Sprint 3/4
Real ERA5 processing revealed that **shadow geometry features dominate** the prediction (contributing ~95% of signal), while atmospheric features provide only marginal improvement (~5%). This validates the importance of accurate geometric feature extraction from imagery.

---

## Workspace Configuration and Data Locations

### Hardware Environment
- **GPU:** NVIDIA GTX 1070 Ti (8 GB VRAM)
- **CPU:** Multi-core (exact specs TBD)
- **RAM:** Sufficient for dataset (933 samples)
- **Storage:** Multiple drives (see below)

### Critical: Multi-Drive Data Architecture

**⚠️ IMPORTANT:** Data is distributed across multiple physical drives. All scripts must handle these paths correctly.

#### Primary Project Directory
```
/home/rylan/Documents/research/NASA/programDirectory/cloudMLPublic/
```
- Source code (`src/`, `scripts/`, `sow_outputs/`)
- Configuration files (`configs/`)
- Model outputs and logs
- Documentation (`docs/`)
- Virtual environment (`venv/`)

#### Raw Flight Data (Separate Drive)
```
/home/rylan/Documents/research/NASA/programDirectory/data/
```
Contains subdirectories for each flight:
- `30Oct24/` - F0: 501 samples (largest flight)
- `10Feb25/` - F1: 191 samples
- `23Oct24/` - F2: 105 samples
- `12Feb25/` - F3: 92 samples
- `18Feb25/` - F4: 44 samples

Each flight directory contains:
- `IRAI_*.h5` - Camera imagery (downward-looking ER-2)
- `CPL_*.hdf5` - Cloud Physics Lidar data (ground truth CBH)
- `CRS_*_nav.hdf` - Navigation data (lat/lon/altitude)

#### ERA5 Reanalysis Data (External Drive)
```
/media/rylan/two/research/NASA/ERA5_data_root/
├── surface/           # 119 daily files (era5_surface_YYYYMMDD.nc)
├── pressure_levels/   # 119 daily files (era5_pressure_YYYYMMDD.nc)
└── processed/         # Processed feature files
```

**Coverage:** October 23, 2024 - February 19, 2025 (all flight dates covered)

**Variables Available:**
- Surface: BLH, T2M, D2M, SP, TCWV
- Pressure levels (37 levels, 1000-1 hPa): T, Q, Z

**Spatial Resolution:** 0.25° × 0.25° (~25 km)  
**Temporal Resolution:** Hourly

**Access Pattern:**
- External drive must be mounted before processing
- Scripts should check for directory existence
- Fallback: Use cached processed features if ERA5 unavailable

#### Processed Features (Project Directory)
```
cloudMLPublic/sow_outputs/
├── wp1_geometric/WP1_Features.hdf5        # Shadow geometry features
├── wp2_atmospheric/
│   ├── WP2_Features.hdf5                   # Current (real ERA5)
│   ├── WP2_Features_REAL_ERA5.hdf5        # Backup
│   └── WP2_Features_SYNTHETIC_BACKUP.hdf5 # Historical (do not use)
├── integrated_features/
│   └── Integrated_Features.hdf5            # Combined feature store
└── wp3_kfold/                              # Physical baseline results
```

### Configuration File Paths

All scripts load from `configs/bestComboConfig.yaml` which specifies:
```yaml
data_directory: "/home/rylan/Documents/research/NASA/programDirectory/data/"
flights:
  - name: "10Feb25"
    iFileName: "/home/rylan/.../10Feb25/GLOVE2025_IRAI_*.h5"
    cFileName: "/home/rylan/.../10Feb25/CPL_*.hdf5"
    nFileName: "/home/rylan/.../10Feb25/CRS_20250210_nav.hdf"
  # ... (4 more flights)
```

**Important:** When creating new scripts, use config-based paths, not hardcoded paths.

---

## Sprint 5 Feature Requirements

### Priority 1: Pre-Trained CNN Architectures (Week 1-3)

#### Feature 1.1: ResNet-50 Backbone
**Objective:** Replace custom CNN with ImageNet pre-trained ResNet-50

**Technical Specifications:**
- Input: 1×440×640 grayscale images (duplicate to 3 channels for pre-trained weights)
- Architecture: ResNet-50 encoder + regression head
- Pre-training: ImageNet weights (torchvision.models.resnet50)
- Fine-tuning strategy: Freeze first 3 blocks, train last block + head
- Expected performance: R² = 0.45-0.50

**Implementation Details:**
```python
import torchvision.models as models

# Load pre-trained ResNet-50
backbone = models.resnet50(pretrained=True)
# Remove final FC layer
backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
# Add regression head
head = torch.nn.Sequential(
    torch.nn.Linear(2048, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, 1)
)
```

**Workspace Considerations:**
- VRAM usage: ~4.5 GB (fits in 8 GB GPU)
- Use mixed precision (FP16) if needed
- Gradient accumulation for effective batch size > 16

**Deliverable:**
- Script: `sow_outputs/wp5_resnet_baseline.py`
- Trained models: `sow_outputs/wp5_resnet/model_fold*.pth` (5 folds)
- Report: `sow_outputs/wp5_resnet/WP5_ResNet_Report.json`

---

#### Feature 1.2: Vision Transformer (ViT-Tiny)
**Objective:** Implement attention-based architecture for image features

**Technical Specifications:**
- Architecture: ViT-Tiny/16 (patch size 16, 12 layers, 192 dim)
- Pre-training: ImageNet-21k → ImageNet-1k fine-tuned weights
- Input handling: Resize 440×640 → 224×224 or use custom patch embedding
- Expected performance: R² = 0.48-0.55

**Why ViT over larger models:**
- ViT-Tiny fits in 8 GB VRAM with batch_size=16
- Attention mechanisms may capture cloud structure better than CNNs
- Recent work shows ViT excels on texture-rich data (clouds)

**Implementation:**
```python
from transformers import ViTForImageClassification, ViTConfig

# Load ViT-Tiny pre-trained
model = ViTForImageClassification.from_pretrained(
    'WinKawaks/vit-tiny-patch16-224',
    num_labels=1,  # Regression
    ignore_mismatched_sizes=True
)
```

**Workspace Considerations:**
- Requires `transformers` library: `pip install transformers`
- VRAM: ~3.8 GB (safe for 8 GB GPU)
- Inference time: ~50ms per image (acceptable)

**Deliverable:**
- Script: `sow_outputs/wp5_vit_baseline.py`
- Trained models: `sow_outputs/wp5_vit/model_fold*.pth`
- Report: `sow_outputs/wp5_vit/WP5_ViT_Report.json`

---

#### Feature 1.3: Mamba/S4 State Space Model (Stretch Goal)
**Objective:** Explore efficient alternative to transformers

**Technical Specifications:**
- Architecture: Mamba-Small or S4 encoder
- Advantage: Linear complexity (vs quadratic for ViT)
- Input: Flatten image patches to sequence
- Expected performance: R² = 0.42-0.50

**Workspace Considerations:**
- Requires `mamba-ssm` library (check GPU compatibility)
- VRAM: ~3.2 GB (very efficient)
- May require CUDA 11.8+ for optimal performance

**Risk Assessment:**
- Medium risk: Library may have compatibility issues
- Fallback: Skip if installation problems arise
- Alternative: Use S4 implementation from state-spaces repo

**Deliverable (if implemented):**
- Script: `sow_outputs/wp5_mamba_baseline.py`
- Report: `sow_outputs/wp5_mamba/WP5_Mamba_Report.json`

---

### Priority 2: Temporal Modeling (Week 3-5)

#### Feature 2.1: Multi-Frame Sequence Processing
**Objective:** Incorporate temporal context (cloud evolution)

**Technical Rationale:**
- Clouds evolve slowly (timescale: minutes)
- Consecutive frames provide redundancy and context
- Temporal consistency can reduce prediction noise

**Technical Specifications:**
- Input: 3-5 consecutive frames (e.g., t-2, t-1, t, t+1, t+2)
- Architecture options:
  - **Option A:** 3D CNN (ConvLSTM)
  - **Option B:** Frame-wise encoder + LSTM
  - **Option C:** Frame-wise encoder + Temporal Attention
- Expected improvement: +5-10% R²

**Dataset Modification:**
```python
class TemporalHDF5Dataset:
    def __init__(self, ..., temporal_frames=5):
        self.temporal_frames = temporal_frames
        self.temporal_offset = temporal_frames // 2
    
    def __getitem__(self, idx):
        # Load frames [idx-2, idx-1, idx, idx+1, idx+2]
        frames = []
        for offset in range(-self.temporal_offset, 
                           self.temporal_offset + 1):
            frame = self._load_frame(idx + offset)
            frames.append(frame)
        return torch.stack(frames), ...
```

**Workspace Considerations:**
- VRAM increases: 3-5× more data in memory
- Use gradient accumulation: batch_size=4 with accumulation=4
- Data loading: Pre-cache adjacent frames to avoid I/O bottleneck

**Deliverable:**
- Modified dataset: `src/temporal_hdf5_dataset.py`
- Temporal models: `sow_outputs/wp5_temporal_*.py`
- Report: Comparison of 3-frame vs 5-frame vs single-frame

---

#### Feature 2.2: Temporal Consistency Regularization
**Objective:** Penalize unrealistic frame-to-frame predictions

**Physics-Informed Loss:**
```python
def temporal_consistency_loss(pred_sequence, lambda_temporal=0.1):
    """
    Penalize large jumps in predicted CBH between consecutive frames.
    
    Physics: CBH should change smoothly (< 50m per frame at 1 Hz)
    """
    temporal_diff = pred_sequence[1:] - pred_sequence[:-1]
    temporal_penalty = torch.mean(torch.abs(temporal_diff))
    return lambda_temporal * temporal_penalty
```

**Total loss:**
```
L_total = L_mse + λ_temporal * L_temporal + λ_physical * L_physical
```

Where:
- L_mse: Standard regression loss
- L_temporal: Temporal smoothness penalty
- L_physical: Physical constraints (e.g., CBH < BLH)

**Deliverable:**
- Loss function: Add to existing training scripts
- Ablation study: λ_temporal ∈ {0.0, 0.05, 0.1, 0.2}

---

### Priority 3: Advanced Fusion with Real ERA5 (Week 4-6)

#### Feature 3.1: FiLM Conditioning on Atmospheric State
**Objective:** Modulate CNN processing based on ERA5 features

**Feature-wise Linear Modulation (FiLM):**
```python
class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(condition_dim, feature_dim)
        self.beta_fc = nn.Linear(condition_dim, feature_dim)
    
    def forward(self, x, condition):
        # x: (B, C, H, W) - CNN features
        # condition: (B, condition_dim) - ERA5 features
        gamma = self.gamma_fc(condition).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(condition).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta
```

**Integration with ResNet:**
- Insert FiLM layers after each ResNet block
- Condition on: BLH, LCL, stability index (3-5 key ERA5 features)
- Expected: Better performance in different atmospheric regimes

**Workspace Considerations:**
- ERA5 features must be loaded per-sample (already implemented)
- Path: `sow_outputs/wp2_atmospheric/WP2_Features.hdf5`
- Ensure features are correctly normalized before FiLM

**Deliverable:**
- Implementation: `sow_outputs/wp5_film_fusion.py`
- Ablation: FiLM vs no FiLM vs simple concatenation

---

#### Feature 3.2: Cross-Modal Attention (Image ↔ ERA5)
**Objective:** Learn bidirectional attention between image and atmospheric features

**Architecture:**
```python
class CrossModalAttention(nn.Module):
    def __init__(self, image_dim=2048, atmo_dim=9):
        self.query_proj = nn.Linear(image_dim, 256)
        self.key_proj = nn.Linear(atmo_dim, 256)
        self.value_proj = nn.Linear(atmo_dim, 256)
        
    def forward(self, image_features, atmo_features):
        Q = self.query_proj(image_features)  # (B, 256)
        K = self.key_proj(atmo_features)      # (B, 256)
        V = self.value_proj(atmo_features)    # (B, 256)
        
        attention = F.softmax(Q @ K.T / sqrt(256), dim=-1)
        attended = attention @ V
        return torch.cat([image_features, attended], dim=1)
```

**Research Question:**
Does the model learn to attend to different ERA5 variables under different conditions?
- High stability → attend to BLH
- Low stability → attend to moisture gradient
- Clear sky → ignore atmospheric features

**Deliverable:**
- Implementation: `sow_outputs/wp5_crossmodal_attention.py`
- Visualization: Attention weights per sample (which ERA5 vars are important?)

---

#### Feature 3.3: Ensemble Methods
**Objective:** Combine strengths of physical GBDT and best CNN

**Ensemble Strategy:**

**Option 1: Simple Weighted Average**
```python
pred_ensemble = alpha * pred_gbdt + (1 - alpha) * pred_cnn
# Optimize alpha on validation set
```

**Option 2: Stacking**
```python
# Train meta-model on GBDT + CNN predictions
meta_features = [pred_gbdt, pred_cnn, confidence_gbdt, confidence_cnn]
pred_final = meta_model.predict(meta_features)
```

**Option 3: Confidence-Weighted**
```python
# Use GBDT when shadow detection is confident
# Use CNN when shadow detection fails (NaN geometric features)
weight_gbdt = shadow_confidence
weight_cnn = 1 - shadow_confidence
pred = weight_gbdt * pred_gbdt + weight_cnn * pred_cnn
```

**Target Performance:**
- Physical GBDT: R² = 0.668
- Best CNN (ResNet-50): R² = 0.50 (estimated)
- **Ensemble Target: R² > 0.72**

**Deliverable:**
- Script: `sow_outputs/wp5_ensemble.py`
- Report: Comparison of ensemble strategies

---

### Priority 4: Cross-Flight Validation (Week 5-7)

#### Feature 4.1: Leave-One-Flight-Out CV (Revisited)
**Objective:** Test true operational deployment scenario

**Previous Result (Sprint 3/4):**
- LOO CV on simple CNN: R² = -3.13 (catastrophic failure)
- Issue: Extreme domain shift in flight F4 (18Feb25)

**New Approach:**
1. **Test with improved models:**
   - ResNet-50 (better generalization than custom CNN)
   - With real ERA5 features (atmospheric context)
   - Ensemble GBDT + CNN (robust fallback)

2. **Quantify domain shift:**
   - Mean CBH per flight: F0=0.85km, F1=0.82km, ..., F4=0.25km
   - Atmospheric differences: Stability index, BLH variation
   - Seasonal effects: October vs February flights

3. **Report realistic performance:**
   - Best case: R² > 0.3 on LOO CV (acceptable for deployment)
   - Document which flights generalize well
   - Identify failure modes (e.g., very low clouds)

**Deliverable:**
- Script: `sow_outputs/wp5_loo_validation.py`
- Report: Per-flight LOO results + domain shift analysis
- Visualization: Prediction scatter plots per held-out flight

---

#### Feature 4.2: Few-Shot Adaptation
**Objective:** Fine-tune on small number of samples from new flight

**Operational Scenario:**
New flight begins → collect first N samples with CPL → fine-tune model → deploy for remainder of flight

**Implementation:**
```python
def few_shot_adapt(pretrained_model, new_flight_samples, n_shots=10):
    """
    Fine-tune model on first n_shots from new flight.
    
    Args:
        pretrained_model: Model trained on other flights
        new_flight_samples: (images, targets) from new flight
        n_shots: Number of adaptation samples (10, 20, 50)
    """
    # Freeze backbone, train only head
    for param in pretrained_model.backbone.parameters():
        param.requires_grad = False
    
    # Fine-tune on n_shots with high learning rate
    optimizer = Adam(pretrained_model.head.parameters(), lr=1e-3)
    for epoch in range(5):  # Quick adaptation
        loss = train_epoch(pretrained_model, new_flight_samples[:n_shots])
    
    return pretrained_model
```

**Experiment:**
- Test N ∈ {5, 10, 20, 50, 100}
- Measure: R² improvement on held-out flight after adaptation
- Compare: No adaptation vs few-shot vs full fine-tuning

**Deliverable:**
- Implementation: Add to `wp5_loo_validation.py`
- Report: Adaptation curves (performance vs N shots)

---

### Priority 5: Uncertainty Quantification (Week 6-7)

#### Feature 5.1: Monte Carlo Dropout
**Objective:** Provide prediction intervals, not just point estimates

**Implementation:**
```python
def mc_dropout_predict(model, x, n_samples=50):
    """
    Run model n_samples times with dropout enabled at test time.
    
    Returns:
        mean_pred: Expected value
        std_pred: Uncertainty estimate
    """
    model.train()  # Keep dropout active
    predictions = []
    for _ in range(n_samples):
        pred = model(x)
        predictions.append(pred)
    
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)
    
    return mean_pred, std_pred
```

**Calibration:**
- 68% of true values should fall within 1σ interval
- 95% within 2σ interval
- Report calibration curves

**Deliverable:**
- Implementation: Add to all CNN models
- Visualization: Prediction intervals on test set

---

#### Feature 5.2: Conformal Prediction
**Objective:** Provide statistically valid confidence intervals

**Theory:**
Conformal prediction provides guaranteed coverage (e.g., 90% of true values within predicted interval) by calibrating on a held-out set.

**Implementation:**
```python
def conformal_calibrate(model, cal_set, alpha=0.1):
    """
    Compute conformal scores on calibration set.
    
    Returns:
        quantile: 1-alpha quantile of absolute errors
    """
    errors = []
    for x, y_true in cal_set:
        y_pred = model(x)
        errors.append(abs(y_true - y_pred))
    
    quantile = np.quantile(errors, 1 - alpha)
    return quantile

def conformal_predict(model, x, quantile):
    """
    Predict with conformal interval.
    """
    y_pred = model(x)
    return y_pred, (y_pred - quantile, y_pred + quantile)
```

**Deliverable:**
- Implementation: `sow_outputs/wp5_conformal.py`
- Report: Coverage validation (does 90% interval actually cover 90%?)

---

### Priority 6: Error Analysis and Visualization (Week 7-8)

#### Feature 6.1: Systematic Failure Mode Analysis
**Objective:** Understand when and why models fail

**Analyses to Perform:**

1. **Error vs Cloud Type:**
   - Manual labeling: Stratocumulus, Cumulus, Cirrus, Broken
   - Hypothesis: Model may fail on broken cloud fields
   
2. **Error vs Atmospheric Conditions:**
   - Correlation with: Stability index, BLH, moisture gradient
   - Hypothesis: Fails in unstable atmospheres
   
3. **Error vs Solar Geometry:**
   - Correlation with: SZA, SAA, time of day
   - Hypothesis: Fails at low solar elevation (long shadows, detection errors)
   
4. **Error vs Shadow Detection Quality:**
   - Correlation with: shadow_detection_confidence
   - Hypothesis: Geometric features fail → CNN must compensate

**Deliverable:**
- Jupyter notebook: `notebooks/error_analysis.ipynb`
- Figures: Error correlation plots for publication

---

#### Feature 6.2: Saliency Maps and Attention Visualization
**Objective:** Interpret what the CNN is looking at

**Grad-CAM Implementation:**
```python
def grad_cam(model, image, target_layer):
    """
    Generate class activation map showing which regions influence prediction.
    """
    # Forward pass
    output = model(image)
    
    # Backward pass
    output.backward()
    
    # Get gradients and activations
    gradients = target_layer.grad
    activations = target_layer.activation
    
    # Weight activations by gradients
    weights = gradients.mean(dim=(2, 3))
    cam = (weights.unsqueeze(-1).unsqueeze(-1) * activations).sum(dim=1)
    
    return cam
```

**Visualizations:**
- Overlay saliency on original images
- Compare: Does model attend to clouds? Shadows? Edges?
- Save examples of good vs bad predictions

**Deliverable:**
- Script: `sow_outputs/wp5_visualize_saliency.py`
- Figures: 20-30 example saliency maps for paper

---

### Priority 7: Advanced Data Augmentation (Week 2-8, Ongoing)

#### Feature 7.1: Physics-Informed Augmentation
**Objective:** Generate training variations that preserve physical relationships

**Augmentations to Implement:**

1. **Photometric (safe):**
   - Brightness: ±20%
   - Contrast: ±20%
   - Gamma correction: [0.8, 1.2]

2. **Geometric (requires solar angle update):**
   - Horizontal flip → SAA' = 360° - SAA
   - Rotation: ±5° (small, to avoid breaking shadow geometry)

3. **Atmospheric simulation (advanced):**
   - Add synthetic thin cirrus (semi-transparent overlay)
   - Vary haze/aerosol loading
   - Simulate different camera sensitivities

**Implementation:**
```python
class PhysicsInformedAugmentation:
    def __call__(self, image, sza, saa):
        # Apply random photometric changes
        image = self.brightness(image)
        image = self.contrast(image)
        
        # Geometric: horizontal flip with SAA correction
        if random.random() < 0.5:
            image = torch.flip(image, dims=[-1])
            saa = 360.0 - saa
        
        return image, sza, saa
```

**Deliverable:**
- Integration into existing DataLoader
- Ablation: Performance with vs without augmentation

---

## Computational Resource Planning

### GPU Memory Budget (8 GB Total)

**ResNet-50:**
- Model parameters: ~1.2 GB
- Batch size 16: ~2.8 GB activations
- Gradients + optimizer: ~1.5 GB
- **Total: ~5.5 GB** (safe margin)

**ViT-Tiny:**
- Model parameters: ~0.4 GB
- Batch size 16: ~2.2 GB activations
- Gradients + optimizer: ~0.8 GB
- **Total: ~3.4 GB** (very safe)

**Temporal Model (5 frames):**
- ResNet-50 + LSTM: ~6.8 GB
- **Mitigation:** Reduce batch size to 8, use gradient accumulation

### Training Time Estimates

**Single model, 5-fold CV:**
- ResNet-50: 2-3 hours (50 epochs × 5 folds)
- ViT-Tiny: 3-4 hours (slower due to attention)
- Temporal model: 4-6 hours (more data per sample)

**Total Sprint 5 compute:**
- ~40-60 hours GPU time (feasible on single GPU over 8 weeks)

### Disk Space Requirements

**Model checkpoints:**
- Per model: ~500 MB
- 5 folds × 4 architectures: ~10 GB

**Intermediate results:**
- Feature caches: ~2 GB
- Visualization outputs: ~1 GB

**Total: ~15 GB additional** (manageable)

---

## Risk Assessment and Mitigation

### High-Risk Items

**Risk 1: Pre-trained models don't transfer to grayscale clouds**
- **Mitigation:** Test on small subset first (100 samples)
- **Fallback:** Self-supervised pre-training on 61K unlabeled images
- **Timeline impact:** +1 week if fallback needed

**Risk 2: Temporal modeling doesn't improve performance**
- **Mitigation:** Quick ablation on single fold before full training
- **Fallback:** Skip temporal modeling, focus on better spatial features
- **Timeline impact:** -1 week if skipped

**Risk 3: Cross-flight generalization remains poor (R² < 0.2)**
- **Mitigation:** Document as fundamental limitation
- **Fallback:** Frame paper around within-flight performance + transfer learning
- **Timeline impact:** None (reframe findings)

### Medium-Risk Items

**Risk 4: VRAM insufficient for larger models**
- **Mitigation:** Mixed precision (FP16), gradient accumulation
- **Fallback:** Use smaller variants (ResNet-34, ViT-Tiny)

**Risk 5: ERA5 external drive unavailable during processing**
- **Mitigation:** Copy processed features to main drive (~74 KB)
- **Fallback:** Use cached features (already extracted)

### Low-Risk Items

**Risk 6: Library compatibility issues (Mamba, transformers)**
- **Mitigation:** Test imports before beginning implementation
- **Fallback:** Skip problematic libraries

---

## Deliverables Checklist

### Code Deliverables
- [ ] `sow_outputs/wp5_resnet_baseline.py` - ResNet-50 implementation
- [ ] `sow_outputs/wp5_vit_baseline.py` - Vision Transformer
- [ ] `sow_outputs/wp5_temporal_models.py` - Temporal sequence processing
- [ ] `sow_outputs/wp5_film_fusion.py` - FiLM conditioning
- [ ] `sow_outputs/wp5_ensemble.py` - Ensemble methods
- [ ] `sow_outputs/wp5_loo_validation.py` - Cross-flight validation
- [ ] `src/temporal_hdf5_dataset.py` - Temporal dataset loader

### Model Artifacts
- [ ] Trained ResNet-50 weights (5 folds)
- [ ] Trained ViT weights (5 folds)
- [ ] Trained temporal model weights (5 folds)
- [ ] Ensemble model weights

### Reports and Documentation
- [ ] `WP5_ResNet_Report.json` - ResNet-50 performance
- [ ] `WP5_ViT_Report.json` - ViT performance
- [ ] `WP5_Temporal_Report.json` - Temporal modeling results
- [ ] `WP5_Ensemble_Report.json` - Ensemble performance
- [ ] `WP5_LOO_Report.json` - Cross-flight validation
- [ ] `WP5_Validation_Summary.json` - Overall Sprint 5 summary

### Visualizations and Analysis
- [ ] Saliency maps (20-30 examples)
- [ ] Attention weight visualizations
- [ ] Error analysis plots
- [ ] Temporal consistency plots
- [ ] Cross-flight performance scatter plots

### Publication Materials
- [ ] Draft methods section (ResNet-50, ViT, temporal)
- [ ] Results tables (all models compared)
- [ ] Publication-quality figures (6-8 figures)
- [ ] Supplementary materials outline

---

## Success Criteria

### Minimum Viable Product (MVP)
- ✅ ResNet-50 baseline: R² > 0.40, MAE < 220m
- ✅ Temporal modeling: ΔR² > +0.05 vs single-frame
- ✅ Ensemble GBDT + CNN: R² > 0.70
- ✅ All code documented and reproducible
- ✅ LOO CV executed (document results, good or bad)

### Target Performance
- 🎯 Best single model: R² > 0.50, MAE < 200m
- 🎯 Ensemble: R² > 0.73, MAE < 130m
- 🎯 LOO CV: R² > 0.30 (acceptable for operational deployment)
- 🎯 Uncertainty quantification: 90% calibrated intervals

### Stretch Goals
- 🌟 ViT or Mamba outperforms ResNet-50 (R² > 0.55)
- 🌟 Temporal model achieves R² > 0.52 (competitive with ResNet-50)
- 🌟 Few-shot adaptation works (ΔR² > +0.15 with 20 samples)
- 🌟 Cross-modal attention reveals interpretable patterns

---

## Timeline and Milestones

### Week 1-2: Pre-Trained Baselines
- Day 1-3: ResNet-50 implementation and testing
- Day 4-7: Full 5-fold training and evaluation
- Day 8-10: ViT implementation and testing
- Day 11-14: ViT 5-fold training

**Milestone 1 (End of Week 2):** ResNet-50 and ViT baselines complete, performance documented

### Week 3-4: Temporal Modeling
- Day 15-18: Temporal dataset loader implementation
- Day 19-21: Temporal architecture testing (LSTM vs Attention)
- Day 22-28: Full temporal model training (5 folds)

**Milestone 2 (End of Week 4):** Temporal modeling complete, temporal vs single-frame comparison

### Week 5-6: Advanced Fusion and Ensemble
- Day 29-32: FiLM conditioning implementation
- Day 33-35: Cross-modal attention implementation
- Day 36-39: Ensemble methods (GBDT + best CNN)
- Day 40-42: Ensemble optimization and validation

**Milestone 3 (End of Week 6):** Ensemble model achieving R² > 0.70

### Week 7: Cross-Flight Validation
- Day 43-45: LOO CV with improved models
- Day 46-48: Few-shot adaptation experiments
- Day 49: Domain shift analysis and documentation

**Milestone 4 (End of Week 7):** Cross-flight validation complete, deployment readiness assessed

### Week 8: Analysis, Visualization, and Documentation
- Day 50-52: Error analysis and failure mode identification
- Day 53-54: Saliency maps and attention visualization
- Day 55-56: Publication figure generation
- Day 57-58: Draft methods and results sections
- Day 59-60: Final Sprint 5 report and documentation

**Final Milestone (End of Week 8):** Sprint 5 complete, publication materials ready

---

## Integration with Existing Codebase

### Files to Modify
- `src/hdf5_dataset.py` - Add temporal frame loading
- `configs/bestComboConfig.yaml` - Add new hyperparameters for Sprint 5 models

### Files to Create (New)
- `src/temporal_hdf5_dataset.py` - Temporal sequence dataset
- `src/models/resnet_baseline.py` - ResNet-50 model definition
- `src/models/vit_baseline.py` - ViT model definition
- `src/models/film_layers.py` - FiLM conditioning layers
- `src/models/temporal_models.py` - LSTM/Temporal attention models

### Directory Structure (Post-Sprint 5)
```
cloudMLPublic/
├── sow_outputs/
│   ├── wp5_resnet/
│   │   ├── model_fold0.pth
│   │   ├── ...
│   │   └── WP5_ResNet_Report.json
│   ├── wp5_vit/
│   ├── wp5_temporal/
│   ├── wp5_ensemble/
│   └── wp5_loo/
├── src/
│   ├── models/
│   │   ├── resnet_baseline.py
│   │   ├── vit_baseline.py
│   │   └── temporal_models.py
│   └── temporal_hdf5_dataset.py
└── docs/
    ├── SPRINT_5_RESULTS.md
    └── sprint_5_figures/
```

---

## Dependencies and Installation

### New Python Packages Required
```bash
# Pre-trained models
pip install timm  # PyTorch Image Models (better than torchvision)
pip install transformers  # Hugging Face (for ViT)

# Uncertainty quantification
pip install mapie  # Conformal prediction library

# Visualization
pip install grad-cam  # Saliency visualization

# Optional (stretch goals)
pip install mamba-ssm  # Mamba state-space models (if compatible)
```

### Version Compatibility
- PyTorch: >= 1.12 (current version)
- CUDA: 11.8 (for GTX 1070 Ti)
- Python: 3.12 (current version)

### Installation Verification Script
```bash
#!/bin/bash
# test_sprint5_dependencies.sh

echo "Testing Sprint 5 dependencies..."

python3 -c "import timm; print('✓ timm:', timm.__version__)"
python3 -c "import transformers; print('✓ transformers:', transformers.__version__)"
python3 -c "import torch; print('✓ PyTorch:', torch.__version__)"
python3 -c "import xgboost; print('✓ XGBoost:', xgboost.__version__)"

echo "All dependencies OK!"
```

---

## Notes for Autonomous Agent Execution

### Critical Path Constraints
1. **ERA5 external drive must be mounted** before any atmospheric feature processing
2. **Config paths must use absolute paths** for multi-drive setup
3. **Virtual environment must be activated** before running scripts
4. **GPU availability should be checked** before training (not just assumed)

### Checkpointing Strategy
- Save model checkpoints every 10 epochs
- Save intermediate results after each fold
- Enable resume-from-checkpoint if training interrupted

### Logging and Monitoring
- Log all training runs to `logs/wp5_training.log`
- Save TensorBoard logs to `outputs/tensorboard/wp5_*`
- Monitor GPU utilization: `nvidia-smi` every 5 minutes

### Error Handling
- If VRAM error: Reduce batch size by half, retry
- If ERA5 drive not found: Use cached features, log warning
- If model diverges (loss > 10): Reduce learning rate, restart from checkpoint

---

## Contact and Support

**Primary Investigator:** Rylan Malarchick  
**Email:** rylan1012@gmail.com  
**Workspace:** `/home/rylan/Documents/research/NASA/programDirectory/cloudMLPublic/`

**Data Issues:** Check drive mounts, verify paths in config  
**CUDA Issues:** GTX 1070 Ti supports CUDA 11.8, verify driver version  
**Library Issues:** Document incompatibilities, use fallback implementations

---

**Document prepared for:** Autonomous Research Agent  
**Approval required:** Before executing Sprint 5 work  
**Last updated:** January 2025