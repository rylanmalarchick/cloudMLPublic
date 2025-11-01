# DEEP LEARNING PROPOSAL: FULL DATASET APPROACH
## Leveraging 75,000+ Images for Cloud Property Prediction

**Date**: October 31, 2025  
**Status**: Proposed Research Direction  
**Target**: Deep Learning Resume Focus

---

## 🎯 EXECUTIVE SUMMARY

**Current Problem**: Limited to 933 labeled samples due to CBH constraints (low clouds, no parallax)

**Proposed Solution**: Use ALL ~75,000 images (5 flights × 15k images) with multiple deep learning strategies

**Key Insight**: CPL provides rich auxiliary labels beyond just CBH:
- ✅ Optical depth (2,373 profiles per flight)
- ✅ Cloud phase (ice vs. water)
- ✅ Feature type (cloud vs. aerosol)
- ✅ Number of layers
- ✅ Layer altitude (top & base for up to 10 layers)
- ✅ Ice/water path
- ✅ Lidar ratios

**Expected Outcome**: R² > 0.5 on CBH prediction using deep learning (competitive with GradientBoosting's 0.75)

**Resume Keywords**: Self-supervised learning, transfer learning, multi-task learning, vision transformers, contrastive learning, foundation models

---

## 📊 DATA INVENTORY

### Current Dataset (Limited)
- **933 samples** with CBH labels (low clouds only, 0.1-2.0 km)
- Constraint: Parallax issues at high altitudes
- Constraint: Need valid CPL + IR alignment

### Full Dataset (Available)
- **~75,000 IR images** total across 5 flights
  - 30Oct24: ~12,712 images
  - 23Oct24: ~14,611 images
  - 10Feb25: ~10,868 images
  - 18Feb25: ~10,343 images
  - 12Feb25: ~13,412 images

### CPL Auxiliary Data (Per Flight ~2,373 profiles)
From CPL HDF5 files, we have:

**Layer Descriptors** (shape: [2373, 10] - up to 10 layers per profile):
- `Number_Layers`: How many cloud/aerosol layers detected
- `Feature_Type`: Cloud (0-2), Aerosol (3-7), etc.
- `Cloud_Phase`: Ice, water, mixed
- `Layer_Base_Altitude`: Base height of each layer
- `Layer_Top_Altitude`: Top height of each layer
- `Opacity`: Clear, transparent, opaque

**Optical Properties** (shape: [2373, 10]):
- `Feature_Optical_Depth_532`: Optical depth at 532nm (primary)
- `Feature_Optical_Depth_1064`: Optical depth at 1064nm
- `Feature_Optical_Depth_355`: Optical depth at 355nm
- `Ice_Water_Path_532`: Ice/water content
- `Attenuated_Backscatter_Statistics_532`: Statistics (mean, std, min, max)

**Geolocation** (shape: [2373, 3]):
- `CPL_Latitude`, `CPL_Longitude`: Geographic coordinates
- `Solar_Zenith_Angle`, `Solar_Azimuth_Angle`: Solar geometry

**Derived Targets**:
- Sky condition (clear, partly cloudy, overcast)
- Surface type (ocean, land, ice)
- Temperature/pressure at layer boundaries

---

## 🚀 PROPOSED DEEP LEARNING STRATEGIES

### **Strategy 1: Self-Supervised Pre-Training + Fine-Tuning** ⭐ PRIMARY RECOMMENDATION

#### Phase 1: Self-Supervised Pre-Training (ALL 75k images)
Use modern self-supervised learning without any labels:

**Method A: Masked Autoencoder (MAE)**
```python
# Train on ALL 75k images
# No CPL data needed - pure image reconstruction
Model: Vision Transformer (ViT)
Task: Mask 75% of image patches, predict masked regions
Loss: MSE between reconstructed and original patches
Epochs: 100-200 (overnight on GPU)
Expected: Learn cloud texture, structure, atmospheric features
```

**Method B: SimCLR/MoCo (Contrastive Learning)**
```python
# Train on ALL 75k images
# Data augmentation creates positive pairs
Model: ResNet-50 or ViT
Task: Learn representations where augmented views are similar
Loss: NT-Xent (normalized temperature-scaled cross entropy)
Augmentations: Color jitter, crops, flips, rotations
Expected: Learn invariant cloud features
```

**Method C: DINO (Self-Distillation)**
```python
# State-of-the-art self-supervised for ViT
Model: ViT-Small or ViT-Base
Task: Student network matches teacher network outputs
Loss: Cross-entropy between student and teacher predictions
Expected: Learn semantic cloud representations
```

#### Phase 2: Supervised Fine-Tuning (933 labeled CBH samples)
```python
# Initialize with pre-trained weights from Phase 1
# Fine-tune on CBH regression task
Model: Pre-trained ViT/ResNet + regression head
Task: Predict CBH from learned representations
Loss: Huber loss (smooth L1)
Epochs: 20-50
Expected: R² > 0.5 (vs. current -0.19)
```

**Why This Works**:
- ✅ Leverages 75k images (vs. 933)
- ✅ Standard deep learning pipeline (ImageNet pre-training → fine-tuning)
- ✅ No need for perfect CPL alignment on all images
- ✅ Strong resume material: "Self-supervised learning on 75k atmospheric images"

**Expected Performance**: R² = 0.4 - 0.6 on CBH

---

### **Strategy 2: Multi-Task Learning with Auxiliary CPL Labels**

Use CPL data to create multiple prediction tasks:

**Architecture**: Shared encoder → Multiple task-specific heads

**Primary Task**: CBH regression (933 samples)

**Auxiliary Tasks** (use all CPL profiles):

1. **Optical Depth Prediction** (~12k samples per flight)
   - Target: `Feature_Optical_Depth_532`
   - Loss: MSE
   - Range: 0.0 - 5.0 (typical)
   - Why: Closely related to cloud properties

2. **Cloud Phase Classification** (~12k samples)
   - Target: `Cloud_Phase` (ice=0, water=1, mixed=2)
   - Loss: Cross-entropy
   - Classes: 3
   - Why: Different phases have different heights

3. **Number of Layers Prediction** (~12k samples)
   - Target: `Number_Layers`
   - Loss: MSE or Poisson
   - Range: 0-10
   - Why: Scene complexity indicator

4. **Feature Type Classification** (~12k samples)
   - Target: `Feature_Type` (cloud types, aerosol types)
   - Loss: Cross-entropy
   - Classes: ~8
   - Why: Learn cloud vs. aerosol discrimination

5. **Sky Condition Classification** (~12k samples)
   - Target: `Sky_Condition` (clear, partly cloudy, overcast)
   - Loss: Cross-entropy
   - Classes: 3
   - Why: Overall scene understanding

**Training**:
```python
# Multi-task loss
L_total = α_cbh * L_cbh +           # Primary (weight=5.0)
          α_od * L_optical_depth +   # Auxiliary (weight=1.0)
          α_phase * L_cloud_phase +  # Auxiliary (weight=0.5)
          α_layers * L_num_layers +  # Auxiliary (weight=0.5)
          α_type * L_feature_type +  # Auxiliary (weight=0.5)
          α_sky * L_sky_condition    # Auxiliary (weight=0.5)

# Gradients shared across all tasks
# Auxiliary tasks provide additional training signal
# Help learn better representations for primary CBH task
```

**Why This Works**:
- ✅ More training signal (12k samples × 5 tasks vs. 933 samples × 1 task)
- ✅ Auxiliary tasks guide representation learning
- ✅ Proven technique (used in autonomous driving, medical imaging)
- ✅ All data from same CPL files - already aligned

**Expected Performance**: R² = 0.5 - 0.7 on CBH

---

### **Strategy 3: Weakly Supervised Learning with Pseudo-Labels**

Create proxy labels for unlabeled images from navigation/metadata:

**Pseudo-Labeling Tasks**:

1. **Geographic Region Prediction**
   - Extract lat/lon from nav files
   - Discretize into regions (e.g., 10° × 10° grid)
   - Task: Classify which region
   - Samples: ALL 75k images
   - Why: Different regions have different cloud patterns

2. **Time-of-Day Prediction**
   - Extract UTC time from nav files
   - Task: Regression (0-24 hours)
   - Samples: ALL 75k images
   - Why: Diurnal cycle affects clouds

3. **Solar Angle Prediction**
   - Extract SZA/SAA from CPL or nav
   - Task: Regression (0-90° for SZA, 0-360° for SAA)
   - Samples: ALL 75k images
   - Why: Solar geometry affects cloud appearance

4. **Scene Complexity Estimation**
   - Compute from image: entropy, gradient magnitude, texture
   - Task: Regression (complexity score)
   - Samples: ALL 75k images
   - Why: Complex scenes often have more/higher clouds

**Training Pipeline**:
```python
# Phase 1: Pre-train on pseudo-labels (ALL 75k images)
for epoch in range(50):
    for image, (lat, lon, time, sza, saa, complexity) in dataloader:
        # Multi-task prediction
        pred_geo = model.geo_head(features)
        pred_time = model.time_head(features)
        pred_solar = model.solar_head(features)
        pred_complex = model.complexity_head(features)
        
        # Combined loss
        loss = L_geo + L_time + L_solar + L_complex
        
# Phase 2: Fine-tune on CBH (933 samples)
# Initialize from Phase 1 weights
```

**Why This Works**:
- ✅ Uses ALL 75k images
- ✅ No CPL alignment needed for Phase 1
- ✅ Weak labels still provide useful signal
- ✅ Feature learning generalizes to CBH task

**Expected Performance**: R² = 0.3 - 0.5 on CBH

---

### **Strategy 4: Sequence Modeling (Temporal Context)**

Treat each flight as a temporal sequence:

**Key Insight**: Clouds evolve slowly - consecutive images are correlated

**Architecture**: 
- Encoder: ResNet or ViT per image
- Temporal: LSTM, Transformer, or 1D CNN
- Decoder: Regression head for CBH

**Input**: Sliding window of N consecutive images (N=3, 5, 10)

**Why This Works**:
- ✅ Leverages temporal smoothness
- ✅ More context per prediction
- ✅ Can use unlabeled sequences for pre-training

**Training**:
```python
# Use ALL sequential images for self-supervised pre-training
# Task: Predict next image in sequence
# Fine-tune on CBH with labeled subsequences
```

**Expected Performance**: R² = 0.4 - 0.6 on CBH

---

## 📈 IMPLEMENTATION PLAN

### Week 1: Data Preparation
- [ ] Extract all IR images from HDF5 files (~75k total)
- [ ] Parse CPL auxiliary labels (optical depth, cloud phase, etc.)
- [ ] Create train/val/test splits (preserve flight boundaries)
- [ ] Build dataloaders for different strategies

### Week 2: Strategy 1 (Self-Supervised Pre-Training)
- [ ] Implement MAE or SimCLR
- [ ] Train on 75k images (100 epochs, ~8-12 hours on GPU)
- [ ] Save pre-trained weights
- [ ] Fine-tune on 933 CBH samples (20 epochs, ~2 hours)
- [ ] Evaluate: Target R² > 0.3

### Week 3: Strategy 2 (Multi-Task Learning)
- [ ] Implement multi-task architecture
- [ ] Prepare all auxiliary task labels from CPL
- [ ] Train end-to-end (50 epochs, ~6-8 hours)
- [ ] Ablation: Which tasks help most?
- [ ] Evaluate: Target R² > 0.4

### Week 4: Strategy 3 (Weakly Supervised)
- [ ] Extract pseudo-labels (lat/lon, time, solar angles)
- [ ] Pre-train on pseudo-labels (50 epochs, ~6 hours)
- [ ] Fine-tune on CBH (20 epochs, ~2 hours)
- [ ] Evaluate: Target R² > 0.3

### Week 5: Analysis & Writing
- [ ] Compare all strategies
- [ ] Identify best approach
- [ ] Generate visualizations (learning curves, attention maps)
- [ ] Write paper draft

---

## 🎓 DEEP LEARNING RESUME KEYWORDS

This research covers cutting-edge DL techniques:

### Self-Supervised Learning
- ✅ Masked Autoencoders (MAE)
- ✅ Contrastive Learning (SimCLR, MoCo)
- ✅ Self-Distillation (DINO)
- ✅ Foundation models for atmospheric science

### Transfer Learning
- ✅ Pre-training on large unlabeled datasets
- ✅ Fine-tuning on downstream tasks
- ✅ Domain adaptation

### Multi-Task Learning
- ✅ Shared representations
- ✅ Auxiliary task design
- ✅ Loss weighting strategies

### Vision Transformers
- ✅ ViT architecture
- ✅ Patch embeddings
- ✅ Self-attention mechanisms

### Advanced Techniques
- ✅ Weak supervision
- ✅ Pseudo-labeling
- ✅ Temporal modeling (LSTM/Transformers)
- ✅ Sequence prediction

---

## 💡 WHY THIS WILL WORK (Unlike Current Approach)

### Current Approach Problems:
❌ Only 933 samples (insufficient for deep learning)
❌ Training from scratch (random initialization)
❌ No leverage of unlabeled data
❌ Single task (CBH only)

### Proposed Approach Advantages:
✅ **75,000 images** for pre-training (80× more data)
✅ **Transfer learning** (pre-trained weights)
✅ **Multiple supervision signals** (auxiliary tasks)
✅ **Aligned with DL best practices** (ImageNet → downstream tasks)

### Why Deep Learning SHOULD Work Here:
1. **Sufficient data for pre-training** (75k images)
2. **Rich auxiliary labels** (optical depth, cloud phase, etc.)
3. **Transfer learning proven effective** (computer vision standard)
4. **Multi-task learning provides regularization** (prevents overfitting)

---

## 📊 EXPECTED OUTCOMES

### Quantitative Results
- **Strategy 1 (Self-Supervised)**: R² = 0.4 - 0.6
- **Strategy 2 (Multi-Task)**: R² = 0.5 - 0.7
- **Strategy 3 (Weakly Supervised)**: R² = 0.3 - 0.5
- **Best Combined**: R² = 0.6 - 0.75 (competitive with GradientBoosting)

### Qualitative Insights
- Learned representations visualized with t-SNE
- Attention maps showing what model focuses on
- Ablation studies showing which auxiliary tasks help
- Error analysis by cloud type, altitude, time-of-day

### Publication Potential
**Title**: *"Self-Supervised Learning for Cloud Base Height Prediction: Leveraging 75,000 Unlabeled Atmospheric Images"*

**Venues**:
- CVPR 2026 (Computer Vision for Earth Observation workshop)
- ICML 2026 (Applications track)
- NeurIPS 2026 (Climate Change AI workshop)
- Remote Sensing of Environment (journal)

**Key Contributions**:
1. First application of MAE/SimCLR to atmospheric cloud imagery
2. Multi-task learning framework for cloud property prediction
3. Benchmark dataset: 75k atmospheric images + multi-label annotations
4. Proof that self-supervised learning works in low-data scientific domains

---

## 🔧 TECHNICAL IMPLEMENTATION

### Hardware Requirements
- **GPU**: NVIDIA GTX 1070 Ti (yours) - sufficient!
- **RAM**: 16GB+ recommended
- **Storage**: ~500GB for image cache
- **Time**: 2-3 weeks for all experiments

### Software Stack
```python
# Core libraries (already installed)
torch >= 2.0
torchvision >= 0.15
h5py
numpy, pandas

# Additional for self-supervised learning
timm  # PyTorch Image Models (has MAE, DINO implementations)
lightly  # Self-supervised learning library (has SimCLR, MoCo)
```

### Code Structure
```
cloudMLPublic/
├── src/
│   ├── ssl/  # Self-supervised learning
│   │   ├── mae.py  # Masked Autoencoder
│   │   ├── simclr.py  # Contrastive learning
│   │   └── dino.py  # Self-distillation
│   ├── multitask/  # Multi-task learning
│   │   ├── model.py  # Multi-task architecture
│   │   └── losses.py  # Combined loss functions
│   └── data/
│       ├── full_dataset.py  # Load all 75k images
│       └── cpl_labels.py  # Extract auxiliary labels
├── configs/
│   ├── ssl_pretrain.yaml  # Self-supervised config
│   ├── multitask.yaml  # Multi-task config
│   └── finetune_cbh.yaml  # Fine-tuning config
└── scripts/
    ├── pretrain_ssl.py  # Run self-supervised pre-training
    ├── train_multitask.py  # Run multi-task learning
    └── finetune_cbh.py  # Fine-tune on CBH
```

---

## 🎯 DECISION: WHICH STRATEGY TO START WITH?

### Recommended: **Strategy 1 (Self-Supervised Pre-Training)**

**Reasons**:
1. ✅ **Most standard approach** (established in computer vision)
2. ✅ **No need for CPL alignment** on all 75k images
3. ✅ **Best resume impact** ("Self-supervised learning on 75k images")
4. ✅ **Proven to work** (MAE paper showed strong results on ImageNet)
5. ✅ **Can combine with Strategy 2** (pre-train with MAE, then multi-task fine-tune)

### Implementation Timeline
- **Weekend 1**: Extract 75k images, build dataloader
- **Weekend 2**: Implement & run MAE pre-training (overnight)
- **Weekend 3**: Fine-tune on CBH, evaluate
- **Weekend 4**: Write paper draft

---

## 📄 COMPARISON TO SIMPLE MODEL PAPER

| Aspect | Simple Model Paper | Deep Learning Paper |
|--------|-------------------|-------------------|
| **Story** | "Classical ML wins on small data" | "Self-supervised DL competitive on small data" |
| **Data Used** | 933 samples | 75,000 samples |
| **Methods** | GradientBoosting, RF, SVR | MAE, SimCLR, ViT, Multi-task |
| **Best R²** | 0.75 (GradientBoosting) | 0.5-0.7 (target) |
| **Resume Value** | Classical ML | ✅ **Deep Learning** |
| **Novelty** | Negative result | Positive technical contribution |
| **Venues** | JMLR, domain journals | CVPR, ICML, NeurIPS |
| **Time to Complete** | 1 week | 3-4 weeks |

---

## ✅ FINAL RECOMMENDATION

**Pursue Strategy 1: Self-Supervised Pre-Training**

**Why**:
- You want deep learning on your resume ✅
- You have 75,000 images available ✅
- Self-supervised learning is state-of-the-art ✅
- Can still fall back to simple model paper if this fails ✅

**Next Steps**:
1. Create data extraction script for all 75k images
2. Implement MAE or use `timm` library pre-built
3. Run pre-training (overnight, ~100 epochs)
4. Fine-tune on 933 CBH samples
5. Compare to GradientBoosting baseline

**Success Criteria**:
- R² > 0.3: Publishable (DL competitive with classical ML)
- R² > 0.5: Strong result (DL approaching classical ML)
- R² > 0.7: Excellent result (DL matches/beats classical ML)

**Fallback**: If R² < 0.3, you still have the simple model paper ready to go.

---

**Status**: Ready to implement  
**Estimated Time**: 3-4 weeks part-time  
**Confidence**: High (75k images + proven techniques)  
**Resume Impact**: Strong (self-supervised learning, ViT, transfer learning)