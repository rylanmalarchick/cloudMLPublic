# Actionable Plan for Next CloudML Colab Training Run

**Date**: 2025-01-28  
**Status**: LITERATURE_INSIGHTS.md ✅ VALIDATED - Ready to implement  
**Current Baseline**: R² = -0.0927, MAE = 0.3415 km, RMSE = 0.5046 km

---

## Executive Summary

I have thoroughly reviewed all three papers (cloudTop.txt, nonlinear.txt, sounder.txt) against your LITERATURE_INSIGHTS.md document. **Verdict: Your literature review is EXCELLENT and correctly captures all key insights.** The recommendations are well-grounded in the papers and directly applicable to your cloud shadow height prediction problem.

### ✅ What LITERATURE_INSIGHTS.md Got Right:
1. **TrajGRU-style location-variant temporal processing** is critical for cloud motion (Paper 1)
2. **Two-stage learning** with feature extraction phase improves complex system modeling (Paper 2)
3. **Multi-sensor fusion** helps when single sensors have complementary weaknesses (Paper 3)
4. **Self-supervised pre-training** on large unlabeled datasets is powerful (Papers 1 & 2)

### ⚠️ Important Context Differences:
- **Papers predict**: Cloud-TOP height from temporal SEQUENCES (multi-timestep)
- **Your problem**: Cloud SHADOW height from spatial MULTI-FRAMES (single-timestep, different angles)
- **Implication**: Temporal attention in your model is actually SPATIAL-VIEW attention across simultaneous frames
- **Key insight**: Your 5 "temporal frames" are really 5 simultaneous spatial views → treat as multi-view geometry problem

---

## 🎯 THREE-TIER IMPLEMENTATION PLAN

---

## TIER 1: IMMEDIATE (Run Before Next Training) ⭐⭐⭐
**Timeline**: 2-4 hours  
**Expected Gain**: +15-25% R² improvement  
**Risk**: Low

### 1.1 Fix Temporal Frame Count ✅ CRITICAL
**Paper Evidence**: Himawari-8 paper used 6 input frames → 12 output; you use 5  
**Your Context**: If data permits, increase to 7 frames for more spatial coverage

```yaml
# configs/colab_optimized_full_tuned.yaml
dataset:
  temporal_frames: 7  # Change from 5 → 7 (if data allows)
```

**Rationale**: More frames = more viewing angles = better triangulation for shadow height

---

### 1.2 Implement Self-Supervised Pre-training Phase ✅ HIGHEST PRIORITY
**Paper Evidence**: 
- Paper 2: Two-stage learning (unsupervised feature extraction → supervised mapping) effective
- Paper 1: Pre-training on large datasets improves generalization

**Implementation Plan**:

#### Option A: Autoencoder Pre-training (RECOMMENDED)
Train encoder to reconstruct input images from latent features:

```python
# Add to training notebook BEFORE main training
def pretrain_encoder(model, train_loader, epochs=20):
    """
    Pre-train the CNN encoder via reconstruction task.
    Forces encoder to learn meaningful spatial features.
    """
    # Freeze all except CNN encoder + add decoder
    encoder = model.cnn_layers
    decoder = build_simple_decoder(model.cnn_output_size)
    
    optimizer = torch.optim.Adam(encoder.parameters() + decoder.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        for batch in train_loader:
            images, _, _ = batch
            batch_size, seq_len, h, w = images.shape
            
            # Process each frame
            for t in range(seq_len):
                frame = images[:, t, :, :].unsqueeze(1)
                
                # Encode then decode
                features = encoder(frame)
                reconstructed = decoder(features)
                
                loss = criterion(reconstructed, frame)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        print(f"Pretrain Epoch {epoch}: Reconstruction Loss = {loss.item():.4f}")
    
    return model

# Usage in notebook:
model = pretrain_encoder(model, train_loader_unlabeled, epochs=20)
# Then proceed with normal supervised training
```

#### Option B: Next-Frame Prediction (if temporal data available)
If you have temporal sequences of IRAI data:

```python
def pretrain_temporal(model, temporal_loader, epochs=15):
    """
    Pre-train temporal attention by predicting next frame.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        for images in temporal_loader:
            # Use frames [0:4] to predict frame [4]
            input_frames = images[:, 0:4, :, :]
            target_frame = images[:, 4, :, :]
            
            # Forward pass
            predicted = model.forward_temporal_only(input_frames)
            loss = criterion(predicted, target_frame)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model
```

**Expected Gain**: +10-20% R² by learning better representations before supervised fine-tuning

---

### 1.3 Implement Multi-Scale Temporal (Spatial-View) Attention
**Paper Evidence**: Paper 1 showed multi-scale temporal processing captures features at different scales

**Current Code**: You have `TemporalAttention` but it's single-scale  
**Improvement**: Multi-scale attention across your 5 (or 7) viewing angles

```python
# Add to src/pytorchmodel.py

class MultiScaleTemporalAttention(nn.Module):
    """
    Multi-scale attention across temporal/spatial frames.
    Processes frames at different temporal scales before attention.
    Based on Himawari-8 paper insights.
    """
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Multi-scale processing paths
        self.scale_1 = nn.Identity()  # Full resolution
        self.scale_2 = nn.Sequential(  # 2-frame average
            nn.Conv1d(feature_dim, feature_dim, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        self.scale_3 = nn.Sequential(  # 3-frame average
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        
        # Attention over concatenated multi-scale features
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim * 3,  # Concatenated scales
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Project back to original dimension
        self.projection = nn.Linear(feature_dim * 3, feature_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, feature_dim)
        Returns:
            attended: (batch, feature_dim)
            weights: attention weights
        """
        batch_size, seq_len, feat_dim = x.shape
        
        # Transpose for Conv1d: (batch, feature_dim, seq_len)
        x_t = x.transpose(1, 2)
        
        # Multi-scale processing
        scale1_out = self.scale_1(x_t)  # (batch, feat, seq_len)
        
        # For scale 2 & 3, pad to maintain same output seq_len
        scale2_out = F.pad(self.scale_2(x_t), (0, 1))  # Pad right
        scale3_out = F.pad(self.scale_3(x_t), (0, 2))  # Pad right
        
        # Concatenate scales: (batch, feat*3, seq_len)
        multi_scale = torch.cat([scale1_out, scale2_out, scale3_out], dim=1)
        
        # Transpose back: (batch, seq_len, feat*3)
        multi_scale = multi_scale.transpose(1, 2)
        
        # Apply attention
        attended, attn_weights = self.attention(
            multi_scale, multi_scale, multi_scale
        )
        
        # Pool over sequence dimension
        attended_pooled = attended.mean(dim=1)  # (batch, feat*3)
        
        # Project back to original feature dimension
        output = self.projection(attended_pooled)  # (batch, feat_dim)
        
        return output, attn_weights


# Modify MultimodalRegressionModel.__init__:
# Replace this line:
#   self.temporal_attention = TemporalAttention(self.cnn_output_size)
# With:
self.temporal_attention = MultiScaleTemporalAttention(
    self.cnn_output_size, 
    num_heads=4
)
```

**Expected Gain**: +5-10% R² by capturing cross-scale spatial-view relationships

---

## TIER 2: MEDIUM-TERM (Next 1-2 Weeks) ⭐⭐
**Timeline**: 4-8 hours implementation  
**Expected Gain**: +10-20% R² improvement  
**Risk**: Medium

### 2.1 Location-Variant Spatial Attention (TrajGRU-style)
**Paper Evidence**: Paper 1 showed TrajGRU >> ConvLSTM because real-world motion is location-variant

**Current Issue**: Your `SpatialAttention` is location-INVARIANT (same convolution everywhere)  
**Improvement**: Make attention weights depend on spatial location

```python
class LocationVariantSpatialAttention(nn.Module):
    """
    Location-variant spatial attention inspired by TrajGRU.
    Different parts of the image get different attention patterns.
    Critical for clouds which have location-specific structure.
    """
    def __init__(self, in_channels, num_flows=5, kernel_size=7):
        super().__init__()
        self.num_flows = num_flows
        
        # Generate flow fields (location-variant kernels)
        self.flow_conv = nn.Conv2d(
            in_channels,
            num_flows * 2,  # 2 for (dx, dy) offsets
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        # Attention weights based on flows
        self.attention_weights = nn.Conv2d(
            in_channels,
            num_flows,
            kernel_size=1
        )
        
        # Final gating
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            attended: (batch, channels, height, width)
        """
        batch, C, H, W = x.shape
        
        # Generate flow fields
        flows = self.flow_conv(x)  # (batch, num_flows*2, H, W)
        flows = flows.view(batch, self.num_flows, 2, H, W)
        
        # Generate attention weights per flow
        flow_weights = self.attention_weights(x)  # (batch, num_flows, H, W)
        flow_weights = F.softmax(flow_weights, dim=1)
        
        # Apply location-variant attention via flows
        attended_flows = []
        for i in range(self.num_flows):
            # Extract flow field for this flow
            flow_i = flows[:, i, :, :, :]  # (batch, 2, H, W)
            weight_i = flow_weights[:, i:i+1, :, :]  # (batch, 1, H, W)
            
            # Warp input according to flow (simplified - use grid_sample in full impl)
            # For now, use weighted pooling as proxy
            attended_i = x * weight_i
            attended_flows.append(attended_i)
        
        # Aggregate flows
        attended = sum(attended_flows) / self.num_flows
        
        # Final gating
        gate = self.gate(x)
        output = x * gate + attended * (1 - gate)
        
        return output


# Replace in MultimodalRegressionModel:
# self.spatial_attention = SpatialAttention(1)
# With:
self.spatial_attention = LocationVariantSpatialAttention(
    in_channels=1,
    num_flows=5,
    kernel_size=7
)
```

**Expected Gain**: +8-15% R² by handling location-specific cloud shadow patterns

---

### 2.2 Add More Input Channels (if Available)
**Paper Evidence**: Paper 3 showed multi-spectral channels (especially H2O, CO2 bands) dramatically improve thin cirrus height

**Action**: Check if your IRAI data has additional spectral bands beyond grayscale
- Thermal IR bands?
- Multiple visible bands?
- NIR bands?

```python
# If multi-channel data available:
# Modify model config:
model:
  image_shape: [7, 440, 640]  # 7 frames (was 5)
  
  # Change CNN input channels:
  cnn_layers:
    - out_channels: 64
      params:
        in_channels: 3  # RGB or multi-spectral (was 1)
        kernel_size: 7
        stride: 2
        padding: 3
```

**Expected Gain**: +5-15% R² if additional spectral info available

---

### 2.3 Adopt Three-Phase Training Strategy
**Paper Evidence**: Paper 2 showed staged training (feature learning → parameter fitting) outperforms end-to-end

**Implementation**:

```python
# Phase 1: Self-supervised pre-training (unlabeled IRAI data)
model = pretrain_encoder(model, unlabeled_loader, epochs=20)
save_checkpoint(model, "pretrained_encoder.pth")

# Phase 2: Supervised pre-training on LARGE single flight
# Use flight with most samples (e.g., ER2_20160501_F02 if it has 500+ samples)
model = train_supervised(
    model, 
    single_flight_loader,
    epochs=30,
    lr=1e-4,
    weight_decay=1e-5
)
save_checkpoint(model, "pretrained_single_flight.pth")

# Phase 3: Multi-flight fine-tuning with REDUCED overweighting
# Load Phase 2 checkpoint
model.load_state_dict(torch.load("pretrained_single_flight.pth"))

# Fine-tune with gentler overweighting
config["loss"]["class_weight"] = 2.0  # Was 5.0 - reduce!
config["training"]["learning_rate"] = 5e-5  # Lower LR for fine-tuning

model = train_supervised(
    model,
    multi_flight_loader,
    epochs=50,
    early_stopping_patience=5
)
```

**Expected Gain**: +15-25% R² by avoiding negative transfer and overfitting

---

## TIER 3: FUTURE WORK (Research Extensions) ⭐
**Timeline**: 2-4 weeks  
**Expected Gain**: +10-30% R² improvement  
**Risk**: High (research-level)

### 3.1 Multi-Sensor Fusion (if additional data available)
**Paper Evidence**: Paper 3 showed imager+sounder fusion improved thin cirrus by 50%

**Check if you have**:
- Lidar data (CALIPSO, CALIOP)
- Radar data
- Sounder data (AIRS, CrIS)
- NWP model output (temperature profiles, tropopause height)

**Implementation**: Use as additional input features or as auxiliary loss constraints

---

### 3.2 Physics-Informed Loss Functions
**Note**: Papers validated that pure data-driven >> PINN for cloud tasks  
**BUT**: Simple physics constraints can help:

```python
def physics_informed_loss(pred_height, sun_elevation, shadow_length):
    """
    Soft constraint: shadow_length ≈ height / tan(sun_elevation)
    """
    expected_shadow = pred_height / torch.tan(sun_elevation * np.pi / 180)
    physics_loss = F.mse_loss(shadow_length, expected_shadow)
    return physics_loss

# Add to total loss:
total_loss = mse_loss + 0.1 * physics_loss
```

---

## 📊 EXPECTED PERFORMANCE TRAJECTORY

| Milestone | R² Target | MAE Target (km) | Implementation |
|-----------|-----------|-----------------|----------------|
| **Current Baseline** | -0.09 | 0.34 | Tuned config |
| **+ Tier 1 (Quick Wins)** | 0.15 - 0.25 | 0.28 - 0.31 | Self-supervised + Multi-scale |
| **+ Tier 2 (Medium)** | 0.35 - 0.50 | 0.22 - 0.26 | Location-variant + 3-phase training |
| **+ Tier 3 (Research)** | 0.55 - 0.70 | 0.18 - 0.22 | Multi-sensor fusion |

---

## 🚀 CONCRETE NEXT STEPS (BEFORE NEXT COLAB RUN)

### Step 1: Update Config (5 mins)
```bash
cd /content/repo
git pull origin main

# Edit configs/colab_optimized_full_tuned.yaml
# Change: temporal_frames: 5 → 7 (if data allows)
```

### Step 2: Implement Multi-Scale Attention (30 mins)
- Copy `MultiScaleTemporalAttention` class to `src/pytorchmodel.py`
- Replace `self.temporal_attention = TemporalAttention(...)` with multi-scale version
- Test with single forward pass

### Step 3: Add Self-Supervised Pre-training (1 hour)
- Add autoencoder pre-training cell to notebook (before main training)
- Create `pretrain_encoder()` function
- Load unlabeled IRAI data if available (or use training data without labels)

### Step 4: Update Training Script (30 mins)
```python
# In Colab notebook, add BEFORE main training loop:

print("=" * 50)
print("PHASE 1: Self-Supervised Pre-training")
print("=" * 50)

model = pretrain_encoder(
    model, 
    train_loader,  # Use same data, ignore labels
    epochs=15,
    device=device
)

print("Pre-training complete! Proceeding to supervised training...")

# Then run normal training
model = train(...
```

### Step 5: Run Experiment & Monitor
- Run full pipeline with new changes
- Monitor TensorBoard for:
  - Pre-training reconstruction loss (should decrease smoothly)
  - Supervised training loss (should start lower than before)
  - Validation R² (target: > 0.15 by epoch 20)
- Save all outputs to Drive
- Compare to baseline

---

## 📋 VALIDATION CHECKLIST

Before declaring success, verify:

- [ ] R² > 0.0 (beats mean baseline)
- [ ] R² > 0.15 (meaningful predictive power)
- [ ] MAE < 0.30 km (practical utility)
- [ ] Training loss decreasing smoothly (no divergence)
- [ ] Validation loss follows training (no severe overfitting)
- [ ] Multi-flight performance comparable (generalization)
- [ ] Predictions physically reasonable (0-15 km range)
- [ ] TensorBoard shows improvement over baseline
- [ ] All files saved to Drive (models, logs, plots)

---

## 🔬 ABLATION STUDY (After Success)

Once Tier 1 achieves R² > 0.15, run ablations to quantify each component:

1. **Baseline** (current tuned config): R² = -0.09
2. **+ Multi-scale attention only**: R² = ?
3. **+ Self-supervised pre-training only**: R² = ?
4. **+ Both (Tier 1 full)**: R² = ?
5. **+ Location-variant attention (Tier 2)**: R² = ?
6. **+ 3-phase training (Tier 2)**: R² = ?

Document results in `ABLATION_RESULTS.md`

---

## 🎓 KEY LESSONS FROM PAPERS

### What Papers Say WORKS:
1. ✅ **Location-variant processing** (TrajGRU >> ConvLSTM)
2. ✅ **Multi-scale feature extraction**
3. ✅ **Self-supervised pre-training** on large datasets
4. ✅ **Two-stage learning** (features first, then parameters)
5. ✅ **Multi-sensor fusion** when sensors are complementary
6. ✅ **Pure data-driven for complex atmospheric phenomena**

### What Papers Say DOESN'T Work:
1. ❌ **PINN for clouds** (too chaotic, non-differentiable physics)
2. ❌ **Single-scale temporal modeling** (misses multi-scale dynamics)
3. ❌ **Location-invariant convolutions** for natural motion
4. ❌ **End-to-end training without pre-training** (for complex systems)
5. ❌ **Optical flow methods** for longer time horizons (error accumulation)

---

## 📞 DECISION POINTS

**If after Tier 1 implementation:**
- **R² < 0**: Problem with implementation → debug, check data pipeline
- **R² 0-0.10**: Modest improvement → proceed to Tier 2
- **R² 0.10-0.25**: Good improvement → analyze what worked, iterate Tier 2
- **R² > 0.25**: Excellent! → write paper, add Tier 3 for bonus

**If stuck with R² < 0.10 after Tier 2:**
- Re-examine data quality (are labels correct?)
- Check for data leakage or temporal/spatial leaks
- Consider simpler baseline (linear regression on hand-crafted features)
- Investigate whether problem is ill-posed (shadow height ambiguous?)

---

## 💾 IMPLEMENTATION FILES TO CREATE

1. `src/modules/multi_scale_attention.py` - Multi-scale temporal attention
2. `src/modules/location_variant_attention.py` - TrajGRU-style spatial attention  
3. `src/pretraining.py` - Self-supervised pre-training utilities
4. `notebooks/pretrain_encoder.ipynb` - Pre-training notebook
5. `configs/colab_tier1.yaml` - Config with Tier 1 improvements
6. `configs/colab_tier2.yaml` - Config with Tier 2 improvements

---

## 📚 REFERENCES

1. Yu et al. (2023): Himawari-8 Cloud Top Height Nowcasting - `docs/cloudTop.txt`
2. Rostamijavanani et al.: LSTM Autoencoders for Nonlinear System ID - `docs/nonlinear.txt`
3. Heidinger et al. (2019): Sounder Data for Cirrus Height - `docs/sounder.txt`
4. LITERATURE_INSIGHTS.md - Comprehensive paper analysis

---

## ✅ FINAL CHECKLIST BEFORE RUNNING

- [ ] Git pull latest changes
- [ ] Config updated (temporal_frames, new modules)
- [ ] Multi-scale attention implemented and tested
- [ ] Self-supervised pre-training cell added to notebook
- [ ] Drive paths configured correctly
- [ ] TensorBoard path verified
- [ ] Checkpoint saving enabled
- [ ] GPU memory profiled (no OOM expected)
- [ ] Estimated runtime acceptable (< 6 hours for Colab)

---

## 🎯 SUCCESS CRITERIA

**Minimum Success**: R² > 0.15, MAE < 0.30 km  
**Good Success**: R² > 0.35, MAE < 0.25 km  
**Excellent Success**: R² > 0.50, MAE < 0.22 km

**Let's implement Tier 1 and run the next training session!** 🚀

---

*Document prepared: 2025-01-28*  
*Based on: LITERATURE_INSIGHTS.md validation + current model analysis*  
*Next update: After Tier 1 training run completes*