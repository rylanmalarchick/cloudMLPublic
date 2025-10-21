# Diagnostic Plan - Is This Task Learnable?

**Date:** October 20, 2024  
**Status:** STOP running experiments - Diagnose fundamental issues first  
**Priority:** CRITICAL - Understand before continuing

---

## The Problem

After 4 training runs, we're stuck:
- **Best R²: -0.045** (Run 1 - the "broken" version)
- **All R² scores are negative** → Model worse than predicting the mean
- **Each "fix" makes it worse** → Suggests we're missing something fundamental

**We've been optimizing hyperparameters without asking: Can a neural network learn this task with this data?**

---

## Fundamental Questions to Answer

### 1. Is the task learnable from these features?

**Features available:**
- IR images (single wavelength, 440x440 pixels)
- Solar Zenith Angle (SZA)
- Solar Azimuth Angle (SAA)
- Temporal sequence (3-7 frames)

**Target:**
- Cloud optical depth (continuous, range 0.15-1.95 km)

**Question:** Do IR intensity patterns + sun angles contain enough information to predict optical depth?

**Diagnostic:**
```python
# 1. Correlation analysis
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Extract simple features from images
mean_intensity = np.mean(images, axis=(1,2,3))
std_intensity = np.std(images, axis=(1,2,3))
max_intensity = np.max(images, axis=(1,2,3))

# Check correlations with target
print(f"Mean intensity vs optical depth: r={pearsonr(mean_intensity, y_true)}")
print(f"Std intensity vs optical depth: r={pearsonr(std_intensity, y_true)}")
print(f"SZA vs optical depth: r={pearsonr(sza, y_true)}")

# Expected: At least r > 0.3 for some feature to be learnable
```

**Success criterion:** At least one feature shows r² > 0.1 (10% variance explained)

---

### 2. Can simpler models learn anything?

**Before throwing deep learning at it, try:**

#### A. Linear Regression Baseline
```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Features: mean/std/max intensity + SZA + SAA
X = np.column_stack([mean_intensity, std_intensity, max_intensity, sza, saa])
y = y_true

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"Linear regression R²: {r2:.4f}")
```

**Success criterion:** R² > 0.0 (beats mean baseline)

#### B. Random Forest Baseline
```python
from sklearn.ensemble import RandomForestRegressor

# Add spatial features (gradients, texture)
X_spatial = extract_spatial_features(images)  # e.g., HOG, Gabor filters
X_full = np.column_stack([X, X_spatial])

rf = RandomForestRegressor(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"Random Forest R²: {r2:.4f}")
print(f"Feature importances: {rf.feature_importances_}")
```

**Success criterion:** R² > 0.2 with clear important features

#### C. Simple CNN Baseline
```python
# Simplest possible CNN - no attention, no temporal
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc = nn.Linear(64 * 107 * 107 + 2, 1)  # +2 for SZA/SAA
    
    def forward(self, img, sza, saa):
        x = F.relu(self.conv1(img))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = torch.cat([x, sza, saa], dim=1)
        return self.fc(x)

# Train for 10 epochs, check R²
```

**Success criterion:** R² > 0.0 with no fancy architecture

**IF ALL THREE FAIL (R² < 0):** The task may not be learnable from these features!

---

### 3. Is the data quality sufficient?

**Check for issues:**

#### A. Label Noise
```python
# Check if some samples have inconsistent labels
# (similar images → very different targets)

from sklearn.metrics.pairwise import cosine_similarity

# Find pairs of similar images
img_flat = images.reshape(len(images), -1)
similarity = cosine_similarity(img_flat)

# For very similar images (>95% similar), check target variance
for i in range(len(images)):
    similar_idx = np.where(similarity[i] > 0.95)[0]
    if len(similar_idx) > 1:
        target_variance = np.var(y_true[similar_idx])
        print(f"Sample {i}: {len(similar_idx)} similar images, target std={np.sqrt(target_variance):.3f}")

# Expected: Similar images should have similar targets (std < 0.2)
```

#### B. Class Imbalance / Distribution Issues
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.hist(y_true, bins=30)
plt.title("Target Distribution")
plt.xlabel("Optical Depth (km)")

plt.subplot(132)
plt.scatter(mean_intensity, y_true, alpha=0.5)
plt.title("Intensity vs Target")
plt.xlabel("Mean Intensity")

plt.subplot(133)
plt.scatter(sza, y_true, alpha=0.5)
plt.title("SZA vs Target")
plt.xlabel("Solar Zenith Angle")

plt.tight_layout()
plt.savefig("data_quality_check.png")
```

**Look for:**
- Severe class imbalance (most samples at one value)
- No clear relationship in scatter plots
- Multimodal distributions that don't make physical sense

---

### 4. Is the architecture appropriate?

**Our current architecture:**
- Multi-scale temporal attention (complex)
- 7 temporal frames (may be too much)
- Multiple attention mechanisms (spatial + temporal + multi-scale)

**Simpler alternatives to test:**

#### A. Single-frame baseline (no temporal)
```yaml
temporal_frames: 1
use_temporal_attention: false
use_multiscale_temporal: false
```

#### B. Simple CNN + MLP (no attention)
```yaml
use_spatial_attention: false
use_temporal_attention: false
use_multiscale_temporal: false
```

#### C. Different architecture family
- Vision Transformer (ViT)
- ResNet-based regressor
- MobileNet (lightweight)

**IF simple architectures also fail:** Problem is in data/features, not architecture complexity.

---

## Diagnostic Experiments (Priority Order)

### Experiment D1: Correlation Analysis (30 minutes)
**Goal:** Check if ANY simple feature correlates with target

**Script:**
```python
# Create: diagnostics/correlation_analysis.py
# - Extract basic features (mean, std, gradients)
# - Compute correlations with target
# - Plot scatter plots
# - Report Pearson r, Spearman ρ
```

**Decision point:**
- If r² < 0.05 for all features → **Task may not be learnable**
- If r² > 0.1 for some features → **Proceed to D2**

---

### Experiment D2: Simple Baselines (1 hour)
**Goal:** Can simple models beat mean baseline?

**Script:**
```python
# Create: diagnostics/simple_baselines.py
# - Linear regression on hand-crafted features
# - Random Forest with feature importance
# - Logistic regression (treat as classification: low/med/high)
```

**Decision point:**
- If all R² < 0 → **Data doesn't contain learnable signal**
- If any R² > 0.1 → **Signal exists, proceed to D3**

---

### Experiment D3: Architecture Ablation (2 hours)
**Goal:** Which components help vs hurt?

**Test configs:**
1. Single frame, no attention → baseline
2. Single frame, spatial attention only
3. Multi-frame (3), no attention
4. Multi-frame (3), temporal attention
5. Current complex architecture

**Script:**
```python
# Create: diagnostics/architecture_ablation.py
# - Train each config for 20 epochs
# - Track R², variance ratio, training time
# - Compare results
```

**Decision point:**
- If simpler = better → **Over-architected for task**
- If complex = better but still R² < 0 → **Need different approach**

---

### Experiment D4: Data Quality Check (1 hour)
**Goal:** Are there fundamental data issues?

**Script:**
```python
# Create: diagnostics/data_quality.py
# - Check for duplicate/near-duplicate samples
# - Check label noise (similar inputs → different outputs?)
# - Check for outliers (impossible values)
# - Check train/val/test distribution shifts
# - Visualize high-error samples
```

**Look for:**
- Mislabeled samples
- Corrupted images
- Systematic biases
- Distribution shifts between flights

---

## Decision Tree

```
START
  ↓
D1: Correlation Analysis
  ↓
   No correlation (r²<0.05) → STOP: Task likely not learnable from these features
                                Consider: Different wavelengths? More metadata?
  ↓
   Some correlation (r²>0.05)
      ↓
    D2: Simple Baselines
      ↓
       All R²<0 → STOP: Data doesn't contain learnable signal
                    Consider: Data collection issues? Need different sensors?
      ↓
       Some R²>0
          ↓
        D3: Architecture Ablation
          ↓
           Simpler is better → Use simple architecture, abandon complex attention
          ↓
           Complex helps but still R²<0.2 → Try different loss/approach
              ↓
            D4: Data Quality
              ↓
               Major issues found → Fix data, re-run
              ↓
               Data looks OK → Consider:
                  - Different task (classification instead of regression?)
                  - Multi-task learning (predict multiple properties)
                  - Semi-supervised learning
                  - Ensemble methods
                  - Domain-specific physics-based features
```

---

## Expected Outcomes

### Scenario A: Task is fundamentally not learnable
**Evidence:**
- Correlation analysis shows r² < 0.05
- Simple models all get R² < 0
- No clear patterns in scatter plots

**Next steps:**
- Consult with domain experts (atmospheric scientists)
- Consider if optical depth is the right target (maybe cloud type? altitude?)
- Investigate if additional features are needed (multi-wavelength? polarization?)
- Consider if this is a measurement/instrumentation issue

---

### Scenario B: Task is learnable, but we're doing it wrong
**Evidence:**
- Simple models get R² = 0.1-0.3
- Clear correlations exist
- Feature importance shows what matters

**Next steps:**
- Use insights from simple models to guide architecture
- Focus on features that matter (maybe SZA is key? Spatial gradients?)
- Try simpler architecture (single frame + MLP might be enough)
- Consider ensemble of simple models

---

### Scenario C: Task is hard but learnable
**Evidence:**
- Simple models get R² = 0.0-0.1 (barely positive)
- Weak but real correlations
- Complex patterns in data

**Next steps:**
- Current architecture might be right, but training approach is wrong
- Try different losses (quantile regression, focal loss, etc.)
- Try different training strategies (curriculum learning, progressive training)
- Consider data augmentation strategies
- Maybe we DO need Run 5's stronger variance loss

---

## Time Investment

| Phase | Time | Value |
|-------|------|-------|
| D1: Correlation | 30 min | Essential |
| D2: Baselines | 1 hour | Essential |
| D3: Ablation | 2 hours | Very helpful |
| D4: Data Quality | 1 hour | Good to have |
| **Total** | **4-5 hours** | vs many more failed training runs |

**Compare to:** Continuing to tweak hyperparameters = potentially weeks with no guarantee of success.

---

## Recommendation

**STOP running training experiments for now.**

**START with diagnostics:**

1. **TODAY (30 min):** Run correlation analysis
   - If no correlations → Save yourself weeks of frustration
   - If correlations exist → You know signal is there

2. **TODAY (1 hour):** Run simple baselines
   - If they work (R² > 0) → Deep learning might work too
   - If they fail → Deep learning won't magically fix it

3. **IF baselines work:** Then decide if Run 5 or architecture changes

4. **IF baselines fail:** Stop and reconsider the whole approach

---

## What This Gives You

**Confidence** that time spent on deep learning is justified  
**Understanding** of what features matter  
**Baseline** to beat (not just "better than mean")  
**Insights** to guide architecture choices  
**Clear decision points** (continue vs pivot)  

vs

Running Run 5 might get R² = -0.03 with 35% variance, still negative, back to square one

---

## I Can Help With

1. **Write the diagnostic scripts** (correlation_analysis.py, simple_baselines.py)
2. **Run them quickly** (30-60 min total)
3. **Analyze results** and make clear recommendations
4. **Create visualizations** to show what's working/not working

**Then we make an informed decision** instead of another hopeful experiment.

---

**Your call:** Do you want me to write the diagnostic scripts, or do you still want to try Run 5?

My honest recommendation: **Diagnostics first**. 4 hours of diagnostics will save you days/weeks of guessing.

---

**Status:** Awaiting decision  
**Next action:** Create diagnostics/correlation_analysis.py or proceed with Run 5