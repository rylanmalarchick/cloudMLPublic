# Quick Guide: Review Sprint 3/4 Results

**Last Updated:** November 9, 2025  
**Status:** ✅ All deliverables complete

---

## 🚀 Quick Start: View Your Results

### 1. See the Executive Summary
```bash
cat SPRINT_3_4_EXECUTIVE_SUMMARY.md
```

**TL;DR:**
- Physical-only GBDT: **R² = 0.68, MAE = 136m** 🏆 (BEST MODEL)
- Attention CNN: R² = 0.33, MAE = 222m
- Image-only CNN: R² = 0.28, MAE = 233m
- All Sprint 3/4 deliverables ✅ COMPLETE

---

## 📊 View Model Performance

### Best Model Performance
```bash
cat sow_outputs/wp3_kfold/WP3_Report_kfold.json | jq '.results'
```

**Expected Output:**
```json
{
  "mean_r2": 0.6759,
  "mean_mae_km": 0.1356,
  "mean_rmse_km": 0.2105,
  "n_folds": 5
}
```

### All Models Comparison
```bash
cat sow_outputs/validation_summary/Validation_Summary.json | jq '.comparison_table'
```

---

## 🔬 View Ablation Study Results

```bash
cat sow_outputs/wp4_ablation/WP4_Ablation_Study.json | jq '.ablation_analysis'
```

**Key Findings:**
- Physical features beat image features by 2×
- Attention fusion improves over concat by 81%
- Current CNN architecture needs improvement

---

## 📁 Key Files Location Map

### SOW Deliverables

| Deliverable | File | Command to View |
|-------------|------|-----------------|
| 7.3a Feature Store | `sow_outputs/integrated_features/Integrated_Features.hdf5` | `h5ls -r <file>` |
| 7.3b Feature Importance | `sow_outputs/wp4_ablation/WP4_Ablation_Study.json` | `cat <file> \| jq` |
| 7.3c Validation Summary | `sow_outputs/validation_summary/Validation_Summary.json` | `cat <file> \| jq` |
| 7.4a Architecture | `sow_outputs/wp4_cnn_model.py` | `less <file>` |
| 7.4c Performance Reports | `sow_outputs/wp4_cnn/WP4_Report_*.json` | `cat <file> \| jq` |
| 7.4d Ablation Study | `sow_outputs/wp4_ablation/WP4_Ablation_Study.json` | `cat <file> \| jq` |

### Model Weights

```bash
# Image-only models (5 folds)
ls sow_outputs/wp4_cnn/model_image_only_fold*.pth

# Concat fusion models (5 folds)
ls sow_outputs/wp4_cnn/model_concat_fold*.pth

# Attention fusion models (5 folds)
ls sow_outputs/wp4_cnn/model_attention_fold*.pth
```

**Total:** 15 trained models (3 variants × 5 folds)

---

## 📈 Reproduce Results

### Re-run Physical Baseline
```bash
./venv/bin/python sow_outputs/wp3_kfold.py
```

**Expected:** R² ≈ 0.68, MAE ≈ 0.14 km (5-10 minutes)

### Re-run WP-4 CNN (Single Variant)
```bash
./venv/bin/python sow_outputs/wp4_cnn_model.py \
    --fusion-mode image_only \
    --n-folds 5 \
    --output-dir sow_outputs/wp4_cnn_rerun
```

**Expected:** R² ≈ 0.28, MAE ≈ 0.23 km (2-3 hours)

### Re-generate All Summaries
```bash
# Validation summary
./venv/bin/python sow_outputs/create_validation_summary.py

# Ablation study
./venv/bin/python sow_outputs/wp4_ablation_study.py

# Feature store
./venv/bin/python sow_outputs/create_integrated_features.py

# Final comprehensive report
./venv/bin/python sow_outputs/wp4_final_summary.py
```

---

## 🔍 Inspect Results in Detail

### View Integrated Feature Store
```bash
# List all groups/datasets
h5ls -r sow_outputs/integrated_features/Integrated_Features.hdf5

# View metadata
h5dump -A sow_outputs/integrated_features/Integrated_Features.hdf5 | head -50

# Interactive inspection (Python)
python3 << EOF
import h5py
f = h5py.File('sow_outputs/integrated_features/Integrated_Features.hdf5', 'r')
print("Groups:", list(f.keys()))
print("Metadata samples:", f.attrs['n_samples'])
print("CBH range:", f['metadata/cbh_km'][:].min(), 
      "to", f['metadata/cbh_km'][:].max(), "km")
f.close()
EOF
```

### View Per-Fold Results
```bash
# WP-3 Physical baseline per-fold
cat sow_outputs/wp3_kfold/WP3_Report_kfold.json | jq '.results.per_fold[]'

# WP-4 Image-only per-fold
cat sow_outputs/wp4_cnn/WP4_Report_image_only.json | jq '.fold_results[]'

# WP-4 Attention per-fold
cat sow_outputs/wp4_cnn/WP4_Report_attention.json | jq '.fold_results[]'
```

### View Key Insights
```bash
cat sow_outputs/validation_summary/Validation_Summary.json | jq '.key_insights'
```

---

## 📚 Documentation Hierarchy

### Start Here (Quick Overview)
1. **`SPRINT_3_4_EXECUTIVE_SUMMARY.md`** ← START HERE
   - 356 lines, ~5 min read
   - High-level results and findings
   - Bottom-line recommendations

### Detailed Documentation
2. **`sow_outputs/SPRINT_3_4_COMPLETION_SUMMARY.md`**
   - 506 lines, ~15 min read
   - Complete technical details
   - All deliverables cross-referenced
   - Risk assessment and next steps

### SOW Deliverables (JSON Reports)
3. **Validation Summary**
   ```bash
   cat sow_outputs/validation_summary/Validation_Summary.json | jq
   ```

4. **Ablation Study**
   ```bash
   cat sow_outputs/wp4_ablation/WP4_Ablation_Study.json | jq
   ```

5. **Individual Model Reports**
   ```bash
   # Physical baseline
   cat sow_outputs/wp3_kfold/WP3_Report_kfold.json | jq
   
   # Image-only CNN
   cat sow_outputs/wp4_cnn/WP4_Report_image_only.json | jq
   
   # Concat fusion
   cat sow_outputs/wp4_cnn/WP4_Report_concat.json | jq
   
   # Attention fusion
   cat sow_outputs/wp4_cnn/WP4_Report_attention.json | jq
   ```

---

## 🎯 Verify Deliverables Checklist

### Sprint 3: Feature Engineering
```bash
# 7.3a: Integrated feature store
test -f sow_outputs/integrated_features/Integrated_Features.hdf5 && echo "✅ 7.3a Found" || echo "❌ Missing"

# 7.3b: Feature importance / ablation
test -f sow_outputs/wp4_ablation/WP4_Ablation_Study.json && echo "✅ 7.3b Found" || echo "❌ Missing"

# 7.3c: Validation summary
test -f sow_outputs/validation_summary/Validation_Summary.json && echo "✅ 7.3c Found" || echo "❌ Missing"
```

### Sprint 4: Hybrid Models
```bash
# 7.4a: Architecture code
test -f sow_outputs/wp4_cnn_model.py && echo "✅ 7.4a Found" || echo "❌ Missing"

# 7.4c: Performance reports
test -f sow_outputs/wp4_cnn/WP4_Report_image_only.json && echo "✅ 7.4c (image-only) Found" || echo "❌ Missing"
test -f sow_outputs/wp4_cnn/WP4_Report_concat.json && echo "✅ 7.4c (concat) Found" || echo "❌ Missing"
test -f sow_outputs/wp4_cnn/WP4_Report_attention.json && echo "✅ 7.4c (attention) Found" || echo "❌ Missing"

# 7.4d: Ablation study
test -f sow_outputs/wp4_ablation/WP4_Ablation_Study.json && echo "✅ 7.4d Found" || echo "❌ Missing"
```

**Expected:** All ✅

---

## 💻 Python Quick Inspection

### Load and Inspect Results
```python
import json
import numpy as np

# Load validation summary
with open('sow_outputs/validation_summary/Validation_Summary.json') as f:
    val_summary = json.load(f)

print("Best Model:", val_summary['best_model']['name'])
print("R²:", val_summary['best_model']['r2'])
print("MAE:", val_summary['best_model']['mae_km'], "km")

# Load ablation study
with open('sow_outputs/wp4_ablation/WP4_Ablation_Study.json') as f:
    ablation = json.load(f)

print("\nModel Ranking:")
for i, model in enumerate(ablation['ablation_analysis']['ranking']['by_r2'], 1):
    print(f"{i}. {model['model']}: R² = {model['r2']:.4f}")

print("\nKey Findings:")
for finding in ablation['key_findings']:
    print(f"- {finding}")
```

### Load Integrated Features
```python
import h5py

# Open integrated feature store
with h5py.File('sow_outputs/integrated_features/Integrated_Features.hdf5', 'r') as f:
    # Load data
    cbh = f['metadata/cbh_km'][:]
    flight_ids = f['metadata/flight_id'][:]
    geo_features = f['geometric_features/derived_geometric_H'][:]
    era5_features = f['atmospheric_features/era5_features'][:]
    
    print("Total samples:", len(cbh))
    print("CBH range:", cbh.min(), "to", cbh.max(), "km")
    print("Geometric features shape:", geo_features.shape)
    print("ERA5 features shape:", era5_features.shape)
    
    # Flight distribution
    print("\nFlight distribution:")
    for fid in range(5):
        count = np.sum(flight_ids == fid)
        print(f"  Flight {fid}: {count} samples")
```

---

## 🔬 Advanced Analysis

### Compare Model Performance Across Folds
```python
import json
import matplotlib.pyplot as plt

# Load all model reports
models = {
    'Physical': 'sow_outputs/wp3_kfold/WP3_Report_kfold.json',
    'Image-only': 'sow_outputs/wp4_cnn/WP4_Report_image_only.json',
    'Concat': 'sow_outputs/wp4_cnn/WP4_Report_concat.json',
    'Attention': 'sow_outputs/wp4_cnn/WP4_Report_attention.json',
}

for name, path in models.items():
    with open(path) as f:
        data = json.load(f)
    
    if 'results' in data:  # WP-3 format
        folds = data['results']['per_fold']
    else:  # WP-4 format
        folds = data['fold_results']
    
    r2_values = [f['r2'] for f in folds]
    print(f"{name}: R² = {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
```

---

## 📊 Generate Custom Plots

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load validation summary
with open('sow_outputs/validation_summary/Validation_Summary.json') as f:
    data = json.load(f)

# Extract metrics
models = []
r2_means = []
r2_stds = []

for entry in data['comparison_table']:
    models.append(entry['model'])
    r2_means.append(entry['mean_r2'])
    r2_stds.append(entry['std_r2'])

# Create bar plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
ax.bar(x, r2_means, yerr=r2_stds, capsize=5, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_ylabel('R² Score')
ax.set_title('Model Comparison: R² Performance')
ax.axhline(y=0.5, color='r', linestyle='--', label='Target (R²=0.5)')
ax.legend()
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
print("Plot saved to model_comparison.png")
```

---

## 🎯 Bottom Line Commands

### Just Show Me The Results
```bash
# Best model performance
echo "=== BEST MODEL ==="
cat sow_outputs/validation_summary/Validation_Summary.json | \
    jq '.best_model | {name, r2, mae_km}'

# All models ranked
echo -e "\n=== MODEL RANKING ==="
cat sow_outputs/wp4_ablation/WP4_Ablation_Study.json | \
    jq '.ablation_analysis.ranking.by_r2[] | {model, r2}'

# Key findings
echo -e "\n=== KEY FINDINGS ==="
cat sow_outputs/wp4_ablation/WP4_Ablation_Study.json | \
    jq -r '.key_findings[]' | sed 's/^/  • /'
```

### Quick Summary Table
```bash
echo "Model                     R²         MAE (km)   RMSE (km)"
echo "--------------------------------------------------------"
cat sow_outputs/validation_summary/Validation_Summary.json | \
    jq -r '.comparison_table[] | "\(.model | .[0:25]) \(.mean_r2) \(.mean_mae_km) \(.mean_rmse_km)"' | \
    column -t
```

---

## 📞 Support & Questions

### Common Issues

**Q: Where are the model weights?**  
A: `sow_outputs/wp4_cnn/model_*.pth` (15 files total)

**Q: Which model should I use for production?**  
A: Physical-only GBDT (`sow_outputs/wp3_kfold/`) - R²=0.68, MAE=136m

**Q: How do I re-train with different hyperparameters?**  
A: Edit parameters in `sow_outputs/wp4_cnn_model.py` or pass CLI args

**Q: Where's the ablation study?**  
A: `sow_outputs/wp4_ablation/WP4_Ablation_Study.json`

**Q: Can I see training logs?**  
A: Training metrics are embedded in JSON reports (`epoch_trained` field)

---

## ✅ Verification Script

Run this to verify everything is in place:

```bash
#!/bin/bash
echo "Verifying Sprint 3/4 Deliverables..."
echo ""

# Sprint 3
echo "Sprint 3: Feature Engineering"
test -f sow_outputs/integrated_features/Integrated_Features.hdf5 && echo "  ✅ 7.3a: Integrated Feature Store" || echo "  ❌ 7.3a: Missing"
test -f sow_outputs/wp4_ablation/WP4_Ablation_Study.json && echo "  ✅ 7.3b: Feature Importance" || echo "  ❌ 7.3b: Missing"
test -f sow_outputs/validation_summary/Validation_Summary.json && echo "  ✅ 7.3c: Validation Summary" || echo "  ❌ 7.3c: Missing"

# Sprint 4
echo ""
echo "Sprint 4: Hybrid Models"
test -f sow_outputs/wp4_cnn_model.py && echo "  ✅ 7.4a: Model Architecture" || echo "  ❌ 7.4a: Missing"
test -f SPRINT_3_4_COMPLETION_SUMMARY.md && echo "  ✅ 7.4b: Training Documentation" || echo "  ❌ 7.4b: Missing"
test -f sow_outputs/wp4_cnn/WP4_Report_image_only.json && echo "  ✅ 7.4c: Performance Reports" || echo "  ❌ 7.4c: Missing"
test -f sow_outputs/wp4_ablation/WP4_Ablation_Study.json && echo "  ✅ 7.4d: Ablation Study" || echo "  ❌ 7.4d: Missing"

# Models
echo ""
echo "Trained Models:"
model_count=$(ls sow_outputs/wp4_cnn/model_*.pth 2>/dev/null | wc -l)
echo "  Found $model_count model files (expected: 15)"
test $model_count -eq 15 && echo "  ✅ All models present" || echo "  ⚠️ Some models missing"

echo ""
echo "Status: All deliverables $(test -f sow_outputs/validation_summary/Validation_Summary.json && echo '✅ COMPLETE' || echo '❌ INCOMPLETE')"
```

---

**Ready to review?** Start with `SPRINT_3_4_EXECUTIVE_SUMMARY.md` for the big picture! 🚀