# Production Model Documentation

**Generated**: 2025-11-11T20:30:57.037289

---

## Model Information

**Model Type**: GradientBoostingRegressor
**Model Name**: CBH_Production_GBDT
**Checkpoint Path**: `/home/rylan/Documents/research/NASA/programDirectory/cloudMLPublic/sow_outputs/sprint6/checkpoints/production_model.pkl`
**Scaler Path**: `/home/rylan/Documents/research/NASA/programDirectory/cloudMLPublic/sow_outputs/sprint6/checkpoints/production_scaler.pkl`

---

## Training Dataset

**Number of Samples**: 933
**Number of Features**: 18
**CBH Range**: [0.120, 1.950] km
**CBH Mean**: 0.830 km
**CBH Std**: 0.371 km

### Features

- blh
- lcl
- inversion_height
- moisture_gradient
- stability_index
- t2m
- d2m
- sp
- tcwv
- cloud_edge_x
- cloud_edge_y
- saa_deg
- shadow_angle_deg
- shadow_detection_confidence
- shadow_edge_x
- shadow_edge_y
- shadow_length_pixels
- sza_deg

---

## Hyperparameters

```json
{
  "n_estimators": 200,
  "max_depth": 8,
  "learning_rate": 0.05,
  "min_samples_split": 10,
  "min_samples_leaf": 4,
  "subsample": 0.8,
  "random_state": 42,
  "verbose": 0
}
```

---

## Performance Metrics

**R²**: 0.9932
**MAE**: 20.6 m
**RMSE**: 30.6 m
**Training Time**: 1.53 seconds

---

## Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | d2m | 0.1873 |
| 2 | t2m | 0.1803 |
| 3 | moisture_gradient | 0.0759 |
| 4 | sza_deg | 0.0721 |
| 5 | blh | 0.0616 |
| 6 | shadow_angle_deg | 0.0563 |
| 7 | saa_deg | 0.0496 |
| 8 | stability_index | 0.0423 |
| 9 | inversion_height | 0.0396 |
| 10 | sp | 0.0393 |

---

## Inference Performance

### CPU Inference

| Batch Size | Mean Time (ms) | Std Time (ms) | Throughput (samples/s) |
|------------|----------------|---------------|------------------------|
| 1 | 0.25 | 0.02 | 4054.4 |
| 4 | 0.29 | 0.04 | 14016.5 |
| 16 | 0.39 | 0.06 | 40652.5 |
| 32 | 0.50 | 0.06 | 64092.1 |
| 64 | 0.72 | 0.09 | 88923.8 |
| 128 | 1.05 | 0.02 | 121842.1 |

### Single Sample Latency

- **Mean**: 0.259 ms
- **Median**: 0.245 ms
- **P95**: 0.370 ms
- **P99**: 0.402 ms

---

## System Information

**Platform**: Linux-6.14.0-33-generic-x86_64-with-glibc2.39
**Python Version**: 3.12.3
**CPU**: x86_64
**CPU Count**: 6 physical, 12 logical
**RAM**: 15.56 GB

---

## Usage Example

```python
import pickle
import numpy as np

# Load model and scaler
with open('/home/rylan/Documents/research/NASA/programDirectory/cloudMLPublic/sow_outputs/sprint6/checkpoints/production_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('/home/rylan/Documents/research/NASA/programDirectory/cloudMLPublic/sow_outputs/sprint6/checkpoints/production_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare features (18 features in correct order)
# Feature order: blh, lcl, inversion_height, moisture_gradient, stability_index...
features = np.array([[...]])  # Shape: (n_samples, 18)

# Normalize and predict
features_scaled = scaler.transform(features)
cbh_km = model.predict(features_scaled)
```

---

## Model Card

### Intended Use

This model is designed for Cloud Base Height (CBH) retrieval from
integrated features including atmospheric (ERA5) and geometric (shadow-based)
measurements. It is intended for research purposes and operational CBH
prediction in the NASA ACTIVATE field campaign.

### Limitations

- Trained on specific flight conditions (Flights F1, F2, F4)
- Performance may degrade on out-of-distribution samples
- Assumes availability of all 18 input features
- Feature normalization is required before inference

### Ethical Considerations

- This model is for atmospheric science research
- No personal data or sensitive information is used
- Results should be validated against ground truth when available

---

**Documentation generated**: 2025-11-11T20:30:57.037437
