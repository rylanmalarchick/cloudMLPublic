# Model Card: Cloud Base Height Retrieval System

## Model Details

### Model Information
- **Model Name**: CBH Retrieval Production Model
- **Model Version**: 2.0.0 (Validated Restudy)
- **Model Type**: Gradient Boosting Decision Tree (GBDT) Regressor
- **Framework**: scikit-learn 1.7.0
- **Date**: January 2026
- **Developer**: Rylan Malarchick
- **Contact**: malarchr@my.erau.edu
- **License**: MIT

### Model Description
This model predicts Cloud Base Height (CBH) in meters above ground level using atmospheric features from ERA5 reanalysis data and physics-based derived features. The model is designed for operational cloud base height retrieval during airborne field campaigns, with documented limitations for cross-regime deployment.

### Model Architecture
- **Algorithm**: Gradient Boosting Regressor (sklearn.ensemble.GradientBoostingRegressor)
- **Number of Estimators**: 200
- **Max Depth**: 8
- **Learning Rate**: 0.05
- **Loss Function**: Least Squares
- **Input Features**: 38 features (10 base ERA5 + 28 physics-based derived)
- **Output**: Single continuous value (CBH in meters)

---

## Training Data

- **Dataset Size**: 1,426 labeled samples from 3 research flights
- **Data Source**: NASA ER-2 aircraft observations with CPL lidar ground truth
- **Campaigns**: GLOVE 2025 (Feb), WHYMSIE 2024 (Oct)
- **Temporal Coverage**: October 2024 - February 2025
- **Label Source**: Cloud Physics Lidar (CPL) cloud base retrievals
- **Data Format**: HDF5 (Enhanced_Features.hdf5)

### Flight Distribution
| Flight | Campaign | Samples | CBH (km) |
|--------|----------|---------|----------|
| Flight 1 | GLOVE 2025 | 1,021 | 1.34 ± 0.22 |
| Flight 2 | GLOVE 2025 | 129 | 0.85 ± 0.16 |
| Flight 3 | WHYMSIE 2024 | 276 | 0.88 ± 0.23 |

**Exclusions**: Flights 0 and 4 excluded due to insufficient samples (n=2 and n=8).

### Feature Engineering

**Base ERA5 Features (10):**
1. `t2m` - Temperature at 2 meters (K)
2. `d2m` - Dewpoint at 2 meters (K)
3. `sp` - Surface pressure (Pa)
4. `blh` - Boundary Layer Height (m)
5. `tcwv` - Total column water vapor (kg/m²)
6. `lcl` - Lifting condensation level (computed from t2m, d2m)
7. `stability_index` - Derived stability indicator
8. `moisture_gradient` - Vertical moisture structure
9. `sza_deg` - Solar zenith angle (degrees)
10. `saa_deg` - Solar azimuth angle (degrees)

**Physics-Based Derived Features (28):**
- Thermodynamic: virtual_temperature, potential_temperature, saturation_vapor_pressure, mixing_ratio
- Stability: stability_x_tcwv, stability_dpd_product, stability_anomaly
- Moisture: dew_point_depression, relative_humidity_2m, vapor_pressure
- Solar/Temporal: sza_cos, sza_sin, saa_cos, saa_sin, solar_heating_proxy, hour_sin, hour_cos
- Interactions: t2m_x_tcwv, blh_x_lcl, t2m_x_sza_cos, polynomial terms

---

## Performance Metrics

### Validation Strategy Comparison

| Validation Strategy | R² | MAE (m) | Assessment |
|--------------------|-----|---------|------------|
| Pooled K-fold | 0.924 | 49.7 | **Inflated** (temporal autocorrelation ρ=0.94) |
| Per-flight shuffled | **0.744** | **117.4** | **Primary honest metric** |
| Per-flight time-ordered | -0.055 | 129.8 | Strict temporal holdout |
| Leave-one-flight-out | **-15.4** | 422 | **Catastrophic domain shift** |

**Critical insight**: The 0.18 R² inflation from pooled to per-flight validation demonstrates that temporal autocorrelation (lag-1 ρ = 0.94) invalidates standard cross-validation.

### Feature Importance (Enhanced Model)
1. **virtual_temperature**: 33%
2. **stability_x_tcwv**: 22%
3. **t2m**: 14%
4. **saturation_vapor_pressure**: 4.4%
5. **tcwv**: 2.7%

### Feature Importance (Base Model)
1. **t2m** (Temperature): 72% - dominates predictions, consistent with LCL physics
2. **d2m** (Dewpoint): 6.5%
3. **tcwv** (Total column water vapor): 4.3%
4. **blh** (Boundary layer height): 4.1%

### Uncertainty Quantification

| Method | Coverage | Target | Width (m) | Assessment |
|--------|----------|--------|-----------|------------|
| Split Conformal | 27% | 90% | 278 | **Fails** (exchangeability violated) |
| Adaptive Conformal | 11% | 90% | 58 | **Fails** (intervals collapse) |
| Quantile Regression | 58% | 90% | 510 | Moderate under-coverage |
| Per-flight Calibration | **86%** | 90% | 313 | **Recommended** |

**Root cause of conformal failure**: Temporal autocorrelation (ρ=0.94) and domain shift violate the exchangeability assumption required for conformal prediction guarantees.

---

## Domain Adaptation

### Leave-One-Flight-Out (LOFO) Results
| Test Flight | n_test | R² | MAE (km) |
|-------------|--------|-----|----------|
| Flight 1 | 1,021 | -6.61 | 0.577 |
| Flight 2 | 129 | 0.15 | 0.119 |
| Flight 3 | 276 | -0.80 | 0.210 |
| **Mean** | -- | **-15.4** | **0.422** |

### Few-Shot Adaptation (Recommended Solution)
| Target Flight | 5-shot | 10-shot | 20-shot | 50-shot |
|---------------|--------|---------|---------|---------|
| Flight 1 | 0.47 | 0.76 | 0.81 | **0.85** |
| Flight 2 | 0.14 | 0.22 | 0.39 | **0.64** |
| Flight 3 | -0.37 | -0.14 | 0.02 | **0.23** |
| **Mean** | 0.08 | 0.28 | 0.41 | **0.57** |

**Operational protocol**: Collect 20-50 labeled samples from target regime before deployment. This recovers R² = 0.41-0.57 on average.

---

## Limitations

### Critical Limitations

1. **Catastrophic Domain Shift** (LOFO R² = -15.4)
   - Model fails completely when deployed to unseen atmospheric regimes
   - Predictions worse than constant baseline without adaptation
   - **Mitigation**: Few-shot adaptation with 50 samples recovers R² = 0.57-0.85

2. **Temporal Autocorrelation Inflation** (lag-1 ρ = 0.94)
   - Standard cross-validation inflates R² from 0.744 to 0.924
   - Adjacent samples are nearly identical, causing information leakage
   - **Mitigation**: Use per-flight validation as honest metric

3. **Conformal Prediction Failure** (27% coverage vs 90% target)
   - Exchangeability assumption violated by temporal structure
   - Split conformal and adaptive conformal fail catastrophically
   - **Mitigation**: Use per-flight calibration (86% coverage)

4. **Temporal Extrapolation Failure** (R² = -0.055)
   - Model cannot predict forward in time even within same flight
   - Time-ordered holdout shows near-zero predictive skill
   - **Mitigation**: Only valid for interpolation, not forecasting

### Data Limitations

5. **Limited Regime Diversity**
   - Only 3 flights from 2 campaigns
   - Generalization to tropical/polar/oceanic regimes unvalidated
   - **Mitigation**: Expand dataset with diverse campaigns

6. **ERA5 Spatial Resolution** (25 km)
   - Cannot capture fine-scale boundary layer variability
   - Low-altitude clouds controlled by micro-meteorology are challenging
   - **Mitigation**: Consider higher-resolution reanalysis (ERA5-Land)

### Sensor Limitations

7. **Camera Auto-Scaling**
   - Automatic exposure adjustment creates inconsistent brightness
   - Complicates shadow detection across frames
   - **Mitigation**: Use ERA5-only features (vision not recommended)

8. **Shadow Detection Assumptions**
   - Brightness thresholds fail in complex illumination
   - Thin clouds, multiple cloud layers, low solar elevation problematic
   - **Mitigation**: ERA5 features are more robust than shadow-based features

9. **CPL Ground Truth Uncertainty**
   - ~30 m vertical resolution
   - Cloud edge detection ambiguity
   - **Mitigation**: Accept ~30 m as floor for achievable MAE

---

## Inference

### Input Specification
- **Format**: NumPy array or Pandas DataFrame
- **Shape**: (n_samples, 38) for enhanced model or (n_samples, 10) for base model
- **Feature Order**: Must match training feature order
- **Preprocessing**: Apply StandardScaler from training
- **Missing Values**: Mean imputation (from training set)

### Output Specification
- **Format**: NumPy array
- **Shape**: (n_samples,)
- **Units**: Meters above ground level
- **Range**: Typically 200-2000 m

### Inference Performance
- **Latency**: 0.28 ms per sample (CPU)
- **Throughput**: ~3,500 samples/second (CPU)
- **Memory**: 1.3 MB model size
- **Hardware**: CPU-only (no GPU required)

---

## Ethical Considerations

1. **Fairness**: Model performance varies by atmospheric regime; do not assume uniform accuracy
2. **Transparency**: All features are physically interpretable; predictions explainable via SHAP
3. **Safety**: NOT certified for safety-critical aviation applications
4. **Reproducibility**: All experiments reproducible with seed=42

---

## Model Governance

### Version History
- **v1.0.0** (Nov 2025): Initial production model with inflated metrics
- **v2.0.0** (Jan 2026): Restudy with validated metrics, domain adaptation, honest UQ

### Artifacts
- **Model**: `outputs/tabular_model/production_model.joblib`
- **Scaler**: `outputs/tabular_model/production_scaler.joblib`
- **Enhanced Features**: `outputs/feature_engineering/Enhanced_Features.hdf5`
- **Results**: `outputs/*/results.json`

### Documentation
- **Preprint**: `preprint/cloudml_academic_preprint.tex`
- **README**: Repository root
- **Deployment Guide**: `docs/cbh/DEPLOYMENT_GUIDE.md`

---

**Model Card Version**: 2.0.0  
**Last Updated**: 2026-01-06  
**Author**: Rylan Malarchick
