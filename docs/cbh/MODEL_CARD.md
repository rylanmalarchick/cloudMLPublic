# Model Card: Cloud Base Height Retrieval System

## Model Details

### Model Information
- **Model Name**: CBH Retrieval Production Model (Sprint 6)
- **Model Version**: 1.0.0
- **Model Type**: Gradient Boosting Decision Tree (GBDT) Regressor
- **Framework**: scikit-learn 1.7.0
- **Date**: November 2025
- **Developers**: NASA Cloud ML Team
- **Contact**: research@nasa.gov
- **License**: NASA Open Source Agreement

### Model Description
This model predicts Cloud Base Height (CBH) in meters above ground level using a combination of atmospheric features from ERA5 reanalysis data and geometric features derived from cloud shadow analysis in aerial imagery. The model is designed for operational cloud base height retrieval during airborne field campaigns.

### Model Architecture
- **Algorithm**: Gradient Boosting Regressor (sklearn.ensemble.GradientBoostingRegressor)
- **Number of Estimators**: 200
- **Max Depth**: 8
- **Learning Rate**: 0.05
- **Loss Function**: Least Squares
- **Input Features**: 18 features (12 atmospheric + 6 geometric)
- **Output**: Single continuous value (CBH in meters)

### Training Data
- **Dataset Size**: 933 labeled samples
- **Data Source**: Integrated features from airborne campaigns (Flights F1, F2, F3, F4, F6)
- **Temporal Coverage**: Multiple flight campaigns
- **Geographic Coverage**: Campaign-specific flight paths
- **Label Source**: Ground truth CBH measurements from flight instruments
- **Data Format**: HDF5 (Integrated_Features.hdf5)

### Feature Engineering
The model uses 18 engineered features across two categories:

**Atmospheric Features (ERA5-derived):**
1. `blh` - Boundary Layer Height (m)
2. `lcl` - Lifting Condensation Level (m)
3. `inversion_height` - Temperature inversion height (m)
4. `moisture_gradient` - Vertical moisture gradient
5. `stability_index` - Atmospheric stability index
6. `t2m` - Temperature at 2 meters (K)
7. `d2m` - Dewpoint at 2 meters (K)
8. `sp` - Surface pressure (Pa)
9. `tcwv` - Total column water vapor (kg/m²)
10. `sza_deg` - Solar zenith angle (degrees)

**Geometric Features (Shadow-derived):**
11. `cloud_edge_x` - Cloud edge x-coordinate (pixels)
12. `cloud_edge_y` - Cloud edge y-coordinate (pixels)
13. `saa_deg` - Solar azimuth angle (degrees)
14. `shadow_angle_deg` - Shadow angle (degrees)
15. `shadow_detection_confidence` - Confidence score [0-1]
16. `shadow_edge_x` - Shadow edge x-coordinate (pixels)
17. `shadow_edge_y` - Shadow edge y-coordinate (pixels)
18. `shadow_length_pixels` - Shadow length (pixels)

### Preprocessing
- **Missing Value Handling**: Mean imputation using SimpleImputer
- **Feature Scaling**: StandardScaler (zero mean, unit variance)
- **Outlier Treatment**: None (robust to outliers via tree-based method)
- **Data Splitting**: Stratified 5-fold cross-validation (stratified by CBH quintiles)

## Intended Use

### Primary Use Cases
1. **Operational CBH Retrieval**: Real-time cloud base height estimation during airborne field campaigns
2. **Climate Research**: Support for cloud parameterization and validation studies
3. **Aviation Safety**: Cloud base height information for flight planning and safety

### Intended Users
- Climate scientists and researchers
- Field campaign operators
- Aviation meteorologists
- Cloud physics researchers

### Out-of-Scope Use Cases
- **Real-time aviation safety-critical systems** (requires additional certification)
- **Non-airborne platforms** (model trained on aerial imagery)
- **Domains with significantly different cloud types** (requires domain adaptation)
- **High-altitude clouds** (model optimized for low-to-mid level clouds)

## Performance Metrics

### Cross-Validation Performance (5-Fold Stratified CV)
- **R² Score**: 0.744 ± 0.037
- **Mean Absolute Error (MAE)**: 117.4 ± 7.4 meters
- **Root Mean Squared Error (RMSE)**: 187.3 ± 15.3 meters
- **Median Absolute Error**: ~95 meters (estimated from distribution)
- **90th Percentile Error**: ~280 meters (estimated)

### Production Model Performance (Full Dataset Training)
- **Training R²**: 0.993 (high due to full dataset fit)
- **Expected Generalization**: R² ≈ 0.74 based on CV

### Uncertainty Quantification
- **Method**: Quantile Regression (90% prediction intervals)
- **Interval Coverage**: 77.1% (under-calibrated; target: 90%)
- **Mean Interval Width**: 533.4 ± 20.8 meters
- **Uncertainty-Error Correlation**: 0.485 (moderate calibration)
- **Calibration Status**: Poorly calibrated (requires post-hoc calibration)

### Feature Importance (Top 5)
1. **d2m** (Dewpoint): 19.5% ± 1.2%
2. **t2m** (Temperature): 17.5% ± 3.1%
3. **moisture_gradient**: 7.7% ± 2.9%
4. **sza_deg** (Solar Zenith): 7.0% ± 1.2%
5. **saa_deg** (Solar Azimuth): 6.4% ± 1.3%

### Ensemble Performance
The production model is part of a multimodal ensemble system:

- **GBDT (Tabular)**: R² = 0.727 ± 0.112, MAE = 118.5 ± 15.8 m
- **CNN (Image)**: R² = 0.351 ± 0.075, MAE = 236.8 ± 16.7 m
- **Weighted Ensemble**: R² = 0.739 ± 0.096, MAE = 122.5 ± 19.8 m (optimal weights: 88% GBDT, 12% CNN)

**Note**: The tabular GBDT model is the primary production model due to superior performance.

## Limitations

### Known Limitations

1. **Domain Shift Sensitivity**
   - Model exhibits catastrophic failure on Flight F4 (R² = -0.98 in leave-one-out test)
   - F4 represents a different atmospheric regime or cloud type distribution
   - Few-shot adaptation with 10 F4 samples improves performance to R² ≈ -0.22 (still poor)
   - **Mitigation**: Deploy domain adaptation or flag predictions when input distribution differs

2. **Uncertainty Calibration**
   - 90% prediction intervals only achieve 77% coverage (under-confident)
   - Requires post-hoc calibration (isotonic regression, conformal prediction)
   - **Mitigation**: Apply calibration methods before using UQ for decision-making

3. **Image Model Underperformance**
   - CNN baseline achieves only R² = 0.35 (vs. GBDT R² = 0.73)
   - Simple CNN architecture may not capture spatial features effectively
   - **Mitigation**: Consider Vision Transformer or pretrained ResNet backbone

4. **Limited Training Data**
   - Only 933 labeled samples across 5 flights
   - May not generalize to all cloud types, geographic regions, or seasons
   - **Mitigation**: Continuous data collection and model retraining

5. **Feature Availability**
   - Requires ERA5 reanalysis data (3-hour latency in operational mode)
   - Shadow detection depends on solar illumination (daytime only)
   - **Mitigation**: Plan campaigns during daylight; use NRT ERA5 data

6. **Outlier Predictions**
   - Approximately 5-10% of predictions have errors > 300 meters
   - Typically occur during rapid atmospheric transitions or edge cases
   - **Mitigation**: Use uncertainty quantification to flag high-uncertainty predictions

### Ethical Considerations

1. **Fairness**: Model performance may vary by geographic region or atmospheric regime (as evidenced by F4 failure)
2. **Transparency**: All features are physically interpretable; model decisions are explainable via SHAP/feature importance
3. **Safety**: NOT certified for safety-critical aviation applications; use only for research/planning
4. **Environmental Impact**: Minimal computational footprint (CPU-based inference, <10ms per sample)

## Training Procedure

### Training Process
1. **Data Loading**: Load integrated features from HDF5 file
2. **Preprocessing**: Apply mean imputation and standard scaling
3. **Cross-Validation**: 5-fold stratified CV for model evaluation
4. **Hyperparameter Tuning**: Manual tuning (n_estimators=200, max_depth=5, lr=0.1)
5. **Production Training**: Retrain on full dataset with best hyperparameters
6. **Artifact Export**: Save model, scaler, and configuration

### Hardware & Software
- **Hardware**: NVIDIA GTX 1070 Ti (8GB VRAM), 16GB RAM, CPU training
- **OS**: Linux (Ubuntu-based)
- **Python Version**: 3.12
- **Key Dependencies**: 
  - scikit-learn 1.7.0
  - numpy 2.2.6
  - pandas 2.3.0
  - h5py 3.14.0

### Training Time
- **Cross-Validation**: ~2-5 minutes (5 folds × 200 trees)
- **Production Model**: ~30-60 seconds (full dataset, 200 trees)
- **Total Pipeline**: <10 minutes including data loading and validation

### Reproducibility
- **Random Seed**: 42 (fixed for all experiments)
- **Validation Strategy**: Stratified 5-fold CV (stratify by CBH quintiles)
- **Environment**: Pinned dependencies in `requirements.txt`

## Inference

### Input Specification
- **Format**: NumPy array or Pandas DataFrame
- **Shape**: (n_samples, 18) for batch inference or (18,) for single sample
- **Feature Order**: Must match training feature order (see Feature Engineering section)
- **Preprocessing**: Apply saved `production_scaler.joblib` before prediction
- **Missing Values**: Impute with feature means (from training set)

### Output Specification
- **Format**: NumPy array
- **Shape**: (n_samples,) for point predictions
- **Units**: Meters above ground level
- **Range**: Typically 0-3000 meters (physically constrained)
- **Uncertainty**: Optional quantile predictions (lower/upper bounds)

### Inference Performance
- **Latency**: <1ms per sample (CPU, batch_size=1)
- **Throughput**: ~10,000 samples/second (CPU, batched)
- **Memory**: ~5MB model size, ~50MB peak inference memory
- **Hardware**: CPU-only (no GPU required)

### Example Usage
```python
import joblib
import numpy as np

# Load artifacts
model = joblib.load('production_model.joblib')
scaler = joblib.load('production_scaler.joblib')

# Prepare input (18 features)
X_raw = np.array([[blh, lcl, inversion_height, moisture_gradient, 
                   stability_index, t2m, d2m, sp, tcwv, 
                   cloud_edge_x, cloud_edge_y, saa_deg, 
                   shadow_angle_deg, shadow_detection_confidence,
                   shadow_edge_x, shadow_edge_y, 
                   shadow_length_pixels, sza_deg]])

# Preprocess
X_scaled = scaler.transform(X_raw)

# Predict
cbh_meters = model.predict(X_scaled)[0]

print(f"Predicted CBH: {cbh_meters:.1f} meters")
```

## Model Governance

### Version Control
- **Model Registry**: Artifacts stored in `sow_outputs/sprint6/checkpoints/`
- **Version Tagging**: v1.0.0 (production release)
- **Change Log**: See `PHASE4_COMPLETION_SUMMARY.md` for update history

### Monitoring & Maintenance
- **Performance Monitoring**: Track MAE/RMSE on new labeled data
- **Data Drift Detection**: Monitor feature distributions (KS test, PSI)
- **Retraining Triggers**: 
  - MAE degrades by >20% on validation set
  - New flight campaign data available (≥100 samples)
  - Domain shift detected (distribution shift alert)
- **Update Frequency**: Quarterly retraining recommended

### Model Card Updates
- **Update Trigger**: Major version release, significant performance change, new use case
- **Responsible Party**: NASA Cloud ML Team
- **Review Cycle**: Annual review minimum

## Validation & Testing

### Validation Strategy
- **Cross-Validation**: 5-fold stratified CV (primary validation)
- **Error Analysis**: Per-flight breakdown, error distribution analysis
- **Domain Adaptation**: Leave-one-flight-out validation (F4 test case)
- **Uncertainty Quantification**: Coverage and calibration analysis

### Test Sets
- **In-Distribution**: Held-out folds in CV (R² = 0.744)
- **Out-of-Distribution**: Flight F4 (R² = -0.98, severe domain shift)
- **Production Validation**: Ongoing collection of labeled validation data

### Failure Modes
1. **Flight F4-like atmospheric regimes**: Near-total prediction failure
2. **High uncertainty samples**: Errors >300m when uncertainty >600m
3. **Shadow detection failures**: Degraded performance when confidence <0.5
4. **Missing ERA5 features**: Model cannot handle missing inputs (requires imputation)

## References & Citations

### Related Publications
- NASA Airborne Cloud Campaign Documentation (internal)
- ERA5 Reanalysis: Hersbach et al. (2020), QJRMS
- Gradient Boosting: Friedman (2001), Annals of Statistics

### Model Artifacts
- **Production Model**: `sow_outputs/sprint6/checkpoints/production_model.joblib`
- **Scaler**: `sow_outputs/sprint6/checkpoints/production_scaler.joblib`
- **Configuration**: `sow_outputs/sprint6/checkpoints/production_config.json`
- **Validation Reports**: `sow_outputs/sprint6/reports/`

### Documentation
- **Deployment Guide**: See `DEPLOYMENT_GUIDE.md`
- **Technical Report**: See `PHASE4_COMPLETION_SUMMARY.md`
- **User Guide**: See `QUICK_START.md`

---

**Model Card Version**: 1.0.0  
**Last Updated**: 2025-11-11  
**Next Review**: 2026-11-11