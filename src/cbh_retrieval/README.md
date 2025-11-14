# Cloud Base Height (CBH) Retrieval Module

Production-ready machine learning module for cloud base height prediction from satellite imagery and atmospheric data.

## Overview

This module contains the complete implementation of the CBH retrieval system developed in Sprint 6, achieving:
- **GBDT Model R²**: 0.744 (tabular features)
- **Ensemble R²**: 0.7391 (weighted GBDT + CNN)
- **MAE**: 117.4 m
- **Test Coverage**: 93.5%

## Module Structure

```
cbh_retrieval/
 image_dataset.py              # Image data loading and preprocessing
 mc_dropout.py                 # Monte Carlo Dropout for uncertainty quantification
 train_production_model.py    # Production model training pipeline
 offline_validation.py         # Validation framework (combined)
 offline_validation_tabular.py # Tabular model validation
 offline_validation_images.py  # Image model validation
 uncertainty_quantification.py # Uncertainty quantification (combined)
 uncertainty_quantification_tabular.py  # UQ for tabular model
 ensemble_models.py            # Ensemble strategy implementations
 ensemble_tabular_image.py    # Tabular + Image ensemble
 analyze_ensemble_results.py  # Ensemble analysis tools
 error_analysis.py             # Comprehensive error analysis
 few_shot_f4.py               # Domain adaptation for Flight F4
 few_shot_f4_tabular.py       # Tabular-based few-shot learning
 ablation_plots.py            # Ablation study visualizations
 performance_plots.py         # Performance metric visualizations
 spatial_attention_viz.py     # Spatial attention analysis
 temporal_attention_viz.py    # Temporal attention analysis
 __init__.py                  # Module initialization
```

## Key Components

### Models
- **Tabular GBDT**: Gradient boosting on atmospheric features (primary model)
- **Image CNN**: Simple convolutional network for satellite imagery
- **Ensemble**: Weighted averaging (88.75% GBDT, 11.25% CNN)

### Features
- ERA5 atmospheric reanalysis (temperature, pressure, humidity profiles)
- Boundary layer height (BLH)
- Lifting condensation level (LCL)
- Solar zenith angle (SZA)
- Flight metadata (altitude, position)

### Validation
- 5-fold stratified cross-validation
- Uncertainty quantification (90% confidence intervals)
- Comprehensive error analysis
- Domain adaptation experiments

## Usage

### Training Production Model

```python
from cbh_retrieval.train_production_model import train_production_model

# Train on all available data
model, scaler = train_production_model(
    features_path="outputs/preprocessed_data/Integrated_Features.hdf5"
)
```

### Running Validation

```python
from cbh_retrieval.offline_validation_tabular import main as validate_tabular

# 5-fold cross-validation
validate_tabular()
```

### Ensemble Prediction

```python
from cbh_retrieval.ensemble_models import WeightedEnsemble

ensemble = WeightedEnsemble(
    gbdt_model=gbdt_model,
    cnn_model=cnn_model,
    weights=[0.8875, 0.1125]
)

predictions = ensemble.predict(features, images)
```

### Uncertainty Quantification

```python
from cbh_retrieval.uncertainty_quantification_tabular import quantify_uncertainty

uq_results = quantify_uncertainty(
    model=model,
    features=features,
    labels=labels,
    confidence_level=0.90
)
```

## Requirements

See `docs/cbh/requirements_production.txt` for dependencies:
- scikit-learn >= 1.3.0
- torch >= 2.0.0
- h5py >= 3.8.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0

## Performance Benchmarks

| Model | R² | MAE (m) | RMSE (m) | Inference (ms) |
|-------|----|---------|-----------|--------------------|
| GBDT | 0.744 | 117.4 | 187.3 | 2.5 (CPU) |
| CNN | 0.351 | 236.8 | 299.1 | 15.3 (GPU) |
| Ensemble | 0.7391 | 122.5 | 195.0 | 18.0 (hybrid) |

## Documentation

Full documentation available in `docs/cbh/`:
- `MODEL_CARD.md` - Model specifications and performance
- `DEPLOYMENT_GUIDE.md` - Production deployment instructions
- `REPRODUCIBILITY_GUIDE.md` - Reproduction instructions
- `SPRINT6_FINAL_DELIVERY.md` - Complete Sprint 6 results

## Testing

Run tests from project root:

```bash
# Run all CBH tests
pytest tests/cbh/

# With coverage
pytest tests/cbh/ --cov=src/cbh_retrieval --cov-report=html
```

## Known Limitations

1. **Image Model**: SimpleCNN underperforms (R²=0.35). Upgrade to ResNet/ViT recommended.
2. **UQ Calibration**: 77% coverage vs. 90% target. Conformal prediction recommended.
3. **Flight F4 Domain Shift**: Severe performance degradation (R²=-0.98). Root cause analysis needed.

## Future Work

See `docs/cbh/FUTURE_WORK.md` for detailed roadmap:
- Temporal Vision Transformer implementation
- Cross-modal attention (ERA5 + imagery)
- Improved uncertainty calibration
- Domain adaptation strategies

## Citation

If you use this module, please cite:

```
NASA Cloud Base Height Retrieval System
Sprint 6 Production Release
https://github.com/rylanmalarchick/cloudMLPublic
```

## License

[Add license information]

## Contact

- Issues: GitHub Issues
- Documentation: `docs/cbh/`
- Main Project: [cloudMLPublic](../../README.md)