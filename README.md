# Cloud Base Height Retrieval from Airborne Observations

Machine learning system for cloud base height (CBH) retrieval using NASA ER-2 aircraft data, featuring rigorous validation methodology and honest assessment of performance across atmospheric regimes.

**Author:** Rylan Malarchick | Embry-Riddle Aeronautical University  
**Contact:** malarchr@my.erau.edu

---

## Key Findings

This project demonstrates that **atmospheric features substantially outperform image-based approaches** for CBH retrieval, while documenting critical limitations in cross-regime generalization.

### Performance Summary

| Validation Strategy | R² | MAE | Assessment |
|--------------------|----|-----|------------|
| Pooled K-fold | 0.924 | 49.7 m | **Inflated** (temporal autocorrelation) |
| Per-flight shuffled | **0.715** | **49.0 m** | Primary honest metric |
| Per-flight time-ordered | -0.055 | 129.8 m | Strict temporal holdout |
| Leave-one-flight-out (LOFO) | **-15.4** | 422 m | **Catastrophic domain shift** |

### Why This Matters

1. **Validation methodology is critical:** Pooled K-fold inflates R² by 0.21 due to lag-1 temporal autocorrelation of 0.94
2. **Domain shift is catastrophic:** Models trained on one atmospheric regime fail completely when deployed to another (R² = -15.4)
3. **Few-shot adaptation works:** 50 labeled samples from a target flight recovers R² = 0.57-0.85
4. **Conformal prediction fails:** Split conformal achieves only 27% coverage (target: 90%) due to exchangeability violations

---

## Project Highlights

### What Works
- **Within-flight deployment:** R² = 0.715, MAE = 49.0 m (per-flight shuffled validation)
- **Few-shot domain adaptation:** 50 samples → R² = 0.57 (mean), up to 0.85 for similar regimes
- **Per-flight uncertainty calibration:** 86% coverage with 277 m intervals
- **Real-time inference:** 0.28 ms per prediction, 1.3 MB model, CPU-only

### What Fails
- **Cross-regime generalization:** R² = -15.4 without adaptation
- **Temporal extrapolation:** R² = -0.055 on time-ordered holdout within flights
- **Split conformal prediction:** 27% coverage (exchangeability violated by autocorrelation)
- **Vision baselines:** ResNet-18 achieves R² = 0.617, underperforming GBDT

---

## Dataset

- **Samples:** 1,426 CPL-aligned observations from 3 research flights
- **Campaigns:** GLOVE 2025 (Feb), WHYMSIE 2024 (Oct)
- **Features:** 10 base ERA5 variables + 28 physics-based derived features (38 total)
- **Target:** Cloud base height from CPL lidar (0.21-1.95 km)
- **Key insight:** Surface temperature (t2m) dominates base model predictions (72% importance), consistent with lifting condensation level physics

### Flight Distribution
| Flight | Campaign | Samples | CBH (km) |
|--------|----------|---------|----------|
| Flight 1 | GLOVE 2025 | 1,021 | 1.34 ± 0.22 |
| Flight 2 | GLOVE 2025 | 129 | 0.85 ± 0.16 |
| Flight 3 | WHYMSIE 2024 | 276 | 0.88 ± 0.23 |

---

## Technical Approach

### Feature Engineering
Physics-based derived features from 10 base ERA5 variables:
- **Thermodynamic:** virtual temperature, potential temperature, saturation vapor pressure
- **Stability:** stability-moisture interactions, stability anomaly
- **Moisture:** dew point depression, relative humidity, mixing ratio
- **Solar/Temporal:** solar geometry transformations, diurnal cycle encodings

Top predictors in enhanced model:
1. `virtual_temperature` (33% importance)
2. `stability_x_tcwv` (22%)
3. `t2m` (14%)

### Domain Adaptation
Evaluated 5 methods for cross-regime generalization:
- **Few-shot learning** (recommended): 50 samples → R² = 0.57-0.85
- **TrAdaBoost:** R² = -0.41 (modest improvement)
- **Instance weighting:** R² = -19.9 to -21.4 (fails)
- **MMD alignment:** R² = -39.4 (destroys signal)

### Uncertainty Quantification
| Method | Coverage | Target | Width (m) |
|--------|----------|--------|-----------|
| Split Conformal | 27% | 90% | 278 |
| Adaptive Conformal | 11% | 90% | 58 |
| Quantile Regression | 58% | 90% | 510 |
| **Per-flight Calibration** | **86%** | 90% | 313 |

---

## Known Limitations

### Data & Methodology
1. **Temporal autocorrelation:** Lag-1 ρ = 0.94 invalidates pooled cross-validation
2. **Limited regime diversity:** 3 flights from 2 campaigns; generalization to tropical/polar/oceanic regimes unvalidated
3. **ERA5 resolution:** 25 km horizontal grid cannot capture fine-scale boundary layer variability

### Sensor & Processing
4. **Camera auto-scaling:** Automatic exposure adjustment creates inconsistent brightness across frames, complicating shadow detection
5. **Shadow detection thresholds:** Brightness-based detection fails in complex illumination (thin clouds, low solar elevation)
6. **CPL ground truth uncertainty:** ~30 m vertical resolution, cloud edge detection ambiguity

### Model & Deployment
7. **Domain shift:** Catastrophic failure (R² = -15.4) when deployed to unseen atmospheric regimes without adaptation
8. **Conformal prediction assumptions:** Exchangeability violated by temporal structure and domain shift
9. **Vision model limitations:** CNNs underperform tabular models despite comparable sample sizes

---

## Repository Structure

```
cloudMLPublic/
├── preprint/                    # Academic preprint (LaTeX)
├── src/cbh_retrieval/           # Production model code
├── scripts/                     # Training & analysis scripts
├── outputs/
│   ├── feature_engineering/     # Enhanced features dataset
│   ├── domain_adaptation/       # LOFO & few-shot results
│   ├── uncertainty/             # Conformal prediction results
│   └── tabular_model/           # Training results
├── tests/cbh/                   # Test suite
└── docs/cbh/                    # Documentation
```

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/rylanmalarchick/CloudMLPublic.git
cd CloudMLPublic

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run training with per-flight validation
python scripts/train_tabular_model.py

# Run domain adaptation experiments
python scripts/run_domain_adaptation.py

# Run uncertainty quantification
python scripts/run_uncertainty_quantification.py
```

---

## Reproducibility

All experiments are fully reproducible:
- **Random seed:** 42 (fixed throughout)
- **Validated dataset:** `outputs/feature_engineering/Enhanced_Features.hdf5` (1,426 × 38 features)
- **Results JSON files:** All metrics traceable to source

### Key Result Files
| File | Contents |
|------|----------|
| `outputs/tabular_model/training_results.json` | Per-flight CV metrics |
| `outputs/domain_adaptation/domain_adaptation_results.json` | LOFO & few-shot results |
| `outputs/uncertainty/uncertainty_quantification_results.json` | UQ method comparison |
| `outputs/feature_engineering/ablation_study_results.json` | Feature importance |

---

## Citation

If you use this work, please cite:

```bibtex
@article{malarchick2026cbh,
  title={Atmospheric Features Outperform Images for Cloud Base Height Retrieval: 
         A Systematic Comparison Using NASA Airborne Observations},
  author={Malarchick, Rylan},
  journal={arXiv preprint},
  year={2026}
}
```

---

## Acknowledgments

This work builds upon methods developed during a NASA OSTEM internship (May-August 2025) with NASA Goddard Space Flight Center High Altitude Research Program. Thanks to Dr. Dong Wu and the NASA ER-2 flight team for data access and technical discussions.

**Data Sources:**
- ERA5 reanalysis: ECMWF Copernicus Climate Data Store
- CPL lidar: NASA Goddard Space Flight Center
- ER-2 imagery: NASA High Altitude Research Program

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Contact

**Rylan Malarchick**  
Embry-Riddle Aeronautical University  
Email: malarchr@my.erau.edu | rylan1012@gmail.com  
GitHub: [@rylanmalarchick](https://github.com/rylanmalarchick)
