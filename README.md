# Cloud Base Height Retrieval from NASA ER-2 Airborne Observations

Machine learning for cloud base height (CBH) retrieval using NASA ER-2 aircraft data from the WHySMIE (Oct 2024) and GLOVE (Feb 2025) campaigns, with honest assessment of performance and failure modes across atmospheric regimes.

**Author:** Rylan Malarchick | Embry-Riddle Aeronautical University  
**Contact:** malarchr@my.erau.edu

---

## Two Papers

This repository supports two complementary papers:

### Paper 1: CNN-Based CBH from Thermal IR Imagery
- **Approach:** ResNet-18 and EfficientNet-B0 on 20×22px thermal IR cutouts
- **Dataset:** 380 samples from 7 flights (2 campaigns), 80/20 train/val 5-fold CV
- **Best result:** ResNet-18 pretrained, R² = 0.432 ± 0.094, MAE = 172.7 ± 17.6 m
- **Key finding:** CNNs on small thermal crops struggle — limited spatial context and small sample size yield modest performance

### Paper 2: ERA5 Feature Engineering & Domain Shift
- **Approach:** GBDT on 34 ERA5-derived features (5 base + 29 thermodynamic)
- **Dataset:** 5,500 ocean-only BL cloud observations from 6 flights
- **Key finding:** Catastrophic domain shift — LOFO R² = −5.36 across flights
- **Adaptation:** Few-shot (50 samples) recovers R² = +0.35; other methods fail

---

## Key Findings

### Performance Summary (Paper 2)

| Validation Strategy | R² | MAE | Assessment |
|--------------------|----|-----|------------|
| Pooled 5-fold CV | −2.05 | — | Inflated (cross-flight leakage) |
| Within-flight 5-fold CV | −0.51 | — | Per-flight mean, high variance |
| Leave-one-flight-out (LOFO) | **−5.36** | **518 m** | True cross-regime performance |
| Few-shot (50 samples) | **+0.35** | — | Best adaptation method |

### Performance Summary (Paper 1)

| Model | R² | MAE (m) | RMSE (m) |
|-------|----|---------|----------|
| ResNet-18 pretrained | **0.432 ± 0.094** | **172.7 ± 17.6** | 239.5 ± 23.7 |
| ResNet-18 scratch | 0.414 ± 0.127 | 169.5 ± 15.8 | 242.7 ± 28.4 |
| EfficientNet-B0 pretrained | 0.311 ± 0.109 | 201.4 ± 26.9 | 263.9 ± 26.3 |

### Why This Matters

1. **Domain shift is catastrophic:** Models trained on one atmospheric regime fail completely on another (LOFO R² = −5.36); all 6 held-out flights produce negative R²
2. **Validation methodology is critical:** Pooled CV (R² = −2.05) substantially overestimates cross-regime performance relative to LOFO (R² = −5.36)
3. **Few-shot adaptation works:** 50 labeled samples from a target flight recovers R² = +0.35 (from −5.36)
4. **Conformal prediction fails under shift:** Cross-flight coverage = 34% (target: 90%); within-flight calibration recovers 90%
5. **Feature engineering aids interpretation, not accuracy:** 34 features vs 5 base: per-flight CV R² = −0.51 vs −2.04 (marginal, both negative)

---

## Dataset

### Paper 2 (ERA5 Tabular)
- **Samples:** 5,500 ocean-only boundary-layer cloud observations
- **Flights:** 6 flights across 2 campaigns
- **Features:** 5 base ERA5 (t2m, d2m, sp, blh, tcwv) + 29 derived = **34 total**
- **Target:** CBH from CPL lidar (≤ 2 km, ocean only)

| Flight | Campaign | Samples | CBH Mean (m) |
|--------|----------|---------|-------------|
| Oct 23, 2024 | WHySMIE | 857 | 138 |
| Oct 30, 2024 | WHySMIE | 1,808 | 941 |
| Nov 4, 2024 | WHySMIE | 1,388 | 89 |
| Feb 10, 2025 | GLOVE | 608 | 380 |
| Feb 12, 2025 | GLOVE | 654 | 783 |
| Feb 18, 2025 | GLOVE | 185 | 94 |

### Paper 1 (Vision)
- **Samples:** 380 from 7 flights (5-fold CV, 304 train / 76 val)
- **Input:** 20×22 pixel thermal IR cutouts
- **Models:** ResNet-18, EfficientNet-B0 (pretrained/scratch, with/without augmentation)

---

## Domain Shift Analysis (Paper 2)

### K-S Divergence (Oct 23 vs Feb 10)
14 of 34 features show K-S = 1.0 (completely non-overlapping distributions). Only solar angle features (sza, saa) show K-S = 0.0.

### Adaptation Methods

| Method | Mean R² | Assessment |
|--------|---------|------------|
| No adaptation (LOFO) | −5.36 | Baseline |
| Instance weighting (KNN) | −3.5 | Marginal improvement |
| Instance weighting (density) | −5.5 | Comparable to baseline |
| MMD alignment | −7.9 | Worse |
| Feature selection | −6.9 | Worse |
| TrAdaBoost | +0.04 | Marginal positive |
| **Few-shot (50 samples)** | **+0.35** | **Effective** |

### Feature Importance (Full 34-Feature Model)
| Feature | Importance |
|---------|-----------|
| blh_sq | 32.3% |
| blh | 16.9% |
| stability_tcwv | 8.0% |
| moisture_gradient | 8.0% |
| blh_lcl_ratio | 4.4% |

---

## Uncertainty Quantification

| Method | Coverage | Target | Width (m) |
|--------|----------|--------|-----------|
| Split conformal (cross-flight) | 34% | 90% | 557 |
| **Per-flight calibration** | **90%** | 90% | 538 |

Conformal prediction fails across flights due to exchangeability violations. Within-flight calibration recovers the 90% target.

---

## Repository Structure

```
programDirectory/
├── preprint/                           # Both papers (LaTeX)
│   ├── paper1_nasa_er2_cbh.tex        # Paper 1: CNN vision
│   └── paper2_era5_domain_shift.tex   # Paper 2: ERA5 domain shift
├── scripts/
│   ├── paper2_rerun_v2.py             # Paper 2 reproducible pipeline
│   ├── feature_engineering.py         # Feature derivation
│   └── train_image_model.py           # Paper 1 training
├── results/
│   └── paper2_rerun_v2/               # Paper 2 v2 rerun results
├── outputs/
│   └── vision_baselines/reports/      # Paper 1 results (380-sample)
└── data/ (../data/)                   # CPL HDF5 flight data
```

---

## Reproducibility

All experiments use `np.random.seed(42)` and are fully reproducible from raw data.

### Paper 2 Rerun
```bash
# Requires ERA5 data on desktop at /mnt/two/research/NASA/ERA5_data_root/surface/
ssh desktop "cd /path/to/programDirectory && python3 -u scripts/paper2_rerun_v2.py"
# Results → results/paper2_rerun_v2/paper2_all_results_v2.json
```

### Key Result Files
| File | Contents |
|------|----------|
| `results/paper2_rerun_v2/paper2_all_results_v2.json` | All Paper 2 metrics (v2 audit) |
| `outputs/vision_baselines/reports/*.json` | Paper 1 per-model results (380 samples) |

---

## Citation

```bibtex
@article{malarchick2026cbh_vision,
  title={CNN-Based Cloud Base Height Retrieval from Thermal Infrared Imagery: 
         Lessons from NASA ER-2 Observations},
  author={Malarchick, Rylan},
  year={2026}
}

@article{malarchick2026cbh_domain,
  title={Physics-Informed Feature Engineering and Domain Shift Challenges 
         for Atmospheric Machine Learning},
  author={Malarchick, Rylan},
  year={2026}
}
```

---

## Acknowledgments

This work was conducted independently following the author's NASA OSTEM internship (May–August 2025) with NASA Goddard Space Flight Center. ERA5 data from ECMWF Copernicus Climate Data Store. CPL lidar data from NASA Goddard.

## License

MIT License — See [LICENSE](LICENSE) for details.

## Contact

**Rylan Malarchick**  
Embry-Riddle Aeronautical University  
Email: malarchr@my.erau.edu  
GitHub: [@rylanmalarchick](https://github.com/rylanmalarchick)
