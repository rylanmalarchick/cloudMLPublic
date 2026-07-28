# Cloud Base Height Retrieval from NASA ER-2 Airborne Observations

> Status: complete. Research code behind two papers, kept as a reference. Not under active development.

Machine-learning retrieval of cloud base height (CBH) from NASA ER-2 data taken
during the WHySMIE (Oct 2024) and GLOVE (Feb 2025) campaigns. The headline result
is a negative one: models trained on one atmospheric regime do not transfer to
another. The repo documents that failure and what does and does not recover from
it, rather than reporting a single optimistic pooled score.

Two papers are supported here:

- Vision (thermal IR): ResNet-18 / EfficientNet-B0 on 20x22 px thermal cutouts.
  380 samples, 7 flights, 5-fold CV. Best is ResNet-18 pretrained at R2 = 0.43,
  MAE = 173 m. Small crops and a small sample cap performance.
- ERA5 tabular + domain shift: gradient-boosted trees on 34 ERA5-derived
  features, 5,500 ocean boundary-layer observations, 6 flights. Leave-one-flight-out
  R2 = -5.36. A 50-sample few-shot fit recovers R2 = +0.35.

## Results

ERA5 model, by validation strategy:

| Validation | R2 | MAE | Note |
|---|---|---|---|
| Pooled 5-fold CV | -2.05 | - | inflated by cross-flight leakage |
| Within-flight 5-fold CV | -0.51 | - | per-flight mean, high variance |
| Leave-one-flight-out | -5.36 | 518 m | true cross-regime performance |
| Few-shot (50 samples) | +0.35 | - | best adaptation |

Vision models (5-fold CV):

| Model | R2 | MAE (m) | RMSE (m) |
|---|---|---|---|
| ResNet-18 pretrained | 0.432 +/- 0.094 | 172.7 +/- 17.6 | 239.5 +/- 23.7 |
| ResNet-18 scratch | 0.414 +/- 0.127 | 169.5 +/- 15.8 | 242.7 +/- 28.4 |
| EfficientNet-B0 pretrained | 0.311 +/- 0.109 | 201.4 +/- 26.9 | 263.9 +/- 26.3 |

What the numbers say: domain shift dominates. All six held-out flights give
negative R2, and 14 of 34 features have a K-S statistic of 1.0 between Oct and Feb.
Validation choice matters: pooled CV hides the cross-regime gap that LOFO exposes.
Few-shot adaptation is the only method that recovers positive skill. Instance
weighting, MMD alignment, and feature selection do not. Split-conformal intervals
miss badly across flights (34% coverage against a 90% target) but calibrate within
a single flight.

Top features (full 34-feature model): blh_sq 32%, blh 17%, stability_tcwv 8%,
moisture_gradient 8%, blh_lcl_ratio 4%.

## Dataset

ERA5 tabular: 5,500 ocean-only boundary-layer observations, CBH from CPL lidar
(<= 2 km). Features are 5 base ERA5 fields (t2m, d2m, sp, blh, tcwv) plus 29 derived.

| Flight | Campaign | Samples | CBH mean (m) |
|---|---|---|---|
| Oct 23, 2024 | WHySMIE | 857 | 138 |
| Oct 30, 2024 | WHySMIE | 1,808 | 941 |
| Nov 4, 2024 | WHySMIE | 1,388 | 89 |
| Feb 10, 2025 | GLOVE | 608 | 380 |
| Feb 12, 2025 | GLOVE | 654 | 783 |
| Feb 18, 2025 | GLOVE | 185 | 94 |

Vision: 380 thermal-IR cutouts (20x22 px) from 7 flights.

## Layout

```
src/        models, datasets, validation (cbh_retrieval package)
scripts/    data pipeline, training, and analysis entry points
configs/    experiment configs (YAML)
outputs/    figures and result tables
docs/cbh/   model card, reproducibility and deployment guides
tests/      pytest suite
```

## Reproducing

Everything is seeded with `np.random.seed(42)`. The ERA5 rerun needs the ERA5
surface data, which is kept off-repo:

```bash
python3 -u scripts/paper2_rerun_v2.py
# -> results/paper2_rerun_v2/paper2_all_results_v2.json
```

Per-model vision results are under `outputs/vision_baselines/reports/*.json`.

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

## License and credits

MIT (see [LICENSE](LICENSE)). Rylan Malarchick, Embry-Riddle Aeronautical
University (malarchr@my.erau.edu). Independent work after a NASA OSTEM
internship (summer 2025) at Goddard Space Flight Center. ERA5 data from ECMWF
Copernicus. CPL lidar from NASA Goddard.
