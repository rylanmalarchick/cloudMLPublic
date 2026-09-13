# Cloud Base Height Retrieval from NASA ER-2 Airborne Observations

Status: complete. This repository holds the code for two papers. It is not under active development.

Machine-learning retrieval of cloud base height (CBH) from NASA ER-2 data taken
during the WHyMSIE (Oct-Nov 2024) and GLOVE (Feb 2025) campaigns. CBH labels come
from the Cloud Physics Lidar (CPL).

- Paper 1, thermal IR imagery: ResNet-18 and EfficientNet-B0 on 20x22 px IRAI
  cutouts. 380 samples from 6 flights (7 processed), stratified 5-fold CV.
  Best: ResNet-18 pretrained, R2 = 0.43, MAE = 173 m.
- Paper 2, ERA5 features and domain shift: gradient-boosted trees on 34
  ERA5-derived features. 5,500 ocean boundary-layer samples from 6 flights.
  Leave-one-flight-out R2 = -5.36. A 50-sample few-shot fit reaches R2 = +0.35.

## Results

ERA5 model, by validation strategy:

| Validation | R2 | MAE | Note |
|---|---|---|---|
| Contiguous-block 5-fold CV, flights stacked | -2.05 | - | unshuffled folds can span flights |
| Within-flight contiguous-block 5-fold CV | -0.51 | - | mean over flights |
| Leave-one-flight-out | -5.36 | 518 m | every test flight unseen |
| Few-shot (50 samples) | +0.35 | - | random target samples, upper bound |

Vision models (stratified 5-fold CV, random fold assignment):

| Model | R2 | MAE (m) | RMSE (m) |
|---|---|---|---|
| ResNet-18 pretrained | 0.432 +/- 0.094 | 172.7 +/- 17.6 | 239.5 +/- 23.7 |
| ResNet-18 scratch | 0.414 +/- 0.127 | 169.5 +/- 15.8 | 242.7 +/- 28.4 |
| EfficientNet-B0 pretrained | 0.311 +/- 0.109 | 201.4 +/- 26.9 | 263.9 +/- 26.3 |

## Dataset

ERA5 tabular: 5,500 ocean-only boundary-layer samples with CPL CBH <= 2 km.
Features are 5 base ERA5 fields (t2m, d2m, sp, blh, tcwv) plus 29 derived.

| Flight | Campaign | Samples | CBH mean (m) |
|---|---|---|---|
| Oct 23, 2024 | WHyMSIE | 857 | 138 |
| Oct 30, 2024 | WHyMSIE | 1,808 | 941 |
| Nov 4, 2024 | WHyMSIE | 1,388 | 89 |
| Feb 10, 2025 | GLOVE | 608 | 380 |
| Feb 12, 2025 | GLOVE | 654 | 783 |
| Feb 18, 2025 | GLOVE | 185 | 94 |

Vision: 380 IRAI cutouts (20x22 px) with matched CPL CBH.

The data are not in this repository. ERA5 comes from the Copernicus Climate
Data Store. CPL, IRAI, and CRS navigation files come from NASA.

## Layout

```
scripts/paper2_rerun_v2.py            Paper 2: all table and figure numbers
scripts/verify_adaptive_conformal.py  Paper 2: adaptive conformal coverage
scripts/create_integrated_features.py Paper 1: CPL/IRAI matching -> Integrated_Features.hdf5
scripts/extract_all_images.py         Paper 1: IRAI frame extraction -> data_ssl/images/
src/cbh_retrieval/vision_baselines.py Paper 1: CNN cross-validation
src/cbh_retrieval/image_dataset.py    Paper 1: image/label dataset
src/cplCompareSub.py                  CPL time conversion
scripts/figures/                      figure scripts for both papers
configs/vision_baselines_config.yaml  flight file list for Paper 1
```

## Reproducing

Install: `pip install -r requirements.txt`. Runs are seeded (42).

Paper 2:

```bash
export CLOUDML_DATA_DIR=/path/to/flights   # 23Oct24/CPL_L2_...hdf5, ...
export CLOUDML_ERA5_ROOT=/path/to/era5     # era5_surface_YYYYMMDD.nc
python scripts/paper2_rerun_v2.py          # -> results/paper2_rerun_v2/paper2_all_results_v2.json
python scripts/verify_adaptive_conformal.py
python scripts/figures/generate_paper2_v2_figures.py
```

Paper 1 (set `data_directory` in the config first):

```bash
python scripts/create_integrated_features.py --config configs/vision_baselines_config.yaml \
    --output outputs/preprocessed_data/Integrated_Features.hdf5
python scripts/extract_all_images.py --config configs/vision_baselines_config.yaml --output-dir data_ssl/images
python src/cbh_retrieval/vision_baselines.py   # -> outputs/vision_baselines/reports/*.json
python scripts/figures/generate_paper1_cnn_figures.py
```

Result files, figures, and data directories are git-ignored. There is no test suite.

## License and credits

MIT (see [LICENSE](LICENSE)). Rylan Malarchick, Embry-Riddle Aeronautical
University (malarchr@my.erau.edu). The work started during a NASA OSTEM
internship at Goddard Space Flight Center (May-Aug 2025) and continued
independently. `src/cplCompareSub.py` is by Peter Pantina.
