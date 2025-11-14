# Archive Directory

This directory contains legacy code, experimental implementations, and historical documentation from the CloudML project's development sprints (Sprints 1-5, October 2024 - February 2025).

## Purpose

The archive preserves:
- **Research provenance**: Documents what approaches were tried and why they succeeded or failed
- **Negative results**: Important null findings (e.g., MAE R²=0.09, solar angles R²=-4.46)
- **Architecture exploration**: Alternative model implementations (GNN, SSM, Multi-scale attention)
- **Historical context**: Sprint reports and development documentation

## Structure

```
archive/
├── legacy_models/          # Research model implementations (Sprints 3-5)
├── legacy_training/        # Pre-training/fine-tuning pipeline code
├── experiments/            # Experimental scripts (MAE, hybrid models)
├── utilities/              # One-time analysis utilities
├── configs/                # Deprecated configuration files
└── docs/                   # Historical documentation & sprint reports
```

## What's in Production

The production Sprint 6 code is in:
- `src/cbh_retrieval/` - Production GBDT + Ensemble models (R²=0.744)
- `tests/cbh/` - Test suite (93.5% coverage)
- `scripts/cbh/` - Production utilities
- `docs/cbh/` - Current documentation

## Why Archive vs Delete?

These files are archived rather than deleted because:

1. **Paper reproducibility**: The academic preprint references these experimental approaches
2. **Negative results matter**: Documenting failed approaches (SSL, MAE, pure image models) is valuable
3. **Future research**: Someone may want to build upon or understand why certain approaches didn't work
4. **Knowledge preservation**: Shows the evolution from image-based models to physics-informed features

## Key Archived Findings

### Failed Approaches (Documented in preprint)
- **Self-Supervised Learning (MAE)**: R² = 0.09 - reconstruction objective misaligned with CBH prediction
- **Solar angles only**: Within-flight R² = 0.70, cross-flight R² = -4.46 (temporal confounding)
- **Image-only CNNs**: R² = 0.320 ± 0.152 (vs GBDT: R² = 0.744 ± 0.037)

### Successful Evolution
- Sprint 3: Physical baseline (GBDT) R² = 0.668
- Sprint 4: Hybrid CNNs failed to beat baseline
- Sprint 5: Temporal ViT R² = 0.728 (breakthrough, but complex)
- Sprint 6: **Production GBDT R² = 0.744** (simpler, better)

## Reference

For current project status, see:
- `README.md` (project root)
- `docs/cbh/README.md` (production documentation)
- `preprint/cloudml_academic_preprint.tex` (academic paper)
