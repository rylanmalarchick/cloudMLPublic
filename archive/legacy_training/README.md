# Legacy Training Code

This directory contains the pre-training/fine-tuning pipeline code from Sprints 3-5.

## Files

### `main.py`
Original training entry point with comprehensive CLI for research experiments.

**Features:**
- Command-line overrides for all hyperparameters
- Ablation study support (angles, attention, loss, augmentation, architectures)
- Pre-training + fine-tuning workflow
- Leave-one-out cross-validation

**Replaced by:** Production training in `src/cbh_retrieval/train_production_model.py`

### `pipeline.py`
Training pipeline orchestration for multi-stage training.

**Workflow:**
1. **Pre-training:** Train on single large flight
2. **Final training:** Fine-tune on all flights with pre-training flight overweighted
3. **LOO validation:** Cross-flight generalization testing

**Result:** This complex workflow was ultimately abandoned in favor of simpler stratified k-fold CV.

### `pretraining.py`
Self-supervised pre-training using SSL/MAE approaches.

**Approach:** Pre-train encoder on unlabeled images, then fine-tune on labeled CBH data.

**Result:** Failed to improve performance (MAE embeddings R² = 0.09).

### `train_model.py`
Core training loop for PyTorch models.

**Features:**
- Early stopping
- Learning rate scheduling
- Gradient clipping
- TensorBoard logging
- Multi-flight weighting

**Replaced by:** Simple scikit-learn GBDT training proved more effective.

## Why Archived?

This code is preserved to document:

1. **Multi-stage training attempt**: Shows the progression from simple to complex training schemes
2. **Failed SSL approach**: Documents why self-supervised pre-training didn't help
3. **Research workflow**: Full experimental pipeline with all options
4. **Negative results**: Complex deep learning pipeline underperformed simple GBDT

## Key Finding

The evolution from complex multi-stage training to simple GBDT demonstrates an important lesson: **domain-informed feature engineering + simple models outperforms complex end-to-end learning in data-limited regimes** (933 samples).

## Production Training

Current production training is much simpler:
- `src/cbh_retrieval/train_production_model.py` - GBDT training (50 lines)
- `src/cbh_retrieval/offline_validation_tabular.py` - Stratified k-fold CV
- No pre-training, no fine-tuning, no multi-stage pipeline

## References

- Academic preprint: `preprint/cloudml_academic_preprint.tex` (Section 3.3)
- Sprint reports: `archive/docs/sprint_reports/sprint_3_4_5_status_report.pdf`
