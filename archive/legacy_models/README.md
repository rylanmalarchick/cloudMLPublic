# Legacy Models

This directory contains research model implementations from Sprints 3-5 (October 2024 - January 2025).

## Files

### `pytorchmodel.py` (766 lines)
Multi-modal regression models with various architectures tested during research:

**Models Implemented:**
- `MultimodalRegressionModel` - Base CNN with FiLM layers + Spatial/Temporal Attention
- `GNNModel` - Graph Neural Network variant
- `SSMModel` - Mamba State Space Model variant
- `SimpleCNNModel` - Baseline CNN architecture

**Key Components:**
- `SpatialAttention` - Pixel-level attention mechanism
- `TemporalAttention` - Frame-level attention for temporal sequences
- `MultiScaleTemporalAttention` - Multi-scale temporal processing
- `FiLMLayer` - Feature-wise Linear Modulation for conditioning on scalar metadata

**Results:** Best image model achieved R² = 0.389 (ResNet-18 ImageNet pre-trained), substantially worse than GBDT baseline (R² = 0.744).

### `mae_model.py`
Masked Autoencoder for Self-Supervised Learning (Phase 2).

**Architecture:**
- 1D patch embedding for cloud imagery (440-pixel width)
- Transformer encoder/decoder
- Optimized for GTX 1070 Ti (8GB VRAM)
- Pre-trained on ~60k unlabeled images

**Result:** SSL embeddings showed R² = 0.09 correlation with CBH - **major negative finding**. The reconstruction objective optimizes for texture/appearance rather than physically meaningful features.

**Finding:** This documented failure is cited in the academic preprint as evidence that self-supervised learning on imagery alone is insufficient for CBH retrieval.

### `scene_complexity.py`
Scene complexity analysis tools for understanding why image models failed.

**Features:**
- GLCM texture features
- Entropy calculations
- Local contrast metrics
- Image-based and lidar-based complexity metrics

**Use Case:** Potential future research to understand which scenes/conditions cause image model failures.

## Why Archived?

These implementations are preserved because:

1. **Paper documentation**: The preprint references these approaches and their failure modes
2. **Negative results**: MAE and pure CNN failures are scientifically valuable
3. **Architecture diversity**: Shows breadth of approaches attempted (CNN, GNN, SSM, Transformer)
4. **Future research**: May inform why physics-informed features outperform learned representations

## Production Models

The current production models are in `src/cbh_retrieval/`:
- GBDT (R² = 0.744, MAE = 117.4m) - **Production**
- Ensemble GBDT+CNN (R² = 0.739) - **Production**
- Simple CNN for ensemble (R² = 0.351) - Used only as ensemble component

## References

- Academic preprint: `preprint/cloudml_academic_preprint.tex`
- Sprint reports: `archive/docs/sprint_reports/`
- Production docs: `docs/cbh/`
