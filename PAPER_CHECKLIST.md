# Paper Checklist: Cloud Base Height Prediction

## Overview

This checklist ensures all experiments, results, and analyses are complete for publication.

---

## 🎯 Phase 1: Baseline Model (REQUIRED)

**Goal**: Establish the strongest possible model as your reference point.

### Training
- [ ] Run baseline training with optimized settings (2-3 hours)
  - Configuration: `colab_optimized.yaml`
  - Batch size: 32
  - Temporal frames: 5
  - Epochs: 50 (with early stopping)
  - All features enabled (spatial attention, temporal attention, augmentation)

### Verification
- [ ] Baseline training completed without OOM errors
- [ ] GPU utilization reached 70-80% (10-12GB)
- [ ] Early stopping triggered (model converged)
- [ ] Final validation metrics recorded:
  - [ ] MAE (Mean Absolute Error)
  - [ ] RMSE (Root Mean Squared Error)
  - [ ] R² (Coefficient of Determination)
  - [ ] MAPE (Mean Absolute Percentage Error)

### Outputs to Check
- [ ] Model checkpoint saved: `/content/drive/MyDrive/CloudML/models/trained/baseline_paper_*.pth`
- [ ] Training curves saved: `/content/drive/MyDrive/CloudML/logs/tensorboard/`
- [ ] Evaluation plots saved: `/content/drive/MyDrive/CloudML/plots/`
- [ ] Metrics CSV saved: `/content/drive/MyDrive/CloudML/logs/csv/`

---

## 🔬 Phase 2: Ablation Studies (REQUIRED)

**Goal**: Isolate the contribution of each component to justify design choices.

### Ablation 1: Solar Angle Modes
- [ ] **Zenith Only (SZA)**: Remove solar azimuth angle
  - Expected: Small performance drop if azimuth provides directional shadow info
  - [ ] Experiment completed
  - [ ] Metrics recorded
  - [ ] Δ MAE vs baseline: _______

### Ablation 2: Spatial Attention
- [ ] **No Spatial Attention**: Disable spatial attention mechanism
  - Expected: Moderate drop - spatial attention focuses on cloud regions
  - [ ] Experiment completed
  - [ ] Metrics recorded
  - [ ] Δ MAE vs baseline: _______

### Ablation 3: Temporal Attention
- [ ] **No Temporal Attention**: Disable temporal attention mechanism
  - Expected: Moderate drop - temporal attention weighs informative frames
  - [ ] Experiment completed
  - [ ] Metrics recorded
  - [ ] Δ MAE vs baseline: _______

### Ablation 4: Combined Attention
- [ ] **No Attention (Both)**: Disable spatial AND temporal attention
  - Expected: Significant drop - demonstrates full attention mechanism value
  - [ ] Experiment completed
  - [ ] Metrics recorded
  - [ ] Δ MAE vs baseline: _______

### Ablation 5: Data Augmentation
- [ ] **No Augmentation**: Disable data augmentation
  - Expected: Small-moderate drop - augmentation aids generalization
  - [ ] Experiment completed
  - [ ] Metrics recorded
  - [ ] Δ MAE vs baseline: _______

### Ablation 6: Loss Function
- [ ] **MAE Loss**: Replace Huber loss with simple MAE
  - Expected: Slight drop - Huber loss is robust to outliers
  - [ ] Experiment completed
  - [ ] Metrics recorded
  - [ ] Δ MAE vs baseline: _______

### Ablation 7: Temporal Context
- [ ] **Fewer Temporal Frames**: Reduce from 5 to 3 frames
  - Expected: Moderate drop - less temporal context available
  - [ ] Experiment completed
  - [ ] Metrics recorded
  - [ ] Δ MAE vs baseline: _______

### Ablation 8: Architecture Comparison
- [ ] **Simple CNN Baseline**: Replace Transformer with basic CNN
  - Expected: Significant drop - demonstrates architecture superiority
  - [ ] Experiment completed
  - [ ] Metrics recorded
  - [ ] Δ MAE vs baseline: _______

### Ablation Summary
- [ ] All 8 ablations completed (~6-8 hours total)
- [ ] Results aggregated into single table
- [ ] Relative performance (Δ metrics) calculated for each
- [ ] Statistical significance tested (if applicable)

---

## 📊 Phase 3: Results Analysis (REQUIRED)

### Quantitative Results
- [ ] **Table 1**: Main results comparing baseline vs all ablations
  - Columns: Experiment, MAE, RMSE, R², MAPE, Δ MAE (%)
  - [ ] Baseline row (reference)
  - [ ] All ablation rows
  - [ ] Best and worst highlighted

- [ ] **Error Statistics**: Computed for baseline model
  - [ ] Mean error
  - [ ] Median error
  - [ ] Error quartiles (25th, 75th percentile)
  - [ ] Maximum error
  - [ ] Error distribution plots

- [ ] **Per-Flight Performance**: Breakdown by flight
  - [ ] MAE for each flight
  - [ ] Flight characteristics noted (if relevant: cloud types, conditions)
  - [ ] Variance across flights analyzed

### Qualitative Results
- [ ] **Attention Visualizations**: Generate attention maps
  - [ ] Spatial attention maps (3-5 examples)
  - [ ] Temporal attention weights (3-5 examples)
  - [ ] Show attention focusing on cloud regions

- [ ] **Error Analysis**: Visualize model performance
  - [ ] Scatter plot: Predicted vs Ground Truth
  - [ ] Residual plot: Error vs CBH
  - [ ] Error distribution histogram
  - [ ] Worst-case examples identified

- [ ] **Training Curves**: Extract from TensorBoard
  - [ ] Training loss over epochs
  - [ ] Validation loss over epochs
  - [ ] Learning rate schedule (if applicable)
  - [ ] Early stopping point marked

---

## 🔄 Phase 4: Leave-One-Out Cross-Validation (RECOMMENDED)

**Goal**: Demonstrate generalization across different atmospheric conditions.

### LOO Experiments
- [ ] **Flight 1 (10Feb25) held out**: Train on 5 others
  - [ ] Experiment completed
  - [ ] MAE on held-out flight: _______

- [ ] **Flight 2 (30Oct24) held out**: Train on 5 others
  - [ ] Experiment completed
  - [ ] MAE on held-out flight: _______

- [ ] **Flight 3 (04Nov24) held out**: Train on 5 others
  - [ ] Experiment completed
  - [ ] MAE on held-out flight: _______

- [ ] **Flight 4 (23Oct24) held out**: Train on 5 others
  - [ ] Experiment completed
  - [ ] MAE on held-out flight: _______

- [ ] **Flight 5 (18Feb25) held out**: Train on 5 others
  - [ ] Experiment completed
  - [ ] MAE on held-out flight: _______

- [ ] **Flight 6 (12Feb25) held out**: Train on 5 others
  - [ ] Experiment completed
  - [ ] MAE on held-out flight: _______

### LOO Analysis
- [ ] Average LOO MAE computed
- [ ] Standard deviation across folds computed
- [ ] Per-fold performance table created (Table 2)
- [ ] Generalization capability discussed

**Note**: LOO CV takes 8-12 hours. Requires Colab Pro or can be split across sessions.

---

## 📈 Phase 5: Additional Experiments (OPTIONAL BUT RECOMMENDED)

### Architecture Variants
- [ ] **GNN Architecture**: Test graph neural network approach
  - Expected: Different spatial modeling paradigm
  - [ ] Experiment completed
  - [ ] Metrics recorded vs baseline

- [ ] **SSM (Mamba) Architecture**: Test state-space model
  - Expected: Different temporal modeling paradigm
  - [ ] Experiment completed
  - [ ] Metrics recorded vs baseline

### Hyperparameter Sensitivity
- [ ] **Learning Rate**: Test 0.0001, 0.001, 0.01
  - [ ] Results documented
  
- [ ] **Batch Size Impact**: Test 16, 32, 48
  - [ ] GPU memory usage recorded
  - [ ] Training speed recorded
  - [ ] Final performance recorded

- [ ] **Temporal Frames**: Test 3, 5, 7 frames
  - [ ] Memory impact documented
  - [ ] Performance vs frames plotted

### Robustness Tests
- [ ] **Noise Robustness**: Add synthetic noise to inputs
- [ ] **Missing Data**: Test with missing angles or degraded images
- [ ] **Extreme CBH Values**: Performance on edge cases

---

## 📝 Phase 6: Paper Writing

### Figures
- [ ] **Figure 1**: Model architecture diagram
  - CNN backbone, FiLM layers, attention mechanisms
  
- [ ] **Figure 2**: Attention visualization examples
  - 2x3 grid: 2 examples × (image, spatial attn, temporal weights)
  
- [ ] **Figure 3**: Ablation results bar chart
  - Δ MAE for each ablation vs baseline
  
- [ ] **Figure 4**: Predicted vs Ground Truth scatter
  - Color-coded by flight or error magnitude
  
- [ ] **Figure 5**: Training curves
  - Train/val loss over epochs with early stopping marked

### Tables
- [ ] **Table 1**: Main Results - Baseline & Ablations
  ```
  | Experiment              | MAE (km) | RMSE (km) | R²    | MAPE (%) | Δ MAE (%) |
  |-------------------------|----------|-----------|-------|----------|-----------|
  | Baseline (Full Model)   | X.XXX    | X.XXX     | 0.XXX | XX.XX    | -         |
  | - w/o Spatial Attn      | X.XXX    | X.XXX     | 0.XXX | XX.XX    | +XX.X     |
  | - w/o Temporal Attn     | X.XXX    | X.XXX     | 0.XXX | XX.XX    | +XX.X     |
  | ...                     | ...      | ...       | ...   | ...      | ...       |
  ```

- [ ] **Table 2**: Leave-One-Out Cross-Validation (if completed)
  ```
  | Held-Out Flight | Training Flights | MAE (km) | RMSE (km) | R²    |
  |-----------------|------------------|----------|-----------|-------|
  | 10Feb25         | 5 others         | X.XXX    | X.XXX     | 0.XXX |
  | ...             | ...              | ...      | ...       | ...   |
  ```

- [ ] **Table 3**: Computational Requirements
  ```
  | Configuration   | GPU Memory | Training Time | Inference Time |
  |-----------------|------------|---------------|----------------|
  | Baseline        | ~11 GB     | ~2.5 hours    | ~X ms/sample   |
  | CNN Baseline    | ~8 GB      | ~1.5 hours    | ~X ms/sample   |
  ```

### Sections to Write
- [ ] **Abstract**: 150-250 words summarizing contribution
- [ ] **Introduction**: Problem motivation, related work, contributions
- [ ] **Methods**:
  - [ ] Dataset description (flights, sensors, preprocessing)
  - [ ] Model architecture (CNN, FiLM, attention mechanisms)
  - [ ] Training procedure (pretraining, fine-tuning, hyperparameters)
- [ ] **Experiments**:
  - [ ] Baseline results
  - [ ] Ablation studies with analysis
  - [ ] LOO cross-validation (if completed)
- [ ] **Results**:
  - [ ] Quantitative metrics with comparisons
  - [ ] Qualitative visualizations with discussion
  - [ ] Error analysis and failure cases
- [ ] **Discussion**:
  - [ ] Key findings interpretation
  - [ ] Limitations and future work
  - [ ] Broader impact
- [ ] **Conclusion**: Summary of contributions

---

## 🗂️ Phase 7: Reproducibility & Code Release

### Code Documentation
- [ ] README.md up to date with:
  - [ ] Installation instructions
  - [ ] Data requirements and sources
  - [ ] Training commands
  - [ ] Inference examples
  
- [ ] Requirements.txt pinned to exact versions
- [ ] Config files documented with comments
- [ ] Model checkpoints prepared for release (if sharing)

### Supplementary Materials
- [ ] **Supplementary PDF**: Additional results, plots, ablations
- [ ] **Code Repository**: GitHub link in paper
- [ ] **Model Weights**: Hosted (Zenodo, Hugging Face, etc.)
- [ ] **Demo Notebook**: Inference example on sample data

### Data Sharing
- [ ] Data usage permissions verified
- [ ] Sample dataset prepared (if allowed)
- [ ] Data preprocessing scripts documented
- [ ] Instructions for accessing full dataset

---

## ✅ Pre-Submission Checklist

### Results Verification
- [ ] All experiments ran to completion (no crashes)
- [ ] No obvious errors in logs or outputs
- [ ] Results are reproducible (reran key experiments)
- [ ] Numbers in paper match experimental outputs
- [ ] Figures are high resolution (300 DPI minimum)
- [ ] Table formatting is clean and consistent

### Code Quality
- [ ] Code runs without errors on fresh environment
- [ ] All dependencies listed in requirements.txt
- [ ] README instructions tested by independent party
- [ ] No hardcoded paths (all configurable)
- [ ] License file included (e.g., MIT)

### Paper Quality
- [ ] No typos or grammatical errors
- [ ] All figures/tables referenced in text
- [ ] Consistent notation throughout
- [ ] Related work adequately cited
- [ ] Contributions clearly stated
- [ ] Limitations honestly discussed

### Ethics & Compliance
- [ ] Data usage complies with NASA/agency policies
- [ ] No proprietary/confidential information leaked
- [ ] Author contributions clearly stated
- [ ] Funding sources acknowledged
- [ ] Conflicts of interest declared (if any)

---

## 📦 Final Deliverables

### For Paper Submission
- [ ] Paper PDF (formatted for venue)
- [ ] Supplementary materials PDF
- [ ] High-res figures (separate files)
- [ ] BibTeX file with all references
- [ ] Cover letter (if required)

### For Code Release
- [ ] GitHub repository public
- [ ] Tagged release version (e.g., v1.0-paper)
- [ ] Pre-trained model weights uploaded
- [ ] Documentation complete (README, docs/)
- [ ] Example outputs included

### For Archival
- [ ] All raw experiment outputs backed up
- [ ] Training logs archived
- [ ] Config files for each experiment saved
- [ ] Jupyter notebooks with analysis saved
- [ ] Citation information prepared

---

## 📊 Expected Results Summary

Fill this in after completing experiments:

### Baseline Performance
```
MAE:  _______ km
RMSE: _______ km
R²:   _______
MAPE: _______ %
```

### Top 3 Ablation Impacts (by Δ MAE)
1. ________________: +_____% (justify architecture choice)
2. ________________: +_____% (justify component)
3. ________________: +_____% (justify design decision)

### LOO CV Average (if completed)
```
Average MAE across 6 folds: _______ km
Standard deviation:         _______ km
Min/Max MAE:                _______ / _______ km
```

---

## 🎓 Publication Checklist

### Target Venue
- [ ] Venue selected: _____________________
- [ ] Submission deadline: _____________________
- [ ] Page limit verified: _____ pages
- [ ] Template downloaded and used
- [ ] Author guidelines reviewed

### Review Process
- [ ] Paper submitted
- [ ] Reviews received
- [ ] Rebuttal prepared (if applicable)
- [ ] Revised manuscript submitted
- [ ] Camera-ready version finalized

### Post-Acceptance
- [ ] Copyright form signed
- [ ] Presentation prepared (poster/slides)
- [ ] Code made public
- [ ] Model weights released
- [ ] Blog post / Twitter thread (optional)

---

## 💡 Tips for Success

1. **Start with Baseline**: Get one strong model working before ablations
2. **Document Everything**: Save configs, logs, commands used
3. **Monitor GPU**: Ensure you're using resources efficiently (~10-12GB)
4. **Check Intermediate Results**: Don't wait for all experiments to verify outputs
5. **Backup Regularly**: Google Drive auto-saves, but double-check
6. **Compare Fairly**: Same hyperparameters across ablations except tested component
7. **Statistical Rigor**: Run multiple seeds if variance is high (optional but good)
8. **Visualize Early**: Generate plots during training to catch issues
9. **Time Management**: Baseline (3h) + Ablations (8h) = ~11 hours minimum
10. **Session Persistence**: Use Colab Pro or break into multiple sessions

---

## 📞 Support

- **Documentation**: See `README.md`, `COLAB_SETUP.md`, `GPU_OPTIMIZATION.md`
- **Issues**: https://github.com/rylanmalarchick/cloudMLPublic/issues
- **Email**: rylan1012@gmail.com

**Good luck with your paper! 🚀📄**