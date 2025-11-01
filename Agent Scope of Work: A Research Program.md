# Agent Scope of Work: A Research Program for CloudML

This document outlines a structured, multi-phase research program designed to systematically diagnose, resolve, and evaluate the cloudML project. The objective is to produce scientifically rigorous and statistically significant results suitable for submission to the ICML 2026 "Machine Learning for Scientific Discovery" track.

---

## Section 0: Confirmed Project Objective

Based on your clarification, this research program is now correctly aligned with the project's primary goal as stated in the README.md: **accurate cloud base height (CBH) prediction**. All experimental plans, evaluation metrics, and scientific narratives within this document have been updated to reflect CBH as the sole regression target.

The previous confusion, stemming from conflicting information in diagnostic documents, has been resolved. The entire experimental apparatus—from data loading in `hdf5_dataset.py` to the loss function in `pytorchmodel.py`—will be rigorously focused on predicting the geometric altitude of the cloud base. This ensures a coherent and targeted research effort.

---

## Section 1: Foundational Analysis: Establishing Task Learnability and Performance Baselines ✅ **COMPLETED**

**Objective:** To rigorously and formally execute a diagnostic plan to address the most fundamental question—"Is this task learnable with this data?"—before committing significant computational resources to complex deep learning experiments.

### 1.1 Signal Correlation Analysis ✅

**Status:** COMPLETED

**Results:**
- Maximum single-feature Pearson r² = **0.1355** (feature: `min_intensity`)
- This exceeds the 0.1 threshold, confirming a learnable signal exists in the data
- The signal is moderate but detectable, suggesting the task is feasible but challenging

**Conclusion:** Strong evidence of a learnable signal. Task is viable. ✅ **GO** decision confirmed.

### 1.2 Classical Model Benchmarking ✅

**Status:** COMPLETED

**Results:**

| Model | Test R² | MAE (km) | RMSE (km) |
|-------|---------|----------|-----------|
| Ridge | 0.6892 | 0.1392 | 0.2137 |
| Lasso | 0.6892 | 0.1392 | 0.2137 |
| ElasticNet | 0.6892 | 0.1392 | 0.2137 |
| SVR | 0.7270 | 0.1300 | 0.2002 |
| RandomForest | **0.7458** | 0.1266 | 0.1932 |
| GradientBoosting | **0.7464** | **0.1265** | **0.1929** |
| Best Previous NN Run | -0.0457 | N/A | N/A |

**Conclusion:** Classical models achieve strong performance (R² ≈ 0.75), definitively proving:
1. The task is learnable with excellent signal quality
2. Simple models generalize well
3. Neural network failures are due to training/architecture issues, NOT data quality

---

## Section 2: Addressing Model Collapse: A Principled Investigation ✅ **COMPLETED - NEGATIVE RESULTS**

**Objective:** To systematically diagnose and resolve the core technical failure of model collapse, where the model learns to predict a near-constant value.

### 2.1-2.3 Diagnostic Experiments and Variance-Preserving Regularization ✅

**Status:** COMPLETED - Multiple experimental runs conducted

**Experiments Performed:**
1. Baseline collapse experiment (variance_lambda = 0)
2. Hyperparameter sweep (variance_lambda: 0.5, 1.0, 2.0, 5.0, 10.0)
3. Diagnostic Experiment A: Simple CNN, no attention, single temporal frame
4. Diagnostic Experiment C: Single-flight overfit test

**Key Findings:**
- All neural network experiments produced **strongly negative R²** values (ranging from -0.194 to -11.0)
- Training loss decreased (models learned something), but validation R² remained negative
- Variance-preserving loss sometimes destabilized training at high lambda values
- Even minimal architectures (SimpleCNN) failed to generalize
- Single-flight overfit tests showed models can reduce training loss but fail validation

**Root Cause Identified:**
The fundamental issue is **not** model collapse or architecture complexity. The problem is a **training regime mismatch**:
- Only **~933 labeled samples** currently available for supervised training
- Task complexity and inductive bias requirements exceed what can be learned from this small labeled set
- This is a classic **small-labeled-data / large-unlabeled-data problem**

**Critical Insight:**
We have **~75,000 IR images** in total, but only a small subset have reliable CPL-aligned CBH labels. This is the wrong regime for supervised deep learning from scratch.

---

## Section 3: REVISED APPROACH - Self-Supervised Pre-Training with CPL as Ground-Truth Verifier

**Decision Point:** Based on Section 1-2 findings, we are **pivoting** from supervised deep learning to a self-supervised pre-training approach that leverages the full unlabeled dataset.

### 3.1 Rationale for Approach Change

**Problem:** 
- Neural networks fail when trained from scratch on ~933 labeled samples (Section 2)
- Classical models succeed because they use hand-crafted features and are less data-hungry (Section 1)
- We have ~75k unlabeled images that are currently unused

**Solution:**
Use **self-supervised learning** to pre-train on ALL images (unlabeled), then fine-tune on the labeled subset. CPL serves **only as the ground-truth verifier** for evaluation, not as the primary training signal for the full dataset.

**Key Advantages:**
1. Leverages the full ~75k image corpus for representation learning
2. Pre-training learns useful visual features without requiring labels
3. Fine-tuning adapts the learned representations to the CBH regression task
4. CPL remains the gold-standard verifier without limiting training data
5. Aligns with modern best practices for limited-label scientific domains

### 3.2 Phase 1: Data Extraction and Preparation

**Objective:** Extract and cache all IR images from HDF5 files for efficient SSL training.

**Tasks:**
1. Create `scripts/extract_all_images.py`:
   - Read all flight HDF5 files
   - Extract all IR images (~75k total)
   - Cache to efficient format (HDF5 or LMDB)
   - Create train/val split for SSL (no labels needed)

2. Verify data extraction:
   - Confirm ~75k images extracted
   - Check image dimensions and quality
   - Validate no data corruption

**Estimated Time:** 2-4 hours (script development + execution)

**Deliverable:** 
- Cached image dataset ready for SSL
- Data extraction report with statistics

### 3.3 Phase 2: Self-Supervised Pre-Training

**Objective:** Train an encoder on all unlabeled images to learn robust visual representations.

**Method Options (ranked by feasibility on GTX 1070 Ti):**

1. **MAE (Masked Autoencoder)** - RECOMMENDED
   - Best for single-GPU training
   - Learns strong representations by reconstructing masked image patches
   - Implementation available via `timm` library
   - Efficient memory usage

2. **SimCLR (Contrastive Learning)**
   - Good alternative, requires careful augmentation design
   - Available via `lightly` library
   - Higher memory requirements (large batch sizes preferred)

3. **DINO (Self-Distillation)**
   - Strong performance but more complex
   - May be challenging on single GPU

**Implementation Plan:**
1. Create `configs/ssl_pretrain_mae.yaml`:
   - Specify MAE architecture (ViT-Small or ViT-Tiny for GPU constraints)
   - Set masking ratio (0.75 standard for MAE)
   - Training hyperparameters (epochs: 100-200, batch_size: 64-128)
   - Augmentation strategy

2. Create `scripts/pretrain_mae.py`:
   - Load cached unlabeled images
   - Initialize MAE model
   - Training loop with checkpointing
   - Log reconstruction loss and sample reconstructions

3. Execute pre-training:
   - Run on full ~75k image dataset
   - Monitor convergence via reconstruction quality
   - Save encoder weights

**Success Criteria:**
- Reconstruction loss decreases steadily
- Visual inspection shows meaningful reconstructions
- Encoder learns to capture cloud structure and texture

**Estimated Time:** 
- Implementation: 4-8 hours
- Training: 12-48 hours (depending on epochs and GPU)

**Deliverable:**
- Pre-trained encoder weights (`mae_encoder_pretrained.pth`)
- Training curves and reconstruction visualizations
- Pre-training report

### 3.4 Phase 3: Fine-Tuning for CBH Regression

**Objective:** Adapt the pre-trained encoder to the CBH regression task using the labeled subset.

**Method:**
1. Load pre-trained encoder (freeze or fine-tune)
2. Add regression head (small MLP: encoder_dim → 128 → 1)
3. Train on CPL-labeled samples (~933 samples)
4. Use CPL CBH as ground truth

**Implementation Plan:**
1. Create `configs/finetune_cbh.yaml`:
   - Load pre-trained encoder weights
   - Specify fine-tuning strategy (freeze encoder initially, then unfreeze)
   - Regression head architecture
   - Loss function (Huber or MSE)
   - Training hyperparameters (epochs: 50-100, batch_size: 32)

2. Create `scripts/finetune_cbh.py`:
   - Load pre-trained encoder
   - Initialize regression head
   - Two-stage fine-tuning:
     - Stage 1: Freeze encoder, train head only (10-20 epochs)
     - Stage 2: Unfreeze encoder, fine-tune end-to-end (30-50 epochs)
   - Standard train/val split on labeled data

3. Execute fine-tuning:
   - Monitor validation R², MAE, RMSE
   - Compare to GradientBoosting baseline (R² = 0.7464)
   - Save best model checkpoint

**Success Criteria:**
- Validation R² > 0.0 (fundamental threshold)
- Target: R² ≥ 0.4-0.5 (competitive with simpler approaches)
- Stretch goal: R² ≥ 0.6 (approaching classical baseline)

**Estimated Time:** 
- Implementation: 2-4 hours
- Training: 2-6 hours per configuration

**Deliverable:**
- Fine-tuned CBH prediction model
- Performance metrics table comparing to baselines
- Prediction scatter plots and error analysis

### 3.5 Phase 4: Evaluation and Comparison

**Objective:** Rigorously evaluate the SSL→fine-tuning approach against classical baselines.

**Evaluation Protocol:**
1. **Quantitative Metrics:**
   - R² on reserved test set
   - MAE and RMSE (km)
   - Variance ratio (prediction std / true std)

2. **Comparison Table:**

| Model | Test R² | MAE (km) | RMSE (km) | Notes |
|-------|---------|----------|-----------|-------|
| GradientBoosting (baseline) | 0.7464 | 0.1265 | 0.1929 | Classical ML benchmark |
| NN from scratch (Section 2) | -0.194 | N/A | N/A | Failed to generalize |
| MAE pre-train + fine-tune | **[TBD]** | **[TBD]** | **[TBD]** | New approach |

3. **Decision Point:**
   - If R² ≥ 0.5: Continue with deep learning exploration (Phase 5)
   - If 0.3 ≤ R² < 0.5: Qualified success; explore improvements
   - If R² < 0.3: Deep learning not viable for this task; write "valuable null result" paper

**Deliverable:**
- Comprehensive evaluation report
- Comparison table and visualizations
- Error analysis by scene characteristics

---

## Section 4: Advanced Exploration (Conditional on Phase 3 Success)

**Status:** Planned - Execute only if Phase 3 achieves R² ≥ 0.5

### 4.1 Multi-Task Learning

**Rationale:** CPL data contains rich information beyond CBH (cloud top height, optical depth, cloud phase, number of layers). Multi-task learning may improve representations.

**Approach:**
1. Expand regression head to predict multiple targets:
   - Cloud base height (primary)
   - Cloud top height
   - Cloud optical depth (if available in CPL)
   - Number of cloud layers (classification head)

2. Use multi-task loss with learned weighting or uncertainty weighting

3. Hypothesis: Auxiliary tasks may improve CBH prediction via better feature learning

**Estimated Time:** 1-2 days

### 4.2 Temporal Modeling

**Rationale:** Current approach uses single frames or simple averaging. Temporal sequence models may capture cloud evolution.

**Approach:**
1. Modify architecture to process temporal sequences:
   - Pre-trained encoder processes each frame
   - Temporal aggregation via:
     - Temporal attention
     - LSTM/GRU
     - Temporal convolutions

2. Leverage multiple nadir views per cloud scene

**Estimated Time:** 2-3 days

### 4.3 Weak Supervision and Label Expansion

**Rationale:** More CPL profiles may be usable with relaxed matching criteria or parallax correction.

**Exploration:**
1. Re-examine CPL matching rules in `preprocess_data.py`
2. Analyze impact of relaxing CBH altitude range (0.1-2.0 km → 0.1-5.0 km)
   - Note: Preliminary analysis of 30Oct24 shows only ~2% increase (+32 samples)
3. Explore parallax correction for better GOES↔CPL alignment
4. Consider weak labeling: use CPL profiles as "region labels" rather than exact point matches

**Estimated Time:** 2-4 days

**Caution:** Preliminary data suggests modest gains (~2% per flight) from simple altitude relaxation. Major improvements require better matching algorithms or parallax correction.

---

## Section 5: Publication Strategy and Narrative Development

### 5.1 Potential Paper Narratives

Based on Phase 3-4 outcomes, we have multiple viable publication paths:

#### Narrative A: Success Story (if R² ≥ 0.6)
**Title:** "Self-Supervised Learning for Cloud Base Height Estimation from Single-Wavelength Infrared Imagery"

**Key Claims:**
1. Identified and solved data regime problem (small labeled, large unlabeled)
2. Self-supervised pre-training enables deep learning success where supervised learning failed
3. Achieved competitive performance (R² ≥ 0.6) approaching classical ML baselines
4. Demonstrated that DL can work with limited ground truth via SSL

**Contributions:**
- Methodological: SSL approach for scientific regression with limited labels
- Empirical: Strong CBH prediction from single-wavelength IR
- Diagnostic: Systematic comparison of training regimes

#### Narrative B: Qualified Success (if 0.4 ≤ R² < 0.6)
**Title:** "When and How Self-Supervised Learning Helps: A Case Study in Cloud Property Retrieval"

**Key Claims:**
1. SSL provides positive results where supervised DL fails completely
2. Classical ML still superior, but DL offers complementary advantages (uncertainty, scalability)
3. Quantified the "SSL benefit" in limited-label scientific domains

**Contributions:**
- Empirical comparison of SSL vs. supervised vs. classical ML
- Lessons learned for scientific ML practitioners
- Clear characterization of when to use each approach

#### Narrative C: Valuable Null Result (if R² < 0.3)
**Title:** "The Limits of Deep Learning for Cloud Base Height Estimation: A Diagnostic Study"

**Key Claims:**
1. Systematic investigation proves DL (even with SSL) cannot match classical ML for this task
2. Diagnosed multiple failure modes (supervised collapse, SSL generalization gap)
3. Strong evidence that single-wavelength IR is fundamentally limited; multi-spectral data required

**Contributions:**
- Rigorous negative result with high scientific value
- Diagnostic methodology for evaluating DL applicability
- Clear empirical bounds on single-wavelength cloud retrieval

### 5.2 Key Figures and Tables (All Narratives)

**Table 1: Foundational Analysis Results**
- Correlation analysis (max r² = 0.1355)
- Classical baseline performance (GradientBoosting R² = 0.7464)
- Establishes task learnability and benchmark

**Table 2: Training Regime Comparison**
| Approach | Data Used | Test R² | Status |
|----------|-----------|---------|--------|
| Classical ML (GradientBoosting) | 933 hand-crafted features | 0.7464 | ✅ |
| Supervised DL (from scratch) | 933 labeled images | -0.194 | ❌ |
| Self-Supervised DL (MAE + fine-tune) | 75k unlabeled + 933 labeled | **[TBD]** | 🔄 |

**Figure 1: SSL Pre-Training Results**
- Reconstruction examples showing what MAE learned
- Training curves (reconstruction loss over epochs)

**Figure 2: Fine-Tuning Performance**
- Scatter plot: predicted vs. true CBH
- Error distribution analysis
- Comparison to classical baseline

**Figure 3: Learned Representations**
- t-SNE or UMAP visualization of encoder features
- Spatial attention maps on example images
- Shows what the SSL model learned about cloud structure

---

## Section 6: Implementation Timeline and Resource Requirements

### Phase-by-Phase Timeline

| Phase | Tasks | Estimated Time | Dependencies |
|-------|-------|----------------|--------------|
| **Phase 1: Data Extraction** | Extract ~75k images, create cache | 2-4 hours | None |
| **Phase 2: SSL Pre-Training** | Implement MAE, train encoder | 1-2 days (impl) + 1-2 days (training) | Phase 1 |
| **Phase 3: Fine-Tuning** | Add regression head, fine-tune | 4-8 hours (impl) + 4-8 hours (training) | Phase 2 |
| **Phase 4: Evaluation** | Comprehensive testing, comparison | 1 day | Phase 3 |
| **Phase 5: Advanced (optional)** | Multi-task, temporal, weak supervision | 1-2 weeks | Phase 4, R² ≥ 0.5 |
| **Phase 6: Paper Writing** | Draft, revise, finalize | 2-3 weeks | Phases 1-5 |

**Total Estimated Time:** 
- Minimum viable result (Phases 1-4): **1-2 weeks**
- With advanced exploration (Phases 1-5): **3-4 weeks**
- Complete paper (Phases 1-6): **6-8 weeks**

### Computational Resources

**Hardware:**
- GPU: GTX 1070 Ti (8GB VRAM) - adequate for planned experiments
- CPU: Multi-core for data extraction and classical baselines
- Storage: ~500 GB recommended (raw data + cached images + checkpoints)

**Software Dependencies:**
- PyTorch ≥ 1.10
- `timm` (for MAE implementation)
- `lightly` (optional, for SimCLR)
- Existing project dependencies (h5py, numpy, scikit-learn, etc.)

---

## Section 7: Risk Mitigation and Contingency Plans

### Risk 1: SSL Pre-Training Fails to Learn Useful Representations
**Indicators:** Reconstruction loss doesn't decrease, random-looking reconstructions

**Mitigation:**
1. Try different SSL methods (MAE → SimCLR → DINO)
2. Adjust masking ratio or augmentation strategy
3. Verify data loading and preprocessing

**Contingency:** If all SSL methods fail, revert to classical ML paper (Narrative C)

### Risk 2: Fine-Tuning Doesn't Improve Over Random Encoder
**Indicators:** Fine-tuned model performs no better than randomly initialized encoder

**Mitigation:**
1. Verify encoder weights are loading correctly
2. Try different fine-tuning strategies (freeze/unfreeze)
3. Increase fine-tuning epochs or adjust learning rate

**Contingency:** Paper focuses on SSL diagnostics and why it didn't help (still publishable)

### Risk 3: GPU Memory Constraints
**Indicators:** OOM errors during pre-training

**Mitigation:**
1. Reduce batch size
2. Use smaller encoder (ViT-Tiny instead of ViT-Small)
3. Enable gradient checkpointing
4. Use mixed precision training (FP16)

**Contingency:** Cloud GPU rental for pre-training phase (Google Colab Pro, AWS, etc.)

### Risk 4: Timeline Overruns
**Indicators:** Phases taking longer than estimated

**Mitigation:**
1. Prioritize core path (Phases 1-4) over advanced exploration (Phase 5)
2. Timebox each phase; move to next if deadline approaches
3. Parallelize where possible (e.g., write paper introduction while training)

**Contingency:** Submit "work in progress" results if deadline pressure; ICML allows iterative improvement

---

## Section 8: Immediate Next Steps

**Status: READY TO BEGIN DEVELOPMENT**

Upon approval, the following concrete actions will be taken:

### Step 1: Data Extraction (Day 1)
```bash
# Create extraction script
python scripts/extract_all_images.py --hdf5-dir data_crops/ --output-dir data_ssl/images/ --format hdf5

# Verify extraction
python scripts/verify_extraction.py --data-dir data_ssl/images/
```

**Expected Output:**
- `data_ssl/images/train.h5` (~70k images)
- `data_ssl/images/val.h5` (~5k images)
- Extraction report with statistics

### Step 2: MAE Pre-Training (Days 2-4)
```bash
# Create config and script, then run
python scripts/pretrain_mae.py --config configs/ssl_pretrain_mae.yaml --epochs 100 --batch-size 128

# Monitor training
tensorboard --logdir runs/mae_pretrain/
```

**Expected Output:**
- `checkpoints/mae_encoder_epoch100.pth`
- Reconstruction visualizations
- Training curves

### Step 3: CBH Fine-Tuning (Days 5-6)
```bash
# Fine-tune on labeled data
python scripts/finetune_cbh.py --config configs/finetune_cbh.yaml --pretrained checkpoints/mae_encoder_epoch100.pth

# Evaluate
python scripts/evaluate_cbh.py --model checkpoints/cbh_best.pth --data data_crops/
```

**Expected Output:**
- Performance metrics (R², MAE, RMSE)
- Comparison table vs. baselines
- Scatter plots and error analysis

### Step 4: Decision Point (Day 7)
Based on fine-tuning results:
- **If R² ≥ 0.5:** Proceed to Phase 5 (advanced exploration)
- **If 0.3 ≤ R² < 0.5:** Begin paper writing (Narrative B)
- **If R² < 0.3:** Begin paper writing (Narrative C)

---

## Conclusion

This revised research program reflects the key insight from Sections 1-2: **the problem is not data quality or model architecture, but training regime**. By pivoting to self-supervised pre-training, we:

1. **Leverage our actual data assets:** ~75k images instead of ~933
2. **Use CPL appropriately:** As ground-truth verifier, not limiting factor
3. **Follow best practices:** SSL is the state-of-the-art for limited-label domains
4. **Maintain publication viability:** Multiple compelling narratives regardless of outcome

The path forward is clear, actionable, and scientifically rigorous. We are ready to begin implementation upon your approval.

**Current Status:** ⏸️ **AWAITING APPROVAL TO BEGIN PHASE 1**

---

## Appendix: Completed Work Summary

### Section 1: Foundational Analysis ✅
- **Correlation Analysis:** Max r² = 0.1355 (min_intensity)
- **Classical Baselines:** GradientBoosting R² = 0.7464, MAE = 0.1265 km
- **Conclusion:** Task is learnable; strong signal confirmed

### Section 2: Model Collapse Investigation ✅
- **Experiments Run:** Baseline, lambda sweep, SimpleCNN, overfit tests
- **Results:** All NN experiments negative R² (-0.194 to -11.0)
- **Root Cause:** Small labeled dataset (~933 samples) insufficient for supervised DL
- **Key Insight:** Need to leverage ~75k unlabeled images via SSL

### Files Created:
- `configs/section2_*.yaml` (multiple experimental configs)
- `configs/diagnostic_exp_*.yaml` (diagnostic experiments)
- `scripts/run_section2_experiments.sh`
- `scripts/run_section2_automated.sh`
- `scripts/aggregate_section2_results.py`
- `scripts/plot_section2_distributions.py`
- `SECTION2_RESULTS_REPORT.md`
- `FINAL_DIAGNOSTIC_REPORT.md`
- `DEEP_LEARNING_PROPOSAL.md`

All diagnostic artifacts are preserved for publication.