# SECTION 2: MODEL COLLAPSE INVESTIGATION
## Execution Guide and Summary

---

## Overview

Section 2 of the cloudML Research Program systematically investigates the **variance collapse** problem through controlled experiments with variance-preserving regularization. This section provides definitive evidence that the neural network's failure was due to training pathology, not fundamental data limitations.

### Scientific Motivation

Standard regression losses (e.g., Huber, MSE) optimize for **point-wise error minimization**, which can be satisfied by predicting a constant value near the mean. The **variance-preserving loss term** introduces a second objective:

$$L_{var} = (1 - \sigma^2_{pred} / \sigma^2_{true})^2$$

This forces the model into **multi-objective optimization**: minimize prediction error while preserving the statistical distribution of the target variable. This is especially critical in scientific domains where capturing the full range and variability of natural phenomena is as important as minimizing average error.

---

## Section 2.1: Baseline Collapse Experiment

### Objective
Quantify the model collapse phenomenon when `variance_lambda = 0` (no regularization).

### Configuration
- **File**: `configs/section2_baseline_collapse.yaml`
- **Key Setting**: `variance_lambda: 0.0`
- **Epochs**: 15 (sufficient to observe collapse)
- **Architecture**: Full transformer (to isolate training issue, not architecture)

### Expected Results
- **Validation R²**: Negative or near zero
- **Variance Ratio**: Drops toward 0-20% (model predicts near-constant)
- **Prediction Distribution**: Extremely narrow, centered on mean

### Execution
```bash
# Activate environment
source venv/bin/activate  # or appropriate for your system

# Run baseline
python main.py --config configs/section2_baseline_collapse.yaml
```

### Deliverables
1. Final negative R² score (control measurement)
2. Training log showing variance ratio decline
3. Prediction histogram showing collapsed distribution

---

## Section 2.2: Variance Lambda Hyperparameter Sweep

### Objective
Determine the optimal strength of variance-preserving regularization through systematic ablation.

### Experimental Design

| Experiment | Lambda Value | Expected Behavior |
|------------|--------------|-------------------|
| 2.2.1      | 0.5          | Weak penalty - may partially collapse |
| 2.2.2      | 1.0          | Moderate penalty - balanced trade-off |
| 2.2.3      | 2.0          | Moderate-strong - robust variance preservation |
| 2.2.4      | 5.0          | Strong penalty - watch for over-correction |
| 2.2.5      | 10.0         | Very strong - high risk of training instability |

### Configuration Files
- `configs/section2_lambda_0.5.yaml`
- `configs/section2_lambda_1.0.yaml`
- `configs/section2_lambda_2.0.yaml`
- `configs/section2_lambda_5.0.yaml`
- `configs/section2_lambda_10.0.yaml`

### Execution Options

#### Option 1: Automated Sequential Execution
```bash
bash scripts/run_section2_experiments.sh
```

This master script runs all experiments sequentially with interactive prompts.

#### Option 2: Manual Individual Execution
```bash
# Run each experiment individually
python main.py --config configs/section2_lambda_0.5.yaml
python main.py --config configs/section2_lambda_1.0.yaml
python main.py --config configs/section2_lambda_2.0.yaml
python main.py --config configs/section2_lambda_5.0.yaml
python main.py --config configs/section2_lambda_10.0.yaml
```

### Metrics to Monitor

For each run, track:
1. **Final Validation R²**: Primary performance metric
2. **Final Variance Ratio**: Should approach 100% (perfect variance matching)
3. **Average Base Loss**: Huber loss component (last 5 epochs)
4. **Average Variance Loss**: Variance penalty component (last 5 epochs)
5. **Training Stability**: Smooth curves vs. explosions/NaNs

---

## Section 2.3: Analysis and Selection

### Objective
Select the optimal `variance_lambda` through quantitative and qualitative analysis.

### Step 1: Aggregate Results

```bash
python scripts/aggregate_section2_results.py
```

**Output**:
- `diagnostics/results/section2_table2.csv` - Raw data
- `diagnostics/results/section2_summary.json` - Structured summary
- `diagnostics/results/section2_table2.txt` - Formatted for paper

This generates **Table 2** from the research program.

### Step 2: Visualize Distributions

```bash
python scripts/plot_section2_distributions.py
```

**Output**:
- `diagnostics/results/section2_distributions.png` - Grid of prediction histograms
- `diagnostics/results/section2_training_curves.png` - R² and variance ratio over epochs

### Step 3: Selection Criteria

Choose the optimal `variance_lambda` based on:

1. **Highest Validation R²** (primary)
2. **Variance Ratio Closest to 100%** (distribution matching)
3. **Training Stability** (smooth curves, no explosions)
4. **Visual Distribution Matching** (histogram comparison)

### Decision Matrix

| Lambda | R² > 0.3 | Variance 80-120% | Stable | → Decision |
|--------|----------|------------------|--------|------------|
| 0.0    | ❌       | ❌               | ✅     | Reject (collapse) |
| 0.5    | ?        | ?                | ?      | Evaluate |
| 1.0    | ?        | ?                | ?      | Evaluate |
| 2.0    | ?        | ?                | ?      | **Likely optimal** |
| 5.0    | ?        | ?                | ?      | Evaluate |
| 10.0   | ?        | ?                | ❌?    | Likely unstable |

> **Note**: The `?` entries will be filled after experiments complete.

---

## Table 2: Results Summary Template

```
variance_lambda | Final R² | Variance Ratio (%) | Avg Base Loss | Avg Var Loss | Stability
----------------|----------|-------------------|---------------|--------------|------------------
0.0 (Baseline)  | [result] | [result]          | [result]      | 0.0          | Stable (Collapsed)
0.5             | [result] | [result]          | [result]      | [result]     | [result]
1.0             | [result] | [result]          | [result]      | [result]     | [result]
2.0             | [result] | [result]          | [result]      | [result]     | [result]
5.0             | [result] | [result]          | [result]      | [result]     | [result]
10.0            | [result] | [result]          | [result]      | [result]     | [result]
```

---

## Expected Outcomes

### Hypothesis
- **λ = 0.0**: Negative R², variance ratio < 20% (collapse confirmed)
- **λ = 0.5-1.0**: Partial recovery, R² > 0, variance ratio 40-80%
- **λ = 2.0-5.0**: Full recovery, R² > 0.3, variance ratio 80-120%
- **λ = 10.0**: Possible instability, erratic training

### Success Criteria for Section 2

✅ **Go to Section 3 if:**
- At least one lambda achieves **R² > 0.2** (beats simple baselines partially)
- Variance ratio > 60% (demonstrates variance preservation)
- Training is stable (no NaNs/explosions)

❌ **Re-evaluate if:**
- All lambdas fail to achieve R² > 0
- Training consistently unstable across all values
- No improvement over baseline collapse

---

## Integration with Research Program

### Key Findings to Document

1. **Quantitative Evidence of Collapse**: "Without variance regularization (λ=0), the model achieved R² = [value], with prediction variance collapsing to [X]% of the target variance."

2. **Effectiveness of Intervention**: "Introducing variance-preserving regularization with λ=[optimal] increased R² to [value] and restored variance ratio to [X]%."

3. **Scientific Contribution**: "This demonstrates that variance collapse is a tractable training pathology, not a fundamental limitation of the architecture or data."

### Paper Narrative Integration

**Figure 2 (Research Program)**: Grid of prediction histograms showing qualitative effect of variance_lambda
- Baseline: Narrow spike at mean
- Low λ: Partial spread
- Optimal λ: Distribution matches target
- High λ: Possible over-dispersal or instability

**Table 2 (Research Program)**: Quantitative results from hyperparameter sweep

---

## Transition to Section 3

Once the optimal `variance_lambda` is determined:

1. **Update all Section 3 configs** with the selected value
2. **Document the decision** in the research log
3. **Proceed to architectural ablation** with confidence that training is stable

### Files to Update for Section 3
- `configs/phase_2_simple_cnn.yaml`
- All ablation configs in Section 3
- Any future experimental configs

---

## Troubleshooting

### Issue: All experiments show R² < 0

**Possible Causes**:
- Insufficient training epochs (try 20-30)
- Learning rate too high/low
- Data loading issues

**Diagnostics**:
```bash
# Check data loading
python -c "from src.hdf5_dataset import UnifiedHDF5Dataset; print('Data loads successfully')"

# Verify loss computation
grep "loss" logs/*.log | tail -20
```

### Issue: Training instability (NaN/Inf losses)

**Possible Causes**:
- variance_lambda too high
- Learning rate too high
- Gradient explosion

**Solutions**:
- Reduce variance_lambda (try λ < 2.0)
- Enable gradient clipping
- Reduce learning rate by 2-5x

### Issue: Predictions not saved

**Solution**:
Ensure the training script saves predictions. Add to `main.py` or training loop:

```python
import pickle
with open('predictions.pkl', 'wb') as f:
    pickle.dump({'predictions': preds, 'targets': targets}, f)
```

---

## Timeline Estimate

- **Section 2.1** (Baseline): ~30-60 min (15 epochs)
- **Section 2.2** (5 experiments): ~2.5-5 hours (15 epochs each)
- **Section 2.3** (Analysis): ~15-30 min (automated scripts)

**Total**: ~3-6 hours for complete Section 2

---

## Checkpoint and Continuation

If experiments are interrupted:

1. **Check completed experiments**:
   ```bash
   ls -lh logs/section2*.log
   ```

2. **Re-run only missing experiments**:
   ```bash
   python main.py --config configs/section2_lambda_[X].yaml
   ```

3. **Analysis works with partial data**:
   - Scripts will mark missing experiments as `[pending]`
   - You can proceed with partial results if needed

---

## Quick Reference Commands

```bash
# Full automated run
bash scripts/run_section2_experiments.sh

# Aggregate results
python scripts/aggregate_section2_results.py

# Generate plots
python scripts/plot_section2_distributions.py

# View results
cat diagnostics/results/section2_table2.txt
```

---

## Success Indicators

✅ Section 2 is **complete** when you have:

1. **Table 2** populated with results from all experiments
2. **Figure 2** showing distribution grid
3. **Selected optimal variance_lambda** with documented justification
4. **Recommendation** ready for Section 3 integration

---

## Next: Section 3 - Architectural Ablation Study

With variance collapse resolved, Section 3 will systematically evaluate:
- Simple CNN baseline
- + Spatial attention
- + Temporal attention  
- + Multi-scale temporal attention

Each building on the previous, quantifying the contribution of architectural complexity.

**Proceed when ready**: `SECTION3_EXECUTION_GUIDE.md` (to be created)

---

**Document Version**: 1.0  
**Last Updated**: Research Program Phase 1  
**Status**: Ready for Execution