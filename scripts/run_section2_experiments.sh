#!/bin/bash

################################################################################
# SECTION 2: MODEL COLLAPSE INVESTIGATION
# Master script to execute all variance_lambda experiments
################################################################################

echo "======================================================================="
echo "SECTION 2: ADDRESSING MODEL COLLAPSE"
echo "Variance-Preserving Regularization Investigation"
echo "======================================================================="
echo ""

# Set the virtual environment path
VENV_PATH="./venv/bin/activate"

# Check if running locally or on Colab
if [ -d "./venv" ]; then
    echo "Detected local environment. Activating virtual environment..."
    source "$VENV_PATH"
    MODE="local"
else
    echo "Detected Colab/HPC environment."
    MODE="colab"
fi

echo ""
echo "======================================================================="
echo "SECTION 2.1: BASELINE COLLAPSE EXPERIMENT (variance_lambda = 0.0)"
echo "======================================================================="
echo ""
echo "Objective: Quantify model collapse without variance-preserving regularizer"
echo "Expected: Negative R², variance ratio drops to ~0%"
echo ""
read -p "Press Enter to start baseline experiment, or Ctrl+C to skip..."

python main.py --config configs/section2_baseline_collapse.yaml

echo ""
echo "Baseline experiment complete. Check logs for:"
echo "  - Final R² (should be negative)"
echo "  - Variance ratio trend (should decrease)"
echo ""
read -p "Press Enter to continue to hyperparameter sweep..."

echo ""
echo "======================================================================="
echo "SECTION 2.2: VARIANCE_LAMBDA HYPERPARAMETER SWEEP"
echo "======================================================================="
echo ""
echo "Testing lambda values: 0.5, 1.0, 2.0, 5.0, 10.0"
echo "Each run: 15 epochs to observe stability and performance"
echo ""

# Lambda = 0.5
echo ""
echo "-----------------------------------------------------------------------"
echo "Experiment 2.2.1: variance_lambda = 0.5 (Weak penalty)"
echo "-----------------------------------------------------------------------"
echo ""
read -p "Press Enter to start, or Ctrl+C to skip..."
python main.py --config configs/section2_lambda_0.5.yaml

# Lambda = 1.0
echo ""
echo "-----------------------------------------------------------------------"
echo "Experiment 2.2.2: variance_lambda = 1.0 (Moderate penalty)"
echo "-----------------------------------------------------------------------"
echo ""
read -p "Press Enter to start, or Ctrl+C to skip..."
python main.py --config configs/section2_lambda_1.0.yaml

# Lambda = 2.0
echo ""
echo "-----------------------------------------------------------------------"
echo "Experiment 2.2.3: variance_lambda = 2.0 (Moderate-Strong penalty)"
echo "-----------------------------------------------------------------------"
echo ""
read -p "Press Enter to start, or Ctrl+C to skip..."
python main.py --config configs/section2_lambda_2.0.yaml

# Lambda = 5.0
echo ""
echo "-----------------------------------------------------------------------"
echo "Experiment 2.2.4: variance_lambda = 5.0 (Strong penalty)"
echo "-----------------------------------------------------------------------"
echo ""
read -p "Press Enter to start, or Ctrl+C to skip..."
python main.py --config configs/section2_lambda_5.0.yaml

# Lambda = 10.0
echo ""
echo "-----------------------------------------------------------------------"
echo "Experiment 2.2.5: variance_lambda = 10.0 (Very Strong penalty)"
echo "-----------------------------------------------------------------------"
echo "WARNING: This may cause training instability or loss explosion"
echo ""
read -p "Press Enter to start, or Ctrl+C to skip..."
python main.py --config configs/section2_lambda_10.0.yaml

echo ""
echo "======================================================================="
echo "SECTION 2: ALL EXPERIMENTS COMPLETE"
echo "======================================================================="
echo ""
echo "Experiments completed:"
echo "  1. Baseline (lambda=0.0) - Quantified collapse"
echo "  2. Lambda=0.5 - Weak regularization"
echo "  3. Lambda=1.0 - Moderate regularization"
echo "  4. Lambda=2.0 - Moderate-strong regularization"
echo "  5. Lambda=5.0 - Strong regularization"
echo "  6. Lambda=10.0 - Very strong regularization"
echo ""
echo "======================================================================="
echo "NEXT STEPS: SECTION 2.3 ANALYSIS"
echo "======================================================================="
echo ""
echo "To complete Section 2, you should now:"
echo ""
echo "1. Aggregate results from all runs:"
echo "   python scripts/aggregate_section2_results.py"
echo ""
echo "2. Generate prediction distribution plots:"
echo "   python scripts/plot_section2_distributions.py"
echo ""
echo "3. Select optimal variance_lambda based on:"
echo "   - Highest validation R²"
echo "   - Variance ratio closest to 100%"
echo "   - Training stability (no explosions)"
echo "   - Prediction distribution matching target distribution"
echo ""
echo "4. Document findings in Table 2 of the research program"
echo ""
echo "5. Proceed to Section 3: Architectural Ablation Study"
echo ""
echo "======================================================================="
