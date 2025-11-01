#!/bin/bash

################################################################################
# DIAGNOSTIC EXPERIMENTS RUNNER
# Purpose: Isolate the root cause of Section 2 failures
################################################################################

echo "======================================================================="
echo "DIAGNOSTIC EXPERIMENTS - ROOT CAUSE ANALYSIS"
echo "======================================================================="
echo ""
echo "Start time: $(date)"
echo ""

# Change to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"
echo ""

# Activate virtual environment
if [ -d "./venv" ]; then
    echo "Activating virtual environment..."
    source ./venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "ERROR: Virtual environment not found at ./venv"
    exit 1
fi

echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Create output directories
mkdir -p logs models results diagnostics/results
echo "✓ Output directories ready"
echo ""

echo "======================================================================="
echo "DIAGNOSTIC PLAN"
echo "======================================================================="
echo ""
echo "These experiments will isolate the failure mode:"
echo ""
echo "Experiment A: Simplest Possible CNN"
echo "  - No attention, no variance loss, pure MSE"
echo "  - Tests: Is architectural complexity the problem?"
echo "  - Expected time: ~30 minutes"
echo ""
echo "Experiment C: Single-Flight Overfitting Test"
echo "  - Train AND validate on same flight (30Oct24)"
echo "  - Tests: Can the model learn ANYTHING?"
echo "  - Expected time: ~60 minutes"
echo ""
echo "Total estimated time: ~90 minutes"
echo ""

read -p "Press Enter to start Experiment A, or Ctrl+C to cancel..."

echo ""
echo "======================================================================="
echo "EXPERIMENT A: SIMPLEST POSSIBLE CNN"
echo "======================================================================="
echo "Config: configs/diagnostic_exp_a_simple.yaml"
echo "Started: $(date)"
echo "-----------------------------------------------------------------------"
echo ""

log_file="logs/diagnostic_exp_a_simple_$(date +%Y%m%d_%H%M%S).log"

if python main.py --config configs/diagnostic_exp_a_simple.yaml 2>&1 | tee "$log_file"; then
    echo ""
    echo "-----------------------------------------------------------------------"
    echo "✓ Experiment A COMPLETED"
    echo "  Finished: $(date)"
    echo "  Log saved: ${log_file}"
    echo "-----------------------------------------------------------------------"
    echo ""

    # Quick analysis
    echo "Quick Results:"
    grep "Epoch.*Train Loss" "$log_file" | tail -5
    echo ""
    best_r2=$(grep "R²:" "$log_file" | grep -oP "R²: -?\d+\.\d+" | sort -t':' -k2 -rn | head -1)
    echo "Best R²: $best_r2"
    echo ""
else
    echo ""
    echo "-----------------------------------------------------------------------"
    echo "✗ Experiment A FAILED"
    echo "  Check log: ${log_file}"
    echo "-----------------------------------------------------------------------"
fi

sleep 5

echo ""
read -p "Press Enter to start Experiment C, or Ctrl+C to skip..."

echo ""
echo "======================================================================="
echo "EXPERIMENT C: SINGLE-FLIGHT OVERFITTING TEST"
echo "======================================================================="
echo "Config: configs/diagnostic_exp_c_overfit.yaml"
echo "Started: $(date)"
echo "-----------------------------------------------------------------------"
echo ""

log_file="logs/diagnostic_exp_c_overfit_$(date +%Y%m%d_%H%M%S).log"

if python main.py --config configs/diagnostic_exp_c_overfit.yaml 2>&1 | tee "$log_file"; then
    echo ""
    echo "-----------------------------------------------------------------------"
    echo "✓ Experiment C COMPLETED"
    echo "  Finished: $(date)"
    echo "  Log saved: ${log_file}"
    echo "-----------------------------------------------------------------------"
    echo ""

    # Quick analysis
    echo "Quick Results (last 5 epochs):"
    grep "Epoch.*Train Loss" "$log_file" | tail -5
    echo ""
    best_r2=$(grep "R²:" "$log_file" | grep -oP "R²: -?\d+\.\d+" | sort -t':' -k2 -rn | head -1)
    echo "Best R²: $best_r2"
    echo ""
else
    echo ""
    echo "-----------------------------------------------------------------------"
    echo "✗ Experiment C FAILED"
    echo "  Check log: ${log_file}"
    echo "-----------------------------------------------------------------------"
fi

echo ""
echo "======================================================================="
echo "DIAGNOSTICS COMPLETE"
echo "======================================================================="
echo "End time: $(date)"
echo ""
echo "======================================================================="
echo "ANALYSIS & DECISION CRITERIA"
echo "======================================================================="
echo ""
echo "Review the logs and R² scores from both experiments:"
echo ""
echo "SCENARIO 1: Experiment A achieves R² > 0"
echo "  → Architecture complexity was the problem"
echo "  → Proceed with simplified Section 3 (simple CNN)"
echo "  → Gradually add complexity back"
echo ""
echo "SCENARIO 2: Experiment C achieves R² > 0.8 (overfitting successful)"
echo "  → Model CAN learn, problem is generalization"
echo "  → Investigate data distribution differences between flights"
echo "  → Consider flight-specific normalization"
echo ""
echo "SCENARIO 3: Both experiments fail (R² < 0)"
echo "  → Fundamental problem with model or data pipeline"
echo "  → Debug: data loading, loss calculation, gradient flow"
echo "  → May need to pivot to simple model paper"
echo ""
echo "SCENARIO 4: Both experiments succeed (R² > 0)"
echo "  → Original Section 2 configs had multiple bugs"
echo "  → Proceed to Section 3 with fixed configs"
echo ""
echo "======================================================================="
echo ""
echo "Next steps:"
echo "  1. Review logs in logs/diagnostic_*.log"
echo "  2. Check R² trends (should increase over epochs)"
echo "  3. Make decision based on scenarios above"
echo ""
echo "======================================================================="
