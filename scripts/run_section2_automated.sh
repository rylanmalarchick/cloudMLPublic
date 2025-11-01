#!/bin/bash

################################################################################
# SECTION 2: AUTOMATED NON-INTERACTIVE EXECUTION
# Master script to run all variance_lambda experiments overnight
################################################################################

echo "======================================================================="
echo "SECTION 2: ADDRESSING MODEL COLLAPSE - AUTOMATED EXECUTION"
echo "Variance-Preserving Regularization Investigation"
echo "======================================================================="
echo ""
echo "Start time: $(date)"
echo ""

# Set error handling
set -e  # Exit on error
set -u  # Exit on undefined variable

# Change to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

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

# Array of experiments
declare -a experiments=(
    "section2_baseline_collapse:0.0:Baseline (no regularization)"
    "section2_lambda_0.5:0.5:Weak penalty"
    "section2_lambda_1.0:1.0:Moderate penalty"
    "section2_lambda_2.0:2.0:Moderate-strong penalty"
    "section2_lambda_5.0:5.0:Strong penalty"
    "section2_lambda_10.0:10.0:Very strong penalty"
)

total_experiments=${#experiments[@]}
current_experiment=0

echo "======================================================================="
echo "EXPERIMENT QUEUE: ${total_experiments} experiments"
echo "======================================================================="
for exp in "${experiments[@]}"; do
    IFS=':' read -r config lambda desc <<< "$exp"
    echo "  - ${desc} (λ=${lambda})"
done
echo ""

# Run each experiment
for exp in "${experiments[@]}"; do
    IFS=':' read -r config lambda desc <<< "$exp"
    current_experiment=$((current_experiment + 1))

    echo ""
    echo "======================================================================="
    echo "EXPERIMENT ${current_experiment}/${total_experiments}: ${desc}"
    echo "======================================================================="
    echo "Config: configs/${config}.yaml"
    echo "Lambda: ${lambda}"
    echo "Started: $(date)"
    echo "-----------------------------------------------------------------------"
    echo ""

    # Run experiment with output logging
    log_file="logs/${config}_$(date +%Y%m%d_%H%M%S).log"

    if python main.py --config "configs/${config}.yaml" 2>&1 | tee "$log_file"; then
        echo ""
        echo "-----------------------------------------------------------------------"
        echo "✓ Experiment ${current_experiment}/${total_experiments} COMPLETED"
        echo "  Finished: $(date)"
        echo "  Log saved: ${log_file}"
        echo "-----------------------------------------------------------------------"
    else
        echo ""
        echo "-----------------------------------------------------------------------"
        echo "✗ Experiment ${current_experiment}/${total_experiments} FAILED"
        echo "  Check log: ${log_file}"
        echo "-----------------------------------------------------------------------"
        echo ""
        echo "WARNING: Experiment failed but continuing with remaining experiments..."
    fi

    # Brief pause between experiments
    sleep 5
done

echo ""
echo "======================================================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "======================================================================="
echo "End time: $(date)"
echo ""
echo "Completed experiments:"
for exp in "${experiments[@]}"; do
    IFS=':' read -r config lambda desc <<< "$exp"
    log_pattern="logs/${config}_*.log"
    if ls $log_pattern 1> /dev/null 2>&1; then
        latest_log=$(ls -t $log_pattern | head -1)
        echo "  ✓ ${desc} (λ=${lambda}) - Log: ${latest_log}"
    else
        echo "  ✗ ${desc} (λ=${lambda}) - NO LOG FOUND"
    fi
done

echo ""
echo "======================================================================="
echo "NEXT STEPS: ANALYSIS"
echo "======================================================================="
echo ""
echo "Run the following commands to analyze results:"
echo ""
echo "1. Aggregate results into Table 2:"
echo "   python scripts/aggregate_section2_results.py"
echo ""
echo "2. Generate prediction distribution plots:"
echo "   python scripts/plot_section2_distributions.py"
echo ""
echo "3. View formatted results:"
echo "   cat diagnostics/results/section2_table2.txt"
echo ""
echo "======================================================================="
echo "SECTION 2 EXECUTION COMPLETE"
echo "======================================================================="
