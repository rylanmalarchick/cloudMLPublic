#!/bin/bash
# Sprint 6 Master Execution Script
# Executes all implemented Sprint 6 tasks in sequence
#
# Usage:
#   ./run_sprint6.sh [--phase PHASE] [--task TASK] [--dry-run]
#
# Options:
#   --phase PHASE     Run specific phase (1 or 2)
#   --task TASK       Run specific task (1.1, 1.2, 1.3, 1.4, 2.1, 2.2)
#   --dry-run         Print commands without executing
#   --help            Show this help message
#
# Author: Sprint 6 Execution Agent
# Date: 2025-01-10

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SPRINT6_DIR="$SCRIPT_DIR"

DRY_RUN=false
RUN_PHASE=""
RUN_TASK=""

# Function to print colored headers
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

# Function to execute or print command
run_command() {
    local cmd="$1"
    local description="$2"

    print_info "$description"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] $cmd"
    else
        echo "  [EXECUTING] $cmd"
        eval "$cmd"
        if [ $? -eq 0 ]; then
            print_success "$description completed"
        else
            print_error "$description failed"
            return 1
        fi
    fi
}

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check Python environment
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please activate virtual environment."
        exit 1
    fi
    print_success "Python found: $(python --version)"

    # Check if we're in the right directory
    if [ ! -f "$SPRINT6_DIR/README.md" ]; then
        print_error "Sprint 6 directory not found. Please run from sprint6/"
        exit 1
    fi
    print_success "Sprint 6 directory validated"

    # Check integrated features file
    INTEGRATED_FEATURES="$PROJECT_ROOT/sow_outputs/integrated_features/Integrated_Features.hdf5"
    if [ ! -f "$INTEGRATED_FEATURES" ]; then
        print_warning "Integrated features file not found: $INTEGRATED_FEATURES"
        print_warning "Some tasks may fail. Please ensure data is available."
    else
        print_success "Integrated features file found"
    fi

    # Check required Python packages
    print_info "Checking Python packages..."
    python -c "import torch; import transformers; import sklearn; import h5py" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "Required Python packages installed"
    else
        print_error "Missing required packages. Run: pip install torch transformers scikit-learn h5py matplotlib seaborn scipy"
        exit 1
    fi
}

# Function to run Phase 1 Task 1.1
run_task_1_1() {
    print_header "Phase 1, Task 1.1: Offline Validation"
    print_info "Expected runtime: 2-4 hours (GPU required)"
    print_info "Deliverables: validation_report.json, 4 figures, 5 model checkpoints"

    run_command \
        "cd $SPRINT6_DIR && python validation/offline_validation.py" \
        "Task 1.1: Offline Validation"
}

# Function to run Phase 1 Task 1.2
run_task_1_2() {
    print_header "Phase 1, Task 1.2: Uncertainty Quantification"
    print_info "Expected runtime: 30-60 minutes"
    print_info "Deliverables: uncertainty_quantification_report.json, 4 figures"
    print_info "Prerequisites: Task 1.1 must be completed (requires fold_0_model.pth)"

    # Check prerequisite
    if [ ! -f "$SPRINT6_DIR/checkpoints/fold_0_model.pth" ] && [ "$DRY_RUN" = false ]; then
        print_warning "Checkpoint not found. Task 1.1 may not have completed."
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "Skipping Task 1.2"
            return
        fi
    fi

    run_command \
        "cd $SPRINT6_DIR && python validation/uncertainty_quantification.py" \
        "Task 1.2: Uncertainty Quantification"
}

# Function to run Phase 1 Task 1.3
run_task_1_3() {
    print_header "Phase 1, Task 1.3: Comprehensive Error Analysis"
    print_info "Expected runtime: 10-20 minutes"
    print_info "Deliverables: error_analysis_report.json, systematic_bias_report.md, 5 figures"
    print_info "Prerequisites: Task 1.1 must be completed (requires fold_0_model.pth)"

    # Check prerequisite
    if [ ! -f "$SPRINT6_DIR/checkpoints/fold_0_model.pth" ] && [ "$DRY_RUN" = false ]; then
        print_warning "Checkpoint not found. Task 1.1 may not have completed."
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "Skipping Task 1.3"
            return
        fi
    fi

    run_command \
        "cd $SPRINT6_DIR && python analysis/error_analysis.py" \
        "Task 1.3: Comprehensive Error Analysis"
}

# Function to run Phase 1 Task 1.4
run_task_1_4() {
    print_header "Phase 1, Task 1.4: Final Production Model Training"
    print_info "Expected runtime: 1-2 hours"
    print_info "Deliverables: final_production_model.pth, hyperparameters.json, benchmark.json, REPRODUCIBILITY.md"

    run_command \
        "cd $SPRINT6_DIR && python training/train_production_model.py" \
        "Task 1.4: Final Production Model Training"
}

# Function to run Phase 2 Task 2.1
run_task_2_1() {
    print_header "Phase 2, Task 2.1: Ensemble Methods"
    print_info "Expected runtime: 2-3 hours (5-fold CV with GBDT + ViT)"
    print_info "Deliverables: ensemble_results.json, ensemble model checkpoints"
    print_info "Target: R² > 0.74"

    run_command \
        "cd $SPRINT6_DIR && python ensemble/ensemble_models.py" \
        "Task 2.1: Ensemble Methods"
}

# Function to run Phase 2 Task 2.2
run_task_2_2() {
    print_header "Phase 2, Task 2.2: Domain Adaptation for Flight F4"
    print_info "Expected runtime: 30-60 minutes"
    print_info "Deliverables: domain_adaptation_results.json, fine-tuned models, 2 figures"
    print_info "Baseline: LOO R² = -3.13 (catastrophic failure)"

    run_command \
        "cd $SPRINT6_DIR && python domain_adaptation/few_shot_f4.py" \
        "Task 2.2: Domain Adaptation for Flight F4"
}

# Function to run all Phase 1 tasks
run_phase_1() {
    print_header "Phase 1: Core Validation & Analysis"
    print_info "Total expected runtime: 4-6 hours"

    run_task_1_1
    run_task_1_2
    run_task_1_3
    run_task_1_4

    print_success "Phase 1 complete!"
}

# Function to run all Phase 2 tasks
run_phase_2() {
    print_header "Phase 2: Model Improvements & Comparisons"
    print_info "Total expected runtime: 2-4 hours"

    run_task_2_1
    run_task_2_2

    print_success "Phase 2 complete!"
}

# Function to display summary of outputs
display_summary() {
    print_header "Execution Summary"

    echo "Reports generated:"
    ls -lh "$SPRINT6_DIR/reports/"*.json 2>/dev/null || echo "  (none yet)"

    echo ""
    echo "Figures generated:"
    find "$SPRINT6_DIR/figures/" -name "*.png" 2>/dev/null | wc -l | xargs echo "  PNG files:"

    echo ""
    echo "Models saved:"
    find "$SPRINT6_DIR/models/" -name "*.pth" 2>/dev/null | wc -l | xargs echo "  Checkpoints:"
    find "$SPRINT6_DIR/checkpoints/" -name "*.pth" 2>/dev/null | wc -l | xargs echo "  Fold checkpoints:"

    echo ""
    echo "Documentation:"
    ls -lh "$SPRINT6_DIR/docs/"*.md 2>/dev/null || echo "  (none yet)"
}

# Function to show help
show_help() {
    cat << EOF
Sprint 6 Master Execution Script

Usage:
  $0 [OPTIONS]

Options:
  --phase PHASE     Run specific phase (1 or 2)
  --task TASK       Run specific task (1.1, 1.2, 1.3, 1.4, 2.1, 2.2)
  --dry-run         Print commands without executing
  --help            Show this help message

Examples:
  # Run all tasks (Phase 1 + Phase 2)
  $0

  # Run only Phase 1
  $0 --phase 1

  # Run only Task 1.1
  $0 --task 1.1

  # Dry run to see what would be executed
  $0 --dry-run

  # Run Phase 2 tasks only
  $0 --phase 2

Tasks:
  Phase 1: Core Validation & Analysis (4-6 hours)
    1.1: Offline Validation (2-4 hours)
    1.2: Uncertainty Quantification (30-60 min)
    1.3: Comprehensive Error Analysis (10-20 min)
    1.4: Final Production Model Training (1-2 hours)

  Phase 2: Model Improvements (2-4 hours)
    2.1: Ensemble Methods (2-3 hours)
    2.2: Domain Adaptation for Flight F4 (30-60 min)

Total estimated runtime: 6-10 hours (GPU required)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            RUN_PHASE="$2"
            shift 2
            ;;
        --task)
            RUN_TASK="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_header "Sprint 6 Master Execution Script"
    echo "Project Root: $PROJECT_ROOT"
    echo "Sprint 6 Directory: $SPRINT6_DIR"

    if [ "$DRY_RUN" = true ]; then
        print_warning "DRY RUN MODE - Commands will be printed but not executed"
    fi

    # Check prerequisites
    check_prerequisites

    # Execute based on arguments
    if [ -n "$RUN_TASK" ]; then
        # Run specific task
        case $RUN_TASK in
            1.1) run_task_1_1 ;;
            1.2) run_task_1_2 ;;
            1.3) run_task_1_3 ;;
            1.4) run_task_1_4 ;;
            2.1) run_task_2_1 ;;
            2.2) run_task_2_2 ;;
            *)
                print_error "Unknown task: $RUN_TASK"
                echo "Valid tasks: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2"
                exit 1
                ;;
        esac
    elif [ -n "$RUN_PHASE" ]; then
        # Run specific phase
        case $RUN_PHASE in
            1) run_phase_1 ;;
            2) run_phase_2 ;;
            *)
                print_error "Unknown phase: $RUN_PHASE"
                echo "Valid phases: 1, 2"
                exit 1
                ;;
        esac
    else
        # Run all tasks
        print_info "Running all implemented tasks (Phase 1 + Phase 2)"
        print_warning "This will take 6-10 hours. Press Ctrl+C to cancel."
        sleep 3

        run_phase_1
        run_phase_2
    fi

    # Display summary
    if [ "$DRY_RUN" = false ]; then
        display_summary
    fi

    print_success "Sprint 6 execution complete!"
    print_info "Review outputs in: $SPRINT6_DIR"
}

# Run main function
main
