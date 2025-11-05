#!/bin/bash
# SOW Sprint 3 - Quick Start Script
# Executes Work Packages 1-4 for Physics-Constrained CBH Model Validation

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="configs/bestComboConfig.yaml"
OUTPUT_DIR="sow_outputs"

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

# Check if we're in the right directory
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Configuration file not found: $CONFIG_FILE"
    echo "Please run this script from the cloudMLPublic directory"
    exit 1
fi

# Create output directories
mkdir -p $OUTPUT_DIR/wp1_geometric
mkdir -p $OUTPUT_DIR/wp2_atmospheric
mkdir -p $OUTPUT_DIR/wp3_baseline
mkdir -p $OUTPUT_DIR/wp4_hybrid
mkdir -p $OUTPUT_DIR/models/final_gbdt_models

print_header "SOW Sprint 3: Physics-Constrained CBH Model Validation"
echo "Configuration: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Parse command line arguments
RUN_WP1=true
RUN_WP2=true
RUN_WP3=false  # Requires WP1 and WP2 to complete first
RUN_WP4=false  # Requires WP3 to pass first
VERBOSE="--verbose"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-wp1)
            RUN_WP1=false
            shift
            ;;
        --skip-wp2)
            RUN_WP2=false
            shift
            ;;
        --run-wp3)
            RUN_WP3=true
            shift
            ;;
        --run-wp4)
            RUN_WP4=true
            shift
            ;;
        --quiet)
            VERBOSE=""
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-wp1      Skip Work Package 1 (Geometric Features)"
            echo "  --skip-wp2      Skip Work Package 2 (Atmospheric Features)"
            echo "  --run-wp3       Run Work Package 3 (Physical Baseline)"
            echo "  --run-wp4       Run Work Package 4 (Hybrid Models)"
            echo "  --quiet         Suppress verbose output"
            echo "  --help          Show this help message"
            echo ""
            echo "Default behavior: Run WP1 and WP2 only"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ==============================================================================
# WORK PACKAGE 1: Geometric Feature Engineering
# ==============================================================================

if [ "$RUN_WP1" = true ]; then
    print_header "Work Package 1: Geometric Feature Engineering"

    WP1_OUTPUT="$OUTPUT_DIR/wp1_geometric/WP1_Features.hdf5"

    if [ -f "$WP1_OUTPUT" ]; then
        print_warning "WP1 output already exists: $WP1_OUTPUT"
        read -p "Overwrite? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "Skipping WP1"
            RUN_WP1=false
        fi
    fi

    if [ "$RUN_WP1" = true ]; then
        echo "Extracting shadow-based geometric features..."
        echo "- Shadow detection and cloud-shadow pairing"
        echo "- Geometric CBH estimation from shadow length"
        echo "- Confidence scoring for each detection"
        echo ""

        python $OUTPUT_DIR/wp1_geometric_features.py \
            --config $CONFIG_FILE \
            --output $WP1_OUTPUT \
            --scale 50.0 \
            $VERBOSE

        if [ $? -eq 0 ]; then
            print_success "WP1 completed successfully"
            print_success "Output: $WP1_OUTPUT"
        else
            print_error "WP1 failed"
            exit 1
        fi
    fi
fi

# ==============================================================================
# WORK PACKAGE 2: Atmospheric Feature Engineering
# ==============================================================================

if [ "$RUN_WP2" = true ]; then
    print_header "Work Package 2: Atmospheric Feature Engineering"

    WP2_OUTPUT="$OUTPUT_DIR/wp2_atmospheric/WP2_Features.hdf5"

    if [ -f "$WP2_OUTPUT" ]; then
        print_warning "WP2 output already exists: $WP2_OUTPUT"
        read -p "Overwrite? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "Skipping WP2"
            RUN_WP2=false
        fi
    fi

    if [ "$RUN_WP2" = true ]; then
        echo "Extracting atmospheric thermodynamic features..."
        echo "- BLH, LCL, Inversion Height, Stability Index"
        echo "- Currently using SYNTHETIC features (ERA5 not implemented)"
        echo "- See SOW_IMPLEMENTATION_GUIDE.md for production setup"
        echo ""

        print_warning "NOTE: WP2 currently generates synthetic atmospheric features"
        print_warning "For production, configure ERA5 API and navigation file parsing"
        echo ""

        python $OUTPUT_DIR/wp2_atmospheric_features.py \
            --config $CONFIG_FILE \
            --output $WP2_OUTPUT \
            --era5-dir $OUTPUT_DIR/wp2_atmospheric/era5_data \
            $VERBOSE

        if [ $? -eq 0 ]; then
            print_success "WP2 completed successfully"
            print_success "Output: $WP2_OUTPUT"
        else
            print_error "WP2 failed"
            exit 1
        fi
    fi
fi

# ==============================================================================
# WORK PACKAGE 3: Physical Baseline Model Validation
# ==============================================================================

if [ "$RUN_WP3" = true ]; then
    print_header "Work Package 3: Physical Baseline Model Validation"

    # Check prerequisites
    WP1_OUTPUT="$OUTPUT_DIR/wp1_geometric/WP1_Features.hdf5"
    WP2_OUTPUT="$OUTPUT_DIR/wp2_atmospheric/WP2_Features.hdf5"

    if [ ! -f "$WP1_OUTPUT" ]; then
        print_error "WP1 output not found: $WP1_OUTPUT"
        print_error "Run WP1 first or use --skip-wp3"
        exit 1
    fi

    if [ ! -f "$WP2_OUTPUT" ]; then
        print_error "WP2 output not found: $WP2_OUTPUT"
        print_error "Run WP2 first or use --skip-wp3"
        exit 1
    fi

    print_warning "WP3 script not yet implemented"
    print_warning "Next steps:"
    echo "  1. Create sow_outputs/wp3_physical_baseline.py"
    echo "  2. Implement LOO CV framework"
    echo "  3. Train GBDT on physical features only"
    echo "  4. Generate WP3_Report.json"
    echo ""
    print_warning "This is the GO/NO-GO gate: Must achieve R² > 0"
fi

# ==============================================================================
# WORK PACKAGE 4: Hybrid Model Integration
# ==============================================================================

if [ "$RUN_WP4" = true ]; then
    print_header "Work Package 4: Hybrid Model Integration"

    # Check prerequisites
    WP3_REPORT="$OUTPUT_DIR/wp3_baseline/WP3_Report.json"

    if [ ! -f "$WP3_REPORT" ]; then
        print_error "WP3 report not found: $WP3_REPORT"
        print_error "Run WP3 first and verify it passes (R² > 0)"
        exit 1
    fi

    print_warning "WP4 script not yet implemented"
    print_warning "Next steps:"
    echo "  1. Create sow_outputs/wp4_hybrid_models.py"
    echo "  2. Extract MAE spatial embeddings (NOT CLS token)"
    echo "  3. Train 4 model variants with ablation"
    echo "  4. Run feature importance analysis"
    echo "  5. Generate all final deliverables"
fi

# ==============================================================================
# Summary
# ==============================================================================

print_header "Execution Summary"

echo "Completed Work Packages:"
if [ "$RUN_WP1" = true ]; then
    print_success "WP1: Geometric Feature Engineering"
fi
if [ "$RUN_WP2" = true ]; then
    print_success "WP2: Atmospheric Feature Engineering"
fi

echo ""
echo "Next Steps:"

if [ ! -f "$OUTPUT_DIR/wp1_geometric/WP1_Features.hdf5" ]; then
    echo "  1. Run WP1 to extract geometric features"
elif [ ! -f "$OUTPUT_DIR/wp2_atmospheric/WP2_Features.hdf5" ]; then
    echo "  1. Run WP2 to extract atmospheric features"
else
    echo "  1. Implement wp3_physical_baseline.py"
    echo "  2. Run WP3 with: $0 --run-wp3"
    echo "  3. If WP3 passes (R² > 0), implement wp4_hybrid_models.py"
    echo "  4. Run WP4 with: $0 --run-wp4"
fi

echo ""
echo "For detailed implementation guidance, see:"
echo "  $OUTPUT_DIR/SOW_IMPLEMENTATION_GUIDE.md"
echo ""

print_success "Script completed"
