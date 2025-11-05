#!/bin/bash
# WP-2: Real ERA5 Data Download and Processing
# This script orchestrates the complete ERA5 data acquisition and feature extraction pipeline

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="configs/bestComboConfig.yaml"
OUTPUT_FILE="sow_outputs/wp2_atmospheric/WP2_Features.hdf5"
ERA5_DATA_ROOT="/media/rylan/two/research/NASA/ERA5_data_root"

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

# Print banner
clear
echo -e "${CYAN}"
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║              WP-2: REAL ERA5 ATMOSPHERIC FEATURE ENGINEERING               ║"
echo "║                                                                            ║"
echo "║  This script will download 30-70 GB of ERA5 reanalysis data from CDS      ║"
echo "║  Expected duration: 2-8 hours (depending on CDS queue)                    ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

# Check if we're in the right directory
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Configuration file not found: $CONFIG_FILE"
    echo "Please run this script from the cloudMLPublic directory"
    exit 1
fi

# Display configuration
print_info "Configuration:"
echo "  Config file:      $CONFIG_FILE"
echo "  Output HDF5:      $OUTPUT_FILE"
echo "  ERA5 data root:   $ERA5_DATA_ROOT"
echo ""

# Check ERA5 data directory
if [ ! -d "$ERA5_DATA_ROOT" ]; then
    print_error "ERA5 data directory does not exist: $ERA5_DATA_ROOT"
    echo "Creating directory..."
    mkdir -p "$ERA5_DATA_ROOT/surface"
    mkdir -p "$ERA5_DATA_ROOT/pressure_levels"
    mkdir -p "$ERA5_DATA_ROOT/processed"
    print_success "Directories created"
fi

# Check disk space on ERA5 drive
print_info "Checking disk space on ERA5 drive..."
AVAILABLE_SPACE=$(df -BG "$ERA5_DATA_ROOT" | tail -1 | awk '{print $4}' | sed 's/G//')
echo "  Available space: ${AVAILABLE_SPACE} GB"

if [ "$AVAILABLE_SPACE" -lt 80 ]; then
    print_warning "Low disk space! Recommend at least 80 GB free."
    print_warning "Available: ${AVAILABLE_SPACE} GB"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Aborted by user"
        exit 1
    fi
fi

# Check Python dependencies
print_header "Checking Dependencies"

echo "Checking Python packages..."

# Check cdsapi
if python3 -c "import cdsapi" 2>/dev/null; then
    print_success "cdsapi installed"
else
    print_error "cdsapi not installed"
    echo ""
    echo "Install with:"
    echo "  pip install cdsapi"
    echo ""
    exit 1
fi

# Check xarray
if python3 -c "import xarray" 2>/dev/null; then
    print_success "xarray installed"
else
    print_error "xarray not installed"
    echo ""
    echo "Install with:"
    echo "  pip install xarray netCDF4"
    echo ""
    exit 1
fi

# Check netCDF4
if python3 -c "import netCDF4" 2>/dev/null; then
    print_success "netCDF4 installed"
else
    print_error "netCDF4 not installed"
    echo ""
    echo "Install with:"
    echo "  pip install netCDF4"
    echo ""
    exit 1
fi

# Check CDS API credentials
print_info "Checking CDS API credentials..."
if [ -f "$HOME/.cdsapirc" ]; then
    print_success "CDS API credentials found: ~/.cdsapirc"
else
    print_error "CDS API credentials not found!"
    echo ""
    echo "You need to set up CDS API access:"
    echo ""
    echo "1. Register at: https://cds.climate.copernicus.eu/user/register"
    echo "2. Login and get your API key from: https://cds.climate.copernicus.eu/user"
    echo "3. Create ~/.cdsapirc with:"
    echo ""
    echo "   url: https://cds.climate.copernicus.eu/api"
    echo "   key: YOUR_UID:YOUR_API_KEY"
    echo ""
    echo "Replace YOUR_UID and YOUR_API_KEY with your credentials"
    echo ""
    exit 1
fi

# Parse command line arguments
SKIP_DOWNLOAD=false
FORCE_DOWNLOAD=false
SURFACE_ONLY=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --force-download)
            FORCE_DOWNLOAD=true
            shift
            ;;
        --surface-only)
            SURFACE_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-download     Use existing ERA5 data (skip download)"
            echo "  --force-download    Re-download all ERA5 files even if they exist"
            echo "  --surface-only      Download only surface data (faster, fewer features)"
            echo "  --dry-run           Show what would be done without executing"
            echo "  --help              Show this help message"
            echo ""
            echo "Default: Download all data (surface + pressure levels)"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display execution plan
print_header "Execution Plan"

echo "Mode:"
if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN MODE (no actual execution)"
fi

if [ "$SKIP_DOWNLOAD" = true ]; then
    print_warning "Skip ERA5 download (use existing data)"
elif [ "$FORCE_DOWNLOAD" = true ]; then
    print_warning "Force re-download all ERA5 files"
fi

if [ "$SURFACE_ONLY" = true ]; then
    print_warning "Surface-only mode (no pressure level data)"
    print_warning "Features will be limited: inversion_height, stability_index will be NaN"
fi

echo ""
echo "Steps:"
echo "  1. Parse navigation files (extract lat/lon/time for 933 samples)"
echo "  2. Download ERA5 surface data (~6-12 GB)"
if [ "$SURFACE_ONLY" = false ]; then
    echo "  3. Download ERA5 pressure level data (~24-60 GB)"
    echo "  4. Extract and derive atmospheric features"
else
    echo "  3. Extract and derive atmospheric features (limited)"
fi
echo "  Final: Save features to HDF5"
echo ""

# Confirmation prompt (unless dry-run)
if [ "$DRY_RUN" = false ]; then
    if [ "$SKIP_DOWNLOAD" = false ]; then
        print_warning "This will download 30-70 GB of data and may take 2-8 hours!"
        echo ""
        read -p "Proceed with ERA5 download? (yes/NO) " -r
        echo
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            print_info "Aborted by user"
            exit 0
        fi
    fi
fi

# Build Python command (use venv if available)
if [ -f "venv/bin/python" ]; then
    PYTHON_CMD="./venv/bin/python sow_outputs/wp2_era5_real.py"
else
    PYTHON_CMD="python3 sow_outputs/wp2_era5_real.py"
fi
PYTHON_CMD="$PYTHON_CMD --config $CONFIG_FILE"
PYTHON_CMD="$PYTHON_CMD --output $OUTPUT_FILE"
PYTHON_CMD="$PYTHON_CMD --era5-dir $ERA5_DATA_ROOT"

if [ "$SKIP_DOWNLOAD" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --skip-download"
fi

if [ "$FORCE_DOWNLOAD" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --force-download"
fi

if [ "$SURFACE_ONLY" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --surface-only"
fi

PYTHON_CMD="$PYTHON_CMD --verbose"

# Display command
print_header "Command"
echo "$PYTHON_CMD"
echo ""

if [ "$DRY_RUN" = true ]; then
    print_info "DRY RUN: Command shown above would be executed"
    exit 0
fi

# Execute
print_header "Executing WP-2 Pipeline"

# Record start time
START_TIME=$(date +%s)

# Run the Python script
echo "Starting at: $(date)"
echo ""

if $PYTHON_CMD; then
    # Record end time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))

    print_header "WP-2 COMPLETE!"
    print_success "ERA5 atmospheric features extracted successfully"
    echo ""
    echo "Output file:    $OUTPUT_FILE"
    echo "ERA5 data:      $ERA5_DATA_ROOT"
    echo "Duration:       ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""

    # Check output file size
    if [ -f "$OUTPUT_FILE" ]; then
        OUTPUT_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "Output size:    $OUTPUT_SIZE"
    fi

    # Check total ERA5 data size
    ERA5_SIZE=$(du -sh "$ERA5_DATA_ROOT" 2>/dev/null | cut -f1)
    echo "ERA5 data size: $ERA5_SIZE"
    echo ""

    print_header "Next Steps"
    echo "1. Inspect the features:"
    echo "   python -c \"import h5py; f = h5py.File('$OUTPUT_FILE', 'r'); print(list(f.keys())); print('Features shape:', f['features'].shape)\""
    echo ""
    echo "2. Run WP-1 diagnostics (check geometric features):"
    echo "   python sow_outputs/diagnose_wp1.py --config $CONFIG_FILE --samples 0 50 100 200 400"
    echo ""
    echo "3. Run WP-3 (Physical Baseline Validation):"
    echo "   python sow_outputs/wp3_physical_baseline.py --wp2-features $OUTPUT_FILE"
    echo ""

else
    # Error occurred
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))

    print_header "WP-2 FAILED"
    print_error "Pipeline failed after ${MINUTES}m ${SECONDS}s"
    echo ""
    echo "Common issues:"
    echo "  • CDS API credentials invalid or expired"
    echo "  • CDS service temporarily unavailable (try again later)"
    echo "  • Network connection interrupted"
    echo "  • Disk full on ERA5 drive"
    echo ""
    echo "Check logs above for specific error messages"
    echo ""
    exit 1
fi
