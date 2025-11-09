#!/bin/bash
#
# Quick-Start Script: Process Real ERA5 Data and Retrain Models
#
# This script processes the real ERA5 data from the external drive
# and retrains all models with the real atmospheric features.
#
# Usage: ./run_real_era5_pipeline.sh
#

set -e  # Exit on error

echo "================================================================================"
echo "SPRINT 3/4 ERA5 DATA PROCESSING PIPELINE"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Process real ERA5 data from /media/rylan/two/research/NASA/ERA5_data_root/"
echo "  2. Create WP2_Features_REAL_ERA5.hdf5 with atmospheric features"
echo "  3. Re-train physical baseline GBDT model"
echo "  4. Re-train all CNN hybrid models"
echo "  5. Generate updated performance reports"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Check if ERA5 data exists
if [ ! -d "/media/rylan/two/research/NASA/ERA5_data_root/surface" ]; then
    echo "ERROR: ERA5 surface data not found at /media/rylan/two/research/NASA/ERA5_data_root/surface"
    echo "Please mount the external drive and try again."
    exit 1
fi

if [ ! -d "/media/rylan/two/research/NASA/ERA5_data_root/pressure_levels" ]; then
    echo "ERROR: ERA5 pressure data not found at /media/rylan/two/research/NASA/ERA5_data_root/pressure_levels"
    echo "Please mount the external drive and try again."
    exit 1
fi

echo "✓ ERA5 data directories found"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "WARNING: Virtual environment not found. Using system Python."
fi
echo ""

# Check for required packages
echo "Checking dependencies..."
python3 -c "import xarray, netCDF4" 2>/dev/null || {
    echo "ERROR: xarray or netCDF4 not installed."
    echo "Install with: pip install xarray netCDF4"
    exit 1
}
echo "✓ Dependencies OK"
echo ""

# ============================================================================
# STEP 1: Process Real ERA5 Data
# ============================================================================
echo "================================================================================"
echo "STEP 1: Processing Real ERA5 Data"
echo "================================================================================"
echo ""

cd sow_outputs/

echo "Running ERA5 feature extraction..."
python3 process_real_era5.py \
    --config ../configs/bestComboConfig.yaml \
    --output wp2_atmospheric/WP2_Features_REAL_ERA5.hdf5 \
    --era5-surface /media/rylan/two/research/NASA/ERA5_data_root/surface \
    --era5-pressure /media/rylan/two/research/NASA/ERA5_data_root/pressure_levels \
    --verbose

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ ERA5 features extracted successfully!"
    echo "✓ Output: sow_outputs/wp2_atmospheric/WP2_Features_REAL_ERA5.hdf5"
else
    echo ""
    echo "✗ ERROR: ERA5 feature extraction failed!"
    exit 1
fi

cd ..
echo ""

# ============================================================================
# STEP 2: Backup Original Synthetic Features
# ============================================================================
echo "================================================================================"
echo "STEP 2: Backing Up Original Synthetic Features"
echo "================================================================================"
echo ""

if [ -f "sow_outputs/wp2_atmospheric/WP2_Features.hdf5" ]; then
    echo "Creating backup of synthetic features..."
    cp sow_outputs/wp2_atmospheric/WP2_Features.hdf5 \
       sow_outputs/wp2_atmospheric/WP2_Features_SYNTHETIC_BACKUP.hdf5
    echo "✓ Backup created: WP2_Features_SYNTHETIC_BACKUP.hdf5"
else
    echo "Note: Original WP2_Features.hdf5 not found (may already be backed up)"
fi
echo ""

# ============================================================================
# STEP 3: Update WP2 Features to Use Real ERA5
# ============================================================================
echo "================================================================================"
echo "STEP 3: Replacing WP2 Features with Real ERA5"
echo "================================================================================"
echo ""

echo "Copying real ERA5 features to WP2_Features.hdf5..."
cp sow_outputs/wp2_atmospheric/WP2_Features_REAL_ERA5.hdf5 \
   sow_outputs/wp2_atmospheric/WP2_Features.hdf5
echo "✓ WP2_Features.hdf5 now contains REAL ERA5 data"
echo ""

# ============================================================================
# STEP 4: Re-train Physical Baseline GBDT
# ============================================================================
echo "================================================================================"
echo "STEP 4: Re-training Physical Baseline (GBDT) with Real ERA5"
echo "================================================================================"
echo ""

cd sow_outputs/

echo "Training WP3 physical baseline..."
python3 wp3_kfold.py \
    --wp1-features wp1_geometric/WP1_Features.hdf5 \
    --wp2-features wp2_atmospheric/WP2_Features.hdf5 \
    --config ../configs/bestComboConfig.yaml \
    --output-dir wp3_kfold_real_era5 \
    --n-folds 5 \
    --verbose

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Physical baseline trained with REAL ERA5!"
    echo "✓ Results: sow_outputs/wp3_kfold_real_era5/WP3_Report_kfold.json"
else
    echo ""
    echo "✗ ERROR: Physical baseline training failed!"
    exit 1
fi

cd ..
echo ""

# ============================================================================
# STEP 5: Re-train CNN Models (Optional - takes ~8 hours)
# ============================================================================
echo "================================================================================"
echo "STEP 5: Re-training CNN Models (Optional)"
echo "================================================================================"
echo ""
echo "CNN training takes ~8 hours for all 3 variants × 5 folds."
echo ""
read -p "Re-train CNN models now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]
then
    cd sow_outputs/

    echo ""
    echo "Training Image-Only CNN..."
    python3 wp4_cnn_model.py \
        --fusion-mode image_only \
        --n-folds 5 \
        --output-dir wp4_cnn_real_era5 \
        --verbose

    echo ""
    echo "Training Concatenation Fusion CNN..."
    python3 wp4_cnn_model.py \
        --fusion-mode concat \
        --n-folds 5 \
        --output-dir wp4_cnn_real_era5 \
        --verbose

    echo ""
    echo "Training Attention Fusion CNN..."
    python3 wp4_cnn_model.py \
        --fusion-mode attention \
        --n-folds 5 \
        --output-dir wp4_cnn_real_era5 \
        --verbose

    echo ""
    echo "✓ All CNN models trained with REAL ERA5!"
    echo "✓ Results: sow_outputs/wp4_cnn_real_era5/"

    cd ..
else
    echo "Skipping CNN training. You can run it later with:"
    echo "  cd sow_outputs/"
    echo "  python3 wp4_cnn_model.py --fusion-mode [image_only|concat|attention] --verbose"
fi

echo ""

# ============================================================================
# STEP 6: Generate Updated Reports
# ============================================================================
echo "================================================================================"
echo "STEP 6: Generating Updated Performance Reports"
echo "================================================================================"
echo ""

cd sow_outputs/

echo "Creating validation summary..."
python3 create_validation_summary.py --verbose || echo "Warning: Validation summary failed"

echo ""
echo "Creating ablation study..."
python3 wp4_ablation_study.py --verbose || echo "Warning: Ablation study failed"

echo ""
echo "Creating final summary..."
python3 wp4_final_summary.py --verbose || echo "Warning: Final summary failed"

cd ..
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "================================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "✓ Real ERA5 data processed"
echo "✓ Physical baseline re-trained with real atmospheric features"
echo ""
echo "Results with REAL ERA5 data:"
echo "  - Physical baseline: sow_outputs/wp3_kfold_real_era5/WP3_Report_kfold.json"
echo ""
echo "Original results (SYNTHETIC data):"
echo "  - Physical baseline: sow_outputs/wp3_kfold/WP3_Report_kfold.json"
echo ""
echo "Compare the two to see the improvement from real ERA5 data!"
echo ""
echo "View results:"
echo "  cat sow_outputs/wp3_kfold_real_era5/WP3_Report_kfold.json | jq '.results'"
echo ""
echo "Next steps:"
echo "  1. Compare synthetic vs. real ERA5 performance"
echo "  2. Re-train CNN models (optional, ~8 hours)"
echo "  3. Update documentation with real ERA5 results"
echo "  4. Proceed with Sprint 5 improvements (pre-trained CNNs, temporal modeling)"
echo ""
echo "================================================================================"
