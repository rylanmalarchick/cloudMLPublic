#!/usr/bin/env python3
"""
Data Inspection Script
======================

Quick inspection of WP1-WP4 output files to understand data structure
before running full analysis scripts.

This script uses only standard library to avoid dependency issues.

Author: AI Research Assistant
Date: 2025-02-19
"""

import sys
import json
from pathlib import Path


def inspect_json_file(filepath):
    """Inspect JSON file (WP3 report)."""
    print(f"\n{'=' * 70}")
    print(f"Inspecting: {filepath}")
    print("=" * 70)

    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        print("\nJSON Structure:")
        print(json.dumps(data, indent=2))

        # Extract key metrics if present
        if "mean_r2" in data:
            print(f"\nKey Result: Mean R² = {data['mean_r2']}")
        if "loo_cv_results" in data:
            print(f"\nLOO CV Folds: {len(data['loo_cv_results'])}")
            for i, fold in enumerate(data["loo_cv_results"]):
                print(f"  Fold {i}: {fold}")

        return data

    except Exception as e:
        print(f"ERROR reading JSON: {e}")
        return None


def inspect_hdf5_file(filepath):
    """Inspect HDF5 file structure."""
    print(f"\n{'=' * 70}")
    print(f"Inspecting: {filepath}")
    print("=" * 70)

    try:
        import h5py

        with h5py.File(filepath, "r") as f:
            print(f"\nDatasets found:")

            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
                    # Print first few values for small arrays
                    if obj.size < 20:
                        print(f"    Values: {obj[:]}")
                    else:
                        print(f"    Range: {obj[:].min():.2f} to {obj[:].max():.2f}")
                        print(f"    Mean: {obj[:].mean():.2f}, Std: {obj[:].std():.2f}")

            f.visititems(print_structure)

    except ImportError:
        print("ERROR: h5py not installed. Install with: pip install h5py")
        print("\nAttempting to show file info without opening...")
        print(f"File exists: {Path(filepath).exists()}")
        print(f"File size: {Path(filepath).stat().st_size / 1024:.1f} KB")

    except Exception as e:
        print(f"ERROR reading HDF5: {e}")


def main():
    """Main inspection routine."""
    print("\n" + "=" * 70)
    print("WP1-WP4 DATA INSPECTION")
    print("=" * 70)

    # Define file paths
    base_dir = Path("sow_outputs")
    files_to_inspect = {
        "WP1 Geometric Features": base_dir / "wp1_geometric" / "WP1_Features.hdf5",
        "WP2 Atmospheric Features": base_dir / "wp2_atmospheric" / "WP2_Features.hdf5",
        "WP3 Baseline Report": base_dir / "wp3_baseline" / "WP3_Report.json",
    }

    # Check which files exist
    print("\nFile Status:")
    for name, path in files_to_inspect.items():
        exists = "✓" if path.exists() else "✗"
        print(f"  {exists} {name}: {path}")

    # Inspect WP3 JSON (easiest, no dependencies)
    wp3_json = files_to_inspect["WP3 Baseline Report"]
    if wp3_json.exists():
        wp3_data = inspect_json_file(wp3_json)

    # Inspect HDF5 files (requires h5py)
    wp1_hdf5 = files_to_inspect["WP1 Geometric Features"]
    if wp1_hdf5.exists():
        inspect_hdf5_file(wp1_hdf5)

    wp2_hdf5 = files_to_inspect["WP2 Atmospheric Features"]
    if wp2_hdf5.exists():
        inspect_hdf5_file(wp2_hdf5)

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)
    print("\nNext Steps:")
    print(
        "  1. Install dependencies: pip install -r sprint4_execution/requirements.txt"
    )
    print(
        "  2. Run ERA5 validation: python3 sprint4_execution/validate_era5_constraints.py"
    )
    print(
        "  3. Run shadow analysis: python3 sprint4_execution/shadow_failure_analysis.py"
    )


if __name__ == "__main__":
    main()
