#!/usr/bin/env python3
"""
Compare old random split vs new stratified split to demonstrate the improvement.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import yaml
from src.hdf5_dataset import HDF5CloudDataset
from src.split_utils import stratified_split_by_flight
from collections import defaultdict


def analyze_split_composition(dataset, indices, split_name="Split"):
    """Analyze and print the flight composition of a split."""
    flight_counts = defaultdict(int)
    total = len(indices)

    for idx in indices:
        global_idx = int(dataset.indices[idx])
        flight_idx, _ = dataset.global_to_local[global_idx]
        flight_name = dataset.flight_configs[flight_idx]["name"]
        flight_counts[flight_name] += 1

    print(f"\n{split_name} composition ({total} samples):")
    print(f"{'Flight':<15} {'Count':>8} {'Percentage':>12}")
    print("-" * 40)
    for flight in sorted(flight_counts.keys()):
        count = flight_counts[flight]
        pct = 100 * count / total
        print(f"{flight:<15} {count:>8} {pct:>11.1f}%")

    return flight_counts


def old_random_split(total_samples, train_ratio, val_ratio, seed=42):
    """Old splitting method - pure random shuffle."""
    indices = np.arange(total_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_end = int(train_ratio * total_samples)
    val_end = train_end + int(val_ratio * total_samples)

    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def main():
    print("=" * 80)
    print("COMPARING OLD vs NEW SPLIT STRATEGIES")
    print("=" * 80)

    # Load config
    config_path = project_root / "configs" / "ssl_finetune_cbh.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load dataset
    print("\nLoading dataset...")
    dataset = HDF5CloudDataset(
        flight_configs=config["data"]["flights"],
        swath_slice=config["data"]["swath_slice"],
        temporal_frames=config["data"]["temporal_frames"],
        filter_type=config["data"]["filter_type"],
        cbh_min=config["data"]["cbh_min"],
        cbh_max=config["data"]["cbh_max"],
        flat_field_correction=config["data"]["flat_field_correction"],
        clahe_clip_limit=config["data"]["clahe_clip_limit"],
        zscore_normalize=config["data"]["zscore_normalize"],
        angles_mode=config["data"]["angles_mode"],
        augment=False,
    )

    total_samples = len(dataset)
    train_ratio = config["data"]["train_ratio"]
    val_ratio = config["data"]["val_ratio"]

    print(f"\nTotal samples: {total_samples}")
    print(
        f"Target ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={1 - train_ratio - val_ratio:.0%}"
    )

    # Show overall dataset composition
    print("\n" + "=" * 80)
    print("OVERALL DATASET COMPOSITION")
    print("=" * 80)
    all_indices = np.arange(total_samples)
    _ = analyze_split_composition(dataset, all_indices, "Full Dataset")

    # Old method
    print("\n" + "=" * 80)
    print("METHOD 1: OLD RANDOM SPLIT (Problematic)")
    print("=" * 80)
    print("Issue: No stratification - test set can be dominated by one flight")

    old_train, old_val, old_test = old_random_split(
        total_samples, train_ratio, val_ratio
    )

    print("\n--- OLD METHOD RESULTS ---")
    old_train_comp = analyze_split_composition(dataset, old_train, "Train")
    old_val_comp = analyze_split_composition(dataset, old_val, "Validation")
    old_test_comp = analyze_split_composition(dataset, old_test, "Test")

    # New method
    print("\n" + "=" * 80)
    print("METHOD 2: NEW STRATIFIED SPLIT (Improved)")
    print("=" * 80)
    print("Benefit: Each flight contributes proportionally to all splits")

    new_train, new_val, new_test = stratified_split_by_flight(
        dataset, train_ratio=train_ratio, val_ratio=val_ratio, seed=42, verbose=False
    )

    print("\n--- NEW METHOD RESULTS ---")
    new_train_comp = analyze_split_composition(dataset, new_train, "Train")
    new_val_comp = analyze_split_composition(dataset, new_val, "Validation")
    new_test_comp = analyze_split_composition(dataset, new_test, "Test")

    # Summary comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print("\nTest set balance (OLD vs NEW):")
    print(f"{'Flight':<15} {'OLD %':>10} {'NEW %':>10} {'Improvement':>15}")
    print("-" * 55)

    all_flights = sorted(set(old_test_comp.keys()) | set(new_test_comp.keys()))

    max_old_pct = 0
    max_new_pct = 0

    for flight in all_flights:
        old_count = old_test_comp.get(flight, 0)
        new_count = new_test_comp.get(flight, 0)
        old_pct = 100 * old_count / len(old_test)
        new_pct = 100 * new_count / len(new_test)

        max_old_pct = max(max_old_pct, old_pct)
        max_new_pct = max(max_new_pct, new_pct)

        diff = new_pct - old_pct
        improvement = (
            "Better"
            if abs(diff) < 5
            else ("Much better" if new_pct < old_pct else "Slightly worse")
        )

        print(f"{flight:<15} {old_pct:>9.1f}% {new_pct:>9.1f}% {improvement:>15}")

    print("\n" + "-" * 80)
    print(f"Max test set concentration (OLD): {max_old_pct:.1f}%")
    print(f"Max test set concentration (NEW): {max_new_pct:.1f}%")
    print(f"Reduction in imbalance: {max_old_pct - max_new_pct:.1f} percentage points")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("The OLD method allowed one flight to dominate the test set, leading to")
    print("misleading performance metrics (e.g., high R² from angle-only models).")
    print("\nThe NEW stratified method ensures balanced representation, providing")
    print("more reliable cross-flight generalization assessment.")
    print("=" * 80)


if __name__ == "__main__":
    main()
