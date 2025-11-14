"""
Utilities for creating stratified train/val/test splits that ensure
balanced representation across flights.
"""

import numpy as np
from typing import Tuple, Dict, List
from collections import defaultdict


def stratified_split_by_flight(
    dataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create stratified train/val/test splits that preserve flight proportions.

    This ensures that each split (train/val/test) contains samples from all flights
    in roughly the same proportions as the overall dataset, preventing test-set
    imbalance issues where one flight dominates.

    Args:
        dataset: HDF5CloudDataset instance with global_to_local mapping
        train_ratio: Fraction of data for training (default: 0.70)
        val_ratio: Fraction of data for validation (default: 0.15)
        seed: Random seed for reproducibility (default: 42)
        verbose: Print split statistics (default: True)

    Returns:
        train_indices: Array of global indices for training
        val_indices: Array of global indices for validation
        test_indices: Array of global indices for testing

    Example:
        >>> train_idx, val_idx, test_idx = stratified_split_by_flight(
        ...     dataset, train_ratio=0.7, val_ratio=0.15, seed=42
        ... )
    """
    np.random.seed(seed)

    # Test ratio is implicit
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be <= 1.0"
        )

    # Group indices by flight
    flight_groups = defaultdict(list)

    for global_idx in range(len(dataset.global_to_local)):
        if global_idx in dataset.indices:
            flight_idx, local_idx = dataset.global_to_local[global_idx]
            flight_name = dataset.flight_data[flight_idx]["name"]
            flight_groups[flight_name].append(global_idx)

    # Prepare lists for final indices
    train_indices = []
    val_indices = []
    test_indices = []

    # Split each flight's samples proportionally
    for flight_name, indices in flight_groups.items():
        indices = np.array(indices)
        n_samples = len(indices)

        # Shuffle within flight
        np.random.shuffle(indices)

        # Calculate split points
        train_end = int(train_ratio * n_samples)
        val_end = train_end + int(val_ratio * n_samples)

        # Ensure at least 1 sample in each split if flight has enough samples
        if n_samples >= 3:
            # Standard split
            flight_train = indices[:train_end]
            flight_val = indices[train_end:val_end]
            flight_test = indices[val_end:]

            # Handle edge case where a split might be empty
            if len(flight_train) == 0 and n_samples > 0:
                flight_train = indices[:1]
                flight_val = indices[1:2] if n_samples > 1 else []
                flight_test = indices[2:] if n_samples > 2 else []
            elif len(flight_val) == 0 and n_samples > 1:
                flight_val = indices[train_end : train_end + 1]
                flight_test = indices[train_end + 1 :]
            elif len(flight_test) == 0 and n_samples > 2:
                flight_test = indices[-1:]
                if len(flight_val) > 1:
                    flight_val = indices[train_end:-1]
        elif n_samples == 2:
            # If only 2 samples, put 1 in train and 1 in test
            flight_train = indices[:1]
            flight_val = []
            flight_test = indices[1:]
        elif n_samples == 1:
            # If only 1 sample, put it in train
            flight_train = indices
            flight_val = []
            flight_test = []
        else:
            continue

        train_indices.extend(flight_train)
        val_indices.extend(flight_val)
        test_indices.extend(flight_test)

    # Convert to numpy arrays
    train_indices = np.array(train_indices, dtype=int)
    val_indices = np.array(val_indices, dtype=int)
    test_indices = np.array(test_indices, dtype=int)

    # Shuffle final splits to mix flights
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    if verbose:
        print("\n" + "=" * 80)
        print("STRATIFIED SPLIT SUMMARY")
        print("=" * 80)
        print(
            f"Total samples: {len(train_indices) + len(val_indices) + len(test_indices)}"
        )
        print(
            f"  Train: {len(train_indices)} ({len(train_indices) / (len(train_indices) + len(val_indices) + len(test_indices)) * 100:.1f}%)"
        )
        print(
            f"  Val:   {len(val_indices)} ({len(val_indices) / (len(train_indices) + len(val_indices) + len(test_indices)) * 100:.1f}%)"
        )
        print(
            f"  Test:  {len(test_indices)} ({len(test_indices) / (len(train_indices) + len(val_indices) + len(test_indices)) * 100:.1f}%)"
        )
        print("\nPer-flight distribution:")
        print(f"{'Flight':<15} {'Total':>8} {'Train':>8} {'Val':>8} {'Test':>8}")
        print("-" * 55)

        for flight_name in sorted(flight_groups.keys()):
            indices = flight_groups[flight_name]
            n_total = len(indices)
            n_train = sum(1 for idx in indices if idx in train_indices)
            n_val = sum(1 for idx in indices if idx in val_indices)
            n_test = sum(1 for idx in indices if idx in test_indices)
            print(f"{flight_name:<15} {n_total:>8} {n_train:>8} {n_val:>8} {n_test:>8}")

        print("=" * 80 + "\n")

    return train_indices, val_indices, test_indices


def get_flight_labels(dataset) -> np.ndarray:
    """
    Extract flight labels for each sample in the dataset.

    Args:
        dataset: HDF5CloudDataset instance

    Returns:
        flight_labels: Array of flight indices (one per sample)
    """
    flight_labels = []

    for global_idx in range(len(dataset)):
        actual_global_idx = dataset.indices[global_idx]
        flight_idx, _ = dataset.global_to_local[actual_global_idx]
        flight_labels.append(flight_idx)

    return np.array(flight_labels)


def analyze_split_balance(
    dataset,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    print_results: bool = True,
) -> Dict[str, Dict]:
    """
    Analyze the balance of flights across train/val/test splits.

    Args:
        dataset: HDF5CloudDataset instance
        train_indices: Training indices
        val_indices: Validation indices
        test_indices: Test indices
        print_results: Whether to print analysis (default: True)

    Returns:
        Dictionary with balance statistics per flight and split
    """
    # Build flight index mapping
    flight_groups = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0})

    for global_idx in range(len(dataset.global_to_local)):
        if global_idx not in dataset.indices:
            continue

        flight_idx, _ = dataset.global_to_local[global_idx]
        flight_name = dataset.flight_data[flight_idx]["name"]
        flight_groups[flight_name]["total"] += 1

        if global_idx in train_indices:
            flight_groups[flight_name]["train"] += 1
        elif global_idx in val_indices:
            flight_groups[flight_name]["val"] += 1
        elif global_idx in test_indices:
            flight_groups[flight_name]["test"] += 1

    if print_results:
        print("\n" + "=" * 80)
        print("SPLIT BALANCE ANALYSIS")
        print("=" * 80)
        print(f"{'Flight':<15} {'Total':>8} {'Train%':>10} {'Val%':>10} {'Test%':>10}")
        print("-" * 80)

        for flight_name in sorted(flight_groups.keys()):
            stats = flight_groups[flight_name]
            total = stats["total"]
            if total > 0:
                train_pct = (stats["train"] / total) * 100
                val_pct = (stats["val"] / total) * 100
                test_pct = (stats["test"] / total) * 100
                print(
                    f"{flight_name:<15} {total:>8} {train_pct:>9.1f}% {val_pct:>9.1f}% {test_pct:>9.1f}%"
                )

        print("=" * 80 + "\n")

    return dict(flight_groups)


def check_split_leakage(
    train_indices: np.ndarray, val_indices: np.ndarray, test_indices: np.ndarray
) -> bool:
    """
    Check for index leakage between splits.

    Args:
        train_indices: Training indices
        val_indices: Validation indices
        test_indices: Test indices

    Returns:
        True if splits are valid (no leakage), False otherwise
    """
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)

    if has_leakage:
        print("WARNING: Split leakage detected!")
        if train_val_overlap:
            print(f"  Train-Val overlap: {len(train_val_overlap)} samples")
        if train_test_overlap:
            print(f"  Train-Test overlap: {len(train_test_overlap)} samples")
        if val_test_overlap:
            print(f"  Val-Test overlap: {len(val_test_overlap)} samples")
        return False
    else:
        print(" No split leakage detected - all splits are disjoint")
        return True
