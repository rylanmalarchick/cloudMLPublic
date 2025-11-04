#!/usr/bin/env python3
"""Check what's in the test set"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import yaml
from src.hdf5_dataset import HDF5CloudDataset
from torch.utils.data import Subset
from src.split_utils import (
    stratified_split_by_flight,
    analyze_split_balance,
    check_split_leakage,
)

# Load config
with open("configs/ssl_finetune_cbh.yaml") as f:
    config = yaml.safe_load(f)

# Load dataset
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

# Create same splits as ablation using stratified splitting
total_samples = len(dataset)
train_ratio = config["data"]["train_ratio"]
val_ratio = config["data"]["val_ratio"]

print("\nCreating stratified train/val/test splits...")
train_indices, val_indices, test_indices = stratified_split_by_flight(
    dataset, train_ratio=train_ratio, val_ratio=val_ratio, seed=42, verbose=True
)

# Verify no leakage
check_split_leakage(train_indices, val_indices, test_indices)

print(f"Total: {total_samples}, Test: {len(test_indices)}")
print(f"\nTest set composition:")

# Map global indices to flights
flight_counts = {}
for idx in test_indices:
    global_idx = int(dataset.indices[idx])
    flight_idx, local_idx = dataset.global_to_local[global_idx]
    flight_name = dataset.flight_configs[flight_idx]["name"]
    flight_counts[flight_name] = flight_counts.get(flight_name, 0) + 1

for flight, count in sorted(flight_counts.items()):
    print(f"  {flight}: {count} samples ({100 * count / len(test_indices):.1f}%)")
