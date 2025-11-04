#!/usr/bin/env python3
"""Quick check: do scaled angles correlate better with CBH than raw angles?"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import yaml
from src.hdf5_dataset import HDF5CloudDataset

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

print(f"Total samples: {len(dataset)}")

# Extract scaled and raw angles + CBH
scaled_sza, scaled_saa, raw_sza, raw_saa, cbh = [], [], [], [], []

for idx in range(len(dataset)):
    global_idx = int(dataset.indices[idx])
    flight_idx, local_idx = dataset.global_to_local[global_idx]
    flight_info = dataset.flight_data[flight_idx]
    
    # Raw angles (degrees)
    raw_sza.append(flight_info["SZA_full"][local_idx, 0])
    raw_saa.append(flight_info["SAA_full"][local_idx, 0])
    
    # CBH (km)
    cbh.append(flight_info["Y_full"][local_idx].flatten()[0])
    
    # Scaled angles (z-score)
    sza_scaled = dataset.sza_scaler.transform([[raw_sza[-1]]])[0, 0]
    saa_scaled = dataset.saa_scaler.transform([[raw_saa[-1]]])[0, 0]
    scaled_sza.append(sza_scaled)
    scaled_saa.append(saa_scaled)

# Convert to arrays
raw_sza = np.array(raw_sza)
raw_saa = np.array(raw_saa)
scaled_sza = np.array(scaled_sza)
scaled_saa = np.array(scaled_saa)
cbh = np.array(cbh)

# Compute correlations
print("\n" + "="*60)
print("RAW ANGLES (degrees):")
print(f"  SZA vs CBH: r = {np.corrcoef(raw_sza, cbh)[0,1]:.4f}")
print(f"  SAA vs CBH: r = {np.corrcoef(raw_saa, cbh)[0,1]:.4f}")

print("\nSCALED ANGLES (z-score):")
print(f"  SZA vs CBH: r = {np.corrcoef(scaled_sza, cbh)[0,1]:.4f}")
print(f"  SAA vs CBH: r = {np.corrcoef(scaled_saa, cbh)[0,1]:.4f}")

print("\nSCALED ANGLES vs SCALED CBH:")
cbh_scaled = dataset.y_scaler.transform(cbh.reshape(-1, 1)).flatten()
print(f"  SZA vs CBH: r = {np.corrcoef(scaled_sza, cbh_scaled)[0,1]:.4f}")
print(f"  SAA vs CBH: r = {np.corrcoef(scaled_saa, cbh_scaled)[0,1]:.4f}")
print("="*60)

# Check SZA distribution for flight 10Feb25
print("\nFlight 10Feb25 SZA distribution:")
print(f"  Min: {raw_sza[:163].min():.2f}°")
print(f"  Max: {raw_sza[:163].max():.2f}°")
print(f"  Std: {raw_sza[:163].std():.4f}°")
print("  (NaN correlation likely due to near-constant SZA)")
