#!/usr/bin/env python3
"""Quick test: Stratified K-Fold vs LOO CV"""
import sys
sys.path.insert(0, '.')
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import h5py

# Load features
with h5py.File("sow_outputs/wp2_atmospheric/WP2_Features.hdf5", "r") as f:
    era5_features = f["features"][:]

with h5py.File("sow_outputs/wp1_geometric/WP1_Features.hdf5", "r") as f:
    geo_features = np.column_stack([
        f["derived_geometric_H"][:],
        f["shadow_length_pixels"][:],
        f["shadow_detection_confidence"][:]
    ])

# Handle NaNs
for j in range(geo_features.shape[1]):
    col = geo_features[:, j]
    if np.any(np.isnan(col)):
        geo_features[np.isnan(col), j] = np.nanmedian(col)

# Load targets
from src.hdf5_dataset import HDF5CloudDataset
import yaml

with open("configs/bestComboConfig.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset = HDF5CloudDataset(
    flight_configs=config["flights"],
    indices=None,
    augment=False,
    swath_slice=[40, 480],
    temporal_frames=1,
)
y = dataset.get_unscaled_y()

# Combine features
X = np.column_stack([era5_features, geo_features])

# Get flight IDs
flight_ids = np.zeros(len(dataset), dtype=int)
flight_mapping = {0: "30Oct24", 1: "10Feb25", 2: "23Oct24", 3: "12Feb25", 4: "18Feb25"}
flight_name_to_id = {v: k for k, v in flight_mapping.items()}

for i in range(len(dataset)):
    _, _, _, _, global_idx, _ = dataset[i]
    flight_idx, _ = dataset.global_to_local[int(global_idx)]
    flight_name = dataset.flight_data[flight_idx]["name"]
    flight_ids[i] = flight_name_to_id.get(flight_name, flight_idx)

print("="*70)
print("Comparison: LOO CV vs Stratified 5-Fold CV")
print("="*70)
print("Model: Random Forest (ERA5 + Geometric features)")
print()

# LOO CV (actually leave-one-flight-out)
print("Leave-One-Flight-Out CV:")
print("-"*70)
loo_results = []
for fold_id in range(5):
    test_mask = flight_ids == fold_id
    train_mask = ~test_mask
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"  Fold {fold_id} ({flight_mapping[fold_id]:8s}): R²={r2:7.4f}, MAE={mae:.4f} km, n={len(y_test)}")
    loo_results.append(r2)

print(f"\n  LOO Mean R²: {np.mean(loo_results):.4f} ± {np.std(loo_results):.4f}")

# Stratified K-Fold (stratify by binned CBH values)
print("\nStratified 5-Fold CV (randomized, stratified by CBH):")
print("-"*70)

# Bin CBH for stratification
cbh_bins = np.digitize(y, bins=[0.3, 0.6, 0.9, 1.2])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

kfold_results = []
for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, cbh_bins)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"  Fold {fold_id}: R²={r2:7.4f}, MAE={mae:.4f} km, n_train={len(y_train)}, n_test={len(y_test)}")
    kfold_results.append(r2)

print(f"\n  K-Fold Mean R²: {np.mean(kfold_results):.4f} ± {np.std(kfold_results):.4f}")

print("\n" + "="*70)
print("Conclusion:")
print(f"  Improvement with stratified K-fold: {np.mean(kfold_results) - np.mean(loo_results):+.4f} R²")
print("="*70)
