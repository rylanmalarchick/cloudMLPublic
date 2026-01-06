#!/usr/bin/env python3
"""
Image CNN Model Training for CBH Retrieval

Trains a simple CNN on 20x22 pixel camera images with the same validation
strategies as the tabular model for fair comparison.

Key limitations of image data:
1. Very small images (20x22 = 440 pixels)
2. 8-bit autoscaling per-image (destroys absolute brightness information)
3. Limited spatial context for cloud detection

Expected results:
- Pooled K-fold: May show moderate R² due to autocorrelation
- Per-flight shuffled: Lower than tabular (less informative features)
- LOFO-CV: Expected to fail (domain shift + autoscaling)

Author: AgentBible-assisted development
Date: 2026-01-06
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class SimpleCNN(nn.Module):
    """Simple CNN for 20x22 grayscale images."""
    
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        
        # Input: (batch, 1, 20, 22)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # After 2 pooling layers: 20x22 -> 10x11 -> 5x5 (approx)
        # Actually: 20->10->5, 22->11->5
        self.fc1 = nn.Linear(128 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv blocks
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        x = self.relu(self.conv3(x))
        
        # Flatten and FC
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x.squeeze(-1)


def load_image_data(
    ssl_path: str,
    features_path: str,
) -> dict[str, Any]:
    """Load and match image data with CBH labels."""
    
    # Load SSL images
    with h5py.File(ssl_path, 'r') as f:
        ssl_images = f['images'][:]  # (N, 440)
        ssl_metadata = f['metadata'][:]  # (N, 4) - flight_id, sample_id, ...
    
    # Load labeled samples
    with h5py.File(features_path, 'r') as f:
        labeled_flight_ids = f['metadata/flight_id'][:]
        labeled_sample_ids = f['metadata/sample_id'][:]
        labeled_cbh_km = f['metadata/cbh_km'][:]
        flight_mapping = {int(k): v for k, v in json.loads(f.attrs['flight_mapping']).items()}
    
    # Create lookup: (flight_id, sample_id) -> ssl_index
    ssl_lookup = {}
    for i, (fid, sid) in enumerate(ssl_metadata[:, :2].astype(int)):
        ssl_lookup[(fid, sid)] = i
    
    # Match labeled samples to images
    images = []
    cbh_values = []
    flight_ids = []
    
    for i in range(len(labeled_cbh_km)):
        key = (int(labeled_flight_ids[i]), int(labeled_sample_ids[i]))
        if key in ssl_lookup:
            ssl_idx = ssl_lookup[key]
            img = ssl_images[ssl_idx].reshape(20, 22)
            
            # Z-score normalize
            mean, std = img.mean(), img.std()
            if std > 0:
                img = (img - mean) / std
            
            images.append(img)
            cbh_values.append(labeled_cbh_km[i])
            flight_ids.append(labeled_flight_ids[i])
    
    images = np.array(images)[:, np.newaxis, :, :]  # (N, 1, 20, 22)
    cbh_values = np.array(cbh_values)
    flight_ids = np.array(flight_ids)
    
    print(f"Matched {len(images)} images to CBH labels")
    print(f"Image shape: {images.shape}")
    print(f"CBH range: [{cbh_values.min():.3f}, {cbh_values.max():.3f}] km")
    
    return {
        'images': images,
        'cbh': cbh_values,
        'flight_ids': flight_ids,
        'flight_mapping': flight_mapping,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(targets)
    
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate model and return predictions."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    return np.array(all_preds), np.array(all_targets)


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for numpy arrays."""
    
    def __init__(self, images: np.ndarray, targets: np.ndarray):
        self.images = torch.from_numpy(images).float()
        self.targets = torch.from_numpy(targets).float()
    
    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.targets[idx]


def run_validation(
    images: np.ndarray,
    cbh: np.ndarray,
    flight_ids: np.ndarray,
    flight_mapping: dict[int, str],
    output_dir: Path,
    n_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """Run all validation strategies."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(cbh),
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'device': str(device),
    }
    
    # ========================================
    # Strategy 1: Pooled K-fold
    # ========================================
    print("\n" + "="*60)
    print("STRATEGY 1: Pooled K-fold (CNN)")
    print("="*60)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pooled_y_true, pooled_y_pred = [], []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(images)):
        dataset = SimpleDataset(images, cbh)
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size)
        
        model = SimpleCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(n_epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
        
        preds, targets = evaluate(model, test_loader, device)
        pooled_y_pred.extend(preds)
        pooled_y_true.extend(targets)
        
        fold_r2 = r2_score(targets, preds)
        print(f"  Fold {fold_idx+1}: R² = {fold_r2:.4f}")
    
    pooled_y_true = np.array(pooled_y_true)
    pooled_y_pred = np.array(pooled_y_pred)
    
    results['pooled_kfold'] = {
        'r2': float(r2_score(pooled_y_true, pooled_y_pred)),
        'mae_km': float(mean_absolute_error(pooled_y_true, pooled_y_pred)),
        'rmse_km': float(np.sqrt(mean_squared_error(pooled_y_true, pooled_y_pred))),
    }
    print(f"Pooled R² = {results['pooled_kfold']['r2']:.4f}")
    
    # ========================================
    # Strategy 2: Per-flight shuffled K-fold
    # ========================================
    print("\n" + "="*60)
    print("STRATEGY 2: Per-flight K-fold shuffled (CNN)")
    print("="*60)
    
    unique_flights = np.unique(flight_ids)
    flight_r2s = {}
    shuffled_y_true, shuffled_y_pred = [], []
    
    for fid in unique_flights:
        flight_name = flight_mapping.get(fid, f"flight_{fid}")
        mask = flight_ids == fid
        X_f, y_f = images[mask], cbh[mask]
        
        if len(y_f) < 20:
            print(f"  {flight_name}: skipped ({len(y_f)} samples)")
            continue
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        f_true, f_pred = [], []
        
        for train_idx, test_idx in kf.split(X_f):
            dataset = SimpleDataset(X_f, y_f)
            train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(Subset(dataset, test_idx.tolist()), batch_size=batch_size)
            
            model = SimpleCNN().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            for epoch in range(n_epochs):
                train_epoch(model, train_loader, optimizer, criterion, device)
            
            preds, targets = evaluate(model, test_loader, device)
            f_pred.extend(preds)
            f_true.extend(targets)
        
        f_r2 = r2_score(f_true, f_pred)
        flight_r2s[flight_name] = f_r2
        shuffled_y_true.extend(f_true)
        shuffled_y_pred.extend(f_pred)
        print(f"  {flight_name}: R² = {f_r2:.4f} ({len(y_f)} samples)")
    
    avg_r2 = np.mean(list(flight_r2s.values()))
    results['per_flight_shuffled'] = {
        'r2': float(avg_r2),
        'mae_km': float(mean_absolute_error(shuffled_y_true, shuffled_y_pred)),
        'rmse_km': float(np.sqrt(mean_squared_error(shuffled_y_true, shuffled_y_pred))),
        'per_flight_r2': flight_r2s,
    }
    print(f"Average R² = {avg_r2:.4f}")
    
    # ========================================
    # Strategy 3: LOFO-CV
    # ========================================
    print("\n" + "="*60)
    print("STRATEGY 3: Leave-One-Flight-Out (CNN)")
    print("="*60)
    
    lofo_r2s = {}
    lofo_y_true, lofo_y_pred = [], []
    
    for test_fid in unique_flights:
        flight_name = flight_mapping.get(test_fid, f"flight_{test_fid}")
        train_mask = flight_ids != test_fid
        test_mask = flight_ids == test_fid
        
        X_train, y_train = images[train_mask], cbh[train_mask]
        X_test, y_test = images[test_mask], cbh[test_mask]
        
        if len(y_test) < 5:
            print(f"  {flight_name}: skipped ({len(y_test)} samples)")
            continue
        
        train_dataset = SimpleDataset(X_train, y_train)
        test_dataset = SimpleDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        model = SimpleCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(n_epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
        
        preds, targets = evaluate(model, test_loader, device)
        f_r2 = r2_score(targets, preds)
        lofo_r2s[flight_name] = f_r2
        lofo_y_true.extend(targets)
        lofo_y_pred.extend(preds)
        print(f"  Test on {flight_name}: R² = {f_r2:.4f}")
    
    avg_r2_lofo = np.mean(list(lofo_r2s.values()))
    results['lofo_cv'] = {
        'r2': float(avg_r2_lofo),
        'mae_km': float(mean_absolute_error(lofo_y_true, lofo_y_pred)),
        'rmse_km': float(np.sqrt(mean_squared_error(lofo_y_true, lofo_y_pred))),
        'per_flight_r2': lofo_r2s,
    }
    print(f"Average LOFO R² = {avg_r2_lofo:.4f}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*60)
    print("CNN VALIDATION SUMMARY")
    print("="*60)
    print(f"{'Strategy':<30} {'R²':>10} {'MAE (m)':>12}")
    print("-"*60)
    for key, label in [('pooled_kfold', 'Pooled K-fold'),
                        ('per_flight_shuffled', 'Per-flight shuffled'),
                        ('lofo_cv', 'LOFO-CV')]:
        r = results[key]
        print(f"{label:<30} {r['r2']:>10.4f} {r['mae_km']*1000:>12.1f}")
    
    # Save results
    results_path = output_dir / 'cnn_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train CNN on camera images for CBH")
    parser.add_argument('--ssl-path', type=str, default='data_ssl/images/train.h5')
    parser.add_argument('--features-path', type=str, 
                        default='outputs/preprocessed_data/Clean_933_Integrated_Features.hdf5')
    parser.add_argument('--output-dir', type=str, default='outputs/image_model')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_image_data(args.ssl_path, args.features_path)
    
    # Run validation
    run_validation(
        images=data['images'],
        cbh=data['cbh'],
        flight_ids=data['flight_ids'],
        flight_mapping=data['flight_mapping'],
        output_dir=output_dir,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
