#!/usr/bin/env python3
"""
Hybrid Model Training (Tabular + Image)

Combines ERA5 tabular features with CNN image features for CBH prediction.
Tests whether multi-modal fusion provides any benefit.

Expected: Minimal improvement over tabular-only, since:
1. Images provide weak signal (R² ≈ 0 per-flight)
2. Tabular features already capture most predictable variance
3. Late fusion may not effectively combine modalities

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
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class HybridCNN(nn.Module):
    """Hybrid model combining image CNN with tabular features."""
    
    def __init__(self, n_tabular_features: int = 10, dropout: float = 0.3):
        super().__init__()
        
        # Image branch (same as SimpleCNN)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Image feature extractor output: 128 * 5 * 5 = 3200 -> 64
        self.img_fc = nn.Linear(128 * 5 * 5, 64)
        
        # Tabular branch
        self.tab_fc1 = nn.Linear(n_tabular_features, 32)
        self.tab_fc2 = nn.Linear(32, 32)
        
        # Fusion (64 + 32 = 96)
        self.dropout = nn.Dropout(dropout)
        self.fusion_fc1 = nn.Linear(96, 64)
        self.fusion_fc2 = nn.Linear(64, 1)
        
    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        # Image branch
        x_img = self.relu(self.conv1(image))
        x_img = self.pool(x_img)
        x_img = self.relu(self.conv2(x_img))
        x_img = self.pool(x_img)
        x_img = self.relu(self.conv3(x_img))
        x_img = x_img.view(x_img.size(0), -1)
        x_img = self.relu(self.img_fc(x_img))
        
        # Tabular branch
        x_tab = self.relu(self.tab_fc1(tabular))
        x_tab = self.relu(self.tab_fc2(x_tab))
        
        # Fusion
        x = torch.cat([x_img, x_tab], dim=1)
        x = self.dropout(self.relu(self.fusion_fc1(x)))
        x = self.fusion_fc2(x)
        
        return x.squeeze(-1)


class HybridDataset(Dataset):
    """Dataset for hybrid model with images and tabular features."""
    
    def __init__(self, images: np.ndarray, tabular: np.ndarray, targets: np.ndarray):
        self.images = torch.from_numpy(images).float()
        self.tabular = torch.from_numpy(tabular).float()
        self.targets = torch.from_numpy(targets).float()
    
    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.images[idx], self.tabular[idx], self.targets[idx]


def load_hybrid_data(
    ssl_path: str,
    features_path: str,
) -> dict[str, Any]:
    """Load matched image and tabular data."""
    
    # Load SSL images
    with h5py.File(ssl_path, 'r') as f:
        ssl_images = f['images'][:]
        ssl_metadata = f['metadata'][:]
    
    # Load tabular features
    with h5py.File(features_path, 'r') as f:
        feature_names = json.loads(f.attrs['feature_names'])
        flight_mapping = {int(k): v for k, v in json.loads(f.attrs['flight_mapping']).items()}
        
        labeled_flight_ids = f['metadata/flight_id'][:]
        labeled_sample_ids = f['metadata/sample_id'][:]
        labeled_cbh_km = f['metadata/cbh_km'][:]
        
        # Build tabular features
        tabular_parts = []
        for name in feature_names:
            if name in f['atmospheric_features']:
                tabular_parts.append(f['atmospheric_features'][name][:])
            elif name in f['geometric_features']:
                tabular_parts.append(f['geometric_features'][name][:])
        tabular_all = np.column_stack(tabular_parts)
    
    # Create lookup
    ssl_lookup = {}
    for i, (fid, sid) in enumerate(ssl_metadata[:, :2].astype(int)):
        ssl_lookup[(fid, sid)] = i
    
    # Match samples
    images, tabular, cbh_values, flight_ids = [], [], [], []
    
    for i in range(len(labeled_cbh_km)):
        key = (int(labeled_flight_ids[i]), int(labeled_sample_ids[i]))
        if key in ssl_lookup:
            ssl_idx = ssl_lookup[key]
            img = ssl_images[ssl_idx].reshape(20, 22)
            
            # Normalize image
            mean, std = img.mean(), img.std()
            if std > 0:
                img = (img - mean) / std
            
            images.append(img)
            tabular.append(tabular_all[i])
            cbh_values.append(labeled_cbh_km[i])
            flight_ids.append(labeled_flight_ids[i])
    
    images = np.array(images)[:, np.newaxis, :, :]
    tabular = np.array(tabular)
    
    # Normalize tabular features
    tab_mean = tabular.mean(axis=0)
    tab_std = tabular.std(axis=0)
    tab_std[tab_std == 0] = 1
    tabular = (tabular - tab_mean) / tab_std
    
    print(f"Matched {len(images)} samples")
    print(f"Image shape: {images.shape}")
    print(f"Tabular shape: {tabular.shape}")
    
    return {
        'images': images,
        'tabular': tabular,
        'cbh': np.array(cbh_values),
        'flight_ids': np.array(flight_ids),
        'flight_mapping': flight_mapping,
        'feature_names': feature_names,
    }


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, tabular, targets in loader:
        images, tabular, targets = images.to(device), tabular.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images, tabular)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, tabular, target in loader:
            images, tabular = images.to(device), tabular.to(device)
            output = model(images, tabular)
            preds.extend(output.cpu().numpy())
            targets.extend(target.numpy())
    return np.array(preds), np.array(targets)


def run_hybrid_validation(data: dict, output_dir: Path, n_epochs: int = 30, batch_size: int = 32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    images = data['images']
    tabular = data['tabular']
    cbh = data['cbh']
    flight_ids = data['flight_ids']
    flight_mapping = data['flight_mapping']
    
    results = {'timestamp': datetime.now().isoformat(), 'n_samples': len(cbh)}
    
    # Pooled K-fold
    print("\n" + "="*60)
    print("HYBRID: Pooled K-fold")
    print("="*60)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pooled_y_true, pooled_y_pred = [], []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(images)):
        dataset = HybridDataset(images, tabular, cbh)
        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, train_idx), batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            torch.utils.data.Subset(dataset, test_idx), batch_size=batch_size
        )
        
        model = HybridCNN(n_tabular_features=tabular.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        for _ in range(n_epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
        
        preds, targets = evaluate(model, test_loader, device)
        pooled_y_pred.extend(preds)
        pooled_y_true.extend(targets)
        print(f"  Fold {fold_idx+1}: R² = {r2_score(targets, preds):.4f}")
    
    results['pooled_kfold'] = {
        'r2': float(r2_score(pooled_y_true, pooled_y_pred)),
        'mae_km': float(mean_absolute_error(pooled_y_true, pooled_y_pred)),
    }
    print(f"Pooled R² = {results['pooled_kfold']['r2']:.4f}")
    
    # Per-flight shuffled
    print("\n" + "="*60)
    print("HYBRID: Per-flight shuffled K-fold")
    print("="*60)
    
    unique_flights = np.unique(flight_ids)
    flight_r2s = {}
    
    for fid in unique_flights:
        flight_name = flight_mapping.get(fid, f"flight_{fid}")
        mask = flight_ids == fid
        if mask.sum() < 20:
            continue
        
        X_img, X_tab, y = images[mask], tabular[mask], cbh[mask]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        f_true, f_pred = [], []
        
        for train_idx, test_idx in kf.split(X_img):
            dataset = HybridDataset(X_img, X_tab, y)
            train_loader = DataLoader(
                torch.utils.data.Subset(dataset, train_idx.tolist()), batch_size=batch_size, shuffle=True
            )
            test_loader = DataLoader(
                torch.utils.data.Subset(dataset, test_idx.tolist()), batch_size=batch_size
            )
            
            model = HybridCNN(n_tabular_features=X_tab.shape[1]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            
            for _ in range(n_epochs):
                train_epoch(model, train_loader, optimizer, criterion, device)
            
            preds, targets = evaluate(model, test_loader, device)
            f_pred.extend(preds)
            f_true.extend(targets)
        
        f_r2 = r2_score(f_true, f_pred)
        flight_r2s[flight_name] = f_r2
        print(f"  {flight_name}: R² = {f_r2:.4f}")
    
    results['per_flight_shuffled'] = {
        'r2': float(np.mean(list(flight_r2s.values()))),
        'per_flight_r2': flight_r2s,
    }
    print(f"Average R² = {results['per_flight_shuffled']['r2']:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("HYBRID MODEL SUMMARY")
    print("="*60)
    print(f"Pooled K-fold R²: {results['pooled_kfold']['r2']:.4f}")
    print(f"Per-flight shuffled R²: {results['per_flight_shuffled']['r2']:.4f}")
    
    # Save
    with open(output_dir / 'hybrid_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ssl-path', default='data_ssl/images/train.h5')
    parser.add_argument('--features-path', default='outputs/preprocessed_data/Clean_933_Integrated_Features.hdf5')
    parser.add_argument('--output-dir', default='outputs/hybrid_model')
    parser.add_argument('--epochs', type=int, default=20)
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = load_hybrid_data(args.ssl_path, args.features_path)
    run_hybrid_validation(data, output_dir, n_epochs=args.epochs)


if __name__ == '__main__':
    main()
