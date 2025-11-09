#!/usr/bin/env python3
"""Monitor K-Fold CV training progress."""
import json
import os
import time
from pathlib import Path
from datetime import datetime

output_dir = Path("sow_outputs/wp4_cnn")

print("=" * 80)
print("WP-4 K-FOLD CV TRAINING MONITOR")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Check for completed reports
modes = ["image_only", "concat", "attention"]
completed = []
in_progress = []

for mode in modes:
    report_path = output_dir / f"WP4_Report_{mode}.json"
    if report_path.exists():
        completed.append(mode)
    else:
        # Check for model checkpoints to infer progress
        checkpoints = list(output_dir.glob(f"model_{mode}_fold*.pth"))
        if checkpoints:
            in_progress.append((mode, len(checkpoints)))
        else:
            in_progress.append((mode, 0))

print(f"{'Model':<15} | {'Status':<20}")
print("-" * 80)
for mode in modes:
    if mode in completed:
        print(f"{mode:<15} | Completed ✓")
    else:
        n_folds = next((n for m, n in in_progress if m == mode), 0)
        if n_folds > 0:
            print(f"{mode:<15} | In progress ({n_folds}/5 folds)")
        else:
            print(f"{mode:<15} | Not started")

print("=" * 80)

# Show completed results
if completed:
    print("\nCOMPLETED RESULTS:")
    print("-" * 80)
    print(f"{'Model':<15} | {'Mean R²':<12} | {'MAE (km)':<12} | {'RMSE (km)':<12}")
    print("-" * 80)
    
    for mode in completed:
        report_path = output_dir / f"WP4_Report_{mode}.json"
        with open(report_path, "r") as f:
            report = json.load(f)
        
        agg = report["aggregate_metrics"]
        print(f"{mode:<15} | {agg['mean_r2']:>6.4f} ± {agg['std_r2']:<4.4f} | "
              f"{agg['mean_mae_km']:>6.4f} ± {agg['std_mae_km']:<4.4f} | "
              f"{agg['mean_rmse_km']:>6.4f} ± {agg['std_rmse_km']:<4.4f}")
    print("=" * 80)

# Check log file
log_file = "sow_outputs/wp4_kfold_training.log"
if os.path.exists(log_file):
    print("\nLAST 20 LINES OF LOG:")
    print("-" * 80)
    os.system(f"tail -20 {log_file}")
