#!/usr/bin/env python3
"""
WP-4 Training Monitor and Progress Tracker

Monitors training logs and reports progress in real-time.
"""

import time
import os
import json
from pathlib import Path
from datetime import datetime
import sys


def parse_log_for_metrics(log_path):
    """Extract metrics from training log."""
    if not os.path.exists(log_path):
        return None

    with open(log_path, "r") as f:
        lines = f.readlines()

    results = {
        "folds_complete": 0,
        "current_fold": None,
        "fold_results": [],
        "status": "initializing",
        "errors": [],
    }

    for line in lines:
        # Check for fold start
        if "Fold" in line and "Test on" in line:
            try:
                fold_id = int(line.split("Fold")[1].split(":")[0].strip())
                flight = line.split("Test on")[1].strip().split()[0]
                results["current_fold"] = {"fold_id": fold_id, "flight": flight}
                results["status"] = "training"
            except:
                pass

        # Check for results
        if "Results:" in line:
            results["status"] = "evaluating"

        if "R² =" in line or "R2 =" in line:
            try:
                r2 = float(line.split("=")[1].strip().split()[0])
                if results["current_fold"] is not None:
                    results["current_fold"]["r2"] = r2
            except:
                pass

        if "MAE =" in line:
            try:
                mae = float(line.split("=")[1].strip().split()[0])
                if results["current_fold"] is not None:
                    results["current_fold"]["mae"] = mae
            except:
                pass

        if "RMSE =" in line:
            try:
                rmse = float(line.split("=")[1].strip().split()[0])
                if results["current_fold"] is not None:
                    results["current_fold"]["rmse"] = rmse
                    # Fold complete
                    results["fold_results"].append(results["current_fold"].copy())
                    results["folds_complete"] += 1
                    results["current_fold"] = None
            except:
                pass

        # Check for errors
        if "Error" in line or "Traceback" in line:
            results["errors"].append(line.strip())
            results["status"] = "error"

    return results


def format_status(mode, metrics):
    """Format status string."""
    if metrics is None:
        return f"{mode:12s} | Not started"

    status = metrics["status"]
    folds = metrics["folds_complete"]

    if status == "error":
        return f"{mode:12s} | ERROR (see log)"
    elif status == "initializing":
        return f"{mode:12s} | Initializing..."
    elif folds == 5:
        # Complete - show aggregate
        r2_vals = [f["r2"] for f in metrics["fold_results"] if "r2" in f]
        if r2_vals:
            mean_r2 = sum(r2_vals) / len(r2_vals)
            return f"{mode:12s} | ✓ COMPLETE | Mean R²: {mean_r2:+.4f}"
        return f"{mode:12s} | ✓ COMPLETE"
    else:
        curr = metrics.get("current_fold")
        if curr:
            fold_id = curr["fold_id"]
            flight = curr["flight"]
            return f"{mode:12s} | Fold {fold_id}/4 ({flight}) | {folds} folds done"
        else:
            return f"{mode:12s} | {folds}/5 folds complete"


def print_progress():
    """Print current progress."""
    base_dir = Path("sow_outputs/wp4_hybrid")

    modes = ["image_only", "concat", "attention"]

    print("\n" + "=" * 80)
    print("WP-4 TRAINING PROGRESS")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"{'Model':<12s} | Status")
    print("-" * 80)

    all_complete = True
    any_running = False

    for mode in modes:
        log_path = base_dir / f"training_{mode}.log"
        metrics = parse_log_for_metrics(log_path)
        status_str = format_status(mode, metrics)
        print(status_str)

        if metrics and metrics["folds_complete"] < 5:
            all_complete = False
            if metrics["status"] in ["training", "evaluating"]:
                any_running = True
        elif metrics is None:
            all_complete = False

    print("=" * 80)

    # Check for report files
    reports = []
    for mode in modes:
        report_path = base_dir / f"WP4_Report_{mode}.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
                agg = report.get("aggregate_metrics", {})
                reports.append(
                    {
                        "mode": mode,
                        "r2": agg.get("mean_r2"),
                        "mae": agg.get("mean_mae_km"),
                        "rmse": agg.get("mean_rmse_km"),
                    }
                )

    if reports:
        print("\nCOMPLETED MODELS:")
        print("-" * 80)
        print(f"{'Model':<12s} | {'R²':>10s} | {'MAE (km)':>10s} | {'RMSE (km)':>10s}")
        print("-" * 80)
        for r in reports:
            if r["r2"] is not None:
                print(
                    f"{r['mode']:<12s} | {r['r2']:>+10.4f} | {r['mae']:>10.4f} | {r['rmse']:>10.4f}"
                )
        print("=" * 80)

    return all_complete, any_running


def monitor_loop(interval=30):
    """Monitor training in a loop."""
    try:
        while True:
            all_complete, any_running = print_progress()

            if all_complete:
                print("\n🎉 ALL TRAINING COMPLETE!")
                break
            elif not any_running:
                print("\n⚠️  No training appears to be running. Check logs.")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--loop":
        monitor_loop(30)
    else:
        print_progress()
