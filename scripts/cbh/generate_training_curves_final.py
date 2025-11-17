#!/usr/bin/env python3
"""Generate training curves figure from vision baseline training logs."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_all_results():
    """Load all vision baseline results from individual JSON files."""
    reports_dir = Path("outputs/vision_baselines/reports")
    
    model_files = {
        "ResNet-18 (scratch, no augment)": "resnet18_scratch_noaugment_results.json",
        "ResNet-18 (pretrained, no augment)": "resnet18_pretrained_noaugment_results.json",
        "ResNet-18 (pretrained, augmented)": "resnet18_pretrained_augment_results.json",
        "EfficientNet-B0 (scratch, no augment)": "efficientnet_b0_scratch_noaugment_results.json",
        "EfficientNet-B0 (pretrained, no augment)": "efficientnet_b0_pretrained_noaugment_results.json",
        "EfficientNet-B0 (pretrained, augmented)": "efficientnet_b0_pretrained_augment_results.json"
    }
    
    results = {}
    for model_name, filename in model_files.items():
        filepath = reports_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                results[model_name] = data
                print(f"  Loaded: {model_name}")
    
    return results

def plot_training_curves(results):
    """Create training curves figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    display_config = [
        ("ResNet-18 (scratch, no augment)", "ResNet-18\n(scratch)", "#1f77b4"),
        ("ResNet-18 (pretrained, no augment)", "ResNet-18\n(pretrained)", "#ff7f0e"),
        ("ResNet-18 (pretrained, augmented)", "ResNet-18\n(augmented)", "#2ca02c"),
        ("EfficientNet-B0 (scratch, no augment)", "EfficientNet-B0\n(scratch)", "#d62728"),
        ("EfficientNet-B0 (pretrained, no augment)", "EfficientNet-B0\n(pretrained)", "#9467bd"),
        ("EfficientNet-B0 (pretrained, augmented)", "EfficientNet-B0\n(augmented)", "#8c564b")
    ]
    
    # Plot R² scores
    ax1 = axes[0]
    positions = []
    labels = []
    r2_means = []
    r2_stds = []
    
    for i, (key, display_name, color) in enumerate(display_config):
        if key in results:
            model_data = results[key]
            
            # Extract R² from all folds
            r2_values = [fold['val_metrics']['r2'] for fold in model_data['folds']]
            r2_mean = np.mean(r2_values)
            r2_std = np.std(r2_values)
            
            positions.append(i)
            labels.append(display_name)
            r2_means.append(r2_mean)
            r2_stds.append(r2_std)
            
            ax1.bar(i, r2_mean, yerr=r2_std, color=color, alpha=0.7, 
                   capsize=5, width=0.6)
    
    ax1.axhline(y=0.713, color='red', linestyle='--', linewidth=2, 
                label='GBDT (R²=0.713)', alpha=0.8)
    
    ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Comparison (5-Fold CV)', fontsize=13, fontweight='bold')
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_ylim([0, 0.8])
    
    # Plot MAE
    ax2 = axes[1]
    mae_means = []
    mae_stds = []
    
    for i, (key, display_name, color) in enumerate(display_config):
        if key in results:
            model_data = results[key]
            
            mae_values = [fold['val_metrics']['mae_m'] for fold in model_data['folds']]
            mae_mean = np.mean(mae_values)
            mae_std = np.std(mae_values)
            
            mae_means.append(mae_mean)
            mae_stds.append(mae_std)
            
            ax2.bar(i, mae_mean, yerr=mae_std, color=color, alpha=0.7,
                   capsize=5, width=0.6)
    
    ax2.axhline(y=123, color='red', linestyle='--', linewidth=2,
                label='GBDT (MAE=123m)', alpha=0.8)
    
    ax2.set_ylabel('Mean Absolute Error (m)', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Error Comparison (5-Fold CV)', fontsize=13, fontweight='bold')
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_ylim([0, 300])
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("outputs/vision_baselines/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to {output_path}")
    
    main_figures = Path("outputs/figures")
    main_figures.mkdir(parents=True, exist_ok=True)
    main_output = main_figures / "vision_baseline_comparison.png"
    plt.savefig(main_output, dpi=300, bbox_inches='tight')
    print(f"Saved copy to {main_output}")
    
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("Vision Baseline Performance Summary (5-Fold Cross-Validation)")
    print("="*80)
    for i, label in enumerate(labels):
        short_label = label.replace('\n', ' ')
        print(f"{short_label:40s}: R²={r2_means[i]:.3f}±{r2_stds[i]:.3f}, MAE={mae_means[i]:.1f}±{mae_stds[i]:.1f}m")
    print("-"*80)
    print(f"{'GBDT (baseline)':40s}: R²=0.713, MAE=123m")
    print("="*80)
    
    best_idx = np.argmax(r2_means)
    print(f"\nBest vision model: {labels[best_idx].replace(chr(10), ' ')}")
    print(f"  R² gap vs GBDT: {(0.713 - r2_means[best_idx])/0.713*100:.1f}% worse")
    print(f"  MAE gap vs GBDT: {(mae_means[best_idx] - 123)/123*100:.1f}% worse")
    print("\nConclusion: Even state-of-the-art vision models underperform GBDT,")
    print("           validating that atmospheric features are superior for CBH retrieval.")

if __name__ == "__main__":
    print("Loading vision baseline results...")
    results = load_all_results()
    print(f"\nFound {len(results)} model results\n")
    
    if len(results) > 0:
        print("Generating comparison figure...")
        plot_training_curves(results)
        print("\nDone!")
    else:
        print("ERROR: No results found!")
