#!/usr/bin/env python3
"""
Generate training curves figure from vision baseline training logs.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results():
    """Load all vision baseline results."""
    reports_dir = Path("outputs/vision_baselines/reports")
    results = {}
    
    # Load each model's results
    models = [
        "resnet18_scratch_noaugment",
        "resnet18_pretrained_noaugment", 
        "resnet18_pretrained_augment",
        "efficientnet_b0_scratch_noaugment",
        "efficientnet_b0_pretrained_noaugment",
        "efficientnet_b0_pretrained_augment"
    ]
    
    for model in models:
        filepath = reports_dir / f"{model}_results.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Get the actual model name (first key)
                model_name = list(data.keys())[0]
                results[model_name] = data[model_name]
    
    return results

def plot_training_curves(results):
    """Create training curves figure."""
    # Set up figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Model display names and colors
    display_names = {
        "ResNet-18 (scratch, no augment)": ("ResNet-18 (scratch)", "#1f77b4"),
        "ResNet-18 (pretrained, no augment)": ("ResNet-18 (pretrained)", "#ff7f0e"),
        "ResNet-18 (pretrained, augmented)": ("ResNet-18 (augmented)", "#2ca02c"),
        "EfficientNet-B0 (scratch, no augment)": ("EfficientNet-B0 (scratch)", "#d62728"),
        "EfficientNet-B0 (pretrained, no augment)": ("EfficientNet-B0 (pretrained)", "#9467bd"),
        "EfficientNet-B0 (pretrained, augmented)": ("EfficientNet-B0 (augmented)", "#8c564b")
    }
    
    # Plot R² scores
    ax1 = axes[0]
    positions = []
    labels = []
    r2_means = []
    r2_stds = []
    
    for i, (model_name, model_data) in enumerate(results.items()):
        if model_name in display_names:
            display_name, color = display_names[model_name]
            
            # Extract R² from all folds
            r2_values = [fold['val_metrics']['r2'] for fold in model_data['folds']]
            r2_mean = np.mean(r2_values)
            r2_std = np.std(r2_values)
            
            positions.append(i)
            labels.append(display_name)
            r2_means.append(r2_mean)
            r2_stds.append(r2_std)
            
            # Bar plot with error bars
            ax1.bar(i, r2_mean, yerr=r2_std, color=color, alpha=0.7, 
                   capsize=5, width=0.6)
    
    # Add GBDT baseline reference line
    ax1.axhline(y=0.713, color='red', linestyle='--', linewidth=2, 
                label='GBDT (baseline)', alpha=0.8)
    
    ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(loc='lower right')
    ax1.set_ylim([0, 0.8])
    
    # Plot MAE
    ax2 = axes[1]
    mae_means = []
    mae_stds = []
    
    for i, (model_name, model_data) in enumerate(results.items()):
        if model_name in display_names:
            display_name, color = display_names[model_name]
            
            # Extract MAE from all folds
            mae_values = [fold['val_metrics']['mae_m'] for fold in model_data['folds']]
            mae_mean = np.mean(mae_values)
            mae_std = np.std(mae_values)
            
            mae_means.append(mae_mean)
            mae_stds.append(mae_std)
            
            # Bar plot with error bars
            ax2.bar(i, mae_mean, yerr=mae_std, color=color, alpha=0.7,
                   capsize=5, width=0.6)
    
    # Add GBDT baseline reference line (assuming ~123m from completion report)
    ax2.axhline(y=123, color='red', linestyle='--', linewidth=2,
                label='GBDT (baseline)', alpha=0.8)
    
    ax2.set_ylabel('Mean Absolute Error (m)', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Error Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_ylim([0, 300])
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("outputs/vision_baselines/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {output_path}")
    
    # Also save to main figures directory for manuscript
    main_figures = Path("outputs/figures")
    main_figures.mkdir(parents=True, exist_ok=True)
    main_output = main_figures / "vision_baseline_comparison.png"
    plt.savefig(main_output, dpi=300, bbox_inches='tight')
    print(f"Saved copy to {main_output}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("Vision Baseline Performance Summary")
    print("="*70)
    for i, label in enumerate(labels):
        print(f"{label:35s}: R²={r2_means[i]:.3f}±{r2_stds[i]:.3f}, MAE={mae_means[i]:.1f}±{mae_stds[i]:.1f}m")
    print(f"{'GBDT (baseline)':35s}: R²=0.713, MAE=123m")
    print("="*70)

if __name__ == "__main__":
    print("Loading vision baseline results...")
    results = load_results()
    print(f"Found {len(results)} model results")
    
    print("\nGenerating training curves figure...")
    plot_training_curves(results)
    
    print("\nDone!")
