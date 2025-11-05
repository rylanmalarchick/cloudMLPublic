#!/usr/bin/env python3
"""
WP-1 Diagnostic Tool

Visualizes shadow detection results to diagnose why geometric CBH is failing.

This script loads sample images and visualizes:
1. Original image
2. Detected cloud regions (bright)
3. Detected shadow regions (dark)
4. Expected shadow direction based on solar azimuth
5. Intensity histograms and thresholds
6. Geometric calculations

Usage:
    python sow_outputs/diagnose_wp1.py --config configs/bestComboConfig.yaml --samples 0 50 100 200 400

Author: Autonomous Agent
Date: 2025-06-04
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from scipy import ndimage
from skimage import measure
import argparse

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hdf5_dataset import HDF5CloudDataset


def diagnose_sample(dataset, idx, scale=50.0, save_dir="sow_outputs/wp1_diagnostics"):
    """
    Diagnose a single sample - visualize shadow detection and geometric calculation.

    Args:
        dataset: HDF5CloudDataset instance
        idx: Sample index to diagnose
        scale: Meters per pixel scale factor
        save_dir: Directory to save diagnostic plots
    """
    # Create output directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Get sample data
    img_tensor, sza, saa, cbh_true, global_idx, local_idx = dataset[idx]

    # Convert to numpy
    img = img_tensor[1].numpy()  # Central frame
    sza_deg = sza.item() if hasattr(sza, "item") else float(sza)
    saa_deg = saa.item() if hasattr(saa, "item") else float(saa)
    cbh_true_km = cbh_true.item() if hasattr(cbh_true, "item") else float(cbh_true)

    # Normalize image to [0, 1]
    img_min, img_max = img.min(), img.max()
    img = (img - img_min) / (img_max - img_min + 1e-6)

    # Simple cloud detection (bright regions above threshold)
    cloud_thresh = img.mean() + 0.5 * img.std()
    cloud_mask = (img > cloud_thresh).astype(np.uint8)

    # Clean up cloud mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, kernel)

    # Simple shadow detection (dark regions below threshold)
    shadow_thresh = img.mean() - 0.5 * img.std()
    shadow_mask = (img < shadow_thresh).astype(np.uint8)

    # Clean up shadow mask
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel_small)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel_small)

    # Expected shadow direction (AWAY from sun, so SAA + 180 degrees)
    shadow_dir_deg = (saa_deg + 180) % 360
    shadow_dir_rad = np.deg2rad(shadow_dir_deg)

    # Direction vector (image coordinates: x=col (east), y=row (south, positive down))
    dx = np.sin(shadow_dir_rad)  # East component
    dy = np.cos(shadow_dir_rad)  # South component (positive y is down in image)

    # Label connected components
    cloud_labels = measure.label(cloud_mask, connectivity=2)
    shadow_labels = measure.label(shadow_mask, connectivity=2)

    cloud_props = measure.regionprops(cloud_labels)
    shadow_props = measure.regionprops(shadow_labels)

    # Find cloud-shadow pairs
    pairs = []
    for cloud_region in cloud_props:
        if cloud_region.area < 50:
            continue

        cloud_centroid = cloud_region.centroid  # (row, col)

        for shadow_region in shadow_props:
            if shadow_region.area < 30:
                continue

            shadow_centroid = shadow_region.centroid

            # Vector from cloud to shadow
            cloud_to_shadow = np.array(
                [
                    shadow_centroid[0] - cloud_centroid[0],  # dy (rows)
                    shadow_centroid[1] - cloud_centroid[1],  # dx (cols)
                ]
            )

            distance_pixels = np.linalg.norm(cloud_to_shadow)

            if distance_pixels < 5 or distance_pixels > 200:
                continue

            # Expected direction
            expected_dir = np.array([dy, dx])
            actual_dir = cloud_to_shadow / (distance_pixels + 1e-6)

            # Cosine similarity (alignment)
            alignment = np.dot(expected_dir, actual_dir)

            if alignment > 0.3:
                pairs.append(
                    {
                        "cloud": cloud_region,
                        "shadow": shadow_region,
                        "distance_pixels": distance_pixels,
                        "alignment": alignment,
                        "cloud_centroid": cloud_centroid,
                        "shadow_centroid": shadow_centroid,
                    }
                )

    # Sort pairs by alignment
    pairs = sorted(pairs, key=lambda p: p["alignment"], reverse=True)

    # Compute geometric CBH for best pair (if any)
    derived_cbh_km = np.nan
    shadow_length_pixels = np.nan

    if pairs:
        best_pair = pairs[0]
        shadow_length_pixels = best_pair["distance_pixels"]
        shadow_length_m = shadow_length_pixels * scale

        # Solar elevation angle
        solar_elevation_deg = 90.0 - sza_deg
        solar_elevation_rad = np.deg2rad(solar_elevation_deg)

        if solar_elevation_deg > 5.0:
            # H = L * tan(elevation)
            cbh_m = shadow_length_m * np.tan(solar_elevation_rad)
            derived_cbh_km = cbh_m / 1000.0
        else:
            derived_cbh_km = np.nan  # Sun too low

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Row 1: Original, Cloud mask, Shadow mask, Combined
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax1.set_title(f"Original Image\nSample {idx}", fontsize=12, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img, cmap="gray", alpha=0.5)
    ax2.imshow(cloud_mask, cmap="Reds", alpha=0.5)
    ax2.set_title(
        f"Cloud Detection\n{cloud_mask.sum()} pixels ({len(cloud_props)} regions)",
        fontsize=12,
    )
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img, cmap="gray", alpha=0.5)
    ax3.imshow(shadow_mask, cmap="Blues", alpha=0.5)
    ax3.set_title(
        f"Shadow Detection\n{shadow_mask.sum()} pixels ({len(shadow_props)} regions)",
        fontsize=12,
    )
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(img, cmap="gray")
    ax4.imshow(cloud_mask, cmap="Reds", alpha=0.3)
    ax4.imshow(shadow_mask, cmap="Blues", alpha=0.3)

    # Draw expected shadow direction arrow
    h, w = img.shape
    center_y, center_x = h // 2, w // 2
    arrow_len = 100
    ax4.arrow(
        center_x,
        center_y,
        dx * arrow_len,
        dy * arrow_len,
        color="yellow",
        width=3,
        head_width=15,
        head_length=10,
        label="Expected Shadow Dir",
        zorder=10,
    )
    ax4.set_title(
        f"Combined + Shadow Direction\nSAA={saa_deg:.1f}°, Shadow Dir={shadow_dir_deg:.1f}°",
        fontsize=12,
    )
    ax4.legend(loc="upper right", fontsize=9)
    ax4.axis("off")

    # Row 2: Cloud-shadow pairs, Geometry diagram, Histogram
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(img, cmap="gray")

    # Draw all detected pairs
    for i, pair in enumerate(pairs[:5]):  # Top 5 pairs
        cloud_y, cloud_x = pair["cloud_centroid"]
        shadow_y, shadow_x = pair["shadow_centroid"]

        color = ["lime", "cyan", "magenta", "orange", "white"][i]

        # Draw centroids
        ax5.plot(
            cloud_x, cloud_y, "o", color=color, markersize=8, label=f"Pair {i + 1}"
        )
        ax5.plot(shadow_x, shadow_y, "s", color=color, markersize=8)

        # Draw line connecting them
        ax5.plot(
            [cloud_x, shadow_x], [cloud_y, shadow_y], "--", color=color, linewidth=2
        )

        # Annotate with alignment score
        mid_x, mid_y = (cloud_x + shadow_x) / 2, (cloud_y + shadow_y) / 2
        ax5.text(
            mid_x,
            mid_y,
            f"{pair['alignment']:.2f}",
            color=color,
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        )

    ax5.set_title(f"Cloud-Shadow Pairs\n{len(pairs)} pairs detected", fontsize=12)
    if pairs:
        ax5.legend(loc="upper right", fontsize=8)
    ax5.axis("off")

    # Geometry diagram
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.set_xlim(-1, 6)
    ax6.set_ylim(-1, 5)
    ax6.set_aspect("equal")
    ax6.axis("off")

    # Draw geometry
    # Sun rays
    sun_x, sun_y = 1, 4
    ax6.scatter(
        sun_x,
        sun_y,
        s=300,
        c="yellow",
        marker="*",
        edgecolors="orange",
        linewidths=2,
        zorder=10,
    )
    ax6.text(sun_x, sun_y + 0.5, "Sun", ha="center", fontsize=10, fontweight="bold")

    # Cloud
    cloud_x, cloud_y = 2, 2.5
    circle = plt.Circle(
        (cloud_x, cloud_y), 0.3, color="white", ec="black", linewidth=2, zorder=5
    )
    ax6.add_patch(circle)
    ax6.text(cloud_x, cloud_y + 0.6, "Cloud", ha="center", fontsize=9)

    # Ground
    ax6.plot([0, 6], [0, 0], "k-", linewidth=3)
    ax6.text(3, -0.4, "Ground", ha="center", fontsize=9)

    # Shadow on ground
    if not np.isnan(shadow_length_pixels) and pairs:
        shadow_ground_x = cloud_x + (shadow_length_pixels / 100)  # Scale down for viz
        ax6.plot(
            [cloud_x, shadow_ground_x], [0, 0], "b-", linewidth=4, label="Shadow (L)"
        )
        ax6.scatter(shadow_ground_x, 0, s=100, c="blue", marker="s", zorder=5)

        # Height line
        ax6.plot(
            [cloud_x, cloud_x], [0, cloud_y], "r--", linewidth=2, label="Height (H)"
        )

        # Sun ray to cloud
        ax6.plot([sun_x, cloud_x], [sun_y, cloud_y], "y--", linewidth=1, alpha=0.7)

        # Sun ray to shadow
        ax6.plot([sun_x, shadow_ground_x], [sun_y, 0], "y--", linewidth=1, alpha=0.7)

        # Solar zenith angle arc
        from matplotlib.patches import Arc

        arc = Arc(
            (sun_x, sun_y),
            1.5,
            1.5,
            angle=0,
            theta1=270 - sza_deg,
            theta2=270,
            color="orange",
            linewidth=2,
        )
        ax6.add_patch(arc)
        ax6.text(
            sun_x - 0.5, sun_y - 0.7, f"SZA={sza_deg:.1f}°", fontsize=9, color="orange"
        )

        ax6.legend(loc="upper left", fontsize=8)

    ax6.set_title(
        "Geometric Formula\nH = L × tan(90° - SZA)", fontsize=12, fontweight="bold"
    )

    # Histogram
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.hist(img.flatten(), bins=60, alpha=0.7, color="gray", edgecolor="black")
    ax7.axvline(
        cloud_thresh,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Cloud thresh={cloud_thresh:.3f}",
    )
    ax7.axvline(
        shadow_thresh,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Shadow thresh={shadow_thresh:.3f}",
    )
    ax7.axvline(
        img.mean(),
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Mean={img.mean():.3f}",
    )
    ax7.set_title("Intensity Distribution", fontsize=12)
    ax7.set_xlabel("Normalized Intensity", fontsize=10)
    ax7.set_ylabel("Pixel Count", fontsize=10)
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # Metadata and calculations
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis("off")

    info_text = f"""SAMPLE METADATA
─────────────────────
Sample Index:     {idx}
Flight:           {dataset.flight_data[dataset.sample_to_flight[idx]]["name"]}
Local Index:      {local_idx}

SOLAR GEOMETRY
─────────────────────
Solar Zenith:     {sza_deg:.2f}°
Solar Azimuth:    {saa_deg:.2f}°
Solar Elevation:  {90 - sza_deg:.2f}°
Shadow Direction: {shadow_dir_deg:.2f}°

GROUND TRUTH
─────────────────────
True CBH:         {cbh_true_km:.3f} km

DETECTION RESULTS
─────────────────────
Cloud regions:    {len(cloud_props)}
Shadow regions:   {len(shadow_props)}
Valid pairs:      {len(pairs)}

GEOMETRIC CALCULATION
─────────────────────
Scale factor:     {scale} m/pixel
Shadow length:    {shadow_length_pixels:.1f} pixels
                  ({shadow_length_pixels * scale:.1f} m)

Derived CBH:      {derived_cbh_km:.3f} km

ERROR ANALYSIS
─────────────────────
CBH Error:        {derived_cbh_km - cbh_true_km:.3f} km
Relative Error:   {((derived_cbh_km - cbh_true_km) / cbh_true_km * 100) if not np.isnan(derived_cbh_km) else np.nan:.1f}%

DIAGNOSTIC FLAGS
─────────────────────
Valid detection:  {"YES" if pairs else "NO"}
Sun elevation OK: {"YES" if (90 - sza_deg) > 5 else "NO (too low)"}
CBH reasonable:   {"YES" if 0 < derived_cbh_km < 10 else "NO" if not np.isnan(derived_cbh_km) else "N/A"}
"""

    ax8.text(
        0.05,
        0.95,
        info_text,
        transform=ax8.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Row 3: Detailed pair analysis
    ax9 = fig.add_subplot(gs[2, :])
    ax9.axis("off")

    if pairs:
        pair_details = "TOP 5 CLOUD-SHADOW PAIRS:\n"
        pair_details += "─" * 120 + "\n"
        pair_details += f"{'#':<3} {'Cloud Area':<12} {'Shadow Area':<12} {'Distance (px)':<15} {'Distance (m)':<15} {'Alignment':<10} {'Derived H (km)':<15}\n"
        pair_details += "─" * 120 + "\n"

        for i, pair in enumerate(pairs[:5]):
            dist_px = pair["distance_pixels"]
            dist_m = dist_px * scale
            align = pair["alignment"]

            # Compute H for this pair
            solar_elev = 90.0 - sza_deg
            if solar_elev > 5:
                h_m = dist_m * np.tan(np.deg2rad(solar_elev))
                h_km = h_m / 1000.0
            else:
                h_km = np.nan

            pair_details += f"{i + 1:<3} {pair['cloud'].area:<12} {pair['shadow'].area:<12} {dist_px:<15.1f} {dist_m:<15.1f} {align:<10.3f} {h_km:<15.3f}\n"
    else:
        pair_details = "NO VALID CLOUD-SHADOW PAIRS DETECTED\n\n"
        pair_details += "Possible reasons:\n"
        pair_details += "  • No clear shadows (ocean surface, thin clouds, overcast)\n"
        pair_details += "  • Multi-layer clouds causing ambiguous shadows\n"
        pair_details += "  • Low sun angle (poor shadow contrast)\n"
        pair_details += "  • Detection thresholds need tuning\n"

    ax9.text(
        0.02,
        0.95,
        pair_details,
        transform=ax9.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    # Overall title
    fig.suptitle(
        f"WP-1 Geometric Feature Diagnostic: Sample {idx}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Save figure
    output_file = Path(save_dir) / f"wp1_diagnostic_sample_{idx:04d}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved diagnostic for sample {idx} -> {output_file}")
    print(
        f"    True CBH: {cbh_true_km:.3f} km | Derived CBH: {derived_cbh_km:.3f} km | Error: {derived_cbh_km - cbh_true_km:.3f} km"
    )
    print(f"    SZA: {sza_deg:.1f}° | Pairs detected: {len(pairs)}")

    return {
        "sample_idx": idx,
        "true_cbh": cbh_true_km,
        "derived_cbh": derived_cbh_km,
        "error": derived_cbh_km - cbh_true_km,
        "n_pairs": len(pairs),
        "shadow_length_pixels": shadow_length_pixels,
    }


def main():
    parser = argparse.ArgumentParser(
        description="WP-1 Diagnostic Tool: Visualize shadow detection and geometric CBH calculation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bestComboConfig.yaml",
        help="Path to configuration YAML",
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[0, 50, 100, 200, 400, 600, 800],
        help="Sample indices to diagnose",
    )
    parser.add_argument(
        "--scale", type=float, default=50.0, help="Meters per pixel scale factor"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sow_outputs/wp1_diagnostics",
        help="Directory to save diagnostic plots",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("WP-1 GEOMETRIC FEATURE DIAGNOSTIC TOOL")
    print("=" * 80)
    print(f"Configuration:  {args.config}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Scale factor:   {args.scale} m/pixel")
    print(f"Samples:        {args.samples}")
    print("=" * 80)

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load dataset
    print("\nLoading dataset...")
    dataset = HDF5CloudDataset(
        flight_configs=config["flights"],
        indices=None,
        augment=False,
        swath_slice=config.get("swath_slice", [40, 480]),
    )

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"\nDiagnosing {len(args.samples)} samples...\n")

    results = []

    for idx in args.samples:
        if idx < len(dataset):
            try:
                result = diagnose_sample(
                    dataset, idx, scale=args.scale, save_dir=args.output_dir
                )
                results.append(result)
            except Exception as e:
                print(f"✗ Error processing sample {idx}: {e}")
        else:
            print(f"✗ Skipping idx={idx} (out of range, max={len(dataset) - 1})")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    if results:
        valid_results = [r for r in results if not np.isnan(r["derived_cbh"])]

        print(f"\nSamples processed:      {len(results)}")
        print(f"Valid detections:       {len(valid_results)}/{len(results)}")

        if valid_results:
            errors = [r["error"] for r in valid_results]
            print(f"\nCBH Error Statistics:")
            print(f"  Mean error:     {np.mean(errors):.3f} km")
            print(f"  Median error:   {np.median(errors):.3f} km")
            print(f"  Std deviation:  {np.std(errors):.3f} km")
            print(f"  Min error:      {np.min(errors):.3f} km")
            print(f"  Max error:      {np.max(errors):.3f} km")

            derived_cbhs = [r["derived_cbh"] for r in valid_results]
            print(f"\nDerived CBH Statistics:")
            print(f"  Mean:           {np.mean(derived_cbhs):.3f} km")
            print(f"  Median:         {np.median(derived_cbhs):.3f} km")
            print(f"  Min:            {np.min(derived_cbhs):.3f} km")
            print(f"  Max:            {np.max(derived_cbhs):.3f} km")

    print(f"\n✓ Diagnostics saved to: {args.output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
