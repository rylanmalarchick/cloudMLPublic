#!/usr/bin/env python3
"""
Angle-CBH Correlation Visualization
====================================

This script visualizes the relationship between solar angles (SZA, SAA) and
Cloud Base Height (CBH) across all flights to understand:

1. How solar angles correlate with CBH
2. Whether this correlation is consistent across flights
3. If there are diurnal patterns in cloud base height
4. Feature importance for angle-based predictions

This helps validate whether using solar angles as features is appropriate
and explains why "Angles_only" performs well in ablation studies.

Author: CloudML Analysis Suite
Date: 2024
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from scipy import stats

# Import project modules
from src.hdf5_dataset import HDF5CloudDataset


class AngleCBHVisualizer:
    """
    Visualize and analyze angle-CBH correlations.
    """

    def __init__(self, config_path):
        """
        Initialize visualizer.

        Args:
            config_path: Path to SSL fine-tuning config
        """
        self.config = self.load_config(config_path)

        print("=" * 80)
        print("ANGLE-CBH CORRELATION ANALYSIS")
        print("=" * 80)
        print(f"Config: {config_path}")
        print()

        # Create output directory
        self.output_dir = Path("outputs/angle_cbh_analysis")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / timestamp
        self.run_dir.mkdir(exist_ok=True, parents=True)

        print(f"Output directory: {self.run_dir}")
        print()

    def load_config(self, config_path):
        """Load YAML configuration."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_data(self):
        """
        Load all flight data with unified scaler.

        Returns:
            data: Dict with per-flight and combined data
        """
        print("Loading flight data...")

        # Load combined dataset (unified scaler)
        combined_dataset = HDF5CloudDataset(
            flight_configs=self.config["data"]["flights"],
            swath_slice=self.config["data"]["swath_slice"],
            temporal_frames=self.config["data"]["temporal_frames"],
            filter_type=self.config["data"]["filter_type"],
            cbh_min=self.config["data"]["cbh_min"],
            cbh_max=self.config["data"]["cbh_max"],
            flat_field_correction=self.config["data"]["flat_field_correction"],
            clahe_clip_limit=self.config["data"]["clahe_clip_limit"],
            zscore_normalize=self.config["data"]["zscore_normalize"],
            angles_mode=self.config["data"]["angles_mode"],
            augment=False,
        )

        print(f"Total samples: {len(combined_dataset)}")

        # Extract unscaled data
        print("Extracting data from dataset...")

        sza_list = []
        saa_list = []
        cbh_list = []
        flight_list = []

        # Extract only CPL-aligned samples (using the dataset's indices)
        for idx in range(len(combined_dataset)):
            global_idx = int(combined_dataset.indices[idx])
            flight_idx, local_idx = combined_dataset.global_to_local[global_idx]
            flight_info = combined_dataset.flight_data[flight_idx]
            flight_name = combined_dataset.flight_configs[flight_idx]["name"]

            # Get raw angles and CBH for this specific sample
            sza = flight_info["SZA_full"][local_idx, 0]
            saa = flight_info["SAA_full"][local_idx, 0]
            cbh = flight_info["Y_full"][local_idx].flatten()[0]

            sza_list.append(sza)
            saa_list.append(saa)
            cbh_list.append(cbh)
            flight_list.append(flight_name)

        # Count samples per flight
        flight_counts = pd.Series(flight_list).value_counts().sort_index()
        for flight_name, count in flight_counts.items():
            print(f"  {flight_name}: {count} samples")

        # Create DataFrame
        df = pd.DataFrame(
            {
                "SZA": sza_list,
                "SAA": saa_list,
                "CBH": cbh_list,
                "Flight": flight_list,
            }
        )

        print(f"\n Extracted {len(df)} samples total")
        print()

        return df

    def compute_correlations(self, df):
        """
        Compute correlation statistics.

        Args:
            df: DataFrame with SZA, SAA, CBH, Flight columns

        Returns:
            correlations: Dict with correlation results
        """
        print("=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)
        print()

        correlations = {}

        # Overall correlations
        overall_sza_corr = df["SZA"].corr(df["CBH"])
        overall_saa_corr = df["SAA"].corr(df["CBH"])

        print("Overall Correlations:")
        print(f"  SZA vs CBH: r = {overall_sza_corr:.4f}")
        print(f"  SAA vs CBH: r = {overall_saa_corr:.4f}")
        print()

        correlations["overall"] = {
            "sza_cbh": overall_sza_corr,
            "saa_cbh": overall_saa_corr,
        }

        # Per-flight correlations
        print("Per-Flight Correlations:")
        per_flight = {}

        for flight in df["Flight"].unique():
            flight_df = df[df["Flight"] == flight]

            sza_corr = flight_df["SZA"].corr(flight_df["CBH"])
            saa_corr = flight_df["SAA"].corr(flight_df["CBH"])

            print(f"  {flight}:")
            print(f"    SZA vs CBH: r = {sza_corr:.4f}")
            print(f"    SAA vs CBH: r = {saa_corr:.4f}")
            print(f"    N = {len(flight_df)}")

            per_flight[flight] = {
                "sza_cbh": sza_corr,
                "saa_cbh": saa_corr,
                "n_samples": len(flight_df),
            }

        correlations["per_flight"] = per_flight
        print()

        # Statistical tests
        print("Statistical Significance Tests:")

        # Pearson correlation test for SZA (handle NaN)
        try:
            sza_r, sza_p = stats.pearsonr(df["SZA"], df["CBH"])
            print(f"  SZA vs CBH: r = {sza_r:.4f}, p = {sza_p:.4e}")
        except:
            sza_r, sza_p = np.nan, np.nan
            print(f"  SZA vs CBH: r = nan (likely constant SZA)")

        # Pearson correlation test for SAA
        try:
            saa_r, saa_p = stats.pearsonr(df["SAA"], df["CBH"])
            print(f"  SAA vs CBH: r = {saa_r:.4f}, p = {saa_p:.4e}")
        except:
            saa_r, saa_p = np.nan, np.nan
            print(f"  SAA vs CBH: r = nan")
        print()

        correlations["significance"] = {
            "sza": {
                "r": float(sza_r) if not np.isnan(sza_r) else None,
                "p": float(sza_p) if not np.isnan(sza_p) else None,
            },
            "saa": {
                "r": float(saa_r) if not np.isnan(saa_r) else None,
                "p": float(saa_p) if not np.isnan(saa_p) else None,
            },
        }

        return correlations

    def plot_comprehensive_analysis(self, df, correlations):
        """
        Create comprehensive visualization of angle-CBH relationships.

        Args:
            df: DataFrame with data
            correlations: Correlation statistics
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        fig.suptitle(
            "Solar Angle - Cloud Base Height Correlation Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # 1. SZA vs CBH scatter (all flights)
        ax1 = fig.add_subplot(gs[0, 0])
        flights = df["Flight"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(flights)))

        for i, flight in enumerate(flights):
            flight_df = df[df["Flight"] == flight]
            ax1.scatter(
                flight_df["SZA"],
                flight_df["CBH"],
                alpha=0.6,
                s=20,
                label=flight,
                color=colors[i],
            )

        ax1.set_xlabel("Solar Zenith Angle (degrees)")
        ax1.set_ylabel("Cloud Base Height (km)")
        ax1.set_title(f"SZA vs CBH (r={correlations['overall']['sza_cbh']:.3f})")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(alpha=0.3)

        # 2. SAA vs CBH scatter (all flights)
        ax2 = fig.add_subplot(gs[0, 1])

        for i, flight in enumerate(flights):
            flight_df = df[df["Flight"] == flight]
            ax2.scatter(
                flight_df["SAA"],
                flight_df["CBH"],
                alpha=0.6,
                s=20,
                label=flight,
                color=colors[i],
            )

        ax2.set_xlabel("Solar Azimuth Angle (degrees)")
        ax2.set_ylabel("Cloud Base Height (km)")
        ax2.set_title(f"SAA vs CBH (r={correlations['overall']['saa_cbh']:.3f})")
        ax2.legend(loc="best", fontsize=8)
        ax2.grid(alpha=0.3)

        # 3. Per-flight correlation heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        per_flight = correlations["per_flight"]
        flight_names = list(per_flight.keys())
        sza_corrs = [per_flight[f]["sza_cbh"] for f in flight_names]
        saa_corrs = [per_flight[f]["saa_cbh"] for f in flight_names]

        corr_matrix = np.array([sza_corrs, saa_corrs]).T
        im = ax3.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(["SZA", "SAA"])
        ax3.set_yticks(range(len(flight_names)))
        ax3.set_yticklabels(flight_names, fontsize=9)
        ax3.set_title("Per-Flight Correlations")
        plt.colorbar(im, ax=ax3, label="Correlation (r)")

        # Add correlation values as text
        for i in range(len(flight_names)):
            for j in range(2):
                text = ax3.text(
                    j,
                    i,
                    f"{corr_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                    fontsize=9,
                )

        # 4. CBH distribution by flight
        ax4 = fig.add_subplot(gs[1, 0])
        df.boxplot(column="CBH", by="Flight", ax=ax4)
        ax4.set_xlabel("Flight")
        ax4.set_ylabel("Cloud Base Height (km)")
        ax4.set_title("CBH Distribution by Flight")
        plt.suptitle("")  # Remove automatic title

        # 5. SZA distribution by flight
        ax5 = fig.add_subplot(gs[1, 1])
        df.boxplot(column="SZA", by="Flight", ax=ax5)
        ax5.set_xlabel("Flight")
        ax5.set_ylabel("Solar Zenith Angle (degrees)")
        ax5.set_title("SZA Distribution by Flight")
        plt.suptitle("")

        # 6. SAA distribution by flight
        ax6 = fig.add_subplot(gs[1, 2])
        df.boxplot(column="SAA", by="Flight", ax=ax6)
        ax6.set_xlabel("Flight")
        ax6.set_ylabel("Solar Azimuth Angle (degrees)")
        ax6.set_title("SAA Distribution by Flight")
        plt.suptitle("")

        # 7. Joint distribution (SZA vs SAA colored by CBH)
        ax7 = fig.add_subplot(gs[2, 0])
        scatter = ax7.scatter(
            df["SZA"], df["SAA"], c=df["CBH"], cmap="viridis", alpha=0.6, s=20
        )
        ax7.set_xlabel("Solar Zenith Angle (degrees)")
        ax7.set_ylabel("Solar Azimuth Angle (degrees)")
        ax7.set_title("Angle Distribution (colored by CBH)")
        plt.colorbar(scatter, ax=ax7, label="CBH (km)")
        ax7.grid(alpha=0.3)

        # 8. Correlation bar chart
        ax8 = fig.add_subplot(gs[2, 1])
        x = np.arange(len(flight_names))
        width = 0.35
        ax8.bar(x - width / 2, sza_corrs, width, label="SZA", alpha=0.8)
        ax8.bar(x + width / 2, saa_corrs, width, label="SAA", alpha=0.8)
        ax8.set_xticks(x)
        ax8.set_xticklabels(flight_names, rotation=45, ha="right")
        ax8.set_ylabel("Correlation with CBH")
        ax8.set_title("Per-Flight Angle-CBH Correlations")
        ax8.legend()
        ax8.axhline(0, color="black", linestyle="--", alpha=0.3)
        ax8.grid(alpha=0.3, axis="y")

        # 9. Summary statistics
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis("off")

        summary_text = (
            "SUMMARY STATISTICS\n"
            "=" * 40 + "\n\n"
            f"Total Samples: {len(df)}\n"
            f"Flights: {len(flights)}\n\n"
            "Overall Correlations:\n"
            f"  SZA-CBH: r = {correlations['overall']['sza_cbh']:.4f}\n"
            f"  SAA-CBH: r = {correlations['overall']['saa_cbh']:.4f}\n\n"
            "Significance:\n"
            f"  SZA: p = {correlations['significance']['sza']['p']:.2e}\n"
            f"  SAA: p = {correlations['significance']['saa']['p']:.2e}\n\n"
            "Data Ranges:\n"
            f"  SZA: {df['SZA'].min():.1f}° - {df['SZA'].max():.1f}°\n"
            f"  SAA: {df['SAA'].min():.1f}° - {df['SAA'].max():.1f}°\n"
            f"  CBH: {df['CBH'].min():.2f} - {df['CBH'].max():.2f} km\n\n"
            "Interpretation:\n"
        )

        # Add interpretation
        overall_corr = max(
            abs(correlations["overall"]["sza_cbh"]),
            abs(correlations["overall"]["saa_cbh"]),
        )

        if overall_corr > 0.7:
            summary_text += "  Strong correlation - angles\n"
            summary_text += "  are highly predictive of CBH.\n"
        elif overall_corr > 0.4:
            summary_text += "  Moderate correlation - angles\n"
            summary_text += "  provide useful information.\n"
        else:
            summary_text += "  Weak correlation - angles\n"
            summary_text += "  have limited predictive power.\n"

        ax9.text(
            0.05,
            0.95,
            summary_text,
            transform=ax9.transAxes,
            fontsize=10,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
        )

        # Save figure
        save_path = self.run_dir / "angle_cbh_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f" Plot saved to {save_path}")

        plt.close()

    def save_results(self, df, correlations):
        """
        Save analysis results to files.

        Args:
            df: DataFrame with data
            correlations: Correlation statistics
        """
        # Save raw data
        csv_path = self.run_dir / "angle_cbh_data.csv"
        df.to_csv(csv_path, index=False)
        print(f" Data saved to {csv_path}")

        # Save correlation results (handle NaN values)
        import json

        def convert_to_json_safe(obj):
            """Convert numpy types and NaN to JSON-safe types."""
            if isinstance(obj, dict):
                return {k: convert_to_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_safe(item) for item in obj]
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj) if not np.isnan(obj) else None
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            return obj

        corr_path = self.run_dir / "correlations.json"
        with open(corr_path, "w") as f:
            json.dump(convert_to_json_safe(correlations), f, indent=2)
        print(f" Correlations saved to {corr_path}")

        # Save summary statistics
        summary_path = self.run_dir / "summary_statistics.txt"
        with open(summary_path, "w") as f:
            f.write("ANGLE-CBH CORRELATION ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            f.write("Overall Statistics:\n")
            f.write(f"  Total samples: {len(df)}\n")
            f.write(f"  Number of flights: {df['Flight'].nunique()}\n\n")

            f.write("Angle Ranges:\n")
            f.write(f"  SZA: {df['SZA'].min():.2f}° to {df['SZA'].max():.2f}°\n")
            f.write(f"  SAA: {df['SAA'].min():.2f}° to {df['SAA'].max():.2f}°\n\n")

            f.write("CBH Statistics:\n")
            f.write(f"  Mean: {df['CBH'].mean():.3f} km\n")
            f.write(f"  Std:  {df['CBH'].std():.3f} km\n")
            f.write(f"  Min:  {df['CBH'].min():.3f} km\n")
            f.write(f"  Max:  {df['CBH'].max():.3f} km\n\n")

            f.write("Overall Correlations:\n")
            sza_p = correlations["significance"]["sza"]["p"]
            saa_p = correlations["significance"]["saa"]["p"]
            sza_p_str = f"{sza_p:.2e}" if sza_p is not None else "nan"
            saa_p_str = f"{saa_p:.2e}" if saa_p is not None else "nan"
            f.write(
                f"  SZA vs CBH: r = {correlations['overall']['sza_cbh']:.4f}, "
                f"p = {sza_p_str}\n"
            )
            f.write(
                f"  SAA vs CBH: r = {correlations['overall']['saa_cbh']:.4f}, "
                f"p = {saa_p_str}\n\n"
            )

            f.write("Per-Flight Correlations:\n")
            for flight, stats in correlations["per_flight"].items():
                f.write(f"  {flight}:\n")
                f.write(f"    SZA-CBH: r = {stats['sza_cbh']:.4f}\n")
                f.write(f"    SAA-CBH: r = {stats['saa_cbh']:.4f}\n")
                f.write(f"    N = {stats['n_samples']}\n")

        print(f" Summary saved to {summary_path}")

    def run(self):
        """Run complete analysis pipeline."""
        # Load data
        df = self.load_data()

        # Compute correlations
        correlations = self.compute_correlations(df)

        # Create visualizations
        print("Creating visualizations...")
        self.plot_comprehensive_analysis(df, correlations)

        # Save results
        print("\nSaving results...")
        self.save_results(df, correlations)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {self.run_dir}")
        print()

        return df, correlations


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze angle-CBH correlations across flights"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ssl_finetune_cbh.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    # Run analysis
    visualizer = AngleCBHVisualizer(config_path=args.config)
    df, correlations = visualizer.run()


if __name__ == "__main__":
    main()
