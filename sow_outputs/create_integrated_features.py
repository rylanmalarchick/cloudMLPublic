#!/usr/bin/env python3
"""
Work Package 3: Integrated Feature Store

Creates a unified HDF5 file containing all features:
- WP1: Geometric features (shadow-derived CBH, solar angles)
- WP2: Atmospheric features (ERA5 reanalysis)
- Image features: Placeholder for CNN embeddings (future)
- Metadata: Sample IDs, flight IDs, timestamps, targets

This is a required deliverable (7.3a) for Sprint 3.

Author: Autonomous Agent
Date: 2025
SOW: SOW-AGENT-CBH-WP-001 Section 7.3a
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import json
from datetime import datetime
from typing import Dict


class IntegratedFeatureStoreBuilder:
    """
    Builds integrated feature store from WP1 and WP2 outputs.

    The integrated store provides a single unified interface for all features,
    making it easy to:
    1. Load consistent feature sets across experiments
    2. Track feature provenance and versions
    3. Support future feature additions (e.g., CNN embeddings)
    """

    def __init__(
        self,
        wp1_features_path: str,
        wp2_features_path: str,
        output_path: str = "sow_outputs/integrated_features/Integrated_Features.hdf5",
        verbose: bool = True,
    ):
        """
        Initialize the builder.

        Args:
            wp1_features_path: Path to WP1_Features.hdf5
            wp2_features_path: Path to WP2_Features.hdf5
            output_path: Path for output integrated HDF5
            verbose: Verbose logging
        """
        self.wp1_path = Path(wp1_features_path)
        self.wp2_path = Path(wp2_features_path)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def load_wp1_features(self) -> Dict:
        """Load WP1 geometric features."""
        if self.verbose:
            print("\n" + "=" * 80)
            print("Loading WP1 Features (Geometric)")
            print("=" * 80)

        with h5py.File(self.wp1_path, "r") as f:
            if self.verbose:
                print(f"Source: {self.wp1_path}")
                print(f"Keys: {list(f.keys())}")

            # Load all geometric features
            data = {
                "sample_id": f["sample_id"][:],
                "derived_geometric_H": f["derived_geometric_H"][:],
                "shadow_length_pixels": f["shadow_length_pixels"][:],
                "shadow_detection_confidence": f["shadow_detection_confidence"][:],
                "sza_deg": f["sza_deg"][:],
                "saa_deg": f["saa_deg"][:],
                "shadow_angle_deg": f["shadow_angle_deg"][:],
                "cloud_edge_x": f["cloud_edge_x"][:],
                "cloud_edge_y": f["cloud_edge_y"][:],
                "shadow_edge_x": f["shadow_edge_x"][:],
                "shadow_edge_y": f["shadow_edge_y"][:],
                "true_cbh_km": f["true_cbh_km"][:],
            }

            if self.verbose:
                print(f"Loaded {len(data)} geometric feature arrays")
                print(f"Samples: {len(data['sample_id'])}")

        return data

    def load_wp2_features(self) -> Dict:
        """Load WP2 atmospheric features."""
        if self.verbose:
            print("\n" + "=" * 80)
            print("Loading WP2 Features (Atmospheric)")
            print("=" * 80)

        with h5py.File(self.wp2_path, "r") as f:
            if self.verbose:
                print(f"Source: {self.wp2_path}")
                print(f"Keys: {list(f.keys())}")

            # Load ERA5 features
            features = f["features"][:]
            feature_names = [
                n.decode() if isinstance(n, bytes) else n for n in f["feature_names"][:]
            ]
            latitudes = f["latitudes"][:]
            longitudes = f["longitudes"][:]
            timestamps = f["timestamps"][:]
            sample_indices = f["sample_indices"][:]

            data = {
                "era5_features": features,
                "era5_feature_names": feature_names,
                "latitudes": latitudes,
                "longitudes": longitudes,
                "timestamps": timestamps,
                "sample_indices": sample_indices,
            }

            if self.verbose:
                print(f"ERA5 features: {features.shape}")
                print(f"Feature names: {feature_names}")

        return data

    def create_flight_ids(self, n_samples: int) -> np.ndarray:
        """
        Create flight ID array based on sample order.

        Based on known distribution: F0=501, F1=191, F2=105, F3=92, F4=44
        """
        flight_sizes = [501, 191, 105, 92, 44]
        flight_ids = np.zeros(n_samples, dtype=np.int32)

        idx = 0
        for fid, size in enumerate(flight_sizes):
            flight_ids[idx : idx + size] = fid
            idx += size

        return flight_ids

    def build_integrated_store(self):
        """Build the integrated feature store."""
        if self.verbose:
            print("\n" + "=" * 80)
            print("Building Integrated Feature Store")
            print("=" * 80)

        # Load WP1 and WP2 features
        wp1_data = self.load_wp1_features()
        wp2_data = self.load_wp2_features()

        n_samples = len(wp1_data["sample_id"])

        # Create flight IDs
        flight_ids = self.create_flight_ids(n_samples)

        if self.verbose:
            print("\n" + "-" * 80)
            print("Creating Integrated HDF5 File")
            print("-" * 80)
            print(f"Output: {self.output_path}")
            print(f"Total samples: {n_samples}")

        # Create integrated HDF5 file
        with h5py.File(self.output_path, "w") as f:
            # Metadata group
            meta_group = f.create_group("metadata")
            meta_group.create_dataset("sample_id", data=wp1_data["sample_id"])
            meta_group.create_dataset("flight_id", data=flight_ids)
            meta_group.create_dataset("latitude", data=wp2_data["latitudes"])
            meta_group.create_dataset("longitude", data=wp2_data["longitudes"])
            meta_group.create_dataset("timestamp", data=wp2_data["timestamps"])

            # Target variable
            meta_group.create_dataset("cbh_km", data=wp1_data["true_cbh_km"])

            # Geometric features group
            geo_group = f.create_group("geometric_features")
            geo_group.create_dataset(
                "derived_geometric_H", data=wp1_data["derived_geometric_H"]
            )
            geo_group.create_dataset(
                "shadow_length_pixels", data=wp1_data["shadow_length_pixels"]
            )
            geo_group.create_dataset(
                "shadow_detection_confidence",
                data=wp1_data["shadow_detection_confidence"],
            )
            geo_group.create_dataset("sza_deg", data=wp1_data["sza_deg"])
            geo_group.create_dataset("saa_deg", data=wp1_data["saa_deg"])
            geo_group.create_dataset(
                "shadow_angle_deg", data=wp1_data["shadow_angle_deg"]
            )
            geo_group.create_dataset("cloud_edge_x", data=wp1_data["cloud_edge_x"])
            geo_group.create_dataset("cloud_edge_y", data=wp1_data["cloud_edge_y"])
            geo_group.create_dataset("shadow_edge_x", data=wp1_data["shadow_edge_x"])
            geo_group.create_dataset("shadow_edge_y", data=wp1_data["shadow_edge_y"])

            # Atmospheric features group
            atmo_group = f.create_group("atmospheric_features")
            atmo_group.create_dataset("era5_features", data=wp2_data["era5_features"])
            atmo_group.create_dataset(
                "era5_feature_names",
                data=np.array(wp2_data["era5_feature_names"], dtype="S"),
            )

            # Image features group (placeholder for future)
            img_group = f.create_group("image_features")
            img_group.attrs["status"] = "placeholder"
            img_group.attrs["note"] = (
                "Future work: Add CNN embeddings from WP-4 image encoder"
            )

            # Add global attributes
            f.attrs["title"] = "Integrated Cloud Base Height Feature Store"
            f.attrs["created"] = datetime.now().isoformat()
            f.attrs["sow_deliverable"] = "7.3a"
            f.attrs["n_samples"] = n_samples
            f.attrs["n_flights"] = 5
            f.attrs["wp1_source"] = str(self.wp1_path)
            f.attrs["wp2_source"] = str(self.wp2_path)
            f.attrs["flight_mapping"] = json.dumps(
                {
                    "0": "30Oct24",
                    "1": "10Feb25",
                    "2": "23Oct24",
                    "3": "12Feb25",
                    "4": "18Feb25",
                }
            )

        if self.verbose:
            print(f"\n✓ Integrated feature store created: {self.output_path}")

    def print_summary(self):
        """Print summary of integrated feature store."""
        if not self.output_path.exists():
            print("ERROR: Feature store not found!")
            return

        print("\n" + "=" * 80)
        print("INTEGRATED FEATURE STORE SUMMARY")
        print("=" * 80)

        with h5py.File(self.output_path, "r") as f:
            print(f"\nFile: {self.output_path}")
            print(f"Created: {f.attrs['created']}")
            print(f"Total samples: {f.attrs['n_samples']}")
            print(f"Number of flights: {f.attrs['n_flights']}")

            print("\nTop-level groups:")
            for group_name in f.keys():
                group = f[group_name]
                print(f"  /{group_name}/")
                if isinstance(group, h5py.Group):
                    for dataset_name in group.keys():
                        dataset = group[dataset_name]
                        if isinstance(dataset, h5py.Dataset):
                            print(
                                f"    - {dataset_name}: {dataset.shape} ({dataset.dtype})"
                            )

            print("\nMetadata:")
            n_samples = f.attrs["n_samples"]
            cbh = f["metadata/cbh_km"][:]
            flight_ids = f["metadata/flight_id"][:]

            print(f"  CBH range: [{cbh.min():.3f}, {cbh.max():.3f}] km")
            print(f"  CBH mean: {cbh.mean():.3f} km")
            print(f"  CBH std: {cbh.std():.3f} km")

            print("\n  Flight distribution:")
            flight_map = json.loads(f.attrs["flight_mapping"])
            for fid in sorted(map(int, flight_map.keys())):
                count = np.sum(flight_ids == fid)
                fname = flight_map[str(fid)]
                print(f"    F{fid} ({fname}): {count} samples")

            print("\nGeometric features:")
            geo_group = f["geometric_features"]
            for key in geo_group.keys():
                dataset = geo_group[key]
                n_nan = np.sum(np.isnan(dataset[:]))
                print(
                    f"  - {key}: {dataset.shape}, {n_nan} NaNs ({100 * n_nan / len(dataset):.1f}%)"
                )

            print("\nAtmospheric features:")
            era5_features = f["atmospheric_features/era5_features"][:]
            era5_names = [
                n.decode() if isinstance(n, bytes) else n
                for n in f["atmospheric_features/era5_feature_names"][:]
            ]
            print(f"  ERA5 features: {era5_features.shape}")
            print(f"  Feature names: {era5_names}")

        print("\n" + "=" * 80)

    def run(self):
        """Execute complete feature store creation."""
        print("\n" + "=" * 80)
        print("WP-3: INTEGRATED FEATURE STORE CREATION")
        print("=" * 80)
        print(f"SOW Deliverable: 7.3a")

        # Build integrated store
        self.build_integrated_store()

        # Print summary
        self.print_summary()

        print("\n✓ Integrated feature store created successfully!")


def main():
    """Main entry point."""
    # Paths
    project_root = Path(__file__).parent.parent
    wp1_features = project_root / "sow_outputs" / "wp1_geometric" / "WP1_Features.hdf5"
    wp2_features = (
        project_root / "sow_outputs" / "wp2_atmospheric" / "WP2_Features.hdf5"
    )
    output_path = (
        project_root
        / "sow_outputs"
        / "integrated_features"
        / "Integrated_Features.hdf5"
    )

    # Check input paths
    if not wp1_features.exists():
        print(f"ERROR: WP1 features not found: {wp1_features}")
        return 1

    if not wp2_features.exists():
        print(f"ERROR: WP2 features not found: {wp2_features}")
        return 1

    # Build integrated store
    builder = IntegratedFeatureStoreBuilder(
        wp1_features_path=str(wp1_features),
        wp2_features_path=str(wp2_features),
        output_path=str(output_path),
        verbose=True,
    )

    builder.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
