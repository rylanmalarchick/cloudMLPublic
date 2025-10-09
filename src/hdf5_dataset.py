import gc

import h5py
import numpy as np
import torch
import torchvision.transforms as T
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .cplCompareSub import cplTimeConvert
from .data_preprocessing import linear_vignetting_correction, preprocess_image


class HDF5CloudDataset(Dataset):
    """
    Lazily loads HDF5 images + metadata, applies corrections, and handles
    data augmentation carefully by linking geometric transforms to solar angle data.
    Now includes advanced augmentations like Random Erasing ("chunking").
    """

    def __init__(
        self,
        flight_configs,
        indices=None,
        swath_slice=(40, 480),
        augment=True,
        temporal_frames=3,
        filter_type="basic",
        cbh_min=None,
        cbh_max=None,
        sza_scaler=None,
        saa_scaler=None,
        y_scaler=None,
        flat_field_correction=True,
        clahe_clip_limit=0.01,
        zscore_normalize=True,
        angles_mode: str = "both",  # NEW: control which angles are fed to the model
    ):
        self.flight_configs = flight_configs
        for i, config in enumerate(self.flight_configs):
            required_keys = ["iFileName", "cFileName", "nFileName"]
            if not all(key in config for key in required_keys):
                raise ValueError(f"Flight config {i} missing required keys.")
            if "name" not in config:
                config["name"] = f"flight_{i}"

        print(f"Initializing dataset with {len(self.flight_configs)} flight(s).")
        for config in self.flight_configs:
            print(f"  - {config['name']}")

        self.start, self.end = swath_slice
        self.augment = augment
        self.temporal_frames = temporal_frames
        self.temporal_offset = self.temporal_frames // 2

        self.filter_type = filter_type
        # new ablation flags
        self.flat_field_correction = flat_field_correction
        self.clahe_clip_limit = clahe_clip_limit
        self.zscore_normalize = zscore_normalize
        self.cbh_min, self.cbh_max = self._get_cbh_range(filter_type, cbh_min, cbh_max)
        print(f"Using CBH range: [{self.cbh_min:.2f} - {self.cbh_max:.2f}] km")

        # NEW: store angle mode
        valid_modes = {"both", "sza_only", "saa_only", "none"}
        if angles_mode not in valid_modes:
            raise ValueError(
                f"angles_mode must be one of {valid_modes}, got {angles_mode}"
            )
        self.angles_mode = angles_mode
        print(f"Angles mode: {self.angles_mode}")

        # Define "safe" augmentations that don't affect geometry
        if self.augment:
            # Transforms that work on PIL Images
            self.pil_transforms = T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1
            )
            # Transforms that work on PyTorch Tensors
            self.tensor_transforms = T.RandomErasing(
                p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0
            )
            print(
                "Augmentation enabled (Color Jitter, Random Erasing, Horizontal Flips)."
            )
        else:
            self.pil_transforms = None
            self.tensor_transforms = None
            print("Augmentation disabled.")

        print("Phase 1: Loading metadata from all flights...")
        self._load_all_flights_metadata()

        self.sza_scaler_pre = sza_scaler
        self.saa_scaler_pre = saa_scaler
        self.y_scaler_pre = y_scaler

        print("Phase 2: Fitting unified scalers...")
        self._fit_unified_scalers()

        print("Phase 3: Creating unified dataset...")
        self._create_unified_dataset(indices)

    def _load_all_flights_metadata(self):
        """Load metadata from all flights for global scaler fitting"""
        self.flight_data = []
        all_sza, all_saa, all_y_valid = [], [], []

        for i, config in enumerate(self.flight_configs):
            try:
                print(f"  Loading flight {config['name']}...")
                flight_info = self._load_single_flight_metadata(config, i)
                self.flight_data.append(flight_info)
                all_sza.append(flight_info["SZA_full"])
                all_saa.append(flight_info["SAA_full"])
                all_y_valid.append(flight_info["y_valid"])
                print(f"    Success: {flight_info['n_samples']} samples")
            except Exception as e:
                print(f"    Error loading flight {config['name']}: {e}")
                continue

        if not self.flight_data:
            raise ValueError("No valid flights loaded.")

        self.global_sza = np.vstack(all_sza)
        self.global_saa = np.vstack(all_saa)
        self.global_y_valid = np.vstack(all_y_valid)

    def _load_single_flight_metadata(self, config, flight_idx):
        """Load metadata for a single flight."""
        with h5py.File(config["iFileName"], "r") as hf:
            ds = hf["Product/Signal"]
            total = ds.shape[0]
            block = ds[0 : min(total, 100), self.start : self.end, :].astype(np.float32)
            times_all = hf["Time/TimeUTC"][:total]

        flat_ref = np.mean(block, axis=0)
        corrected_flat = linear_vignetting_correction(flat_ref, flat_ref)
        median_cf = np.median(corrected_flat)

        with h5py.File(config["cFileName"], "r") as cf:
            cTimePre = cf["layer_descriptor/Profile_Decimal_Julian_Day"][:, 0]
            cTime = cplTimeConvert(cTimePre, config["cFileName"])
            raw_base_all_layers = cf["layer_descriptor/Layer_Base_Altitude"][
                :, :
            ].astype(np.float32)
            raw_base_all_layers[raw_base_all_layers == -9999.0] = np.nan
            base_layer = raw_base_all_layers[:, 0]
            lidar_km = cf["layer_descriptor/Lidar_Surface_Altitude"][:] / 1000.0
            dem_km = cf["layer_descriptor/DEM_Surface_Altitude"][:] / 1000.0
            surf_mask = (
                (np.abs(lidar_km) <= 0.9)
                & (dem_km == 0)
                & np.isfinite(base_layer)
                & (base_layer >= self.cbh_min)
                & (base_layer <= self.cbh_max)
            )
            layer_base = np.where(surf_mask, base_layer, np.nan)
            valid_cpl_mask = np.isfinite(layer_base)
            cTime_clean = cTime[valid_cpl_mask]
            layer_base_clean = layer_base[valid_cpl_mask]

        with h5py.File(config["nFileName"], "r") as nf:
            SZA_raw = nf["nav/solarZenith"][:total].reshape(-1, 1)
            SZA = np.nan_to_num(SZA_raw, nan=np.nanmean(SZA_raw))
            SAA_raw = nf["nav/sunAzGrd"][:total].reshape(-1, 1)
            SAA = np.mod(np.nan_to_num(SAA_raw, nan=0.0), 360.0)

        Y_full = np.full((times_all.shape[0], 1), np.nan, dtype=np.float32)
        time_tolerance = 0.5
        matched_img_indices = []
        matched_time_diffs = []
        for i, cTime_val in enumerate(cTime_clean):
            closest_idx = np.argmin(np.abs(times_all - cTime_val))
            dt = float(np.abs(times_all[closest_idx] - cTime_val))
            if dt <= time_tolerance:
                Y_full[closest_idx, 0] = layer_base_clean[i]
                matched_img_indices.append(int(closest_idx))
                matched_time_diffs.append(dt)

        valid_mask = ~np.isnan(Y_full)
        Y_full[valid_mask] = np.clip(Y_full[valid_mask], self.cbh_min, self.cbh_max)
        y_valid = Y_full[valid_mask].reshape(-1, 1)

        del block
        gc.collect()

        return {
            "name": config["name"],
            "flight_idx": flight_idx,
            "n_samples": total,
            "SZA_full": SZA,
            "SAA_full": SAA,
            "Y_full": Y_full,
            "y_valid": y_valid,
            "times_all": times_all,
            "flat_ref": flat_ref,
            "corrected_flat": corrected_flat,
            "median_cf": median_cf,
            "iFileName": config["iFileName"],
            "nFileName": config["nFileName"],
            # Diagnostics
            "cTime_clean": cTime_clean,
            "matched_img_indices": np.array(matched_img_indices, dtype=np.int32),
            "matched_time_diffs": np.array(matched_time_diffs, dtype=np.float32),
        }

    def get_time_sync_diagnostics(self):
        """Return per-flight arrays for matched image indices and |Δt| in seconds."""
        diags = []
        for info in self.flight_data:
            diags.append(
                {
                    "name": info.get("name", "unknown"),
                    "n_samples": info.get("n_samples", 0),
                    "matched_img_indices": info.get(
                        "matched_img_indices", np.array([], dtype=np.int32)
                    ),
                    "matched_time_diffs": info.get(
                        "matched_time_diffs", np.array([], dtype=np.float32)
                    ),
                }
            )
        return diags

    def _fit_unified_scalers(self):
        """Fit scalers on combined data or use pre-fitted ones."""
        if self.sza_scaler_pre and self.saa_scaler_pre and self.y_scaler_pre:
            print("  Using pre-fitted unified scalers.")
            self.sza_scaler = self.sza_scaler_pre
            self.saa_scaler = self.saa_scaler_pre
            self.y_scaler = self.y_scaler_pre
        else:
            print("  Fitting new unified scalers.")
            self.sza_scaler = StandardScaler().fit(self.global_sza)
            self.saa_scaler = StandardScaler().fit(self.global_saa)
            self.y_scaler = StandardScaler().fit(self.global_y_valid)

        if hasattr(self.sza_scaler, "mean_"):
            print("  Unified scalers fitted/loaded:")
            print(
                f"    SZA: mean={self.sza_scaler.mean_[0]:.3f}, std={self.sza_scaler.scale_[0]:.3f}"
            )
            print(
                f"    SAA: mean={self.saa_scaler.mean_[0]:.3f}, std={self.saa_scaler.scale_[0]:.3f}"
            )
            print(
                f"    Y:   mean={self.y_scaler.mean_[0]:.3f}, std={self.y_scaler.scale_[0]:.3f}"
            )

    def _create_unified_dataset(self, indices):
        """Create unified dataset with global indexing."""
        self.global_to_local = []
        for flight_info in self.flight_data:
            for local_idx in range(flight_info["n_samples"]):
                if not np.isnan(flight_info["Y_full"][local_idx]).any():
                    self.global_to_local.append((flight_info["flight_idx"], local_idx))

        if indices is not None:
            self.indices = np.array(indices, dtype=int)[
                indices < len(self.global_to_local)
            ]
        else:
            self.indices = np.arange(len(self.global_to_local))

        first_flight = self.flight_data[0]
        self.image_shape = (
            self.temporal_frames,
            first_flight["flat_ref"].shape[0],
            first_flight["flat_ref"].shape[1],
        )
        print(
            f"  Unified dataset: {len(self.indices)} samples across {len(self.flight_data)} flights."
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = int(self.indices[idx])
        flight_idx, local_idx = self.global_to_local[global_idx]
        flight_info = self.flight_data[flight_idx]

        frame_indices = [
            local_idx - self.temporal_offset + i for i in range(self.temporal_frames)
        ]

        temporal_stack = []
        with h5py.File(flight_info["iFileName"], "r") as hf:
            for frame_idx in frame_indices:
                img = hf["Product/Signal"][frame_idx, self.start : self.end, :].astype(
                    np.float32
                )
                if self.flat_field_correction:
                    IR_corr = (
                        img * flight_info["median_cf"] / flight_info["corrected_flat"]
                    )
                else:
                    IR_corr = img
                IR_norm = preprocess_image(
                    IR_corr,
                    clip_limit=self.clahe_clip_limit,
                    zscore=self.zscore_normalize,
                )
                temporal_stack.append(torch.from_numpy(IR_norm).unsqueeze(0))

        img_stack = torch.cat(temporal_stack, dim=0)

        # Flags for which angles we actually use
        use_sza = self.angles_mode in ("both", "sza_only")
        use_saa = self.angles_mode in ("both", "saa_only")

        # Apply augmentations if enabled
        if self.augment:
            if self.pil_transforms:
                pil_frames = [T.ToPILImage()(frame) for frame in img_stack]
                transformed_pil_frames = [
                    self.pil_transforms(frame) for frame in pil_frames
                ]
                img_stack = torch.stack(
                    [T.ToTensor()(frame) for frame in transformed_pil_frames]
                ).squeeze(1)

            if self.tensor_transforms:
                img_stack = self.tensor_transforms(img_stack)

            # Geometric augmentation (Horizontal Flip)
            if torch.rand(1) < 0.5:
                img_stack = T.functional.hflip(img_stack)
                # If SAA is being used, update it to remain geometrically consistent
                if use_saa:
                    saa_scalar = flight_info["SAA_full"][local_idx, 0]
                    saa_scalar = 360.0 - saa_scalar
                else:
                    saa_scalar = None
            else:
                saa_scalar = flight_info["SAA_full"][local_idx, 0] if use_saa else None
        else:
            saa_scalar = flight_info["SAA_full"][local_idx, 0] if use_saa else None

        # Collect and scale angles as needed
        if use_sza:
            sza_scalar = flight_info["SZA_full"][local_idx, 0]
            sza_scaled = self.sza_scaler.transform([[sza_scalar]])[0, 0].astype(
                np.float32
            )
        else:
            sza_scaled = None

        if use_saa and saa_scalar is not None:
            saa_scaled = self.saa_scaler.transform([[saa_scalar]])[0, 0].astype(
                np.float32
            )
        else:
            saa_scaled = None

        # Prepare output tensors for scalars, keeping interface (two scalars) stable
        zero_scalar = torch.tensor(0.0, dtype=torch.float32).unsqueeze(0)
        sza_tensor = (
            torch.tensor(sza_scaled, dtype=torch.float32).unsqueeze(0)
            if sza_scaled is not None
            else zero_scalar
        )
        saa_tensor = (
            torch.tensor(saa_scaled, dtype=torch.float32).unsqueeze(0)
            if saa_scaled is not None
            else zero_scalar
        )

        # Target variable Y (always used)
        Y = flight_info["Y_full"][local_idx]
        Y_scaled = (
            self.y_scaler.transform(Y.reshape(1, -1)).flatten().astype(np.float32)
        )

        return (
            img_stack,
            sza_tensor,
            saa_tensor,
            torch.tensor(Y_scaled, dtype=torch.float32),
            global_idx,
            local_idx,
        )

    def get_unscaled_y(self):
        """
        Return the CPL cloud-base height (km) for every sample in this dataset,
        ordered to match self.indices. This reads from in-memory metadata and
        avoids invoking __getitem__ so it's side-effect free and fast.
        """
        print("Fetching unscaled true labels (km)...")
        y_km = np.zeros(len(self.indices), dtype=np.float32)
        for i, gidx in enumerate(self.indices):
            flight_idx, local_idx = self.global_to_local[int(gidx)]
            y_val = self.flight_data[flight_idx]["Y_full"][local_idx]
            # y_val is shape (1,) per sample
            y_km[i] = float(y_val[0]) if np.ndim(y_val) > 0 else float(y_val)
        return y_km

    def get_scaled_y(self):
        """
        Return the standardized Y (same scaling used for training), aligned to self.indices.
        """
        y_km = self.get_unscaled_y().reshape(-1, 1)
        return self.y_scaler.transform(y_km).astype(np.float32).flatten()

    def get_raw_indices(self):
        """
        Helper function to get the original HDF5 frame indices for all samples.
        """
        print("Fetching raw frame indices...")
        raw_indices = []
        for i in range(len(self)):
            # __getitem__ returns (img, sza, saa, y_true_scaled, global_idx, local_idx)
            # We want the 6th element (index 5), which is the absolute local_idx
            _, _, _, _, _, local_idx = self[i]
            raw_indices.append(local_idx)

        return np.array(raw_indices)

    def _get_cbh_range(self, filter_type, cbh_min_override, cbh_max_override):
        if cbh_min_override is not None and cbh_max_override is not None:
            return cbh_min_override, cbh_max_override
        filter_configs = {
            "basic": (0.1, 2.0),
            "practical": (0.05, 2.5),
            "comprehensive": (0.1, 2.0),
        }
        default_min, default_max = filter_configs.get(filter_type, (0.1, 2.0))
        return cbh_min_override or default_min, cbh_max_override or default_max
