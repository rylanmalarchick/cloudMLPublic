#!/usr/bin/env python3
"""
Feature Engineering for CBH Retrieval

Creates physics-based derived features from ERA5 atmospheric data to potentially
improve CBH prediction. Uses AgentBible validators for data quality assurance.

Feature Categories:
1. LCL-based features (cloud formation physics)
2. Thermodynamic features (moisture, stability)
3. Temporal features (diurnal cycle)
4. Interaction features (polynomial, physics-motivated)

Author: CBH Restudy Agent
Date: 2026-01-06
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import AgentBible - note: it uses decorator-based validators
try:
    import agentbible
    from agentbible.errors import ValidationError
    AGENTBIBLE_AVAILABLE = True
    AGENTBIBLE_VERSION = agentbible.__version__
    print(f"AgentBible v{AGENTBIBLE_VERSION} loaded successfully")
except ImportError:
    AGENTBIBLE_AVAILABLE = False
    AGENTBIBLE_VERSION = "N/A"
    print("Warning: AgentBible not available")
    
    class ValidationError(Exception):
        pass


# Direct array validators (AgentBible decorators are for functions, not arrays)
# These are numpy-based validators following AgentBible philosophy
def validate_finite(arr: np.ndarray, name: str = "array") -> np.ndarray:
    """Validate array contains only finite values."""
    if not np.all(np.isfinite(arr)):
        nan_count = np.sum(np.isnan(arr))
        inf_count = np.sum(np.isinf(arr))
        raise ValidationError(
            f"{name} contains non-finite values: {nan_count} NaN, {inf_count} Inf"
        )
    return arr


def validate_range(arr: np.ndarray, min_val: float, max_val: float, 
                   name: str = "array") -> np.ndarray:
    """Validate array values are within range."""
    if np.any(arr < min_val) or np.any(arr > max_val):
        actual_min, actual_max = np.min(arr), np.max(arr)
        raise ValidationError(
            f"{name} out of range [{min_val}, {max_val}], "
            f"got [{actual_min:.3f}, {actual_max:.3f}]"
        )
    return arr


def validate_non_negative(arr: np.ndarray, name: str = "array") -> np.ndarray:
    """Validate array contains only non-negative values."""
    if np.any(arr < 0):
        n_neg = np.sum(arr < 0)
        raise ValidationError(f"{name} contains {n_neg} negative values")
    return arr


def validate_positive(arr: np.ndarray, name: str = "array") -> np.ndarray:
    """Validate array contains only positive values."""
    if np.any(arr <= 0):
        n_nonpos = np.sum(arr <= 0)
        raise ValidationError(f"{name} contains {n_nonpos} non-positive values")
    return arr

# Physical constants
R_DRY = 287.05  # J/(kg·K) - gas constant for dry air
R_VAPOR = 461.5  # J/(kg·K) - gas constant for water vapor
CP_DRY = 1004.0  # J/(kg·K) - specific heat at constant pressure
L_V = 2.5e6  # J/kg - latent heat of vaporization
P0 = 100000.0  # Pa - reference pressure for potential temperature
EPSILON = R_DRY / R_VAPOR  # ~0.622


class FeatureEngineer:
    """
    Physics-based feature engineering for CBH prediction.
    
    Uses ERA5 atmospheric variables to derive features relevant to
    cloud base height estimation.
    """
    
    def __init__(self, input_path: Path, output_dir: Path, random_seed: int = 42):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track which features were created
        self.feature_log = {
            "timestamp": datetime.now().isoformat(),
            "input_file": str(self.input_path),
            "agentbible_version": AGENTBIBLE_VERSION,
            "features_created": [],
            "validation_errors": [],
            "physics_notes": {},
        }
        
    def load_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
        """Load features and metadata from clean HDF5 file."""
        print("\n" + "=" * 80)
        print("Loading Clean Dataset")
        print("=" * 80)
        
        with h5py.File(self.input_path, "r") as f:
            # Load atmospheric features
            atmo_features = {}
            for key in f["atmospheric_features"].keys():
                atmo_features[key] = f[f"atmospheric_features/{key}"][:]
                
            # Load geometric features
            geo_features = {}
            for key in f["geometric_features"].keys():
                geo_features[key] = f[f"geometric_features/{key}"][:]
                
            # Load metadata
            cbh_km = f["metadata/cbh_km"][:]
            flight_ids = f["metadata/flight_id"][:].astype(str)
            timestamps = f["metadata/timestamp"][:] if "timestamp" in f["metadata"] else None
            
        n_samples = len(cbh_km)
        print(f"  Loaded {n_samples} samples")
        print(f"  Atmospheric features: {list(atmo_features.keys())}")
        print(f"  Geometric features: {list(geo_features.keys())}")
        
        return atmo_features, geo_features, cbh_km, flight_ids, timestamps
    
    def _validate_temperature(self, T: np.ndarray, name: str) -> np.ndarray:
        """Validate temperature values (in Kelvin)."""
        T = validate_finite(T, name=name)
        T = validate_range(T, 200.0, 350.0, name=name)  # Reasonable atmospheric range
        return T
    
    def _validate_pressure(self, P: np.ndarray, name: str) -> np.ndarray:
        """Validate pressure values (in Pa)."""
        P = validate_finite(P, name=name)
        P = validate_positive(P, name=name)
        P = validate_range(P, 50000.0, 110000.0, name=name)  # Surface to ~5km
        return P
    
    def _validate_height(self, H: np.ndarray, name: str) -> np.ndarray:
        """Validate height values (in meters)."""
        H = validate_finite(H, name=name)
        # Heights can be negative (below sea level) but should be reasonable
        H = validate_range(H, -500.0, 20000.0, name=name)
        return H
    
    # ========================================================================
    # LCL-based Features
    # ========================================================================
    
    def compute_lcl_features(self, atmo: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute Lifting Condensation Level (LCL) related features.
        
        The LCL is the height at which an air parcel becomes saturated when
        lifted adiabatically. It's closely related to cloud base height.
        
        Features:
        - lcl_deficit: BLH - LCL (positive means BLH above LCL)
        - lcl_ratio: LCL / BLH (ratio of condensation to boundary layer)
        - lcl_normalized: LCL / CBH (how well LCL predicts CBH)
        """
        print("\n  Computing LCL-based features...")
        
        blh = self._validate_height(atmo["blh"], "BLH")
        lcl = self._validate_height(atmo["lcl"], "LCL")
        
        features = {}
        
        # LCL deficit: difference between BLH and LCL
        # Positive = BLH extends above LCL (favorable for cloud formation)
        features["lcl_deficit"] = blh - lcl
        self.feature_log["physics_notes"]["lcl_deficit"] = (
            "BLH - LCL: Positive values indicate boundary layer extends above "
            "condensation level, favorable for cloud formation in BL"
        )
        
        # LCL ratio: how does LCL compare to BLH
        # Avoid division by zero with small epsilon
        features["lcl_ratio"] = lcl / np.maximum(blh, 10.0)
        self.feature_log["physics_notes"]["lcl_ratio"] = (
            "LCL / BLH: Values < 1 mean LCL below BLH (clouds likely in BL), "
            "> 1 means free tropospheric clouds"
        )
        
        # Validate outputs
        for name, arr in features.items():
            try:
                validate_finite(arr, name=name)
                self.feature_log["features_created"].append(name)
            except Exception as e:
                self.feature_log["validation_errors"].append(f"{name}: {str(e)}")
                # Impute with median
                median = np.nanmedian(arr)
                arr[~np.isfinite(arr)] = median
                features[name] = arr
                
        print(f"    Created: {list(features.keys())}")
        return features
    
    # ========================================================================
    # Thermodynamic Features
    # ========================================================================
    
    def compute_thermodynamic_features(self, atmo: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute thermodynamic features from ERA5 data.
        
        Features:
        - dew_point_depression: T2m - Td (larger = drier air)
        - relative_humidity_2m: Approximate RH from T and Td
        - saturation_vapor_pressure: e_s(T)
        - vapor_pressure: e(Td)
        - mixing_ratio: Water vapor mixing ratio
        - potential_temperature: Theta at surface
        - virtual_temperature: Tv accounting for moisture
        """
        print("\n  Computing thermodynamic features...")
        
        t2m = self._validate_temperature(atmo["t2m"], "t2m")
        d2m = self._validate_temperature(atmo["d2m"], "d2m")
        sp = self._validate_pressure(atmo["sp"], "sp")
        
        features = {}
        
        # Dew point depression (already implicit in t2m - d2m)
        features["dew_point_depression"] = t2m - d2m
        validate_non_negative(features["dew_point_depression"], "dew_point_depression")
        self.feature_log["physics_notes"]["dew_point_depression"] = (
            "T - Td: Measure of atmospheric dryness. Larger values = drier air, "
            "which pushes LCL higher"
        )
        
        # Saturation vapor pressure using Clausius-Clapeyron (Magnus formula)
        # e_s = 611.2 * exp(17.67 * (T - 273.15) / (T - 29.65))
        t_celsius = t2m - 273.15
        td_celsius = d2m - 273.15
        
        e_sat = 611.2 * np.exp(17.67 * t_celsius / (t_celsius + 243.5))
        e_actual = 611.2 * np.exp(17.67 * td_celsius / (td_celsius + 243.5))
        
        features["saturation_vapor_pressure"] = e_sat
        features["vapor_pressure"] = e_actual
        
        # Relative humidity
        features["relative_humidity_2m"] = 100.0 * (e_actual / e_sat)
        features["relative_humidity_2m"] = np.clip(features["relative_humidity_2m"], 0, 100)
        self.feature_log["physics_notes"]["relative_humidity_2m"] = (
            "RH = e/e_s * 100%: Higher RH means air closer to saturation, "
            "lower LCL, potentially lower CBH"
        )
        
        # Mixing ratio: r = epsilon * e / (p - e)
        features["mixing_ratio"] = EPSILON * e_actual / (sp - e_actual)
        features["mixing_ratio"] = np.maximum(features["mixing_ratio"], 0)  # Ensure non-negative
        self.feature_log["physics_notes"]["mixing_ratio"] = (
            "r = 0.622 * e / (p - e): Mass of water vapor per mass of dry air (kg/kg)"
        )
        
        # Potential temperature: theta = T * (P0/P)^(R/Cp)
        features["potential_temperature"] = t2m * (P0 / sp) ** (R_DRY / CP_DRY)
        self.feature_log["physics_notes"]["potential_temperature"] = (
            "Theta = T * (P0/P)^(R/Cp): Temperature parcel would have if brought "
            "adiabatically to 1000 hPa. Useful for stability assessment"
        )
        
        # Virtual temperature: Tv = T * (1 + 0.61 * r)
        features["virtual_temperature"] = t2m * (1 + 0.61 * features["mixing_ratio"])
        self.feature_log["physics_notes"]["virtual_temperature"] = (
            "Tv = T * (1 + 0.61*r): Temperature dry air would need to have same "
            "density as moist air. Important for buoyancy calculations"
        )
        
        # Validate outputs
        for name, arr in features.items():
            try:
                validate_finite(arr, name=name)
                self.feature_log["features_created"].append(name)
            except Exception as e:
                self.feature_log["validation_errors"].append(f"{name}: {str(e)}")
                median = np.nanmedian(arr)
                arr[~np.isfinite(arr)] = median
                features[name] = arr
                
        print(f"    Created: {list(features.keys())}")
        return features
    
    # ========================================================================
    # Stability Features
    # ========================================================================
    
    def compute_stability_features(self, atmo: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute atmospheric stability features.
        
        Features:
        - lifted_index_proxy: Simplified stability measure
        - k_index_proxy: Related to thunderstorm potential
        - total_totals_proxy: Another stability index
        """
        print("\n  Computing stability features...")
        
        t2m = self._validate_temperature(atmo["t2m"], "t2m")
        d2m = self._validate_temperature(atmo["d2m"], "d2m")
        stability_index = atmo["stability_index"]
        
        features = {}
        
        # Enhanced stability index: combine with dew point depression
        dpd = t2m - d2m
        features["stability_dpd_product"] = stability_index * dpd
        self.feature_log["physics_notes"]["stability_dpd_product"] = (
            "Stability * DPD: Combined measure - high stability + dry air = "
            "strong capping, inhibits cloud formation"
        )
        
        # Stability relative to typical values
        stability_mean = np.mean(stability_index)
        features["stability_anomaly"] = stability_index - stability_mean
        self.feature_log["physics_notes"]["stability_anomaly"] = (
            "Deviation from mean stability: Positive = more stable than average"
        )
        
        # Moisture-weighted stability
        tcwv = atmo["tcwv"]
        tcwv_norm = tcwv / np.maximum(np.mean(tcwv), 1.0)
        features["stability_moisture_ratio"] = stability_index / np.maximum(tcwv_norm, 0.1)
        self.feature_log["physics_notes"]["stability_moisture_ratio"] = (
            "Stability / normalized TCWV: High values = stable + dry = suppressed clouds"
        )
        
        # Validate outputs
        for name, arr in features.items():
            try:
                validate_finite(arr, name=name)
                self.feature_log["features_created"].append(name)
            except Exception as e:
                self.feature_log["validation_errors"].append(f"{name}: {str(e)}")
                median = np.nanmedian(arr)
                arr[~np.isfinite(arr)] = median
                features[name] = arr
                
        print(f"    Created: {list(features.keys())}")
        return features
    
    # ========================================================================
    # Solar/Temporal Features
    # ========================================================================
    
    def compute_solar_features(self, geo: Dict[str, np.ndarray], 
                               timestamps: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute solar geometry and temporal features.
        
        Features:
        - sza_cos: Cosine of solar zenith angle (more direct radiation at low SZA)
        - sza_sin: Sine of SZA
        - solar_heating_proxy: Inverse SZA as heating indicator
        - hour_sin/cos: Diurnal cycle (if timestamps available)
        """
        print("\n  Computing solar/temporal features...")
        
        sza = geo["sza_deg"]
        saa = geo["saa_deg"]
        
        features = {}
        
        # Trigonometric transforms of solar angles
        sza_rad = np.deg2rad(sza)
        saa_rad = np.deg2rad(saa)
        
        features["sza_cos"] = np.cos(sza_rad)
        features["sza_sin"] = np.sin(sza_rad)
        features["saa_cos"] = np.cos(saa_rad)
        features["saa_sin"] = np.sin(saa_rad)
        
        self.feature_log["physics_notes"]["sza_cos"] = (
            "cos(SZA): Proportional to incoming solar radiation intensity. "
            "Higher values (low SZA) = more heating = deeper boundary layer"
        )
        
        # Solar heating proxy: high when sun is high
        features["solar_heating_proxy"] = features["sza_cos"] ** 2
        self.feature_log["physics_notes"]["solar_heating_proxy"] = (
            "cos^2(SZA): Rough proxy for surface heating intensity"
        )
        
        # If timestamps available, extract diurnal cycle
        if timestamps is not None and len(timestamps) > 0:
            try:
                # Assuming timestamps are in hours or can be converted
                hours = np.array([float(t) % 24 for t in timestamps])
                features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
                features["hour_cos"] = np.cos(2 * np.pi * hours / 24)
                self.feature_log["physics_notes"]["hour_sin"] = (
                    "sin(2*pi*hour/24): Diurnal cycle encoding"
                )
            except:
                print("    Warning: Could not parse timestamps for diurnal features")
        
        # Validate outputs
        for name, arr in features.items():
            try:
                validate_finite(arr, name=name)
                self.feature_log["features_created"].append(name)
            except Exception as e:
                self.feature_log["validation_errors"].append(f"{name}: {str(e)}")
                median = np.nanmedian(arr)
                arr[~np.isfinite(arr)] = median
                features[name] = arr
                
        print(f"    Created: {list(features.keys())}")
        return features
    
    # ========================================================================
    # Interaction Features
    # ========================================================================
    
    def compute_interaction_features(self, atmo: Dict[str, np.ndarray], 
                                     geo: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute interaction features between variables.
        
        These capture non-linear relationships that may improve predictions.
        """
        print("\n  Computing interaction features...")
        
        t2m = atmo["t2m"]
        d2m = atmo["d2m"]
        blh = atmo["blh"]
        lcl = atmo["lcl"]
        tcwv = atmo["tcwv"]
        stability = atmo["stability_index"]
        sza = geo["sza_deg"]
        
        features = {}
        
        # Temperature-moisture interaction
        features["t2m_x_tcwv"] = t2m * tcwv / 1000  # Scale for numerical stability
        self.feature_log["physics_notes"]["t2m_x_tcwv"] = (
            "T2m * TCWV: Warm + moist air = high cloud potential"
        )
        
        # BLH-LCL interaction
        features["blh_x_lcl"] = blh * lcl / 1e6  # Scale
        
        # Stability-moisture interaction
        features["stability_x_tcwv"] = stability * tcwv
        self.feature_log["physics_notes"]["stability_x_tcwv"] = (
            "Stability * TCWV: High stability suppresses convection even with moisture"
        )
        
        # Solar-temperature interaction
        sza_cos = np.cos(np.deg2rad(sza))
        features["t2m_x_sza_cos"] = t2m * sza_cos / 300  # Scale
        self.feature_log["physics_notes"]["t2m_x_sza_cos"] = (
            "T2m * cos(SZA): Surface temperature modified by solar intensity"
        )
        
        # BLH-stability interaction
        features["blh_x_stability"] = blh * stability / 100
        
        # Polynomial features for key predictors
        features["t2m_squared"] = ((t2m - 280) / 10) ** 2  # Centered and scaled
        features["blh_squared"] = (blh / 500) ** 2
        features["lcl_squared"] = (lcl / 500) ** 2
        
        # Dew point depression squared (non-linear moisture effect)
        dpd = t2m - d2m
        features["dpd_squared"] = (dpd / 5) ** 2
        
        # Validate outputs
        for name, arr in features.items():
            try:
                validate_finite(arr, name=name)
                self.feature_log["features_created"].append(name)
            except Exception as e:
                self.feature_log["validation_errors"].append(f"{name}: {str(e)}")
                median = np.nanmedian(arr)
                arr[~np.isfinite(arr)] = median
                features[name] = arr
                
        print(f"    Created: {list(features.keys())}")
        return features
    
    # ========================================================================
    # Main Pipeline
    # ========================================================================
    
    def engineer_features(self) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
        """
        Run full feature engineering pipeline.
        
        Returns:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
            cbh_km: Target variable
            flight_ids: Flight identifiers
        """
        print("\n" + "=" * 80)
        print("Feature Engineering Pipeline")
        print("=" * 80)
        
        # Load data
        atmo, geo, cbh_km, flight_ids, timestamps = self.load_data()
        
        # Compute all feature categories
        lcl_features = self.compute_lcl_features(atmo)
        thermo_features = self.compute_thermodynamic_features(atmo)
        stability_features = self.compute_stability_features(atmo)
        solar_features = self.compute_solar_features(geo, timestamps)
        interaction_features = self.compute_interaction_features(atmo, geo)
        
        # Combine all features
        all_features = {}
        
        # Original atmospheric features
        for key, arr in atmo.items():
            all_features[key] = arr
            
        # Original geometric features
        for key, arr in geo.items():
            all_features[key] = arr
            
        # New engineered features
        for feature_dict in [lcl_features, thermo_features, stability_features,
                            solar_features, interaction_features]:
            all_features.update(feature_dict)
        
        # Create feature matrix
        feature_names = sorted(all_features.keys())
        X = np.column_stack([all_features[name] for name in feature_names])
        
        # Final validation
        print("\n" + "=" * 80)
        print("Final Feature Validation")
        print("=" * 80)
        
        n_original = len(atmo) + len(geo)
        n_engineered = len(feature_names) - n_original
        
        print(f"  Original features: {n_original}")
        print(f"  Engineered features: {n_engineered}")
        print(f"  Total features: {len(feature_names)}")
        print(f"  Feature matrix shape: {X.shape}")
        
        # Check for NaN/Inf
        nan_count = np.sum(np.isnan(X))
        inf_count = np.sum(np.isinf(X))
        
        if nan_count > 0 or inf_count > 0:
            print(f"  WARNING: {nan_count} NaN, {inf_count} Inf values detected")
            # Impute with column medians
            for i in range(X.shape[1]):
                mask = ~np.isfinite(X[:, i])
                if np.any(mask):
                    X[mask, i] = np.nanmedian(X[:, i])
            print("  Imputed non-finite values with column medians")
        else:
            print("  All values are finite")
        
        self.feature_log["n_original_features"] = n_original
        self.feature_log["n_engineered_features"] = n_engineered
        self.feature_log["total_features"] = len(feature_names)
        self.feature_log["feature_names"] = feature_names
        
        return X, feature_names, cbh_km, flight_ids
    
    def save_enhanced_dataset(self, X: np.ndarray, feature_names: List[str],
                              cbh_km: np.ndarray, flight_ids: np.ndarray):
        """Save enhanced dataset to HDF5."""
        output_path = self.output_dir / "Enhanced_Features.hdf5"
        
        print(f"\n  Saving enhanced dataset to {output_path}")
        
        with h5py.File(output_path, "w") as f:
            # Save feature matrix
            f.create_dataset("features", data=X, compression="gzip")
            
            # Save feature names
            f.create_dataset("feature_names", 
                           data=np.array(feature_names, dtype="S50"))
            
            # Save metadata
            meta = f.create_group("metadata")
            meta.create_dataset("cbh_km", data=cbh_km)
            meta.create_dataset("flight_id", data=flight_ids.astype("S10"))
            
            # Save provenance
            f.attrs["created"] = datetime.now().isoformat()
            f.attrs["source_file"] = str(self.input_path)
            f.attrs["agentbible_version"] = AGENTBIBLE_VERSION
            f.attrs["n_samples"] = X.shape[0]
            f.attrs["n_features"] = X.shape[1]
            
        print(f"  Saved: {X.shape[0]} samples x {X.shape[1]} features")
        
        # Save feature log
        log_path = self.output_dir / "feature_engineering_log.json"
        with open(log_path, "w") as f:
            json.dump(self.feature_log, f, indent=2)
        print(f"  Saved log: {log_path}")
        
        return output_path


def run_ablation_study(X: np.ndarray, feature_names: List[str],
                       cbh_km: np.ndarray, flight_ids: np.ndarray,
                       output_dir: Path):
    """
    Run ablation study comparing original vs enhanced features.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score, GroupKFold
    from sklearn.metrics import r2_score, mean_absolute_error
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 80)
    print("Ablation Study: Original vs Enhanced Features")
    print("=" * 80)
    
    # Identify original vs engineered features
    original_features = ["blh", "t2m", "d2m", "sp", "tcwv", "lcl", 
                        "stability_index", "moisture_gradient", "sza_deg", "saa_deg"]
    
    original_idx = [i for i, name in enumerate(feature_names) if name in original_features]
    engineered_idx = [i for i, name in enumerate(feature_names) if name not in original_features]
    
    X_original = X[:, original_idx]
    X_engineered = X[:, engineered_idx]
    
    print(f"  Original features: {len(original_idx)}")
    print(f"  Engineered features: {len(engineered_idx)}")
    
    # Model configuration
    model_params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "random_state": 42,
    }
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_original_scaled = StandardScaler().fit_transform(X_original)
    
    # Per-flight CV (same as in training script)
    results = {}
    
    # 1. Original features only
    print("\n  Testing original features...")
    model = GradientBoostingRegressor(**model_params)
    
    # Use GroupKFold for per-flight validation
    gkf = GroupKFold(n_splits=3)
    
    scores_orig = []
    mae_orig = []
    for train_idx, val_idx in gkf.split(X_original_scaled, cbh_km, groups=flight_ids):
        model.fit(X_original_scaled[train_idx], cbh_km[train_idx])
        pred = model.predict(X_original_scaled[val_idx])
        scores_orig.append(r2_score(cbh_km[val_idx], pred))
        mae_orig.append(mean_absolute_error(cbh_km[val_idx], pred))
    
    results["original"] = {
        "r2_mean": float(np.mean(scores_orig)),
        "r2_std": float(np.std(scores_orig)),
        "mae_mean_km": float(np.mean(mae_orig)),
        "mae_std_km": float(np.std(mae_orig)),
    }
    print(f"    R² = {results['original']['r2_mean']:.4f} ± {results['original']['r2_std']:.4f}")
    print(f"    MAE = {results['original']['mae_mean_km']*1000:.1f} ± {results['original']['mae_std_km']*1000:.1f} m")
    
    # 2. All features (original + engineered)
    print("\n  Testing all features (original + engineered)...")
    model = GradientBoostingRegressor(**model_params)
    
    scores_all = []
    mae_all = []
    for train_idx, val_idx in gkf.split(X_scaled, cbh_km, groups=flight_ids):
        model.fit(X_scaled[train_idx], cbh_km[train_idx])
        pred = model.predict(X_scaled[val_idx])
        scores_all.append(r2_score(cbh_km[val_idx], pred))
        mae_all.append(mean_absolute_error(cbh_km[val_idx], pred))
    
    results["all_features"] = {
        "r2_mean": float(np.mean(scores_all)),
        "r2_std": float(np.std(scores_all)),
        "mae_mean_km": float(np.mean(mae_all)),
        "mae_std_km": float(np.std(mae_all)),
    }
    print(f"    R² = {results['all_features']['r2_mean']:.4f} ± {results['all_features']['r2_std']:.4f}")
    print(f"    MAE = {results['all_features']['mae_mean_km']*1000:.1f} ± {results['all_features']['mae_std_km']*1000:.1f} m")
    
    # 3. Feature importance on full model
    print("\n  Computing feature importance...")
    model = GradientBoostingRegressor(**model_params)
    model.fit(X_scaled, cbh_km)
    
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
        "category": ["original" if name in original_features else "engineered" 
                    for name in feature_names]
    }).sort_values("importance", ascending=False)
    
    results["feature_importance"] = importance_df.to_dict("records")
    
    # Top 15 features
    print("\n  Top 15 Features:")
    for i, row in importance_df.head(15).iterrows():
        print(f"    {row['feature']:30s} {row['importance']:.4f} ({row['category']})")
    
    # Improvement analysis
    r2_improvement = results["all_features"]["r2_mean"] - results["original"]["r2_mean"]
    mae_improvement = results["original"]["mae_mean_km"] - results["all_features"]["mae_mean_km"]
    
    results["improvement"] = {
        "r2_delta": float(r2_improvement),
        "mae_delta_km": float(mae_improvement),
        "mae_delta_m": float(mae_improvement * 1000),
    }
    
    print(f"\n  Improvement with engineered features:")
    print(f"    R² delta: {r2_improvement:+.4f}")
    print(f"    MAE delta: {mae_improvement*1000:+.1f} m")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: R² comparison
    ax = axes[0]
    x_pos = [0, 1]
    r2_vals = [results["original"]["r2_mean"], results["all_features"]["r2_mean"]]
    r2_errs = [results["original"]["r2_std"], results["all_features"]["r2_std"]]
    colors = ["#3498db", "#2ecc71"]
    
    bars = ax.bar(x_pos, r2_vals, yerr=r2_errs, capsize=5, color=colors, edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Original\n(10 features)", "Enhanced\n({} features)".format(len(feature_names))])
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("Model Performance: Original vs Enhanced Features", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
               f"{val:.4f}", ha="center", fontweight="bold")
    
    # Right: Feature importance by category
    ax = axes[1]
    orig_imp = importance_df[importance_df["category"] == "original"]["importance"].sum()
    eng_imp = importance_df[importance_df["category"] == "engineered"]["importance"].sum()
    
    ax.pie([orig_imp, eng_imp], labels=["Original", "Engineered"],
          colors=["#3498db", "#e74c3c"], autopct="%1.1f%%",
          startangle=90, explode=(0, 0.05))
    ax.set_title("Feature Importance by Category", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    fig_path = output_dir / "ablation_study.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved figure: {fig_path}")
    
    # Save results
    results_path = output_dir / "ablation_study_results.json"
    # Convert numpy types for JSON
    results_serializable = json.loads(json.dumps(results, default=str))
    with open(results_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"  Saved results: {results_path}")
    
    return results


def main():
    """Main execution."""
    print("=" * 80)
    print("CBH Feature Engineering Pipeline")
    print("=" * 80)
    
    # Paths
    input_path = PROJECT_ROOT / "outputs/preprocessed_data/Clean_933_Integrated_Features.hdf5"
    output_dir = PROJECT_ROOT / "outputs/feature_engineering"
    
    print(f"\nInput: {input_path}")
    print(f"Output: {output_dir}")
    
    if not input_path.exists():
        print(f"\nERROR: Input file not found: {input_path}")
        return 1
    
    # Initialize engineer
    engineer = FeatureEngineer(input_path, output_dir)
    
    # Run feature engineering
    X, feature_names, cbh_km, flight_ids = engineer.engineer_features()
    
    # Save enhanced dataset
    engineer.save_enhanced_dataset(X, feature_names, cbh_km, flight_ids)
    
    # Run ablation study
    run_ablation_study(X, feature_names, cbh_km, flight_ids, output_dir)
    
    print("\n" + "=" * 80)
    print("Feature Engineering Complete!")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - Enhanced_Features.hdf5")
    print(f"  - feature_engineering_log.json")
    print(f"  - ablation_study.png")
    print(f"  - ablation_study_results.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
