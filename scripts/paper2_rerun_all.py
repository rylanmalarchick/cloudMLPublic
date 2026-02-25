#!/usr/bin/env python3
"""
Paper 2 Complete Rerun: All experiments from raw data.

Single reproducible script that:
1. Loads CPL HDF5 flight data (lat/lon/time/CBH)
2. Matches to ERA5 NetCDF surface data (t2m, d2m, sp, blh, tcwv)
3. Engineers 29 derived features
4. Runs LOFO baseline (6 flights)
5. Runs all domain adaptation experiments (few-shot, instance weighting, TrAdaBoost, MMD)
6. Runs conformal prediction (cross-flight and within-flight)
7. Outputs single reproducible results JSON

Run on desktop where ERA5 data lives:
    ssh desktop "cd /home/rylan/dev/research/NASA/cloudML/programDirectory && python3 scripts/paper2_rerun_all.py"

Author: Rylan (audit rerun)
Date: 2026-02-09
"""

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any

import h5py
import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CPL_DATA_DIR = PROJECT_ROOT.parent / "data"
ERA5_ROOT = Path("/mnt/two/research/NASA/ERA5_data_root/surface")
OUTPUT_DIR = PROJECT_ROOT / "results" / "paper2_rerun_20260209"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Flight configs: CPL HDF5 filename -> ERA5 date
FLIGHTS = {
    "23Oct24": {
        "cpl": "23Oct24/CPL_L2_V1-02_01kmLay_259004_23oct24.hdf5",
        "era5_date": "20241023",
        "campaign": "WHySMIE",
    },
    "30Oct24": {
        "cpl": "30Oct24/CPL_L2_V1-02_01kmLay_259006_30oct24.hdf5",
        "era5_date": "20241030",
        "campaign": "WHySMIE",
    },
    "04Nov24": {
        "cpl": "04Nov24/CPL_L2_V1-02_01kmLay_259008_04nov24.hdf5",
        "era5_date": "20241104",
        "campaign": "WHySMIE",
    },
    "10Feb25": {
        "cpl": "10Feb25/CPL_L2_V1-02_01kmLay_259015_10feb25.hdf5",
        "era5_date": "20250210",
        "campaign": "GLOVE",
    },
    "12Feb25": {
        "cpl": "12Feb25/CPL_L2_V1-02_01kmLay_259016_12feb25.hdf5",
        "era5_date": "20250212",
        "campaign": "GLOVE",
    },
    "18Feb25": {
        "cpl": "18Feb25/CPL_L2_V1-02_01kmLay_259017_18feb25.hdf5",
        "era5_date": "20250218",
        "campaign": "GLOVE",
    },
}

# Ocean filter (rough coastline)
def is_ocean(lat, lon):
    if lat < 30:   return lon < -117
    elif lat < 34: return lon < -118.5
    elif lat < 36: return lon < -120.5
    elif lat < 38: return lon < -122
    elif lat < 40: return lon < -123
    elif lat < 42: return lon < -124
    else:          return lon < -124.5


# === STEP 1: Load CPL data ===
def load_cpl_flight(flight_key: str) -> Dict[str, np.ndarray]:
    """Load CPL HDF5, filter to ocean-only valid CBH."""
    config = FLIGHTS[flight_key]
    path = CPL_DATA_DIR / config["cpl"]
    
    with h5py.File(path, "r") as f:
        lat = f["geolocation/CPL_Latitude"][:, 0]
        lon = f["geolocation/CPL_Longitude"][:, 0]
        # Decimal Julian Day (3 columns: day.fraction for each resolution)
        djd = f["layer_descriptor/Profile_Decimal_Julian_Day"][:, 0]
        cbh_km = f["layer_descriptor/Layer_Base_Altitude"][:, 0]
        sza = f["geolocation/Solar_Zenith_Angle"][:]
        saa = f["geolocation/Solar_Azimuth_Angle"][:]
    
    # Extract UTC hour from decimal julian day (fractional part * 24)
    day_frac_hours = (djd % 1) * 24.0
    
    # Valid CBH: positive, boundary layer only (below 2 km)
    # The paper focused on boundary-layer cloud base height
    valid = (cbh_km > 0) & (cbh_km <= 2.0)
    
    # Ocean filter
    ocean = np.array([is_ocean(lat[i], lon[i]) if valid[i] else False 
                      for i in range(len(lat))])
    
    mask = valid & ocean
    
    return {
        "lat": lat[mask],
        "lon": lon[mask],
        "day_frac": day_frac_hours[mask],
        "sza": sza[mask],
        "saa": saa[mask],
        "cbh_km": cbh_km[mask],
        "n_total": int(valid.sum()),
        "n_ocean": int(mask.sum()),
        "n_land": int(valid.sum() - mask.sum()),
    }


# === STEP 2: Match to ERA5 ===
def load_era5_for_flight(flight_key: str) -> Dict:
    """Load ERA5 NetCDF for a flight date using h5py (avoids netCDF4/h5py libhdf5 conflict)."""
    config = FLIGHTS[flight_key]
    era5_path = ERA5_ROOT / f"era5_surface_{config['era5_date']}.nc"
    
    # Use h5py to read netCDF4 files (they're HDF5 underneath)
    f = h5py.File(str(era5_path), "r")
    
    # valid_time is in seconds since 1970-01-01; we just need hour indices (0-23)
    data = {
        "lat": f["latitude"][:],
        "lon": f["longitude"][:],
        "t2m": f["t2m"][:],    # K, shape (24, nlat, nlon)
        "d2m": f["d2m"][:],    # K
        "sp": f["sp"][:],      # Pa
        "blh": f["blh"][:],    # m
        "tcwv": f["tcwv"][:],  # kg/m²
    }
    f.close()
    return data


def match_era5(cpl: Dict, era5: Dict, flight_key: str) -> np.ndarray:
    """Match CPL obs to nearest ERA5 grid point + hour. Returns feature array."""
    n = len(cpl["lat"])
    
    # ERA5 grid
    era5_lats = era5["lat"]
    era5_lons = era5["lon"]
    
    # Base features: t2m, d2m, sp, blh, tcwv (5 variables)
    features = np.zeros((n, 5))
    
    for i in range(n):
        # Nearest lat/lon
        lat_idx = np.argmin(np.abs(era5_lats - cpl["lat"][i]))
        lon_idx = np.argmin(np.abs(era5_lons - cpl["lon"][i]))
        
        # Nearest hour from day fraction (UTC)
        hour = int(cpl["day_frac"][i]) % 24
        hour_idx = min(hour, 23)
        
        features[i, 0] = era5["t2m"][hour_idx, lat_idx, lon_idx]
        features[i, 1] = era5["d2m"][hour_idx, lat_idx, lon_idx]
        features[i, 2] = era5["sp"][hour_idx, lat_idx, lon_idx]
        features[i, 3] = era5["blh"][hour_idx, lat_idx, lon_idx]
        features[i, 4] = era5["tcwv"][hour_idx, lat_idx, lon_idx]
    
    return features

BASE_FEATURE_NAMES = ["t2m", "d2m", "sp", "blh", "tcwv"]


# === STEP 3: Feature engineering (29 derived features) ===
def engineer_features(base: np.ndarray, cpl: Dict) -> Tuple[np.ndarray, List[str]]:
    """Engineer 29 derived features from 5 base ERA5 + position/time."""
    t2m = base[:, 0]  # K
    d2m = base[:, 1]  # K
    sp  = base[:, 2]  # Pa
    blh = base[:, 3]  # m
    tcwv = base[:, 4] # kg/m²
    lat = cpl["lat"]
    lon = cpl["lon"]
    day_frac = cpl["day_frac"]
    
    derived = {}
    
    # --- LCL features (2) ---
    lcl = 125.0 * ((t2m - 273.15) - (d2m - 273.15))
    lcl = np.clip(lcl, 0, 15000)
    derived["lcl"] = lcl
    derived["lcl_deficit"] = blh - lcl
    
    # --- Thermodynamic features (8) ---
    # Virtual temperature
    e_s = 611.2 * np.exp(17.67 * (d2m - 273.15) / (d2m - 29.65))
    w = 0.622 * e_s / (sp - e_s)
    t_v = t2m * (1 + 0.61 * w)
    derived["t_virtual"] = t_v
    
    # Dewpoint depression
    derived["dewpoint_depression"] = t2m - d2m
    
    # Relative humidity (approximate)
    rh = np.exp(17.67 * (d2m - 273.15) / (d2m - 29.65)) / \
         np.exp(17.67 * (t2m - 273.15) / (t2m - 29.65))
    derived["rh"] = np.clip(rh, 0, 1)
    
    # Stability index
    derived["stability_index"] = (t2m - d2m) / 10.0
    
    # Moisture gradient
    derived["moisture_gradient"] = ((d2m - 250) / 30) + (tcwv / 30)
    
    # Pressure altitude (hypsometric)
    derived["pressure_altitude"] = 44330 * (1 - (sp / 101325) ** 0.1903)
    
    # Theta-e (equivalent potential temperature, simplified)
    theta = t2m * (100000 / sp) ** 0.286
    derived["theta_e"] = theta * np.exp(2.5e6 * w / (1004 * t2m))
    
    # BLH/LCL ratio
    derived["blh_lcl_ratio"] = blh / (lcl + 1)
    
    # --- Stability features (4) ---
    derived["stability_tcwv"] = derived["stability_index"] * tcwv
    derived["dd_blh"] = derived["dewpoint_depression"] * blh / 1000
    derived["t2m_d2m_ratio"] = t2m / (d2m + 0.01)
    derived["inversion_strength"] = blh * derived["stability_index"]
    
    # --- Solar/temporal features (7) ---
    sza_rad = np.radians(cpl["sza"])
    derived["sza_cos"] = np.cos(sza_rad)
    derived["sza_sin"] = np.sin(sza_rad)
    
    saa_rad = np.radians(cpl["saa"])
    derived["saa_cos"] = np.cos(saa_rad)
    derived["saa_sin"] = np.sin(saa_rad)
    
    hour = day_frac  # already in hours
    derived["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    derived["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    
    derived["solar_heating_proxy"] = np.maximum(0, derived["sza_cos"]) * (t2m - 273.15)
    
    # --- Interaction features (8) ---
    derived["t2m_tcwv"] = (t2m - 273.15) * tcwv
    derived["rh_blh"] = derived["rh"] * blh
    derived["lcl_sq"] = lcl ** 2 / 1e6  # Scale down
    derived["blh_sq"] = blh ** 2 / 1e6
    derived["t2m_sp"] = (t2m - 273.15) * sp / 1e5
    derived["lat_abs"] = np.abs(lat)
    derived["lon_abs"] = np.abs(lon)
    derived["lat_lon"] = lat * lon / 1000
    
    # Stack all derived
    derived_names = list(derived.keys())
    derived_arr = np.column_stack([derived[k] for k in derived_names])
    
    # Combine base + derived
    full = np.hstack([base, derived_arr])
    full_names = BASE_FEATURE_NAMES + derived_names
    
    # Handle NaN/Inf
    mask = np.all(np.isfinite(full), axis=1)
    if mask.sum() < len(mask):
        print(f"    Warning: dropping {(~mask).sum()} non-finite rows")
    
    return full, full_names, mask


# === STEP 4: Run all experiments ===
def run_lofo_baseline(X_by_flight, y_by_flight, flight_keys):
    """Leave-one-flight-out baseline."""
    print("\n=== LOFO Baseline ===")
    results = {}
    
    for i, target in enumerate(flight_keys):
        # Source = all other flights
        X_source = np.vstack([X_by_flight[k] for k in flight_keys if k != target])
        y_source = np.concatenate([y_by_flight[k] for k in flight_keys if k != target])
        X_target = X_by_flight[target]
        y_target = y_by_flight[target]
        
        scaler = StandardScaler()
        X_source_s = scaler.fit_transform(X_source)
        X_target_s = scaler.transform(X_target)
        
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
        )
        model.fit(X_source_s, y_source)
        y_pred = model.predict(X_target_s)
        
        r2 = float(r2_score(y_target, y_pred))
        mae_m = float(mean_absolute_error(y_target, y_pred) * 1000)
        
        results[target] = {
            "r2": r2, "mae_m": mae_m, "n_test": len(y_target),
            "campaign": FLIGHTS[target]["campaign"],
        }
        print(f"  {target} ({FLIGHTS[target]['campaign']}): R²={r2:.3f}, MAE={mae_m:.0f}m, n={len(y_target)}")
    
    mean_r2 = float(np.mean([v["r2"] for v in results.values()]))
    mean_mae = float(np.mean([v["mae_m"] for v in results.values()]))
    print(f"  MEAN: R²={mean_r2:.3f}, MAE={mean_mae:.0f}m")
    
    return {"per_flight": results, "mean_r2": mean_r2, "mean_mae_m": mean_mae}


def run_fewshot(X_by_flight, y_by_flight, flight_keys, shot_sizes=[5, 10, 20, 50]):
    """Few-shot adaptation: add N target samples to source training."""
    print("\n=== Few-Shot Adaptation ===")
    results = {n: {} for n in shot_sizes}
    
    for target in flight_keys:
        X_source = np.vstack([X_by_flight[k] for k in flight_keys if k != target])
        y_source = np.concatenate([y_by_flight[k] for k in flight_keys if k != target])
        X_target = X_by_flight[target]
        y_target = y_by_flight[target]
        
        scaler = StandardScaler()
        X_source_s = scaler.fit_transform(X_source)
        X_target_s = scaler.transform(X_target)
        
        for n_shots in shot_sizes:
            if n_shots >= len(X_target) * 0.8:
                continue
            
            trial_r2s = []
            for trial in range(20):  # 20 random trials
                idx = np.random.permutation(len(X_target))
                adapt_idx, test_idx = idx[:n_shots], idx[n_shots:]
                
                X_train = np.vstack([X_source_s, X_target_s[adapt_idx]])
                y_train = np.concatenate([y_source, y_target[adapt_idx]])
                
                model = GradientBoostingRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.1,
                    random_state=42 + trial
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_target_s[test_idx])
                trial_r2s.append(float(r2_score(y_target[test_idx], y_pred)))
            
            mean_r2 = float(np.mean(trial_r2s))
            std_r2 = float(np.std(trial_r2s))
            results[n_shots][target] = {"mean_r2": mean_r2, "std_r2": std_r2}
            print(f"  {target} {n_shots}-shot: R²={mean_r2:.3f}±{std_r2:.3f}")
    
    # Aggregate means
    agg = {}
    for n_shots in shot_sizes:
        if results[n_shots]:
            agg[n_shots] = float(np.mean([v["mean_r2"] for v in results[n_shots].values()]))
    
    print(f"  MEANS: {', '.join(f'{k}-shot={v:.3f}' for k,v in agg.items())}")
    return {"per_flight": {str(k): v for k, v in results.items()}, "mean_by_shots": agg}


def run_instance_weighting(X_by_flight, y_by_flight, flight_keys):
    """Instance weighting: KNN and density-ratio weighting."""
    print("\n=== Instance Weighting ===")
    results = {"knn": {}, "density": {}}
    
    for target in flight_keys:
        X_source = np.vstack([X_by_flight[k] for k in flight_keys if k != target])
        y_source = np.concatenate([y_by_flight[k] for k in flight_keys if k != target])
        X_target = X_by_flight[target]
        y_target = y_by_flight[target]
        
        scaler = StandardScaler()
        X_source_s = scaler.fit_transform(X_source)
        X_target_s = scaler.transform(X_target)
        
        for method in ["knn", "density"]:
            if method == "knn":
                k = min(5, len(X_target_s))
                nn = NearestNeighbors(n_neighbors=k)
                nn.fit(X_target_s)
                dists, _ = nn.kneighbors(X_source_s)
                weights = 1.0 / (1.0 + dists.mean(axis=1))
                weights = 0.1 + 0.9 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
            else:
                from sklearn.neighbors import KernelDensity
                kde_s = KernelDensity(bandwidth=0.5).fit(X_source_s)
                kde_t = KernelDensity(bandwidth=0.5).fit(X_target_s)
                log_w = kde_t.score_samples(X_source_s) - kde_s.score_samples(X_source_s)
                weights = np.exp(np.clip(log_w, -5, 5))
                weights = weights / weights.mean()
            
            model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
            model.fit(X_source_s, y_source, sample_weight=weights)
            y_pred = model.predict(X_target_s)
            
            r2 = float(r2_score(y_target, y_pred))
            results[method][target] = {"r2": r2, "mae_m": float(mean_absolute_error(y_target, y_pred) * 1000)}
            print(f"  {target} {method}: R²={r2:.3f}")
    
    agg = {}
    for method in ["knn", "density"]:
        agg[method] = float(np.mean([v["r2"] for v in results[method].values()]))
    
    print(f"  MEANS: KNN={agg['knn']:.3f}, density={agg['density']:.3f}")
    return {"per_flight": results, "mean_r2": agg}


def run_tradaboost(X_by_flight, y_by_flight, flight_keys):
    """TrAdaBoost transfer learning."""
    print("\n=== TrAdaBoost ===")
    results = {}
    
    for target in flight_keys:
        X_source = np.vstack([X_by_flight[k] for k in flight_keys if k != target])
        y_source = np.concatenate([y_by_flight[k] for k in flight_keys if k != target])
        X_target = X_by_flight[target]
        y_target = y_by_flight[target]
        
        if len(X_target) < 30:
            print(f"  {target}: skipped (n={len(X_target)})")
            continue
        
        scaler = StandardScaler()
        X_source_s = scaler.fit_transform(X_source)
        X_target_s = scaler.transform(X_target)
        
        # Split target: 20% adapt, 80% test
        n_adapt = max(10, int(0.2 * len(X_target)))
        idx = np.random.permutation(len(X_target))
        adapt_idx, test_idx = idx[:n_adapt], idx[n_adapt:]
        
        X_adapt, y_adapt = X_target_s[adapt_idx], y_target[adapt_idx]
        X_test, y_test = X_target_s[test_idx], y_target[test_idx]
        
        # TrAdaBoost iterations
        n_source = len(X_source_s)
        n_adapt_samples = len(X_adapt)
        X_combined = np.vstack([X_source_s, X_adapt])
        y_combined = np.concatenate([y_source, y_adapt])
        
        w_source = np.ones(n_source) / n_source
        w_target = np.ones(n_adapt_samples) / n_adapt_samples
        beta_s = 1.0 / (1.0 + np.sqrt(2 * np.log(n_source) / 20))
        
        models, betas = [], []
        for t in range(20):
            weights = np.concatenate([w_source, w_target])
            weights = weights / weights.sum()
            
            m = GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=42+t)
            m.fit(X_combined, y_combined, sample_weight=weights)
            
            y_p = m.predict(X_combined)
            errors = np.abs(y_combined - y_p)
            max_err = errors.max() + 1e-8
            norm_errors = errors / max_err
            
            tw = w_target / w_target.sum()
            eps = np.clip(np.sum(tw * norm_errors[n_source:]), 0.01, 0.99)
            beta_t = eps / (1 - eps)
            
            w_source *= np.power(beta_s, norm_errors[:n_source])
            w_target *= np.power(1.0 / (beta_t + 1e-8), -norm_errors[n_source:])
            
            models.append(m)
            betas.append(beta_t)
        
        # Weighted prediction
        preds = np.array([m.predict(X_test) for m in models])
        ws = np.log(1.0 / (np.array(betas) + 1e-8))
        ws = ws / ws.sum()
        y_pred = np.average(preds, axis=0, weights=ws)
        
        r2 = float(r2_score(y_test, y_pred))
        results[target] = {"r2": r2, "mae_m": float(mean_absolute_error(y_test, y_pred) * 1000)}
        print(f"  {target}: R²={r2:.3f}")
    
    mean_r2 = float(np.mean([v["r2"] for v in results.values()]))
    print(f"  MEAN: R²={mean_r2:.3f}")
    return {"per_flight": results, "mean_r2": mean_r2}


def run_mmd_alignment(X_by_flight, y_by_flight, flight_keys):
    """MMD-based feature alignment."""
    print("\n=== MMD Alignment ===")
    from sklearn.decomposition import PCA
    
    results = {}
    for target in flight_keys:
        X_source = np.vstack([X_by_flight[k] for k in flight_keys if k != target])
        y_source = np.concatenate([y_by_flight[k] for k in flight_keys if k != target])
        X_target = X_by_flight[target]
        y_target = y_by_flight[target]
        
        scaler = StandardScaler()
        X_source_s = scaler.fit_transform(X_source)
        X_target_s = scaler.transform(X_target)
        
        n_comp = min(X_source_s.shape[1], 20)
        pca = PCA(n_components=n_comp, random_state=42)
        X_all = np.vstack([X_source_s, X_target_s])
        pca.fit(X_all)
        
        X_src_pca = pca.transform(X_source_s)
        X_tgt_pca = pca.transform(X_target_s)
        
        # Mean alignment
        X_src_aligned = X_src_pca - X_src_pca.mean(0) + X_tgt_pca.mean(0)
        
        model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_src_aligned, y_source)
        y_pred = model.predict(X_tgt_pca)
        
        r2 = float(r2_score(y_target, y_pred))
        results[target] = {"r2": r2, "mae_m": float(mean_absolute_error(y_target, y_pred) * 1000)}
        print(f"  {target}: R²={r2:.3f}")
    
    mean_r2 = float(np.mean([v["r2"] for v in results.values()]))
    print(f"  MEAN: R²={mean_r2:.3f}")
    return {"per_flight": results, "mean_r2": mean_r2}


def run_feature_selection(X_by_flight, y_by_flight, flight_keys, feature_names):
    """Feature selection: use only most stable features across flights."""
    print("\n=== Feature Selection ===")
    
    # Find features with lowest K-S divergence across flight pairs
    all_flights = list(flight_keys)
    n_features = X_by_flight[all_flights[0]].shape[1]
    
    # Mean KS stat per feature
    ks_per_feature = np.zeros(n_features)
    n_pairs = 0
    for i in range(len(all_flights)):
        for j in range(i+1, len(all_flights)):
            for f in range(n_features):
                stat, _ = ks_2samp(X_by_flight[all_flights[i]][:, f],
                                    X_by_flight[all_flights[j]][:, f])
                ks_per_feature[f] += stat
            n_pairs += 1
    ks_per_feature /= n_pairs
    
    # Select bottom 50% (most stable)
    n_select = n_features // 2
    selected_idx = np.argsort(ks_per_feature)[:n_select]
    sel_names = [feature_names[i] for i in selected_idx]
    print(f"  Selected {n_select} most stable features: {sel_names[:5]}...")
    
    results = {}
    for target in flight_keys:
        X_source = np.vstack([X_by_flight[k][:, selected_idx] for k in flight_keys if k != target])
        y_source = np.concatenate([y_by_flight[k] for k in flight_keys if k != target])
        X_target = X_by_flight[target][:, selected_idx]
        y_target = y_by_flight[target]
        
        scaler = StandardScaler()
        X_source_s = scaler.fit_transform(X_source)
        X_target_s = scaler.transform(X_target)
        
        model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_source_s, y_source)
        y_pred = model.predict(X_target_s)
        
        r2 = float(r2_score(y_target, y_pred))
        results[target] = {"r2": r2}
        print(f"  {target}: R²={r2:.3f}")
    
    mean_r2 = float(np.mean([v["r2"] for v in results.values()]))
    print(f"  MEAN: R²={mean_r2:.3f}")
    return {"per_flight": results, "mean_r2": mean_r2, "selected_features": sel_names}


def run_conformal_prediction(X_by_flight, y_by_flight, flight_keys):
    """Conformal prediction: both cross-flight and within-flight."""
    print("\n=== Conformal Prediction ===")
    
    results = {"cross_flight": {}, "within_flight": {}}
    
    # Cross-flight conformal: calibrate on source flights, evaluate on target
    for target in flight_keys:
        X_source = np.vstack([X_by_flight[k] for k in flight_keys if k != target])
        y_source = np.concatenate([y_by_flight[k] for k in flight_keys if k != target])
        X_target = X_by_flight[target]
        y_target = y_by_flight[target]
        
        scaler = StandardScaler()
        X_source_s = scaler.fit_transform(X_source)
        X_target_s = scaler.transform(X_target)
        
        # Split source into train/calibration
        n_cal = len(X_source_s) // 5
        idx = np.random.permutation(len(X_source_s))
        cal_idx, train_idx = idx[:n_cal], idx[n_cal:]
        
        model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_source_s[train_idx], y_source[train_idx])
        
        # Calibration residuals
        cal_resid = np.abs(y_source[cal_idx] - model.predict(X_source_s[cal_idx]))
        q = np.quantile(cal_resid, 0.9)  # 90% coverage target
        
        # Test on target
        y_pred = model.predict(X_target_s)
        resid = np.abs(y_target - y_pred)
        coverage = float((resid <= q).mean())
        width_m = float(2 * q * 1000)
        
        results["cross_flight"][target] = {"coverage": coverage, "width_m": width_m}
        print(f"  Cross-flight {target}: coverage={coverage:.1%}, width={width_m:.0f}m")
    
    cross_mean_cov = float(np.mean([v["coverage"] for v in results["cross_flight"].values()]))
    cross_mean_width = float(np.mean([v["width_m"] for v in results["cross_flight"].values()]))
    results["cross_flight"]["mean"] = {"coverage": cross_mean_cov, "width_m": cross_mean_width}
    print(f"  Cross-flight MEAN: coverage={cross_mean_cov:.1%}, width={cross_mean_width:.0f}m")
    
    # Within-flight conformal: calibrate and evaluate within same flight
    for target in flight_keys:
        X = X_by_flight[target]
        y = y_by_flight[target]
        
        if len(X) < 50:
            continue
        
        scaler = StandardScaler()
        
        # 60% train, 20% cal, 20% test
        idx = np.random.permutation(len(X))
        n1 = int(0.6 * len(X))
        n2 = int(0.8 * len(X))
        train_idx, cal_idx, test_idx = idx[:n1], idx[n1:n2], idx[n2:]
        
        X_s = scaler.fit_transform(X)
        model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_s[train_idx], y[train_idx])
        
        cal_resid = np.abs(y[cal_idx] - model.predict(X_s[cal_idx]))
        q = np.quantile(cal_resid, 0.9)
        
        test_resid = np.abs(y[test_idx] - model.predict(X_s[test_idx]))
        coverage = float((test_resid <= q).mean())
        width_m = float(2 * q * 1000)
        
        r2 = float(r2_score(y[test_idx], model.predict(X_s[test_idx])))
        results["within_flight"][target] = {"coverage": coverage, "width_m": width_m, "r2": r2}
        print(f"  Within-flight {target}: coverage={coverage:.1%}, width={width_m:.0f}m, R²={r2:.3f}")
    
    within_mean_cov = float(np.mean([v["coverage"] for v in results["within_flight"].values()]))
    within_mean_width = float(np.mean([v["width_m"] for v in results["within_flight"].values()]))
    results["within_flight"]["mean"] = {"coverage": within_mean_cov, "width_m": within_mean_width}
    print(f"  Within-flight MEAN: coverage={within_mean_cov:.1%}, width={within_mean_width:.0f}m")
    
    return results


def run_ks_divergence(X_by_flight, flight_keys, feature_names):
    """K-S divergence between Oct 23 and Feb 10."""
    print("\n=== K-S Divergence (Oct23 vs Feb10) ===")
    X1 = X_by_flight["23Oct24"]
    X2 = X_by_flight["10Feb25"]
    
    ks_results = []
    for i, name in enumerate(feature_names):
        stat, pval = ks_2samp(X1[:, i], X2[:, i])
        ks_results.append({"feature": name, "ks_stat": float(stat), "pval": float(pval)})
    
    ks_results.sort(key=lambda x: -x["ks_stat"])
    print("  Top 10 most shifted features:")
    for r in ks_results[:10]:
        print(f"    {r['feature']}: KS={r['ks_stat']:.3f}")
    
    return ks_results


def run_pooled_cv(X_by_flight, y_by_flight, flight_keys):
    """Pooled 5-fold CV to show autocorrelation inflation."""
    print("\n=== Pooled 5-Fold CV ===")
    from sklearn.model_selection import cross_val_score
    
    X_all = np.vstack([X_by_flight[k] for k in flight_keys])
    y_all = np.concatenate([y_by_flight[k] for k in flight_keys])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    scores = cross_val_score(model, X_scaled, y_all, cv=5, scoring="r2")
    
    mean_r2 = float(np.mean(scores))
    print(f"  Pooled CV R²: {mean_r2:.3f} ± {np.std(scores):.3f}")
    print(f"  Per-fold: {[f'{s:.3f}' for s in scores]}")
    
    return {"mean_r2": mean_r2, "std_r2": float(np.std(scores)), "per_fold": [float(s) for s in scores]}


def run_base_vs_enhanced(X_by_flight, y_by_flight, flight_keys):
    """Compare base 10 features (5 ERA5 + lat/lon/time proxies) vs all 34."""
    print("\n=== Base vs Enhanced Feature Comparison (Per-Flight CV) ===")
    
    # Per-flight CV with all features
    results = {}
    for nf_label, n_features in [("base_5", 5), ("all", None)]:
        r2s = []
        for flight in flight_keys:
            X = X_by_flight[flight]
            if n_features is not None:
                X = X[:, :n_features]
            y = y_by_flight[flight]
            
            if len(X) < 20:
                continue
            
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            
            from sklearn.model_selection import cross_val_score
            model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
            scores = cross_val_score(model, X_s, y, cv=min(5, len(X)//5), scoring="r2")
            r2s.append(float(np.mean(scores)))
        
        mean_r2 = float(np.mean(r2s))
        results[nf_label] = {"mean_r2": mean_r2, "per_flight": r2s}
        print(f"  {nf_label}: Per-flight CV R² = {mean_r2:.3f}")
    
    return results


# === MAIN ===
def main():
    print("=" * 80)
    print("Paper 2 Complete Rerun — All Experiments from Raw Data")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)
    
    # Load all flights
    X_by_flight = {}
    y_by_flight = {}
    stats = {}
    feature_names = None
    
    for flight_key in FLIGHTS:
        print(f"\nLoading {flight_key} ({FLIGHTS[flight_key]['campaign']})...")
        
        # Load CPL
        cpl = load_cpl_flight(flight_key)
        print(f"  CPL: {cpl['n_ocean']} ocean / {cpl['n_land']} land / {cpl['n_total']} total valid CBH")
        
        if cpl["n_ocean"] == 0:
            print(f"  SKIPPING {flight_key}: no ocean points")
            continue
        
        # Load ERA5
        era5 = load_era5_for_flight(flight_key)
        
        # Match
        base_features = match_era5(cpl, era5, flight_key)
        print(f"  ERA5 matched: {base_features.shape}")
        
        # Engineer features
        full_features, names, finite_mask = engineer_features(base_features, cpl)
        if feature_names is None:
            feature_names = names
        
        X_by_flight[flight_key] = full_features[finite_mask]
        y_by_flight[flight_key] = cpl["cbh_km"][finite_mask]
        
        stats[flight_key] = {
            "campaign": FLIGHTS[flight_key]["campaign"],
            "n_ocean_cbh": cpl["n_ocean"],
            "n_era5_matched": int(finite_mask.sum()),
            "n_land_excluded": cpl["n_land"],
            "cbh_mean_m": float(cpl["cbh_km"][finite_mask].mean() * 1000),
            "cbh_std_m": float(cpl["cbh_km"][finite_mask].std() * 1000),
        }
        print(f"  Final: {finite_mask.sum()} samples, CBH mean={stats[flight_key]['cbh_mean_m']:.0f}m")
    
    flight_keys = list(X_by_flight.keys())
    total_n = sum(len(X_by_flight[k]) for k in flight_keys)
    print(f"\n{'='*80}")
    print(f"Total: {total_n} samples across {len(flight_keys)} flights")
    print(f"Features: {len(feature_names)} ({', '.join(feature_names[:10])}...)")
    print(f"{'='*80}")
    
    # Run all experiments
    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_flights": len(flight_keys),
            "flight_keys": flight_keys,
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "total_samples": total_n,
        },
        "flight_stats": stats,
    }
    
    all_results["pooled_cv"] = run_pooled_cv(X_by_flight, y_by_flight, flight_keys)
    all_results["base_vs_enhanced"] = run_base_vs_enhanced(X_by_flight, y_by_flight, flight_keys)
    all_results["lofo_baseline"] = run_lofo_baseline(X_by_flight, y_by_flight, flight_keys)
    all_results["few_shot"] = run_fewshot(X_by_flight, y_by_flight, flight_keys)
    all_results["instance_weighting"] = run_instance_weighting(X_by_flight, y_by_flight, flight_keys)
    all_results["tradaboost"] = run_tradaboost(X_by_flight, y_by_flight, flight_keys)
    all_results["mmd_alignment"] = run_mmd_alignment(X_by_flight, y_by_flight, flight_keys)
    all_results["feature_selection"] = run_feature_selection(X_by_flight, y_by_flight, flight_keys, feature_names)
    all_results["ks_divergence"] = run_ks_divergence(X_by_flight, flight_keys, feature_names)
    all_results["conformal"] = run_conformal_prediction(X_by_flight, y_by_flight, flight_keys)
    
    # Save
    out_path = OUTPUT_DIR / "paper2_all_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*80}")
    print(f"ALL RESULTS SAVED: {out_path}")
    print(f"{'='*80}")
    
    # Print summary table
    print("\n=== SUMMARY TABLE FOR PAPER ===")
    print(f"{'Method':<30} {'Mean R²':>10} {'Notes':>20}")
    print("-" * 62)
    print(f"{'Pooled 5-fold CV':<30} {all_results['pooled_cv']['mean_r2']:>10.3f} {'(inflated)':>20}")
    print(f"{'LOFO Baseline':<30} {all_results['lofo_baseline']['mean_r2']:>10.3f} {'':>20}")
    for ns in [5, 10, 20, 50]:
        if ns in all_results["few_shot"]["mean_by_shots"]:
            print(f"{'Few-shot (' + str(ns) + ')':<30} {all_results['few_shot']['mean_by_shots'][ns]:>10.3f}")
    print(f"{'IW (KNN)':<30} {all_results['instance_weighting']['mean_r2']['knn']:>10.3f}")
    print(f"{'IW (density)':<30} {all_results['instance_weighting']['mean_r2']['density']:>10.3f}")
    print(f"{'TrAdaBoost':<30} {all_results['tradaboost']['mean_r2']:>10.3f}")
    print(f"{'MMD Alignment':<30} {all_results['mmd_alignment']['mean_r2']:>10.3f}")
    print(f"{'Feature Selection':<30} {all_results['feature_selection']['mean_r2']:>10.3f}")
    print(f"\nConformal (cross-flight): coverage={all_results['conformal']['cross_flight']['mean']['coverage']:.1%}")
    print(f"Conformal (within-flight): coverage={all_results['conformal']['within_flight']['mean']['coverage']:.1%}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
