#!/usr/bin/env python3
"""
Adaptive conformal prediction verification.

Runs adaptive conformal (online quantile adjustment) cross-flight
to verify/replace the unverified 11% coverage claim in Paper 2.

Uses the same LOFO structure as paper2_rerun_v2.py:
- Train on 5 flights, calibrate on held-out portion of source, test on 6th flight
- Standard conformal: fixed quantile from calibration
- Adaptive conformal: online quantile adjustment using exponential update

Run on desktop:
    ssh desktop "cd /home/rylan/dev/research/NASA/cloudML/programDirectory && \
        nohup ./venv/bin/python3 scripts/verify_adaptive_conformal.py > results/paper2_rerun_v2/adaptive_conformal.log 2>&1 &"

Author: Rylan (audit item 4)
Date: 2026-02-24
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CPL_DATA_DIR = PROJECT_ROOT.parent / "data"
ERA5_ROOT = Path("/mnt/two/research/NASA/ERA5_data_root/surface")
OUTPUT_DIR = PROJECT_ROOT / "results" / "paper2_rerun_v2"

FLIGHTS = {
    "23Oct24": {"cpl": "23Oct24/CPL_L2_V1-02_01kmLay_259004_23oct24.hdf5", "era5_date": "20241023", "campaign": "WHySMIE"},
    "30Oct24": {"cpl": "30Oct24/CPL_L2_V1-02_01kmLay_259006_30oct24.hdf5", "era5_date": "20241030", "campaign": "WHySMIE"},
    "04Nov24": {"cpl": "04Nov24/CPL_L2_V1-02_01kmLay_259008_04nov24.hdf5", "era5_date": "20241104", "campaign": "WHySMIE"},
    "10Feb25": {"cpl": "10Feb25/CPL_L2_V1-02_01kmLay_259015_10feb25.hdf5", "era5_date": "20250210", "campaign": "GLOVE"},
    "12Feb25": {"cpl": "12Feb25/CPL_L2_V1-02_01kmLay_259016_12feb25.hdf5", "era5_date": "20250212", "campaign": "GLOVE"},
    "18Feb25": {"cpl": "18Feb25/CPL_L2_V1-02_01kmLay_259017_18feb25.hdf5", "era5_date": "20250218", "campaign": "GLOVE"},
}

BASE_FEATURE_NAMES = ["t2m", "d2m", "sp", "blh", "tcwv"]


def is_ocean(lat, lon):
    if lat < 30:   return lon < -117
    elif lat < 34: return lon < -118.5
    elif lat < 36: return lon < -120.5
    elif lat < 38: return lon < -122
    elif lat < 40: return lon < -123
    elif lat < 42: return lon < -124
    else:          return lon < -124.5


def load_cpl_flight(flight_key):
    config = FLIGHTS[flight_key]
    path = CPL_DATA_DIR / config["cpl"]
    with h5py.File(path, "r") as f:
        lat = f["geolocation/CPL_Latitude"][:, 0]
        lon = f["geolocation/CPL_Longitude"][:, 0]
        djd = f["layer_descriptor/Profile_Decimal_Julian_Day"][:, 0]
        cbh_km = f["layer_descriptor/Layer_Base_Altitude"][:, 0]
        sza = f["geolocation/Solar_Zenith_Angle"][:]
        saa = f["geolocation/Solar_Azimuth_Angle"][:]
    day_frac_hours = (djd % 1) * 24.0
    valid = (cbh_km > 0) & (cbh_km <= 2.0)
    ocean = np.array([is_ocean(lat[i], lon[i]) if valid[i] else False for i in range(len(lat))])
    mask = valid & ocean
    return {"lat": lat[mask], "lon": lon[mask], "day_frac": day_frac_hours[mask],
            "sza": sza[mask], "saa": saa[mask], "cbh_km": cbh_km[mask]}


def load_era5_for_flight(flight_key):
    config = FLIGHTS[flight_key]
    era5_path = ERA5_ROOT / f"era5_surface_{config['era5_date']}.nc"
    f = h5py.File(str(era5_path), "r")
    data = {"lat": f["latitude"][:], "lon": f["longitude"][:],
            "t2m": f["t2m"][:], "d2m": f["d2m"][:],
            "sp": f["sp"][:], "blh": f["blh"][:], "tcwv": f["tcwv"][:]}
    f.close()
    return data


def match_era5(cpl, era5, flight_key):
    n = len(cpl["lat"])
    features = np.zeros((n, 5))
    for i in range(n):
        lat_idx = np.argmin(np.abs(era5["lat"] - cpl["lat"][i]))
        lon_idx = np.argmin(np.abs(era5["lon"] - cpl["lon"][i]))
        hour_idx = min(int(cpl["day_frac"][i]) % 24, 23)
        features[i, 0] = era5["t2m"][hour_idx, lat_idx, lon_idx]
        features[i, 1] = era5["d2m"][hour_idx, lat_idx, lon_idx]
        features[i, 2] = era5["sp"][hour_idx, lat_idx, lon_idx]
        features[i, 3] = era5["blh"][hour_idx, lat_idx, lon_idx]
        features[i, 4] = era5["tcwv"][hour_idx, lat_idx, lon_idx]
    return features


def engineer_features(base, cpl):
    t2m, d2m, sp, blh, tcwv = base[:, 0], base[:, 1], base[:, 2], base[:, 3], base[:, 4]
    lat, lon = cpl["lat"], cpl["lon"]
    day_frac = cpl["day_frac"]
    derived = {}
    lcl = 125.0 * ((t2m - 273.15) - (d2m - 273.15))
    lcl = np.clip(lcl, 0, 15000)
    derived["lcl"] = lcl
    derived["lcl_deficit"] = blh - lcl
    e_s = 611.2 * np.exp(17.67 * (d2m - 273.15) / (d2m - 29.65))
    w = 0.622 * e_s / (sp - e_s)
    derived["t_virtual"] = t2m * (1 + 0.61 * w)
    derived["dewpoint_depression"] = t2m - d2m
    rh = np.exp(17.67 * (d2m - 273.15) / (d2m - 29.65)) / np.exp(17.67 * (t2m - 273.15) / (t2m - 29.65))
    derived["rh"] = np.clip(rh, 0, 1)
    derived["stability_index"] = (t2m - d2m) / 10.0
    derived["moisture_gradient"] = ((d2m - 250) / 30) + (tcwv / 30)
    derived["pressure_altitude"] = 44330 * (1 - (sp / 101325) ** 0.1903)
    theta = t2m * (100000 / sp) ** 0.286
    derived["theta_e"] = theta * np.exp(2.5e6 * w / (1004 * t2m))
    derived["blh_lcl_ratio"] = blh / (lcl + 1)
    derived["stability_tcwv"] = derived["stability_index"] * tcwv
    derived["dd_blh"] = derived["dewpoint_depression"] * blh / 1000
    derived["t2m_d2m_ratio"] = t2m / (d2m + 0.01)
    derived["inversion_strength"] = blh * derived["stability_index"]
    sza_rad = np.radians(cpl["sza"])
    derived["sza_cos"] = np.cos(sza_rad)
    derived["sza_sin"] = np.sin(sza_rad)
    saa_rad = np.radians(cpl["saa"])
    derived["saa_cos"] = np.cos(saa_rad)
    derived["saa_sin"] = np.sin(saa_rad)
    hour = day_frac
    derived["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    derived["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    derived["solar_heating_proxy"] = np.maximum(0, derived["sza_cos"]) * (t2m - 273.15)
    derived["t2m_tcwv"] = (t2m - 273.15) * tcwv
    derived["rh_blh"] = derived["rh"] * blh
    derived["lcl_sq"] = lcl ** 2 / 1e6
    derived["blh_sq"] = blh ** 2 / 1e6
    derived["t2m_sp"] = (t2m - 273.15) * sp / 1e5
    derived["lat_abs"] = np.abs(lat)
    derived["lon_abs"] = np.abs(lon)
    derived["lat_lon"] = lat * lon / 1000
    derived_names = list(derived.keys())
    derived_arr = np.column_stack([derived[k] for k in derived_names])
    full = np.hstack([base, derived_arr])
    full_names = BASE_FEATURE_NAMES + derived_names
    mask = np.all(np.isfinite(full), axis=1)
    return full, full_names, mask


def make_gbdt(**kwargs):
    defaults = dict(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    defaults.update(kwargs)
    return GradientBoostingRegressor(**defaults)


def run_adaptive_conformal(X_by_flight, y_by_flight, flight_keys, alpha=0.1, gamma=0.05):
    """
    Adaptive conformal with online quantile adjustment.
    
    For each target flight:
    1. Train GBDT on source flights (80%), calibrate on source (20%)
    2. Standard: fixed q from calibration residuals
    3. Adaptive: start from q, then adjust online as test samples arrive
       - If prediction covers true value: decrease q (tighten)
       - If prediction misses: increase q (widen)
       - Update rule: q_t+1 = q_t * exp(gamma * (alpha - miss_t))
    """
    print("\n=== Adaptive Conformal Prediction ===")
    results = {"standard": {}, "adaptive": {}}
    target_coverage = 1 - alpha

    for target in flight_keys:
        X_source = np.vstack([X_by_flight[k] for k in flight_keys if k != target])
        y_source = np.concatenate([y_by_flight[k] for k in flight_keys if k != target])
        X_target, y_target = X_by_flight[target], y_by_flight[target]

        scaler = StandardScaler()
        X_source_s = scaler.fit_transform(X_source)
        X_target_s = scaler.transform(X_target)

        # Train/cal split on source
        n_cal = len(X_source_s) // 5
        idx = np.random.permutation(len(X_source_s))
        cal_idx, train_idx = idx[:n_cal], idx[n_cal:]

        model = make_gbdt()
        model.fit(X_source_s[train_idx], y_source[train_idx])

        # Calibration residuals
        cal_resid = np.abs(y_source[cal_idx] - model.predict(X_source_s[cal_idx]))
        q_standard = np.quantile(cal_resid, target_coverage)

        # Test predictions
        y_pred = model.predict(X_target_s)
        test_resid = np.abs(y_target - y_pred)

        # Standard conformal
        coverage_std = float((test_resid <= q_standard).mean())
        width_std = float(2 * q_standard * 1000)
        results["standard"][target] = {"coverage": coverage_std, "width_m": width_std}

        # Adaptive conformal (online)
        q_t = q_standard  # start from standard quantile
        covered_count = 0
        for t in range(len(y_target)):
            is_covered = test_resid[t] <= q_t
            covered_count += int(is_covered)
            # Update: miss=1 if not covered, miss=0 if covered
            miss_t = 1 - int(is_covered)
            q_t = q_t * np.exp(gamma * (alpha - miss_t))
            # Clip to prevent degenerate quantiles
            q_t = np.clip(q_t, 0.001, 10.0)

        coverage_adapt = covered_count / len(y_target)
        results["adaptive"][target] = {"coverage": float(coverage_adapt), "final_q_km": float(q_t)}

        print(f"  {target}: standard={coverage_std:.1%}, adaptive={coverage_adapt:.1%} (final_q={q_t*1000:.0f}m)")

    # Means
    for method in ["standard", "adaptive"]:
        vals = [v["coverage"] for k, v in results[method].items() if k != "mean"]
        results[method]["mean_coverage"] = float(np.mean(vals))
    
    print(f"\n  Standard mean: {results['standard']['mean_coverage']:.1%}")
    print(f"  Adaptive mean: {results['adaptive']['mean_coverage']:.1%}")
    return results


def main():
    print("=" * 70)
    print("Adaptive Conformal Prediction Verification")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    X_by_flight, y_by_flight = {}, {}
    flight_keys = []

    for flight_key in FLIGHTS:
        print(f"\nLoading {flight_key}...")
        cpl = load_cpl_flight(flight_key)
        if cpl["cbh_km"].shape[0] == 0:
            continue
        era5 = load_era5_for_flight(flight_key)
        base = match_era5(cpl, era5, flight_key)
        full, names, mask = engineer_features(base, cpl)
        X_by_flight[flight_key] = full[mask]
        y_by_flight[flight_key] = cpl["cbh_km"][mask]
        flight_keys.append(flight_key)
        print(f"  {flight_key}: {mask.sum()} samples, {full.shape[1]} features")

    results = run_adaptive_conformal(X_by_flight, y_by_flight, flight_keys)

    # Also try different gamma values
    print("\n=== Sensitivity to gamma ===")
    for gamma in [0.01, 0.02, 0.05, 0.1, 0.2]:
        r = run_adaptive_conformal(X_by_flight, y_by_flight, flight_keys, gamma=gamma)
        print(f"  gamma={gamma}: adaptive mean coverage={r['adaptive']['mean_coverage']:.1%}")

    # Save results
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "purpose": "Verify/replace 11% adaptive conformal claim",
            "gamma_default": 0.05,
            "alpha": 0.1,
            "target_coverage": 0.9,
        },
        "results": results,
    }
    out_path = OUTPUT_DIR / "adaptive_conformal_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
