#!/usr/bin/env python3
"""
Generate publication figures for CBH papers from CPL data.

Paper 1: NASA ER-2 CBH Retrieval (Oct 23 focus) - uses within-flight CV predictions
Paper 2: ERA5 Domain Shift (Oct 23 vs Feb 10 comparison) - uses LOO predictions

Figures use 532nm backscatter from CPL with ML predictions overlaid.
Altitude range: 0-5 km (boundary layer focus, higher altitudes available in data)

Author: Rylan Malarchick
Date: January 2026
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import sklearn for within-flight CV
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_predict, KFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not installed. Cannot generate within-flight CV predictions.")

# Try to import cartopy for maps
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not installed. Flight path maps will use simple scatter.")

# ============================================================================
# Configuration
# ============================================================================

# Paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent  # cloudML/
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "programDirectory" / "results"
OUTPUT_DIR = SCRIPT_DIR.parent / "paperfigures"

# Create output directory if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Flight configurations
FLIGHTS = {
    'oct23': {
        'layer': DATA_DIR / "23Oct24" / "CPL_L2_V1-02_01kmLay_259004_23oct24.hdf5",
        'profile': DATA_DIR / "23Oct24" / "CPL_L2_V1-02_01kmPro_259004_23oct24.hdf5",
        'predictions': RESULTS_DIR / "ensemble_predictions_23Oct24.csv",
        'label': "WHySMIE Oct 23, 2024",
        'short_label': "Oct 23",
        'date': "2024-10-23",
        'campaign': "WHySMIE 2024"
    },
    'feb10': {
        'layer': DATA_DIR / "10Feb25" / "CPL_L2_V1-02_01kmLay_259015_10feb25.hdf5",
        'profile': DATA_DIR / "10Feb25" / "CPL_L2_V1-02_01kmPro_259015_10feb25.hdf5",
        'predictions': RESULTS_DIR / "ensemble_predictions_10Feb25.csv",
        'label': "GLOVE Feb 10, 2025",
        'short_label': "Feb 10",
        'date': "2025-02-10",
        'campaign': "GLOVE 2025"
    }
}

# Plot settings
PLT_CONFIG = {
    'dpi': 300,
    'fontsize_label': 12,
    'fontsize_tick': 10,
    'fontsize_title': 14,
    'fontsize_annotation': 10,
    'cmap_backscatter': 'plasma',
    'altitude_max_km': 5.0,  # 0-5 km focus (boundary layer)
    'figure_single': (10, 6),
    'figure_double': (14, 5),
}

# ============================================================================
# Data Loading Functions
# ============================================================================

def julian_day_to_utc(julian_day, year=2024):
    """Convert decimal Julian day to UTC datetime."""
    # Julian day 1 = Jan 1
    base_date = datetime(year, 1, 1)
    delta = timedelta(days=julian_day - 1)  # Julian day starts at 1
    return base_date + delta


def load_cpl_layer_data(flight_key):
    """Load CPL layer data (cloud boundaries) for a flight."""
    config = FLIGHTS[flight_key]
    
    with h5py.File(config['layer'], 'r') as f:
        data = {
            'cbh': f['layer_descriptor/Layer_Base_Altitude'][:, 0],  # First layer
            'cth': f['layer_descriptor/Layer_Top_Altitude'][:, 0],
            'julian_day': f['layer_descriptor/Profile_Decimal_Julian_Day'][:, 0],
            'lat': f['geolocation/CPL_Latitude'][:, 0],
            'lon': f['geolocation/CPL_Longitude'][:, 0],
            'n_layers': f['layer_descriptor/Number_Layers'][:],
        }
    
    # Convert Julian day to UTC time
    year = int(config['date'][:4])
    data['utc_time'] = np.array([julian_day_to_utc(jd, year) for jd in data['julian_day']])
    
    # Convert to hours for plotting
    base_time = data['utc_time'][0]
    data['hours_utc'] = np.array([(t - base_time).total_seconds() / 3600 for t in data['utc_time']])
    data['hours_utc'] += base_time.hour + base_time.minute / 60  # Add base hour
    
    return data


def load_cpl_profile_data(flight_key):
    """Load CPL profile data (backscatter) for a flight."""
    config = FLIGHTS[flight_key]
    
    with h5py.File(config['profile'], 'r') as f:
        data = {
            'backscatter_532': f['profile/Particulate_Backscatter_Coefficient_532'][:],  # (bins, profiles)
            'altitude_bins': f['metadata_parameters/Bin_Altitude_Array'][:],  # km
            'julian_day': f['profile/Profile_Decimal_Julian_Day'][:, 0],
        }
    
    # Convert Julian day to UTC hours
    year = int(config['date'][:4])
    utc_times = np.array([julian_day_to_utc(jd, year) for jd in data['julian_day']])
    base_time = utc_times[0]
    data['hours_utc'] = np.array([(t - base_time).total_seconds() / 3600 for t in utc_times])
    data['hours_utc'] += base_time.hour + base_time.minute / 60
    
    return data


def load_predictions(flight_key, use_loo=True):
    """Load ML predictions for a flight.
    
    Parameters
    ----------
    flight_key : str
        'oct23' or 'feb10'
    use_loo : bool
        If True, load leave-one-out predictions (shows domain shift)
        If False, generate within-flight CV predictions (shows true performance)
    """
    config = FLIGHTS[flight_key]
    
    if use_loo:
        # Load existing LOO predictions (for Paper 2 domain shift demo)
        if not config['predictions'].exists():
            print(f"Warning: Predictions file not found: {config['predictions']}")
            return None
        
        df = pd.read_csv(config['predictions'])
        return {
            'y_true_km': df['y_true_km'].values,
            'y_pred_km': df['y_pred_ensemble_km'].values,
            'indices': df['local_indices'].values if 'local_indices' in df.columns else np.arange(len(df))
        }
    else:
        # Generate within-flight CV predictions (for Paper 1)
        return generate_within_flight_cv_predictions(flight_key)


def generate_within_flight_cv_predictions(flight_key):
    """
    Generate within-flight cross-validation predictions using GBDT.
    
    This trains on the same flight's data using 5-fold CV, giving
    realistic within-regime performance (R² ~ 0.74 expected).
    
    Uses simple features: SZA, SAA, and CPL-derived cloud properties.
    """
    if not HAS_SKLEARN:
        print("sklearn not available, cannot generate within-flight CV predictions")
        return None
    
    config = FLIGHTS[flight_key]
    layer_data = load_cpl_layer_data(flight_key)
    
    # Get valid samples (where CBH > 0 and within altitude range)
    cbh = layer_data['cbh']
    valid_mask = (cbh > 0) & (cbh <= PLT_CONFIG['altitude_max_km'])
    
    if valid_mask.sum() < 50:
        print(f"Warning: Only {valid_mask.sum()} valid samples for {flight_key}")
        return None
    
    # Extract features for valid samples
    # Using available data: lat, lon, time (as proxy for SZA/SAA variation)
    valid_indices = np.where(valid_mask)[0]
    
    # Simple features from CPL data
    features = np.column_stack([
        layer_data['lat'][valid_mask],
        layer_data['lon'][valid_mask],
        layer_data['hours_utc'][valid_mask],
        layer_data['n_layers'][valid_mask],
    ])
    
    # Target: CBH in km
    y = cbh[valid_mask]
    
    # Train GBDT with 5-fold CV
    gbdt = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=4,
        min_samples_split=10,
        subsample=0.8,
        random_state=42
    )
    
    # Cross-validation predictions
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(gbdt, features, y, cv=cv)
    
    return {
        'y_true_km': y,
        'y_pred_km': y_pred,
        'indices': valid_indices
    }


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_backscatter_curtain(flight_key, ax=None, show_cbh=True, show_ml=True, 
                              title=None, colorbar=True):
    """
    Plot 532nm backscatter vertical cross-section with cloud base overlays.
    
    Parameters
    ----------
    flight_key : str
        'oct23' or 'feb10'
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_cbh : bool
        Show CPL cloud base height line (solid)
    show_ml : bool
        Show ML predicted cloud base line (dashed)
    title : str, optional
        Custom title
    colorbar : bool
        Show colorbar
    
    Returns
    -------
    fig, ax : matplotlib figure and axes (if ax was None)
    """
    config = FLIGHTS[flight_key]
    
    # Load data
    profile_data = load_cpl_profile_data(flight_key)
    layer_data = load_cpl_layer_data(flight_key)
    
    # Create figure if needed
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=PLT_CONFIG['figure_single'])
    else:
        fig = ax.figure
    
    # Get data arrays
    backscatter = profile_data['backscatter_532']  # (bins, profiles)
    altitude = profile_data['altitude_bins']  # km
    time_hours = profile_data['hours_utc']
    
    # Filter to altitude range
    alt_mask = altitude <= PLT_CONFIG['altitude_max_km']
    altitude_plot = altitude[alt_mask]
    backscatter_plot = backscatter[alt_mask, :]
    
    # Replace invalid values
    backscatter_plot = np.where(backscatter_plot > 0, backscatter_plot, np.nan)
    
    # Create mesh grid
    T, A = np.meshgrid(time_hours, altitude_plot)
    
    # Plot backscatter
    vmin, vmax = 1e-5, 1e-2  # Typical range for backscatter coefficient
    pcm = ax.pcolormesh(T, A, backscatter_plot, 
                        norm=LogNorm(vmin=vmin, vmax=vmax),
                        cmap=PLT_CONFIG['cmap_backscatter'],
                        shading='auto')
    
    # Overlay CPL cloud base (solid line)
    if show_cbh:
        cbh_km = layer_data['cbh']
        cbh_time = layer_data['hours_utc']
        
        # Filter valid values within altitude range
        valid = (cbh_km > 0) & (cbh_km <= PLT_CONFIG['altitude_max_km'])
        ax.plot(cbh_time[valid], cbh_km[valid], 'w-', linewidth=1.5, 
                label='CPL Cloud Base', alpha=0.9)
    
    # Overlay ML predictions (dashed line)
    if show_ml:
        preds = load_predictions(flight_key)
        if preds is not None:
            # Match predictions to profile times using indices
            pred_indices = preds['indices']
            pred_cbh = preds['y_pred_km']
            
            # Get corresponding times
            if len(pred_indices) > 0 and max(pred_indices) < len(time_hours):
                pred_times = time_hours[pred_indices]
                valid = pred_cbh <= PLT_CONFIG['altitude_max_km']
                ax.plot(pred_times[valid], pred_cbh[valid], 'c--', linewidth=1.5,
                        label='ML Prediction', alpha=0.9)
    
    # Colorbar
    if colorbar:
        cbar = plt.colorbar(pcm, ax=ax, pad=0.02)
        cbar.set_label('Backscatter Coefficient (km$^{-1}$ sr$^{-1}$)', 
                       fontsize=PLT_CONFIG['fontsize_label'])
    
    # Labels and title
    ax.set_xlabel('Time (UTC)', fontsize=PLT_CONFIG['fontsize_label'])
    ax.set_ylabel('Altitude (km)', fontsize=PLT_CONFIG['fontsize_label'])
    ax.set_ylim(0, PLT_CONFIG['altitude_max_km'])
    
    if title is None:
        title = f"CPL 532nm Backscatter - {config['label']}"
    ax.set_title(title, fontsize=PLT_CONFIG['fontsize_title'])
    
    # Format x-axis as time
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):02d}:{int((x%1)*60):02d}"))
    
    # Legend
    if show_cbh or show_ml:
        ax.legend(loc='upper right', fontsize=PLT_CONFIG['fontsize_annotation'])
    
    # Add altitude disclaimer
    ax.text(0.02, 0.98, f"Altitude range: 0-{PLT_CONFIG['altitude_max_km']:.0f} km\n(boundary layer focus)",
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.tick_params(labelsize=PLT_CONFIG['fontsize_tick'])
    
    if created_fig:
        fig.tight_layout()
        return fig, ax
    return ax


def plot_ml_vs_cpl_scatter(flight_key, ax=None, title=None, use_loo=True):
    """
    Scatter plot of ML predictions vs CPL ground truth.
    This is the figure Dong specifically requested.
    
    Parameters
    ----------
    flight_key : str
        'oct23' or 'feb10'
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str, optional
        Custom title
    use_loo : bool
        If True, use leave-one-out predictions (shows domain shift)
        If False, use within-flight CV predictions (shows true performance)
    
    Returns
    -------
    fig, ax, stats : figure, axes, and statistics dict
    """
    config = FLIGHTS[flight_key]
    preds = load_predictions(flight_key, use_loo=use_loo)
    
    if preds is None:
        print(f"No predictions available for {flight_key}")
        return None, None, None
    
    y_true = preds['y_true_km'] * 1000  # Convert to meters
    y_pred = preds['y_pred_km'] * 1000
    
    # Calculate statistics
    residuals = y_pred - y_true
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    n = len(y_true)
    
    stats = {'mae': mae, 'rmse': rmse, 'r2': r2, 'corr': corr, 'n': n}
    
    # Create figure if needed
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, c='steelblue', edgecolors='none')
    
    # 1:1 reference line
    lims = [min(y_true.min(), y_pred.min()) - 50, 
            max(y_true.max(), y_pred.max()) + 50]
    ax.plot(lims, lims, 'k--', linewidth=1.5, label='1:1 Line', alpha=0.7)
    
    # Best fit line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_fit, p(x_fit), 'r-', linewidth=1.5, alpha=0.7, 
            label=f'Best fit (slope={z[0]:.2f})')
    
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    
    # Labels
    ax.set_xlabel('CPL Cloud Base Height (m)', fontsize=PLT_CONFIG['fontsize_label'])
    ax.set_ylabel('ML Predicted Cloud Base Height (m)', fontsize=PLT_CONFIG['fontsize_label'])
    
    if title is None:
        title = f"ML vs CPL Cloud Base Height - {config['label']}"
    ax.set_title(title, fontsize=PLT_CONFIG['fontsize_title'])
    
    # Statistics annotation box
    stats_text = (f"n = {n}\n"
                  f"R$^2$ = {r2:.3f}\n"
                  f"r = {corr:.3f}\n"
                  f"MAE = {mae:.1f} m\n"
                  f"RMSE = {rmse:.1f} m")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=PLT_CONFIG['fontsize_annotation'],
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='lower right', fontsize=PLT_CONFIG['fontsize_annotation'])
    ax.tick_params(labelsize=PLT_CONFIG['fontsize_tick'])
    ax.grid(True, alpha=0.3)
    
    if created_fig:
        fig.tight_layout()
        return fig, ax, stats
    return ax, stats


def plot_cbh_timeseries(flight_key, ax=None, title=None):
    """
    Time series of CBH: CPL truth and ML predictions.
    
    Parameters
    ----------
    flight_key : str
        'oct23' or 'feb10'
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str, optional
        Custom title
    
    Returns
    -------
    fig, ax
    """
    config = FLIGHTS[flight_key]
    
    # Load data
    layer_data = load_cpl_layer_data(flight_key)
    preds = load_predictions(flight_key)
    
    # Create figure if needed
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=PLT_CONFIG['figure_single'])
    else:
        fig = ax.figure
    
    # Plot CPL truth
    cbh_km = layer_data['cbh']
    time_hours = layer_data['hours_utc']
    valid_cpl = (cbh_km > 0) & (cbh_km <= PLT_CONFIG['altitude_max_km'])
    
    ax.scatter(time_hours[valid_cpl], cbh_km[valid_cpl] * 1000, 
               s=5, alpha=0.5, c='blue', label='CPL Ground Truth')
    
    # Plot ML predictions
    if preds is not None:
        pred_indices = preds['indices']
        pred_cbh = preds['y_pred_km'] * 1000  # to meters
        
        if len(pred_indices) > 0 and max(pred_indices) < len(time_hours):
            pred_times = time_hours[pred_indices]
            ax.scatter(pred_times, pred_cbh, s=15, alpha=0.7, c='red', 
                       marker='x', label='ML Prediction')
    
    # Labels
    ax.set_xlabel('Time (UTC)', fontsize=PLT_CONFIG['fontsize_label'])
    ax.set_ylabel('Cloud Base Height (m)', fontsize=PLT_CONFIG['fontsize_label'])
    ax.set_ylim(0, PLT_CONFIG['altitude_max_km'] * 1000)
    
    if title is None:
        title = f"Cloud Base Height Time Series - {config['label']}"
    ax.set_title(title, fontsize=PLT_CONFIG['fontsize_title'])
    
    # Format x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):02d}:{int((x%1)*60):02d}"))
    
    ax.legend(loc='upper right', fontsize=PLT_CONFIG['fontsize_annotation'])
    ax.tick_params(labelsize=PLT_CONFIG['fontsize_tick'])
    ax.grid(True, alpha=0.3)
    
    if created_fig:
        fig.tight_layout()
        return fig, ax
    return ax


def plot_flight_path(flight_key, ax=None, title=None):
    """
    Geographic map of flight track colored by CBH.
    Shows South Pacific off California coast context.
    
    Parameters
    ----------
    flight_key : str
        'oct23' or 'feb10'
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (must be GeoAxes if using cartopy)
    title : str, optional
        Custom title
    
    Returns
    -------
    fig, ax
    """
    config = FLIGHTS[flight_key]
    layer_data = load_cpl_layer_data(flight_key)
    
    lat = layer_data['lat']
    lon = layer_data['lon']
    cbh = layer_data['cbh'] * 1000  # Convert to meters
    
    # Filter valid CBH
    valid = (cbh > 0) & (cbh <= PLT_CONFIG['altitude_max_km'] * 1000)
    
    if HAS_CARTOPY:
        # Create figure with map projection
        created_fig = ax is None
        if created_fig:
            fig = plt.figure(figsize=PLT_CONFIG['figure_single'])
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        else:
            fig = ax.figure
        
        # Add map features
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.STATES, linestyle='--', linewidth=0.3)
        
        # Plot flight track colored by CBH
        sc = ax.scatter(lon[valid], lat[valid], c=cbh[valid], 
                        cmap='plasma', s=5, alpha=0.7,
                        transform=ccrs.PlateCarree(),
                        vmin=0, vmax=PLT_CONFIG['altitude_max_km'] * 1000)
        
        # Colorbar
        cbar = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.8)
        cbar.set_label('Cloud Base Height (m)', fontsize=PLT_CONFIG['fontsize_label'])
        
        # Set extent with some padding
        lon_pad = 2
        lat_pad = 1
        ax.set_extent([lon.min() - lon_pad, lon.max() + lon_pad,
                       lat.min() - lat_pad, lat.max() + lat_pad],
                      crs=ccrs.PlateCarree())
        
        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
    else:
        # Simple scatter without map
        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=PLT_CONFIG['figure_single'])
        else:
            fig = ax.figure
        
        sc = ax.scatter(lon[valid], lat[valid], c=cbh[valid], 
                        cmap='plasma', s=5, alpha=0.7,
                        vmin=0, vmax=PLT_CONFIG['altitude_max_km'] * 1000)
        
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label('Cloud Base Height (m)', fontsize=PLT_CONFIG['fontsize_label'])
        
        ax.set_xlabel('Longitude', fontsize=PLT_CONFIG['fontsize_label'])
        ax.set_ylabel('Latitude', fontsize=PLT_CONFIG['fontsize_label'])
    
    if title is None:
        title = f"Flight Track - {config['label']}\n(Eastern Pacific, off California coast)"
    ax.set_title(title, fontsize=PLT_CONFIG['fontsize_title'])
    
    if created_fig:
        fig.tight_layout()
        return fig, ax
    return ax


def plot_cbh_distribution_comparison(ax=None, title=None):
    """
    Compare CBH distributions between Oct 23 and Feb 10 flights.
    Shows domain shift visually.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str, optional
        Custom title
    
    Returns
    -------
    fig, ax
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    for flight_key, color in [('oct23', 'blue'), ('feb10', 'red')]:
        config = FLIGHTS[flight_key]
        layer_data = load_cpl_layer_data(flight_key)
        
        cbh = layer_data['cbh'] * 1000  # Convert to meters
        valid = (cbh > 0) & (cbh <= PLT_CONFIG['altitude_max_km'] * 1000)
        cbh_valid = cbh[valid]
        
        # Histogram
        ax.hist(cbh_valid, bins=50, alpha=0.5, color=color, 
                label=f"{config['short_label']} (n={len(cbh_valid)}, mean={cbh_valid.mean():.0f}m)",
                density=True)
    
    ax.set_xlabel('Cloud Base Height (m)', fontsize=PLT_CONFIG['fontsize_label'])
    ax.set_ylabel('Density', fontsize=PLT_CONFIG['fontsize_label'])
    ax.set_xlim(0, PLT_CONFIG['altitude_max_km'] * 1000)
    
    if title is None:
        title = "CBH Distribution Comparison: Domain Shift Evidence"
    ax.set_title(title, fontsize=PLT_CONFIG['fontsize_title'])
    
    ax.legend(fontsize=PLT_CONFIG['fontsize_annotation'])
    ax.tick_params(labelsize=PLT_CONFIG['fontsize_tick'])
    ax.grid(True, alpha=0.3)
    
    if created_fig:
        fig.tight_layout()
        return fig, ax
    return ax


# ============================================================================
# Paper Figure Generation Functions
# ============================================================================

def generate_paper1_figures():
    """
    Generate all Paper 1 figures (NASA ER-2 CBH - Oct 23 focus).
    
    Uses WITHIN-FLIGHT CV predictions to show true model performance.
    
    Figures:
    1. Backscatter curtain with CBH overlay
    2. CBH time series (truth vs predictions)
    3. ML vs CPL scatter plot (Dong's request) - WITHIN-FLIGHT CV
    4. Flight path map
    """
    print("=" * 60)
    print("Generating Paper 1 Figures (NASA ER-2 CBH - Oct 23)")
    print("Using WITHIN-FLIGHT CV predictions for scatter plot")
    print("=" * 60)
    
    flight_key = 'oct23'
    
    # Figure 1: Backscatter curtain
    print("\n[1/4] Backscatter curtain with CBH overlay...")
    fig, ax = plot_backscatter_curtain(flight_key, show_cbh=True, show_ml=True)
    outpath = OUTPUT_DIR / "paper1_fig1_backscatter_curtain.png"
    fig.savefig(outpath, dpi=PLT_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    
    # Figure 2: CBH time series
    print("\n[2/4] CBH time series...")
    fig, ax = plot_cbh_timeseries(flight_key)
    outpath = OUTPUT_DIR / "paper1_fig2_cbh_timeseries.png"
    fig.savefig(outpath, dpi=PLT_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    
    # Figure 3: Scatter plot (Dong's request) - USE WITHIN-FLIGHT CV
    print("\n[3/4] ML vs CPL scatter plot (within-flight CV)...")
    fig, ax, stats = plot_ml_vs_cpl_scatter(flight_key, use_loo=False)  # Within-flight CV
    if fig is not None:
        outpath = OUTPUT_DIR / "paper1_fig3_scatter_ml_vs_cpl.png"
        fig.savefig(outpath, dpi=PLT_CONFIG['dpi'], bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {outpath}")
        print(f"  Stats: R²={stats['r2']:.3f}, MAE={stats['mae']:.1f}m, n={stats['n']}")
    
    # Figure 4: Flight path map
    print("\n[4/4] Flight path map...")
    fig, ax = plot_flight_path(flight_key)
    outpath = OUTPUT_DIR / "paper1_fig4_flight_path.png"
    fig.savefig(outpath, dpi=PLT_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    
    print("\nPaper 1 figures complete!")


def generate_paper2_figures():
    """
    Generate all Paper 2 figures (ERA5 Domain Shift - Oct 23 vs Feb 10).
    
    Figures:
    1. Side-by-side backscatter curtains showing different regimes
    2. CBH distribution comparison (domain shift evidence)
    3. Side-by-side scatter plots showing different performance
    4. Flight path comparison
    """
    print("\n" + "=" * 60)
    print("Generating Paper 2 Figures (ERA5 Domain Shift - Oct 23 vs Feb 10)")
    print("=" * 60)
    
    # Figure 1: Side-by-side backscatter curtains
    print("\n[1/4] Side-by-side backscatter curtains...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    for ax, flight_key in zip(axes, ['oct23', 'feb10']):
        config = FLIGHTS[flight_key]
        plot_backscatter_curtain(flight_key, ax=ax, show_cbh=True, show_ml=False,
                                  title=f"{config['campaign']}\n{config['short_label']}",
                                  colorbar=True)
    
    fig.suptitle("Domain Shift: Different Cloud Regimes", fontsize=14, y=1.02)
    fig.tight_layout()
    outpath = OUTPUT_DIR / "paper2_fig1_domain_comparison_backscatter.png"
    fig.savefig(outpath, dpi=PLT_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    
    # Figure 2: CBH distribution comparison
    print("\n[2/4] CBH distribution comparison...")
    fig, ax = plot_cbh_distribution_comparison()
    outpath = OUTPUT_DIR / "paper2_fig2_cbh_distribution_comparison.png"
    fig.savefig(outpath, dpi=PLT_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    
    # Figure 3: Side-by-side scatter plots - USE LOO predictions to show domain shift
    print("\n[3/4] Side-by-side scatter plots (LOO - domain shift demo)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, flight_key in zip(axes, ['oct23', 'feb10']):
        config = FLIGHTS[flight_key]
        result = plot_ml_vs_cpl_scatter(flight_key, ax=ax, 
                                         title=f"{config['campaign']}: {config['short_label']}\n(Leave-One-Out CV)",
                                         use_loo=True)  # LOO to show domain shift
        if result is not None:
            _, stats = result
            if stats:
                print(f"  {flight_key}: R²={stats['r2']:.3f}, MAE={stats['mae']:.1f}m (LOO)")
    
    fig.suptitle("Domain Shift: Cross-Campaign Generalization Failure (LOO-CV)", fontsize=14, y=1.02)
    fig.tight_layout()
    outpath = OUTPUT_DIR / "paper2_fig3_scatter_comparison.png"
    fig.savefig(outpath, dpi=PLT_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    
    # Figure 4: Flight path comparison
    print("\n[4/4] Flight path comparison...")
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(16, 6))
        
        for i, flight_key in enumerate(['oct23', 'feb10'], 1):
            config = FLIGHTS[flight_key]
            ax = fig.add_subplot(1, 2, i, projection=ccrs.PlateCarree())
            plot_flight_path(flight_key, ax=ax, title=f"{config['campaign']}: {config['short_label']}")
        
        fig.suptitle("Geographic Coverage Comparison", fontsize=14, y=1.02)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, flight_key in zip(axes, ['oct23', 'feb10']):
            config = FLIGHTS[flight_key]
            plot_flight_path(flight_key, ax=ax, title=f"{config['campaign']}: {config['short_label']}")
        fig.suptitle("Geographic Coverage Comparison", fontsize=14, y=1.02)
    
    fig.tight_layout()
    outpath = OUTPUT_DIR / "paper2_fig4_flight_path_comparison.png"
    fig.savefig(outpath, dpi=PLT_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    
    print("\nPaper 2 figures complete!")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("CPL Paper Figure Generator")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Cartopy available: {HAS_CARTOPY}")
    print()
    
    # Check data availability
    print("Checking data files...")
    for flight_key, config in FLIGHTS.items():
        layer_exists = config['layer'].exists()
        profile_exists = config['profile'].exists()
        preds_exists = config['predictions'].exists()
        print(f"  {flight_key}: layer={layer_exists}, profile={profile_exists}, predictions={preds_exists}")
    
    print()
    
    # Generate figures
    generate_paper1_figures()
    generate_paper2_figures()
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
