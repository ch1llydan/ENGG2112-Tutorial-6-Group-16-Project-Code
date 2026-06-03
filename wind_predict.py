"""
Wind Speed Predictor — New England NSW
========================================
Loads the trained XGBoost model and predicts hourly wind speeds
for any specified date or date range.

Produces:
    wind_forecast.csv  — hourly predicted wind speeds, ready for
                         use in the wind turbine power generation script

Usage:
    Set FORECAST_START and FORECAST_END in USER INPUTS, then run:
        python wind_predict.py

Requirements:
    pip install xgboost pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb #type:ignore
import warnings
from config import FORECAST_DATE_STR, FORECAST_YEAR # type: ignore

warnings.filterwarnings('ignore')

# =============================================================
# USER INPUTS
# =============================================================

MODEL_FILE      = 'wind_model.json'     # trained model from wind_ml.py
HISTORY_FILE    = 'wind_raw.csv'        # ERA5 historical data — needed for lag features

# Date range to forecast
# Can be a single day or a multi-day range
# Must be within or adjacent to the historical data range

FORECAST_START   = FORECAST_DATE_STR
FORECAST_END     = FORECAST_DATE_STR
FORECAST_HORIZON = 1

# Output
OUTPUT_FILE     = 'wind_forecast.csv'
PLOT_FILE       = 'wind_forecast_plot.png'
SHOW_PLOT       = False

# =============================================================
# STEP 1 — Load model and historical data
# =============================================================

print("\n" + "="*60)
print("  Wind Speed Predictor — New England NSW")
print("="*60)

print(f"\n[1/4] Loading model and historical data...")

model = xgb.XGBRegressor()
model.load_model(MODEL_FILE)
print(f"  Model loaded  : {MODEL_FILE}")

df_hist = pd.read_csv(HISTORY_FILE)
df_hist['datetime'] = pd.to_datetime(df_hist['datetime'], dayfirst=True)
df_hist = df_hist.sort_values('datetime').reset_index(drop=True)
print(f"  History loaded: {len(df_hist):,} rows  "
      f"({df_hist['datetime'].min().date()} to {df_hist['datetime'].max().date()})")

# =============================================================
# STEP 2 — Build feature set for the forecast window
# =============================================================

print(f"\n[2/4] Building features for forecast window...")

# Parse forecast dates
forecast_start  = pd.Timestamp(FORECAST_START)
forecast_end    = pd.Timestamp(FORECAST_END) + pd.Timedelta(hours=23)

print(f"  Forecast window : {forecast_start.date()} to {forecast_end.date()}")

# To compute lag features for the forecast window we need history
# going back at least 48 hours before the start
lookback_start  = forecast_start - pd.Timedelta(hours=48)

# Extract the relevant slice of history plus the forecast window
df_window = df_hist[df_hist['datetime'] >= lookback_start].copy()

# Check we have enough history
if len(df_window) < 48:
    raise ValueError(
        f"Not enough historical data before {forecast_start.date()} to "
        f"compute lag features. Need at least 48 hours of prior data."
    )

# Check forecast dates exist in history (ERA5 data covers up to end of 2025)
forecast_rows = df_window[
    (df_window['datetime'] >= forecast_start) &
    (df_window['datetime'] <= forecast_end)
]
if len(forecast_rows) == 0:
    raise ValueError(
        f"No data found for {forecast_start.date()} to {forecast_end.date()} "
        f"in {HISTORY_FILE}. Check your date range is within the ERA5 data coverage."
    )

# =============================================================
# STEP 3 — Feature engineering (must match wind_ml.py exactly)
# =============================================================

df = df_window.copy()

# Cyclical time features
df['hour']          = df['datetime'].dt.hour
df['day_of_year']   = df['datetime'].dt.dayofyear
df['month']         = df['datetime'].dt.month
df['day_of_week']   = df['datetime'].dt.dayofweek

df['hour_sin']      = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos']      = np.cos(2 * np.pi * df['hour'] / 24)
df['doy_sin']       = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['doy_cos']       = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
df['month_sin']     = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos']     = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin']       = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos']       = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Wind direction components
df['wind_dir_sin']  = np.sin(np.radians(
    (180 + np.degrees(np.arctan2(df['u100'], df['v100']))) % 360
))
df['wind_dir_cos']  = np.cos(np.radians(
    (180 + np.degrees(np.arctan2(df['u100'], df['v100']))) % 360
))

# Lag features
lag_hours = [1, 2, 3, 4, 6, 12, 24, 48]
for lag in lag_hours:
    df[f'wind_lag_{lag}h']  = df['wind_speed_ms'].shift(lag)
    df[f'u100_lag_{lag}h']  = df['u100'].shift(lag)
    df[f'v100_lag_{lag}h']  = df['v100'].shift(lag)

# Rolling statistics
for window in [3, 6, 12, 24]:
    df[f'wind_roll_{window}h_mean'] = df['wind_speed_ms'].rolling(window).mean()
    df[f'wind_roll_{window}h_std']  = df['wind_speed_ms'].rolling(window).std()
    df[f'wind_roll_{window}h_max']  = df['wind_speed_ms'].rolling(window).max()

# Rate of change
df['wind_change_1h']  = df['wind_speed_ms'].diff(1)
df['wind_change_3h']  = df['wind_speed_ms'].diff(3)
df['wind_change_6h']  = df['wind_speed_ms'].diff(6)
df['wind_change_24h'] = df['wind_speed_ms'].diff(24)

# Drop NaN rows from lag computation
df = df.dropna().reset_index(drop=True)

# Filter to forecast window only
forecast_df = df[
    (df['datetime'] >= forecast_start) &
    (df['datetime'] <= forecast_end)
].copy()

print(f"  Forecast rows   : {len(forecast_df)}")

# =============================================================
# STEP 4 — Generate predictions
# =============================================================

print(f"\n[3/4] Generating predictions...")

FEATURE_COLS = [
    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
    'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    'u100', 'v100', 'wind_dir_sin', 'wind_dir_cos',
    *[f'wind_lag_{h}h'  for h in lag_hours],
    *[f'u100_lag_{h}h'  for h in lag_hours],
    *[f'v100_lag_{h}h'  for h in lag_hours],
    *[f'wind_roll_{w}h_mean' for w in [3, 6, 12, 24]],
    *[f'wind_roll_{w}h_std'  for w in [3, 6, 12, 24]],
    *[f'wind_roll_{w}h_max'  for w in [3, 6, 12, 24]],
    'wind_change_1h', 'wind_change_3h', 'wind_change_6h', 'wind_change_24h',
]

X_forecast = forecast_df[FEATURE_COLS]
predicted   = model.predict(X_forecast)
predicted   = np.maximum(predicted, 0)   # wind speed cannot be negative

# =============================================================
# STEP 5 — Build output DataFrame
# =============================================================

# The prediction is for FORECAST_HORIZON hours ahead of each row
# Shift timestamps forward so the datetime reflects when the wind occurs
output = pd.DataFrame({
    'datetime'          : forecast_df['datetime'].values + pd.Timedelta(hours=FORECAST_HORIZON),
    'wind_speed_actual_ms'  : forecast_df['wind_speed_ms'].values,
    'wind_speed_predicted_ms': predicted,
    'u100'              : forecast_df['u100'].values,
    'v100'              : forecast_df['v100'].values,
})

output['hour']   = pd.to_datetime(output['datetime']).dt.hour
output['date']   = pd.to_datetime(output['datetime']).dt.date

# Summary statistics
print(f"\n  Forecast summary:")
print(f"  {'Hour':<6} {'Predicted (m/s)':<20} {'Actual (m/s)':<20}")
print(f"  {'-'*46}")
for _, row in output.iterrows():
    marker = ' ◄' if abs(row['wind_speed_predicted_ms'] - row['wind_speed_actual_ms']) > 2 else ''
    print(f"  {int(row['hour']):02d}:00  "
          f"{row['wind_speed_predicted_ms']:>8.2f}             "
          f"{row['wind_speed_actual_ms']:>8.2f}{marker}")

mae = np.mean(np.abs(output['wind_speed_predicted_ms'] - output['wind_speed_actual_ms']))
print(f"\n  MAE for this period : {mae:.3f} m/s")

# =============================================================
# STEP 6 — Save output CSV
# =============================================================

print(f"\n[4/4] Saving outputs...")

# Output CSV formatted for use in wind turbine power generation script
# Contains all columns needed for hub height scaling and power curve application
output_save = output[[
    'datetime',
    'wind_speed_predicted_ms',  # primary input for power curve
    'wind_speed_actual_ms',     # for validation comparison
    'u100',                     # U component at 100m
    'v100',                     # V component at 100m
]].copy()

output_save.to_csv(OUTPUT_FILE, index=False, float_format='%.4f')
print(f"  Forecast saved  : {OUTPUT_FILE}")
print(f"  Columns         : datetime, wind_speed_predicted_ms, "
      f"wind_speed_actual_ms, u100, v100")

# =============================================================
# PLOTS
# =============================================================

n_days = (forecast_end.date() - forecast_start.date()).days + 1

times = pd.to_datetime(output['datetime'])

# =============================================================
# FIGURE 1 — FULL FORECAST OVERVIEW
# =============================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# White background
fig.patch.set_facecolor('white')

for ax in axes:

    ax.set_facecolor('white')

    ax.tick_params(
        colors='black',
        labelsize=8
    )

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.title.set_color('black')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    ax.grid(
        True,
        color='lightgray',
        linewidth=0.6,
        linestyle='--',
        alpha=0.7
    )

date_label = (
    str(forecast_start.date())
    if n_days == 1
    else f"{forecast_start.date()} to {forecast_end.date()}"
)

fig.suptitle(
    f'Wind Speed Forecast — New England NSW\n{date_label}',
    fontsize=13,
    color='black',
    fontweight='bold',
    y=0.98
)

# -------------------------------------------------------------
# PANEL 1 — PREDICTED VS ACTUAL WIND SPEED
# -------------------------------------------------------------

ax = axes[0]

ax.fill_between(
    times,
    output['wind_speed_actual_ms'],
    alpha=0.2,
    color='forestgreen'
)

ax.plot(
    times,
    output['wind_speed_actual_ms'],
    color='forestgreen',
    linewidth=1.8,
    label='Actual wind speed'
)

ax.plot(
    times,
    output['wind_speed_predicted_ms'],
    color='darkorange',
    linewidth=1.5,
    linestyle='--',
    label='Predicted wind speed'
)

# Turbine operating regions

ax.axhline(
    3,
    color='navy',
    linewidth=1,
    linestyle='--',
    alpha=0.8,
    label='Cut-in speed (3 m/s)'
)

ax.axhline(
    9,
    color='gold',
    linewidth=1,
    linestyle='--',
    alpha=0.8,
    label='Goldwind rated (9 m/s)'
)

ax.axhline(
    13,
    color='orange',
    linewidth=1,
    linestyle='--',
    alpha=0.8,
    label='Vestas rated (13 m/s)'
)

ax.axhline(
    22,
    color='red',
    linewidth=1,
    linestyle='--',
    alpha=0.8,
    label='Cut-out speed (22 m/s)'
)

ax.set_ylabel('Wind speed (m/s)')

ax.set_title(
    'Predicted vs Actual Wind Speed'
)

ax.legend(
    fontsize=7,
    framealpha=0.9,
    facecolor='white',
    edgecolor='black',
    loc='upper right',
    ncol=2
)

ax.set_ylim(bottom=0)

# -------------------------------------------------------------
# PANEL 2 — PREDICTION ERROR
# -------------------------------------------------------------

ax = axes[1]

error = (
    output['wind_speed_predicted_ms']
    - output['wind_speed_actual_ms']
)

ax.fill_between(
    times,
    error,
    0,
    where=(error >= 0),
    alpha=0.4,
    color='orange',
    label='Overprediction'
)

ax.fill_between(
    times,
    error,
    0,
    where=(error < 0),
    alpha=0.4,
    color='skyblue',
    label='Underprediction'
)

ax.plot(
    times,
    error,
    color='black',
    linewidth=1.0,
    alpha=0.9
)

ax.axhline(
    0,
    color='black',
    linewidth=0.8,
    linestyle='--'
)

ax.set_ylabel('Error (m/s)')
ax.set_xlabel('Time')

ax.set_title(
    f'Prediction Error  (MAE = {mae:.3f} m/s)'
)

ax.legend(
    fontsize=7.5,
    framealpha=0.9,
    facecolor='white',
    edgecolor='black'
)

# -------------------------------------------------------------
# X AXIS FORMATTING
# -------------------------------------------------------------

if n_days == 1:

    axes[1].xaxis.set_major_locator(
        mdates.HourLocator(interval=2)
    )

    axes[1].xaxis.set_major_formatter(
        mdates.DateFormatter('%H:%M')
    )

    axes[1].set_xlabel('Hour of day')

else:

    axes[1].xaxis.set_major_locator(
        mdates.DayLocator(interval=1)
    )

    axes[1].xaxis.set_major_formatter(
        mdates.DateFormatter('%d %b')
    )

    axes[1].set_xlabel('Date')

    # Day boundary lines

    for d in pd.date_range(
        forecast_start,
        forecast_end,
        freq='D'
    ):

        for ax in axes:

            ax.axvline(
                d,
                color='gray',
                linewidth=0.7,
                alpha=0.6
            )

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=150,
    bbox_inches='tight',
    facecolor='white'
)

print(f"  Full forecast plot saved : {PLOT_FILE}")

if SHOW_PLOT:
    plt.show()

plt.close()


# =============================================================
# FIGURE 2 — WIND SPEED COMPARISON ONLY
# =============================================================

fig, ax = plt.subplots(figsize=(14, 5))

fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.grid(
    True,
    color='lightgray',
    linestyle='--',
    linewidth=0.6
)

ax.fill_between(
    times,
    output['wind_speed_actual_ms'],
    alpha=0.2,
    color='forestgreen'
)

ax.plot(
    times,
    output['wind_speed_actual_ms'],
    color='forestgreen',
    linewidth=2,
    label='Actual wind speed'
)

ax.plot(
    times,
    output['wind_speed_predicted_ms'],
    color='darkorange',
    linewidth=1.7,
    linestyle='--',
    label='Predicted wind speed'
)

# Operating limits

ax.axhline(
    3,
    color='navy',
    linewidth=1,
    linestyle='--',
    alpha=0.8,
    label='Cut-in speed (3 m/s)'
)

ax.axhline(
    9,
    color='gold',
    linewidth=1,
    linestyle='--',
    alpha=0.8,
    label='Goldwind rated (9 m/s)'
)

ax.axhline(
    13,
    color='orange',
    linewidth=1,
    linestyle='--',
    alpha=0.8,
    label='Vestas rated (13 m/s)'
)

ax.axhline(
    22,
    color='red',
    linewidth=1,
    linestyle='--',
    alpha=0.8,
    label='Cut-out speed (22 m/s)'
)


ax.set_title(
    'Predicted vs Actual Wind Speed',
    fontsize=13,
    fontweight='bold'
)

ax.set_ylabel('Wind speed (m/s)')
ax.set_xlabel('Time')

ax.legend()

comparison_file = PLOT_FILE.replace(
    '.png',
    '_comparison.png'
)

plt.tight_layout()

plt.savefig(
    comparison_file,
    dpi=150,
    bbox_inches='tight',
    facecolor='white'
)

print(f"  Comparison plot saved : {comparison_file}")

if SHOW_PLOT:
    plt.show()

plt.close()


# =============================================================
# FIGURE 3 — PREDICTION ERROR ONLY
# =============================================================

fig, ax = plt.subplots(figsize=(14, 5))

fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.grid(
    True,
    color='lightgray',
    linestyle='--',
    linewidth=0.6
)

ax.fill_between(
    times,
    error,
    0,
    where=(error >= 0),
    alpha=0.4,
    color='orange',
    label='Overprediction'
)

ax.fill_between(
    times,
    error,
    0,
    where=(error < 0),
    alpha=0.4,
    color='skyblue',
    label='Underprediction'
)

ax.plot(
    times,
    error,
    color='black',
    linewidth=1.0
)

ax.axhline(
    0,
    color='black',
    linestyle='--',
    linewidth=0.8
)

ax.set_title(
    f'Prediction Error (MAE = {mae:.3f} m/s)',
    fontsize=13,
    fontweight='bold'
)

ax.set_ylabel('Error (m/s)')
ax.set_xlabel('Time')

ax.legend()

error_file = PLOT_FILE.replace(
    '.png',
    '_error.png'
)

plt.tight_layout()

plt.savefig(
    error_file,
    dpi=150,
    bbox_inches='tight',
    facecolor='white'
)

print(f"  Error plot saved : {error_file}")

if SHOW_PLOT:
    plt.show()

plt.close()


# =============================================================
# SUMMARY
# =============================================================

print("\n" + "="*60)
print(f"  Forecast complete — {len(output)} hourly predictions")
print(f"  Output file ready for power generation script:")
print(f"  --> {OUTPUT_FILE}")
print("="*60 + "\n")
