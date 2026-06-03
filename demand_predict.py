"""
Demand Model - Prediction Script
==================================
New England NSW - Half-Hour Demand Forecasting

Accepts any day and month as input and produces a 24-hour
half-hourly demand forecast using the trained Random Forest model.

Year is not required - the model uses seasonal and diurnal patterns
learned across all training years. The processed data provides the
lag features needed for prediction.

Requires demand_train.py to have been run first.

Inputs:
    demand_model.pkl      - saved model from demand_train.py
    demand_processed.csv  - preprocessed data from demand_train.py

Output:
    demand_profile.csv    - half-hourly demand forecast in MW
                            ready for use in power_generation.py

Requirements:
    pip install pandas scikit-learn matplotlib joblib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import warnings
from config import FORECAST_DAY, FORECAST_MONTH # type: ignore

warnings.filterwarnings('ignore')

# =============================================================
# USER INPUTS - Set day and month only, year is handled automatically
# =============================================================

MODEL_FILE      = 'demand_model.pkl'
PROCESSED_FILE  = 'demand_processed.csv'

plt.rcParams.update({
    'figure.facecolor' : 'white',
    'axes.facecolor'   : 'white',
    'axes.edgecolor'   : '#cccccc',
    'axes.labelcolor'  : 'black',
    'xtick.color'      : 'black',
    'ytick.color'      : 'black',
    'text.color'       : 'black',
    'grid.color'       : '#eeeeee',
    'grid.linewidth'   : 0.8,
    'grid.linestyle'   : '--',
})
# Day and month to forecast - year is irrelevant
# The model will find the most recent matching day/month in the
# processed data to extract lag features, then predict forward

# Output
OUTPUT_FILE     = 'demand_profile.csv'
PLOT_FILE       = 'demand_forecast_plot.png'
SHOW_PLOT       = False

# =============================================================
# STEP 1 - Load model and processed data
# =============================================================

print("\n" + "="*60)
print("  Demand Predictor - New England NSW")
print("="*60)

print(f"\n[1/3] Loading model and data...")

model = joblib.load(MODEL_FILE)
print(f"  Model loaded  : {MODEL_FILE}")

df = pd.read_csv(PROCESSED_FILE, parse_dates=['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)
print(f"  Data loaded   : {len(df):,} rows  "
      f"({df['Datetime'].min().date()} to {df['Datetime'].max().date()})")

# =============================================================
# STEP 2 - Find the best matching day in processed data
# =============================================================

print(f"\n[2/3] Locating forecast day (day={FORECAST_DAY}, month={FORECAST_MONTH})...")

# Find all occurrences of this day/month in the processed data
# Prefer the most recent occurrence to get the most relevant lag values
matches = df[
    (df['Datetime'].dt.day   == FORECAST_DAY) &
    (df['Datetime'].dt.month == FORECAST_MONTH)
]

if matches.empty:
    # Day might not exist in some years (e.g. Feb 29 in non-leap years)
    # Fall back to nearest available day in that month
    month_data = df[df['Datetime'].dt.month == FORECAST_MONTH]
    if month_data.empty:
        raise ValueError(f"No data found for month {FORECAST_MONTH}.")
    nearest_day = (month_data['Datetime'].dt.day - FORECAST_DAY).abs().min()
    fallback_day = month_data[
        (month_data['Datetime'].dt.day - FORECAST_DAY).abs() == nearest_day
    ]['Datetime'].dt.day.iloc[0]
    print(f"  Day {FORECAST_DAY} not found in month {FORECAST_MONTH} "
          f"- using day {fallback_day} instead")
    matches = df[
        (df['Datetime'].dt.day   == fallback_day) &
        (df['Datetime'].dt.month == FORECAST_MONTH)
    ]

# Use the most recent matching date in the processed data
# This ensures lag features reflect the most up-to-date seasonal patterns
most_recent_date = matches['Datetime'].dt.date.max()
forecast_rows    = df[df['Datetime'].dt.date == most_recent_date].copy()

month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

print(f"  Using data from : {most_recent_date} "
      f"({month_names[FORECAST_MONTH-1]} data from most recent year available)")
print(f"  Intervals found : {len(forecast_rows)} half-hour periods")

if len(forecast_rows) == 0:
    raise ValueError(
        f"No rows found for {most_recent_date}. "
        f"Check FORECAST_DAY and FORECAST_MONTH are valid."
    )

# =============================================================
# STEP 3 - Predict
# =============================================================

print(f"\n[3/3] Generating 24-hour demand forecast...")

FEATURE_COLS = [
    'Hour', 'Minute', 'DayOfWeek', 'Month',
    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
    'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    'lag_1', 'lag_2', 'lag_48', 'lag_336',
    'roll_4h_mean', 'roll_24h_mean'
]

# Check all features are present
missing = [f for f in FEATURE_COLS if f not in forecast_rows.columns]
if missing:
    raise ValueError(
        f"Missing features in processed data: {missing}\n"
        f"Re-run demand_train.py to regenerate demand_processed.csv"
    )

X_forecast   = forecast_rows[FEATURE_COLS]
predicted_kw = model.predict(X_forecast)
predicted_kw = np.maximum(predicted_kw, 0)

# Build a clean 24-hour time index starting from midnight
# We replace the year with a neutral label since year is irrelevant
base_date = pd.Timestamp(f'2025-{FORECAST_MONTH:02d}-{FORECAST_DAY:02d}')
n_intervals = len(forecast_rows)
time_index  = pd.date_range(start=base_date, periods=n_intervals, freq='30min')

output = pd.DataFrame({
    'datetime'         : time_index,
    'demand_mw'        : predicted_kw / 1000,       # MW - for power_generation.py
    'demand_actual_mw' : forecast_rows['kW'].values / 1000,
    'demand_kw'        : predicted_kw,
})

output['error_mw'] = output['demand_mw'] - output['demand_actual_mw']

# Summary
day_label = f"{FORECAST_DAY} {month_names[FORECAST_MONTH-1]}"
print(f"\n  Forecast : {day_label} (any year)")
print(f"  Intervals: {len(output)} x 30min")
print(f"\n  Peak demand   : {output['demand_mw'].max():.2f} MW  "
      f"at {output.loc[output['demand_mw'].idxmax(), 'datetime'].strftime('%H:%M')}")
print(f"  Trough demand : {output['demand_mw'].min():.2f} MW  "
      f"at {output.loc[output['demand_mw'].idxmin(), 'datetime'].strftime('%H:%M')}")
print(f"  Daily average : {output['demand_mw'].mean():.2f} MW")
print(f"  Daily energy  : {output['demand_mw'].sum() * 0.5:.1f} MWh")

# Print half-hourly table
print(f"\n  Half-hourly forecast:")
print(f"  {'Time':<8} {'Predicted (MW)':<18} {'Actual (MW)':<18} {'Error (MW)'}")
print(f"  {'-'*56}")
for _, row in output.iterrows():
    t = pd.Timestamp(row['datetime']).strftime('%H:%M')
    print(f"  {t:<8} {row['demand_mw']:>10.2f}         "
          f"{row['demand_actual_mw']:>10.2f}         "
          f"{row['error_mw']:>+.2f}")

mae_mw = np.mean(np.abs(output['error_mw']))
print(f"\n  MAE vs historical actual : {mae_mw:.3f} MW")

# Save - demand_profile.csv is read directly by power_generation.py
output[['datetime', 'demand_mw', 'demand_actual_mw']].to_csv(
    OUTPUT_FILE, index=False, float_format='%.4f'
)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"  Columns : datetime, demand_mw")
print(f"  Ready for power_generation.py")
print(f"  Set DEMAND_FILE = '{OUTPUT_FILE}' in power_generation.py")

# =============================================================
# FIGURE 1 — COMBINED OVERVIEW
# =============================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# White background styling
fig.patch.set_facecolor('white')

for ax in axes:
    ax.set_facecolor('white')
    ax.tick_params(colors='black', labelsize=9)
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    ax.grid(True, color='lightgray',
            linewidth=0.6, linestyle='--', alpha=0.7)

fig.suptitle(
    f'Demand Forecast - New England NSW\n'
    f'{day_label}  |  Reference year: {most_recent_date.year}  '
    f'|  MAE: {mae_mw:.3f} MW',
    fontsize=13,
    color='black',
    fontweight='bold',
    y=0.98
)

times = pd.to_datetime(output['datetime'])

# -------------------------------------------------------------
# Panel 1: Predicted vs Historical Actual
# -------------------------------------------------------------

ax = axes[0]

ax.fill_between(
    times,
    output['demand_actual_mw'],
    alpha=0.2,
    color='forestgreen'
)

ax.plot(
    times,
    output['demand_actual_mw'],
    color='forestgreen',
    linewidth=1.8,
    label=f'Historical actual ({most_recent_date.year})'
)

ax.plot(
    times,
    output['demand_mw'],
    color='darkorange',
    linewidth=1.5,
    linestyle='--',
    label='Predicted demand'
)

ax.set_ylabel('Demand (MW)')
ax.set_title('Predicted vs Historical Actual Demand')
ax.set_ylim(bottom=0)

ax.legend(
    fontsize=8,
    framealpha=0.9,
    facecolor='white',
    edgecolor='black'
)

# -------------------------------------------------------------
# Panel 2: Prediction Error
# -------------------------------------------------------------

ax = axes[1]

error = output['error_mw']

ax.fill_between(
    times, error, 0,
    where=(error >= 0),
    alpha=0.4,
    color='orange',
    label='Overprediction'
)

ax.fill_between(
    times, error, 0,
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
    linewidth=0.8,
    linestyle='--'
)

ax.set_ylabel('Error (MW)')
ax.set_xlabel('Time of Day')

ax.set_title(
    f'Prediction Error  (MAE = {mae_mw:.3f} MW)'
)

ax.legend(
    fontsize=8,
    framealpha=0.9,
    facecolor='white',
    edgecolor='black'
)

# Time axis formatting
axes[1].xaxis.set_major_locator(
    mdates.HourLocator(interval=2)
)

axes[1].xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M')
)

axes[1].tick_params(
    axis='x',
    colors='black',
    labelsize=9
)

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=150,
    bbox_inches='tight',
    facecolor='white'
)

print(f"  Combined plot saved : {PLOT_FILE}")

if SHOW_PLOT:
    plt.show()

plt.close()


# =============================================================
# FIGURE 2 — DEMAND COMPARISON ONLY
# =============================================================

fig, ax = plt.subplots(figsize=(14, 5))

fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.grid(True, color='lightgray',
        linestyle='--', linewidth=0.6)

ax.fill_between(
    times,
    output['demand_actual_mw'],
    alpha=0.2,
    color='forestgreen'
)

ax.plot(
    times,
    output['demand_actual_mw'],
    color='forestgreen',
    linewidth=2,
    label='Historical actual'
)

ax.plot(
    times,
    output['demand_mw'],
    color='darkorange',
    linewidth=1.8,
    linestyle='--',
    label='Predicted demand'
)

ax.set_title(
    'Demand Forecast Comparison',
    fontsize=13,
    fontweight='bold'
)

ax.set_ylabel('Demand (MW)')
ax.set_xlabel('Time of Day')

ax.legend()

ax.xaxis.set_major_locator(
    mdates.HourLocator(interval=2)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M')
)

comparison_file = PLOT_FILE.replace('.png', '_comparison.png')

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
# FIGURE 3 — ERROR ONLY
# =============================================================

fig, ax = plt.subplots(figsize=(14, 5))

fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.grid(True, color='lightgray',
        linestyle='--', linewidth=0.6)

ax.fill_between(
    times, error, 0,
    where=(error >= 0),
    alpha=0.4,
    color='orange',
    label='Overprediction'
)

ax.fill_between(
    times, error, 0,
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
    f'Prediction Error (MAE = {mae_mw:.3f} MW)',
    fontsize=13,
    fontweight='bold'
)

ax.set_ylabel('Error (MW)')
ax.set_xlabel('Time of Day')

ax.legend()

ax.xaxis.set_major_locator(
    mdates.HourLocator(interval=2)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M')
)

error_file = PLOT_FILE.replace('.png', '_error.png')

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
print(f"  Forecast complete")
print(f"  Day       : {day_label}")
print(f"  Intervals : {len(output)} x 30min")
print(f"  Peak      : {output['demand_mw'].max():.2f} MW")
print(f"  Output    : {OUTPUT_FILE}")
print("="*60 + "\n")