"""
XGBoost Wind Speed Prediction Model
=====================================
Predicts hourly wind speed across a 24-hour cycle for any day of the year.

Data splits:
    Training   : 2020-01-01 to 2022-12-31  (3 years)
    Validation : 2023-01-01 to 2024-12-31  (2 years — hyperparameter tuning)
    Test       : 2025-01-01 to 2025-12-31  (1 year  — final evaluation)

Input:  wind_raw.csv (produced by wind_extract.py)
Output: wind_model.json     — trained XGBoost model
        wind_predictions.csv — hourly predictions vs actuals for test year
        wind_ml_results.png  — diagnostic plots

Requirements:
    pip install xgboost scikit-learn pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================
# USER INPUTS
# =============================================================

CSV_FILE        = 'wind_raw.csv'

# Data split boundaries
TRAIN_START     = '2020-01-01'
TRAIN_END       = '2022-12-31 23:00:00'
VAL_START       = '2023-01-01'
VAL_END         = '2024-12-31 23:00:00'
TEST_START      = '2025-01-01'
TEST_END        = '2025-12-31 23:00:00'

# Prediction horizon — how many hours ahead to predict
# 1  = next hour (short-term)
# 24 = 24 hours ahead (day-ahead dispatch planning)
FORECAST_HORIZON = 1

# XGBoost hyperparameters
# These are tuned against the validation set
XGB_PARAMS = {
    'n_estimators'    : 1000,
    'learning_rate'   : 0.05,
    'max_depth'       : 6,
    'subsample'       : 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha'       : 0.1,    # L1 regularisation
    'reg_lambda'      : 1.0,    # L2 regularisation
    'random_state'    : 42,
    'n_jobs'          : -1,
    'early_stopping_rounds': 50
}

# Output files
MODEL_FILE       = 'wind_model.json'
PREDICTIONS_FILE = 'wind_predictions.csv'
PLOT_FILE        = 'wind_ml_results.png'
SHOW_PLOT        = True

# =============================================================
# STEP 1 — Load and validate data
# =============================================================

print("\n" + "="*60)
print("  XGBoost Wind Speed Prediction Model")
print("="*60)
print(f"\n[1/6] Loading data from {CSV_FILE}...")

df = pd.read_csv(CSV_FILE)
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
df = df.sort_values('datetime').reset_index(drop=True)

# Check for missing hours and fill gaps
full_range = pd.date_range(
    start=df['datetime'].min(),
    end=df['datetime'].max(),
    freq='h'
)
n_missing = len(full_range) - len(df)
if n_missing > 0:
    print(f"  WARNING: {n_missing} missing hourly timestamps detected — filling by interpolation")
    df = df.set_index('datetime').reindex(full_range)
    df.index.name = 'datetime'
    df = df.interpolate(method='time').reset_index()

print(f"  Rows        : {len(df):,}")
print(f"  Date range  : {df['datetime'].min()} to {df['datetime'].max()}")
print(f"  Wind speed  : mean={df['wind_speed_ms'].mean():.2f} m/s  "
      f"max={df['wind_speed_ms'].max():.2f} m/s  "
      f"min={df['wind_speed_ms'].min():.2f} m/s")

# =============================================================
# STEP 2 — Feature engineering
# =============================================================

print(f"\n[2/6] Engineering features...")

# --- Cyclical time encoding ---
# Encodes circular features so hour 23 is numerically close to hour 0,
# and December is close to January. Without this, models treat them as far apart.
df['hour']          = df['datetime'].dt.hour
df['day_of_year']   = df['datetime'].dt.dayofyear
df['month']         = df['datetime'].dt.month
df['day_of_week']   = df['datetime'].dt.dayofweek
df['year']          = df['datetime'].dt.year

df['hour_sin']      = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos']      = np.cos(2 * np.pi * df['hour'] / 24)
df['doy_sin']       = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['doy_cos']       = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
df['month_sin']     = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos']     = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin']       = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos']       = np.cos(2 * np.pi * df['day_of_week'] / 7)

# --- Wind direction components ---
# Keep U and V as features — direction matters for persistence patterns
df['wind_dir_sin']  = np.sin(np.radians(
    (180 + np.degrees(np.arctan2(df['u100'], df['v100']))) % 360
))
df['wind_dir_cos']  = np.cos(np.radians(
    (180 + np.degrees(np.arctan2(df['u100'], df['v100']))) % 360
))

# --- Lag features — past wind speed as predictor of future wind ---
# Wind is strongly autocorrelated: what it was recently predicts what it will be
lag_hours = [1, 2, 3, 4, 6, 12, 24, 48]
for lag in lag_hours:
    df[f'wind_lag_{lag}h']  = df['wind_speed_ms'].shift(lag)
    df[f'u100_lag_{lag}h']  = df['u100'].shift(lag)
    df[f'v100_lag_{lag}h']  = df['v100'].shift(lag)

# --- Rolling statistics — recent wind behaviour ---
for window in [3, 6, 12, 24]:
    df[f'wind_roll_{window}h_mean'] = df['wind_speed_ms'].rolling(window).mean()
    df[f'wind_roll_{window}h_std']  = df['wind_speed_ms'].rolling(window).std()
    df[f'wind_roll_{window}h_max']  = df['wind_speed_ms'].rolling(window).max()

# --- Rate of change ---
df['wind_change_1h']  = df['wind_speed_ms'].diff(1)   # m/s per hour
df['wind_change_3h']  = df['wind_speed_ms'].diff(3)
df['wind_change_6h']  = df['wind_speed_ms'].diff(6)
df['wind_change_24h'] = df['wind_speed_ms'].diff(24)  # same time yesterday

# --- Target variable: wind speed N hours ahead ---
df['target'] = df['wind_speed_ms'].shift(-FORECAST_HORIZON)

# Drop rows with NaN from lag/rolling/target creation
df = df.dropna().reset_index(drop=True)

# Define feature columns (everything except raw inputs and target)
FEATURE_COLS = [
    # Cyclical time
    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
    'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    # Wind components
    'u100', 'v100', 'wind_dir_sin', 'wind_dir_cos',
    # Lags
    *[f'wind_lag_{h}h'  for h in lag_hours],
    *[f'u100_lag_{h}h'  for h in lag_hours],
    *[f'v100_lag_{h}h'  for h in lag_hours],
    # Rolling stats
    *[f'wind_roll_{w}h_mean' for w in [3, 6, 12, 24]],
    *[f'wind_roll_{w}h_std'  for w in [3, 6, 12, 24]],
    *[f'wind_roll_{w}h_max'  for w in [3, 6, 12, 24]],
    # Rate of change
    'wind_change_1h', 'wind_change_3h', 'wind_change_6h', 'wind_change_24h',
]

print(f"  Features    : {len(FEATURE_COLS)}")
print(f"  Rows after dropna: {len(df):,}")

# =============================================================
# STEP 3 — Chronological train / validation / test split
# =============================================================

print(f"\n[3/6] Splitting data chronologically...")

train = df[(df['datetime'] >= TRAIN_START) & (df['datetime'] <= TRAIN_END)]
val   = df[(df['datetime'] >= VAL_START)   & (df['datetime'] <= VAL_END)]
test  = df[(df['datetime'] >= TEST_START)  & (df['datetime'] <= TEST_END)]

X_train, y_train = train[FEATURE_COLS], train['target']
X_val,   y_val   = val[FEATURE_COLS],   val['target']
X_test,  y_test  = test[FEATURE_COLS],  test['target']

print(f"  Training   : {len(train):,} rows  "
      f"({train['datetime'].min().date()} → {train['datetime'].max().date()})")
print(f"  Validation : {len(val):,} rows  "
      f"({val['datetime'].min().date()} → {val['datetime'].max().date()})")
print(f"  Test       : {len(test):,} rows  "
      f"({test['datetime'].min().date()} → {test['datetime'].max().date()})")
print(f"  Forecast horizon: {FORECAST_HORIZON} hour(s) ahead")

# =============================================================
# STEP 4 — Train XGBoost model
# =============================================================

print(f"\n[4/6] Training XGBoost model...")
print(f"  Early stopping patience: {XGB_PARAMS['early_stopping_rounds']} rounds")
print(f"  Max estimators: {XGB_PARAMS['n_estimators']}")

model = xgb.XGBRegressor(**XGB_PARAMS)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=100
)

n_trees = model.best_iteration + 1
print(f"\n  Best iteration : {n_trees} trees (early stopping)")

# =============================================================
# STEP 5 — Evaluate on all three sets
# =============================================================

print(f"\n[5/6] Evaluating model performance...")

def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # Skill score vs persistence (naive baseline: predict current = future)
    print(f"\n  {name}:")
    print(f"    MAE  : {mae:.3f} m/s  (average absolute error)")
    print(f"    RMSE : {rmse:.3f} m/s  (root mean square error)")
    print(f"    R²   : {r2:.4f}       (1.0 = perfect)")
    return mae, rmse, r2

# Predictions
train_pred = model.predict(X_train)
val_pred   = model.predict(X_val)
test_pred  = model.predict(X_test)

# Clamp predictions — wind speed cannot be negative
train_pred = np.maximum(train_pred, 0)
val_pred   = np.maximum(val_pred,   0)
test_pred  = np.maximum(test_pred,  0)

print("\n" + "-"*50)
print("  PERFORMANCE SUMMARY")
print("-"*50)
train_metrics = evaluate("Training   (2020–2022)", y_train, train_pred)
val_metrics   = evaluate("Validation (2023–2024)", y_val,   val_pred)
test_metrics  = evaluate("Test       (2025)",      y_test,  test_pred)

# Persistence baseline — predict next hour = current hour
persistence_pred = test['wind_speed_ms'].values  # current as prediction
persistence_mae  = mean_absolute_error(y_test, persistence_pred)
persistence_rmse = np.sqrt(mean_squared_error(y_test, persistence_pred))
print(f"\n  Persistence baseline (naive — current hour = next hour):")
print(f"    MAE  : {persistence_mae:.3f} m/s")
print(f"    RMSE : {persistence_rmse:.3f} m/s")
print(f"\n  XGBoost improvement over persistence:")
print(f"    MAE  : {(1 - test_metrics[0]/persistence_mae)*100:.1f}% better")
print(f"    RMSE : {(1 - test_metrics[1]/persistence_rmse)*100:.1f}% better")

# Feature importance — top 15
importance = pd.DataFrame({
    'feature'   : FEATURE_COLS,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  Top 15 most important features:")
for _, row in importance.head(15).iterrows():
    bar = '█' * int(row['importance'] * 500)
    print(f"    {row['feature']:<30} {bar} {row['importance']:.4f}")

# =============================================================
# STEP 6 — Save model and predictions
# =============================================================

print(f"\n[6/6] Saving outputs...")

# Save model
model.save_model(MODEL_FILE)
print(f"  Model saved : {MODEL_FILE}")

# Save test predictions
test_out = test[['datetime', 'wind_speed_ms']].copy()
test_out['predicted_ms'] = test_pred
test_out['error_ms']     = test_out['predicted_ms'] - test_out['wind_speed_ms']
test_out.to_csv(PREDICTIONS_FILE, index=False)
print(f"  Predictions : {PREDICTIONS_FILE}")

# =============================================================
# PLOTS
# =============================================================

fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor('#0f1923')

# Layout: 3 rows x 2 cols, last plot spans full bottom row
ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4)
ax5 = fig.add_subplot(3, 1, 3)   # spans full bottom row
axes = [ax1, ax2, ax3, ax4, ax5]

for ax in axes:
    ax.set_facecolor('#131f2e')
    ax.tick_params(colors='#8fa8c8', labelsize=8)
    ax.xaxis.label.set_color('#8fa8c8')
    ax.yaxis.label.set_color('#8fa8c8')
    ax.title.set_color('#dce8f5')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a3f5a')
    ax.grid(True, color='#1e3050', linewidth=0.6, linestyle='--', alpha=0.7)

fig.suptitle(
    f'XGBoost Wind Speed Model — New England NSW\n'
    f'Train 2020–2022  |  Validate 2023–2024  |  Test 2025  '
    f'|  Horizon: {FORECAST_HORIZON}h ahead',
    fontsize=12, color='#dce8f5', fontweight='bold', y=0.98
)

# --- Plot 1: Scatter — predicted vs actual (test set) ---
ax = axes[0]
ax.scatter(y_test, test_pred, alpha=0.15, s=3,
           color='#4caf8a', rasterized=True)
lims = [0, max(y_test.max(), test_pred.max()) * 1.05]
ax.plot(lims, lims, 'r--', linewidth=1.2, alpha=0.8, label='Perfect prediction')
ax.set_xlabel('Actual wind speed (m/s)')
ax.set_ylabel('Predicted wind speed (m/s)')
ax.set_title(f'Predicted vs Actual — Test 2025\nR²={test_metrics[2]:.3f}  '
             f'MAE={test_metrics[0]:.2f} m/s')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.legend(fontsize=7, framealpha=0.2, facecolor='#0f1923',
          edgecolor='#2a3f5a', labelcolor='#dce8f5')

# --- Plot 2: Error distribution ---
ax = axes[1]
errors = test_out['error_ms']
ax.hist(errors, bins=80, color='#4fc3f7', alpha=0.75, edgecolor='none')
ax.axvline(0, color='#ff5252', linewidth=1.2, linestyle='--')
ax.axvline(errors.mean(), color='#f5c842', linewidth=1.0,
           linestyle='--', label=f'Mean={errors.mean():.2f}')
ax.set_xlabel('Prediction error (m/s)')
ax.set_ylabel('Count')
ax.set_title(f'Error Distribution — Test 2025\n'
             f'Std={errors.std():.2f} m/s')
ax.legend(fontsize=7, framealpha=0.2, facecolor='#0f1923',
          edgecolor='#2a3f5a', labelcolor='#dce8f5')

# --- Plot 3: Average diurnal profile by season ---
ax = axes[2]
test_out_plot = test_out.copy()
test_out_plot['hour']   = pd.to_datetime(test_out_plot['datetime']).dt.hour
test_out_plot['month']  = pd.to_datetime(test_out_plot['datetime']).dt.month
test_out_plot['season'] = test_out_plot['month'].map({
    12:'Summer', 1:'Summer', 2:'Summer',
    3:'Autumn',  4:'Autumn', 5:'Autumn',
    6:'Winter',  7:'Winter', 8:'Winter',
    9:'Spring', 10:'Spring', 11:'Spring'
})
season_colors = {
    'Summer': '#f5a623',
    'Autumn': '#e05c5c',
    'Winter': '#4fc3f7',
    'Spring': '#4caf8a'
}
for season, color in season_colors.items():
    s = test_out_plot[test_out_plot['season'] == season]
    actual_diurnal = s.groupby('hour')['wind_speed_ms'].mean()
    pred_diurnal   = s.groupby('hour')['predicted_ms'].mean()
    ax.plot(actual_diurnal.index, actual_diurnal.values,
            color=color, linewidth=1.8, label=f'{season} actual')
    ax.plot(pred_diurnal.index, pred_diurnal.values,
            color=color, linewidth=1.2, linestyle='--', alpha=0.75)
ax.set_xlabel('Hour of day')
ax.set_ylabel('Average wind speed (m/s)')
ax.set_title('Average Diurnal Profile by Season\n(solid=actual, dashed=predicted)')
ax.set_xticks(range(0, 24, 3))
ax.legend(fontsize=6.5, framealpha=0.2, facecolor='#0f1923',
          edgecolor='#2a3f5a', labelcolor='#dce8f5', ncol=2)

# --- Plot 4: Feature importance ---
ax = axes[3]
top_n = 15
top   = importance.head(top_n)
bars  = ax.barh(range(top_n), top['importance'].values,
                color='#4caf8a', alpha=0.8, edgecolor='none')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top['feature'].values, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel('Feature importance')
ax.set_title(f'Top {top_n} Feature Importances')

# --- Plot 5: 2-week time series sample from test year ---
ax = axes[4]
sample_start = pd.Timestamp('2025-07-01')
sample_end   = pd.Timestamp('2025-07-14 23:00:00')
sample = test_out[
    (pd.to_datetime(test_out['datetime']) >= sample_start) &
    (pd.to_datetime(test_out['datetime']) <= sample_end)
].copy()

times = pd.to_datetime(sample['datetime'])
ax.fill_between(times, sample['wind_speed_ms'], alpha=0.25,
                color='#4caf8a')
ax.plot(times, sample['wind_speed_ms'],
        color='#4caf8a', linewidth=1.2, label='Actual')
ax.plot(times, sample['predicted_ms'],
        color='#f5c842', linewidth=1.0, linestyle='--',
        alpha=0.9, label='Predicted')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.set_xlabel('Date')
ax.set_ylabel('Wind speed (m/s)')
ax.set_title('Sample Forecast — July 2025 (2 weeks)')
ax.legend(fontsize=8, framealpha=0.2, facecolor='#0f1923',
          edgecolor='#2a3f5a', labelcolor='#dce8f5')

plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"  Plot saved  : {PLOT_FILE}")

if SHOW_PLOT:
    plt.show()

plt.close()

print("\n" + "="*60)
print("  Training complete.")
print(f"  Test MAE  : {test_metrics[0]:.3f} m/s")
print(f"  Test RMSE : {test_metrics[1]:.3f} m/s")
print(f"  Test R²   : {test_metrics[2]:.4f}")
print("="*60 + "\n")
