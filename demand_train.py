"""
Demand Model — Training Script
================================
New England NSW — Half-Hour Demand Forecasting

Data coverage: 1 Oct 2019 → 30 Sep 2025 (financial years)

Split:
    Training   : 1 Oct 2019 → 30 Sep 2022  (3 financial years)
    Validation : 1 Oct 2022 → 30 Sep 2024  (2 financial years)
    Test       : 1 Oct 2024 → 30 Sep 2025  (1 financial year)

Run this script ONCE. After it completes, use demand_predict.py
for fast predictions on any day and month within the data coverage.

Outputs:
    demand_model.pkl          — trained Random Forest model
    demand_processed.csv      — preprocessed combined dataset
    demand_training_plot.png  — validation diagnostic plot

Requirements:
    pip install pandas scikit-learn matplotlib joblib
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import os
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore')

# =============================================================
# USER INPUTS
# =============================================================

CSV_FILES = [
    'EE Zone Substation Load Data 2019-20.csv',
    'EE Zone Substation Load Data 2020-21.csv',
    'EE Zone Substation Load Data 2021-22.csv',
    'EE Zone Substation Load Data 2022-23.csv',
    'EE Zone Substation Load Data 2023-24.csv',
    'EE Zone Substation Load Data 2024-25.csv',
]

TOWNS = [
    # Original towns
    'Tenterfield',
    'Glen Innes',
    'Moree',
    'Narrabri',
    'Gunnedah',
    'Quirindi 11',
    'Quirindi 33',
    'Wee Waa',
    # Northern Tablelands additions
    'Goddard Lane',
    # Namoi additions
    'Attunga',
    'Boggabri',
    'Coonabarabran',
    'Galloway Street',
    'Hillgrove',
    'Madgwick Drive',
    'Nundle',
    'East Tamworth',
    'South Tamworth',
    'Walcha South 22/11',
    'Walcha South 66/22',
    'Werris Creek',
    # Border Rivers additions
    'Bingara',
    'Borthwick St',
    'Texas 22',
    'Texas 33',
    'Wathagar',
]

# Chronological split boundaries aligned to financial year (1 Oct)
# Training   : 1 Oct 2019 - 30 Sep 2022  (3 years)
# Validation : 1 Oct 2022 - 30 Sep 2024  (2 years)
# Test       : 1 Oct 2024 - 30 Sep 2025  (1 year)
TRAIN_END   = '2022-10-01'
VAL_END     = '2024-10-01'

RF_PARAMS = {
    'n_estimators' : 100,
    'max_depth'    : None,
    'random_state' : 1,
    'n_jobs'       : -1
}

MODEL_FILE      = 'demand_model.pkl'
PROCESSED_FILE  = 'demand_processed.csv'
PLOT_FILE       = 'demand_training_plot.png'
SHOW_PLOT       = True

# =============================================================
# STEP 1 - Load and preprocess raw CSVs
# =============================================================

print("\n" + "="*60)
print("  Demand Model - Training Script")
print("="*60)
print(f"\n[1/4] Loading and preprocessing {len(CSV_FILES)} CSV files...")
print(f"  Towns : {TOWNS}")
print(f"  Note  : This step is slow - it only needs to run once.\n")

def process_csv(filepath, towns):
    """Load one financial year CSV, filter to towns, and return
    a summed half-hourly DataFrame."""
    print(f"  Loading {os.path.basename(filepath)}...", end='', flush=True)

    df_raw = pd.read_csv(filepath, usecols=['Name', 'Date', 'Time', 'kW'])
    print(f" {len(df_raw):,} rows", end='')

    all_town_data = []
    for town in towns:
        df_town = df_raw[df_raw['Name'].str.contains(town, na=False)].copy()
        if df_town.empty:
            continue
        df_town['Datetime'] = pd.to_datetime(
            df_town['Date'] + ' ' + df_town['Time'],
            dayfirst=True, errors='coerce'
        )
        df_town['kW'] = pd.to_numeric(df_town['kW'], errors='coerce')
        df_town = df_town.dropna(subset=['Datetime'])
        df_town = df_town.sort_values('Datetime')
        df_town['kW'] = df_town['kW'].interpolate()
        df_town = df_town[['Datetime', 'kW']]
        df_town = df_town.groupby('Datetime', as_index=False)['kW'].sum()
        all_town_data.append(df_town)

    if not all_town_data:
        print(" - WARNING: no matching town data found")
        return pd.DataFrame()

    combined = pd.concat(all_town_data)
    total_df = combined.groupby('Datetime', as_index=False)['kW'].sum()
    total_df = total_df.sort_values('Datetime').reset_index(drop=True)
    print(f" -> {len(total_df):,} half-hour intervals")
    return total_df


all_data = []
for filepath in CSV_FILES:
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found - skipping")
        continue
    result = process_csv(filepath, TOWNS)
    if not result.empty:
        all_data.append(result)

if not all_data:
    raise FileNotFoundError(
        "No data loaded. Check that CSV files are in the same folder "
        "as this script and filenames match exactly."
    )

df = pd.concat(all_data).sort_values('Datetime').reset_index(drop=True)
df = df.drop_duplicates(subset='Datetime').reset_index(drop=True)

print(f"\n  Combined rows : {len(df):,}")
print(f"  Date range    : {df['Datetime'].min().date()} to {df['Datetime'].max().date()}")
print(f"  Demand range  : {df['kW'].min():.0f} kW to {df['kW'].max():.0f} kW")

# =============================================================
# STEP 2 - Feature engineering
# =============================================================

print(f"\n[2/4] Engineering features...")

df['Hour']      = df['Datetime'].dt.hour
df['Minute']    = df['Datetime'].dt.minute
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Month']     = df['Datetime'].dt.month
df['DayOfYear'] = df['Datetime'].dt.dayofyear

# Cyclical encoding - consistent with wind model
df['hour_sin']  = np.sin(2 * np.pi * df['Hour']      / 24)
df['hour_cos']  = np.cos(2 * np.pi * df['Hour']      / 24)
df['doy_sin']   = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
df['doy_cos']   = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)
df['month_sin'] = np.sin(2 * np.pi * df['Month']     / 12)
df['month_cos'] = np.cos(2 * np.pi * df['Month']     / 12)
df['dow_sin']   = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
df['dow_cos']   = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

# Lag features (30-min intervals)
# lag_1   = 30 min ago
# lag_2   = 60 min ago
# lag_48  = 24 hours ago (same time yesterday)
# lag_336 = 1 week ago  (same time, same day of week)
df['lag_1']   = df['kW'].shift(1)
df['lag_2']   = df['kW'].shift(2)
df['lag_48']  = df['kW'].shift(48)
df['lag_336'] = df['kW'].shift(336)

# Rolling statistics
df['roll_4h_mean']  = df['kW'].rolling(8).mean()
df['roll_24h_mean'] = df['kW'].rolling(48).mean()

df = df.dropna().reset_index(drop=True)

FEATURE_COLS = [
    'Hour', 'Minute', 'DayOfWeek', 'Month',
    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
    'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    'lag_1', 'lag_2', 'lag_48', 'lag_336',
    'roll_4h_mean', 'roll_24h_mean'
]

print(f"  Features          : {len(FEATURE_COLS)}")
print(f"  Rows after dropna : {len(df):,}")

df.to_csv(PROCESSED_FILE, index=False)
print(f"  Processed data saved: {PROCESSED_FILE}")

# =============================================================
# STEP 3 - Train / validate / test split and training
# =============================================================

print(f"\n[3/4] Splitting data and training model...")

train = df[df['Datetime'] <  TRAIN_END]
val   = df[(df['Datetime'] >= TRAIN_END) & (df['Datetime'] < VAL_END)]
test  = df[df['Datetime'] >= VAL_END]

print(f"\n  Training   : {train['Datetime'].min().date()} to "
      f"{train['Datetime'].max().date()}  ({len(train):,} rows)")
print(f"  Validation : {val['Datetime'].min().date()} to "
      f"{val['Datetime'].max().date()}  ({len(val):,} rows)")
print(f"  Test       : {test['Datetime'].min().date()} to "
      f"{test['Datetime'].max().date()}  ({len(test):,} rows)")

X_train, y_train = train[FEATURE_COLS], train['kW']
X_val,   y_val   = val[FEATURE_COLS],   val['kW']
X_test,  y_test  = test[FEATURE_COLS],  test['kW']

print(f"\n  Training model (n_estimators={RF_PARAMS['n_estimators']})...")
model = RandomForestRegressor(**RF_PARAMS)
model.fit(X_train, y_train)

def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n  {name}:")
    print(f"    MAE  : {mae:,.0f} kW  ({mae/1000:.3f} MW)")
    print(f"    RMSE : {rmse:,.0f} kW")
    print(f"    R2   : {r2:.4f}")
    return mae, rmse, r2

train_pred = model.predict(X_train)
val_pred   = model.predict(X_val)
test_pred  = model.predict(X_test)

print("\n" + "-"*50)
print("  PERFORMANCE SUMMARY")
print("-"*50)
evaluate("Training   (Oct 2019 - Sep 2022)", y_train, train_pred)
evaluate("Validation (Oct 2022 - Sep 2024)", y_val,   val_pred)
test_mae, test_rmse, test_r2 = evaluate(
    "Test       (Oct 2024 - Sep 2025)", y_test, test_pred
)

# Overfitting check
train_r2 = r2_score(y_train, train_pred)
val_r2   = r2_score(y_val,   val_pred)
gap      = train_r2 - val_r2
if gap > 0.1:
    print(f"\n  NOTE: R2 gap of {gap:.3f} between train and validation "
          f"suggests mild overfitting.")
    print(f"  Consider reducing n_estimators or adding max_depth limit.")

importance = pd.DataFrame({
    'feature'    : FEATURE_COLS,
    'importance' : model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  Top 10 features:")
for _, row in importance.head(10).iterrows():
    bar = '#' * int(row['importance'] * 300)
    print(f"    {row['feature']:<22} {bar} {row['importance']:.4f}")

joblib.dump(model, MODEL_FILE)
print(f"\n  Model saved: {MODEL_FILE}")

test_result = test[['Datetime', 'kW']].copy()
test_result['Predicted_kW'] = test_pred
test_result['Predicted_MW'] = test_pred / 1000
test_result['Actual_MW']    = test_result['kW'] / 1000

# =============================================================
# STEP 4 - Diagnostic plots
# =============================================================

print(f"\n[4/4] Generating diagnostic plots...")

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#0f1923')

ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4)
ax5 = fig.add_subplot(3, 1, 3)
axes = [ax1, ax2, ax3, ax4, ax5]

for ax in axes:
    ax.set_facecolor('#131f2e')
    ax.tick_params(colors='#8fa8c8', labelsize=8)
    ax.yaxis.label.set_color('#8fa8c8')
    ax.title.set_color('#dce8f5')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a3f5a')
    ax.grid(True, color='#1e3050', linewidth=0.6, linestyle='--', alpha=0.7)

fig.suptitle(
    f'Demand Model - New England NSW  |  Random Forest\n'
    f'Train: Oct 2019-Sep 2022  |  Val: Oct 2022-Sep 2024  '
    f'|  Test: Oct 2024-Sep 2025\n'
    f'Test MAE: {test_mae/1000:.3f} MW  |  Test R2: {test_r2:.4f}',
    fontsize=11, color='#dce8f5', fontweight='bold'
)

# Plot 1: Scatter predicted vs actual (test)
ax = ax1
ax.scatter(y_test / 1000, test_pred / 1000,
           alpha=0.1, s=2, color='#4caf8a', rasterized=True)
lim = max(y_test.max(), test_pred.max()) / 1000 * 1.05
ax.plot([0, lim], [0, lim], 'r--', linewidth=1.0, alpha=0.8)
ax.set_xlabel('Actual (MW)')
ax.set_ylabel('Predicted (MW)')
ax.set_title(f'Predicted vs Actual - Test\nR2={test_r2:.3f}')
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)

# Plot 2: Error distribution
ax = ax2
errors_mw = (test_pred - y_test.values) / 1000
ax.hist(errors_mw, bins=80, color='#4fc3f7', alpha=0.75, edgecolor='none')
ax.axvline(0, color='#ff5252', linewidth=1.2, linestyle='--')
ax.axvline(errors_mw.mean(), color='#f5c842', linewidth=1.0,
           linestyle='--', label=f'Mean={errors_mw.mean():.3f} MW')
ax.set_xlabel('Error (MW)')
ax.set_ylabel('Count')
ax.set_title(f'Error Distribution - Test\nStd={errors_mw.std():.3f} MW')
ax.legend(fontsize=7, framealpha=0.2, facecolor='#0f1923',
          edgecolor='#2a3f5a', labelcolor='#dce8f5')

# Plot 3: Seasonal diurnal profiles
ax = ax3
test_result['hour']   = pd.to_datetime(test_result['Datetime']).dt.hour
test_result['month']  = pd.to_datetime(test_result['Datetime']).dt.month
test_result['season'] = test_result['month'].map({
    12:'Summer', 1:'Summer', 2:'Summer',
    3:'Autumn',  4:'Autumn', 5:'Autumn',
    6:'Winter',  7:'Winter', 8:'Winter',
    9:'Spring', 10:'Spring', 11:'Spring'
})
season_colors = {'Summer':'#f5a623', 'Autumn':'#e05c5c',
                 'Winter':'#4fc3f7', 'Spring':'#4caf8a'}
for season, color in season_colors.items():
    s = test_result[test_result['season'] == season]
    if s.empty:
        continue
    ax.plot(s.groupby('hour')['Actual_MW'].mean(),
            color=color, linewidth=1.8, label=f'{season} actual')
    ax.plot(s.groupby('hour')['Predicted_MW'].mean(),
            color=color, linewidth=1.2, linestyle='--', alpha=0.75)
ax.set_xlabel('Hour of day')
ax.set_ylabel('Average demand (MW)')
ax.set_title('Diurnal Profile by Season\n(solid=actual, dashed=predicted)')
ax.set_xticks(range(0, 24, 3))
ax.legend(fontsize=6.5, framealpha=0.2, facecolor='#0f1923',
          edgecolor='#2a3f5a', labelcolor='#dce8f5', ncol=2)

# Plot 4: Feature importance
ax = ax4
top_n = 12
top   = importance.head(top_n)
ax.barh(range(top_n), top['importance'].values,
        color='#4caf8a', alpha=0.8, edgecolor='none')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top['feature'].values, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_title(f'Top {top_n} Feature Importances')
ax.tick_params(axis='x', colors='#8fa8c8')

# Plot 5: 4-day sample from start of test set
ax = ax5
plot_start = pd.Timestamp(VAL_END)
plot_end   = plot_start + pd.Timedelta(days=4)
plot_data  = test_result[
    (test_result['Datetime'] >= plot_start) &
    (test_result['Datetime'] <= plot_end)
].sort_values('Datetime')

times = pd.to_datetime(plot_data['Datetime'])
ax.fill_between(times, plot_data['Actual_MW'], alpha=0.2, color='#4caf8a')
ax.plot(times, plot_data['Actual_MW'],
        color='#4caf8a', linewidth=1.8, label='Actual')
ax.plot(times, plot_data['Predicted_MW'],
        color='#f5c842', linewidth=1.3, linestyle='--', label='Predicted')
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
ax.tick_params(axis='x', rotation=25, colors='#8fa8c8', labelsize=7)
ax.set_ylabel('Demand (MW)')
ax.set_xlabel('Date / Time')
ax.set_title(f'Sample - First 4 Days of Test Set '
             f'({plot_start.date()} to {plot_end.date()})')
ax.set_ylim(bottom=0)
ax.legend(fontsize=8, framealpha=0.2, facecolor='#0f1923',
          edgecolor='#2a3f5a', labelcolor='#dce8f5')

plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"  Plot saved: {PLOT_FILE}")

if SHOW_PLOT:
    plt.show()

plt.close()

print("\n" + "="*60)
print(f"  Training complete")
print(f"  Test MAE : {test_mae/1000:.3f} MW")
print(f"  Test R2  : {test_r2:.4f}")
print(f"\n  Run demand_predict.py for predictions on any day and month")
print("="*60 + "\n")
