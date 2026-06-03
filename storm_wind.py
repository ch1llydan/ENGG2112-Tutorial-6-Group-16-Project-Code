"""
Synthetic Storm Wind File Generator
=====================================
Generates a synthetic hourly wind forecast CSV simulating a stormy day
with winds reaching 80-100 km/h (22-28 m/s).

This file is used to test turbine cut-out dispatch logic in
power_generation.py — replacing wind_forecast.csv for testing.

Wind profile:
    00:00-05:00  Building storm — moderate winds ramping up
    06:00-10:00  Storm arrival  — rapid ramp to peak
    11:00-14:00  Storm peak     — winds above cut-out (22+ m/s)
    15:00-18:00  Storm easing   — winds dropping back through cut-out
    19:00-23:00  Post-storm     — settling to elevated but manageable winds

Output: wind_forecast.csv (overwrites existing file)
        Columns match wind_predict.py output exactly so
        power_generation.py reads it without any changes.
"""

import pandas as pd
import numpy as np

# =============================================================
# USER INPUTS
# =============================================================

# Date to simulate — uses config if available, otherwise this default
try:
    from config import FORECAST_DATE_STR #type: ignore
    DATE = FORECAST_DATE_STR
except ImportError:
    DATE = '2025-07-15'

# Storm profile — wind speeds in m/s at each hour
# 80 km/h = 22.2 m/s  (cut-out threshold)
# 90 km/h = 25.0 m/s
# 100 km/h = 27.8 m/s
HOURLY_WIND_MS = {
     0:  8.5,    # overnight base — moderate wind
     1:  9.0,
     2:  9.8,
     3: 11.2,    # pre-dawn strengthening
     4: 13.5,
     5: 16.0,    # storm approaching
     6: 18.5,
     7: 20.8,    # rapid ramp — approaching cut-out
     8: 22.5,    # ABOVE CUT-OUT — turbines shutting down
     9: 24.8,
    10: 26.2,    # peak storm (~94 km/h)
    11: 27.5,    # near 100 km/h
    12: 27.8,    # peak (~100 km/h)
    13: 26.5,
    14: 24.1,    # still above cut-out
    15: 22.8,
    16: 21.0,    # dropping below cut-out — turbines restarting
    17: 19.5,
    18: 17.2,    # storm easing
    19: 15.0,
    20: 13.8,
    21: 12.5,
    22: 11.0,
    23:  9.5,    # post-storm settling
}

# Add realistic gusts and variability using small random perturbations
# Seed for reproducibility
RANDOM_SEED     = 42
GUST_AMPLITUDE  = 1.2    # m/s — standard deviation of gust noise

OUTPUT_FILE     = 'wind_forecast.csv'

# =============================================================
# GENERATE HOURLY PROFILE
# =============================================================

np.random.seed(RANDOM_SEED)

hours      = list(range(24))
base_winds = [HOURLY_WIND_MS[h] for h in hours]

# Add smooth correlated noise to simulate gusts
# Use a simple AR(1) process so gusts are correlated hour-to-hour
noise = np.zeros(24)
noise[0] = np.random.normal(0, GUST_AMPLITUDE)
for i in range(1, 24):
    noise[i] = 0.6 * noise[i-1] + np.random.normal(0, GUST_AMPLITUDE * 0.8)

wind_speeds = np.array(base_winds) + noise
wind_speeds = np.maximum(wind_speeds, 0)    # no negative wind speeds

# =============================================================
# BUILD DATAFRAME
# =============================================================

timestamps = pd.date_range(start=DATE, periods=24, freq='h')

# Generate synthetic U and V components from wind speed
# Assume predominantly westerly storm (wind coming from west = positive U)
# Direction varies slightly through the storm
direction_deg = 270 + noise * 5    # westerly with small variations

u100 = wind_speeds * np.cos(np.radians(270 - direction_deg))
v100 = wind_speeds * np.sin(np.radians(270 - direction_deg))

df = pd.DataFrame({
    'datetime'                : timestamps,
    'wind_speed_predicted_ms' : np.round(wind_speeds, 3),
    'wind_speed_actual_ms'    : np.round(wind_speeds, 3),  # same as predicted for synthetic
    'u100'                    : np.round(u100, 3),
    'v100'                    : np.round(v100, 3),
})

# =============================================================
# PRINT PROFILE
# =============================================================

print("\n" + "="*60)
print("  Synthetic Storm Wind Profile")
print("="*60)
print(f"  Date        : {DATE}")
print(f"  Output file : {OUTPUT_FILE}")
print(f"\n  {'Hour':<6} {'m/s':>6} {'km/h':>7}  {'Status'}")
print(f"  {'-'*45}")

for _, row in df.iterrows():
    t     = pd.Timestamp(row['datetime'])
    ms    = row['wind_speed_predicted_ms']
    kmh   = ms * 3.6

    if ms >= 27:
        status = 'STORM PEAK - all turbines cut out'
    elif ms >= 22:
        status = 'ABOVE CUT-OUT - turbines shut down'
    elif ms >= 20:
        status = 'Approaching cut-out - ramping down'
    elif ms >= 13:
        status = 'Vestas rated - full Sapphire output'
    elif ms >= 9:
        status = 'Goldwind rated - full White Rock output'
    elif ms >= 3:
        status = 'Partial generation'
    else:
        status = 'Below cut-in - no generation'

    marker = ' <-- CUT-OUT' if ms >= 22 else ''
    print(f"  {t.strftime('%H:%M'):<6} {ms:>6.1f} {kmh:>7.1f}  {status}{marker}")

peak_ms  = df['wind_speed_predicted_ms'].max()
peak_kmh = peak_ms * 3.6
cutout_hours = (df['wind_speed_predicted_ms'] >= 22).sum()

print(f"\n  Peak wind speed  : {peak_ms:.1f} m/s  ({peak_kmh:.0f} km/h)")
print(f"  Hours above cut-out (>=22 m/s): {cutout_hours} hours")
print(f"  Expected behaviour:")
print(f"    - Turbines ramp up during 00:00-07:00")
print(f"    - Cut-out triggered ~08:00 as wind exceeds 22 m/s")
print(f"    - All turbines offline 08:00-15:00")
print(f"    - Gradual restart from ~16:00 as wind drops below 22 m/s")
print(f"    - Ramp rate limits will slow restart (max {10} turbines/hour)")
print(f"    - Full generation not restored until ~19:00-20:00")

# =============================================================
# SAVE
# =============================================================

df.to_csv(OUTPUT_FILE, index=False, float_format='%.3f')
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"  Run power_generation.py to see dispatch response.")
print("="*60 + "\n")
