"""
Power Generation & Wind Turbine Dispatch Model
================================================
New England NSW — Sapphire & White Rock Wind Farms

Integrates wind forecast, solar generation, and local demand to
determine optimal turbine dispatch at 30-minute intervals, handling
surplus (curtailment/export) and deficit (import/unmet demand).

Inputs:
    wind_forecast.csv     — from wind_predict.py  (hourly)
    solar_irradiance.csv  — from solar_irradiance.py (configurable interval)
    demand_profile.csv    — from demand ML model (30-min intervals)
                            OR uses built-in synthetic demand if not available

Outputs:
    dispatch_log.csv      — full 30-min dispatch record
    dispatch_summary.csv  — daily summary statistics
    dispatch_plot.png     — visualisation

Requirements:
    pip install pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import warnings
from config import NE_THROUGH_FLOW_MW # type: ignore

warnings.filterwarnings('ignore')

# =============================================================
# USER INPUTS
# =============================================================

# --- Input files ---
WIND_FORECAST_FILE  = 'wind_forecast.csv'
SOLAR_FILE          = 'solar_irradiance_output.csv'

# Demand file — set to None to use synthetic demand profile
DEMAND_FILE         = 'demand_profile.csv'      # replace with 'demand_profile.csv' when ready

# --- Transmission parameters ---
NE_LINE_CAPACITY_EXPORT = 1000      # MW — physical line limit (QLD→NSW direction)
NE_LINE_CAPACITY_IMPORT = 1000       # MW — physical line limit (NSW→QLD direction)
NE_AVAILABLE_EXPORT     = NE_LINE_CAPACITY_EXPORT - NE_THROUGH_FLOW_MW  # 900 MW
NE_AVAILABLE_IMPORT     = NE_LINE_CAPACITY_IMPORT - NE_THROUGH_FLOW_MW  # 450 MW

# --- Wind farm parameters ---
SAPPHIRE_TURBINES       = 75
WHITE_ROCK_TURBINES     = 70
SAPPHIRE_CAPACITY_MW    = 270.0
WHITE_ROCK_CAPACITY_MW  = 175.0

# Hub height wind scaling
ERA5_HEIGHT             = 100       # ERA5 data height (m)
SAPPHIRE_HUB_HEIGHT     = 117       # Vestas V126 hub height (m)
WHITE_ROCK_HUB_HEIGHT   = 90        # Goldwind GW121 hub height (m)
ALPHA                   = 0.143     # Hellmann exponent

# Ramp rate — max turbines started or stopped per 30-min interval
MAX_RAMP_SAPPHIRE       = 7        # turbines per 30min
MAX_RAMP_WHITE_ROCK     = 7         # turbines per 30min

# Minimum turbines online for grid stability (even during low demand)
MIN_TURBINES_SAPPHIRE   = 5
MIN_TURBINES_WHITE_ROCK = 4

# Cut-out buffer — begin ramping down before hard cut-out to avoid sudden loss
CUTOUT_BUFFER_MS        = 5.0       # m/s below cut-out to start curtailing
curtailed_wind_mw  = 0.0
solar_curtailed_mw = 0.0

# --- Synthetic demand parameters (used if DEMAND_FILE is None) ---
# Estimated New England region demand in MW
DEMAND_BASE_MW          = 120       # average base load
DEMAND_PEAK_MW          = 220       # peak load (summer afternoon / winter morning)
DEMAND_TROUGH_MW        = 80        # overnight minimum

# --- Output ---
LOG_FILE                = 'dispatch_log.csv'
SUMMARY_FILE            = 'dispatch_summary.csv'
PLOT_FILE               = 'dispatch_plot.png'
SHOW_PLOT               = False

# =============================================================
# POWER CURVES
# =============================================================

# --- Vestas V126 3.6MW — digitised from manufacturer power curve chart ---
# Values read directly from published bar chart (kW at each 1 m/s interval)
# Rated power: ~3,570 kW  |  Cut-in: 3 m/s  |  Cut-out: 22 m/s
_V126_WS = np.array([
     0,  1,  2,    3,    4,    5,    6,    7,    8,
     9, 10,   11,   12,   13,   14,   15,   16,
    17, 18,   19,   20,   21,   22,   23,   24,
    25, 26
])
_V126_KW = np.array([
     0,  0,  0,   0,  140,  330,  600,  1000,  1500,
   2150, 2900, 3400, 3600, 3600, 3600, 3600, 3600,
  3600, 3600, 3600, 3600, 3600, 0, 0, 0,
  0,    0
])

def power_curve_vestas_v126(wind_speed_ms):
    """
    Vestas V126 ~3.57MW — Sapphire Wind Farm
    Digitised from manufacturer published power curve chart.
    Cut-in: ~3 m/s  |  Rated: ~13-14 m/s  |  Cut-out: 25 m/s
    Returns kW per turbine.
    """
    ws = np.asarray(wind_speed_ms, dtype=float)
    return np.interp(ws, _V126_WS, _V126_KW)


# --- Goldwind GW121 2.35MW — digitised from manufacturer power curve chart ---
# S-curve digitised at key inflection points.
# Rated power: ~2,350 kW  |  Cut-in: ~3 m/s  |  Cut-out: 22 m/s
# Note: published chart shows data to 20 m/s — extended flat to 22 m/s
# then zero per standard cut-out specification.
_GW121_WS = np.array([
     0,  1,    2,    3,    4,    5,    6,    7,
     8,  9,   10,   11,   12,   13,   14,   15,
    16, 17,   18,   19,   20,   21,   22,   23
])
_GW121_KW = np.array([
     0,  0,    0,   0,   140,  250,  530,  930,
  1800, 2400, 2500, 2500, 2500, 2500, 2500, 2500,
  2500, 2500, 2500, 2500, 2500, 2500,    0,    0
])

def power_curve_goldwind_gw121(wind_speed_ms):
    """
    Goldwind GW121 ~2.35MW — White Rock Wind Farm
    Digitised from manufacturer published power curve chart.
    Cut-in: ~3 m/s  |  Rated: ~10 m/s  |  Cut-out: 22 m/s
    Returns kW per turbine.
    """
    ws = np.asarray(wind_speed_ms, dtype=float)
    return np.interp(ws, _GW121_WS, _GW121_KW)

def scale_wind_to_hub(wind_speed, measured_height, hub_height, alpha=ALPHA):
    """Scale wind speed from ERA5 height to turbine hub height."""
    return wind_speed * (hub_height / measured_height) ** alpha


# =============================================================
# SYNTHETIC DEMAND PROFILE
# =============================================================

def synthetic_demand(timestamps):
    """
    Generate a synthetic 30-minute demand profile for New England NSW.
    Based on typical residential, agricultural, and industrial load patterns.
    Replace with real ML demand model output when available.

    Returns a Series of demand values in MW indexed by timestamp.
    """
    df = pd.DataFrame({'datetime': timestamps})
    df['hour']  = df['datetime'].dt.hour + df['datetime'].dt.minute / 60
    df['month'] = df['datetime'].dt.month

    # Base diurnal shape — dual-peak pattern (morning and evening)
    # using sum of two Gaussians
    morning_peak = np.exp(-0.5 * ((df['hour'] - 8.0)  / 1.5) ** 2)
    evening_peak = np.exp(-0.5 * ((df['hour'] - 18.5) / 1.5) ** 2)
    overnight    = np.exp(-0.5 * ((df['hour'] - 3.0)  / 2.0) ** 2) * 0.3

    diurnal = morning_peak + evening_peak + overnight
    diurnal = (diurnal - diurnal.min()) / (diurnal.max() - diurnal.min())

    # Seasonal amplitude — higher in winter (heating) and summer (cooling)
    # months 6,7,8 = winter peak; months 12,1,2 = summer peak
    season_factor = 1.0 + 0.25 * np.cos(
        2 * np.pi * (df['month'] - 1) / 12
    )

    # Agricultural irrigation load — summer mornings (Oct–Mar)
    irrigation = np.where(
        (df['month'].isin([10,11,12,1,2,3])) & (df['hour'] >= 6) & (df['hour'] <= 10),
        30.0, 0.0
    )

    demand = (
        DEMAND_TROUGH_MW
        + (DEMAND_PEAK_MW - DEMAND_TROUGH_MW) * diurnal * season_factor
        + irrigation
    )

    return pd.Series(demand.values, index=timestamps, name='demand_mw')


# =============================================================
# DISPATCH LOGIC
# =============================================================

def compute_turbines_needed(target_mw, ws_hub_sapphire, ws_hub_white_rock,
                             prev_sapphire, prev_white_rock):
    """
    Determine how many turbines to run at each farm to meet target_mw.

    Strategy:
        1. Calculate max output per turbine at current wind speed
        2. Proportionally split target between farms by capacity
        3. Round to nearest whole turbine
        4. Apply ramp rate limits
        5. Apply minimum online constraints

    Returns (sapphire_online, white_rock_online, total_mw)
    """
    # Output per turbine at current wind speed (MW)
    mw_per_sap = power_curve_vestas_v126(ws_hub_sapphire)   / 1000
    mw_per_wr  = power_curve_goldwind_gw121(ws_hub_white_rock) / 1000

    # Max available from each farm
    max_sap = mw_per_sap * SAPPHIRE_TURBINES
    max_wr  = mw_per_wr  * WHITE_ROCK_TURBINES
    max_total = max_sap + max_wr

    if max_total == 0:
        return 0, 0, 0.0

    # Target clipped to what's physically possible
    target_clipped = min(target_mw, max_total)
    target_clipped = max(target_clipped, 0)

    # Proportional split by available capacity
    sap_share = max_sap / max_total if max_total > 0 else 0.5
    wr_share  = 1 - sap_share

    # Turbines needed per farm
    sap_needed = (target_clipped * sap_share / mw_per_sap
                  if mw_per_sap > 0 else 0)
    wr_needed  = (target_clipped * wr_share  / mw_per_wr
                  if mw_per_wr  > 0 else 0)

    # Round to whole turbines
    sap_online = int(round(sap_needed))
    wr_online  = int(round(wr_needed))

    # Clamp to valid range
    sap_online = max(0, min(sap_online, SAPPHIRE_TURBINES))
    wr_online  = max(0, min(wr_online,  WHITE_ROCK_TURBINES))

    # Apply ramp rate limits
    sap_online = int(np.clip(sap_online,
                             prev_sapphire - MAX_RAMP_SAPPHIRE,
                             prev_sapphire + MAX_RAMP_SAPPHIRE))
    wr_online  = int(np.clip(wr_online,
                             prev_white_rock - MAX_RAMP_WHITE_ROCK,
                             prev_white_rock + MAX_RAMP_WHITE_ROCK))

    # Apply minimums (if wind is sufficient for cut-in)
    if mw_per_sap > 0:
        sap_online = max(sap_online, MIN_TURBINES_SAPPHIRE)
    if mw_per_wr > 0:
        wr_online  = max(wr_online,  MIN_TURBINES_WHITE_ROCK)

    # Actual output
    total_mw = (mw_per_sap * sap_online) + (mw_per_wr * wr_online)

    return sap_online, wr_online, total_mw


def curtail_to_headroom(total_mw, export_headroom,
                        ws_hub_sap, ws_hub_wr,
                        sap_online, wr_online):
    """
    If generation exceeds export headroom, curtail turbines discretely
    until generation fits within available export capacity.

    Alternates curtailment between farms to spread wear evenly.
    Returns (sap_online, wr_online, actual_mw, curtailed_mw)
    """
    mw_per_sap = power_curve_vestas_v126(ws_hub_sap)    / 1000
    mw_per_wr  = power_curve_goldwind_gw121(ws_hub_wr)  / 1000

    curtailed_mw = 0.0

    # Step down one turbine at a time, alternating farms
    farm_turn = 0   # 0 = Sapphire, 1 = White Rock

    while total_mw > export_headroom:
        if farm_turn == 0 and sap_online > MIN_TURBINES_SAPPHIRE:
            sap_online  -= 1
            removed      = mw_per_sap
            curtailed_mw += removed
            total_mw     -= removed
            farm_turn     = 1
        elif farm_turn == 1 and wr_online > MIN_TURBINES_WHITE_ROCK:
            wr_online    -= 1
            removed       = mw_per_wr
            curtailed_mw += removed
            total_mw     -= removed
            farm_turn     = 0
        else:
            # Both farms at minimum — can't curtail further
            break

    actual_mw = (mw_per_sap * sap_online) + (mw_per_wr * wr_online)
    return sap_online, wr_online, actual_mw, curtailed_mw


# =============================================================
# STEP 1 — Load inputs
# =============================================================

print("\n" + "="*60)
print("  Power Generation & Dispatch Model — New England NSW")
print("="*60)

print("\n[1/5] Loading inputs...")

# --- Wind forecast ---
wind_df = pd.read_csv(WIND_FORECAST_FILE, parse_dates=['datetime'])
wind_df = wind_df.sort_values('datetime').reset_index(drop=True)
print(f"  Wind forecast : {len(wind_df)} rows  "
      f"({wind_df['datetime'].min()} to {wind_df['datetime'].max()})")

# --- Solar ---
solar_df = pd.read_csv(SOLAR_FILE, parse_dates=['datetime'])
solar_df = solar_df.sort_values('datetime').reset_index(drop=True)
# Use farm_output_mw column — produced by solar_irradiance.py
if 'farm_output_mw' not in solar_df.columns:
    raise ValueError("solar_irradiance_output.csv must contain 'farm_output_mw' column. "
                     "Re-run solar_irradiance.py to regenerate.")
print(f"  Solar         : {len(solar_df)} rows  "
      f"({solar_df['datetime'].min()} to {solar_df['datetime'].max()})")

# =============================================================
# STEP 2 — Align all inputs to 30-minute index
# =============================================================

print(f"\n[2/5] Aligning inputs to 30-minute intervals...")

# Master 30-minute time index — driven by forecast window
t_start = wind_df['datetime'].min()
t_end   = wind_df['datetime'].max()
index_30min = pd.date_range(start=t_start, end=t_end, freq='30min')
index_30min = index_30min.tz_localize(None)   # add this line



# Resample wind: hourly → 30-min via linear interpolation
wind_series = wind_df.set_index('datetime')['wind_speed_predicted_ms']
wind_series.index = pd.to_datetime(wind_series.index).tz_localize(None)
#wind_series.index = pd.to_datetime(wind_series.index)

combined_index = wind_series.index.union(index_30min)
combined_index = pd.DatetimeIndex(combined_index)

wind_30min = (
    wind_series
    .reindex(combined_index)
    .interpolate(method='time')
    .reindex(index_30min)
)

solar_series = solar_df.set_index('datetime')['farm_output_mw']
solar_series.index = pd.to_datetime(solar_series.index).tz_localize(None)
#solar_series.index = pd.to_datetime(solar_series.index)

combined_index = solar_series.index.union(index_30min)
combined_index = pd.DatetimeIndex(combined_index)

solar_30min = (
    solar_series
    .reindex(combined_index)
    .interpolate(method='time')
    .reindex(index_30min)
    .clip(lower=0)
)

# Demand: load from file or generate synthetic profile
if DEMAND_FILE is not None:
    demand_df = pd.read_csv(DEMAND_FILE, parse_dates=['datetime'])
    demand_df['datetime'] = pd.to_datetime(demand_df['datetime']).dt.tz_localize(None)
    demand_df = demand_df.set_index('datetime')
    demand_df.index = pd.DatetimeIndex(demand_df.index)
    demand_30min = (
        demand_df['demand_mw']
        .reindex(demand_df.index.union(index_30min))
        .interpolate(method='time')
        .reindex(index_30min)
    )
    print(f"  Demand        : loaded from {DEMAND_FILE}")
else:
    demand_30min = synthetic_demand(index_30min)
    print(f"  Demand        : synthetic profile (replace with ML model output)")

print(f"  Time index    : {len(index_30min)} × 30-min intervals")
print(f"  Period        : {index_30min[0]} to {index_30min[-1]}")

# =============================================================
# STEP 3 — Dispatch loop
# =============================================================

print(f"\n[3/5] Running dispatch loop...")

records = []

# Initialise turbine counts based on first wind speed in the forecast
_ws0     = float(wind_30min.iloc[0])
_ws_sap0 = scale_wind_to_hub(_ws0, ERA5_HEIGHT, SAPPHIRE_HUB_HEIGHT)
_ws_wr0  = scale_wind_to_hub(_ws0, ERA5_HEIGHT, WHITE_ROCK_HUB_HEIGHT)

_mw_sap0 = power_curve_vestas_v126(_ws_sap0)    / 1000
_mw_wr0  = power_curve_goldwind_gw121(_ws_wr0)  / 1000

prev_sap = SAPPHIRE_TURBINES  if _mw_sap0 > 0 else 0
prev_wr  = WHITE_ROCK_TURBINES if _mw_wr0  > 0 else 0

for t in index_30min:
    wind_ms = float(wind_30min[t])
    solar_mw = float(solar_30min.get(t, 0.0))
    demand_mw = float(demand_30min[t])

    # Hub height scaling
    ws_sap = scale_wind_to_hub(wind_ms, ERA5_HEIGHT, SAPPHIRE_HUB_HEIGHT)
    ws_wr  = scale_wind_to_hub(wind_ms, ERA5_HEIGHT, WHITE_ROCK_HUB_HEIGHT)

    # Hard cut-out — zero turbines above 22 m/s
    CUTOUT_SPEED = 22.0

    # Hard cut-out check — must happen BEFORE power curve lookup
    # If wind is at or above cut-out, force turbines offline immediately
    if ws_sap >= CUTOUT_SPEED:
        ws_sap = 0.0    # force power curve to return 0
    if ws_wr >= CUTOUT_SPEED:
        ws_wr  = 0.0    # force power curve to return 0

    # Cut-out buffer — in the approach zone, scale back available turbines
    # proportionally so ramp-down begins before the hard cut-out hits
    elif ws_sap >= (CUTOUT_SPEED - CUTOUT_BUFFER_MS):
        # Scale: at buffer start (17 m/s) = full turbines, at 22 m/s = 0
        buffer_fraction = 1.0 - (ws_sap - (CUTOUT_SPEED - CUTOUT_BUFFER_MS)) / CUTOUT_BUFFER_MS
        prev_sap = int(round(prev_sap * buffer_fraction))

    if ws_wr >= (CUTOUT_SPEED - CUTOUT_BUFFER_MS) and ws_wr < CUTOUT_SPEED:
        buffer_fraction = 1.0 - (ws_wr - (CUTOUT_SPEED - CUTOUT_BUFFER_MS)) / CUTOUT_BUFFER_MS
        prev_wr = int(round(prev_wr * buffer_fraction))

    # Net demand after solar
    net_demand = max(demand_mw - solar_mw, 0)

    # Run maximum turbines the wind allows, subject to ramp rate limits
    # Goal is to maximise generation and export, not just meet local demand
    mw_per_sap = power_curve_vestas_v126(ws_sap)    / 1000
    mw_per_wr  = power_curve_goldwind_gw121(ws_wr)  / 1000

    # Maximum turbines available within ramp rate constraints
    sap_max = int(np.clip(SAPPHIRE_TURBINES,
                        prev_sap - MAX_RAMP_SAPPHIRE,
                        prev_sap + MAX_RAMP_SAPPHIRE))
    wr_max  = int(np.clip(WHITE_ROCK_TURBINES,
                        prev_wr  - MAX_RAMP_WHITE_ROCK,
                        prev_wr  + MAX_RAMP_WHITE_ROCK))

    # Apply minimum online constraints if wind is above cut-in
    sap_online = sap_max if mw_per_sap > 0 else 0
    wr_online  = wr_max  if mw_per_wr  > 0 else 0

    if mw_per_sap > 0:
        sap_online = max(sap_online, MIN_TURBINES_SAPPHIRE)
    if mw_per_wr > 0:
        wr_online  = max(wr_online,  MIN_TURBINES_WHITE_ROCK)

    wind_mw = (mw_per_sap * sap_online) + (mw_per_wr * wr_online)

    # Total local generation
    total_gen = wind_mw + solar_mw

    # --- Surplus / deficit handling ---
    surplus_mw    = max(total_gen - demand_mw, 0)
    deficit_mw    = max(demand_mw - total_gen, 0)
    curtailed_mw  = 0.0
    export_mw     = 0.0
    import_mw     = 0.0
    unmet_mw      = 0.0

    if surplus_mw > 0:
        # Step 1 — export surplus up to available headroom
        export_mw  = min(surplus_mw, NE_AVAILABLE_EXPORT)
        remaining  = surplus_mw - export_mw

        # Step 2 — curtail wind turbines if still surplus after max export
        if remaining > 0:
            sap_online, wr_online, wind_mw, curtailed_wind_mw = curtail_to_headroom(
                wind_mw, 0.2*(demand_mw + export_mw - solar_mw),
                ws_sap, ws_wr, sap_online, wr_online
            )
            curtailed_mw += curtailed_wind_mw
            total_gen     = wind_mw + solar_mw
            remaining     = max(total_gen - demand_mw - export_mw, 0)

        # Step 3 — curtail solar if surplus remains after wind curtailment
        if remaining > 0:
            solar_curtailed_mw = min(remaining, solar_mw)
            solar_mw           = max(solar_mw - solar_curtailed_mw, 0)
            curtailed_mw      += solar_curtailed_mw
            total_gen          = wind_mw + solar_mw
            remaining          = max(total_gen - demand_mw - export_mw, 0)

    elif deficit_mw > 0:
        # Import from grid up to available import capacity
        import_mw = min(deficit_mw, NE_AVAILABLE_IMPORT)
        unmet_mw  = max(deficit_mw - import_mw, 0)

    # Record this interval
    records.append({
        'datetime'          : t,
        'wind_speed_ms'     : round(wind_ms,  3),
        'ws_sapphire_hub'   : round(ws_sap,   3),
        'ws_white_rock_hub' : round(ws_wr,    3),
        'demand_mw'         : round(demand_mw, 2),
        'solar_mw'          : round(solar_mw,  2),
        'wind_mw'           : round(wind_mw,   2),
        'total_gen_mw'      : round(total_gen, 2),
        'sapphire_online'   : sap_online,
        'white_rock_online' : wr_online,
        'sapphire_pct'      : round(100 * sap_online / SAPPHIRE_TURBINES,  1),
        'white_rock_pct'    : round(100 * wr_online  / WHITE_ROCK_TURBINES, 1),
        'export_mw'         : round(export_mw,    2),
        'import_mw'         : round(import_mw,    2),
        'curtailed_wind_mw'  : round(curtailed_wind_mw,  2),
        'curtailed_solar_mw' : round(solar_curtailed_mw, 2),
        'curtailed_mw'       : round(curtailed_mw,        2),
        'unmet_demand_mw'   : round(unmet_mw,     2),
        'net_balance_mw'    : round(total_gen + import_mw - demand_mw - export_mw, 3),
    })

    prev_sap = sap_online
    prev_wr  = wr_online

log = pd.DataFrame(records)

print(f"  Dispatch intervals processed: {len(log)}")

# =============================================================
# STEP 4 — Summary statistics
# =============================================================

print(f"\n[4/5] Computing summary...")

interval_hours = 0.5   # 30-min intervals

total_wind_energy    = log['wind_mw'].sum()         * interval_hours
total_solar_energy   = log['solar_mw'].sum()        * interval_hours
total_demand_energy  = log['demand_mw'].sum()       * interval_hours
total_exported       = log['export_mw'].sum()       * interval_hours
total_imported       = log['import_mw'].sum()       * interval_hours
total_curtailed       = log['curtailed_mw'].sum()      * interval_hours
total_curtailed_wind  = log['curtailed_wind_mw'].sum() * interval_hours
total_curtailed_solar = log['curtailed_solar_mw'].sum()* interval_hours
total_unmet          = log['unmet_demand_mw'].sum() * interval_hours

curtailment_pct = 100 * total_curtailed / (total_wind_energy + total_curtailed) \
                  if (total_wind_energy + total_curtailed) > 0 else 0
unmet_pct       = 100 * total_unmet / total_demand_energy \
                  if total_demand_energy > 0 else 0

avg_sap_online  = log['sapphire_online'].mean()
avg_wr_online   = log['white_rock_online'].mean()

print("\n" + "="*60)
print("  DISPATCH SUMMARY")
print("="*60)
print(f"  Period         : {log['datetime'].min()} to {log['datetime'].max()}")
print(f"  Intervals      : {len(log)} × 30min")
print("-"*60)
print(f"  Wind generated : {total_wind_energy:,.0f} MWh")
print(f"  Solar generated: {total_solar_energy:,.0f} MWh")
print(f"  Total demand   : {total_demand_energy:,.0f} MWh")
print(f"  Exported       : {total_exported:,.0f} MWh")
print(f"  Imported       : {total_imported:,.0f} MWh")
print(f"  Curtailed wind : {total_curtailed_wind:,.0f} MWh")
print(f"  Curtailed solar: {total_curtailed_solar:,.0f} MWh")
print(f"  Total curtailed: {total_curtailed:,.0f} MWh  ({curtailment_pct:.1f}% of available)")
print(f"  Unmet demand   : {total_unmet:,.0f} MWh  ({unmet_pct:.2f}% of total demand)")
print("-"*60)
print(f"  Avg Sapphire online   : {avg_sap_online:.1f} / {SAPPHIRE_TURBINES} turbines "
      f"({100*avg_sap_online/SAPPHIRE_TURBINES:.1f}%)")
print(f"  Avg White Rock online : {avg_wr_online:.1f} / {WHITE_ROCK_TURBINES} turbines "
      f"({100*avg_wr_online/WHITE_ROCK_TURBINES:.1f}%)")
print("="*60)

# Daily summary
log['date'] = pd.to_datetime(log['datetime']).dt.date
daily = log.groupby('date').agg(
    wind_mwh        = ('wind_mw',          lambda x: x.sum() * interval_hours),
    solar_mwh       = ('solar_mw',         lambda x: x.sum() * interval_hours),
    demand_mwh      = ('demand_mw',        lambda x: x.sum() * interval_hours),
    exported_mwh    = ('export_mw',        lambda x: x.sum() * interval_hours),
    imported_mwh    = ('import_mw',        lambda x: x.sum() * interval_hours),
    curtailed_mwh   = ('curtailed_mw',     lambda x: x.sum() * interval_hours),
    unmet_mwh       = ('unmet_demand_mw',  lambda x: x.sum() * interval_hours),
    avg_sap_online  = ('sapphire_online',  'mean'),
    avg_wr_online   = ('white_rock_online','mean'),
    peak_wind_ms    = ('wind_speed_ms',    'max'),
).reset_index()

# =============================================================
# STEP 5 — Save outputs
# =============================================================

print(f"\n[5/5] Saving outputs...")

log.to_csv(LOG_FILE, index=False, float_format='%.3f')
print(f"  Dispatch log    : {LOG_FILE}")

daily.to_csv(SUMMARY_FILE, index=False, float_format='%.2f')
print(f"  Daily summary   : {SUMMARY_FILE}")

# =============================================================
# PLOTS
# =============================================================

times = pd.to_datetime(log['datetime'])
n_intervals = len(log)
multi_day = (times.max() - times.min()).total_seconds() > 86400

# =============================================================
# FIGURE 1 — FULL DISPATCH OVERVIEW
# =============================================================

fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

# White background
fig.patch.set_facecolor('white')

for ax in axes:

    ax.set_facecolor('white')

    ax.tick_params(colors='black', labelsize=8)

    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')

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
    str(times.min().date())
    if not multi_day
    else f"{times.min().date()} to {times.max().date()}"
)

fig.suptitle(
    f'New England NSW — Wind Dispatch Model\n{date_label}',
    fontsize=13,
    color='black',
    fontweight='bold',
    y=0.99
)

# -------------------------------------------------------------
# PANEL 1 — GENERATION STACK VS DEMAND
# -------------------------------------------------------------

ax = axes[0]

ax.fill_between(
    times,
    0,
    log['solar_mw'],
    alpha=0.7,
    color='gold',
    label='Solar'
)

ax.fill_between(
    times,
    log['solar_mw'],
    log['solar_mw'] + log['wind_mw'],
    alpha=0.7,
    color='forestgreen',
    label='Wind'
)

ax.plot(
    times,
    log['demand_mw'],
    color='red',
    linewidth=1.8,
    linestyle='--',
    label='Demand'
)

ax.set_ylabel('Power (MW)')
ax.set_title('Generation Stack vs Demand')

ax.legend(
    fontsize=8,
    framealpha=0.9,
    facecolor='white',
    edgecolor='black',
    loc='upper right'
)

ax.set_ylim(bottom=0)

# -------------------------------------------------------------
# PANEL 2 — TURBINES ONLINE
# -------------------------------------------------------------

ax = axes[1]

ax.fill_between(
    times,
    log['sapphire_online'],
    alpha=0.6,
    color='skyblue',
    label=f'Sapphire (max {SAPPHIRE_TURBINES})'
)

ax.fill_between(
    times,
    log['white_rock_online'],
    alpha=0.6,
    color='mediumpurple',
    label=f'White Rock (max {WHITE_ROCK_TURBINES})'
)

ax.axhline(
    SAPPHIRE_TURBINES,
    color='skyblue',
    linewidth=0.8,
    linestyle=':'
)

ax.axhline(
    WHITE_ROCK_TURBINES,
    color='mediumpurple',
    linewidth=0.8,
    linestyle=':'
)

ax.set_ylabel('Turbines online')

ax.set_title(
    'Turbine Dispatch — Sapphire & White Rock'
)

ax.legend(
    fontsize=8,
    framealpha=0.9,
    facecolor='white',
    edgecolor='black'
)

ax.set_ylim(
    0,
    max(SAPPHIRE_TURBINES, WHITE_ROCK_TURBINES) * 1.1
)

# -------------------------------------------------------------
# PANEL 3 — IMPORT / EXPORT / CURTAILMENT
# -------------------------------------------------------------

ax = axes[2]

ax.fill_between(
    times,
    log['export_mw'],
    alpha=0.5,
    color='forestgreen',
    label='Export to NSW grid'
)

ax.fill_between(
    times,
    -log['import_mw'],
    alpha=0.5,
    color='orange',
    label='Import from NSW grid'
)

ax.fill_between(
    times,
    log['curtailed_mw'],
    alpha=0.5,
    color='red',
    label='Curtailed wind'
)

ax.axhline(
    0,
    color='black',
    linewidth=0.8,
    linestyle='--'
)

ax.axhline(
    NE_AVAILABLE_EXPORT,
    color='forestgreen',
    linewidth=0.8,
    linestyle=':',
    label=f'Export limit ({NE_AVAILABLE_EXPORT} MW)'
)

ax.axhline(
    -NE_AVAILABLE_IMPORT,
    color='orange',
    linewidth=0.8,
    linestyle=':',
    label=f'Import limit ({NE_AVAILABLE_IMPORT} MW)'
)

ax.set_ylabel('MW')

ax.set_title(
    'Grid Interaction & Curtailment  (+ = Export, − = Import)'
)

ax.legend(
    fontsize=7.5,
    framealpha=0.9,
    facecolor='white',
    edgecolor='black',
    ncol=2
)

# -------------------------------------------------------------
# PANEL 4 — WIND SPEED
# -------------------------------------------------------------

ax = axes[3]

ax.fill_between(
    times,
    log['wind_speed_ms'],
    alpha=0.2,
    color='skyblue'
)

ax.plot(
    times,
    log['wind_speed_ms'],
    color='skyblue',
    linewidth=1.4,
    label='Wind speed @ 100m'
)

ax.axhline(
    3,
    color='forestgreen',
    linewidth=0.9,
    linestyle='--',
    label='Cut-in (3 m/s)'
)

ax.axhline(
    9,
    color='gold',
    linewidth=0.9,
    linestyle='--',
    label='GW121 rated (9 m/s)'
)

ax.axhline(
    13,
    color='orange',
    linewidth=0.9,
    linestyle='--',
    label='V126 rated (13 m/s)'
)

ax.axhline(
    22,
    color='red',
    linewidth=0.9,
    linestyle='--',
    label='Cut-out (22 m/s)'
)

ax.set_ylabel('Wind speed (m/s)')

ax.set_title(
    'Wind Speed with Turbine Operating Regions'
)

ax.set_ylim(bottom=0)

ax.legend(
    fontsize=7.5,
    framealpha=0.9,
    facecolor='white',
    edgecolor='black',
    ncol=2
)

# -------------------------------------------------------------
# X AXIS FORMATTING
# -------------------------------------------------------------

if multi_day:

    axes[3].xaxis.set_major_locator(
        mdates.DayLocator(interval=1)
    )

    axes[3].xaxis.set_major_formatter(
        mdates.DateFormatter('%d %b')
    )

    axes[3].set_xlabel('Date')

    for d in pd.date_range(
        times.min().normalize(),
        times.max().normalize() + pd.Timedelta(days=1),
        freq='D'
    ):

        for ax in axes:

            ax.axvline(
                d,
                color='gray',
                linewidth=0.6,
                alpha=0.5
            )

else:

    axes[3].xaxis.set_major_locator(
        mdates.HourLocator(interval=2)
    )

    axes[3].xaxis.set_major_formatter(
        mdates.DateFormatter('%H:%M')
    )

    axes[3].set_xlabel('Time of day')

axes[3].tick_params(
    axis='x',
    colors='black',
    labelsize=8
)

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=150,
    bbox_inches='tight',
    facecolor='white'
)

print(f"  Full dispatch plot saved : {PLOT_FILE}")

if SHOW_PLOT:
    plt.show()

plt.close()


# =============================================================
# FIGURE 2 — GENERATION VS DEMAND ONLY
# =============================================================

fig, ax = plt.subplots(figsize=(16, 6))

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
    0,
    log['solar_mw'],
    alpha=0.7,
    color='gold',
    label='Solar'
)

ax.fill_between(
    times,
    log['solar_mw'],
    log['solar_mw'] + log['wind_mw'],
    alpha=0.7,
    color='forestgreen',
    label='Wind'
)

ax.plot(
    times,
    log['demand_mw'],
    color='red',
    linewidth=1.8,
    linestyle='--',
    label='Demand'
)

ax.set_title(
    'Generation Stack vs Demand',
    fontsize=13,
    fontweight='bold'
)

ax.set_ylabel('Power (MW)')
ax.set_xlabel('Time')

ax.legend()

generation_file = PLOT_FILE.replace(
    '.png',
    '_generation.png'
)

plt.tight_layout()

plt.savefig(
    generation_file,
    dpi=150,
    bbox_inches='tight',
    facecolor='white'
)

print(f"  Generation plot saved : {generation_file}")

if SHOW_PLOT:
    plt.show()

plt.close()


# =============================================================
# FIGURE 3 — GRID INTERACTION ONLY
# =============================================================

fig, ax = plt.subplots(figsize=(16, 6))

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
    log['export_mw'],
    alpha=0.5,
    color='forestgreen',
    label='Export'
)

ax.fill_between(
    times,
    -log['import_mw'],
    alpha=0.5,
    color='orange',
    label='Import'
)

ax.fill_between(
    times,
    log['curtailed_mw'],
    alpha=0.5,
    color='red',
    label='Curtailment'
)

ax.axhline(
    0,
    color='black',
    linestyle='--',
    linewidth=0.8
)

ax.set_title(
    'Grid Interaction & Curtailment',
    fontsize=13,
    fontweight='bold'
)

ax.set_ylabel('Power (MW)')
ax.set_xlabel('Time')

ax.legend()

grid_file = PLOT_FILE.replace(
    '.png',
    '_grid.png'
)

plt.tight_layout()

plt.savefig(
    grid_file,
    dpi=150,
    bbox_inches='tight',
    facecolor='white'
)

print(f"  Grid interaction plot saved : {grid_file}")

if SHOW_PLOT:
    plt.show()

plt.close()


# =============================================================
# SUMMARY
# =============================================================

print("\n" + "="*60)
print(f"  Dispatch complete — {len(log)} intervals processed")
print(f"  Curtailment rate : {curtailment_pct:.1f}%")
print(f"  Unmet demand     : {unmet_pct:.2f}%")
print(f"  Net balance check: {log['net_balance_mw'].abs().max():.3f} MW max error")
print("="*60 + "\n")