"""
Interactive Presentation Dashboard Generator
==============================================
New England NSW — Wind Dispatch Model

Runs the full pipeline for multiple scenarios and generates a
single self-contained HTML file for live presentation demos.


Usage:
    1. Set your scenarios in USER INPUTS below
    2. Run: python generate_dashboard.py
    3. Open new_england_dashboard.html in your browser

Requirements:
    All pipeline scripts must be in the same folder.
    Models must already be trained (wind_model.json, demand_model.pkl).
"""

import subprocess
import sys
import os
import base64
import json
import shutil
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

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

# =============================================================
# USER INPUTS
# =============================================================

SCENARIOS = [
    {
        'name'  : 'Summer Day',
        'day'   : 15,
        'month' : 1,
        'year'  : 2025,
        'storm' : False,
        'icon'  : '☀️',
        'desc'  : 'January — peak solar, moderate wind, high agricultural load',
    },
    {
        'name'  : 'Autumn Day',
        'day'   : 15,
        'month' : 4,
        'year'  : 2025,
        'storm' : False,
        'icon'  : '🍂',
        'desc'  : 'April — transitional solar, variable wind patterns',
    },
    {
        'name'  : 'Winter Day',
        'day'   : 15,
        'month' : 7,
        'year'  : 2025,
        'storm' : False,
        'icon'  : '❄️',
        'desc'  : 'July — low solar, strong westerly winds, high heating demand',
    },
    {
        'name'  : 'Spring Day',
        'day'   : 15,
        'month' : 10,
        'year'  : 2025,
        'storm' : False,
        'icon'  : '🌿',
        'desc'  : 'October — balanced solar and wind, moderate demand',
    },
    {
        'name'  : 'Storm - High Winds',
        'day'   : 15,
        'month' : 7,
        'year'  : 2025,
        'storm' : True,
        'icon'  : '⛈️',
        'desc'  : 'Severe storm — winds 80-100 km/h, turbine cut-out triggered',
    },
]

OUTPUT_HTML = 'new_england_dashboard.html'
TEMP_DIR    = '_dashboard_temp'

# =============================================================
# FIXED Y-AXIS LIMITS — consistent across all scenarios
# =============================================================

YLIM_SOLAR_MW    = (0, 700)      # Solar farm output MW
YLIM_DEMAND_MW   = (0, 200)      # Regional demand MW
YLIM_WIND_MS     = (0, 30)       # Wind speed m/s
YLIM_GEN_MW      = (0, 700)      # Generation stack MW
YLIM_TURBINES    = (0, 85)       # Turbines online
YLIM_ERROR_MW    = (-30, 30)     # Demand error MW
YLIM_WIND_ERR    = (-2.5, 2.5)       # Wind prediction error m/s

# =============================================================
# HELPERS
# =============================================================

def write_config(day, month, year, through_flow=350):
    """Write config.py for this scenario."""
    with open('config.py', 'w') as f:
        f.write(f"""# =============================================================
# PIPELINE CONFIGURATION — change date here to run any day
# =============================================================

FORECAST_DAY   = {day}
FORECAST_MONTH = {month}
FORECAST_YEAR  = {year}

# Derived formats used by different scripts
FORECAST_DATE_STR = '{year}-{month:02d}-{day:02d}'

# Transmission — constant QNI through-flow assumption
# Adjust this value if you want to test different through-flow scenarios
NE_THROUGH_FLOW_MW = {through_flow}
""")


def run(script):
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"\n    WARNING: {script} error:")
        print(result.stderr[-400:] if result.stderr else '(no stderr)')
    return result.returncode == 0


def img_to_base64(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    return f'data:image/png;base64,{data}'


def safe_read_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def parse_dt(df, col):
    """Parse datetime column, strip timezone."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
    if df[col].dt.tz is not None:
        df[col] = df[col].dt.tz_localize(None)
    return df


def style_ax(ax):
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
    ax.grid(True, color='#eeeeee', linewidth=0.7, linestyle='--')
    ax.tick_params(colors='black', labelsize=8)


def fmt_x_hourly(ax, interval=2):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
    ax.tick_params(axis='x', rotation=0)

# =============================================================
# MAIN LOOP
# =============================================================

os.makedirs(TEMP_DIR, exist_ok=True)
scenario_data = []

print("\n" + "="*60)
print("  Dashboard Generator — New England NSW")
print("="*60)
print(f"  Scenarios : {len(SCENARIOS)}")
print(f"  Output    : {OUTPUT_HTML}\n")

for idx, sc in enumerate(SCENARIOS):
    print(f"\n  [{idx+1}/{len(SCENARIOS)}] {sc['icon']}  {sc['name']}")
    print(f"    {sc['day']:02d}/{sc['month']:02d}/{sc['year']}  "
          f"{'— STORM MODE' if sc['storm'] else ''}")

    write_config(sc['day'], sc['month'], sc['year'])

    print(f"    solar...",   end='', flush=True)
    run('solar_irradiance.py');  print(" OK", end='')
    print(f"  demand...",   end='', flush=True)
    run('demand_predict.py');    print(" OK", end='')

    if sc['storm']:
        print(f"  storm wind...", end='', flush=True)
        run('storm_wind.py');    print(" OK", end='')
    else:
        print(f"  wind...",  end='', flush=True)
        run('wind_predict.py');  print(" OK", end='')

    print(f"  dispatch...", end='', flush=True)
    run('power_generation.py');  print(" OK")

    # ── Load outputs ──────────────────────────────────────────
    solar_df    = safe_read_csv('solar_irradiance_output.csv')
    demand_df   = safe_read_csv('demand_profile.csv')
    wind_df     = safe_read_csv('wind_forecast.csv')
    dispatch_df = safe_read_csv('dispatch_log.csv')

    if not solar_df.empty:
        solar_df = parse_dt(solar_df, 'datetime')
    if not demand_df.empty:
        demand_df = parse_dt(demand_df, 'datetime')
    if not wind_df.empty:
        wind_df = parse_dt(wind_df, 'datetime')
    if not dispatch_df.empty:
        dispatch_df = parse_dt(dispatch_df, 'datetime')

    # ── Build figure ──────────────────────────────────────────
    print(f"    Building figure...", end='', flush=True)

    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor('white')
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.52, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])   # solar output
    ax2 = fig.add_subplot(gs[0, 1])   # demand
    ax3 = fig.add_subplot(gs[1, :])   # wind speed
    ax4 = fig.add_subplot(gs[2, :])   # generation stack
    ax5 = fig.add_subplot(gs[3, :])   # turbines online
    ax6 = fig.add_subplot(gs[4, 0])   # demand prediction error
    ax7 = fig.add_subplot(gs[4, 1])   # wind prediction error

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        style_ax(ax)

    date_str = f"{sc['day']:02d}/{sc['month']:02d}/{sc['year']}"
    fig.suptitle(
        f"{sc['icon']}  {sc['name']}  —  {date_str}\n"
        f"New England NSW Wind Dispatch Model",
        fontsize=14, fontweight='bold', color='black', y=0.995
    )

    # ── Panel 1: Solar output ─────────────────────────────────
    if not solar_df.empty and 'farm_output_mw' in solar_df.columns:
        t = solar_df['datetime']
        ax1.fill_between(t, solar_df['farm_output_mw'],
                         alpha=0.4, color='#f5a623')
        ax1.plot(t, solar_df['farm_output_mw'],
                 color='#e67e00', linewidth=1.8)
        ax1.set_ylabel('MW')
        ax1.set_title('Solar Generation — New England Solar Farm')
        ax1.set_ylim(YLIM_SOLAR_MW)
        fmt_x_hourly(ax1, interval=4)

    # ── Panel 2: Demand ───────────────────────────────────────
    if not demand_df.empty and 'demand_mw' in demand_df.columns:
        t = demand_df['datetime']
        ax2.fill_between(t, demand_df['demand_mw'],
                         alpha=0.3, color='#e05c5c')
        ax2.plot(t, demand_df['demand_mw'],
                 color='#c0392b', linewidth=1.8)
        ax2.set_ylabel('MW')
        ax2.set_title('Predicted Regional Demand')
        ax2.set_ylim(YLIM_DEMAND_MW)
        fmt_x_hourly(ax2, interval=4)

    # ── Panel 3: Wind speed ───────────────────────────────────
    if not wind_df.empty and 'wind_speed_predicted_ms' in wind_df.columns:
        t  = wind_df['datetime']
        ws = wind_df['wind_speed_predicted_ms']
        ax3.fill_between(t, ws, alpha=0.2, color='#2980b9')
        ax3.plot(t, ws, color='#2980b9', linewidth=1.8,
                 label='Predicted wind speed @ 100m')
        # Dashed threshold lines as requested
        ax3.axhline(3,  color='#27ae60', linewidth=1.1,
                    linestyle='--', label='Cut-in (3 m/s)')
        ax3.axhline(10, color='#f39c12', linewidth=1.1,
                    linestyle='--', label='GW121 rated (~10 m/s)')
        ax3.axhline(14, color='#e67e22', linewidth=1.1,
                    linestyle='--', label='V126 rated (~14 m/s)')
        ax3.axhline(22, color='#e74c3c', linewidth=1.1,
                    linestyle='--', label='Cut-out (22 m/s)')
        ax3.set_ylabel('m/s')
        ax3.set_title('Wind Speed Forecast with Turbine Operating Regions')
        ax3.set_ylim(YLIM_WIND_MS)
        ax3.legend(fontsize=7.5, ncol=4, framealpha=0.9,
                   facecolor='white', edgecolor='#cccccc',
                   loc='upper right')
        fmt_x_hourly(ax3, interval=2)

    # ── Panel 4: Generation stack vs demand ───────────────────
    if not dispatch_df.empty:
        t = dispatch_df['datetime']
        ax4.fill_between(t, 0, dispatch_df['solar_mw'],
                         alpha=0.75, color='#f5c842', label='Solar')
        ax4.fill_between(t,
                         dispatch_df['solar_mw'],
                         dispatch_df['solar_mw'] + dispatch_df['wind_mw'],
                         alpha=0.65, color='#27ae60', label='Wind')
        if 'import_mw' in dispatch_df.columns and dispatch_df['import_mw'].sum() > 0:
            ax4.fill_between(t,
                             dispatch_df['solar_mw'] + dispatch_df['wind_mw'],
                             dispatch_df['solar_mw'] + dispatch_df['wind_mw']
                             + dispatch_df['import_mw'],
                             alpha=0.5, color='#9b59b6', label='Import')
        if 'curtailed_mw' in dispatch_df.columns and dispatch_df['curtailed_mw'].sum() > 0:
            ax4.fill_between(t,
                             0,
                             dispatch_df['curtailed_mw'],
                             alpha=0.5, color='#e74c3c', label='Curtailed')
        ax4.axhline(650, color='#e74c3c', linewidth=1.1,
                    linestyle='--', label='Max Export (650MW)')
        ax4.plot(t, dispatch_df['demand_mw'],
                 color='#c0392b', linewidth=2.0,
                 linestyle='--', label='Demand')
        ax4.set_ylabel('MW')
        ax4.set_title('Generation Stack vs Demand')
        ax4.set_ylim(YLIM_GEN_MW)
        ax4.legend(fontsize=8, ncol=4, framealpha=0.9,
                   facecolor='white', edgecolor='#cccccc',
                   loc='upper right')
        fmt_x_hourly(ax4, interval=2)

    # ── Panel 5: Turbines online ──────────────────────────────
    if not dispatch_df.empty:
        t = dispatch_df['datetime']
        ax5.fill_between(t, dispatch_df['sapphire_online'],
                         alpha=0.55, color='#2980b9',
                         label=f'Sapphire (max 75)')
        ax5.fill_between(t, dispatch_df['white_rock_online'],
                         alpha=0.55, color='#8e44ad',
                         label=f'White Rock (max 70)')
        ax5.axhline(75, color='#2980b9', linewidth=0.9,
                    linestyle='--', alpha=0.6)
        ax5.axhline(70, color='#8e44ad', linewidth=0.9,
                    linestyle='--', alpha=0.6)
        ax5.set_ylabel('Turbines online')
        ax5.set_title('Turbine Dispatch — Sapphire & White Rock')
        ax5.set_ylim(YLIM_TURBINES)
        ax5.legend(fontsize=8, ncol=2, framealpha=0.9,
                   facecolor='white', edgecolor='#cccccc')
        fmt_x_hourly(ax5, interval=2)

    # ── Panel 6: Demand prediction error ─────────────────────
    if not demand_df.empty and 'demand_actual_mw' in demand_df.columns:
        t     = demand_df['datetime']
        error = demand_df['demand_mw'] - demand_df['demand_actual_mw']
        ax6.fill_between(t, error, 0,
                         where=(error >= 0), alpha=0.45,
                         color='#e74c3c', label='Over-predicted')
        ax6.fill_between(t, error, 0,
                         where=(error < 0), alpha=0.45,
                         color='#3498db', label='Under-predicted')
        ax6.plot(t, error, color='black', linewidth=0.9, alpha=0.6)
        ax6.axhline(0, color='black', linewidth=0.9, alpha=0.4)
        mae_d = np.mean(np.abs(error))
        ax6.set_ylabel('Error (MW)')
        ax6.set_title(f'Demand Prediction Error  (MAE = {mae_d:.3f} MW)')
        ax6.set_ylim(YLIM_ERROR_MW)
        ax6.legend(fontsize=7.5, framealpha=0.9,
                   facecolor='white', edgecolor='#cccccc')
        fmt_x_hourly(ax6, interval=4)
    else:
        ax6.text(0.5, 0.5, 'Demand actual not available\n(no demand_actual_mw column)',
                 ha='center', va='center', transform=ax6.transAxes,
                 color='#888', fontsize=9)
        ax6.set_title('Demand Prediction Error')

    # ── Panel 7: Wind prediction error ───────────────────────
    if (not wind_df.empty
            and 'wind_speed_predicted_ms' in wind_df.columns
            and 'wind_speed_actual_ms' in wind_df.columns):
        t          = wind_df['datetime']
        wind_error = (wind_df['wind_speed_predicted_ms']
                      - wind_df['wind_speed_actual_ms'])
        ax7.fill_between(t, wind_error, 0,
                         where=(wind_error >= 0), alpha=0.45,
                         color='#e74c3c', label='Over-predicted')
        ax7.fill_between(t, wind_error, 0,
                         where=(wind_error < 0), alpha=0.45,
                         color='#3498db', label='Under-predicted')
        ax7.plot(t, wind_error, color='black', linewidth=0.9, alpha=0.6)
        ax7.axhline(0, color='black', linewidth=0.9, alpha=0.4)
        mae_w = np.mean(np.abs(wind_error))
        ax7.set_ylabel('Error (m/s)')
        ax7.set_title(f'Wind Prediction Error  (MAE = {mae_w:.3f} m/s)')
        ax7.set_ylim(YLIM_WIND_ERR)
        ax7.legend(fontsize=7.5, framealpha=0.9,
                   facecolor='white', edgecolor='#cccccc')
        fmt_x_hourly(ax7, interval=4)
    else:
        ax7.text(0.5, 0.5,
                 'Wind actual not available\n(run wind_predict.py, not storm_wind.py)',
                 ha='center', va='center', transform=ax7.transAxes,
                 color='#888', fontsize=9)
        ax7.set_title('Wind Prediction Error')

    # ── Save ──────────────────────────────────────────────────
    fig_path = os.path.join(TEMP_DIR, f'scenario_{idx}.png')
    plt.savefig(fig_path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close()
    print(" OK")

    # ── Stats ─────────────────────────────────────────────────
    stats = {}
    if not dispatch_df.empty:
        ih = 0.5
        stats['wind_mwh']    = round(dispatch_df['wind_mw'].sum()      * ih, 0)
        stats['solar_mwh']   = round(dispatch_df['solar_mw'].sum()     * ih, 0)
        stats['demand_mwh']  = round(dispatch_df['demand_mw'].sum()    * ih, 0)
        stats['export_mwh']  = round(dispatch_df['export_mw'].sum()    * ih, 0)
        stats['import_mwh']  = round(dispatch_df['import_mw'].sum()    * ih, 0)
        stats['curtail_mwh'] = round(dispatch_df['curtailed_mw'].sum() * ih, 0)
        stats['avg_sap']     = round(dispatch_df['sapphire_online'].mean(), 1)
        stats['avg_wr']      = round(dispatch_df['white_rock_online'].mean(), 1)
        stats['peak_wind']   = (round(wind_df['wind_speed_predicted_ms'].max(), 1)
                                if not wind_df.empty else 0)
        stats['peak_solar']  = (round(solar_df['farm_output_mw'].max(), 1)
                                if not solar_df.empty and 'farm_output_mw' in solar_df.columns
                                else 0)
        stats['peak_demand'] = (round(demand_df['demand_mw'].max(), 1)
                                if not demand_df.empty else 0)

    scenario_data.append({
        'name'  : sc['name'],
        'icon'  : sc['icon'],
        'desc'  : sc['desc'],
        'date'  : f"{sc['day']:02d}/{sc['month']:02d}/{sc['year']}",
        'storm' : sc['storm'],
        'stats' : stats,
        'img'   : img_to_base64(fig_path),
    })

# =============================================================
# BUILD HTML
# =============================================================

print(f"\n  Building HTML...", end='', flush=True)

scenarios_json = json.dumps(scenario_data)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>New England NSW — Wind Dispatch Model</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg      : #f4f6f9;
    --surface : #ffffff;
    --border  : #dde1e7;
    --accent  : #1a6b3a;
    --accent2 : #e67e00;
    --accent3 : #2471a3;
    --storm   : #c0392b;
    --text    : #1c2128;
    --muted   : #64748b;
    --radius  : 10px;
    --shadow  : 0 1px 12px rgba(0,0,0,0.06);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Lexend', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }}

  /* Header */
  .header {{
    background: var(--surface);
    border-bottom: 2px solid var(--border);
    padding: 20px 40px;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: var(--shadow);
  }}
  .header-title {{
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.01em;
  }}
  .header-sub {{
    font-size: 0.76rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    margin-top: 3px;
  }}

  /* Container */
  .container {{
    max-width: 1440px;
    margin: 0 auto;
    padding: 30px 40px;
  }}

  /* Tabs */
  .tab-bar {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 26px;
  }}
  .tab {{
    padding: 9px 20px;
    border-radius: 40px;
    border: 2px solid var(--border);
    background: var(--surface);
    cursor: pointer;
    font-family: 'Lexend', sans-serif;
    font-weight: 500;
    font-size: 0.86rem;
    color: var(--muted);
    transition: all 0.16s ease;
    display: flex;
    align-items: center;
    gap: 6px;
    user-select: none;
  }}
  .tab:hover         {{ border-color: var(--accent); color: var(--accent); background: #f0faf4; }}
  .tab.active        {{ background: var(--accent); border-color: var(--accent); color: white; font-weight: 600; }}
  .tab.storm         {{ border-color: var(--storm); color: var(--storm); }}
  .tab.storm:hover   {{ background: #fdf2f1; }}
  .tab.storm.active  {{ background: var(--storm); border-color: var(--storm); color: white; }}

  /* Panel */
  .scenario-panel       {{ display: none; animation: fadeUp 0.22s ease; }}
  .scenario-panel.active{{ display: block; }}
  @keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}

  /* Info bar */
  .info-bar    {{ margin-bottom: 20px; }}
  .info-title  {{
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }}
  .info-desc   {{ font-size: 0.85rem; color: var(--muted); margin-top: 5px; font-weight: 300; }}
  .storm-badge {{
    background: #fdf2f1;
    color: var(--storm);
    border: 1.5px solid var(--storm);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }}

  /* Stats — two centred rows */
  .stats-wrapper {{
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 24px;
  }}
  .stats-row {{
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .stat-card {{
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
    min-width: 170px;
    flex: 0 1 170px;
    transition: box-shadow 0.14s;
  }}
  .stat-card:hover {{ box-shadow: var(--shadow); }}
  .stat-label {{
    font-size: 0.68rem;
    font-family: 'DM Mono', monospace;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 5px;
  }}
  .stat-value {{
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
  }}
  .stat-unit {{
    font-size: 0.7rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    margin-left: 3px;
    font-weight: 400;
  }}
  .stat-card.green  {{ border-left: 4px solid var(--accent);  }}
  .stat-card.orange {{ border-left: 4px solid var(--accent2); }}
  .stat-card.blue   {{ border-left: 4px solid var(--accent3); }}
  .stat-card.red    {{ border-left: 4px solid var(--storm);   }}
  .stat-card.purple {{ border-left: 4px solid #7d3c98;        }}

  /* Figure */
  .figure-card {{
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
  }}
  .figure-card img {{ width: 100%; display: block; }}

  /* Footer */
  .footer {{
    text-align: center;
    padding: 12px;
    font-size: 0.72rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    border-top: 1px solid var(--border);
    margin-top: 17px;
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-title">🌬 New England NSW — Wind Dispatch Model</div>
  <div class="header-sub">
    Sapphire Wind Farm (75 × Vestas V126)  +  White Rock Wind Farm (70 × Goldwind GW121)  +  New England Solar Farm (400 MW)
  </div>
</div>

<div class="container">
  <div class="tab-bar" id="tabBar"></div>
  <div id="panels"></div>
</div>

<div class="footer">
  Generated {datetime.now().strftime('%d %B %Y  %H:%M')}
  &nbsp;|&nbsp; ERA5 Reanalysis Wind Data
  &nbsp;|&nbsp; Essential Energy Substation Demand Data
  &nbsp;|&nbsp; pvlib Solar Model
</div>

<script>
const scenarios = {scenarios_json};

function statCard(label, value, unit, cls) {{
  if (value === undefined || value === null) return '';
  return `<div class="stat-card ${{cls}}">
    <div class="stat-label">${{label}}</div>
    <div class="stat-value">${{value}}<span class="stat-unit">${{unit}}</span></div>
  </div>`;
}}

function buildPanel(sc, idx) {{
  const s = sc.stats || {{}};
  const stormBadge = sc.storm ? '<span class="storm-badge">⚡ Storm Event</span>' : '';

  // Row 1 — 6 cards, Row 2 — 5 cards, both centred
  const row1 = `
    ${{statCard('Wind Generated',  s.wind_mwh,    'MWh', 'green')}}
    ${{statCard('Solar Generated', s.solar_mwh,   'MWh', 'orange')}}
    ${{statCard('Total Demand',    s.demand_mwh,  'MWh', 'red')}}
    ${{statCard('Exported',        s.export_mwh,  'MWh', 'blue')}}
    ${{statCard('Imported',        s.import_mwh,  'MWh', 'purple')}}
    ${{statCard('Curtailed',       s.curtail_mwh, 'MWh', 'red')}}`;

  const row2 = `
    ${{statCard('Avg Sapphire Online',    s.avg_sap,     '/ 75',  'green')}}
    ${{statCard('Avg White Rock Online',  s.avg_wr,      '/ 70',  'blue')}}
    ${{statCard('Peak Wind Speed',        s.peak_wind,   'm/s',   'blue')}}
    ${{statCard('Peak Solar Output',      s.peak_solar,  'MW',    'orange')}}
    ${{statCard('Peak Demand',            s.peak_demand, 'MW',    'red')}}`;

  return `
  <div class="scenario-panel" id="panel-${{idx}}">
    <div class="info-bar">
      <div class="info-title">${{sc.icon}} ${{sc.name}} ${{stormBadge}}</div>
      <div class="info-desc">${{sc.date}} &nbsp;—&nbsp; ${{sc.desc}}</div>
    </div>
    <div class="stats-wrapper">
      <div class="stats-row">${{row1}}</div>
      <div class="stats-row">${{row2}}</div>
    </div>
    <div class="figure-card">
      <img src="${{sc.img}}" alt="${{sc.name}} charts" loading="lazy">
    </div>
  </div>`;
}}

function switchTab(idx) {{
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', i===idx));
  document.querySelectorAll('.scenario-panel').forEach((p,i) => p.classList.toggle('active', i===idx));
}}

const tabBar = document.getElementById('tabBar');
const panels = document.getElementById('panels');

scenarios.forEach((sc, idx) => {{
  const tab = document.createElement('div');
  tab.className = 'tab' + (sc.storm ? ' storm' : '') + (idx===0 ? ' active' : '');
  tab.innerHTML = sc.icon + ' ' + sc.name;
  tab.onclick = () => switchTab(idx);
  tabBar.appendChild(tab);
  panels.innerHTML += buildPanel(sc, idx);
}});

document.getElementById('panel-0').classList.add('active');
</script>
</body>
</html>"""

with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

shutil.rmtree(TEMP_DIR, ignore_errors=True)

print(" OK")
print(f"\n{'='*60}")
print(f"  Dashboard complete")
print(f"  Open: {OUTPUT_HTML}")
for sc in scenario_data:
    print(f"    {'⛈  ' if sc['storm'] else '   '}{sc['icon']} {sc['name']}  ({sc['date']})")
print("="*60 + "\n")
