"""
Master Pipeline - New England NSW Wind Dispatch Model
======================================================
Orchestrates the full pipeline from a single script.
Change the forecast date in config.py, then run this file.

Pipeline order:
    1. wind_predict.py      - hourly wind speed forecast
    2. solar_irradiance.py  - deterministic solar generation
    3. demand_predict.py    - half-hourly demand forecast
    4. power_generation.py  - turbine dispatch optimisation

One-time setup scripts (run manually, not part of daily pipeline):
    wind_extract.py         - extract ERA5 GRIB to CSV (run once)
    wind_ml.py              - train XGBoost wind model (run once)
    demand_train.py         - train Random Forest demand model (run once)

Requirements:
    All scripts and config.py must be in the same folder as this file.
"""

import subprocess
import sys
import os
import time
from datetime import datetime

# =============================================================
# PIPELINE CONTROL - set which steps to run
# =============================================================

# Core pipeline - runs every time you change the forecast date
RUN_WIND_PREDICT    = True
RUN_SOLAR           = True
RUN_DEMAND_PREDICT  = True
RUN_POWER_GEN       = True

# One-time setup - set True only when retraining is needed
RUN_WIND_EXTRACT    = False   # re-extracts ERA5 GRIB file
RUN_WIND_TRAIN      = False   # retrains XGBoost wind model
RUN_DEMAND_TRAIN    = False   # retrains Random Forest demand model

# =============================================================
# HELPER FUNCTIONS
# =============================================================

def run_script(script_name, label=None):
    """
    Run a Python script as a subprocess and track its result.
    Returns (success, elapsed_seconds).
    """
    label = label or script_name
    print(f"\n{'='*60}")
    print(f"  Running : {label}")
    print(f"  Script  : {script_name}")
    print(f"  Time    : {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    t_start = time.time()
    result  = subprocess.run(
        [sys.executable, script_name],
        capture_output=False   # shows live output in terminal
    )
    elapsed = time.time() - t_start
    success = result.returncode == 0

    if success:
        print(f"\n  Completed in {elapsed:.1f}s")
    else:
        print(f"\n  FAILED after {elapsed:.1f}s (exit code {result.returncode})")

    return success, elapsed


def check_file(filepath, created_by):
    """Warn if a required input file is missing."""
    if not os.path.exists(filepath):
        print(f"  MISSING : {filepath}")
        print(f"            Run {created_by} first to create this file.")
        return False
    return True


# =============================================================
# PRE-FLIGHT CHECKS
# =============================================================

print("\n" + "="*60)
print("  NEW ENGLAND NSW - WIND DISPATCH PIPELINE")
print("="*60)

try:
    from config import (FORECAST_DAY, FORECAST_MONTH, FORECAST_YEAR, #type: ignore
                        FORECAST_DATE_STR, NE_THROUGH_FLOW_MW) 
except ImportError:
    print("\n  ERROR: config.py not found.")
    print("  Create config.py in the same folder as this script.")
    sys.exit(1)

print(f"\n  Forecast date  : {FORECAST_DAY:02d}/{FORECAST_MONTH:02d}/{FORECAST_YEAR}")
print(f"  Through-flow   : {NE_THROUGH_FLOW_MW} MW (constant QNI assumption)")
print(f"  Started        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\n  Pre-flight checks...")
all_ok = True

if RUN_WIND_PREDICT and not RUN_WIND_TRAIN:
    all_ok &= check_file('wind_model.json',     'wind_ml.py')
    all_ok &= check_file('wind_raw.csv',        'wind_extract.py')

if RUN_DEMAND_PREDICT and not RUN_DEMAND_TRAIN:
    all_ok &= check_file('demand_model.pkl',    'demand_train.py')
    all_ok &= check_file('demand_processed.csv','demand_train.py')

if not all_ok:
    print("\n  Fix missing files above then re-run. Aborting.")
    sys.exit(1)

print("  All required files present.")

# =============================================================
# BUILD PIPELINE
# =============================================================

pipeline = []

# One-time setup steps (only when flagged)
if RUN_WIND_EXTRACT:
    pipeline.append(('wind_extract.py',   'Extract ERA5 wind data'))
if RUN_WIND_TRAIN:
    pipeline.append(('wind_ml.py',        'Train XGBoost wind model'))
if RUN_DEMAND_TRAIN:
    pipeline.append(('demand_train.py',   'Train demand model'))

# Core pipeline steps
if RUN_WIND_PREDICT:
    pipeline.append(('wind_predict.py',      'Wind speed forecast'))
if RUN_SOLAR:
    pipeline.append(('solar_irradiance.py',  'Solar generation'))
if RUN_DEMAND_PREDICT:
    pipeline.append(('demand_predict.py',    'Demand forecast'))
if RUN_POWER_GEN:
    pipeline.append(('power_generation.py',  'Turbine dispatch'))

one_time = {'wind_extract.py', 'wind_ml.py', 'demand_train.py'}

print(f"\n  Steps to run : {len(pipeline)}")
for i, (script, label) in enumerate(pipeline):
    tag = '(one-time setup)' if script in one_time else ''
    print(f"    {i+1}. {label:<38} {tag}")

# =============================================================
# RUN PIPELINE
# =============================================================

results    = {}
start_time = time.time()
failed_at  = None

for script, label in pipeline:
    if not os.path.exists(script):
        print(f"\n  ERROR: {script} not found in current directory.")
        results[label] = ('MISSING', 0)
        failed_at = label
        break

    success, elapsed = run_script(script, label)
    results[label]   = ('OK' if success else 'FAILED', elapsed)

    if not success:
        failed_at = label
        print(f"\n  Pipeline stopped at : {label}")
        print(f"  Scroll up to see the error from {script}.")
        break

# =============================================================
# FINAL SUMMARY
# =============================================================

total_elapsed = time.time() - start_time

print(f"\n\n{'='*60}")
print(f"  PIPELINE SUMMARY")
print(f"{'='*60}")
print(f"  Date      : {FORECAST_DAY:02d}/{FORECAST_MONTH:02d}/{FORECAST_YEAR}")
print(f"  Finished  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Total time: {total_elapsed:.1f}s")
print(f"  {'-'*56}")

for label, (status, elapsed) in results.items():
    icon = 'OK  ' if status == 'OK' else ('FAIL' if status == 'FAILED' else 'MISS')
    t    = f'{elapsed:.1f}s' if elapsed > 0 else ''
    print(f"  [{icon}]  {label:<38} {t}")

# Show steps that were not reached due to earlier failure
ran = set(results.keys())
for _, label in pipeline:
    if label not in ran:
        print(f"  [----]  {label:<38} (not reached)")

print(f"  {'-'*56}")

if failed_at is None:
    print(f"\n  All steps completed successfully.")
    print(f"\n  Output files:")

    output_files = [
        ('wind_forecast.csv',           'Hourly wind speed predictions'),
        ('wind_forecast_plot.png',      'Wind forecast plot'),
        ('solar_irradiance_output.csv', 'Solar generation profile'),
        ('solar_irradiance_plot.png',   'Solar plot'),
        ('demand_profile.csv',          'Half-hourly demand forecast'),
        ('demand_forecast_plot.png',    'Demand forecast plot'),
        ('dispatch_log.csv',            'Full 30-min dispatch record'),
        ('dispatch_summary.csv',        'Daily summary'),
        ('dispatch_plot.png',           'Dispatch visualisation'),
    ]

    for fname, desc in output_files:
        exists = os.path.exists(fname)
        tag    = 'OK     ' if exists else 'MISSING'
        print(f"    [{tag}]  {fname:<38} {desc}")
else:
    print(f"\n  Pipeline FAILED at step: {failed_at}")
    print(f"  Fix the error above and re-run.")

print("="*60 + "\n")
