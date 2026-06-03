import cfgrib
import xarray as xr
import numpy as np
import pandas as pd

# =============================================================
# USER INPUTS
# =============================================================

GRIB_FILE   = 'wind_data_100m.grib'

# Single extraction point between Sapphire and White Rock
GRID_LAT    = -29.75
GRID_LON    = 151.5

# =============================================================
# STEP 1 — Load the GRIB file
# =============================================================

print("Loading GRIB file...")

all_datasets = cfgrib.open_datasets(GRIB_FILE)

print(f"  Found {len(all_datasets)} dataset(s) in file")

ds = None
for i, d in enumerate(all_datasets):
    print(f"  Dataset {i} variables: {list(d.data_vars)}")
    if 'u100' in d.data_vars and 'v100' in d.data_vars:
        ds = d
        print(f"  --> Using dataset {i}")
        break

if ds is None:
    ds_u, ds_v = None, None
    for d in all_datasets:
        if 'u100' in d.data_vars:
            ds_u = d
        if 'v100' in d.data_vars:
            ds_v = d

    if ds_u is not None and ds_v is not None:
        ds = xr.merge([ds_u, ds_v])
        print("  --> u100 and v100 found in separate datasets, merged")
    else:
        print("\n  Could not find u100/v100. All available variables:")
        for i, d in enumerate(all_datasets):
            print(f"    Dataset {i}: {list(d.data_vars)}")
        raise ValueError("u100 and/or v100 not found in GRIB file.")

print(f"\n  Loaded successfully")
print(f"  Dimensions  : {dict(ds.sizes)}")
print(f"  Time range  : {str(ds.time.values[0])[:16]} "
      f"to {str(ds.time.values[-1])[:16]}")
print(f"  Time steps  : {len(ds.time):,}")

# =============================================================
# STEP 2 — Extract single grid point
# =============================================================

print(f"\nExtracting grid point ({GRID_LAT}, {GRID_LON})...")

point = ds.sel(latitude=GRID_LAT, longitude=GRID_LON, method='nearest')

actual_lat = float(point.latitude)
actual_lon = float(point.longitude)

print(f"  Requested : ({GRID_LAT}, {GRID_LON})")
print(f"  Extracted : ({actual_lat}, {actual_lon})")

# =============================================================
# STEP 3 — Convert to DataFrame with raw wind speed only
# =============================================================

df = pd.DataFrame({
    'datetime'      : point.time.values,
    'u100'          : point.u100.values,
    'v100'          : point.v100.values,
})

# Raw wind speed magnitude from U and V components
df['wind_speed_ms'] = np.sqrt(df['u100']**2 + df['v100']**2)

print(f"\n  Rows        : {len(df):,}")
print(f"  Date range  : {df['datetime'].min()} to {df['datetime'].max()}")
print(f"  Wind speed  : mean={df['wind_speed_ms'].mean():.2f} m/s  "
      f"min={df['wind_speed_ms'].min():.2f} m/s  "
      f"max={df['wind_speed_ms'].max():.2f} m/s")

# =============================================================
# STEP 4 — Save to CSV
# =============================================================

df.to_csv('wind_raw.csv', index=False)

print("\n  Saved: wind_raw.csv")
print("  Columns: datetime, u100, v100, wind_speed_ms")