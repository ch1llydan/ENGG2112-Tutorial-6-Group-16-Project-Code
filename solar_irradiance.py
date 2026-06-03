"""
!!!If you want to run this you need to download pvlib as mentioned below,
and change the name of the folder the files get saved in (line 65 and 66)!!!


24-Hour Solar Irradiance Model using pvlib
==========================================
Models clear-sky solar irradiance for a given location, date, and system parameters.
Designed for use in wind dispatch optimisation as a solar generation input.

References:
    - Duffie & Beckman (2013), Solar Engineering of Thermal Processes, 4th ed., Wiley.
    - Ineichen & Perez (2002), Solar Energy, 73(3), 157-161.
    - Holmgren et al. (2018), pvlib python, J. Open Source Software, 3(29), 884.

!!!Requirements:!!!
    pip install pvlib pandas matplotlib numpy
"""

import pvlib #type:ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from config import FORECAST_DAY, FORECAST_MONTH, FORECAST_YEAR # type: ignore


# =============================================================================
# USER INPUTS — Configure all parameters here
# =============================================================================

# --- Location ---
LATITUDE        = -30.5134      # Decimal degrees (negative = Southern Hemisphere)
LONGITUDE       = 151.6672      # Decimal degrees (positive = East)
ALTITUDE        = 1000          # Metres above sea level
TIMEZONE        = 'Australia/Sydney'
LOCATION_NAME   = 'Armidale, NSW'  # Label for plots only

# --- Date ---

YEAR  = FORECAST_YEAR
MONTH = FORECAST_MONTH
DAY   = FORECAST_DAY
                                # Tip: solstices (Jun 21, Dec 21) and
                                # equinoxes (Mar 20, Sep 22) are useful reference days

# --- Atmospheric parameters ---
# Linke turbidity factor: atmospheric clarity
#   2.0  = very clear, high altitude (New England plateau typical)
#   3.0  = typical rural/regional
#   4.0+ = hazy/urban/coastal
LINKE_TURBIDITY = 2.0

# --- Solar farm parameters ---
FARM_CAPACITY_MW    = 400.0 + 115.0 + 154.0     # Nameplate capacity of the solar farm (MW) (ACEN, Metz, Gunnedah)
PERFORMANCE_RATIO   = 0.8      # System efficiency factor (0-1)
                                # Accounts for inverter losses, wiring, soiling, etc.
                                # 0.75-0.85 is typical for utility-scale farms

# --- Time resolution ---
FREQUENCY           = '30min'    # Time step: '5min', '15min', '30min', '1h'

# --- Clear sky model ---
# Options: 'ineichen' (default, recommended), 'haurwitz', 'simplified_solis'
CLEAR_SKY_MODEL     = 'ineichen'

# --- Output options ---
SAVE_CSV            = True      # Save irradiance data to CSV
SAVE_PLOT           = True      # Save plot as PNG
CSV_FILENAME  = 'solar_irradiance_output.csv'
PLOT_FILENAME = 'solar_irradiance_plot.png'
SHOW_PLOT           = False      # Display plot interactively

# =============================================================================
# MODEL — No changes needed below this line
# =============================================================================

def build_time_index(year, month, day, frequency, timezone):
    """Build a full 24-hour DatetimeIndex at the specified frequency."""
    start = pd.Timestamp(year=year, month=month, day=day, hour=0, minute=0,
                         tz=timezone)
    end   = pd.Timestamp(year=year, month=month, day=day, hour=23, minute=59,
                         tz=timezone)
    return pd.date_range(start=start, end=end, freq=frequency)


def compute_solar_position(times, latitude, longitude, altitude):
    """Compute sun position (zenith, azimuth, elevation) for each time step."""
    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        tz=times.tz,
        altitude=altitude
    )
    solar_position = location.get_solarposition(times)
    return location, solar_position


def compute_clear_sky_irradiance(location, times, linke_turbidity, model):
    """
    Compute clear-sky GHI, DNI, DHI using the specified model.

    Returns:
        DataFrame with columns: ghi, dni, dhi (all in W/m²)
            GHI = Global Horizontal Irradiance  (total on horizontal surface)
            DNI = Direct Normal Irradiance      (beam component)
            DHI = Diffuse Horizontal Irradiance (scattered sky component)
    """
    if model == 'ineichen':
        # Ineichen model requires Linke turbidity
        # pvlib can also retrieve climatological turbidity automatically,
        # but we use the user-specified value for transparency
        clearsky = location.get_clearsky(
            times,
            model='ineichen',
            linke_turbidity=linke_turbidity
        )
    else:
        clearsky = location.get_clearsky(times, model=model)

    return clearsky


def scale_to_farm_output(ghi, capacity_mw, performance_ratio):
    """
    Scale GHI irradiance to farm power output in MW.

    Method:
        Normalise GHI by the standard test condition irradiance (1000 W/m²),
        then multiply by farm capacity and performance ratio.

        P(t) = (GHI(t) / 1000) × Capacity_MW × Performance_Ratio

    This is a DC-to-AC simplified model. For higher fidelity, a full
    PVWatts or 5-parameter diode model could be substituted here.
    """
    return (ghi / 1000.0) * capacity_mw * performance_ratio


def find_sun_times(solar_position):
    """Extract sunrise and sunset times from solar position data."""
    above_horizon = solar_position['elevation'] > 0
    if above_horizon.any():
        sunrise = solar_position.index[above_horizon][0]
        sunset  = solar_position.index[above_horizon][-1]
    else:
        sunrise = sunset = None
    return sunrise, sunset


def print_summary(times, clearsky, farm_output_mw, solar_position,
                  capacity_mw, performance_ratio, linke_turbidity):
    """Print a summary of key statistics to the console."""
    sunrise, sunset = find_sun_times(solar_position)

    peak_ghi   = clearsky['ghi'].max()
    peak_power = farm_output_mw.max()
    total_energy = farm_output_mw.sum() * (
        pd.Timedelta(times.freq) / pd.Timedelta('1h')
    )

    print("\n" + "="*55)
    print("  24-HOUR SOLAR IRRADIANCE MODEL — SUMMARY")
    print("="*55)
    print(f"  Location       : {LOCATION_NAME}")
    print(f"  Coordinates    : {LATITUDE}°, {LONGITUDE}°, {ALTITUDE}m")
    print(f"  Date           : {times[0].strftime('%d %B %Y')}")
    print(f"  Clear-sky model: {CLEAR_SKY_MODEL.capitalize()}")
    print(f"  Linke turbidity: {linke_turbidity}")
    print(f"  Time resolution: {FREQUENCY}")
    print("-"*55)
    print(f"  Sunrise        : {sunrise.strftime('%H:%M %Z') if sunrise else 'N/A'}")
    print(f"  Sunset         : {sunset.strftime('%H:%M %Z') if sunset else 'N/A'}")
    if sunrise and sunset:
        daylight_hrs = (sunset - sunrise).total_seconds() / 3600
        print(f"  Daylight hours : {daylight_hrs:.1f} hrs")
    print(f"  Peak GHI       : {peak_ghi:.1f} W/m²")
    print(f"  Peak farm output: {peak_power:.1f} MW "
          f"({100*peak_power/capacity_mw:.1f}% of capacity)")
    print(f"  Daily energy   : {total_energy:.1f} MWh")
    print(f"  Capacity factor: {100*total_energy/(capacity_mw*24):.1f}%")
    print("="*55 + "\n")


def plot_irradiance(times, clearsky, farm_output_mw, solar_position,
                    capacity_mw, location_name, save_path=None):
    """
    Generate:
    1. Combined irradiance + power output figure
    2. Irradiance-only figure
    3. Power output-only figure
    """

    # =========================================================
    # PREP DATA
    # =========================================================

    clearsky = clearsky.copy()
    clearsky.index = clearsky.index.tz_localize(None)

    farm_output_mw = farm_output_mw.copy()
    farm_output_mw.index = farm_output_mw.index.tz_localize(None)

    date_str = times[0].strftime('%d %B %Y')

    # =========================================================
    # FIGURE 1 — FULL OVERVIEW
    # =========================================================

    fig = plt.figure(figsize=(14, 8))

    # White background
    fig.patch.set_facecolor('white')

    gs = GridSpec(2, 1, figure=fig, hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ---------------------------------------------------------
    # SHARED STYLING
    # ---------------------------------------------------------

    for ax in [ax1, ax2]:

        ax.set_facecolor('white')

        ax.tick_params(
            colors='black',
            labelsize=9
        )

        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')

        ax.title.set_color('black')

        for spine in ax.spines.values():
            spine.set_edgecolor('black')

        ax.grid(
            True,
            color='lightgray',
            linewidth=0.7,
            linestyle='--',
            alpha=0.8
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter('%H:%M')
        )

        ax.xaxis.set_major_locator(
            mdates.HourLocator(interval=2)
        )

    fig.suptitle(
        f'Solar Irradiance Model  ·  {location_name}  ·  {date_str}',
        fontsize=13,
        color='black',
        fontweight='bold',
        y=0.97
    )

    # =========================================================
    # PANEL 1 — IRRADIANCE COMPONENTS
    # =========================================================

    ax1.fill_between(
        clearsky.index,
        clearsky['ghi'],
        alpha=0.25,
        color='gold'
    )

    ax1.plot(
        clearsky.index,
        clearsky['ghi'],
        color='gold',
        linewidth=2.0,
        label='GHI — Global Horizontal'
    )

    ax1.plot(
        clearsky.index,
        clearsky['dni'],
        color='darkorange',
        linewidth=1.5,
        linestyle='--',
        label='DNI — Direct Normal'
    )

    ax1.plot(
        clearsky.index,
        clearsky['dhi'],
        color='skyblue',
        linewidth=1.5,
        linestyle=':',
        label='DHI — Diffuse Horizontal'
    )

    ax1.set_ylabel(
        'Irradiance (W/m²)',
        fontsize=10
    )

    ax1.set_title(
        'Clear-Sky Irradiance Components',
        fontsize=10,
        pad=8
    )

    ax1.set_ylim(bottom=0)

    ax1.legend(
        fontsize=8.5,
        framealpha=0.9,
        facecolor='white',
        edgecolor='black'
    )

    # =========================================================
    # PANEL 2 — SOLAR FARM POWER OUTPUT
    # =========================================================

    ax2.fill_between(
        farm_output_mw.index,
        farm_output_mw.values,
        alpha=0.3,
        color='forestgreen'
    )

    ax2.plot(
        farm_output_mw.index,
        farm_output_mw.values,
        color='forestgreen',
        linewidth=2.2,
        label='Estimated farm output'
    )

    ax2.axhline(
        y=capacity_mw,
        color='red',
        linewidth=1.0,
        linestyle='--',
        alpha=0.8,
        label=f'New England Capacity Total ({capacity_mw:.0f} MW)'
    )

    ax2.set_ylabel(
        'Power Output (MW)',
        fontsize=10
    )

    ax2.set_xlabel(
        'Time of Day',
        fontsize=10
    )

    ax2.set_title(
        f'Estimated Solar Farm Power Output  '
        f'(Capacity: {capacity_mw:.0f} MW, PR: {PERFORMANCE_RATIO:.0%})',
        fontsize=10,
        pad=8
    )

    ax2.set_ylim(
        bottom=0,
        top=capacity_mw * 1.1
    )

    ax2.legend(
        fontsize=8.5,
        framealpha=0.9,
        facecolor='white',
        edgecolor='black'
    )

    plt.tight_layout()

    # Save combined figure
    if save_path:

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches='tight',
            facecolor='white'
        )

        print(f"  Full overview plot saved : {save_path}")

    if SHOW_PLOT:
        plt.show()

    plt.close()

    # =========================================================
    # FIGURE 2 — IRRADIANCE ONLY
    # =========================================================

    fig, ax = plt.subplots(figsize=(14, 5))

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.grid(
        True,
        color='lightgray',
        linestyle='--',
        linewidth=0.7
    )

    ax.fill_between(
        clearsky.index,
        clearsky['ghi'],
        alpha=0.25,
        color='gold'
    )

    ax.plot(
        clearsky.index,
        clearsky['ghi'],
        color='gold',
        linewidth=2.0,
        label='GHI'
    )

    ax.plot(
        clearsky.index,
        clearsky['dni'],
        color='darkorange',
        linewidth=1.5,
        linestyle='--',
        label='DNI'
    )

    ax.plot(
        clearsky.index,
        clearsky['dhi'],
        color='skyblue',
        linewidth=1.5,
        linestyle=':',
        label='DHI'
    )

    ax.set_title(
        'Clear-Sky Irradiance Components',
        fontsize=13,
        fontweight='bold'
    )

    ax.set_ylabel('Irradiance (W/m²)')
    ax.set_xlabel('Time of Day')

    ax.legend()

    irradiance_file = None

    if save_path:

        irradiance_file = save_path.replace(
            '.png',
            '_irradiance.png'
        )

        plt.savefig(
            irradiance_file,
            dpi=150,
            bbox_inches='tight',
            facecolor='white'
        )

        print(f"  Irradiance plot saved : {irradiance_file}")

    if SHOW_PLOT:
        plt.show()

    plt.close()

    # =========================================================
    # FIGURE 3 — POWER OUTPUT ONLY
    # =========================================================

    fig, ax = plt.subplots(figsize=(14, 5))

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.grid(
        True,
        color='lightgray',
        linestyle='--',
        linewidth=0.7
    )

    ax.fill_between(
        farm_output_mw.index,
        farm_output_mw.values,
        alpha=0.3,
        color='forestgreen'
    )

    ax.plot(
        farm_output_mw.index,
        farm_output_mw.values,
        color='forestgreen',
        linewidth=2.2,
        label='Estimated farm output'
    )

    ax.axhline(
        y=capacity_mw,
        color='red',
        linewidth=1.0,
        linestyle='--',
        label=f'Capacity ({capacity_mw:.0f} MW)'
    )

    ax.set_title(
        'Estimated Solar Farm Power Output',
        fontsize=13,
        fontweight='bold'
    )

    ax.set_ylabel('Power Output (MW)')
    ax.set_xlabel('Time of Day')

    ax.set_ylim(
        bottom=0,
        top=capacity_mw * 1.1
    )

    ax.legend()

    power_file = None

    if save_path:

        power_file = save_path.replace(
            '.png',
            '_power.png'
        )

        plt.savefig(
            power_file,
            dpi=150,
            bbox_inches='tight',
            facecolor='white'
        )

        print(f"  Power output plot saved : {power_file}")

    if SHOW_PLOT:
        plt.show()

    plt.close()


def save_to_csv(times, clearsky, farm_output_mw, solar_position, filename):
    """Export key model outputs to a CSV file."""
    df = pd.DataFrame({
        'datetime'          : times,
        'ghi_wm2'           : clearsky['ghi'].values,
        'dni_wm2'           : clearsky['dni'].values,
        'dhi_wm2'           : clearsky['dhi'].values,
        'solar_elevation_deg': solar_position['elevation'].values,
        'solar_azimuth_deg' : solar_position['azimuth'].values,
        'farm_output_mw'    : farm_output_mw.values
    })
    df.to_csv(filename, index=False, float_format='%.3f')
    print(f"  Data saved to : {filename}")
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n  Running solar irradiance model for {LOCATION_NAME}...")
    print(f"  Date: {DAY:02d}/{MONTH:02d}/{YEAR}  |  Model: {CLEAR_SKY_MODEL}")

    # Build time series
    times = build_time_index(YEAR, MONTH, DAY, FREQUENCY, TIMEZONE)

    # Compute solar position and clear-sky irradiance
    location, solar_position = compute_solar_position(
        times, LATITUDE, LONGITUDE, ALTITUDE
    )
    clearsky = compute_clear_sky_irradiance(
        location, times, LINKE_TURBIDITY, CLEAR_SKY_MODEL
    )

    # Scale GHI to farm power output
    farm_output_mw = scale_to_farm_output(
        clearsky['ghi'], FARM_CAPACITY_MW, PERFORMANCE_RATIO
    )
    farm_output_mw.name = 'farm_output_mw'

    # Summary statistics
    print_summary(times, clearsky, farm_output_mw, solar_position,
                  FARM_CAPACITY_MW, PERFORMANCE_RATIO, LINKE_TURBIDITY)

    # Save outputs
    if SAVE_CSV:
        df = save_to_csv(times, clearsky, farm_output_mw,
                         solar_position, CSV_FILENAME)

    if SAVE_PLOT:
        plot_irradiance(times, clearsky, farm_output_mw, solar_position,
                        FARM_CAPACITY_MW, LOCATION_NAME, PLOT_FILENAME)
    else:
        plot_irradiance(times, clearsky, farm_output_mw, solar_position,
                        FARM_CAPACITY_MW, LOCATION_NAME, save_path=None)

    return clearsky, farm_output_mw, solar_position


if __name__ == '__main__':
    clearsky, farm_output_mw, solar_position = main()
