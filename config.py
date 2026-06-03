# =============================================================
# PIPELINE CONFIGURATION — change date here to run any day
# =============================================================

FORECAST_DAY   = 1
FORECAST_MONTH = 2
FORECAST_YEAR  = 2025

# Derived formats used by different scripts
FORECAST_DATE_STR = f'{FORECAST_YEAR}-{FORECAST_MONTH:02d}-{FORECAST_DAY:02d}'

# Transmission — constant QNI through-flow assumption
# Adjust this value if you want to test different through-flow scenarios
NE_THROUGH_FLOW_MW = 350