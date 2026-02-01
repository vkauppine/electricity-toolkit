import os
from dotenv import load_dotenv

load_dotenv()

FINGRID_API_KEY = os.getenv("FINGRID_API_KEY", "")
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY", "")
OILPRICE_API_KEY = os.getenv("OILPRICE_API_KEY", "")

# ENTSO-E Transparency Platform
ENTSOE_API_URL = "https://web-api.tp.entsoe.eu/api"

# OilPrice API (fuel/carbon prices)
OILPRICE_API_URL = "https://api.oilpriceapi.com/v1"
OILPRICE_CODE_GAS = "DUTCH_TTF_EUR"
OILPRICE_CODE_COAL = "COAL_USD"
OILPRICE_CODE_CARBON = "EU_CARBON_EUR"

# Country EIC codes for reservoir data
EIC_FINLAND = "10Y1001A1001A71M"
EIC_SWEDEN = "10YSE-1--------K"
EIC_NORWAY = "10YNO-0--------C"

# Fingrid dataset IDs
WIND_POWER_DATASET = 181
CONSUMPTION_DATASET = 124
NUCLEAR_DATASET = 188
HYDRO_DATASET = 191
CHP_DISTRICT_HEATING_DATASET = 201
CHP_INDUSTRIAL_DATASET = 202
ELECTRIC_BOILER_DATASET = 371

# Fingrid forecast dataset IDs
WIND_FORECAST_DATASET = 245
CONSUMPTION_FORECAST_DATASET = 166
TRANSMISSION_SE1_FI = 24
TRANSMISSION_SE3_FI = 25
TRANSMISSION_FI_SE1 = 26
TRANSMISSION_FI_SE3 = 27
TRANSMISSION_EE_FI = 112
TRANSMISSION_FI_EE = 115

# Fingrid API base URL
FINGRID_API_URL = "https://data.fingrid.fi/api/datasets/{dataset_id}/data"

# Data cache directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Default settings
DEFAULT_CITY = "Tampere"
DEFAULT_DAYS = 7
DEFAULT_SCATTER_YEARS = [2024, 2025, 2026]

# Forecast settings
FORECAST_HORIZON_HOURS = 48
TRAINING_DAYS = 90
CONFIDENCE_LEVEL = 0.95
MODEL_CACHE_DIR = os.path.join(DATA_DIR, "models")
