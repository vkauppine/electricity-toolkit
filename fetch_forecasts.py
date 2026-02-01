"""Fetch forecast data from Fingrid and FMI for price prediction."""

import pandas as pd
import requests
from datetime import datetime, timedelta, timezone

from config import (
    WIND_FORECAST_DATASET,
    CONSUMPTION_FORECAST_DATASET,
    TRANSMISSION_SE1_FI,
    TRANSMISSION_SE3_FI,
    TRANSMISSION_FI_SE1,
    TRANSMISSION_FI_SE3,
    TRANSMISSION_EE_FI,
    TRANSMISSION_FI_EE,
)
from fetch_fingrid import _fetch_dataset
from fetch_fmi import _parse_wfs_simple, FMI_WFS_URL, NATIONAL_STATIONS


# --- Fingrid forecast datasets ---

def fetch_wind_forecast(start: str, end: str) -> pd.DataFrame:
    """Wind power generation forecast (MW). Dataset 245."""
    return _fetch_dataset(WIND_FORECAST_DATASET, "wind forecast", start, end)


def fetch_consumption_forecast(start: str, end: str) -> pd.DataFrame:
    """Electricity consumption forecast (MWh). Dataset 166."""
    return _fetch_dataset(CONSUMPTION_FORECAST_DATASET, "consumption forecast", start, end)


def fetch_transmission_capacity(start: str, end: str) -> pd.DataFrame:
    """Fetch all transmission capacities and merge into one DataFrame.

    Returns DataFrame with timestamp + 6 capacity columns (MW).
    """
    datasets = [
        (TRANSMISSION_SE1_FI, "import_se1"),
        (TRANSMISSION_SE3_FI, "import_se3"),
        (TRANSMISSION_EE_FI, "import_ee"),
        (TRANSMISSION_FI_SE1, "export_se1"),
        (TRANSMISSION_FI_SE3, "export_se3"),
        (TRANSMISSION_FI_EE, "export_ee"),
    ]

    frames = {}
    for ds_id, col_name in datasets:
        df = _fetch_dataset(ds_id, f"transmission {col_name}", start, end)
        if not df.empty:
            df = df.rename(columns={"value": col_name})
            df["hour"] = df["timestamp"].dt.floor("h")
            # Take hourly mean if sub-hourly
            hourly = df.groupby("hour")[col_name].mean().reset_index()
            hourly = hourly.rename(columns={"hour": "timestamp"})
            frames[col_name] = hourly

    if not frames:
        return pd.DataFrame()

    # Merge all on timestamp
    result = None
    for col_name, df in frames.items():
        if result is None:
            result = df
        else:
            result = pd.merge(result, df, on="timestamp", how="outer")

    return result.sort_values("timestamp").reset_index(drop=True)


# --- FMI temperature forecast ---

def fetch_temperature_forecast(place: str = "Helsinki") -> pd.DataFrame:
    """Fetch temperature forecast from FMI Harmonie model.

    Returns DataFrame with 'timestamp' and 'temperature' columns (~50h ahead).
    """
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "getFeature",
        "storedquery_id": "fmi::forecast::harmonie::surface::point::simple",
        "place": place,
        "parameters": "Temperature",
        "timestep": 60,
    }
    print(f"  Fetching temperature forecast for {place}...")
    resp = requests.get(FMI_WFS_URL, params=params, timeout=60)
    if resp.status_code != 200:
        print(f"  Warning: FMI forecast failed for {place} (status {resp.status_code})")
        return pd.DataFrame(columns=["timestamp", "temperature"])

    rows = _parse_wfs_simple(resp.text)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "temperature"])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df = df.rename(columns={"value": "temperature"})
    return df[["timestamp", "temperature"]].sort_values("timestamp").reset_index(drop=True)


def fetch_national_temperature_forecast() -> pd.DataFrame:
    """Fetch temperature forecast from multiple cities, return simple average.

    For forecasts we use simple average (no regression weights needed since
    the weights were derived from historical consumption which is already
    captured in the consumption forecast feature).
    """
    frames = []
    for place in NATIONAL_STATIONS:
        df = fetch_temperature_forecast(place)
        if not df.empty:
            frames.append(df)

    if not frames:
        # Fallback to single station
        print("  Warning: falling back to Helsinki-only temperature forecast")
        return fetch_temperature_forecast("Helsinki")

    combined = pd.concat(frames)
    combined["hour"] = combined["timestamp"].dt.floor("h")
    avg = combined.groupby("hour")["temperature"].mean().reset_index()
    avg = avg.rename(columns={"hour": "timestamp"})
    return avg.sort_values("timestamp").reset_index(drop=True)
