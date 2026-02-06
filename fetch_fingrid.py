"""Fetch energy production/consumption data from Fingrid."""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

from config import (
    FINGRID_API_KEY,
    FINGRID_API_URL,
    WIND_POWER_DATASET,
    CONSUMPTION_DATASET,
    NUCLEAR_DATASET,
    HYDRO_DATASET,
    CHP_DISTRICT_HEATING_DATASET,
    CHP_INDUSTRIAL_DATASET,
    ELECTRIC_BOILER_DATASET,
    SOLAR_POWER_DATASET,
    TOTAL_PRODUCTION_DATASET,
    NET_IMPORT_EXPORT_DATASET,
    DATA_DIR,
)


def _fetch_from_api(dataset_id: int, start: str, end: str) -> pd.DataFrame:
    """Fetch data from Fingrid REST API with pagination."""
    if not FINGRID_API_KEY:
        raise ValueError("No Fingrid API key configured")

    headers = {"x-api-key": FINGRID_API_KEY}
    all_rows = []
    page = 1
    page_size = 20000

    while True:
        params = {
            "startTime": start,
            "endTime": end,
            "format": "json",
            "page": page,
            "pageSize": page_size,
            "locale": "en",
            "sortBy": "startTime",
            "sortOrder": "asc",
        }
        url = FINGRID_API_URL.format(dataset_id=dataset_id)
        for attempt in range(5):
            resp = requests.get(url, headers=headers, params=params, timeout=60)
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", [])
        if not data:
            break
        all_rows.extend(data)
        pagination = payload.get("pagination", {})
        total = pagination.get("total", 0)
        if len(all_rows) >= total:
            break
        page += 1

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "value"])

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"startTime": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["timestamp", "value"]].dropna()


def _load_from_csv(dataset_id: int) -> pd.DataFrame:
    """Load data from a CSV file in the data/ directory."""
    pattern = f"dataset_{dataset_id}"
    for fname in os.listdir(DATA_DIR):
        if pattern in fname.lower() and fname.endswith(".csv"):
            path = os.path.join(DATA_DIR, fname)
            df = pd.read_csv(path)
            ts_col = None
            val_col = None
            for col in df.columns:
                cl = col.lower().strip()
                if cl in ("start time", "starttime", "time", "timestamp", "alkuaika"):
                    ts_col = col
                if cl in ("value", "arvo", "tuotanto", "kulutus"):
                    val_col = col

            if ts_col is None:
                ts_col = df.columns[0]
            if val_col is None:
                val_col = df.columns[-1]

            df = df.rename(columns={ts_col: "timestamp", val_col: "value"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df[["timestamp", "value"]].dropna()

    raise FileNotFoundError(
        f"No CSV file matching '{pattern}*.csv' found in {DATA_DIR}. "
        f"Download from https://data.fingrid.fi/en/datasets/{dataset_id} "
        f"and save to the data/ folder."
    )


def _fetch_dataset(dataset_id: int, label: str, start: str, end: str) -> pd.DataFrame:
    """Generic fetch: API if key available, else CSV fallback."""
    if FINGRID_API_KEY:
        print(f"Fetching {label} from Fingrid API ({start[:10]} to {end[:10]})...")
        return _fetch_from_api(dataset_id, start, end)
    else:
        print(f"No API key — loading {label} from CSV...")
        return _load_from_csv(dataset_id)


def fetch_wind_power(start: str, end: str) -> pd.DataFrame:
    """Wind power production (MW). Dataset 181."""
    return _fetch_dataset(WIND_POWER_DATASET, "wind power", start, end)


def fetch_consumption(start: str, end: str) -> pd.DataFrame:
    """Electricity consumption (MWh). Dataset 124."""
    return _fetch_dataset(CONSUMPTION_DATASET, "consumption", start, end)


def fetch_nuclear(start: str, end: str) -> pd.DataFrame:
    """Nuclear power production (MW). Dataset 188."""
    return _fetch_dataset(NUCLEAR_DATASET, "nuclear power", start, end)


def fetch_hydro(start: str, end: str) -> pd.DataFrame:
    """Hydro power production (MW). Dataset 191."""
    return _fetch_dataset(HYDRO_DATASET, "hydro power", start, end)


def fetch_chp_district_heating(start: str, end: str) -> pd.DataFrame:
    """District heating CHP production (MW). Dataset 201."""
    return _fetch_dataset(CHP_DISTRICT_HEATING_DATASET, "CHP district heating", start, end)


def fetch_chp_industrial(start: str, end: str) -> pd.DataFrame:
    """Industrial CHP production (MW). Dataset 202."""
    return _fetch_dataset(CHP_INDUSTRIAL_DATASET, "CHP industrial", start, end)


def fetch_electric_boiler(start: str, end: str) -> pd.DataFrame:
    """Electric boiler consumption (MW). Dataset 371."""
    return _fetch_dataset(ELECTRIC_BOILER_DATASET, "electric boiler consumption", start, end)


def fetch_solar_power(start: str, end: str) -> pd.DataFrame:
    """Solar power production forecast (MW). Dataset 248.

    Note: This is a forecast dataset updated every 15 minutes for the next 72 hours.
    Fingrid does not publish actual solar production measurements separately.
    """
    return _fetch_dataset(SOLAR_POWER_DATASET, "solar power", start, end)


def fetch_total_production(start: str, end: str) -> pd.DataFrame:
    """Total electricity production in Finland (MW). Dataset 192.

    Real-time measurements from Fingrid's operation control system, updated every 3 minutes.
    """
    return _fetch_dataset(TOTAL_PRODUCTION_DATASET, "total production", start, end)


def fetch_net_import_export(start: str, end: str) -> pd.DataFrame:
    """Net import/export of electricity (MW). Dataset 194.

    Positive values = net import, negative values = net export.
    Real-time measurements updated every 3 minutes.
    """
    return _fetch_dataset(NET_IMPORT_EXPORT_DATASET, "net import/export", start, end)
