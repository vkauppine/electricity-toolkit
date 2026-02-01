"""Fetch Finnish day-ahead electricity spot prices from sahkotin.fi."""

import pandas as pd
import requests
from datetime import datetime, timedelta, timezone


SAHKOTIN_URL = "https://sahkotin.fi/prices"

# sahkotin.fi can handle large ranges but let's chunk by 90 days to be safe
_CHUNK_DAYS = 90


def fetch_spot_price(start: str, end: str) -> pd.DataFrame:
    """Fetch Finnish day-ahead spot prices (EUR/MWh).

    Args:
        start: ISO 8601 start time (e.g. '2024-01-01T00:00:00Z')
        end: ISO 8601 end time

    Returns:
        DataFrame with 'timestamp' and 'price' columns.
    """
    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)

    print(f"Fetching spot prices from sahkotin.fi ({start[:10]} to {end[:10]})...")

    all_rows = []
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS), end_dt)
        params = {
            "start": chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        resp = requests.get(SAHKOTIN_URL, params=params, timeout=60)
        resp.raise_for_status()
        prices = resp.json().get("prices", [])
        all_rows.extend(prices)
        chunk_start = chunk_end

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "price"])

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"date": "timestamp", "value": "price"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df[["timestamp", "price"]].dropna().sort_values("timestamp").reset_index(drop=True)
