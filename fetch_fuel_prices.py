"""Fetch European fuel/carbon prices from OilPrice API.

Available data:
- TTF Natural Gas (DUTCH_TTF_EUR) — EUR/MWh, European gas benchmark
- EU Carbon ETS (EU_CARBON_EUR) — EUR/tCO2, carbon cost component
- Coal API2 CIF ARA (COAL_USD) — USD/ton, European coal import benchmark
"""

import time
import requests
import pandas as pd
from datetime import datetime, timezone

from config import (
    OILPRICE_API_KEY,
    OILPRICE_API_URL,
    OILPRICE_CODE_GAS,
    OILPRICE_CODE_COAL,
    OILPRICE_CODE_CARBON,
)

# In-memory cache: {key: (timestamp, data)}
_cache = {}
_CACHE_TTL = 30 * 60  # 30 minutes


def _cache_get(key: str):
    """Return cached value if still valid, else None."""
    if key in _cache:
        ts, data = _cache[key]
        if time.time() - ts < _CACHE_TTL:
            return data
    return None


def _cache_set(key: str, data):
    """Store value in cache."""
    _cache[key] = (time.time(), data)


def _api_get(endpoint: str, params: dict | None = None) -> dict | None:
    """Make an authenticated GET request to OilPrice API with retry."""
    if not OILPRICE_API_KEY:
        return None
    headers = {"Authorization": f"Token {OILPRICE_API_KEY}"}
    url = f"{OILPRICE_API_URL}/{endpoint}"
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code != 200:
                return None
            return resp.json()
        except (requests.RequestException, ValueError):
            if attempt < 2:
                time.sleep(1)
                continue
            return None
    return None


def fetch_fuel_price(code: str) -> dict | None:
    """Fetch latest price for a commodity code.
    Returns {"price": float, "currency": str, "updated": str} or None.
    """
    cache_key = f"fuel_latest_{code}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    payload = _api_get("prices/latest", params={"by_code": code})
    if payload is None or payload.get("status") != "success":
        return None

    data = payload.get("data", {})
    result = {
        "price": data.get("price"),
        "currency": data.get("currency", ""),
        "updated": data.get("created_at", ""),
    }
    if result["price"] is None:
        return None

    _cache_set(cache_key, result)
    return result


def fetch_fuel_history(code: str, period: str = "past_month") -> pd.DataFrame:
    """Fetch historical prices. period: past_day, past_week, past_month.
    Returns DataFrame [timestamp, price].
    """
    cache_key = f"fuel_hist_{code}_{period}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    endpoint = f"prices/{period}"
    payload = _api_get(endpoint, params={"by_code": code})
    if payload is None or payload.get("status") != "success":
        return pd.DataFrame(columns=["timestamp", "price"])

    data = payload.get("data", [])
    if not isinstance(data, list):
        data = [data]

    rows = []
    for item in data:
        ts = item.get("created_at")
        price = item.get("price")
        if ts is not None and price is not None:
            rows.append({"timestamp": pd.to_datetime(ts, utc=True), "price": float(price)})

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["timestamp", "price"])
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)

    _cache_set(cache_key, df)
    return df


def fetch_all_fuel_prices() -> dict:
    """Fetch latest TTF gas, coal, and carbon prices.
    Returns {"gas": {...}, "coal": {...}, "carbon": {...}}.
    Missing entries have None values.
    """
    codes = {
        "gas": OILPRICE_CODE_GAS,
        "coal": OILPRICE_CODE_COAL,
        "carbon": OILPRICE_CODE_CARBON,
    }
    result = {}
    for name, code in codes.items():
        result[name] = fetch_fuel_price(code)
    return result


def compute_gas_srmc(gas_price: float, carbon_price: float,
                     efficiency: float = 0.45,
                     emission_factor: float = 0.37) -> float:
    """Compute short-run marginal cost of gas generation in EUR/MWh.
    SRMC = gas_price / efficiency + carbon_price * emission_factor
    """
    return gas_price / efficiency + carbon_price * emission_factor
