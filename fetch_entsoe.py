"""
Fetch hydro reservoir filling levels from ENTSO-E Transparency Platform.

Uses direct HTTP requests + stdlib XML parsing (no entsoe-py dependency).
Document type A72 = reservoir filling information, process type A16 = realised.
"""

import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import pandas as pd
import requests

from config import ENTSOE_API_KEY, ENTSOE_API_URL, EIC_FINLAND, EIC_SWEDEN, EIC_NORWAY

_NS = {"ns": "urn:iec62325.351:tc57wg16:451-5:gc_marketdocument:7:3"}

# In-memory cache: (country, year) -> DataFrame
_cache: dict[tuple[str, int], tuple[float, pd.DataFrame]] = {}
_CACHE_TTL = 3600  # 1 hour — weekly data rarely changes


def _parse_reservoir_xml(xml_text: str) -> pd.DataFrame:
    """Parse ENTSO-E GL_MarketDocument XML into a DataFrame of [timestamp, filling_rate]."""
    root = ET.fromstring(xml_text)

    rows = []
    for ts in root.findall(".//ns:TimeSeries", _NS):
        for period in ts.findall("ns:Period", _NS):
            start_el = period.find("ns:timeInterval/ns:start", _NS)
            resolution_el = period.find("ns:resolution", _NS)
            if start_el is None or resolution_el is None:
                continue

            period_start = datetime.fromisoformat(start_el.text.replace("Z", "+00:00"))
            resolution = resolution_el.text  # e.g. "P7D" or "P1W"

            # Parse resolution to timedelta
            if resolution in ("P7D", "P1W"):
                delta = timedelta(days=7)
            elif resolution == "P1D":
                delta = timedelta(days=1)
            else:
                # Try to parse PnD format
                try:
                    days = int(resolution.replace("P", "").replace("D", ""))
                    delta = timedelta(days=days)
                except (ValueError, AttributeError):
                    delta = timedelta(days=7)

            for point in period.findall("ns:Point", _NS):
                pos_el = point.find("ns:position", _NS)
                qty_el = point.find("ns:quantity", _NS)
                if pos_el is None or qty_el is None:
                    continue

                position = int(pos_el.text)
                quantity = float(qty_el.text)
                timestamp = period_start + delta * (position - 1)
                rows.append({"timestamp": timestamp, "filling_rate": quantity})

    if not rows:
        return pd.DataFrame(columns=["timestamp", "filling_rate"])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_reservoir_filling(country_code: str, start: str, end: str) -> pd.DataFrame:
    """Fetch weekly reservoir filling rate (%) for a country.

    Args:
        country_code: EIC area code (e.g. '10Y1001A1001A71M' for Finland)
        start: ISO datetime string for period start
        end: ISO datetime string for period end

    Returns:
        DataFrame with columns: [timestamp, filling_rate]
        filling_rate is percentage (0-100%)
    """
    if not ENTSOE_API_KEY:
        return pd.DataFrame(columns=["timestamp", "filling_rate"])

    # Check cache
    now = time.time()
    cache_key = (country_code, start, end)
    if cache_key in _cache:
        cached_time, cached_df = _cache[cache_key]
        if now - cached_time < _CACHE_TTL:
            return cached_df.copy()

    params = {
        "securityToken": ENTSOE_API_KEY,
        "documentType": "A72",
        "processType": "A16",
        "in_Domain": country_code,
        "periodStart": _to_entsoe_ts(start),
        "periodEnd": _to_entsoe_ts(end),
    }

    for attempt in range(3):
        try:
            resp = requests.get(ENTSOE_API_URL, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 401:
                return pd.DataFrame(columns=["timestamp", "filling_rate"])
            if resp.status_code != 200:
                return pd.DataFrame(columns=["timestamp", "filling_rate"])
            break
        except requests.RequestException:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return pd.DataFrame(columns=["timestamp", "filling_rate"])

    df = _parse_reservoir_xml(resp.text)

    # Cache result
    _cache[cache_key] = (now, df)

    return df


def _to_entsoe_ts(iso_str: str) -> str:
    """Convert ISO datetime string to ENTSO-E format (YYYYMMDDHHmm)."""
    # Handle various input formats
    dt = pd.to_datetime(iso_str, utc=True)
    return dt.strftime("%Y%m%d%H%M")


def fetch_nordic_reservoirs(start: str, end: str) -> pd.DataFrame:
    """Fetch FI + SE + NO reservoirs, return merged DataFrame.

    Args:
        start: ISO datetime string for period start
        end: ISO datetime string for period end

    Returns:
        DataFrame with columns: [timestamp, fi_pct, se_pct, no_pct, nordic_avg]
    """
    countries = {
        "fi_pct": EIC_FINLAND,
        "se_pct": EIC_SWEDEN,
        "no_pct": EIC_NORWAY,
    }

    dfs = {}
    for col, eic in countries.items():
        df = fetch_reservoir_filling(eic, start, end)
        if not df.empty:
            dfs[col] = df.rename(columns={"filling_rate": col})
        time.sleep(0.3)  # Be nice to the API

    if not dfs:
        return pd.DataFrame(columns=["timestamp", "fi_pct", "se_pct", "no_pct", "nordic_avg"])

    # Merge on timestamp (weekly resolution, so timestamps should align)
    merged = None
    for col, df in dfs.items():
        if merged is None:
            merged = df[["timestamp", col]]
        else:
            merged = pd.merge(merged, df[["timestamp", col]], on="timestamp", how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame(columns=["timestamp", "fi_pct", "se_pct", "no_pct", "nordic_avg"])

    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Ensure all columns exist
    for col in ["fi_pct", "se_pct", "no_pct"]:
        if col not in merged.columns:
            merged[col] = None

    # Nordic average (weighted roughly by capacity: NO ~87 TWh, SE ~34 TWh, FI ~5.5 TWh)
    weights = {"no_pct": 87, "se_pct": 34, "fi_pct": 5.5}
    total_weight = 0
    weighted_sum = pd.Series(0.0, index=merged.index)
    for col, w in weights.items():
        if col in merged.columns:
            valid = merged[col].notna()
            weighted_sum = weighted_sum + merged[col].fillna(0) * w * valid.astype(float)
            total_weight += w * valid.astype(float)

    merged["nordic_avg"] = weighted_sum / total_weight.replace(0, float("nan"))

    return merged[["timestamp", "fi_pct", "se_pct", "no_pct", "nordic_avg"]]
