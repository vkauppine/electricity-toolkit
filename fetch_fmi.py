"""Fetch weather data from FMI Open Data."""

import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from sklearn.linear_model import LinearRegression


# FMI WFS base URL
FMI_WFS_URL = "https://opendata.fmi.fi/wfs"

# Representative stations across Finland for weighted national temperature.
# Chosen to cover population centers and geographic spread.
NATIONAL_STATIONS = [
    "Helsinki",
    "Tampere",
    "Oulu",
    "Turku",
    "Jyväskylä",
    "Kuopio",
    "Rovaniemi",
    "Vaasa",
    "Joensuu",
    "Sodankylä",
]


def _parse_wfs_simple(xml_text: str) -> list[dict]:
    """Parse FMI WFS simple feature XML response into a list of dicts."""
    root = ET.fromstring(xml_text)
    ns = {
        "wfs": "http://www.opengis.net/wfs/2.0",
        "BsWfs": "http://xml.fmi.fi/schema/wfs/2.0",
        "gml": "http://www.opengis.net/gml/3.2",
    }
    rows = []
    for member in root.findall(".//BsWfs:BsWfsElement", ns):
        time_el = member.find("BsWfs:Time", ns)
        param_el = member.find("BsWfs:ParameterName", ns)
        value_el = member.find("BsWfs:ParameterValue", ns)
        if time_el is not None and param_el is not None and value_el is not None:
            try:
                val = float(value_el.text)
            except (ValueError, TypeError):
                continue
            rows.append({
                "time": time_el.text,
                "parameter": param_el.text,
                "value": val,
            })
    return rows


def fetch_hourly_temperature(place: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch hourly temperature for a place from FMI.

    Returns DataFrame with columns: timestamp, temperature
    """
    # FMI limits query length; fetch in chunks of 7 days
    all_rows = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=7), end)
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "getFeature",
            "storedquery_id": "fmi::observations::weather::hourly::simple",
            "place": place,
            "starttime": chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endtime": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "parameters": "TA_PT1H_AVG",
        }
        resp = requests.get(FMI_WFS_URL, params=params, timeout=60)
        if resp.status_code == 200:
            rows = _parse_wfs_simple(resp.text)
            all_rows.extend(rows)
        chunk_start = chunk_end

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "temperature"])

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df = df.rename(columns={"value": "temperature"})
    return df[["timestamp", "temperature"]].sort_values("timestamp").reset_index(drop=True)


def fetch_city_temperatures(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch hourly temperature from multiple stations, returning one column per city.

    Returns DataFrame indexed by hourly timestamp, with one column per city.
    """
    city_series = {}
    for place in NATIONAL_STATIONS:
        print(f"  Fetching temperature for {place}...")
        df = fetch_hourly_temperature(place, start, end)
        if not df.empty:
            df["hour"] = df["timestamp"].dt.floor("h")
            series = df.groupby("hour")["temperature"].mean()
            city_series[place] = series

    if not city_series:
        raise RuntimeError("Could not fetch temperature data from any FMI station")

    combined = pd.DataFrame(city_series)
    combined.index.name = "timestamp"
    return combined


def compute_regression_weights(city_temps: pd.DataFrame, consumption: pd.DataFrame) -> dict[str, float]:
    """Use linear regression to find how each city's temperature drives consumption.

    Regresses consumption ~ T_city1 + T_city2 + ... and returns normalized
    absolute coefficients as weights. Cities whose temperature has a larger
    effect on national consumption get higher weight.

    Args:
        city_temps: DataFrame indexed by timestamp, one column per city
        consumption: DataFrame with 'timestamp' and 'value' columns

    Returns:
        Dict mapping city name to its weight (sums to 1.0)
    """
    cons = consumption.copy()
    cons["hour"] = cons["timestamp"].dt.floor("h")
    cons = cons.groupby("hour")["value"].mean()

    # Align on common timestamps
    common_idx = city_temps.index.intersection(cons.index)
    if len(common_idx) < 100:
        print(f"  Warning: only {len(common_idx)} overlapping hours for regression, "
              f"falling back to equal weights")
        cities = list(city_temps.columns)
        return {c: 1.0 / len(cities) for c in cities}

    X = city_temps.loc[common_idx].dropna()
    common_idx = X.index
    y = cons.loc[common_idx]

    # Drop any remaining NaN rows
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    if len(X) < 100:
        cities = list(city_temps.columns)
        print(f"  Warning: insufficient clean data for regression, using equal weights")
        return {c: 1.0 / len(cities) for c in cities}

    model = LinearRegression()
    model.fit(X.values, y.values)

    # Use absolute coefficients — consumption typically rises as temp drops,
    # so coefficients are negative, but the magnitude tells us influence.
    abs_coefs = np.abs(model.coef_)
    total = abs_coefs.sum()
    if total == 0:
        cities = list(X.columns)
        return {c: 1.0 / len(cities) for c in cities}

    weights = abs_coefs / total
    result = {city: float(w) for city, w in zip(X.columns, weights)}

    print("  Regression weights (city temperature influence on consumption):")
    for city, w in sorted(result.items(), key=lambda x: -x[1]):
        print(f"    {city:15s}: {w:.3f}")
    print(f"  R² = {model.score(X.values, y.values):.3f}")

    return result


def compute_weighted_temperature(city_temps: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Compute weighted national temperature from per-city temps and weights.

    Returns DataFrame with columns: timestamp, temperature
    """
    available = [c for c in weights if c in city_temps.columns]
    w = np.array([weights[c] for c in available])
    # Re-normalize in case some cities are missing
    w = w / w.sum()

    weighted = (city_temps[available].values * w).sum(axis=1)
    result = pd.DataFrame({
        "timestamp": city_temps.index,
        "temperature": weighted,
    })
    return result.dropna().reset_index(drop=True)


def fetch_national_temperature_weighted(
    start: datetime, end: datetime, consumption_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fetch city temperatures and compute regression-weighted national temperature.

    Args:
        start: period start
        end: period end
        consumption_df: electricity consumption DataFrame for regression

    Returns:
        (temperature DataFrame with 'timestamp' and 'temperature', weight dict)
    """
    city_temps = fetch_city_temperatures(start, end)
    print(f"  Got temperature data for {len(city_temps.columns)} cities, "
          f"{len(city_temps)} hours")

    print("\n=== Computing regression weights ===")
    weights = compute_regression_weights(city_temps, consumption_df)

    temp_df = compute_weighted_temperature(city_temps, weights)
    return temp_df, weights


def fetch_daily_weather(place: str, days: int = 7) -> pd.DataFrame:
    """Fetch daily weather observations (min/max/avg temp, precipitation) for a city.

    Returns DataFrame with columns: date, t_min, t_max, t_avg, precipitation
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "getFeature",
        "storedquery_id": "fmi::observations::weather::daily::simple",
        "place": place,
        "starttime": start.strftime("%Y-%m-%dT00:00:00Z"),
        "endtime": end.strftime("%Y-%m-%dT23:59:59Z"),
        "parameters": "tmin,tmax,tday,rrday",
    }
    print(f"Fetching daily weather for {place} (last {days} days)...")
    resp = requests.get(FMI_WFS_URL, params=params, timeout=60)
    resp.raise_for_status()
    rows = _parse_wfs_simple(resp.text)

    if not rows:
        raise RuntimeError(f"No daily weather data returned for {place}")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["timestamp"].dt.date

    # Pivot parameters into columns
    pivot = df.pivot_table(index="date", columns="parameter", values="value", aggfunc="first")
    pivot = pivot.reset_index()

    rename_map = {}
    for col in pivot.columns:
        cl = col.lower() if isinstance(col, str) else ""
        if cl == "tmin":
            rename_map[col] = "t_min"
        elif cl == "tmax":
            rename_map[col] = "t_max"
        elif cl == "tday":
            rename_map[col] = "t_avg"
        elif cl == "rrday":
            rename_map[col] = "precipitation"

    pivot = pivot.rename(columns=rename_map)

    expected = ["date", "t_min", "t_max", "t_avg", "precipitation"]
    for c in expected:
        if c not in pivot.columns:
            pivot[c] = float("nan")

    return pivot[expected].sort_values("date").reset_index(drop=True)
