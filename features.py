"""Feature engineering for electricity price forecasting."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from fetch_prices import fetch_spot_price
from fetch_fingrid import (
    fetch_wind_power, fetch_consumption, fetch_nuclear,
    fetch_hydro, fetch_chp_district_heating, fetch_chp_industrial,
    fetch_electric_boiler, fetch_solar_power, fetch_total_production,
    fetch_net_import_export,
)
from fetch_forecasts import (
    fetch_wind_forecast, fetch_consumption_forecast,
    fetch_transmission_capacity, fetch_national_temperature_forecast,
)
from fetch_fmi import fetch_city_temperatures, compute_weighted_temperature
from fetch_fuel_prices import fetch_fuel_history
from fetch_entsoe import fetch_nordic_reservoirs
from config import (
    OILPRICE_CODE_GAS, OILPRICE_CODE_COAL, OILPRICE_CODE_CARBON,
    RESOLUTION_MINUTES, PERIODS_PER_HOUR,
)

# Finnish public holidays (month, day) — fixed-date holidays
_FINNISH_HOLIDAYS = [
    (1, 1), (1, 6), (5, 1), (12, 6), (12, 24), (12, 25), (12, 26),
]


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features from timestamp column."""
    ts = df["timestamp"]
    df = df.copy()
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Finnish public holidays (simplified — fixed dates only)
    md = list(zip(ts.dt.month, ts.dt.day))
    df["is_holiday"] = pd.Series(
        [1 if (m, d) in _FINNISH_HOLIDAYS else 0 for m, d in md],
        index=df.index,
    )

    # For sub-hourly data, add quarter-hour feature (0-3 for 15-min intervals)
    if RESOLUTION_MINUTES < 60:
        df["quarter_hour"] = ts.dt.minute // RESOLUTION_MINUTES
        # Cyclic encoding for time-of-day at sub-hourly resolution
        periods_per_day = 24 * PERIODS_PER_HOUR
        time_of_day = df["hour"] * PERIODS_PER_HOUR + df["quarter_hour"]
        df["time_of_day_sin"] = np.sin(2 * np.pi * time_of_day / periods_per_day)
        df["time_of_day_cos"] = np.cos(2 * np.pi * time_of_day / periods_per_day)

    # Cyclic encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def create_lag_features(df: pd.DataFrame, col: str, lags: list[int]) -> pd.DataFrame:
    """Create lagged versions of a column.

    Args:
        df: DataFrame with the column to lag
        col: Column name to create lags for
        lags: List of lag periods (in hours for backward compatibility)

    Note: Lags are converted from hours to periods based on RESOLUTION_MINUTES.
    E.g., lag=1 hour = 4 periods for 15-min resolution.
    """
    df = df.copy()
    for lag_hours in lags:
        lag_periods = lag_hours * PERIODS_PER_HOUR
        df[f"{col}_lag_{lag_hours}h"] = df[col].shift(lag_periods)
    return df


def create_rolling_features(df: pd.DataFrame, col: str, windows: list[int]) -> pd.DataFrame:
    """Create rolling mean and std features.

    Args:
        df: DataFrame with the column
        col: Column name to create rolling features for
        windows: List of window sizes (in hours for backward compatibility)

    Note: Windows are converted from hours to periods based on RESOLUTION_MINUTES.
    E.g., window=24 hours = 96 periods for 15-min resolution.
    """
    df = df.copy()
    for w_hours in windows:
        w_periods = w_hours * PERIODS_PER_HOUR
        df[f"{col}_rmean_{w_hours}h"] = df[col].rolling(w_periods, min_periods=1).mean()
        df[f"{col}_rstd_{w_hours}h"] = df[col].rolling(w_periods, min_periods=1).std().fillna(0)
    return df


def _resample_to_target_resolution(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Resample data to the target resolution specified in config.

    For RESOLUTION_MINUTES=60, resamples to hourly.
    For RESOLUTION_MINUTES=15, resamples to 15-minute intervals.
    """
    df = df.copy()
    freq_str = f"{RESOLUTION_MINUTES}min" if RESOLUTION_MINUTES < 60 else "h"
    df["period"] = df["timestamp"].dt.floor(freq_str)
    resampled = df.groupby("period")[value_col].mean().reset_index()
    resampled = resampled.rename(columns={"period": "timestamp"})
    return resampled


def fetch_historical_features(start: str, end: str) -> pd.DataFrame:
    """Fetch all historical data and build feature matrix.

    Returns DataFrame with price + all features at the configured resolution.
    Resolution is determined by RESOLUTION_MINUTES in config.
    """
    import time

    # 1. Spot prices (target)
    prices = fetch_spot_price(start, end)
    prices = _resample_to_target_resolution(prices, value_col="price")
    prices = prices.rename(columns={"value": "price"})

    # If price column already exists from fetch, keep it
    if "price" not in prices.columns and "value" in prices.columns:
        prices = prices.rename(columns={"value": "price"})

    # 2. Fingrid production/consumption data
    datasets = {
        "wind": fetch_wind_power,
        "consumption": fetch_consumption,
        "nuclear": fetch_nuclear,
        "hydro": fetch_hydro,
        "chp_dh": fetch_chp_district_heating,
        "chp_ind": fetch_chp_industrial,
        "electric_boiler": fetch_electric_boiler,
        "solar": fetch_solar_power,
        "total_production": fetch_total_production,
        "net_import_export": fetch_net_import_export,
    }

    energy = prices[["timestamp"]].copy()
    for name, fn in datasets.items():
        time.sleep(0.5)  # Rate limit courtesy
        try:
            df = fn(start, end)
            df = _resample_to_target_resolution(df)
            df = df.rename(columns={"value": name})
            energy = pd.merge(energy, df[["timestamp", name]], on="timestamp", how="left")
        except Exception as e:
            print(f"  Warning: could not fetch {name}: {e}")
            energy[name] = np.nan

    # 3. Temperature (weighted national average)
    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)
    try:
        print("\n=== Fetching city temperatures for features ===")
        city_temps = fetch_city_temperatures(start_dt, end_dt)
        # Use consumption for regression weights
        cons_for_weights = energy[["timestamp", "consumption"]].dropna()
        cons_for_weights = cons_for_weights.rename(columns={"consumption": "value"})
        from fetch_fmi import compute_regression_weights
        weights = compute_regression_weights(city_temps, cons_for_weights)
        temp_df = compute_weighted_temperature(city_temps, weights)
    except Exception as e:
        print(f"  Warning: temperature fetch failed: {e}")
        temp_df = pd.DataFrame({"timestamp": energy["timestamp"], "temperature": np.nan})

    energy = pd.merge(energy, temp_df, on="timestamp", how="left")

    # 4. Transmission capacity
    try:
        cap_df = fetch_transmission_capacity(start, end)
        if not cap_df.empty:
            energy = pd.merge(energy, cap_df, on="timestamp", how="left")
    except Exception as e:
        print(f"  Warning: transmission capacity fetch failed: {e}")

    # Ensure transmission columns exist
    for col in ["import_se1", "import_se3", "import_ee", "export_se1", "export_se3", "export_ee"]:
        if col not in energy.columns:
            energy[col] = np.nan

    # 5. Fuel prices (TTF gas, carbon ETS, coal API2)
    fuel_codes = {
        "gas_ttf": OILPRICE_CODE_GAS,
        "carbon_ets": OILPRICE_CODE_CARBON,
        "coal_api2": OILPRICE_CODE_COAL,
    }
    for col_name, code in fuel_codes.items():
        try:
            fuel_df = fetch_fuel_history(code, period="past_month")
            if not fuel_df.empty:
                fuel_df = fuel_df.rename(columns={"price": col_name})
                fuel_df = fuel_df.sort_values("timestamp")
                energy = pd.merge_asof(
                    energy.sort_values("timestamp"),
                    fuel_df[["timestamp", col_name]],
                    on="timestamp",
                    direction="backward",
                )
            else:
                energy[col_name] = np.nan
        except Exception as e:
            print(f"  Warning: could not fetch {col_name}: {e}")
            energy[col_name] = np.nan

    # 6. Hydro reservoir levels (Nordic + Finland)
    try:
        reservoir_df = fetch_nordic_reservoirs(start, end)
        if not reservoir_df.empty:
            res_renamed = reservoir_df.rename(columns={
                "nordic_avg": "reservoir_nordic",
                "fi_pct": "reservoir_fi",
            })[["timestamp", "reservoir_nordic", "reservoir_fi"]]
            res_renamed = res_renamed.sort_values("timestamp")
            energy = pd.merge_asof(
                energy.sort_values("timestamp"),
                res_renamed,
                on="timestamp",
                direction="backward",
            )
        else:
            energy["reservoir_nordic"] = np.nan
            energy["reservoir_fi"] = np.nan
    except Exception as e:
        print(f"  Warning: reservoir data fetch failed: {e}")
        energy["reservoir_nordic"] = np.nan
        energy["reservoir_fi"] = np.nan

    # 7. Merge price target
    result = pd.merge(energy, prices, on="timestamp", how="inner")

    return result.sort_values("timestamp").reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """From raw historical data with price + energy columns, build full feature matrix.

    Input must have: timestamp, price, wind, consumption, nuclear, hydro,
    chp_dh, chp_ind, electric_boiler, temperature, transmission columns.
    """
    df = df.copy()

    # Temporal features
    df = extract_temporal_features(df)

    # Price lag and rolling features
    df = create_lag_features(df, "price", [1, 24, 168])
    df = create_rolling_features(df, "price", [24, 168])

    # Derived energy features
    df["thermal"] = df[["chp_dh", "chp_ind", "electric_boiler"]].sum(axis=1, skipna=True)

    # Calculate total production from components
    # Note: Use total_production if available, otherwise sum components
    if "total_production" in df.columns and df["total_production"].notna().any():
        total_prod = df["total_production"]
    else:
        total_prod = df[["wind", "nuclear", "hydro", "thermal", "solar"]].sum(axis=1, skipna=True)

    df["supply_demand_ratio"] = total_prod / df["consumption"].replace(0, np.nan)
    df["wind_penetration"] = df["wind"] / total_prod.replace(0, np.nan)
    df["solar_penetration"] = df.get("solar", pd.Series(0, index=df.index)) / total_prod.replace(0, np.nan)
    df["renewable_share"] = (df["wind"] + df["hydro"] + df.get("solar", 0)) / total_prod.replace(0, np.nan)

    # Transmission derived
    df["net_import_capacity"] = (
        df[["import_se1", "import_se3", "import_ee"]].sum(axis=1, skipna=True)
        - df[["export_se1", "export_se3", "export_ee"]].sum(axis=1, skipna=True)
    )

    # Fuel derived
    df["gas_srmc"] = df.get("gas_ttf", pd.Series(np.nan, index=df.index)) / 0.45 + \
                     df.get("carbon_ets", pd.Series(np.nan, index=df.index)) * 0.37

    # Temperature derived (convert hour-based lags to periods)
    periods_24h = 24 * PERIODS_PER_HOUR
    df["temperature_lag_24h"] = df["temperature"].shift(periods_24h)
    df["temp_change_24h"] = df["temperature"] - df["temperature_lag_24h"]
    df["heating_degree_hours"] = (18 - df["temperature"]).clip(lower=0)

    # Price momentum
    std_24 = df.get("price_rstd_24h", pd.Series(1, index=df.index))
    df["price_momentum"] = (df["price"] - df.get("price_rmean_24h", df["price"])) / std_24.replace(0, 1)

    return df


def get_feature_columns() -> list[str]:
    """Return the list of feature column names used for model training."""
    base_features = [
        # Temporal
        "hour", "day_of_week", "month", "is_weekend", "is_holiday",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    ]

    # Add sub-hourly features if resolution is < 60 minutes
    if RESOLUTION_MINUTES < 60:
        base_features.extend([
            "quarter_hour", "time_of_day_sin", "time_of_day_cos",
        ])

    base_features.extend([
        # Price lags
        "price_lag_1h", "price_lag_24h", "price_lag_168h",
        "price_rmean_24h", "price_rstd_24h", "price_rmean_168h", "price_rstd_168h",
        # Energy production
        "wind", "consumption", "nuclear", "hydro", "thermal", "solar",
        "total_production", "net_import_export",
        # Weather
        "temperature", "temperature_lag_24h", "heating_degree_hours",
        # Transmission
        "import_se1", "import_se3", "import_ee",
        "export_se1", "export_se3", "export_ee", "net_import_capacity",
        # Derived energy features
        "supply_demand_ratio", "wind_penetration", "solar_penetration",
        "renewable_share", "temp_change_24h", "price_momentum",
        # Fuel / carbon
        "gas_ttf", "carbon_ets", "coal_api2", "gas_srmc",
        # Reservoirs
        "reservoir_nordic", "reservoir_fi",
    ])

    return base_features


def build_forecast_features(
    historical_df: pd.DataFrame,
    wind_forecast: pd.DataFrame,
    consumption_forecast: pd.DataFrame,
    temp_forecast: pd.DataFrame,
    capacity_df: pd.DataFrame | None,
    horizon_hours: int = 48,
) -> pd.DataFrame:
    """Build feature matrix for the forecast horizon.

    Uses the tail of historical data for lags and appends forecast values
    for future periods at the configured resolution.
    """
    now = historical_df["timestamp"].max()
    freq_str = f"{RESOLUTION_MINUTES}min" if RESOLUTION_MINUTES < 60 else "h"
    periods = horizon_hours * PERIODS_PER_HOUR
    period_delta = timedelta(minutes=RESOLUTION_MINUTES)

    future_times = pd.date_range(
        start=now + period_delta,
        periods=periods,
        freq=freq_str,
        tz="UTC",
    )
    future = pd.DataFrame({"timestamp": future_times})

    # Merge forecast inputs
    if not wind_forecast.empty:
        wf = _resample_to_target_resolution(wind_forecast)
        wf = wf.rename(columns={"value": "wind"})
        future = pd.merge(future, wf[["timestamp", "wind"]], on="timestamp", how="left")

    if not consumption_forecast.empty:
        cf = _resample_to_target_resolution(consumption_forecast)
        cf = cf.rename(columns={"value": "consumption"})
        future = pd.merge(future, cf[["timestamp", "consumption"]], on="timestamp", how="left")

    if not temp_forecast.empty:
        future = pd.merge(future, temp_forecast, on="timestamp", how="left")

    if capacity_df is not None and not capacity_df.empty:
        future = pd.merge(future, capacity_df, on="timestamp", how="left")

    # Fill missing energy columns with last known values
    last_row = historical_df.iloc[-1]
    for col in ["wind", "consumption", "nuclear", "hydro", "chp_dh", "chp_ind",
                 "electric_boiler", "solar", "total_production", "net_import_export",
                 "temperature",
                 "import_se1", "import_se3", "import_ee",
                 "export_se1", "export_se3", "export_ee",
                 "gas_ttf", "carbon_ets", "coal_api2",
                 "reservoir_nordic", "reservoir_fi"]:
        if col not in future.columns:
            future[col] = np.nan
        future[col] = future[col].fillna(last_row.get(col, np.nan))

    # Use last known price for lag features
    future["price"] = np.nan

    # Concatenate historical tail (for lags) + future
    # 7 days in periods (e.g., 7*24*4 = 672 periods for 15-min resolution)
    tail_size = 168 * PERIODS_PER_HOUR
    tail = historical_df.tail(tail_size).copy()
    combined = pd.concat([tail, future], ignore_index=True)

    # Build features on combined
    combined = build_feature_matrix(combined)

    # Return only the future rows
    forecast_features = combined.tail(periods).copy()
    return forecast_features


def build_backtest_test_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature matrix for backtest test period.

    Mimics production conditions where future prices are unknown.
    Uses same logic as build_forecast_features but for historical test data.

    Args:
        train_df: Training period data with timestamp, price, and energy columns
        test_df: Test period data (same columns)

    Returns:
        Feature matrix for test period with realistic feature values
        (price rolling features use NaN for test prices)
    """
    # Mask test prices to simulate production (future unknown)
    test_masked = test_df.copy()
    test_masked["price"] = np.nan

    # Use historical tail for lag computation (7 days in periods)
    tail_size = min(168 * PERIODS_PER_HOUR, len(train_df))
    tail = train_df.tail(tail_size).copy()

    # Concatenate: [historical tail] + [test with masked prices]
    combined = pd.concat([tail, test_masked], ignore_index=True)

    # Build features on combined data
    # Price lags will pull from historical data (correct)
    # Price rolling stats will be NaN or partial (matches production)
    combined = build_feature_matrix(combined)

    # Return only the test rows
    test_features = combined.tail(len(test_df)).copy()
    return test_features
