"""Main forecasting orchestration."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from config import FORECAST_HORIZON_HOURS, TRAINING_DAYS, CONFIDENCE_LEVEL
from models import check_package
from models.ensemble import EnsembleModel
from features import (
    fetch_historical_features, build_feature_matrix,
    build_forecast_features, get_feature_columns,
)
from fetch_forecasts import (
    fetch_wind_forecast, fetch_consumption_forecast,
    fetch_transmission_capacity, fetch_national_temperature_forecast,
)


def _prepare_training_data(df: pd.DataFrame):
    """Prepare X, y arrays from feature matrix, dropping rows with NaN in key features."""
    feature_cols = get_feature_columns()
    # Keep only columns that exist
    available_cols = [c for c in feature_cols if c in df.columns]
    subset = df.dropna(subset=["price"] + available_cols[:5])  # Require price + basic features
    X = subset[available_cols].fillna(0).values
    y = subset["price"].values
    return X, y, available_cols


def run_forecast(training_days: int = TRAINING_DAYS,
                 forecast_hours: int = FORECAST_HORIZON_HOURS,
                 confidence: float = CONFIDENCE_LEVEL) -> dict:
    """Run the full forecasting pipeline.

    Returns:
        dict with keys: forecast, models_info, historical_prices, forecast_times
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=training_days)
    start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Fetch historical data
    print("=" * 60)
    print("STEP 1: Fetching historical data")
    print("=" * 60)
    historical = fetch_historical_features(start_str, end_str)
    print(f"\n  Historical data: {len(historical)} hourly rows")

    # 2. Build feature matrix
    print("\n" + "=" * 60)
    print("STEP 2: Building feature matrix")
    print("=" * 60)
    features = build_feature_matrix(historical)

    X, y, feature_cols = _prepare_training_data(features)
    print(f"  Training samples: {len(X)}, features: {len(feature_cols)}")

    # 3. Train/validation split (last 7 days = validation)
    val_hours = min(168, len(X) // 5)
    X_train, X_val = X[:-val_hours], X[-val_hours:]
    y_train, y_val = y[:-val_hours], y[-val_hours:]
    print(f"  Train: {len(X_train)}, Validation: {len(X_val)}")

    # 4. Train available models
    print("\n" + "=" * 60)
    print("STEP 3: Training models")
    print("=" * 60)

    ensemble = EnsembleModel()
    val_predictions = {}
    available_models = {}

    # XGBoost
    if check_package("xgboost"):
        try:
            from models.xgboost_model import XGBoostPriceModel
            xgb_model = XGBoostPriceModel(confidence=confidence)
            xgb_model.fit(X_train, y_train)
            val_pred = xgb_model.predict(X_val)
            val_predictions["XGBoost"] = val_pred
            ensemble.add_model("XGBoost", xgb_model)
            available_models["XGBoost"] = xgb_model
        except Exception as e:
            print(f"  XGBoost training failed: {e}")
    else:
        print("  XGBoost not installed — skipping (pip install xgboost)")

    # SARIMA
    if check_package("statsmodels"):
        try:
            from models.sarima_model import SARIMAPriceModel
            sarima = SARIMAPriceModel()
            sarima.fit(y_train)
            val_pred = sarima.predict(val_hours)
            val_predictions["SARIMA"] = val_pred
            ensemble.add_model("SARIMA", sarima)
            available_models["SARIMA"] = sarima
        except Exception as e:
            print(f"  SARIMA training failed: {e}")
    else:
        print("  statsmodels not installed — skipping SARIMA (pip install statsmodels)")

    # LSTM
    if check_package("torch"):
        try:
            from models.lstm_model import LSTMPriceModel
            lstm = LSTMPriceModel(
                lookback=min(168, len(X_train) // 3),
                forecast_horizon=forecast_hours,
                epochs=30,
            )
            lstm.fit(X_train, y_train)
            if lstm.is_fitted:
                val_pred = lstm.predict(X_val)
                val_predictions["LSTM"] = val_pred
                ensemble.add_model("LSTM", lstm)
                available_models["LSTM"] = lstm
        except Exception as e:
            print(f"  LSTM training failed: {e}")
    else:
        print("  torch not installed — skipping LSTM (pip install torch)")

    if not available_models:
        raise RuntimeError(
            "No forecasting models available. Install at least one of: "
            "xgboost, statsmodels, torch"
        )

    # 5. Compute ensemble weights
    print("\n" + "=" * 60)
    print("STEP 4: Computing ensemble weights")
    print("=" * 60)
    ensemble.compute_weights(val_predictions, y_val)

    # 6. Fetch forecast inputs
    print("\n" + "=" * 60)
    print("STEP 5: Fetching forecast inputs")
    print("=" * 60)
    forecast_start = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    forecast_end = (now + timedelta(hours=forecast_hours + 6)).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        wind_fc = fetch_wind_forecast(forecast_start, forecast_end)
    except Exception as e:
        print(f"  Wind forecast unavailable: {e}")
        wind_fc = pd.DataFrame()

    try:
        cons_fc = fetch_consumption_forecast(forecast_start, forecast_end)
    except Exception as e:
        print(f"  Consumption forecast unavailable: {e}")
        cons_fc = pd.DataFrame()

    try:
        print("\n  Fetching temperature forecast...")
        temp_fc = fetch_national_temperature_forecast()
    except Exception as e:
        print(f"  Temperature forecast unavailable: {e}")
        temp_fc = pd.DataFrame()

    try:
        cap_fc = fetch_transmission_capacity(forecast_start, forecast_end)
    except Exception as e:
        print(f"  Transmission capacity unavailable: {e}")
        cap_fc = None

    # 7. Build forecast feature matrix
    print("\n" + "=" * 60)
    print("STEP 6: Generating forecast")
    print("=" * 60)
    forecast_features = build_forecast_features(
        historical, wind_fc, cons_fc, temp_fc, cap_fc,
        horizon_hours=forecast_hours,
    )

    forecast_X = forecast_features[
        [c for c in feature_cols if c in forecast_features.columns]
    ].fillna(0).values

    forecast_times = forecast_features["timestamp"].values

    # 8. Generate predictions from each model
    model_forecasts = {}

    if "XGBoost" in available_models:
        model_forecasts["XGBoost"] = available_models["XGBoost"].predict_with_intervals(forecast_X)

    if "SARIMA" in available_models:
        model_forecasts["SARIMA"] = available_models["SARIMA"].predict_with_intervals(
            forecast_hours, confidence=confidence
        )

    if "LSTM" in available_models:
        model_forecasts["LSTM"] = available_models["LSTM"].predict_with_intervals(
            forecast_X, confidence=confidence
        )

    # 9. Ensemble
    forecast = ensemble.predict_with_intervals(model_forecasts)

    # 10. Print summary
    _print_summary(forecast, forecast_times, ensemble, confidence)

    return {
        "forecast": forecast,
        "model_forecasts": model_forecasts,
        "models_info": {
            "weights": ensemble.weights,
            "rmses": ensemble.rmses,
        },
        "historical_prices": historical[["timestamp", "price"]].tail(48),
        "forecast_times": forecast_times,
        "feature_importance": (
            available_models["XGBoost"].feature_importance()
            if "XGBoost" in available_models else None
        ),
        "feature_names": feature_cols,
    }


def _print_summary(forecast: dict, times: np.ndarray, ensemble: EnsembleModel,
                    confidence: float):
    """Print Finnish-language forecast summary."""
    mean = forecast["mean"]
    lower = forecast["lower"]
    upper = forecast["upper"]

    # Convert EUR/MWh to snt/kWh (divide by 10)
    mean_24 = mean[:24].mean() / 10
    lower_24 = lower[:24].min() / 10
    upper_24 = upper[:24].max() / 10
    mean_48 = mean[:min(48, len(mean))].mean() / 10
    ci_upper_24 = upper[:24].max() / 10

    pct = int(confidence * 100)

    print("\n" + "=" * 60)
    print("  SÄHKÖN HINTAENNUSTE")
    print("=" * 60)
    print(f"\n  Seuraavat 24 tuntia:")
    print(f"    Hinta todennäköisesti {lower_24:.1f}–{upper_24:.1f} snt/kWh")
    print(f"    Keskihinta: {mean_24:.1f} snt/kWh")
    print(f"    {pct}% varmuus: alle {ci_upper_24:.1f} snt/kWh")

    if len(mean) > 24:
        lower_48 = lower[:48].min() / 10
        upper_48 = upper[:48].max() / 10
        print(f"\n  Seuraavat 48 tuntia:")
        print(f"    Hinta todennäköisesti {lower_48:.1f}–{upper_48:.1f} snt/kWh")
        print(f"    Keskihinta: {mean_48:.1f} snt/kWh")

    print(f"\n  Mallit:")
    for name in sorted(ensemble.weights, key=lambda k: -ensemble.weights[k]):
        w = ensemble.weights[name]
        rmse = ensemble.rmses[name]
        print(f"    {name:10s}: paino {w:.2f}  (RMSE {rmse:.1f} EUR/MWh)")
