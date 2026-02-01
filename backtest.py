"""Walk-forward backtesting framework for electricity price forecasting."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from config import TRAINING_DAYS, FORECAST_HORIZON_HOURS, CONFIDENCE_LEVEL
from models import check_package
from models.ensemble import EnsembleModel
from features import (
    fetch_historical_features, build_feature_matrix,
    get_feature_columns,
)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, and MAPE for a single forecast window."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan}

    errors = y_pred - y_true
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    # MAPE: avoid division by zero
    nonzero = np.abs(y_true) > 0.01
    if nonzero.sum() > 0:
        mape = np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100
    else:
        mape = np.nan

    return {"rmse": rmse, "mae": mae, "mape": mape}


def _train_and_predict(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,
    forecast_hours: int,
    confidence: float = CONFIDENCE_LEVEL,
    lstm_epochs: int = 15,
) -> dict[str, np.ndarray]:
    """Train all available models and return point predictions.

    Returns dict mapping model name to predicted array.
    """
    predictions = {}

    # Validation split for ensemble weights (last 48h of training)
    val_size = min(48, len(X_train) // 5)
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    ensemble = EnsembleModel()
    val_predictions = {}

    # XGBoost
    if check_package("xgboost"):
        try:
            from models.xgboost_model import XGBoostPriceModel
            model = XGBoostPriceModel(confidence=confidence)
            model.fit(X_tr, y_tr)
            val_predictions["XGBoost"] = model.predict(X_val)
            predictions["XGBoost"] = model.predict(X_test)
            ensemble.add_model("XGBoost", model)
        except Exception as e:
            print(f"    XGBoost failed: {e}")

    # SARIMA
    if check_package("statsmodels"):
        try:
            from models.sarima_model import SARIMAPriceModel
            model = SARIMAPriceModel()
            model.fit(y_tr)
            val_predictions["SARIMA"] = model.predict(val_size)
            pred = model.predict(len(X_test))
            predictions["SARIMA"] = pred[:len(X_test)]
            ensemble.add_model("SARIMA", model)
        except Exception as e:
            print(f"    SARIMA failed: {e}")

    # LSTM
    if check_package("torch"):
        try:
            from models.lstm_model import LSTMPriceModel
            lookback = min(168, len(X_tr) // 3)
            model = LSTMPriceModel(
                lookback=lookback,
                forecast_horizon=forecast_hours,
                epochs=lstm_epochs,
            )
            model.fit(X_tr, y_tr)
            if model.is_fitted:
                val_predictions["LSTM"] = model.predict(X_val)
                predictions["LSTM"] = model.predict(X_test)
                ensemble.add_model("LSTM", model)
        except Exception as e:
            print(f"    LSTM failed: {e}")

    # Ensemble
    if val_predictions and len(val_predictions) > 1:
        ensemble.compute_weights(val_predictions, y_val)
        # Compute ensemble prediction as weighted average
        n = min(len(v) for v in predictions.values())
        ens_pred = np.zeros(n)
        total_w = sum(ensemble.weights.get(k, 0) for k in predictions)
        if total_w > 0:
            for name, pred in predictions.items():
                w = ensemble.weights.get(name, 0)
                ens_pred += (w / total_w) * pred[:n]
            predictions["Ensemble"] = ens_pred

    return predictions


def _aggregate_metrics(step_results: list[dict]) -> dict:
    """Aggregate per-step metrics into summary statistics per model.

    Returns {model_name: {rmse_mean, rmse_std, mae_mean, mae_std, mape_mean, mape_std}}.
    """
    # Collect all model names
    model_names = set()
    for sr in step_results:
        model_names.update(sr["metrics"].keys())

    summary = {}
    for name in sorted(model_names):
        rmses, maes, mapes = [], [], []
        for sr in step_results:
            m = sr["metrics"].get(name)
            if m is None:
                continue
            if np.isfinite(m["rmse"]):
                rmses.append(m["rmse"])
            if np.isfinite(m["mae"]):
                maes.append(m["mae"])
            if np.isfinite(m["mape"]):
                mapes.append(m["mape"])

        summary[name] = {
            "rmse_mean": np.mean(rmses) if rmses else np.nan,
            "rmse_std": np.std(rmses) if rmses else np.nan,
            "mae_mean": np.mean(maes) if maes else np.nan,
            "mae_std": np.std(maes) if maes else np.nan,
            "mape_mean": np.mean(mapes) if mapes else np.nan,
            "mape_std": np.std(mapes) if mapes else np.nan,
            "n_windows": len(rmses),
        }

    return summary


def run_backtest(
    backtest_days: int = 30,
    step_hours: int = 24,
    training_days: int = TRAINING_DAYS,
    forecast_hours: int = FORECAST_HORIZON_HOURS,
    confidence: float = CONFIDENCE_LEVEL,
    feature_columns: list[str] | None = None,
) -> dict:
    """Run walk-forward backtesting.

    1. Fetch entire (training_days + backtest_days) window ONCE
    2. Build full feature matrix ONCE
    3. Slide cutoff forward by step_hours, train + predict each window
    4. Aggregate and return results

    Args:
        backtest_days: number of days in the test period
        step_hours: retrain interval in hours
        training_days: training window size in days
        forecast_hours: forecast horizon per step
        confidence: confidence level for intervals
        feature_columns: override feature columns (for compare mode)

    Returns:
        dict with keys: summary, step_results, config
    """
    now = datetime.now(timezone.utc)
    total_days = training_days + backtest_days
    start = now - timedelta(days=total_days)
    start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Fetch all data once
    print("=" * 60)
    print(f"BACKTEST: Fetching {total_days} days of historical data")
    print("=" * 60)
    historical = fetch_historical_features(start_str, end_str)
    print(f"  Got {len(historical)} hourly rows")

    # 2. Build full feature matrix once
    print("\nBuilding feature matrix...")
    features = build_feature_matrix(historical)

    # Determine feature columns
    if feature_columns is None:
        feature_columns = get_feature_columns()
    available_cols = [c for c in feature_columns if c in features.columns]
    print(f"  Using {len(available_cols)} features")

    # Prepare full arrays
    subset = features.dropna(subset=["price"] + available_cols[:5])
    X_full = subset[available_cols].fillna(0).values
    y_full = subset["price"].values
    timestamps = subset["timestamp"].values

    # 3. Walk-forward loop
    train_size = training_days * 24
    backtest_start_idx = max(train_size, 0)
    total_hours = len(X_full)

    step_results = []
    step_num = 0
    cutoff = backtest_start_idx

    print(f"\nWalk-forward: train={train_size}h, step={step_hours}h, "
          f"forecast={forecast_hours}h")
    print(f"Total data points: {total_hours}")
    print("-" * 60)

    while cutoff + forecast_hours <= total_hours:
        step_num += 1
        train_start = max(0, cutoff - train_size)
        train_end = cutoff
        test_end = min(cutoff + forecast_hours, total_hours)

        X_train = X_full[train_start:train_end]
        y_train = y_full[train_start:train_end]
        X_test = X_full[cutoff:test_end]
        y_true = y_full[cutoff:test_end]
        step_timestamps = timestamps[cutoff:test_end]

        if len(X_train) < 100 or len(X_test) == 0:
            cutoff += step_hours
            continue

        print(f"\n  Step {step_num}: train [{train_start}:{train_end}], "
              f"test [{cutoff}:{test_end}]")

        # Train and predict
        preds = _train_and_predict(
            X_train, y_train, X_test,
            forecast_hours=forecast_hours,
            confidence=confidence,
            lstm_epochs=15,
        )

        # Compute metrics per model
        metrics = {}
        for model_name, y_pred in preds.items():
            n = min(len(y_true), len(y_pred))
            metrics[model_name] = _compute_metrics(y_true[:n], y_pred[:n])

        step_results.append({
            "step": step_num,
            "cutoff_idx": cutoff,
            "timestamps": step_timestamps,
            "y_true": y_true,
            "predictions": preds,
            "metrics": metrics,
        })

        cutoff += step_hours

    if not step_results:
        print("\nNo backtest steps could be run. Check data availability.")
        return {"summary": {}, "step_results": [], "config": {}}

    # 4. Aggregate
    summary = _aggregate_metrics(step_results)

    results = {
        "summary": summary,
        "step_results": step_results,
        "config": {
            "backtest_days": backtest_days,
            "step_hours": step_hours,
            "training_days": training_days,
            "forecast_hours": forecast_hours,
            "n_features": len(available_cols),
            "feature_columns": available_cols,
        },
    }

    _print_backtest_report(results)
    return results


def _print_backtest_report(results: dict):
    """Print formatted backtest results to console."""
    summary = results["summary"]
    config = results["config"]

    print("\n" + "=" * 70)
    print("  BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Period: {config['backtest_days']} days, "
          f"Step: {config['step_hours']}h, "
          f"Horizon: {config['forecast_hours']}h")
    print(f"  Training window: {config['training_days']} days, "
          f"Features: {config['n_features']}")
    print(f"  Windows evaluated: {len(results['step_results'])}")
    print()
    print(f"  {'Model':<12} {'RMSE':>10} {'MAE':>10} {'MAPE':>10} {'Windows':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for name, m in sorted(summary.items(), key=lambda x: x[1].get("rmse_mean", 999)):
        rmse_str = f"{m['rmse_mean']:.2f}±{m['rmse_std']:.2f}" if np.isfinite(m['rmse_mean']) else "N/A"
        mae_str = f"{m['mae_mean']:.2f}±{m['mae_std']:.2f}" if np.isfinite(m['mae_mean']) else "N/A"
        mape_str = f"{m['mape_mean']:.1f}%±{m['mape_std']:.1f}" if np.isfinite(m['mape_mean']) else "N/A"
        print(f"  {name:<12} {rmse_str:>10} {mae_str:>10} {mape_str:>10} {m['n_windows']:>8}")

    print()


def plot_backtest_results(results: dict, output_path: str = "backtest_results.html") -> str:
    """Generate interactive Plotly chart of backtest results.

    Shows actual vs predicted prices across all backtest windows.
    Returns the output file path.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  Plotly not installed — skipping chart (pip install plotly)")
        return ""

    step_results = results["step_results"]
    summary = results["summary"]
    if not step_results:
        return ""

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Backtest: Actual vs Predicted", "Per-Window RMSE"),
        row_heights=[0.65, 0.35],
        vertical_spacing=0.12,
    )

    # Top: actual vs predicted prices
    # Collect all actual values
    all_times = []
    all_actual = []
    for sr in step_results:
        ts = pd.to_datetime(sr["timestamps"])
        all_times.extend(ts)
        all_actual.extend(sr["y_true"])

    fig.add_trace(
        go.Scatter(x=all_times, y=all_actual, name="Actual",
                   line=dict(color="black", width=1.5)),
        row=1, col=1,
    )

    # Predicted per model
    colors = {"XGBoost": "#1f77b4", "SARIMA": "#ff7f0e", "LSTM": "#2ca02c",
              "Ensemble": "#d62728"}
    model_names = set()
    for sr in step_results:
        model_names.update(sr["predictions"].keys())

    for model_name in sorted(model_names):
        color = colors.get(model_name, "#9467bd")
        times_list = []
        preds_list = []
        for sr in step_results:
            if model_name not in sr["predictions"]:
                continue
            ts = pd.to_datetime(sr["timestamps"])
            pred = sr["predictions"][model_name]
            n = min(len(ts), len(pred))
            times_list.extend(ts[:n])
            preds_list.extend(pred[:n])

        if times_list:
            fig.add_trace(
                go.Scatter(x=times_list, y=preds_list, name=model_name,
                           line=dict(color=color, width=1, dash="dot"),
                           opacity=0.7),
                row=1, col=1,
            )

    # Bottom: per-window RMSE
    for model_name in sorted(model_names):
        color = colors.get(model_name, "#9467bd")
        window_times = []
        window_rmses = []
        for sr in step_results:
            m = sr["metrics"].get(model_name)
            if m and np.isfinite(m["rmse"]):
                window_times.append(pd.to_datetime(sr["timestamps"][0]))
                window_rmses.append(m["rmse"])

        if window_times:
            fig.add_trace(
                go.Bar(x=window_times, y=window_rmses, name=f"{model_name} RMSE",
                       marker_color=color, opacity=0.6, showlegend=False),
                row=2, col=1,
            )

    fig.update_layout(
        title="Walk-Forward Backtest Results",
        height=800,
        template="plotly_white",
    )
    fig.update_yaxes(title_text="EUR/MWh", row=1, col=1)
    fig.update_yaxes(title_text="RMSE", row=2, col=1)

    fig.write_html(output_path)
    print(f"\n  Backtest chart saved: {output_path}")
    return output_path


def run_feature_comparison(
    backtest_days: int = 30,
    step_hours: int = 24,
    training_days: int = TRAINING_DAYS,
    forecast_hours: int = FORECAST_HORIZON_HOURS,
) -> dict:
    """Compare all 37 features vs original 31 features.

    Runs walk-forward twice and prints RMSE delta.
    """
    # Original 31 features (without fuel/reservoir)
    original_features = [
        "hour", "day_of_week", "month", "is_weekend", "is_holiday",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "price_lag_1h", "price_lag_24h", "price_lag_168h",
        "price_rmean_24h", "price_rstd_24h", "price_rmean_168h", "price_rstd_168h",
        "wind", "consumption", "nuclear", "hydro", "thermal",
        "temperature", "temperature_lag_24h", "heating_degree_hours",
        "import_se1", "import_se3", "import_ee",
        "export_se1", "export_se3", "export_ee", "net_import_capacity",
        "supply_demand_ratio", "wind_penetration", "temp_change_24h", "price_momentum",
    ]

    all_features = get_feature_columns()

    print("\n" + "=" * 70)
    print("  FEATURE COMPARISON: Original vs Extended")
    print("=" * 70)

    print("\n--- Run 1: All features ({} total) ---".format(len(all_features)))
    results_all = run_backtest(
        backtest_days=backtest_days,
        step_hours=step_hours,
        training_days=training_days,
        forecast_hours=forecast_hours,
        feature_columns=all_features,
    )

    print("\n--- Run 2: Original features ({} total) ---".format(len(original_features)))
    results_orig = run_backtest(
        backtest_days=backtest_days,
        step_hours=step_hours,
        training_days=training_days,
        forecast_hours=forecast_hours,
        feature_columns=original_features,
    )

    # Print comparison
    print("\n" + "=" * 70)
    print("  FEATURE COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n  {'Model':<12} {'All RMSE':>12} {'Orig RMSE':>12} {'Delta':>10} {'Change':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    all_models = set(results_all["summary"].keys()) | set(results_orig["summary"].keys())
    for name in sorted(all_models):
        rmse_all = results_all["summary"].get(name, {}).get("rmse_mean", np.nan)
        rmse_orig = results_orig["summary"].get(name, {}).get("rmse_mean", np.nan)
        if np.isfinite(rmse_all) and np.isfinite(rmse_orig):
            delta = rmse_all - rmse_orig
            pct = (delta / rmse_orig) * 100
            direction = "better" if delta < 0 else "worse"
            print(f"  {name:<12} {rmse_all:>12.2f} {rmse_orig:>12.2f} "
                  f"{delta:>+10.2f} {pct:>+9.1f}% ({direction})")
        else:
            print(f"  {name:<12} {'N/A':>12} {'N/A':>12} {'N/A':>10} {'N/A':>10}")

    print()

    return {
        "all_features": results_all,
        "original_features": results_orig,
    }
