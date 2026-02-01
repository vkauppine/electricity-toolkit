"""Forecast visualization with Plotly."""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

OUTPUT_DIR = os.path.dirname(__file__)


def plot_price_forecast(result: dict, output: str = "price_forecast.html") -> str:
    """Generate interactive price forecast chart.

    Args:
        result: output dict from forecast.run_forecast()
        output: HTML filename
    """
    forecast = result["forecast"]
    historical = result["historical_prices"]
    forecast_times = pd.to_datetime(result["forecast_times"], utc=True)

    fig = go.Figure()

    # Historical prices (last 48h)
    fig.add_trace(go.Scatter(
        x=historical["timestamp"],
        y=historical["price"] / 10,
        mode="lines+markers",
        name="Toteutunut hinta",
        line=dict(color="gray", width=2),
        marker=dict(size=3),
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=forecast_times,
        y=forecast["upper"] / 10,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_times,
        y=forecast["lower"] / 10,
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(0, 100, 255, 0.15)",
        line=dict(width=0),
        name="95% luottamusväli",
    ))

    # Forecast mean
    fig.add_trace(go.Scatter(
        x=forecast_times,
        y=forecast["mean"] / 10,
        mode="lines",
        name="Ennuste",
        line=dict(color="blue", width=3),
    ))

    # Individual model forecasts (lighter lines)
    colors = {"XGBoost": "green", "SARIMA": "orange", "LSTM": "purple"}
    for name, mf in result.get("model_forecasts", {}).items():
        n = min(len(mf["mean"]), len(forecast_times))
        fig.add_trace(go.Scatter(
            x=forecast_times[:n],
            y=mf["mean"][:n] / 10,
            mode="lines",
            name=name,
            line=dict(color=colors.get(name, "red"), width=1, dash="dot"),
            opacity=0.6,
        ))

    fig.update_layout(
        title="Sähkön hintaennuste — Suomi",
        xaxis_title="Aika",
        yaxis_title="Hinta (snt/kWh)",
        template="plotly_white",
        font=dict(size=12),
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )

    path = os.path.join(OUTPUT_DIR, output)
    fig.write_html(path, auto_open=False)
    print(f"\nForecast chart saved: {path}")
    return path


def plot_feature_importance(result: dict, output: str = "feature_importance.html") -> str | None:
    """Plot XGBoost feature importance if available."""
    importance = result.get("feature_importance")
    feature_names = result.get("feature_names")
    if importance is None or feature_names is None:
        return None

    n = min(len(importance), len(feature_names))
    importance = importance[:n]
    feature_names = feature_names[:n]

    # Sort by importance
    idx = np.argsort(importance)[-15:]  # Top 15
    fig = go.Figure(go.Bar(
        x=importance[idx],
        y=[feature_names[i] for i in idx],
        orientation="h",
        marker_color="steelblue",
    ))
    fig.update_layout(
        title="Tärkeimmät ennustepiirteet (XGBoost)",
        xaxis_title="Tärkeys",
        template="plotly_white",
        font=dict(size=12),
        height=500,
    )

    path = os.path.join(OUTPUT_DIR, output)
    fig.write_html(path, auto_open=False)
    print(f"Feature importance chart saved: {path}")
    return path
