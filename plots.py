"""Generate interactive Plotly charts."""

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


OUTPUT_DIR = os.path.dirname(__file__)


def plot_wind_vs_temperature(wind_df: pd.DataFrame, temp_df: pd.DataFrame, output: str = "wind_vs_temperature.html"):
    """Chart 1: Scatter — wind power production vs temperature, colored by year.

    Args:
        wind_df: DataFrame with 'timestamp' and 'value' (MW)
        temp_df: DataFrame with 'timestamp' and 'temperature' (°C)
        output: output HTML filename
    """
    # Merge on hourly timestamps
    wind_df = wind_df.copy()
    temp_df = temp_df.copy()
    wind_df["hour"] = wind_df["timestamp"].dt.floor("h")
    temp_df["hour"] = temp_df["timestamp"].dt.floor("h")

    merged = pd.merge(wind_df, temp_df, on="hour", how="inner")
    merged["year"] = merged["hour"].dt.year.astype(str)

    fig = px.scatter(
        merged,
        x="temperature",
        y="value",
        color="year",
        opacity=0.4,
        labels={
            "temperature": "Average Temperature (°C)",
            "value": "Wind Power Production (MW)",
            "year": "Year",
        },
        title="Wind Power Production vs Temperature in Finland",
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        legend=dict(title="Year"),
    )
    fig.update_traces(marker=dict(size=4))

    path = os.path.join(OUTPUT_DIR, output)
    fig.write_html(path, auto_open=False)
    print(f"Chart saved: {path}")
    return path


def plot_consumption_vs_temperature(cons_df: pd.DataFrame, temp_df: pd.DataFrame, output: str = "consumption_vs_temperature.html"):
    """Chart 2: Scatter — electricity consumption vs temperature, colored by year.

    Args:
        cons_df: DataFrame with 'timestamp' and 'value' (MWh)
        temp_df: DataFrame with 'timestamp' and 'temperature' (°C)
        output: output HTML filename
    """
    cons_df = cons_df.copy()
    temp_df = temp_df.copy()
    cons_df["hour"] = cons_df["timestamp"].dt.floor("h")
    temp_df["hour"] = temp_df["timestamp"].dt.floor("h")

    merged = pd.merge(cons_df, temp_df, on="hour", how="inner")
    merged["year"] = merged["hour"].dt.year.astype(str)

    fig = px.scatter(
        merged,
        x="temperature",
        y="value",
        color="year",
        opacity=0.4,
        labels={
            "temperature": "Average Temperature (°C)",
            "value": "Electricity Consumption (MWh)",
            "year": "Year",
        },
        title="Electricity Consumption vs Temperature in Finland",
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        legend=dict(title="Year"),
    )
    fig.update_traces(marker=dict(size=4))

    path = os.path.join(OUTPUT_DIR, output)
    fig.write_html(path, auto_open=False)
    print(f"Chart saved: {path}")
    return path


def plot_weather_timeline(weather_df: pd.DataFrame, city: str = "Tampere", output: str = "weather_timeline.html"):
    """Chart 3: Weather timeline — temperature lines + precipitation bars.

    Args:
        weather_df: DataFrame with date, t_min, t_max, t_avg, precipitation
        city: city name for the title
        output: output HTML filename
    """
    fig = go.Figure()

    # Temperature range as filled area
    fig.add_trace(go.Scatter(
        x=weather_df["date"],
        y=weather_df["t_max"],
        mode="lines",
        name="Max Temp",
        line=dict(color="rgba(255, 80, 80, 0.8)", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=weather_df["date"],
        y=weather_df["t_min"],
        mode="lines",
        name="Min Temp",
        line=dict(color="rgba(80, 80, 255, 0.8)", width=1),
        fill="tonexty",
        fillcolor="rgba(180, 180, 255, 0.2)",
    ))
    fig.add_trace(go.Scatter(
        x=weather_df["date"],
        y=weather_df["t_avg"],
        mode="lines+markers",
        name="Avg Temp",
        line=dict(color="rgba(50, 50, 50, 0.9)", width=2),
        marker=dict(size=5),
    ))

    # Precipitation bars on secondary y-axis
    fig.add_trace(go.Bar(
        x=weather_df["date"],
        y=weather_df["precipitation"],
        name="Precipitation (mm)",
        marker_color="rgba(100, 149, 237, 0.6)",
        yaxis="y2",
    ))

    fig.update_layout(
        title=f"Weather in {city}",
        template="plotly_white",
        font=dict(size=12),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Temperature (°C)", side="left"),
        yaxis2=dict(
            title="Precipitation (mm)",
            side="right",
            overlaying="y",
            showgrid=False,
            rangemode="tozero",
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        barmode="overlay",
    )

    path = os.path.join(OUTPUT_DIR, output)
    fig.write_html(path, auto_open=False)
    print(f"Chart saved: {path}")
    return path
