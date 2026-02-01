"""CLI entry point for energy data visualization."""

import argparse
import os
import webbrowser
from datetime import datetime, timedelta, timezone

from config import DEFAULT_CITY, DEFAULT_DAYS, DEFAULT_SCATTER_YEARS
from fetch_fingrid import fetch_wind_power, fetch_consumption
from fetch_fmi import fetch_national_temperature_weighted, fetch_daily_weather
from plots import plot_wind_vs_temperature, plot_consumption_vs_temperature, plot_weather_timeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finnish energy data visualization"
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS,
        help=f"Number of days for weather timeline (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--city", type=str, default=DEFAULT_CITY,
        help=f"City for weather timeline (default: {DEFAULT_CITY})",
    )
    parser.add_argument(
        "--scatter-years", type=str, default=None,
        help="Comma-separated years for scatter plots (e.g. 2024,2025,2026)",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Don't open charts in browser automatically",
    )
    parser.add_argument(
        "--chart", type=str, default="all",
        choices=["all", "wind", "consumption", "weather"],
        help="Which chart to generate (default: all)",
    )
    parser.add_argument(
        "--forecast", action="store_true",
        help="Generate price forecast for next 24-48 hours",
    )
    parser.add_argument(
        "--training-days", type=int, default=None,
        help="Days of historical data for forecast training (default: 90)",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run walk-forward backtest instead of forecast",
    )
    parser.add_argument(
        "--backtest-days", type=int, default=30,
        help="Test period length in days for backtest (default: 30)",
    )
    parser.add_argument(
        "--step-hours", type=int, default=24,
        help="Retrain interval in hours for backtest (default: 24)",
    )
    parser.add_argument(
        "--compare-features", action="store_true",
        help="Compare all features vs original features in backtest",
    )
    return parser.parse_args()


def build_scatter_time_range(years: list[int]) -> tuple[str, str]:
    """Build start/end ISO strings covering the given years."""
    start_year = min(years)
    end_year = max(years)
    start = f"{start_year}-01-01T00:00:00Z"
    now = datetime.now(timezone.utc)
    if end_year >= now.year:
        end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        end = f"{end_year + 1}-01-01T00:00:00Z"
    return start, end


def run_forecast_mode(args):
    """Run the price forecasting pipeline."""
    from config import TRAINING_DAYS, FORECAST_HORIZON_HOURS
    from forecast import run_forecast
    from plots_forecast import plot_price_forecast, plot_feature_importance

    training_days = args.training_days or TRAINING_DAYS
    result = run_forecast(training_days=training_days)

    charts_to_open = []
    path = plot_price_forecast(result)
    charts_to_open.append(path)

    fi_path = plot_feature_importance(result)
    if fi_path:
        charts_to_open.append(fi_path)

    if not args.no_open:
        for path in charts_to_open:
            webbrowser.open(f"file://{os.path.abspath(path)}")

    print(f"\nDone! Generated {len(charts_to_open)} forecast chart(s).")


def run_backtest_mode(args):
    """Run the backtesting pipeline."""
    from config import TRAINING_DAYS, FORECAST_HORIZON_HOURS
    from backtest import run_backtest, run_feature_comparison, plot_backtest_results

    training_days = args.training_days or TRAINING_DAYS

    if args.compare_features:
        results = run_feature_comparison(
            backtest_days=args.backtest_days,
            step_hours=args.step_hours,
            training_days=training_days,
        )
        # Plot the "all features" run
        plot_backtest_results(results["all_features"])
    else:
        results = run_backtest(
            backtest_days=args.backtest_days,
            step_hours=args.step_hours,
            training_days=training_days,
        )
        path = plot_backtest_results(results)
        if path and not args.no_open:
            webbrowser.open(f"file://{os.path.abspath(path)}")

    print("\nBacktest complete.")


def main():
    args = parse_args()

    if args.backtest:
        run_backtest_mode(args)
        return

    if args.forecast:
        run_forecast_mode(args)
        return

    if args.scatter_years:
        years = [int(y.strip()) for y in args.scatter_years.split(",")]
    else:
        years = DEFAULT_SCATTER_YEARS

    charts_to_open = []
    do_scatter = args.chart in ("all", "wind", "consumption")
    do_weather = args.chart in ("all", "weather")

    # --- Scatter plots (wind & consumption vs temperature) ---
    if do_scatter:
        start, end = build_scatter_time_range(years)
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))

        # Fetch consumption first — needed for regression-based temp weighting
        print("\n=== Fetching consumption data ===")
        cons_df = fetch_consumption(start, end)
        print(f"  Got {len(cons_df)} consumption records")

        # Fetch city temperatures and compute regression-weighted national temp
        print("\n=== Fetching city temperatures ===")
        temp_df, weights = fetch_national_temperature_weighted(start_dt, end_dt, cons_df)
        print(f"  Got {len(temp_df)} weighted temperature records")

        if args.chart in ("all", "wind"):
            print("\n=== Fetching wind power data ===")
            wind_df = fetch_wind_power(start, end)
            print(f"  Got {len(wind_df)} wind power records")
            path = plot_wind_vs_temperature(wind_df, temp_df)
            charts_to_open.append(path)

        if args.chart in ("all", "consumption"):
            path = plot_consumption_vs_temperature(cons_df, temp_df)
            charts_to_open.append(path)

    # --- Weather timeline ---
    if do_weather:
        print(f"\n=== Fetching weather for {args.city} ({args.days} days) ===")
        weather_df = fetch_daily_weather(args.city, args.days)
        print(f"  Got {len(weather_df)} daily records")
        path = plot_weather_timeline(weather_df, city=args.city)
        charts_to_open.append(path)

    # Open charts in browser
    if not args.no_open:
        for path in charts_to_open:
            webbrowser.open(f"file://{os.path.abspath(path)}")

    print(f"\nDone! Generated {len(charts_to_open)} chart(s).")


if __name__ == "__main__":
    main()
