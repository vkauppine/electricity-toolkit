# Energia

Finnish electricity market dashboard and price forecasting toolkit. Real-time monitoring of spot prices, production, consumption, weather, and Nordic hydro reservoir levels — all from the terminal.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
python sahko.py       # Launch dashboard
```

## Commands

### sahko.py — Terminal Dashboard

```bash
python sahko.py                # Full dashboard (default: today)
python sahko.py now            # Current status only
python sahko.py tomorrow       # Tomorrow's forecast
python sahko.py week           # Weekly summary
python sahko.py dash           # Rich panel dashboard
python sahko.py live           # Interactive dashboard (keyboard controls)
python sahko.py table          # Hourly price table
python sahko.py table --hours 48  # Extended table
python sahko.py reservoir      # Nordic hydro reservoir levels
```

Language: add `--lang en` or `--en` for English (default is Finnish).

#### Interactive mode keys (`live`)

| Key | Action |
|-----|--------|
| `r` | Refresh |
| `t` | Today view |
| `w` | Week view |
| `h` | Tomorrow view |
| `q` | Quit |

### main.py — Charts & Forecasts

```bash
python main.py                          # All scatter plots + weather
python main.py --chart wind             # Wind vs temperature
python main.py --chart consumption      # Consumption vs temperature
python main.py --chart weather          # Weather timeline
python main.py --forecast               # 48-hour price forecast
python main.py --forecast --training-days 180  # Longer training window
python main.py --backtest                      # Walk-forward backtest (30 days)
python main.py --backtest --backtest-days 7 --step-hours 24  # Quick backtest
python main.py --backtest --compare-features   # Compare new vs original features
python main.py --city Helsinki --days 14       # Custom city & range
python main.py --scatter-years 2024,2025,2026  # Custom years
python main.py --no-open                # Don't open browser
```

## Configuration

### API Keys

Create a `.env` file in the project root:

```env
FINGRID_API_KEY=your_key_here
ENTSOE_API_KEY=your_key_here
OILPRICE_API_KEY=your_key_here
```

| Key | Source | Required | Purpose |
|-----|--------|----------|---------|
| `FINGRID_API_KEY` | [data.fingrid.fi](https://data.fingrid.fi/) | For real-time data | Wind, nuclear, hydro, consumption, transmission |
| `ENTSOE_API_KEY` | [transparency.entsoe.eu](https://transparency.entsoe.eu/) | For reservoir data | Nordic hydro reservoir filling levels |
| `OILPRICE_API_KEY` | [oilpriceapi.com](https://www.oilpriceapi.com/) | For fuel prices | TTF gas, EU carbon ETS, coal API2 |

**Fingrid**: Register at data.fingrid.fi to get an API key immediately.

**ENTSO-E**: Register at transparency.entsoe.eu, then email `transparency@entsoe.eu` with subject "Restful API access" to request API access. Token is generated from account settings after approval.

All features degrade gracefully if keys are missing — Fingrid falls back to CSV files in `data/`, reservoir view shows a missing-key message, fuel prices fill with NaN.

### Defaults (config.py)

| Setting | Default |
|---------|---------|
| City | Tampere |
| History days | 7 |
| Forecast horizon | 48 hours |
| Training window | 90 days |
| Confidence level | 95% |

## Data Sources

| Source | Data | Resolution |
|--------|------|------------|
| [sahkotin.fi](https://sahkotin.fi/) | Day-ahead spot prices (EUR/MWh) | Hourly |
| [Fingrid API](https://data.fingrid.fi/) | Wind, nuclear, hydro, CHP, consumption, transmission | 3-min / hourly |
| [FMI Open Data](https://opendata.fmi.fi/) | Temperature, wind speed, cloud cover, forecasts | Hourly |
| [ENTSO-E](https://transparency.entsoe.eu/) | Hydro reservoir filling levels (FI, SE, NO) | Weekly |
| [OilPrice API](https://www.oilpriceapi.com/) | TTF gas, EU carbon ETS, coal API2 prices | Daily |

## Price Forecasting

Three models combined via inverse-variance ensemble weighting:

| Model | Package | Description |
|-------|---------|-------------|
| XGBoost | `xgboost` | Gradient boosting with quantile regression |
| SARIMA | `statsmodels` | Seasonal ARIMA(2,1,2)(1,1,1,24) |
| LSTM | `torch` | 2-layer LSTM with Monte Carlo Dropout |

37 engineered features including temporal cycles, price lags, rolling statistics, production mix, weather, transmission capacity, fuel/carbon prices, Nordic reservoir levels, and derived ratios (supply/demand, wind penetration, gas SRMC).

### Backtesting

Walk-forward validation that slides a training window forward through historical data:

```bash
python main.py --backtest                          # 30-day test, daily retraining
python main.py --backtest --backtest-days 7        # Quick 7-day test
python main.py --backtest --step-hours 48          # Retrain every 48h
python main.py --backtest --compare-features       # New (37) vs original (31) features
```

Each step trains all models on the preceding window, predicts the next horizon, and records RMSE/MAE/MAPE. Results are printed as a summary table and saved as an interactive Plotly chart (`backtest_results.html`).

LSTM epochs are reduced to 15 during backtesting for speed. Ensemble weights are recomputed per window using a 48-hour internal validation split.

Install only what you need — each model is optional:

```bash
pip install xgboost        # XGBoost model
pip install statsmodels     # SARIMA model
pip install torch           # LSTM model (large download)
```

## Project Structure

```
energia/
  sahko.py              # Terminal dashboard (main CLI)
  main.py               # Chart generation & forecast CLI
  config.py             # Configuration & constants
  fetch_prices.py       # Spot prices from sahkotin.fi
  fetch_fingrid.py      # Production/consumption from Fingrid
  fetch_fmi.py          # Weather from FMI
  fetch_forecasts.py    # Wind/consumption/temperature forecasts
  fetch_entsoe.py       # Hydro reservoir levels from ENTSO-E
  fetch_fuel_prices.py  # Fuel & carbon prices from OilPrice API
  features.py           # Feature engineering pipeline
  forecast.py           # Forecast orchestration
  backtest.py           # Walk-forward backtesting framework
  plots.py              # Scatter & weather charts
  plots_forecast.py     # Forecast visualization
  models/
    ensemble.py         # Inverse-variance ensemble
    xgboost_model.py    # XGBoost price model
    sarima_model.py     # SARIMA time series model
    lstm_model.py       # LSTM neural network
  data/                 # CSV fallback cache
  .env                  # API keys (not committed)
```

## Requirements

Core:
- Python 3.10+
- requests, pandas, plotly, python-dotenv, scikit-learn, fmiopendata

Optional:
- `rich` — Rich terminal dashboard (`sahko dash`/`live`)
- `statsmodels` — SARIMA forecasting
- `xgboost` — XGBoost forecasting
- `torch` — LSTM forecasting
