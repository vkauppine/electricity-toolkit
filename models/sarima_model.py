"""SARIMA time series model for price forecasting."""

import numpy as np
import warnings


class SARIMAPriceModel:
    """Seasonal ARIMA model using only historical price series."""

    def __init__(self, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24)):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        self.SARIMAX = SARIMAX
        self.order = order
        self.seasonal_order = seasonal_order
        self.result = None
        self.is_fitted = False

    def fit(self, y: np.ndarray):
        """Fit SARIMA on price series. Uses last 720 hours (30 days) max
        to keep fitting tractable."""
        print("  Training SARIMA model...")
        y = y[-720:]  # Limit series length for speed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = self.SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.result = model.fit(disp=False, maxiter=200)
        self.is_fitted = True
        print(f"  SARIMA AIC: {self.result.aic:.1f}")

    def predict(self, steps: int) -> np.ndarray:
        forecast = self.result.get_forecast(steps=steps)
        mean = forecast.predicted_mean
        return np.asarray(mean)

    def predict_with_intervals(self, steps: int, confidence: float = 0.95) -> dict:
        forecast = self.result.get_forecast(steps=steps)
        ci = forecast.conf_int(alpha=1 - confidence)
        mean = np.asarray(forecast.predicted_mean)
        ci = np.asarray(ci)
        return {
            "mean": mean,
            "lower": ci[:, 0],
            "upper": ci[:, 1],
        }
