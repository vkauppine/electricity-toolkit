"""Ensemble model combining SARIMA, XGBoost, and LSTM predictions."""

import numpy as np


class EnsembleModel:
    """Weighted ensemble with inverse-variance weighting and variance pooling."""

    def __init__(self):
        self.models = {}     # name -> model
        self.weights = {}    # name -> weight
        self.rmses = {}      # name -> validation RMSE

    def add_model(self, name: str, model):
        self.models[name] = model

    def compute_weights(self, val_predictions: dict[str, np.ndarray], y_val: np.ndarray):
        """Compute weights from validation performance.

        Args:
            val_predictions: dict mapping model name to predicted values on validation set
            y_val: actual validation values
        """
        for name, pred in val_predictions.items():
            # Align lengths (SARIMA may produce different length)
            n = min(len(pred), len(y_val))
            rmse = np.sqrt(np.mean((pred[:n] - y_val[:n]) ** 2))
            self.rmses[name] = max(rmse, 0.01)  # Floor to avoid division by zero

        # Inverse variance weighting
        inv_var = {name: 1.0 / (rmse ** 2) for name, rmse in self.rmses.items()}
        total = sum(inv_var.values())
        self.weights = {name: v / total for name, v in inv_var.items()}

        print("\n  Ensemble weights:")
        for name in sorted(self.weights, key=lambda k: -self.weights[k]):
            print(f"    {name:10s}: {self.weights[name]:.3f}  (RMSE: {self.rmses[name]:.2f} EUR/MWh)")

    def predict_with_intervals(self, model_forecasts: dict[str, dict]) -> dict:
        """Combine model forecasts into ensemble prediction with confidence intervals.

        Args:
            model_forecasts: dict mapping model name to
                {'mean': array, 'lower': array, 'upper': array}

        Returns:
            {'mean': array, 'lower': array, 'upper': array}
        """
        if not model_forecasts:
            raise RuntimeError("No model forecasts to ensemble")

        # Use only models that have weights
        available = {k: v for k, v in model_forecasts.items() if k in self.weights}
        if not available:
            # Fallback: equal weights
            available = model_forecasts
            n = len(available)
            weights = {k: 1.0 / n for k in available}
        else:
            # Re-normalize weights for available models
            total = sum(self.weights[k] for k in available)
            weights = {k: self.weights[k] / total for k in available}

        # Find common forecast length
        lengths = [len(v["mean"]) for v in available.values()]
        n = min(lengths)

        # Weighted mean
        mean = np.zeros(n)
        for name, forecast in available.items():
            mean += weights[name] * forecast["mean"][:n]

        # Variance pooling
        within_var = np.zeros(n)
        between_var = np.zeros(n)
        for name, forecast in available.items():
            w = weights[name]
            # Estimate variance from interval width
            interval_width = forecast["upper"][:n] - forecast["lower"][:n]
            var_i = (interval_width / (2 * 1.96)) ** 2
            within_var += w * var_i
            between_var += w * (forecast["mean"][:n] - mean) ** 2

        total_std = np.sqrt(within_var + between_var)

        return {
            "mean": mean,
            "lower": mean - 1.96 * total_std,
            "upper": mean + 1.96 * total_std,
        }
