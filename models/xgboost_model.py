"""XGBoost price forecasting model with quantile regression."""

import numpy as np


class XGBoostPriceModel:
    """Gradient boosting model that trains 3 sub-models for point + interval prediction."""

    def __init__(self, confidence: float = 0.95):
        import xgboost as xgb
        self.xgb = xgb
        alpha = (1 - confidence) / 2  # e.g. 0.025 for 95%

        base_params = dict(
            max_depth=6,
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        self.model_mean = xgb.XGBRegressor(
            **base_params, objective="reg:squarederror"
        )
        self.model_lower = xgb.XGBRegressor(
            **base_params, objective="reg:quantileerror", quantile_alpha=alpha,
        )
        self.model_upper = xgb.XGBRegressor(
            **base_params, objective="reg:quantileerror", quantile_alpha=1 - alpha,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        print("  Training XGBoost mean model...")
        self.model_mean.fit(X, y)
        print("  Training XGBoost lower quantile...")
        self.model_lower.fit(X, y)
        print("  Training XGBoost upper quantile...")
        self.model_upper.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_mean.predict(X)

    def predict_with_intervals(self, X: np.ndarray) -> dict:
        return {
            "mean": self.model_mean.predict(X),
            "lower": self.model_lower.predict(X),
            "upper": self.model_upper.predict(X),
        }

    def feature_importance(self) -> np.ndarray:
        return self.model_mean.feature_importances_
