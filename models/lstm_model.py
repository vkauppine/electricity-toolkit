"""LSTM neural network model for price forecasting."""

import numpy as np


class _LSTMNet:
    """PyTorch LSTM network."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.nn = nn

        class Net(nn.Module):
            def __init__(self_net):
                super().__init__()
                self_net.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self_net.drop1 = nn.Dropout(dropout)
                self_net.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
                self_net.drop2 = nn.Dropout(dropout)
                self_net.fc = nn.Linear(hidden_dim // 2, output_dim)

            def forward(self_net, x):
                x, _ = self_net.lstm1(x)
                x = self_net.drop1(x)
                x, _ = self_net.lstm2(x)
                x = self_net.drop2(x)
                x = x[:, -1, :]
                return self_net.fc(x)

        self.net = Net()

    def parameters(self):
        return self.net.parameters()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def __call__(self, x):
        return self.net(x)


class LSTMPriceModel:
    """LSTM model with Monte Carlo Dropout for uncertainty estimation."""

    def __init__(self, lookback: int = 168, forecast_horizon: int = 48,
                 hidden_dim: int = 128, dropout: float = 0.2, epochs: int = 50):
        import torch
        self.torch = torch
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.is_fitted = False

    def _create_sequences(self, X: np.ndarray, y: np.ndarray):
        """Create (lookback, features) input sequences and (horizon,) targets."""
        Xs, ys = [], []
        for i in range(self.lookback, len(X) - self.forecast_horizon + 1):
            Xs.append(X[i - self.lookback:i])
            ys.append(y[i:i + self.forecast_horizon])
        return np.array(Xs), np.array(ys)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train LSTM on feature matrix X and price target y."""
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        print("  Training LSTM model...")

        # Scale features and target
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        if len(X_seq) < 10:
            print("  Warning: insufficient data for LSTM training")
            return

        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq)

        self.model = _LSTMNet(X.shape[1], self.hidden_dim, self.forecast_horizon, self.dropout)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point prediction using last lookback rows of X."""
        import torch
        self.model.eval()
        X_scaled = self.scaler_X.transform(X[-self.lookback:])
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(X_tensor).numpy().ravel()
        return self.scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()

    def predict_with_intervals(self, X: np.ndarray, n_samples: int = 100,
                                confidence: float = 0.95) -> dict:
        """Monte Carlo Dropout: run n_samples forward passes with dropout active."""
        import torch
        X_scaled = self.scaler_X.transform(X[-self.lookback:])
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)

        # Enable dropout during inference
        self.model.train()
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(X_tensor).numpy().ravel()
                pred = self.scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()
                predictions.append(pred)

        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)

        z = 1.96 if confidence == 0.95 else __import__("scipy").stats.norm.ppf((1 + confidence) / 2)

        return {
            "mean": mean,
            "lower": mean - z * std,
            "upper": mean + z * std,
        }
