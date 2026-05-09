import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
class Config:
    seq_len       = 60
    d_model       = 64
    n_heads       = 4
    n_layers      = 3
    d_ff          = 256
    dropout       = 0.1
    batch_size    = 32
    epochs        = 50
    lr            = 1e-4
    train_split   = 0.8
    device        = "cpu"

cfg = Config()

# ─────────────────────────────────────────────
# 2. FEATURE COLUMNS
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "macd", "macd_signal", "macd_hist",
    "rsi", "stoch_k", "stoch_d",
    "bb_upper", "bb_lower", "bb_mid", "bb_pct", "bb_width",
    "atr", "obv",
    "price_range", "close_open",
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "ret_lag10",
    "roll_mean5", "roll_std5", "roll_vol5",
    "roll_mean10", "roll_std10", "roll_vol10",
    "roll_mean20", "roll_std20", "roll_vol20"
]
TARGET_COL = "close"


# ─────────────────────────────────────────────
# 3. BUILD SEQUENCES (flatten for sklearn)
# ─────────────────────────────────────────────
def build_sequences(df, seq_len, scaler_X=None, scaler_y=None):
    df = df[FEATURE_COLS].copy()
    df.dropna(inplace=True)

    features = df[FEATURE_COLS].values
    target   = df[[TARGET_COL]].values

    if scaler_X is None:
        scaler_X = MinMaxScaler()
        features = scaler_X.fit_transform(features)
    else:
        features = scaler_X.transform(features)

    if scaler_y is None:
        scaler_y = MinMaxScaler()
        target = scaler_y.fit_transform(target)
    else:
        target = scaler_y.transform(target)

    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len : i].flatten())  # flatten for sklearn
        y.append(target[i][0])

    return np.array(X), np.array(y), scaler_X, scaler_y


# ─────────────────────────────────────────────
# 4. SKLEARN MODEL (replaces Transformer)
# ─────────────────────────────────────────────
class StockTransformer:
    """Wrapper around GradientBoostingRegressor to mimic torch model interface."""
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


# ─────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(df):
    df = df.copy()
    df.dropna(inplace=True)

    split = int(len(df) * cfg.train_split)
    df_train, df_val = df.iloc[:split], df.iloc[split:]

    X_train, y_train, scaler_X, scaler_y = build_sequences(df_train, cfg.seq_len)
    X_val,   y_val,   _,        _        = build_sequences(df_val,   cfg.seq_len, scaler_X, scaler_y)

    print(f"Train samples: {len(X_train)}  |  Val samples: {len(X_val)}")

    model = StockTransformer()
    model.fit(X_train, y_train)

    preds_scaled = model.predict(X_val).reshape(-1, 1)
    actuals_scaled = y_val.reshape(-1, 1)

    preds_inv   = scaler_y.inverse_transform(preds_scaled)
    actuals_inv = scaler_y.inverse_transform(actuals_scaled)

    mae  = np.mean(np.abs(preds_inv - actuals_inv))
    rmse = np.sqrt(np.mean((preds_inv - actuals_inv) ** 2))
    mape = np.mean(np.abs((actuals_inv - preds_inv) / (actuals_inv + 1e-9))) * 100

    print(f"\n── Validation Metrics ──")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.2f}%")

    return model, scaler_X, scaler_y, preds_inv, actuals_inv


# ─────────────────────────────────────────────
# 6. PREDICT NEXT DAY
# ─────────────────────────────────────────────
def predict_next_day(model, df_recent, scaler_X, scaler_y):
    df_recent = df_recent.copy()
    df_recent.dropna(inplace=True)
    features = scaler_X.transform(df_recent[FEATURE_COLS].values)
    seq = features[-cfg.seq_len:].flatten().reshape(1, -1)
    pred_scaled = model.predict(seq).reshape(-1, 1)
    return float(scaler_y.inverse_transform(pred_scaled)[0, 0])
