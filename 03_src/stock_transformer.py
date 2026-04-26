import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import math

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
    device        = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

# ─────────────────────────────────────────────
# 2. YOUR EXACT FEATURE COLUMNS
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
# 3. DATASET
# ─────────────────────────────────────────────
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
        X.append(features[i - seq_len : i])
        y.append(target[i])

    return np.array(X), np.array(y), scaler_X, scaler_y


# ─────────────────────────────────────────────
# 4. POSITIONAL ENCODING
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────
# 5. TRANSFORMER MODEL
# ─────────────────────────────────────────────
class StockTransformer(nn.Module):
    def __init__(self, n_features, cfg):
        super().__init__()
        self.input_proj = nn.Linear(n_features, cfg.d_model)
        self.pos_enc    = PositionalEncoding(cfg.d_model, dropout=cfg.dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff, dropout=cfg.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, 64),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :])


# ─────────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, actuals = 0, [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        pred = model(X_b)
        total_loss += criterion(pred, y_b).item() * len(X_b)
        preds.append(pred.cpu())
        actuals.append(y_b.cpu())
    return total_loss / len(loader.dataset), torch.cat(preds), torch.cat(actuals)


def train(model, train_loader, val_loader, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    best_val  = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, cfg.device)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "04_models/best_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg.epochs}  Train MSE: {train_loss:.6f}  Val MSE: {val_loss:.6f}")

    print(f"\n✅ Best Val MSE: {best_val:.6f}  →  Model saved to 04_models/best_model.pt")


# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(df):
    df = df.copy()
    df.dropna(inplace=True)

    split = int(len(df) * cfg.train_split)
    df_train, df_val = df.iloc[:split], df.iloc[split:]

    X_train, y_train, scaler_X, scaler_y = build_sequences(df_train, cfg.seq_len)
    X_val,   y_val,   _,        _        = build_sequences(df_val,   cfg.seq_len, scaler_X, scaler_y)

    print(f"Train samples: {len(X_train)}  |  Val samples: {len(X_val)}")

    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(StockDataset(X_val,   y_val),   batch_size=cfg.batch_size)

    n_features = X_train.shape[2]
    model = StockTransformer(n_features, cfg).to(cfg.device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}  |  Device: {cfg.device}\n")

    train(model, train_loader, val_loader, cfg)

    model.load_state_dict(torch.load("04_models/best_model.pt", map_location=cfg.device))
    _, preds, actuals = evaluate(model, val_loader, nn.MSELoss(), cfg.device)

    preds_inv   = scaler_y.inverse_transform(preds.numpy())
    actuals_inv = scaler_y.inverse_transform(actuals.numpy())

    mae  = np.mean(np.abs(preds_inv - actuals_inv))
    rmse = np.sqrt(np.mean((preds_inv - actuals_inv) ** 2))
    mape = np.mean(np.abs((actuals_inv - preds_inv) / (actuals_inv + 1e-9))) * 100

    print(f"\n── Validation Metrics ──")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.2f}%")

    return model, scaler_X, scaler_y, preds_inv, actuals_inv


# ─────────────────────────────────────────────
# 8. PREDICT NEXT DAY
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_next_day(model, df_recent, scaler_X, scaler_y):
    model.eval()
    df_recent = df_recent.copy()
    df_recent.dropna(inplace=True)
    features = scaler_X.transform(df_recent[FEATURE_COLS].values)
    seq  = features[-cfg.seq_len:]
    x    = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(cfg.device)
    pred = model(x).cpu().numpy()
    return float(scaler_y.inverse_transform(pred)[0, 0])