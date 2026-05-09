import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

class Config:
    seq_len      = 60
    d_model      = 64
    n_heads      = 4
    n_layers     = 3
    d_ff         = 256
    dropout      = 0.1
    batch_size   = 32
    epochs       = 50
    lr           = 1e-4
    train_split  = 0.8
    device       = "cpu"

cfg = Config()

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "macd", "macd_signal", "macd_hist",
    "rsi", "stoch_k", "stoch_d",
    "bb_upper", "bb_lower", "bb_mid", "bb_pct", "bb_width",
    "atr", "obv",
    "price
