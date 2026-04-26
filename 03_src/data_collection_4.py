import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR      = os.path.join(BASE_DIR, "01_data", "raw")
PROCESSED_DIR= os.path.join(BASE_DIR, "01_data", "processed")
FEATURES_DIR = os.path.join(BASE_DIR, "01_data", "features")

os.makedirs(RAW_DIR,       exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR,  exist_ok=True)

START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"
MIN_ROWS   = 500
WINDOW     = 60

# ── STOCK LIST ─────────────────────────────────────────────────────────────────
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS",        "HDFCBANK.NS",  "INFY.NS",      "ICICIBANK.NS",
    "HINDUNILVR.NS","SBIN.NS",       "BHARTIARTL.NS","KOTAKBANK.NS", "ITC.NS",
    "LT.NS",        "AXISBANK.NS",   "ASIANPAINT.NS","MARUTI.NS",    "TITAN.NS",
    "SUNPHARMA.NS", "BAJFINANCE.NS", "HCLTECH.NS",   "WIPRO.NS",     "ULTRACEMCO.NS",
    "POWERGRID.NS", "NTPC.NS",       "ONGC.NS",      "JSWSTEEL.NS",  "TATAMOTORS.NS",
    "TATASTEEL.NS", "ADANIENT.NS",   "ADANIPORTS.NS","COALINDIA.NS", "DIVISLAB.NS",
    "DRREDDY.NS",   "EICHERMOT.NS",  "CIPLA.NS",     "GRASIM.NS",    "HEROMOTOCO.NS",
    "BAJAJFINSV.NS","BAJAJ-AUTO.NS", "BPCL.NS",      "BRITANNIA.NS", "HINDALCO.NS",
    "INDUSINDBK.NS","NESTLEIND.NS",  "SBILIFE.NS",   "TECHM.NS",     "TATACONSUM.NS",
    "APOLLOHOSP.NS","HDFCLIFE.NS",   "LTIM.NS",      "TRENT.NS",     "SHRIRAMFIN.NS",
]

EXTRA = [
    "ABB.NS",       "AMBUJACEM.NS",  "AUROPHARMA.NS","BANDHANBNK.NS","BERGEPAINT.NS",
    "CHOLAFIN.NS",  "COLPAL.NS",     "DABUR.NS",     "DLF.NS",       "GODREJCP.NS",
    "HAVELLS.NS",   "IRCTC.NS",      "LUPIN.NS",      "MUTHOOTFIN.NS","NAUKRI.NS",
    "OFSS.NS",      "PIDILITIND.NS", "POLYCAB.NS",   "RECLTD.NS",    "SIEMENS.NS",
    "VEDL.NS",      "VOLTAS.NS",     "YESBANK.NS",   "INDIGO.NS",    "MPHASIS.NS",
    "PERSISTENT.NS","LTTS.NS",       "COFORGE.NS",   "FEDERALBNK.NS","TVSMOTOR.NS",
]

ALL_STOCKS = list(dict.fromkeys(NIFTY_50 + EXTRA))


# ── HELPERS ────────────────────────────────────────────────────────────────────
def fix_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df


def download_one(ticker):
    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return None
        df = fix_columns(df)
        if len(df) < MIN_ROWS:
            return None
        df.index = pd.to_datetime(df.index)
        return df.dropna()
    except Exception:
        return None


def preprocess(df):
    df = fix_columns(df.copy())
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.ffill().dropna()
    df = df[df["volume"] > 0]
    df["returns"]     = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    return df.dropna()


def add_indicators(df):
    df    = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    df["ema_9"]       = EMAIndicator(c, 9).ema_indicator()
    df["ema_21"]      = EMAIndicator(c, 21).ema_indicator()
    df["ema_50"]      = EMAIndicator(c, 50).ema_indicator()
    df["sma_20"]      = SMAIndicator(c, 20).sma_indicator()
    df["sma_50"]      = SMAIndicator(c, 50).sma_indicator()

    m = MACD(c)
    df["macd"]        = m.macd()
    df["macd_signal"] = m.macd_signal()
    df["macd_hist"]   = m.macd_diff()

    df["rsi"]         = RSIIndicator(c, 14).rsi()

    st = StochasticOscillator(h, l, c)
    df["stoch_k"]     = st.stoch()
    df["stoch_d"]     = st.stoch_signal()

    bb = BollingerBands(c, 20, 2)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_mid"]      = bb.bollinger_mavg()
    df["bb_pct"]      = bb.bollinger_pband()
    df["bb_width"]    = bb.bollinger_wband()

    df["atr"]         = AverageTrueRange(h, l, c, 14).average_true_range()
    df["obv"]         = OnBalanceVolumeIndicator(c, v).on_balance_volume()

    df["price_range"] = (h - l) / c
    df["close_open"]  = (c - df["open"]) / df["open"]

    for lag in [1, 2, 3, 5, 10]:
        df[f"ret_lag{lag}"] = df["returns"].shift(lag)

    for w in [5, 10, 20]:
        df[f"roll_mean{w}"] = c.rolling(w).mean()
        df[f"roll_std{w}"]  = c.rolling(w).std()
        df[f"roll_vol{w}"]  = df["returns"].rolling(w).std()

    return df.dropna()


def normalize(df):
    df   = df.copy()
    skip = {"open", "high", "low", "close", "volume"}
    for col in df.select_dtypes(include=np.number).columns:
        if col not in skip:
            mn, mx = df[col].min(), df[col].max()
            if mx > mn:
                df[col] = (df[col] - mn) / (mx - mn)
    return df


def make_sequences(df, target="close"):
    cols = [c for c in df.columns if c != target]
    data = df[cols + [target]].values
    X, y = [], []
    for i in range(len(data) - WINDOW):
        X.append(data[i: i + WINDOW, :-1])
        y.append(data[i + WINDOW, -1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── QUICK TEST (single stock) ──────────────────────────────────────────────────
def quick_test():
    ticker = "RELIANCE.NS"
    print(f"\nTesting with {ticker} ...")
    print(f"Saving to: {BASE_DIR}\n")

    raw = download_one(ticker)
    if raw is None:
        print("Download failed. Check your internet connection.")
        return

    clean    = preprocess(raw)
    features = add_indicators(clean)
    normed   = normalize(features)
    X, y     = make_sequences(normed)

    safe = ticker.replace(".", "_")
    features.to_csv(os.path.join(PROCESSED_DIR, f"{safe}_processed.csv"))
    normed.to_csv(  os.path.join(FEATURES_DIR,  f"{safe}_features.csv"))

    print(f"  Raw rows     : {len(raw)}")
    print(f"  Clean rows   : {len(clean)}")
    print(f"  Features     : {features.shape[1]} columns")
    print(f"  X shape      : {X.shape}")
    print(f"  y shape      : {y.shape}")
    print(f"\n  Files saved to 01_data folder")
    print(f"  SUCCESS!")


# ── FULL PIPELINE ──────────────────────────────────────────────────────────────
def run_pipeline(tickers):
    print(f"\n{'='*50}")
    print(f"  Downloading {len(tickers)} stocks")
    print(f"  {START_DATE} to {END_DATE}")
    print(f"{'='*50}\n")

    raw_data = {}
    failed   = []

    for ticker in tqdm(tickers, desc="Downloading", ncols=60):
        df = download_one(ticker)
        if df is not None:
            safe = ticker.replace(".", "_")
            df.to_csv(os.path.join(RAW_DIR, f"{safe}.csv"))
            raw_data[ticker] = df
        else:
            failed.append(ticker)
        time.sleep(0.3)

    print(f"\n  Downloaded : {len(raw_data)}")
    print(f"  Failed     : {len(failed)}")

    summary = []
    print("\nProcessing ...")
    for ticker, df in tqdm(raw_data.items(), desc="Processing", ncols=60):
        try:
            clean    = preprocess(df)
            features = add_indicators(clean)
            normed   = normalize(features)
            safe     = ticker.replace(".", "_")

            features.to_csv(os.path.join(PROCESSED_DIR, f"{safe}_processed.csv"))
            normed.to_csv(  os.path.join(FEATURES_DIR,  f"{safe}_features.csv"))

            X, y = make_sequences(normed)
            summary.append({"ticker": ticker, "rows": len(features),
                            "features": features.shape[1],
                            "sequences": len(X), "status": "OK"})
        except Exception as e:
            summary.append({"ticker": ticker, "rows": 0,
                            "features": 0, "sequences": 0,
                            "status": f"ERROR: {e}"})

    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(os.path.join(PROCESSED_DIR, "summary.csv"), index=False)
    ok = df_sum[df_sum["status"] == "OK"]
    print(f"\n  Done! {len(ok)}/{len(df_sum)} stocks processed successfully")
    print(f"  Summary saved to 01_data/processed/summary.csv")


# ── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    if mode == "test":
        quick_test()
    elif mode == "nifty50":
        run_pipeline(NIFTY_50)
    elif mode == "all":
        run_pipeline(ALL_STOCKS)
    else:
        print("Usage:")
        print("  python data_collection.py test      <- single stock test")
        print("  python data_collection.py nifty50   <- NIFTY 50 stocks")
        print("  python data_collection.py all       <- all 80 stocks")
