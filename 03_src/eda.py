import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# ─────────────────────────────────────────────
# PATH
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "01_data", "processed")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

    if not files:
        print("No data found!")
        return None

    file_path = os.path.join(DATA_DIR, files[0])
    print(f"\nUsing file: {files[0]}")

    df = pd.read_csv(file_path)

    # Ensure date column exists properly
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'date'}, inplace=True)

    df['date'] = pd.to_datetime(df['date'])

    return df


# ─────────────────────────────────────────────
# BASIC INFO
# ─────────────────────────────────────────────
def print_basic_info(df):
    print("\n===== BASIC INFO =====")
    print(df.info())

    print("\n===== STATS =====")
    print(df.describe())

    print("\nMissing values:\n", df.isnull().sum())

    print("\nInfinite values:\n", (df == float("inf")).sum())


# ─────────────────────────────────────────────
# PRICE TREND (EMA)
# ─────────────────────────────────────────────
def plot_price(df):
    plt.figure(figsize=(12, 6))

    plt.plot(df['date'], df['close'], label='Close')
    plt.plot(df['date'], df['ema_9'], label='EMA 9')
    plt.plot(df['date'], df['ema_21'], label='EMA 21')

    plt.title("Price + EMA Trend")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    plt.show()


# ─────────────────────────────────────────────
# CORRELATION HEATMAP
# ─────────────────────────────────────────────
def plot_correlation(df):
    plt.figure(figsize=(12, 8))

    corr = df.corr(numeric_only=True)

    sns.heatmap(corr, cmap='coolwarm')
    plt.title("Correlation Matrix")

    plt.show()


# ─────────────────────────────────────────────
# STATIONARITY TEST (ADF)
# ─────────────────────────────────────────────
def check_stationarity(series, name=""):
    result = adfuller(series)

    print(f"\n{name} ADF Test")
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])

    if result[1] < 0.05:
        print("Stationary ✅")
    else:
        print("Not Stationary ❌")


# ─────────────────────────────────────────────
# MAIN EDA FUNCTION
# ─────────────────────────────────────────────
def run_eda():
    df = load_data()

    if df is None:
        return

    print_basic_info(df)

    # Trend
    plot_price(df)

    # Correlation
    plot_correlation(df)

    # Stationarity
    check_stationarity(df['close'], "Close Price")
    check_stationarity(df['returns'], "Returns")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_eda()