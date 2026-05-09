import sys
import os
sys.path.append("03_src")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stock_transformer import run_pipeline, predict_next_day, cfg

# ── Page Config ─────────────────────────────
st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2e3450;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4aa;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8890a4;
        margin-top: 4px;
    }
    .predict-box {
        background: linear-gradient(135deg, #00d4aa22, #0066ff22);
        border: 1px solid #00d4aa55;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin: 16px 0;
    }
    .predict-price {
        font-size: 3rem;
        font-weight: 700;
        color: #00d4aa;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa, #0066ff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────
st.markdown("## 📈 NSE Stock Price Predictor")
st.markdown("Transformer-based next-day close price prediction")
st.divider()

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    data_folder = "01_data/processed"
    all_files = sorted([f.replace("_processed.csv", "")
                        for f in os.listdir(data_folder)
                        if f.endswith("_processed.csv")])

    selected_stock = st.selectbox("Select Stock", all_files, index=all_files.index("RELIANCE_NS") if "RELIANCE_NS" in all_files else 0)

    st.markdown("---")
    st.markdown("### 🔧 Model Config")
    st.markdown(f"- Seq Length: `{cfg.seq_len}`")
    st.markdown(f"- Epochs: `{cfg.epochs}`")
    st.markdown(f"- d_model: `{cfg.d_model}`")
    st.markdown(f"- Layers: `{cfg.n_layers}`")
    st.markdown(f"- Device: `{cfg.device}`")

    st.markdown("---")
    run_btn = st.button("🚀 Train & Predict")

# ── Main Area ────────────────────────────────
if run_btn:
    with st.spinner(f"Loading {selected_stock} data..."):
        df = pd.read_csv(f"{data_folder}/{selected_stock}_processed.csv")
        df.dropna(inplace=True)

    st.success(f"✅ Loaded {selected_stock} — {len(df)} rows")

    with st.spinner("Training transformer model... (this may take a few minutes)"):
        model, scaler_X, scaler_y, preds, actuals = run_pipeline(df)

    with st.spinner("Predicting next day..."):
        next_price = predict_next_day(model, df.tail(90), scaler_X, scaler_y)

    # ── Metrics ──────────────────────────────
    mae  = float(np.mean(np.abs(preds - actuals)))
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    mape = float(np.mean(np.abs((actuals - preds) / (actuals + 1e-9))) * 100)

    # ── Next Day Prediction Box ───────────────
    st.markdown(f"""
    <div class="predict-box">
        <div style="color:#8890a4; font-size:1rem; margin-bottom:8px;">📅 Next Day Predicted Close</div>
        <div class="predict-price">₹{next_price:,.2f}</div>
        <div style="color:#8890a4; font-size:0.85rem; margin-top:8px;">{selected_stock}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric Cards ─────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">₹{mae:.2f}</div>
            <div class="metric-label">Mean Absolute Error</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">₹{rmse:.2f}</div>
            <div class="metric-label">Root Mean Squared Error</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{mape:.2f}%</div>
            <div class="metric-label">Mean Absolute % Error</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Plots ─────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📈 Predicted vs Actual", "📊 Error Chart", "🔵 Scatter Plot"])

    plt.style.use("dark_background")

    with tab1:
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")
        ax.plot(actuals, label="Actual", color="#4fc3f7", linewidth=1.5)
        ax.plot(preds,   label="Predicted", color="#00d4aa", linewidth=1.5, linestyle="--")
        ax.set_title(f"{selected_stock} — Predicted vs Actual", color="white")
        ax.set_xlabel("Days (Validation Set)", color="#8890a4")
        ax.set_ylabel("Price (₹)", color="#8890a4")
        ax.legend()
        ax.tick_params(colors="#8890a4")
        ax.spines[:].set_color("#2e3450")
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")
        errors = preds.flatten() - actuals.flatten()
        colors = ["#00d4aa" if e >= 0 else "#ff5370" for e in errors]
        ax.bar(range(len(errors)), errors, color=colors, alpha=0.8)
        ax.axhline(0, color="white", linewidth=0.8)
        ax.set_title("Prediction Error", color="white")
        ax.set_xlabel("Days", color="#8890a4")
        ax.set_ylabel("Error (₹)", color="#8890a4")
        ax.tick_params(colors="#8890a4")
        ax.spines[:].set_color("#2e3450")
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")
        ax.scatter(actuals, preds, alpha=0.5, color="#9c6eff", s=15)
        mn = min(actuals.min(), preds.min())
        mx = max(actuals.max(), preds.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect Prediction")
        ax.set_title("Actual vs Predicted", color="white")
        ax.set_xlabel("Actual (₹)", color="#8890a4")
        ax.set_ylabel("Predicted (₹)", color="#8890a4")
        ax.legend()
        ax.tick_params(colors="#8890a4")
        ax.spines[:].set_color("#2e3450")
        st.pyplot(fig)

    # ── Save results ──────────────────────────
    os.makedirs("05_results", exist_ok=True)
    pd.DataFrame([{
        "stock": selected_stock,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 2),
        "next_day_prediction": round(next_price, 2)
    }]).to_csv(f"05_results/{selected_stock}_metrics.csv", index=False)

    st.success(f"✅ Results saved to 05_results/{selected_stock}_metrics.csv")

else:
    st.info("👈 Select a stock from the sidebar and click **Train & Predict** to start!")
    st.markdown("""
    ### How it works:
    1. Select any NSE stock from the dropdown
    2. Click **Train & Predict**
    3. Model trains on historical OHLCV + technical indicators
    4. See predicted vs actual chart + next day price
    """)
