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
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2e3450;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #00d4aa; }
    .metric-label { font-size: 0.85rem; color: #8890a4; margin-top: 4px; }
    .predict-box {
        background: linear-gradient(135deg, #00d4aa22, #0066ff22);
        border: 1px solid #00d4aa55;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin: 16px 0;
    }
    .predict-price { font-size: 3rem; font-weig
