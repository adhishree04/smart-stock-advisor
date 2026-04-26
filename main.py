import sys
sys.path.append("03_src")
import os
import pandas as pd
import torch
from stock_transformer import run_pipeline, predict_next_day, cfg

data_folder = "01_data/processed"
all_files = [f for f in os.listdir(data_folder) if f.endswith("_processed.csv")]

for file in all_files:
    stock = file.replace("_processed.csv", "")
    print(f"\nTraining: {stock}")
    try:
        df = pd.read_csv(f"{data_folder}/{file}")
        df.dropna(inplace=True)
        model, scaler_X, scaler_y, preds, actuals = run_pipeline(df)
        next_price = predict_next_day(model, df.tail(90), scaler_X, scaler_y)

        # Save model
        torch.save({
            "model": model.state_dict(),
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "next_price": next_price
        }, f"04_models/{stock}_model.pt")

        print(f"✅ {stock} saved! Next day: ₹{next_price:.2f}")
    except Exception as e:
        print(f"❌ {stock} failed: {e}")