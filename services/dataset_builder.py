import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from services.data_fetcher import fetch_stock_data
from services.feature_engineering import add_technical_indicators


def build_ml_dataset(symbol: str = "INFY.NS"):
    """
    Prepare ML-ready dataset from stock data.
    SAFE version: never crashes app startup.
    """

    # 1️⃣ Fetch raw stock data
    df = fetch_stock_data(symbol)

    # ❗ IMPORTANT: do NOT crash if market data unavailable
    if df is None:
        return None, None, None, None

    # 2️⃣ Add technical indicators
    df = add_technical_indicators(df)

    # 3️⃣ Target: next-day price movement
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # 4️⃣ Drop NaN rows (from indicators + shift)
    df.dropna(inplace=True)

    # 5️⃣ Feature columns
    features = [
        "sma_20",
        "sma_50",
        "rsi",
        "macd",
        "macd_signal",
        "volatility"
    ]

    X = df[features]
    y = df["target"]

    # 6️⃣ Time-based split (no data leakage)
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = build_ml_dataset()

    if X_train is None:
        print("⚠️ Market data unavailable. Dataset not built.")
    else:
        print("Training data shape:", X_train.shape)
        print("Testing data shape:", X_test.shape)
        print("\nSample training rows:")
        print(X_train.head())
