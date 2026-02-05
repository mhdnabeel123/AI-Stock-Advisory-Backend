import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from services.data_fetcher import fetch_stock_data
from services.feature_engineering import add_technical_indicators


def build_ml_dataset(symbol: str = "INFY.NS"):
    """
    Prepare ML-ready dataset from stock data
    """

    # Fetch and engineer features
    df = fetch_stock_data(symbol)
    df = add_technical_indicators(df)

    # Target: next day price direction
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Drop rows with NaN values (due to indicators)
    df.dropna(inplace=True)

    # Feature set
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

    # Time-based train-test split (NO random split)
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = build_ml_dataset()

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    print("\nSample training rows:")
    print(X_train.head())
