import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import ta

import pandas as pd
import ta


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to stock price data
    """
    df = df.copy()

    # Moving Averages
    df["sma_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["sma_50"] = ta.trend.sma_indicator(df["Close"], window=50)

    # RSI
    df["rsi"] = ta.momentum.rsi(df["Close"], window=14)

    # MACD
    df["macd"] = ta.trend.macd(df["Close"])
    df["macd_signal"] = ta.trend.macd_signal(df["Close"])

    # Volatility (Risk)
    df["volatility"] = df["Close"].rolling(window=20).std()

    return df


if __name__ == "__main__":
    from services.data_fetcher import fetch_stock_data

    df = fetch_stock_data("INFY.NS")
    df = add_technical_indicators(df)

    print(df.tail())
