import yfinance as yf
import pandas as pd


def fetch_stock_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch stock data using Yahoo Finance (FREE).
    Near real-time (15 min delay).
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError("No data found for symbol")

    df.reset_index(inplace=True)
    return df


if __name__ == "__main__":
    # Manual test
    data = fetch_stock_data("INFY.NS")
    print(data.head())
