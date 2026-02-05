import yfinance as yf
import pandas as pd


def fetch_stock_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d"
) -> pd.DataFrame | None:
    """
    Fetch stock data using Yahoo Finance (FREE).
    Safe version: never crashes the app.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df is None or df.empty:
            print(f"⚠️ No data returned for {symbol}")
            return None

        df.reset_index(inplace=True)
        return df

    except Exception as e:
        # VERY IMPORTANT: never crash production app
        print(f"⚠️ Yahoo Finance unavailable for {symbol}: {e}")
        return None


if __name__ == "__main__":
    # Manual test (local only)
    data = fetch_stock_data("INFY.NS")
    if data is not None:
        print(data.head())
    else:
        print("Failed to fetch data")
