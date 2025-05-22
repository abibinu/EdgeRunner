import yfinance as yf
import pandas as pd

def fetch_yahoo_data(symbol, interval="1d", start=None, end=None):
    """
    Fetch historical data from Yahoo Finance.
    interval: '1m', '5m', '15m', '1h', '1d', etc.
    start, end: 'YYYY-MM-DD'
    """
    df = yf.download(symbol, interval=interval, start=start, end=end, progress=False, auto_adjust=True)
    if not df.empty:
        # If columns are MultiIndex, flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df.reset_index(inplace=True)
    return df

if __name__ == "__main__":
    # Example: Fetch daily data for RELIANCE.NS
    df = fetch_yahoo_data(
        symbol="RELIANCE.NS",
        interval="1d",
        start="2024-05-10",
        end="2024-05-17"
    )
    print(df.head())
