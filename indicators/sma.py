import pandas as pd

def sma(df, period=14, price_col="Close"):
    """
    Calculate Simple Moving Average (SMA).
    df: DataFrame with price data
    period: window length
    price_col: column to use for price (default: 'Close')
    Returns a pandas Series with the SMA values.
    """
    return df[price_col].rolling(window=period, min_periods=1).mean()

if __name__ == "__main__":
    # Example usage with Yahoo Finance data
    from data.yahoo_fetcher import fetch_yahoo_data
    df = fetch_yahoo_data(
        symbol="RELIANCE.NS",
        interval="1d",
        start="2024-05-10",
        end="2024-05-17"
    )
    df["SMA_3"] = sma(df, period=3)
    print(df[["Date", "Close", "SMA_3"]])
