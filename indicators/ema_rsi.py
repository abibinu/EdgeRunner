import pandas as pd

def ema(df, period=14, price_col="Close"):
    """
    Calculate Exponential Moving Average (EMA).
    df: DataFrame with price data
    period: window length
    price_col: column to use for price (default: 'Close')
    Returns a pandas Series with the EMA values.
    """
    return df[price_col].ewm(span=period, adjust=False).mean()

def rsi(df, period=14, price_col="Close"):
    """
    Calculate Relative Strength Index (RSI).
    df: DataFrame with price data
    period: window length
    price_col: column to use for price (default: 'Close')
    Returns a pandas Series with the RSI values.
    """
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

if __name__ == "__main__":
    from data.yahoo_fetcher import fetch_yahoo_data
    df = fetch_yahoo_data(
        symbol="RELIANCE.NS",
        interval="1d",
        start="2024-05-01",
        end="2024-05-17"
    )
    df["EMA_5"] = ema(df, period=5)
    df["RSI_5"] = rsi(df, period=5)
    print(df[["Date", "Close", "EMA_5", "RSI_5"]])
