import pandas as pd

# MACD Indicator

def macd(df, fast_period=12, slow_period=26, signal_period=9, price_col="Close"):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    Returns MACD line, Signal line, and Histogram as DataFrame columns.
    """
    exp1 = df[price_col].ewm(span=fast_period, adjust=False).mean()
    exp2 = df[price_col].ewm(span=slow_period, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# Bollinger Bands

def bollinger_bands(df, period=20, std_dev=2, price_col="Close"):
    """
    Calculate Bollinger Bands.
    Returns middle band (SMA), upper band, and lower band as DataFrame columns.
    """
    sma = df[price_col].rolling(window=period, min_periods=1).mean()
    std = df[price_col].rolling(window=period, min_periods=1).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return sma, upper_band, lower_band

if __name__ == "__main__":
    from data.yahoo_fetcher import fetch_yahoo_data
    df = fetch_yahoo_data(
        symbol="RELIANCE.NS",
        interval="1d",
        start="2024-05-01",
        end="2024-05-17"
    )
    # MACD
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df)
    # Bollinger Bands
    df["BB_Mid"], df["BB_Upper"], df["BB_Lower"] = bollinger_bands(df)
    print(df[["Date", "Close", "MACD", "MACD_Signal", "MACD_Hist", "BB_Mid", "BB_Upper", "BB_Lower"]])
