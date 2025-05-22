import pandas as pd
from indicators.sma import sma

# Example strategy: SMA crossover
# Buy when Close crosses above SMA, Sell when Close crosses below SMA

def sma_crossover_strategy(df, sma_period=5):
    df = df.copy()
    df['SMA'] = sma(df, period=sma_period)
    df['Signal'] = 0
    # Buy signal: Close crosses above SMA
    df.loc[(df['Close'] > df['SMA']) & (df['Close'].shift(1) <= df['SMA'].shift(1)), 'Signal'] = 1
    # Sell signal: Close crosses below SMA
    df.loc[(df['Close'] < df['SMA']) & (df['Close'].shift(1) >= df['SMA'].shift(1)), 'Signal'] = -1
    return df[['Date', 'Close', 'SMA', 'Signal']]

if __name__ == "__main__":
    from data.yahoo_fetcher import fetch_yahoo_data
    df = fetch_yahoo_data(
        symbol="RELIANCE.NS",
        interval="1d",
        start="2024-05-01",
        end="2024-05-17"
    )
    result = sma_crossover_strategy(df, sma_period=3)
    print(result)
