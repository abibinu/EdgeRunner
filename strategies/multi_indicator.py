import pandas as pd
from indicators.sma import sma
from indicators.ema_rsi import ema, rsi
from indicators.macd_bbands import macd, bollinger_bands

# Multi-indicator strategy example:
# Buy when:
#   - Close > SMA
#   - EMA > SMA
#   - RSI > 50
#   - MACD > MACD Signal
#   - Close > Bollinger Mid
# Sell when:
#   - Close < SMA
#   - EMA < SMA
#   - RSI < 50
#   - MACD < MACD Signal
#   - Close < Bollinger Mid

def multi_indicator_strategy(df, sma_period=5, ema_period=5, rsi_period=5, macd_fast=12, macd_slow=26, macd_signal=9, bb_period=20, bb_std=2):
    # No-trade zone: require ATR and price range to be above threshold
    price_range = df['Close'].rolling(window=10).max() - df['Close'].rolling(window=10).min()
    price_range_threshold = price_range.median()
    # Relative Volume (current volume / 20-bar average)
    df['Rel_Vol'] = df['Volume'] / (df['Volume'].rolling(window=20).mean() + 1e-9)
    rel_vol_threshold = df['Rel_Vol'].median()
    # ATR (Average True Range)
    df['TR'] = df['Close'].diff().abs()
    df['ATR'] = df['TR'].rolling(window=5).mean()
    # Volume (already handled for filter, but add as feature)
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    df['Vol_MA'] = df['Volume'].rolling(window=5).mean()
    # Momentum (3-day and 5-day returns)
    df['Momentum_3'] = df['Close'].pct_change(3)
    df['Momentum_5'] = df['Close'].pct_change(5)
    df = df.copy().reset_index(drop=True)
    # Use best parameters from grid search for SUZLON
    sma_period = 3
    ema_period = 3
    rsi_period = 3
    macd_fast = 6
    macd_slow = 13
    macd_signal = 5
    bb_period = 10
    bb_std = 2
    df['SMA'] = sma(df, period=sma_period).values
    df['EMA'] = ema(df, period=ema_period).values
    df['RSI'] = rsi(df, period=rsi_period).values
    macd_line, macd_signal_line, macd_hist = macd(df, fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal)
    df['MACD'] = macd_line.values
    df['MACD_Signal'] = macd_signal_line.values
    df['MACD_Hist'] = macd_hist.values
    bb_mid, bb_upper, bb_lower = bollinger_bands(df, period=bb_period, std_dev=bb_std)
    df['BB_Mid'] = bb_mid.values
    df['BB_Upper'] = bb_upper.values
    df['BB_Lower'] = bb_lower.values

    # ATR filter (volatility)
    df['TR'] = df['Close'].diff().abs()
    df['ATR'] = df['TR'].rolling(window=5).mean()
    atr_threshold = df['ATR'].median()  # Only trade if ATR above median

    # Volume filter
    if 'Volume' in df.columns:
        df['Vol_MA'] = df['Volume'].rolling(window=5).mean()
        vol_threshold = df['Vol_MA'].median()
    else:
        df['Vol_MA'] = 1
        vol_threshold = 0

    # ML integration
    from utils.config_loader import load_config
    config = load_config()
    use_ml = config.get('ml', {}).get('use_ml', False)
    # Allow skipping ML prediction for training
    skip_ml = config.get('ml', {}).get('skip_ml', False)
    if use_ml and not skip_ml:
        from ml.model import EdgeRunnerMLModel
        ml_model = EdgeRunnerMLModel()
        features = df[[
            'SMA', 'EMA', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower',
            'ATR', 'Volume', 'Vol_MA', 'Momentum_3', 'Momentum_5', 'Rel_Vol'
        ]].copy()
        features = features.rename(columns={
            'SMA': 'sma_10',
            'EMA': 'ema_10',
            'RSI': 'rsi_14',
            'MACD': 'macd',
            'BB_Upper': 'bb_upper',
            'BB_Lower': 'bb_lower',
            'ATR': 'atr',
            'Volume': 'volume',
            'Vol_MA': 'vol_ma',
            'Momentum_3': 'momentum_3',
            'Momentum_5': 'momentum_5',
            'Rel_Vol': 'rel_vol',
        })
        features = features.fillna(0)
        # Align features with model's expected columns if available
        if hasattr(ml_model, 'feature_names') and ml_model.feature_names is not None:
            missing = [col for col in ml_model.feature_names if col not in features.columns]
            for col in missing:
                features[col] = 0
            features = features[ml_model.feature_names]
        # ML confidence filter: only take trades if model is confident
        if hasattr(ml_model.model, 'predict_proba'):
            proba = ml_model.model.predict_proba(features)
            preds = ml_model.model.predict(features)
            conf = proba.max(axis=1)
            min_conf = 0.6
            preds_filtered = [p if c >= min_conf else 0 for p, c in zip(preds, conf)]
            df['Signal'] = preds_filtered
        else:
            df['Signal'] = ml_model.predict_signal(features)
    else:
        df['Signal'] = 0
        # Buy signal with ATR/volume/rel_vol/no-trade zone filter
        df.loc[
            (df['Close'] > df['SMA']) &
            (df['EMA'] > df['SMA']) &
            (df['RSI'] > 50) &
            (df['MACD'] > df['MACD_Signal']) &
            (df['Close'] > df['BB_Mid']) &
            (df['ATR'] > atr_threshold) &
            (df['Vol_MA'] > vol_threshold) &
            (df['Rel_Vol'] > rel_vol_threshold) &
            (price_range > price_range_threshold),
            'Signal'] = 1
        # Sell signal with ATR/volume/rel_vol/no-trade zone filter
        df.loc[
            (df['Close'] < df['SMA']) &
            (df['EMA'] < df['SMA']) &
            (df['RSI'] < 50) &
            (df['MACD'] < df['MACD_Signal']) &
            (df['Close'] < df['BB_Mid']) &
            (df['ATR'] > atr_threshold) &
            (df['Vol_MA'] > vol_threshold) &
            (df['Rel_Vol'] > rel_vol_threshold) &
            (price_range > price_range_threshold),
            'Signal'] = -1
    return df[['Date', 'Close', 'SMA', 'EMA', 'RSI', 'MACD', 'MACD_Signal', 'BB_Mid', 'BB_Upper', 'BB_Lower', 'ATR', 'Volume', 'Vol_MA', 'Momentum_3', 'Momentum_5', 'Rel_Vol', 'Signal']]

if __name__ == "__main__":
    from data.yahoo_fetcher import fetch_yahoo_data
    df = fetch_yahoo_data(
        symbol="RELIANCE.NS",
        interval="1d",
        start="2024-05-01",
        end="2024-05-17"
    )
    result = multi_indicator_strategy(df)
    print(result)
