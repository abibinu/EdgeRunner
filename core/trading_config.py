"""
Centralized configuration for all trading strategies and parameters.
"""

# General trading configuration
INITIAL_CAPITAL = 100000
COMMISSION_RATE = 0.0003  # 0.03% commission
BACKTEST_PERIOD_DAYS = 180  # 6 months of data
TRAIN_TEST_SPLIT = 0.7  # 70% training data, 30% testing data

# Stock universe configuration
STOCK_UNIVERSE = {
    'INFY': 'NSE_EQ|INE009A01021',    # Infosys
    'TCS': 'NSE_EQ|INE467B01029',     # TCS
    'RELIANCE': 'NSE_EQ|INE002A01018'  # Reliance
}

# ML Mean Reversion Strategy Parameters
ML_MEAN_REVERSION_PARAMS = {
    'lookback_period': 12,        # Balanced lookback for better signal quality
    'entry_zscore': 2.0,         # More extreme deviation required for entry
    'exit_zscore': 0.5,          # Quick profit taking
    'stop_loss_pct': 0.4,        # Tight stop loss
    'min_volume_ratio': 0.8,     # Higher volume requirement for confirmation
    'ml_threshold': 0.6,         # More conservative signal threshold
    'price_deviation_threshold': 0.6,  # Stronger price movement required
    'trend_window': 8            # Short window for trend context
}

# Adaptive Trend Strategy Parameters
ADAPTIVE_TREND_PARAMS = {
    'lookback_period': 8,        # Very short lookback
    'volatility_window': 10,     # Quick volatility adaptation
    'trend_threshold': 0.006,    # Highly sensitive to trends
    'stop_loss_atr_mult': 1.8,   # Tighter stops
    'profit_target_atr_mult': 1.5, # Quick profits
    'rsi_upper': 65,            # More aggressive overbought level
    'rsi_lower': 35,            # More aggressive oversold level
    'min_trend_strength': 0.10   # Minimal trend requirement
}

# Risk Management Parameters
RISK_MANAGEMENT = {
    'max_position_size': 0.08,      # Very conservative sizing
    'max_drawdown_pct': 0.010,      # Ultra-tight drawdown control
    'daily_drawdown_limit': 0.010,  # Ultra-tight daily limit
    'max_open_positions': 8,        # Maximum diversification
    'correlation_threshold': 0.35,   # Very strict correlation control
    'risk_per_trade': 0.006        # Minimal risk per trade
}