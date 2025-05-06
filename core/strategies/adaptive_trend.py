import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy

class AdaptiveTrendStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__()
        self.params = params
    
    def calculate_indicators(self, data):
        """Calculate technical indicators for trend analysis"""
        indicators = {}
        
        # ATR calculation
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        indicators['atr'] = true_range.rolling(window=self.params['volatility_window']).mean()
        
        # EMAs for multiple timeframes
        indicators['ema_short'] = data['close'].ewm(span=self.params['lookback_period'], adjust=False).mean()
        indicators['ema_medium'] = data['close'].ewm(span=self.params['lookback_period'] * 2, adjust=False).mean()
        indicators['ema_long'] = data['close'].ewm(span=self.params['lookback_period'] * 3, adjust=False).mean()
        
        # RSI calculation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # On-Balance Volume (OBV)
        obv = pd.Series(0, index=data.index)
        obv.iloc[0] = data['volume'].iloc[0]
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
            elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        indicators['obv'] = obv
        indicators['obv_ma'] = obv.rolling(window=self.params['lookback_period']).mean()
        
        # MACD calculation
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        indicators['macd'] = exp1 - exp2
        indicators['signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
        
        # ADX calculation
        plus_dm = data['high'].diff()
        minus_dm = -data['low'].diff()
        plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
        tr = pd.DataFrame([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift()).abs(),
            (data['low'] - data['close'].shift()).abs()
        ]).max()
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / tr.rolling(14).mean())
        minus_di = 100 * (minus_dm.rolling(14).mean() / tr.rolling(14).mean())
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        indicators['adx'] = dx.rolling(window=14).mean()
        
        return indicators
    
    def calculate_trend_strength(self, indicators):
        """Calculate overall trend strength using multiple indicators"""
        # EMA alignment score
        ema_score = (
            (indicators['ema_short'] > indicators['ema_medium']).astype(int) +
            (indicators['ema_medium'] > indicators['ema_long']).astype(int)
        ) / 2
        
        # Volume trend confirmation
        volume_trend = (indicators['obv'] > indicators['obv_ma']).astype(int)
        
        # MACD trend
        macd_trend = (indicators['macd'] > indicators['signal']).astype(int)
        
        # Normalize ADX to 0-1 range
        adx_norm = indicators['adx'] / 100.0
        
        # Combined trend strength
        trend_strength = (ema_score * 0.4 + 
                         volume_trend * 0.2 + 
                         macd_trend * 0.2 + 
                         adx_norm * 0.2)
        
        return trend_strength
    
    def generate_signals(self, data):
        """Generate trading signals based on adaptive trend analysis"""
        signals = pd.Series(0, index=data.index)
        indicators = self.calculate_indicators(data)
        
        # Calculate trend strength
        trend_strength = self.calculate_trend_strength(indicators)
        
        # Previous positions for exit conditions
        prev_position = signals.shift(1).fillna(0)
        
        # Entry conditions
        long_condition = (
            (trend_strength > self.params['min_trend_strength']) &  # Strong uptrend
            (indicators['rsi'] < self.params['rsi_upper']) &  # Not overbought
            (data['close'].pct_change() > self.params['trend_threshold'])  # Momentum confirmation
        )
        
        short_condition = (
            (trend_strength < -self.params['min_trend_strength']) &  # Strong downtrend
            (indicators['rsi'] > self.params['rsi_lower']) &  # Not oversold
            (data['close'].pct_change() < -self.params['trend_threshold'])  # Momentum confirmation
        )
        
        # Exit conditions
        exit_long = (
            (trend_strength < 0) |  # Trend reversal
            (indicators['rsi'] > self.params['rsi_upper']) |  # Overbought
            (self.calculate_drawdown(data) < -indicators['atr'] * self.params['stop_loss_atr_mult'])  # Stop loss
        )
        
        exit_short = (
            (trend_strength > 0) |  # Trend reversal
            (indicators['rsi'] < self.params['rsi_lower']) |  # Oversold
            (self.calculate_drawdown(data) < -indicators['atr'] * self.params['stop_loss_atr_mult'])  # Stop loss
        )
        
        # Apply signals with profit targets
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        # Exit conditions
        signals[(prev_position == 1) & exit_long] = 0
        signals[(prev_position == -1) & exit_short] = 0
        
        # Take profit at target
        long_profit_target = data['close'] > (data['close'].shift(1) + 
                                            indicators['atr'] * self.params['profit_target_atr_mult'])
        short_profit_target = data['close'] < (data['close'].shift(1) - 
                                             indicators['atr'] * self.params['profit_target_atr_mult'])
        
        signals[(prev_position == 1) & long_profit_target] = 0
        signals[(prev_position == -1) & short_profit_target] = 0
        
        return signals
    
    @staticmethod
    def calculate_drawdown(data):
        """Calculate drawdown for dynamic stop loss"""
        high_water_mark = data['close'].expanding().max()
        drawdown = (data['close'] - high_water_mark) / high_water_mark
        return drawdown