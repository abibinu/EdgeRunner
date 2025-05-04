from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler

class AdaptiveTrendStrategy(BaseStrategy):
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize Adaptive Trend strategy with dynamic parameter adjustment
        """
        default_params = {
            'lookback_period': 20,
            'volatility_window': 20,
            'trend_threshold': 0.01,  # Reduced for more sensitivity
            'stop_loss_atr_mult': 1.5,  # Tighter stop loss
            'profit_target_atr_mult': 2.5,  # More realistic profit target
            'rsi_upper': 70,
            'rsi_lower': 30,
            'min_trend_strength': 0.3  # Minimum trend strength for entry
        }
        super().__init__(params or default_params)
        self.scaler = StandardScaler()
        
    def calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend strength using multiple indicators"""
        df = data.copy()
        
        # Price vs EMAs
        df['short_trend'] = ((df['close'] > df['ema_9']) & 
                           (df['ema_9'].diff() > 0)).astype(int)
        
        df['medium_trend'] = ((df['close'] > df['ema_21']) & 
                            (df['ema_21'].diff() > 0)).astype(int)
        
        df['long_trend'] = ((df['close'] > df['ema_50']) & 
                          (df['ema_50'].diff() > 0)).astype(int)
        
        # Price momentum using multiple timeframes
        df['mom_1d'] = df['close'].pct_change() > 0
        df['mom_5d'] = df['close'].pct_change(5) > 0
        df['price_momentum'] = (df['mom_1d'] & df['mom_5d']).astype(int)
        
        # Enhanced volume analysis
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_trend'] = ((df['volume'] > df['volume_ma']) & 
                             (df['volume'] > df['volume'].shift())).astype(int)
        
        # RSI momentum
        df['rsi_trend'] = ((df['rsi'] > 50) & (df['rsi'].diff() > 0)).astype(int)
        
        # MACD momentum
        df['macd_trend'] = ((df['macd'] > df['macd_signal']) & 
                           (df['macd'].diff() > 0)).astype(int)
        
        # Combine trends with dynamic weights based on volatility
        volatility = df['atr'] / df['close']
        vol_rank = volatility.rank(pct=True)
        
        # Adjust weights based on volatility regime
        short_weight = 0.35 * (1 - vol_rank) + 0.25 * vol_rank
        medium_weight = 0.35 * (1 - vol_rank) + 0.25 * vol_rank
        long_weight = 0.1 + 0.2 * vol_rank
        
        trend_strength = (
            df['short_trend'] * short_weight +
            df['medium_trend'] * medium_weight +
            df['long_trend'] * long_weight +
            df['price_momentum'] * 0.1 +
            df['volume_trend'] * 0.1 +
            df['rsi_trend'] * 0.05 +
            df['macd_trend'] * 0.05
        )
        
        return trend_strength

    def calculate_volatility_regime(self, data: pd.DataFrame) -> pd.Series:
        """Determine market volatility regime"""
        df = data.copy()
        
        # Calculate historical volatility
        returns = df['close'].pct_change()
        hist_vol = returns.rolling(window=self.params['volatility_window']).std() * np.sqrt(252)
        
        # Calculate ATR-based volatility
        atr_vol = df['atr'] / df['close']
        
        # Calculate Bollinger Band width
        bb_width = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Normalize metrics
        hist_vol_norm = (hist_vol - hist_vol.rolling(100).mean()) / hist_vol.rolling(100).std()
        atr_vol_norm = (atr_vol - atr_vol.rolling(100).mean()) / atr_vol.rolling(100).std()
        bb_width_norm = (bb_width - bb_width.rolling(100).mean()) / bb_width.rolling(100).std()
        
        # Combine volatility metrics
        volatility_regime = (hist_vol_norm + atr_vol_norm + bb_width_norm) / 3
        return volatility_regime
        
    def adjust_parameters(self, data: pd.DataFrame, volatility_regime: pd.Series):
        """Dynamically adjust strategy parameters based on market conditions"""
        current_volatility = volatility_regime.iloc[-1]
        
        # Adjust trend threshold based on volatility
        vol_factor = 1 + current_volatility * 0.2  # ±20% adjustment
        self.params['trend_threshold'] = max(0.008, min(0.02, 
            self.params['trend_threshold'] * vol_factor))
        
        # Adjust stop loss and profit targets
        if current_volatility > 1:  # High volatility regime
            self.params['stop_loss_atr_mult'] = max(1.2, 1.5 * vol_factor)
            self.params['profit_target_atr_mult'] = max(1.8, 2.5 * vol_factor)
        else:  # Low volatility regime
            self.params['stop_loss_atr_mult'] = min(2.0, 1.5 / vol_factor)
            self.params['profit_target_atr_mult'] = min(3.0, 2.5 / vol_factor)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using adaptive trend strategy"""
        df = self.add_technical_indicators(data)
        
        # Calculate trend strength and volatility
        df['trend_strength'] = self.calculate_trend_strength(df)
        df['volatility_regime'] = self.calculate_volatility_regime(df)
        
        # Adjust parameters dynamically
        self.adjust_parameters(df, df['volatility_regime'])
        
        # Initialize signals
        df['signal'] = 0
        
        # More responsive trend conditions
        strong_uptrend = (
            (df['trend_strength'] > self.params['min_trend_strength']) &
            (df['close'] > df['ema_21']) &
            (df['macd'] > 0) &
            (df['rsi'] > 40) & (df['rsi'] < 75)  # Wider RSI range
        )
        
        strong_downtrend = (
            (df['trend_strength'] < -self.params['min_trend_strength']) &
            (df['close'] < df['ema_21']) &
            (df['macd'] < 0) &
            (df['rsi'] < 60) & (df['rsi'] > 25)  # Wider RSI range
        )
        
        # Volume confirmation with more flexibility
        volume_confirmation = df['volume'] > df['volume'].rolling(window=10).mean()
        
        # Entry signals
        long_entry = strong_uptrend & volume_confirmation
        short_entry = strong_downtrend & volume_confirmation
        
        # Set entry signals
        df.loc[long_entry, 'signal'] = 1
        df.loc[short_entry, 'signal'] = -1
        
        # Calculate dynamic stop loss and profit targets
        df['stop_loss'] = np.nan
        df['profit_target'] = np.nan
        
        # For long positions
        long_mask = df['signal'] == 1
        df.loc[long_mask, 'stop_loss'] = (
            df.loc[long_mask, 'close'] - 
            df.loc[long_mask, 'atr'] * self.params['stop_loss_atr_mult']
        )
        df.loc[long_mask, 'profit_target'] = (
            df.loc[long_mask, 'close'] + 
            df.loc[long_mask, 'atr'] * self.params['profit_target_atr_mult']
        )
        
        # For short positions
        short_mask = df['signal'] == -1
        df.loc[short_mask, 'stop_loss'] = (
            df.loc[short_mask, 'close'] + 
            df.loc[short_mask, 'atr'] * self.params['stop_loss_atr_mult']
        )
        df.loc[short_mask, 'profit_target'] = (
            df.loc[short_mask, 'close'] - 
            df.loc[short_mask, 'atr'] * self.params['profit_target_atr_mult']
        )
        
        # More forgiving exit conditions
        long_exit = (
            (df['close'] < df['ema_50']) |  # Use longer-term MA for exits
            (df['close'] < df['stop_loss'].shift(1)) |
            (df['close'] > df['profit_target'].shift(1)) |
            (df['trend_strength'] < -0.1)  # Less strict trend reversal
        )
        
        short_exit = (
            (df['close'] > df['ema_50']) |  # Use longer-term MA for exits
            (df['close'] > df['stop_loss'].shift(1)) |
            (df['close'] < df['profit_target'].shift(1)) |
            (df['trend_strength'] > 0.1)  # Less strict trend reversal
        )
        
        # Set exit signals
        df.loc[long_exit & (df['signal'].shift(1) == 1), 'signal'] = 0
        df.loc[short_exit & (df['signal'].shift(1) == -1), 'signal'] = 0
        
        # Debug info
        print(f"Long signals generated: {(df['signal'] == 1).sum()}")
        print(f"Short signals generated: {(df['signal'] == -1).sum()}")
        
        return df