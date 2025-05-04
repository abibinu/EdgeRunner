from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime

class BaseStrategy(ABC):
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize strategy with parameters
        Args:
            params: Dictionary of strategy parameters
        """
        self.params = params or {}
        self.position = 0  # Current position: 1 (long), -1 (short), 0 (neutral)
        self.positions = []  # Track all positions
        self.trades = []  # Track all trades
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the strategy logic
        Args:
            data: DataFrame with OHLCV data
        Returns:
            DataFrame with signals column added
        """
        pass

    def calculate_rsi(self, close_prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using pandas"""
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, close_prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD using pandas"""
        exp1 = close_prices.ewm(span=fast, adjust=False).mean()
        exp2 = close_prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return pd.DataFrame({
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        })

    def calculate_bollinger_bands(self, close_prices: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Bands using pandas"""
        middle = close_prices.rolling(window=period).mean()
        std_dev = close_prices.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return pd.DataFrame({
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower
        })

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        df = data.copy()
        
        # Calculate EMAs
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Calculate Momentum and ROC
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Fill NaN values with 0 or appropriate values
        df = df.fillna(method='bfill')  # Backward fill first
        df = df.fillna(0)  # Fill remaining NaNs with 0
        
        return df

    def calculate_position_size(self, capital: float, risk_per_trade: float, 
                              entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk management rules
        """
        if not stop_loss or not entry_price:
            return 0
            
        risk_amount = capital * (risk_per_trade / 100)
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
            
        position_size = risk_amount / price_risk
        return position_size

    def log_trade(self, timestamp: datetime, side: str, price: float, 
                  quantity: float, reason: str):
        """
        Log trade details for analysis
        """
        trade = {
            'timestamp': timestamp,
            'side': side,
            'price': price,
            'quantity': quantity,
            'reason': reason
        }
        self.trades.append(trade)