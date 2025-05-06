import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from .base_strategy import BaseStrategy

class MLMeanReversionStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.model = RandomForestClassifier(
            n_estimators=150,  # Increased for better model stability
            max_depth=4,       # Reduced to prevent overfitting
            min_samples_leaf=5,  # Increased for more robust predictions
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.stock_specific_params = {
            'INFY': {
                'entry_zscore': 2.2,     # More extreme for INFY
                'exit_zscore': 0.6,
                'min_volume_ratio': 0.85
            },
            'TCS': {
                'entry_zscore': 1.8,     # Less extreme for TCS
                'exit_zscore': 0.4,
                'min_volume_ratio': 0.75
            },
            'RELIANCE': {
                'entry_zscore': 2.0,     # Balanced for Reliance
                'exit_zscore': 0.5,
                'min_volume_ratio': 0.8
            }
        }
    
    def adjust_params_for_stock(self, stock_name):
        """Adjust parameters based on stock characteristics"""
        if stock_name in self.stock_specific_params:
            self.params.update(self.stock_specific_params[stock_name])
    
    def calculate_features(self, data):
        """Calculate technical features for ML model"""
        df = pd.DataFrame()
        
        # Price-based features
        df['returns'] = data['close'].pct_change()
        df['log_returns'] = np.log(data['close']).diff()
        
        # Volatility features
        df['rolling_std'] = data['close'].rolling(window=self.params['lookback_period']).std()
        
        # Calculate ATR manually instead of using pandas-ta
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=self.params['lookback_period']).mean()
        
        # Mean reversion features
        rolling_mean = data['close'].rolling(window=self.params['lookback_period']).mean()
        rolling_std = data['close'].rolling(window=self.params['lookback_period']).std()
        df['zscore'] = (data['close'] - rolling_mean) / rolling_std
        
        # Volume features
        df['volume_ratio'] = (data['volume'] / 
                            data['volume'].rolling(window=self.params['lookback_period']).mean())
        
        # Calculate RSI manually
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Money Flow Index (MFI)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        raw_money_flow = typical_price * data['volume']
        
        money_flow_pos = pd.Series(np.where(typical_price > typical_price.shift(1), raw_money_flow, 0))
        money_flow_neg = pd.Series(np.where(typical_price < typical_price.shift(1), raw_money_flow, 0))
        
        mf_pos_sum = money_flow_pos.rolling(window=14).sum()
        mf_neg_sum = money_flow_neg.rolling(window=14).sum()
        
        money_ratio = mf_pos_sum / mf_neg_sum
        df['mfi'] = 100 - (100 / (1 + money_ratio))
        
        # Calculate MACD manually
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Calculate ADX
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
        df['adx'] = dx.rolling(window=14).mean()
        
        # Calculate Bollinger Bands
        middle = data['close'].rolling(window=20).mean()
        std_dev = data['close'].rolling(window=20).std()
        df['bb_upper'] = middle + (std_dev * 2)
        df['bb_middle'] = middle
        df['bb_lower'] = middle - (std_dev * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df.fillna(0)
    
    def generate_labels(self, data, lookahead=5):
        """Generate binary labels for ML training"""
        future_returns = data['close'].shift(-lookahead).div(data['close']) - 1
        # Using dynamic thresholds based on volatility
        rolling_std = data['close'].pct_change().rolling(window=self.params['lookback_period']).std()
        upper_threshold = rolling_std * 0.5
        lower_threshold = -rolling_std * 0.5
        
        # Create three classes: -1 (down), 0 (neutral), 1 (up)
        labels = pd.Series(0, index=data.index)
        labels[future_returns > upper_threshold] = 1
        labels[future_returns < lower_threshold] = -1
        return labels
    
    def train_model(self, data):
        """Train the ML model"""
        features = self.calculate_features(data)
        labels = self.generate_labels(data)
        
        # Remove any remaining NaN values
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_idx]
        labels = labels[valid_idx]
        
        if len(features) > 0 and len(np.unique(labels)) > 1:
            # Scale features
            self.scaler.fit(features)
            scaled_features = self.scaler.transform(features)
            
            # Train model
            self.model.fit(scaled_features, labels)
            self.is_trained = True
            print(f"Model trained successfully with {len(np.unique(labels))} unique classes")
        else:
            print("Warning: Insufficient data variation for training")
            self.is_trained = False
    
    def generate_signals(self, data):
        """Generate trading signals"""
        if not self.is_trained:
            return pd.Series(0, index=data.index)
        
        features = self.calculate_features(data)
        if len(features) == 0:
            return pd.Series(0, index=data.index)
        
        # Scale features and predict
        scaled_features = self.scaler.transform(features)
        try:
            # Try to get probability predictions
            pred_proba = self.model.predict_proba(scaled_features)
            if pred_proba.shape[1] > 1:
                predictions = pred_proba[:, 1]  # Probability of positive class
            else:
                predictions = self.model.predict(scaled_features)
        except (AttributeError, IndexError):
            predictions = self.model.predict(scaled_features)
        
        # Generate signals based on predictions and mean reversion conditions
        signals = pd.Series(0, index=data.index)
        
        # Calculate zscore with dynamic lookback
        volatility = data['close'].pct_change().rolling(window=20).std()
        lookback = max(5, min(20, int(1 / volatility.mean() if not np.isnan(volatility.mean()) else 10)))
        
        zscore = (data['close'] - data['close'].rolling(window=lookback).mean()) / \
                data['close'].rolling(window=lookback).std()
        
        volume_ratio = data['volume'] / data['volume'].rolling(window=self.params['lookback_period']).mean()
        
        # Convert predictions to signal strength (0 to 1)
        if isinstance(predictions[0], (int, np.integer)):
            signal_strength = (predictions + 1) / 2
        else:
            signal_strength = predictions
        
        # Dynamic thresholds based on recent volatility
        vol_scale = volatility / volatility.rolling(window=50).mean()
        vol_scale = vol_scale.fillna(1)
        
        # Adjust entry threshold based on volatility
        entry_zscore = self.params['entry_zscore'] * np.sqrt(vol_scale)
        exit_zscore = self.params['exit_zscore'] * np.sqrt(vol_scale)
        
        # Long signals with dynamic thresholds
        long_condition = (
            (zscore < -entry_zscore) &  # Price is below mean
            (signal_strength > self.params['ml_threshold']) &  # ML model predicts uptick
            (volume_ratio > self.params['min_volume_ratio'])  # Sufficient volume
        )
        
        # Short signals with dynamic thresholds
        short_condition = (
            (zscore > entry_zscore) &  # Price is above mean
            (signal_strength < (1 - self.params['ml_threshold'])) &  # ML model predicts downtick
            (volume_ratio > self.params['min_volume_ratio'])  # Sufficient volume
        )
        
        # Exit conditions with dynamic thresholds
        exit_long_condition = (
            (zscore > -exit_zscore) |  # Price reverted to mean
            (self.calculate_drawdown(data) < -self.params['stop_loss_pct']/100)  # Stop loss
        )
        
        exit_short_condition = (
            (zscore < exit_zscore) |  # Price reverted to mean
            (self.calculate_drawdown(data) < -self.params['stop_loss_pct']/100)  # Stop loss
        )
        
        # Apply signals with confirmation
        signals[long_condition] = 1
        signals[short_condition] = -1
        signals[exit_long_condition & (signals.shift(1) == 1)] = 0
        signals[exit_short_condition & (signals.shift(1) == -1)] = 0
        
        # Add trend filter
        trend = data['close'].rolling(window=self.params['trend_window']).mean()
        signals[signals == 1] = np.where(data['close'][signals == 1] > trend[signals == 1], 1, 0)
        signals[signals == -1] = np.where(data['close'][signals == -1] < trend[signals == -1], -1, 0)
        
        return signals
    
    @staticmethod
    def calculate_drawdown(data):
        """Calculate drawdown for stop loss"""
        high_water_mark = data['close'].expanding().max()
        drawdown = (data['close'] - high_water_mark) / high_water_mark
        return drawdown