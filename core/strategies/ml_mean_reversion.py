from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any

class MLMeanReversionStrategy(BaseStrategy):
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize ML-enhanced mean reversion strategy"""
        default_params = {
            'lookback_period': 20,
            'entry_zscore': 1.0,
            'exit_zscore': 0.3,
            'stop_loss_pct': 1.0,
            'min_volume_ratio': 0.8,
            'ml_threshold': 0.4,
            'price_deviation_threshold': 0.02,
            'trend_window': 5
        }
        super().__init__(params or default_params)
        
        # Initialize model with default configuration
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            min_samples_split=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.ml_features = [
            # Price-based features
            'zscore', 'returns', 'returns_5', 'returns_10', 'returns_std',
            'trend_strength', 'price_deviation', 'bb_deviation',
            
            # Volume-based features
            'volume_ratio', 'volume_trend', 'volume_zscore',
            
            # Technical indicators
            'rsi', 'macd', 'macd_hist', 'macd_signal',
            'bb_upper', 'bb_lower', 'bb_middle',
            'momentum', 'roc',
            
            # Mean reversion indicators
            'mean_reversion_score', 'price_velocity', 'price_acceleration',
            
            # Volatility features
            'volatility', 'volatility_ratio'
        ]

    def prepare_features(self, data: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """Prepare features for ML model with enhanced mean reversion indicators"""
        df = self.add_technical_indicators(data)
        
        # Calculate all technical features
        df = self._calculate_features(df)
        
        # Prepare ML features
        feature_data = df[self.ml_features].copy()
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
        feature_data = feature_data.dropna()
        
        if len(feature_data) > 0:
            if fit_scaler:
                self.scaler.fit(feature_data)
            # Only transform if we have fitted the scaler
            if hasattr(self.scaler, 'mean_'):
                feature_data = pd.DataFrame(
                    self.scaler.transform(feature_data),
                    index=feature_data.index,
                    columns=feature_data.columns
                )
                for col in feature_data.columns:
                    df.loc[feature_data.index, col] = feature_data[col]
        
        return df
        
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical features"""
        # Price trend features
        df['rolling_mean'] = df['close'].rolling(window=self.params['lookback_period']).mean()
        df['rolling_std'] = df['close'].rolling(window=self.params['lookback_period']).std()
        df['zscore'] = (df['close'] - df['rolling_mean']) / df['rolling_std']
        
        # Enhanced momentum features
        df['returns'] = df['close'].pct_change()
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        df['returns_std'] = df['returns'].rolling(window=20).std()
        
        # Short-term trend features
        trend_window = self.params.get('trend_window', 5)
        df['short_trend'] = df['close'].diff(trend_window).rolling(trend_window).mean()
        df['trend_strength'] = df['short_trend'] / df['close']
        
        # Price deviation features
        df['price_deviation'] = (df['close'] - df['rolling_mean']).abs() / df['rolling_mean']
        df['bb_deviation'] = (df['close'] - df['bb_middle']) / (df['bb_upper'] - df['bb_lower'])
        
        # Enhanced volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_trend'] = df['volume'].pct_change(5)
        df['volume_zscore'] = (df['volume'] - df['volume_ma']) / df['volume'].rolling(window=20).std()
        
        # Mean reversion indicators
        df['mean_reversion_score'] = -1 * df['zscore'] * df['volume_ratio']
        df['price_velocity'] = df['returns'].rolling(window=5).mean()
        df['price_acceleration'] = df['price_velocity'].diff()
        
        # Volatility features
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=60).mean()
        
        return df

    def train_model(self, data: pd.DataFrame, target_returns: int = 5):
        """Train the ML model with enhanced feature engineering and balanced sampling"""
        # Prepare features with fitting the scaler
        df = self.prepare_features(data, fit_scaler=True)
        
        # Create target variable with more generous profit threshold
        df['future_return'] = df['close'].shift(-target_returns) / df['close'] - 1
        df['target'] = 0
        
        # More relaxed conditions for identifying opportunities
        long_opp = (
            (df['zscore'] < -self.params['entry_zscore']) & 
            (df['future_return'] > 0.003)  # Reduced from 0.005 to 0.3%
        )
        
        short_opp = (
            (df['zscore'] > self.params['entry_zscore']) & 
            (df['future_return'] < -0.003)  # Reduced from -0.005 to -0.3%
        )
        
        # Set target labels
        df.loc[long_opp | short_opp, 'target'] = 1
        
        # Prepare features for training
        feature_data = df[self.ml_features].copy()
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
        feature_data = feature_data.dropna()
        
        if len(feature_data) > 0:
            targets = df['target'].loc[feature_data.index].fillna(0)
            
            try:
                # Enhanced class weights for better balance
                n_samples = len(targets)
                n_positives = targets.sum()
                
                if n_positives > 0:
                    weight_multiplier = max(2, min(5, n_samples / (2 * n_positives)))
                    class_weights = {
                        0: 1,
                        1: weight_multiplier
                    }
                    
                    # Configure model with balanced class weights
                    self.model = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=4,
                        min_samples_split=5,
                        class_weight=class_weights,
                        random_state=42
                    )
                else:
                    # Use default model configuration for no positive examples
                    class_weights = {0: 1, 1: 1}
                
                # Fit the model
                self.model.fit(feature_data, targets)
                print(f"Model trained successfully on {len(feature_data)} samples")
                print(f"Positive examples: {targets.sum()} ({targets.mean()*100:.1f}%)")
                print(f"Class weights: {class_weights}")
                
                if n_positives > 0:
                    # Print feature importances
                    importances = pd.Series(
                        self.model.feature_importances_,
                        index=self.ml_features
                    ).sort_values(ascending=False)
                    print("\nTop 5 important features:")
                    print(importances.head())
                
            except Exception as e:
                print(f"Error training model: {str(e)}")
                print("Using default model configuration")
                # Ensure the model is fitted even with errors
                self.model.fit(feature_data, targets)
                
        else:
            print("No valid features for training")
            # Create dummy features and fit model to prevent errors
            dummy_features = np.zeros((1, len(self.ml_features)))
            dummy_targets = np.zeros(1)
            self.model.fit(dummy_features, dummy_targets)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals with enhanced mean reversion criteria"""
        df = self.prepare_features(data, fit_scaler=False)
        
        feature_data = df[self.ml_features].copy()
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
        feature_data = feature_data.dropna()
        
        # Initialize signals
        df['signal'] = 0
        df['ml_signal'] = 0.5
        
        if len(feature_data) > 0:
            try:
                ml_predictions = self.model.predict_proba(feature_data)[:, 1]
                df.loc[feature_data.index, 'ml_signal'] = ml_predictions
                
                print(f"ML predictions range: {ml_predictions.min():.2f} to {ml_predictions.max():.2f}")
                print(f"Number of high confidence predictions (>{self.params['ml_threshold']}): "
                      f"{(ml_predictions > self.params['ml_threshold']).sum()}")
                
            except Exception as e:
                print(f"Error generating predictions: {str(e)}")
                # Keep default ml_signal value of 0.5
        
        # More relaxed entry conditions
        long_condition = (
            (df['zscore'] < -self.params['entry_zscore']) &
            (df['ml_signal'] > self.params['ml_threshold']) &
            (df['volume_ratio'] > self.params['min_volume_ratio']) &
            (df['mean_reversion_score'] > 0)
        )
        
        short_condition = (
            (df['zscore'] > self.params['entry_zscore']) &
            (df['ml_signal'] > self.params['ml_threshold']) &
            (df['volume_ratio'] > self.params['min_volume_ratio']) &
            (df['mean_reversion_score'] < 0)
        )
        
        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1
        
        # Dynamic stop loss
        stop_loss_pct = self.params['stop_loss_pct'] * (1 + df['volatility_ratio'])
        
        # Exit conditions
        exit_long_condition = (
            (df['zscore'] > -self.params['exit_zscore']) |  # More generous exit
            (df['returns'] < -stop_loss_pct / 100) |
            (df['mean_reversion_score'] < 0)
        )
        
        exit_short_condition = (
            (df['zscore'] < self.params['exit_zscore']) |  # More generous exit
            (df['returns'] > stop_loss_pct / 100) |
            (df['mean_reversion_score'] > 0)
        )
        
        df.loc[exit_long_condition & (df['signal'].shift(1) == 1), 'signal'] = 0
        df.loc[exit_short_condition & (df['signal'].shift(1) == -1), 'signal'] = 0
        
        print(f"Long signals generated: {(df['signal'] == 1).sum()}")
        print(f"Short signals generated: {(df['signal'] == -1).sum()}")
        
        return df