from itertools import product
import pandas as pd
import numpy as np
from ml.model import EdgeRunnerMLModel

def test_ml_model():
    # Use real historical data and indicators for model training
    from data.yahoo_fetcher import fetch_yahoo_data
    from strategies.multi_indicator import multi_indicator_strategy
    import os
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # Download and combine historical data for SUZLON (no no-trade zone filter)
    symbols = ["SUZLON.NS"]

    # Reduced grid for fast iteration (expand for final run)
    sma_periods = [5, 10]
    ema_periods = [5, 10]
    rsi_periods = [7, 14]
    macd_fasts = [6, 12]
    macd_slows = [13, 26]
    macd_signals = [5, 9]

    # Additional features
    def add_more_features(df):
        # Ensure required columns exist
        for col in ['High', 'Low', 'Open', 'Close']:
            if col not in df.columns:
                df[col] = 0
        # Price-based features
        df['High_Low'] = df['High'] - df['Low']
        df['Close_Open'] = df['Close'] - df['Open']
        df['Return_1'] = df['Close'].pct_change(1)
        df['Return_3'] = df['Close'].pct_change(3)
        df['Volatility_5'] = df['Close'].rolling(5).std()
        df['Volatility_10'] = df['Close'].rolling(10).std()
        df['Rolling_Max_5'] = df['Close'].rolling(5).max()
        df['Rolling_Min_5'] = df['Close'].rolling(5).min()
        df['Rolling_Max_10'] = df['Close'].rolling(10).max()
        df['Rolling_Min_10'] = df['Close'].rolling(10).min()
        # Avoid division by zero
        df['Range_Pct'] = (df['High'] - df['Low']) / df['Open'].replace(0, np.nan)
        return df

    # Use only one ML algorithm best suited for this task: XGBoost (if available), else fallback to RandomForest
    try:
        from xgboost import XGBClassifier
        try_algos = [
            ('XGBoost', XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss'))
        ]
    except ImportError:
        try_algos = [
            ('RandomForest', RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'))
        ]

    for algo_name, algo in try_algos:
        print(f"\n[ML] Trying algorithm: {algo_name}")
        best_f1 = 0
        best_params = None
        best_clf = None
        best_report = None
        best_features = None
        best_labels = None
        total_combos = len(sma_periods) * len(ema_periods) * len(rsi_periods) * len(macd_fasts) * len(macd_slows) * len(macd_signals)
        combo_idx = 0
        for sma_p, ema_p, rsi_p, macd_f, macd_s, macd_sig in product(sma_periods, ema_periods, rsi_periods, macd_fasts, macd_slows, macd_signals):
            combo_idx += 1
            print(f"[GridSearch] Combo {combo_idx}/{total_combos}: SMA={sma_p}, EMA={ema_p}, RSI={rsi_p}, MACD=({macd_f},{macd_s},{macd_sig})")
            all_features = []
            all_labels = []
            for symbol in symbols:
                df = fetch_yahoo_data(
                    symbol=symbol,
                    interval="1d",
                    start="2022-01-01",
                    end="2024-01-01"
                )
                strat_df = multi_indicator_strategy(
                    df,
                    sma_period=sma_p,
                    ema_period=ema_p,
                    rsi_period=rsi_p,
                    macd_fast=macd_f,
                    macd_slow=macd_s,
                    macd_signal=macd_sig
                )
                strat_df = add_more_features(strat_df)
                required_cols = [
                    'SMA', 'EMA', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower',
                    'ATR', 'Volume', 'Vol_MA', 'Momentum_3', 'Momentum_5', 'Rel_Vol',
                    'High_Low', 'Close_Open', 'Return_1', 'Return_3', 'Volatility_5', 'Volatility_10',
                    'Rolling_Max_5', 'Rolling_Min_5', 'Rolling_Max_10', 'Rolling_Min_10', 'Range_Pct'
                ]
                for col in required_cols:
                    if col not in strat_df.columns:
                        strat_df[col] = 0
                features = strat_df[required_cols].copy()
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
                # Target: next day's return (profitability)
                future_close = strat_df['Close'].shift(-1)
                returns = (future_close - strat_df['Close']) / strat_df['Close']
                y = returns.apply(lambda x: 1 if x > 0 else -1)
                features = features.iloc[:-1]
                y = y.iloc[:-1]
                all_features.append(features)
                all_labels.append(y)
            features = pd.concat(all_features, ignore_index=True)
            y = pd.concat(all_labels, ignore_index=True)
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42, stratify=y)
            clf = algo
            # XGBoost expects labels 0/1, not -1/1
            if algo_name == 'XGBoost':
                y_train_fit = y_train.replace(-1, 0)
                y_test_fit = y_test.replace(-1, 0)
                clf.fit(X_train, y_train_fit)
                y_pred = clf.predict(X_test)
                y_pred = pd.Series(y_pred).replace(0, -1)  # Map back for reporting
                report = classification_report(y_test, y_pred, output_dict=True)
            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)
            f1 = report['1']['f1-score'] if '1' in report else 0
            if f1 > best_f1:
                best_f1 = f1
                best_params = (sma_p, ema_p, rsi_p, macd_f, macd_s, macd_sig)
                best_clf = clf
                best_report = report
                best_features = features
                best_labels = y
        print(f"Best F1 (buy) for {algo_name}: {best_f1:.3f} with params: {best_params}")
        print(best_report)
        # Save the best parameters and report
        with open(f'models/edgerunner_best_params_{algo_name}.txt', 'w') as f:
            f.write(f"Best Params: {best_params}\n")
            f.write(f"Best F1: {best_f1:.3f}\n")
            f.write(str(best_report))
        # Retrain best model on all data with best params
        if best_features is not None and best_labels is not None:
            # XGBoost expects 0/1 labels
            if algo_name == 'XGBoost':
                best_labels_fit = best_labels.replace(-1, 0)
                best_clf.fit(best_features, best_labels_fit)
            else:
                best_clf.fit(best_features, best_labels)
            model_path = f'models/edgerunner_model_{algo_name}.pkl'
            os.makedirs('models', exist_ok=True)
            # Save both model and feature columns
            joblib.dump({'model': best_clf, 'feature_names': list(best_features.columns)}, model_path)
            # Test ML model loading and prediction (only for RandomForest, for now)
            if algo_name == 'RandomForest':
                ml_model = EdgeRunnerMLModel()
                preds = ml_model.predict_signal(best_features)
                print('ML Model Predictions (first 10):', preds[:10])

if __name__ == '__main__':
    test_ml_model()
