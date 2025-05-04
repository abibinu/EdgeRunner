import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from core.strategies.ml_mean_reversion import MLMeanReversionStrategy
from core.strategies.adaptive_trend import AdaptiveTrendStrategy
from core.backtest.strategy_evaluator import StrategyEvaluator
from core.upstox_client import get_historical_candles

def test_strategies():
    # Test configuration
    test_stocks = {
        'HDFC': 'NSE_EQ|INE040A01034',   # HDFC Bank
        'INFY': 'NSE_EQ|INE009A01021',    # Infosys
        'SBIN': 'NSE_EQ|INE062A01020'     # State Bank of India
    }
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    # Initialize strategies with custom parameters
    ml_strategy = MLMeanReversionStrategy({
        'lookback_period': 20,
        'entry_zscore': 2.0,
        'exit_zscore': 0.5,
        'stop_loss_pct': 2.0
    })
    
    adaptive_strategy = AdaptiveTrendStrategy({
        'lookback_period': 20,
        'volatility_window': 20,
        'trend_threshold': 0.02,
        'stop_loss_atr_mult': 2.0,
        'profit_target_atr_mult': 3.0
    })
    
    # Initialize evaluator
    evaluator = StrategyEvaluator(
        initial_capital=100000,
        commission=0.0003  # 0.03% commission
    )
    
    for stock_name, instrument_key in test_stocks.items():
        print(f"\nTesting strategies on {stock_name}...")
        
        # Fetch historical data
        data = get_historical_candles(
            instrument_key=instrument_key,
            interval='day',
            to_date_str=end_date.strftime('%Y-%m-%d'),
            from_date_str=start_date.strftime('%Y-%m-%d')
        )
        
        if data.empty:
            print(f"No data available for {stock_name}")
            continue
        
        # Split data for ML strategy training
        train_size = int(len(data) * 0.7)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # Train ML strategy
        print(f"Training ML strategy on {len(train_data)} days of data...")
        ml_strategy.train_model(train_data)
        
        # Evaluate both strategies
        strategies = [ml_strategy, adaptive_strategy]
        results = evaluator.evaluate_strategies(strategies, test_data)
        
        # Generate and save performance report
        performance_report = evaluator.generate_performance_report(results)
        print(f"\n{stock_name} Performance Report:")
        print(performance_report)
        print("\nStrategy Comparison:")
        print("1. ML Mean Reversion Strategy:")
        if 'MLMeanReversionStrategy' in results:
            result = results['MLMeanReversionStrategy']
            print(f"   - Total Return: {result.get('total_return_pct', 0):.2f}%")
            print(f"   - Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
            print(f"   - Max Drawdown: {result.get('max_drawdown_pct', 0):.2f}%")
            print(f"   - Win Rate: {result.get('win_rate', 0)*100:.2f}%")
        
        print("\n2. Adaptive Trend Strategy:")
        if 'AdaptiveTrendStrategy' in results:
            result = results['AdaptiveTrendStrategy']
            print(f"   - Total Return: {result.get('total_return_pct', 0):.2f}%")
            print(f"   - Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
            print(f"   - Max Drawdown: {result.get('max_drawdown_pct', 0):.2f}%")
            print(f"   - Win Rate: {result.get('win_rate', 0)*100:.2f}%")

if __name__ == "__main__":
    try:
        test_strategies()
    except Exception as e:
        print(f"An error occurred during testing: {e}")