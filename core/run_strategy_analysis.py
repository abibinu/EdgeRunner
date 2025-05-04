import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add the core directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.ml_mean_reversion import MLMeanReversionStrategy
from strategies.adaptive_trend import AdaptiveTrendStrategy
from backtest.strategy_evaluator import StrategyEvaluator
from upstox_client import get_historical_candles

def main():
    # Create output directory for plots
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Fetch historical data for multiple stocks
    stocks = {
        'INFY': 'NSE_EQ|INE009A01021',  # Infosys
        'TCS': 'NSE_EQ|INE467B01029',   # TCS
        'RELIANCE': 'NSE_EQ|INE002A01018'  # Reliance
    }
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)  # 6 months of data
    
    results_by_stock = {}
    
    for stock_name, instrument_key in stocks.items():
        print(f"\nAnalyzing {stock_name}...")
        
        # Fetch daily data
        data = get_historical_candles(
            instrument_key=instrument_key,
            interval='day',
            to_date_str=end_date.strftime('%Y-%m-%d'),
            from_date_str=start_date.strftime('%Y-%m-%d')
        )
        
        if data.empty:
            print(f"No data available for {stock_name}")
            continue
            
        # Initialize strategies with complete parameters
        ml_strategy = MLMeanReversionStrategy({
            'lookback_period': 20,
            'entry_zscore': 1.0,
            'exit_zscore': 0.3,
            'stop_loss_pct': 1.0,
            'min_volume_ratio': 0.8,
            'ml_threshold': 0.4,  # Lowered from 0.55 to generate more signals
            'price_deviation_threshold': 0.5,  # New parameter for price deviation
            'trend_window': 5  # New parameter for trend calculation
        })
        
        adaptive_strategy = AdaptiveTrendStrategy({
            'lookback_period': 20,
            'volatility_window': 20,
            'trend_threshold': 0.01,
            'stop_loss_atr_mult': 1.5,
            'profit_target_atr_mult': 2.5,
            'rsi_upper': 75,  # Increased from 70
            'rsi_lower': 25,  # Decreased from 30
            'min_trend_strength': 0.25  # Decreased from 0.3 for more signals
        })
        
        # Train ML strategy on first 70% of data
        train_size = int(len(data) * 0.7)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        print(f"Training ML strategy on {len(train_data)} days of data...")
        ml_strategy.train_model(train_data)
        
        # Initialize evaluator
        evaluator = StrategyEvaluator(initial_capital=100000, commission=0.0003)  # 0.03% commission
        
        # Evaluate strategies
        strategies = [ml_strategy, adaptive_strategy]
        results = evaluator.evaluate_strategies(strategies, test_data)
        results_by_stock[stock_name] = results
        
        # Generate plots for this stock
        evaluator.plot_equity_curves(
            results,
            test_data,  # Pass the test data for proper date indexing
            save_path=os.path.join(output_dir, f'{stock_name}_equity_curves.png')
        )
        
        evaluator.plot_drawdown_analysis(
            results,
            test_data,  # Pass the test data for proper date indexing
            save_path=os.path.join(output_dir, f'{stock_name}_drawdowns.png')
        )
        
        evaluator.plot_monthly_returns(
            results,
            test_data,  # Pass the test data for proper date indexing
            save_path=os.path.join(output_dir, f'{stock_name}_monthly_returns.png')
        )
        
        evaluator.plot_trade_analysis(
            results,
            save_path=os.path.join(output_dir, f'{stock_name}_trade_analysis.png')
        )
        
        # Generate performance report
        performance_report = evaluator.generate_performance_report(results)
        print(f"\n{stock_name} Performance Report:")
        print(performance_report)
        
        # Save performance report to CSV
        performance_report.to_csv(
            os.path.join(output_dir, f'{stock_name}_performance_report.csv')
        )
        
        # Print detailed strategy comparison
        print("\nStrategy Details:")
        for strategy_name, result in results.items():
            print(f"\n{strategy_name}:")
            print(f"Total Return: {result.get('total_return_pct', 0):.2f}%")
            print(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {result.get('max_drawdown_pct', 0):.2f}%")
            print(f"Win Rate: {result.get('win_rate', 0)*100:.2f}%")
            print(f"Total Trades: {result.get('total_trades', 0)}")
            print(f"Average Trade PnL: ₹{result.get('avg_trade', 0):.2f}")
            print(f"Profit Factor: {result.get('profit_factor', 0):.2f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")