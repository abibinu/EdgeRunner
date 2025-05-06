import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the core directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.ml_mean_reversion import MLMeanReversionStrategy
from strategies.adaptive_trend import AdaptiveTrendStrategy
from backtest.strategy_evaluator import StrategyEvaluator
from upstox_client import get_historical_candles
from trading_config import *  # Import all configuration parameters

def check_risk_limits(portfolio_value, daily_returns):
    """Check if risk management limits are breached"""
    if len(daily_returns) == 0:
        return True, ""  # No data yet, no risk breach
        
    latest_return = daily_returns.iloc[-1] if not daily_returns.empty else 0
    if latest_return < -RISK_MANAGEMENT['daily_drawdown_limit']:
        return False, f"Daily drawdown limit breached: {latest_return:.2%}"
    return True, ""

def analyze_portfolio_correlation(results_by_stock):
    """Analyze correlation between different strategies and stocks"""
    returns_data = {}
    
    for stock, results in results_by_stock.items():
        for strategy_name, result in results.items():
            if 'equity_curve' in result and len(result['equity_curve']) > 1:
                # Calculate returns from equity curve
                equity_series = pd.Series(result['equity_curve'])
                returns = equity_series.pct_change().fillna(0)
                key = f"{stock}_{strategy_name}"
                returns_data[key] = returns
    
    if returns_data:
        returns_df = pd.DataFrame(returns_data).ffill().fillna(0)  # Using ffill() instead of fillna(method='ffill')
        correlation_matrix = returns_df.corr()
        
        # Add summary statistics
        volatility = returns_df.std() * np.sqrt(252)  # Annualized volatility
        sharpe = returns_df.mean() / returns_df.std() * np.sqrt(252)  # Annualized Sharpe
        
        print("\nStrategy Statistics:")
        stats_df = pd.DataFrame({
            'Annualized Volatility': volatility,
            'Annualized Sharpe': sharpe
        })
        print(stats_df)
        
        return correlation_matrix
    return pd.DataFrame()

def calculate_portfolio_metrics(results_by_stock):
    """Calculate aggregate portfolio metrics"""
    total_pnl = 0
    total_trades = 0
    winning_trades = 0
    max_drawdown = 0
    
    for stock, results in results_by_stock.items():
        for strategy_name, result in results.items():
            total_pnl += result.get('avg_trade', 0) * result.get('total_trades', 0)
            trades = result.get('total_trades', 0)
            total_trades += trades
            winning_trades += trades * result.get('win_rate', 0)
            max_drawdown = max(max_drawdown, result.get('max_drawdown_pct', 0))
    
    portfolio_win_rate = (winning_trades / total_trades if total_trades > 0 else 0)
    
    print("\nPortfolio Summary:")
    print(f"Total PnL: ₹{total_pnl:.2f}")
    print(f"Total Trades: {total_trades}")
    print(f"Portfolio Win Rate: {portfolio_win_rate*100:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")

def main():
    # Create output directory for plots
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=BACKTEST_PERIOD_DAYS)
    
    results_by_stock = {}
    portfolio_value = INITIAL_CAPITAL
    daily_returns = pd.Series(dtype=float)  # Initialize with float dtype
    
    for stock_name, instrument_key in STOCK_UNIVERSE.items():
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
        
        # Initialize strategies with parameters from config
        ml_strategy = MLMeanReversionStrategy(ML_MEAN_REVERSION_PARAMS)
        adaptive_strategy = AdaptiveTrendStrategy(ADAPTIVE_TREND_PARAMS)
        
        # Train ML strategy on first portion of data
        train_size = int(len(data) * TRAIN_TEST_SPLIT)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        print(f"Training ML strategy on {len(train_data)} days of data...")
        ml_strategy.train_model(train_data)
        
        # Initialize evaluator with config parameters
        evaluator = StrategyEvaluator(
            initial_capital=INITIAL_CAPITAL, 
            commission=COMMISSION_RATE
        )
        
        # Evaluate strategies
        strategies = [ml_strategy, adaptive_strategy]
        results = evaluator.evaluate_strategies(strategies, test_data)
        results_by_stock[stock_name] = results
        
        # Risk management checks - Calculate returns properly
        if results:
            strategy_returns = []
            for result in results.values():
                if 'equity_curve' in result:
                    equity_series = pd.Series(result['equity_curve'])
                    returns = equity_series.pct_change().fillna(0)
                    strategy_returns.append(returns)
            
            if strategy_returns:
                # Average returns across strategies
                combined_returns = pd.concat(strategy_returns, axis=1).mean(axis=1)
                if daily_returns.empty:
                    daily_returns = combined_returns
                else:
                    daily_returns = daily_returns.add(combined_returns, fill_value=0)
        
        risk_ok, risk_message = check_risk_limits(portfolio_value, daily_returns)
        if not risk_ok:
            print(f"Risk limit breached for {stock_name}: {risk_message}")
            continue
        
        # Generate and save visualization plots
        evaluator.plot_equity_curves(
            results, test_data,
            save_path=os.path.join(output_dir, f'{stock_name}_equity_curves.png')
        )
        
        evaluator.plot_drawdown_analysis(
            results, test_data,
            save_path=os.path.join(output_dir, f'{stock_name}_drawdowns.png')
        )
        
        evaluator.plot_monthly_returns(
            results, test_data,
            save_path=os.path.join(output_dir, f'{stock_name}_monthly_returns.png')
        )
        
        evaluator.plot_trade_analysis(
            results,
            save_path=os.path.join(output_dir, f'{stock_name}_trade_analysis.png')
        )
        
        # Generate and save performance reports
        performance_report = evaluator.generate_performance_report(results)
        print(f"\n{stock_name} Performance Report:")
        print(performance_report)
        
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
            print(f"Average Trade PnL: ₹{result.get('avg_trade', 0):.2f}")  # Fixed double colon
            print(f"Profit Factor: {result.get('profit_factor', 0):.2f}")
    
    # Portfolio-wide analysis
    print("\nPortfolio-wide Analysis:")
    calculate_portfolio_metrics(results_by_stock)
    
    print("\nPortfolio Correlation Analysis:")
    correlation_matrix = analyze_portfolio_correlation(results_by_stock)
    print("\nStrategy Correlations:")
    print(correlation_matrix)
    
    # Save correlation matrix
    correlation_matrix.to_csv(os.path.join(output_dir, 'portfolio_correlation.csv'))
    
    # Create strategy comparison visualization
    if results_by_stock:
        strategy_comparison = pd.DataFrame([
            {
                'Stock': stock,
                'Strategy': strategy,
                'Return (%)': result.get('total_return_pct', 0),
                'Sharpe': result.get('sharpe_ratio', 0),
                'Win Rate (%)': result.get('win_rate', 0) * 100,
                'Trades': result.get('total_trades', 0)
            }
            for stock, results in results_by_stock.items()
            for strategy, result in results.items()
        ])
        
        print("\nStrategy Comparison Summary:")
        print(strategy_comparison.sort_values('Return (%)', ascending=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        raise