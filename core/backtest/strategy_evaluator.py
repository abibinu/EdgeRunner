import pandas as pd
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.backtest_engine import BacktestEngine
from strategies.base_strategy import BaseStrategy

class StrategyEvaluator:
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.0):
        """
        Initialize strategy evaluator
        Args:
            initial_capital: Starting capital for backtesting
            commission: Commission per trade (percentage)
        """
        self.backtest_engine = BacktestEngine(initial_capital, commission)
        
    def evaluate_strategies(self, strategies: List[BaseStrategy], data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate multiple strategies on the same dataset
        Args:
            strategies: List of strategy instances to evaluate
            data: Historical price data
        Returns:
            Dict containing evaluation results for all strategies
        """
        results = {}
        
        for strategy in strategies:
            strategy_name = strategy.__class__.__name__
            backtest_result = self.backtest_engine.run_backtest(strategy, data)
            results[strategy_name] = backtest_result
            
        return results
    
    def plot_equity_curves(self, results: Dict[str, Dict[str, Any]], data: pd.DataFrame, save_path: str = None):
        """Plot equity curves for all strategies"""
        plt.figure(figsize=(12, 6))
        
        for strategy_name, result in results.items():
            if 'equity_curve' in result:
                equity_series = pd.Series(
                    result['equity_curve'],
                    index=pd.to_datetime(data['timestamp'])[:len(result['equity_curve'])]
                )
                plt.plot(equity_series.index, equity_series.values, label=f"{strategy_name}")
        
        plt.title('Strategy Performance Comparison')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def plot_drawdown_analysis(self, results: Dict[str, Dict[str, Any]], data: pd.DataFrame, save_path: str = None):
        """Plot drawdown analysis for all strategies"""
        plt.figure(figsize=(12, 6))
        
        for strategy_name, result in results.items():
            if 'equity_curve' in result:
                equity_series = pd.Series(
                    result['equity_curve'],
                    index=pd.to_datetime(data['timestamp'])[:len(result['equity_curve'])]
                )
                running_max = equity_series.cummax()
                drawdowns = (running_max - equity_series) / running_max * 100
                plt.plot(drawdowns.index, drawdowns.values, label=f"{strategy_name}")
        
        plt.title('Drawdown Analysis')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def plot_monthly_returns(self, results: Dict[str, Dict[str, Any]], data: pd.DataFrame, save_path: str = None):
        """Plot monthly returns heatmap"""
        plt.figure(figsize=(15, 5 * len(results)))
        
        for i, (strategy_name, result) in enumerate(results.items()):
            if 'equity_curve' in result:
                # Create a Series with datetime index
                equity_series = pd.Series(
                    result['equity_curve'],
                    index=pd.to_datetime(data['timestamp'])[:len(result['equity_curve'])]
                )
                returns = equity_series.pct_change()
                
                # Calculate monthly returns using datetime index
                monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
                
                # Create monthly returns matrix
                monthly_returns_table = pd.pivot_table(
                    monthly_returns.reset_index(),
                    values=0,
                    index=monthly_returns.index.year,
                    columns=monthly_returns.index.month,
                    fill_value=0
                ) * 100
                
                plt.subplot(len(results), 1, i+1)
                sns.heatmap(monthly_returns_table, 
                           annot=True, 
                           fmt='.1f', 
                           center=0, 
                           cmap='RdYlGn',
                           cbar_kws={'label': 'Monthly Return (%)'})
                plt.title(f'{strategy_name} - Monthly Returns (%)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def generate_performance_report(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Generate a performance comparison report"""
        metrics = []
        
        for strategy_name, result in results.items():
            if 'error' not in result:
                metrics.append({
                    'Strategy': strategy_name,
                    'Total Return (%)': round(result['total_return_pct'], 2),
                    'Sharpe Ratio': round(result['sharpe_ratio'], 2),
                    'Sortino Ratio': round(result['sortino_ratio'], 2),
                    'Max Drawdown (%)': round(result['max_drawdown_pct'], 2),
                    'Win Rate (%)': round(result['win_rate'] * 100, 2),
                    'Total Trades': result['total_trades'],
                    'Avg Trade': round(result['avg_trade'], 2),
                    'Profit Factor': round(result['profit_factor'], 2)
                })
        
        return pd.DataFrame(metrics).set_index('Strategy')
    
    def plot_trade_analysis(self, results: Dict[str, Dict[str, Any]], save_path: str = None):
        """Plot trade analysis for all strategies"""
        fig = plt.figure(figsize=(15, 5 * len(results)))
        
        for i, (strategy_name, result) in enumerate(results.items()):
            if 'trades' in result and result['trades']:
                trades_df = pd.DataFrame(result['trades'])
                pnls = trades_df['pnl']
                
                plt.subplot(len(results), 2, 2*i+1)
                sns.histplot(pnls, bins=50)
                plt.title(f'{strategy_name} - PnL Distribution')
                plt.xlabel('Trade PnL')
                plt.ylabel('Frequency')
                
                plt.subplot(len(results), 2, 2*i+2)
                cumulative_pnl = pnls.cumsum()
                plt.plot(cumulative_pnl)
                plt.title(f'{strategy_name} - Cumulative PnL')
                plt.xlabel('Trade Number')
                plt.ylabel('Cumulative PnL')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()