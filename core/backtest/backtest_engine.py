import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.base_strategy import BaseStrategy

class BacktestEngine:
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.0):
        """
        Initialize backtesting engine
        Args:
            initial_capital: Starting capital for backtesting
            commission: Commission per trade (percentage)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset()
        
    def reset(self):
        """Reset backtest state"""
        self.capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, strategy: BaseStrategy, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest for a strategy on historical data
        Args:
            strategy: Trading strategy instance
            data: Historical price data
        Returns:
            Dict containing backtest results and performance metrics
        """
        self.reset()
        
        # Generate signals
        df = strategy.generate_signals(data)
        
        # Initialize position tracking
        position = 0
        entry_price = 0
        entry_time = None
        
        # Track performance
        equity = []
        returns = []
        drawdowns = []
        peak = self.capital
        
        for idx, row in df.iterrows():
            current_price = row['close']
            signal = row['signal']
            
            # Process exits
            if position != 0 and signal == 0:
                # Calculate profit/loss
                if position > 0:
                    pnl = (current_price - entry_price) * position
                else:
                    pnl = (entry_price - current_price) * abs(position)
                    
                # Apply commission
                commission_cost = abs(current_price * position) * self.commission
                pnl -= commission_cost
                
                # Update capital
                self.capital += pnl
                
                # Log trade
                self.trades.append({
                    'entry_time': entry_time,
                    'exit_time': idx,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': position,
                    'pnl': pnl,
                    'commission': commission_cost
                })
                
                position = 0
            
            # Process entries
            elif position == 0 and signal != 0:
                position = signal  # 1 for long, -1 for short
                entry_price = current_price
                entry_time = idx
                
                # Apply entry commission
                commission_cost = current_price * abs(position) * self.commission
                self.capital -= commission_cost
            
            # Track equity and drawdown
            current_value = self.capital
            if position != 0:
                unrealized_pnl = (current_price - entry_price) * position
                current_value += unrealized_pnl
            
            equity.append(current_value)
            peak = max(peak, current_value)
            drawdown = (peak - current_value) / peak * 100
            drawdowns.append(drawdown)
            
            if len(equity) > 1:
                daily_return = (current_value / equity[-2]) - 1
                returns.append(daily_return)
        
        # Calculate performance metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        if len(returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            max_drawdown = max(drawdowns)
            
            total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
            
            # Calculate sortino ratio (downside risk only)
            negative_returns = [r for r in returns if r < 0]
            sortino_ratio = np.sqrt(252) * np.mean(returns) / np.std(negative_returns) if len(negative_returns) > 0 and np.std(negative_returns) > 0 else 0
            
            # Average trade metrics
            avg_trade = np.mean([t['pnl'] for t in self.trades]) if self.trades else 0
            avg_win = np.mean([t['pnl'] for t in self.trades if t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in self.trades if t['pnl'] < 0]) if losing_trades > 0 else 0
            
            return {
                'total_return_pct': total_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown_pct': max_drawdown,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_trade': avg_trade,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
                'equity_curve': equity,
                'trades': self.trades
            }
        
        return {'error': 'Insufficient data for backtest'}