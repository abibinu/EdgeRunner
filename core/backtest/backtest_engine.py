import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.base_strategy import BaseStrategy
from trading_config import RISK_MANAGEMENT

class BacktestEngine:
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.0):
        """Initialize backtesting engine"""
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset()
    
    def reset(self):
        """Reset backtest state"""
        self.capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.peak_capital = self.initial_capital
        self.open_positions = 0
    
    def calculate_position_size(self, current_price: float, atr: float, risk_per_trade: float = 0.01) -> float:
        """Calculate position size based on ATR and risk per trade"""
        if atr == 0:
            return 0
        
        # Risk amount in currency
        risk_amount = self.capital * risk_per_trade
        
        # Use ATR for stop loss distance
        stop_distance = atr * 2
        
        # Calculate position size
        position_size = risk_amount / stop_distance if stop_distance > 0 else 0
        
        # Adjust for maximum position size limit
        max_position_value = self.capital * RISK_MANAGEMENT['max_position_size']
        position_value = position_size * current_price
        
        if position_value > max_position_value:
            position_size = max_position_value / current_price
            
        return position_size
    
    def _should_exit_position(self, current_price: float, entry_price: float, position: float, daily_pnl: List[float], window: int = 5) -> bool:
        """Determine if we should exit a position based on various risk metrics"""
        # Calculate current position P&L
        if position > 0:
            unrealized_pnl = (current_price - entry_price) * abs(position)
        else:
            unrealized_pnl = (entry_price - current_price) * abs(position)
        
        # Calculate drawdown from entry
        drawdown = (unrealized_pnl / (entry_price * abs(position))) * 100
        
        # Check recent performance
        recent_pnl_sum = sum(daily_pnl[-window:]) if len(daily_pnl) >= window else 0
        
        # Exit conditions
        return (
            drawdown < -RISK_MANAGEMENT['max_drawdown_pct'] * 100 or  # Stop loss hit
            recent_pnl_sum < -self.capital * RISK_MANAGEMENT['max_drawdown_pct'] or  # Poor recent performance
            self.capital < self.initial_capital * 0.95  # Capital protection
        )
    
    def run_backtest(self, strategy: BaseStrategy, data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest for a strategy"""
        self.reset()
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Initialize tracking variables
        position = 0
        entry_price = 0
        entry_time = None
        daily_pnl = []
        
        # Calculate ATR for position sizing
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=20).mean()
        
        # Track performance
        equity = []
        returns = []
        drawdowns = []
        peak = self.capital
        
        # Iterate through the data
        for idx in data.index:
            current_price = data.loc[idx, 'close']
            signal = signals[idx]
            current_atr = atr[idx]
            
            # Process exits
            if position != 0 and (signal == 0 or self._should_exit_position(current_price, entry_price, position, daily_pnl)):
                # Calculate profit/loss
                if position > 0:
                    pnl = (current_price - entry_price) * abs(position)
                else:
                    pnl = (entry_price - current_price) * abs(position)
                
                # Apply commission
                commission_cost = abs(current_price * position) * self.commission
                pnl -= commission_cost
                
                # Update capital and track daily PnL
                self.capital += pnl
                daily_pnl.append(pnl)
                
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
                self.open_positions -= 1
            
            # Process entries
            elif position == 0 and signal != 0:
                # Check if we can take new positions
                if self.open_positions >= RISK_MANAGEMENT['max_open_positions']:
                    continue
                
                # Calculate position size
                position_size = self.calculate_position_size(current_price, current_atr)
                position = signal * position_size
                
                entry_price = current_price
                entry_time = idx
                
                # Apply entry commission
                commission_cost = current_price * abs(position) * self.commission
                self.capital -= commission_cost
                self.open_positions += 1
            
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
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            avg_win = np.mean([t['pnl'] for t in self.trades if t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in self.trades if t['pnl'] < 0]) if losing_trades > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # Calculate mean return and standard deviation for ratios
            mean_return = np.mean(returns) if len(returns) > 0 else 0
            std_return = np.std(returns) if len(returns) > 0 else float('inf')
            
            # Calculate Sortino ratio more safely
            negative_returns = [r for r in returns if r < 0]
            if len(negative_returns) > 0:
                downside_std = np.std(negative_returns)
                sortino_ratio = mean_return / downside_std * np.sqrt(252) if downside_std > 0 else (
                    float('inf') if mean_return > 0 else float('-inf')
                )
            else:
                # No negative returns - strategy hasn't lost money
                sortino_ratio = float('inf') if mean_return > 0 else 0
            
            # Calculate Sharpe ratio safely
            sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else (
                float('inf') if mean_return > 0 else 0
            )
            
            return {
                'total_return_pct': (self.capital - self.initial_capital) / self.initial_capital * 100,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown_pct': max(drawdowns),
                'win_rate': win_rate,
                'total_trades': total_trades,
                'avg_trade': np.mean([t['pnl'] for t in self.trades]),
                'profit_factor': profit_factor,
                'equity_curve': equity,
                'trades': self.trades
            }
        
        return {
            'total_return_pct': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown_pct': 0,
            'win_rate': 0,
            'total_trades': 0,
            'avg_trade': 0,
            'profit_factor': float('inf'),
            'equity_curve': equity,
            'trades': []
        }