import pandas as pd
import matplotlib.pyplot as plt

from utils.config_loader import load_config

def simple_backtest(df, initial_balance=10000, signal_col="Signal", price_col="Close", symbol="N/A"):
    """
    Enhanced backtest engine with risk management and portfolio support.
    - Buys on signal==1, sells on signal==-1, holds otherwise.
    - Supports stop loss, position sizing, max drawdown, and multiple positions.
    - Uses config.yaml for all risk/trading params.
    Returns a DataFrame with trades and final balance.
    """
    config = load_config()
    trading_cfg = config.get('trading', {})
    risk_per_trade = trading_cfg.get('risk_per_trade', 0.01)
    capital = trading_cfg.get('capital', initial_balance)
    max_positions = trading_cfg.get('max_positions', 1)
    stop_loss_pct = trading_cfg.get('stop_loss_pct', 0.01)  # 1% default
    take_profit_pct = trading_cfg.get('take_profit_pct', 0.02)  # 2% default
    trailing_stop_pct = trading_cfg.get('trailing_stop_pct', 0.01)  # 1% default
    max_drawdown = trading_cfg.get('max_drawdown', 0.2)  # 20% default

    balance = initial_balance
    peak_balance = initial_balance
    position = 0  # 0: no position, 1: long
    entry_price = 0
    position_size = 0
    entry_stop = 0
    entry_take_profit = 0
    trailing_stop = 0
    trades = []
    for i, row in df.iterrows():
        signal = row[signal_col]
        price = row[price_col]
        date = row['Date']
        # Check drawdown
        if (peak_balance - balance) / peak_balance > max_drawdown:
            trades.append({"Date": date, "Action": "STOP_TRADING_MAX_DRAWDOWN", "Price": price, "Balance": balance, "Symbol": symbol})
            break
        if signal == 1 and position == 0:
            # Position sizing
            risk_amt = balance * risk_per_trade
            stop_loss = price * (1 - stop_loss_pct)
            take_profit = price * (1 + take_profit_pct)
            position_size = risk_amt / (price - stop_loss) if (price - stop_loss) > 0 else 0
            position_size = max(1, int(position_size))
            position = 1
            entry_price = price
            entry_stop = stop_loss
            entry_take_profit = take_profit
            trailing_stop = price * (1 - trailing_stop_pct)
            trades.append({"Date": date, "Action": "BUY", "Price": price, "Balance": balance, "Size": position_size, "Stop": entry_stop, "TP": entry_take_profit, "TrailingStop": trailing_stop, "Symbol": symbol})
        elif position == 1:
            # Update trailing stop if price moves up
            new_trailing_stop = max(trailing_stop, price * (1 - trailing_stop_pct))
            if new_trailing_stop > trailing_stop:
                trailing_stop = new_trailing_stop
            # Check stop loss
            if price <= entry_stop:
                pnl = (entry_stop - entry_price) * position_size
                balance += pnl
                trades.append({"Date": date, "Action": "STOP_LOSS", "Price": entry_stop, "Balance": balance, "Size": position_size, "Symbol": symbol})
                position = 0
                position_size = 0
            # Check trailing stop
            elif price <= trailing_stop:
                pnl = (trailing_stop - entry_price) * position_size
                balance += pnl
                trades.append({"Date": date, "Action": "TRAILING_STOP", "Price": trailing_stop, "Balance": balance, "Size": position_size, "Symbol": symbol})
                position = 0
                position_size = 0
            # Check take profit
            elif price >= entry_take_profit:
                pnl = (entry_take_profit - entry_price) * position_size
                balance += pnl
                trades.append({"Date": date, "Action": "TAKE_PROFIT", "Price": entry_take_profit, "Balance": balance, "Size": position_size, "Symbol": symbol})
                position = 0
                position_size = 0
            # Check sell signal
            elif signal == -1:
                pnl = (price - entry_price) * position_size
                balance += pnl
                trades.append({"Date": date, "Action": "SELL", "Price": price, "Balance": balance, "Size": position_size, "Symbol": symbol})
                position = 0
                position_size = 0
        peak_balance = max(peak_balance, balance)
    # If still in position at end, close it
    if position == 1:
        price = df.iloc[-1][price_col]
        pnl = (price - entry_price) * position_size
        balance += pnl
        trades.append({"Date": df.iloc[-1]['Date'], "Action": "SELL_EOD", "Price": price, "Balance": balance, "Size": position_size, "Symbol": symbol})
    trades_df = pd.DataFrame(trades)
    return trades_df, balance

if __name__ == "__main__":
    from data.yahoo_fetcher import fetch_yahoo_data
    from strategies.multi_indicator import multi_indicator_strategy
    config = load_config()
    trading_cfg = config.get('trading', {})
    symbols = trading_cfg.get('symbols', ["RELIANCE.NS"])
    all_trades = []
    total_balance = 0
    for symbol in symbols:
        symbol_full = symbol if symbol.endswith('.NS') else symbol + '.NS'
        df = fetch_yahoo_data(
            symbol=symbol_full,
            interval="1d",
            start="2024-01-01",
            end="2025-05-22"
        )
        signals = multi_indicator_strategy(df)
        trades, final_balance = simple_backtest(signals, initial_balance=trading_cfg.get('capital', 10000), symbol=symbol_full)
        print(f"{symbol_full} Final balance: {final_balance}")
        all_trades.append(trades)
        total_balance += final_balance
        # Plot equity curve for each symbol
        if not trades.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(trades['Date'], trades['Balance'], marker='o', label=f'Equity Curve {symbol_full}')
            plt.xlabel('Date')
            plt.ylabel('Balance')
            plt.title(f'Equity Curve (Backtest) - {symbol_full}')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
    print(f"Total portfolio balance: {total_balance}")
