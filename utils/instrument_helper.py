import pandas as pd
import os

def load_instrument_master(path=None):
    """
    Loads the Upstox instrument master CSV file.
    Returns a DataFrame.
    """
    if path is None:
        # Default location
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'instrument_master.csv')
    return pd.read_csv(path)

def get_instrument_key(symbol, exchange="NSE_EQ", master_df=None):
    """
    Returns the instrument_key for a given symbol and exchange.
    Looks for 'tradingsymbol' and 'exchange' columns in the master file.
    """
    if master_df is None:
        master_df = load_instrument_master()
    row = master_df[(master_df['tradingsymbol'] == symbol) & (master_df['exchange'] == exchange)]
    if not row.empty:
        return row.iloc[0]['instrument_key']
    else:
        raise ValueError(f"Instrument key not found for {symbol} on {exchange}")

if __name__ == "__main__":
    df = load_instrument_master()
    print(df.head())
    print(get_instrument_key("RELIANCE", "NSE_EQ", df))
