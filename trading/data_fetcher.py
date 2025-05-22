import requests
from utils.config_loader import load_config
import pandas as pd
from utils.instrument_helper import get_instrument_key

class UpstoxDataFetcher:
    BASE_URL = "https://api.upstox.com/v2"

    def __init__(self, config=None):
        if config is None:
            config = load_config()
        upstox_cfg = config['upstox']
        self.access_token = upstox_cfg['access_token']
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }

    def get_historical_candles(self, instrument_key, interval="1minute", start_date=None, end_date=None, debug=False):
        """
        Fetch historical candle data for an instrument_key.
        interval: '1minute', '30minute', 'day', 'week', 'month'
        start_date, end_date: 'YYYY-MM-DD' (default: last 5 days)
        """
        url = f"{self.BASE_URL}/historical-candle/{instrument_key}/{interval}"
        params = {}
        if start_date:
            params['from'] = start_date
        if end_date:
            params['to'] = end_date
        if debug:
            print(f"Requesting URL: {url}")
            print(f"With params: {params}")
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code == 200:
            data = response.json()['data']['candles']
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=["datetime", "open", "high", "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df
        else:
            raise Exception(f"API Error: {response.status_code} {response.text}")

if __name__ == "__main__":
    fetcher = UpstoxDataFetcher()
    from utils.instrument_helper import load_instrument_master, get_instrument_key
    master_df = load_instrument_master()
    test_cases = [
        ("RELIANCE", "NSE_EQ"),
        ("TCS", "NSE_EQ"),
    ]
    # Try to find a NIFTY F&O instrument automatically
    nifty_fo = master_df[(master_df['tradingsymbol'].str.startswith('NIFTY')) & (master_df['exchange'] == 'NSE_FO')]
    if not nifty_fo.empty:
        first_nifty = nifty_fo.iloc[0]
        test_cases.append((first_nifty['tradingsymbol'], 'NSE_FO'))
        print(f"\nAuto-selected NIFTY F&O instrument: {first_nifty['tradingsymbol']} ({first_nifty['instrument_key']})")
    else:
        print("No NIFTY F&O instrument found in instrument master.")

    for symbol, exchange in test_cases:
        print(f"\n=== Testing {symbol} ({exchange}) ===")
        try:
            instrument_key = get_instrument_key(symbol, exchange, master_df)
            print(f"Instrument key: {instrument_key}")
            for interval in ["1minute", "day"]:
                try:
                    print(f"Trying interval: {interval}")
                    df = fetcher.get_historical_candles(
                        instrument_key=instrument_key,
                        interval=interval,
                        start_date="2024-05-10",
                        end_date="2024-05-17",
                        debug=True
                    )
                    print(df.head())
                except Exception as e:
                    print(f"{interval} interval failed: {e}")
        except Exception as e:
            print(f"Failed to fetch for {symbol} ({exchange}): {e}")
