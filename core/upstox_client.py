import requests
import os
import json
import configparser
import pandas as pd
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
from time import sleep
from functools import wraps
import time

# --- Configuration & Constants ---
script_dir = os.path.dirname(__file__) 
project_dir = os.path.dirname(script_dir) 

TOKEN_FILE = os.path.join(project_dir, 'access_token.txt')
CONFIG_FILE = os.path.join(project_dir, 'config.ini')
BASE_URL = "https://api.upstox.com/v2"

# Market timing constants (IST)
MARKET_OPEN_HOUR = 9  # 9:15 AM IST
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15  # 3:30 PM IST
MARKET_CLOSE_MINUTE = 30

# Rate limiting constants
RATE_LIMIT_REQUESTS = 100  # Maximum requests per minute (adjust based on API limits)
REQUEST_TIMESTAMPS = []

# --- Helper Functions ---

def load_access_token():
    """Loads the access token from the file."""
    if not os.path.exists(TOKEN_FILE):
        raise FileNotFoundError(f"Error: Access token file '{TOKEN_FILE}' not found. " 
                                f"Run authenticate.py first.")
    with open(TOKEN_FILE, 'r') as f:
        access_token = f.read().strip()
    if not access_token:
         raise ValueError(f"Error: Access token file '{TOKEN_FILE}' is empty.")
    return access_token

def rate_limit_decorator(func):
    """Decorator to implement rate limiting"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        global REQUEST_TIMESTAMPS
        REQUEST_TIMESTAMPS = [ts for ts in REQUEST_TIMESTAMPS if current_time - ts < 60]
        
        # Check if we've hit the rate limit
        if len(REQUEST_TIMESTAMPS) >= RATE_LIMIT_REQUESTS:
            sleep_time = 61 - (current_time - REQUEST_TIMESTAMPS[0])
            if sleep_time > 0:
                print(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                sleep(sleep_time)
        
        # Add current timestamp
        REQUEST_TIMESTAMPS.append(current_time)
        
        return func(*args, **kwargs)
    return wrapper

def is_market_hours():
    """
    Check if the market is currently open.
    Returns:
        bool: True if market is open, False otherwise
    """
    now = datetime.now(ZoneInfo('Asia/Kolkata'))
    
    # Check if it's a trading day
    if not is_trading_day(now.date()):
        return False
    
    # Convert current time to minutes since midnight
    current_minutes = now.hour * 60 + now.minute
    market_open_minutes = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE
    market_close_minutes = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE
    
    return market_open_minutes <= current_minutes <= market_close_minutes

@rate_limit_decorator
def _make_api_request(method, endpoint, params=None, data=None):
    """Generic function to make authenticated API requests."""
    access_token = load_access_token()
    url = f"{BASE_URL}{endpoint}"
    headers = {
        'accept': 'application/json',
        'Api-Version': '2.0',
        'Authorization': f'Bearer {access_token}'
    }
    if method.upper() == 'POST' or method.upper() == 'PUT':
         headers['Content-Type'] = 'application/json' 

    try:
        print(f"Request: {method.upper()} {url} | Params: {params} | Data: {data}")
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, params=params)
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=data) 
        # Add PUT, DELETE etc. if needed
        else:
             raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status() 
        
        # Handle potential empty responses for certain actions (like order cancellation)
        if response.status_code == 204 or not response.content:
             return {"status": "success", "data": None} 
             
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response status code: {http_err.response.status_code}")
        error_details = None
        try:
            error_details = http_err.response.json()
            print(f"Response content: {json.dumps(error_details, indent=4)}")
        except json.JSONDecodeError:
             error_text = http_err.response.text
             print(f"Response content: {error_text}")
             error_details = {"error_message": error_text}
             
        if http_err.response.status_code == 401:
            print("\n>>> Received 401 Unauthorized. Your access token might be invalid or expired. "
                  "Try running authenticate.py again. <<<")
        raise Exception(f"API request failed: {http_err.response.status_code}") from http_err
        
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
        raise Exception(f"API Request failed: {req_err}") from req_err
    except Exception as e:
        print(f"An unexpected error occurred during API request: {e}")
        raise Exception(f"Unexpected error: {e}") from e


# --- API Call Functions ---

def get_user_profile():
    """Fetches user profile information."""
    return _make_api_request('GET', '/user/profile')

def get_funds_and_margins():
     """Fetches user funds and margins for equity segment."""
     return _make_api_request('GET', '/user/funds_and_margin') 

def is_trading_day(check_date):
    """
    Check if a given date is likely to be a trading day (Mon-Fri).
    This is a basic check - doesn't account for holidays.
    """
    return check_date.weekday() < 5  # 0-4 are Monday to Friday

def get_previous_trading_day(from_date):
    """
    Get the most recent previous trading day from a given date.
    """
    current = from_date
    while not is_trading_day(current):
        current -= timedelta(days=1)
    return current

def get_historical_candles(instrument_key, interval, to_date_str, from_date_str, check_market_hours=True):
    """
    Fetches historical candle data and returns it as a Pandas DataFrame.
    Args:
        instrument_key (str): Instrument identifier (e.g., "NSE_EQ|INE009A01021")
        interval (str): Time interval ('1minute', '5minute', '30minute', 'day', 'week', 'month')
        to_date_str (str): End date in 'YYYY-MM-DD' format
        from_date_str (str): Start date in 'YYYY-MM-DD' format
        check_market_hours (bool): If True, validates market hours for intraday data
    Returns:
        pandas.DataFrame: DataFrame with columns [timestamp, open, high, low, close, volume, open_interest]
    """
    # Validate interval
    valid_intervals = ['1minute', '5minute', '30minute', 'day', 'week', 'month']
    if interval not in valid_intervals:
        print(f"Error: Invalid interval. Must be one of {valid_intervals}")
        return pd.DataFrame()

    # For intraday data, check market hours if requested
    if 'minute' in interval and check_market_hours:
        if not is_market_hours():
            print("Warning: Market is currently closed. Using historical data only.")

    # Convert date strings to date objects for validation
    try:
        to_date = datetime.strptime(to_date_str, '%Y-%m-%d').date()
        from_date = datetime.strptime(from_date_str, '%Y-%m-%d').date()
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        print("Please ensure dates are in YYYY-MM-DD format")
        return pd.DataFrame()

    # Validate dates
    if to_date < from_date:
        print("Error: 'to_date' must be later than or equal to 'from_date'")
        return pd.DataFrame()
    
    # For minute data, adjust dates to ensure we're requesting trading days
    if 'minute' in interval:
        if not is_trading_day(to_date):
            adjusted_date = get_previous_trading_day(to_date)
            print(f"Warning: {to_date} is not a trading day. Adjusting to {adjusted_date}")
            to_date = adjusted_date
            to_date_str = to_date.strftime('%Y-%m-%d')
        # For minute data, from_date should be same as to_date
        from_date = to_date
        from_date_str = to_date_str

    endpoint = f"/historical-candle/{instrument_key}/{interval}/{to_date_str}/{from_date_str}"
    response_data = _make_api_request('GET', endpoint)
    
    if response_data and response_data.get('status') == 'success' and response_data.get('data', {}).get('candles'):
        candles_list = response_data['data']['candles']
        
        # Define column names based on Upstox API documentation order
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'open_interest']
        
        df = pd.DataFrame(candles_list, columns=columns)
        
        # Convert timestamp to datetime objects with IST timezone
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('Asia/Kolkata')
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'open_interest']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])

        print(f"Successfully converted {len(df)} candles to DataFrame.")
        return df
    else:
        error_msg = "No data available. "
        if not is_trading_day(to_date):
            error_msg += f"Note: {to_date_str} is not a trading day (weekend). "
        error_msg += "Try a different date range or verify the instrument key and interval."
        print(f"Warning: {error_msg}")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'open_interest'])

if __name__ == "__main__":
    print("Testing Upstox Client...")
    try:
        # 1. Test Get Profile
        print("\n--- Testing Get Profile ---")
        profile = get_user_profile()
        print("Profile Data:")
        print(json.dumps(profile, indent=4))

        # 2. Test Get Funds (Uncomment when ready)
        # print("\n--- Testing Get Funds & Margins ---")
        # funds = get_funds_and_margins()
        # print("Funds Data:")
        # print(json.dumps(funds, indent=4))
        
        print("\n--- Testing Historical Data (Pandas) ---")
        try:
            instrument_key_infy = "NSE_EQ|INE009A01021" # Verified Key
            
            # Get the most recent trading day
            today = date.today()
            end_date = get_previous_trading_day(today - timedelta(days=1))
            start_date = get_previous_trading_day(end_date - timedelta(days=1))

            to_date_str = end_date.strftime('%Y-%m-%d')
            from_date_str = start_date.strftime('%Y-%m-%d')

            # Fetch daily data
            print(f"Fetching DAILY candles for {instrument_key_infy} from {from_date_str} to {to_date_str}")
            daily_candles_df = get_historical_candles(instrument_key_infy, 'day', to_date_str, from_date_str)
            
            print("Daily Candle DataFrame:")
            if not daily_candles_df.empty:
                print(daily_candles_df.head())
                print("\nDataFrame Info:")
                daily_candles_df.info()
            else:
                print("No daily data received.")

            # Fetch 1-minute data for the most recent trading day
            last_trading_day = get_previous_trading_day(today)
            last_trading_day_str = last_trading_day.strftime('%Y-%m-%d')
            
            print(f"\nFetching 1-MINUTE candles for {instrument_key_infy} for {last_trading_day_str}")
            minute_candles_df = get_historical_candles(instrument_key_infy, '1minute', last_trading_day_str, last_trading_day_str)
            
            print("\nMinute Candle DataFrame (first 5 rows):")
            if not minute_candles_df.empty:
                print(minute_candles_df.head())
                print(f"\nTotal 1-minute candles received: {len(minute_candles_df)}")
            else:
                print("No 1-minute data received.")

        except Exception as e:
            print(f"An error occurred during historical data testing: {e}")

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred during testing: {e}")