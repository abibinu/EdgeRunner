import configparser
import webbrowser
import requests
import os 

# --- Configuration ---
script_dir = os.path.dirname(__file__) 
project_dir = os.path.dirname(script_dir) 

CONFIG_FILE = os.path.join(project_dir, 'config.ini')
TOKEN_FILE = os.path.join(project_dir, 'access_token.txt')

def load_config():
    """Loads API credentials from the config file."""
    config = configparser.ConfigParser()
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Error: Configuration file '{CONFIG_FILE}' not found.")
    config.read(CONFIG_FILE)
    
    try:
        api_key = config['UPSTOX']['API_KEY']
        api_secret = config['UPSTOX']['API_SECRET']
        redirect_uri = config['UPSTOX']['REDIRECT_URI']
        return api_key, api_secret, redirect_uri
    except KeyError as e:
        raise KeyError(f"Error: Missing key {e} in '{CONFIG_FILE}'. Please check the file.")

def get_authorization_code(api_key, redirect_uri):
    """Opens the browser for user login and returns the manually entered auth code."""
    auth_base_url = "https://api.upstox.com/v2/login/authorization/dialog" 
    
    auth_url = f"{auth_base_url}?client_id={api_key}&redirect_uri={redirect_uri}&response_type=code" 
    # Optional: add '&state=YOUR_RANDOM_STRING' for security if desired/required by API

    print("-" * 80)
    print("Opening browser for Upstox login...")
    print("1. Log in to Upstox.")
    print("2. Authorize the application ('EdgeRunner' or your app name).")
    print("3. You will be redirected to a URL like:")
    print(f"   {redirect_uri}/?code=YOUR_CODE")
    print("4. Copy the value of the 'code' parameter from the browser's address bar.")
    print("-" * 80)
    
    webbrowser.open(auth_url)
    
    authorization_code = input("Paste the authorization code here: ").strip()
    return authorization_code

def get_access_token(api_key, api_secret, redirect_uri, authorization_code):
    """Exchanges the authorization code for an access token."""
    token_url = "https://api.upstox.com/v2/login/authorization/token" 
    
    headers = {
        'accept': 'application/json',
        'Api-Version': '2.0', 
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    data = {
        'client_id': api_key,
        'client_secret': api_secret,
        'redirect_uri': redirect_uri,
        'code': authorization_code,
        'grant_type': 'authorization_code'
    }
    
    try:
        response = requests.post(token_url, headers=headers, data=data)
        response.raise_for_status() 
        
        token_data = response.json()
        print("\nSuccessfully obtained access token!")
        
        access_token = token_data.get('access_token')
        if not access_token:
            raise ValueError("Error: 'access_token' not found in the response.")
            
        with open(TOKEN_FILE, 'w') as f:
            f.write(access_token)
        print(f"Access token saved to {TOKEN_FILE}")
            
        return access_token

    except requests.exceptions.RequestException as e:
        print(f"Error during access token request: {e}")
        if e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            try:
                print(f"Response content: {e.response.json()}") 
            except requests.exceptions.JSONDecodeError:
                print(f"Response content: {e.response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    try:
        print("Starting Upstox authentication process...")
        api_key, api_secret, redirect_uri = load_config()
        
        auth_code = get_authorization_code(api_key, redirect_uri)
        
        if auth_code:
            get_access_token(api_key, api_secret, redirect_uri, auth_code)
        else:
            print("Authentication process aborted.")
            
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred in the main block: {e}")