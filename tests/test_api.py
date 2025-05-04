import requests
import os
import json 

# --- Configuration ---
TOKEN_FILE = 'access_token.txt'
BASE_URL = "https://api.upstox.com/v2" 

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

def get_user_profile(access_token):
    """Fetches user profile information from Upstox."""
    profile_url = f"{BASE_URL}/user/profile"
    
    headers = {
        'accept': 'application/json',
        'Api-Version': '2.0',
        'Authorization': f'Bearer {access_token}' 
    }
    
    try:
        print(f"Making request to: {profile_url}")
        response = requests.get(profile_url, headers=headers)
        response.raise_for_status() 
        
        profile_data = response.json()
        print("\nSuccessfully fetched user profile!")

        print(json.dumps(profile_data, indent=4))
        
        return profile_data

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response status code: {http_err.response.status_code}")
        try:
            print(f"Response content: {http_err.response.json()}")
        except json.JSONDecodeError:
            print(f"Response content: {http_err.response.text}")
        if http_err.response.status_code == 401:
            print("\n>>> Received 401 Unauthorized. Your access token might be invalid or expired. "
                  "Try running authenticate.py again. <<<")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
    return None

if __name__ == "__main__":
    try:
        print("Attempting to fetch Upstox user profile...")
        token = load_access_token()
        get_user_profile(token)
        
    except (FileNotFoundError, ValueError) as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred in the main block: {e}")