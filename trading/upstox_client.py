import requests
from utils.config_loader import load_config

class UpstoxV2Client:
    BASE_URL = "https://api.upstox.com/v2"

    def __init__(self, config=None):
        if config is None:
            config = load_config()
        upstox_cfg = config['upstox']
        self.api_key = upstox_cfg['api_key']
        self.api_secret = upstox_cfg['api_secret']
        self.redirect_uri = upstox_cfg['redirect_uri']
        self.access_token = upstox_cfg['access_token']

    def get_headers(self):
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }

    def get_profile(self):
        url = f"{self.BASE_URL}/user/profile"
        response = requests.get(url, headers=self.get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} {response.text}")

if __name__ == "__main__":
    client = UpstoxV2Client()
    try:
        profile = client.get_profile()
        print("Upstox API v2 connection successful! Profile:")
        print(profile)
    except Exception as e:
        print("Failed to connect to Upstox API v2:", e)
