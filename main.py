# EdgeRunner - Main Entry Point

from utils.config_loader import load_config

if __name__ == "__main__":
    print("Welcome to EdgeRunner - Your Personal Intraday Trading Bot!")
    config = load_config()
    print("Loaded config:")
    print(config)
    # TODO: Initialize modules, and start the workflow
