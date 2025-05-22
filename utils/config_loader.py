import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')

def load_config(path=CONFIG_PATH):
    """
    Loads the YAML configuration file and returns it as a dictionary.
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    config = load_config()
    print("Loaded config:")
    print(config)
