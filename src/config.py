# config_loader.py
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Load config ONCE when the module is imported
CONFIG = load_config()
