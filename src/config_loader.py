import yaml
from pathlib import Path

def load_config():
    # Finds the root folder regardless of where the script is run
    project_root = Path(__file__).resolve().parents[1] 
    config_path = project_root / "config.yaml"
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    return config