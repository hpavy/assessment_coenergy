import yaml
from box import Box

def load_config(config_path) -> Box:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return Box(config)