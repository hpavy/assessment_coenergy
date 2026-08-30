import yaml
from box import Box
import numpy as np
import torch

def load_config(config_path) -> Box:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return Box(config)

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)