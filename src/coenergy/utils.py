import json
import yaml
from pathlib import Path
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

def save_results(losses: dict, metrics, output_dir: str = "output") -> None:
    """Save training losses and final metrics to a JSON file in the output folder."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=False)

    # Convert metrics object to dictionary if it has as_dict method
    if hasattr(metrics, 'as_dict'):
        metrics_dict = metrics.as_dict()
    elif hasattr(metrics, '__dict__'):
        metrics_dict = metrics.__dict__
    else:
        metrics_dict = metrics

    results = {
        "losses": losses,
        "final_metrics": metrics_dict
    }

    file_path = out_path / "results.json"
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {file_path}")