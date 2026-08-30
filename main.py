import torch
import numpy as np

from coenergy.utils import load_config, set_seed
from coenergy.util_assessement import load_simulation_data
from coenergy.dataset import split_data


if __name__ == "__main__":
    config_path = "config.yaml"
    config = load_config(config_path)
    set_seed(config.seed)

    data = load_simulation_data("data/")

    dataset_train, dataset_val, dataset_test = split_data(data, config.val_rate, config.test_rate)

    print("piche")