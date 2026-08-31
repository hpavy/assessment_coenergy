import torch
import numpy as np

from coenergy.utils import load_config, set_seed
from coenergy.util_assessement import load_simulation_data
from coenergy.dataset import split_data
from coenergy.training import train_loop
from coenergy.model import load_model


if __name__ == "__main__":
    config_path = "config.yaml"
    config = load_config(config_path)
    set_seed(config.seed)

    data = load_simulation_data("data/")

    dataset_train, dataset_val, dataset_test = split_data(
        data, config.val_rate, config.test_rate, seed=config.seed
    )

    model = load_model(
        input_dim=dataset_train[0][0].flatten().shape[0],
        n_outputs=len(dataset_train[0][1]),
        config=config
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    try:
        model_dict = train_loop(model, dataset_train, dataset_val, optimizer, config)
    except KeyboardInterrupt:
        print("Training interrupted by user.")


    # Quick sanity check
    x, y = dataset_train[0]
    print(f"Train size : {len(dataset_train)}")
    print(f"Val size   : {len(dataset_val)}")
    print(f"Test size  : {len(dataset_test)}")
    print(f"x shape    : {x.shape}  (2 channels, T timesteps)")
    print(f"y shape    : {y.shape}  (Hb components + 3 CDT)")
