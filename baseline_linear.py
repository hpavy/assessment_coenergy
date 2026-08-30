"""Baseline 2: Linear Regression on piecewise time-series features.

Runs end-to-end: loads data, splits, extracts features, trains linear model,
and reports MAE in real physical units.
"""

from coenergy.utils import load_config, set_seed
from coenergy.util_assessement import load_simulation_data
from coenergy.dataset import split_data
from coenergy.baseline_linear import train_linear

if __name__ == "__main__":
    config = load_config("config.yaml")
    set_seed(config.seed)

    print("Loading data...")
    data = load_simulation_data("data/")

    print("Splitting data...")
    train_ds, val_ds, test_ds = split_data(
        data, config.val_rate, config.test_rate, seed=config.seed
    )

    print("\n--- Baseline 2: Linear Regression (10 parts × 4 stats) ---")
    metrics_val = train_linear(train_ds, val_ds, n_parts=10)
    metrics_test = train_linear(train_ds, test_ds, n_parts=10)

    print(f"\nValidation: {metrics_val}")
    print(f"Test:       {metrics_test}")
