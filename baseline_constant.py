"""Baseline 1: Constant predictor (mean of training targets).

Runs end-to-end: loads data, splits, predicts the training mean,
and reports MAE in real physical units.
"""

from coenergy.utils import load_config, set_seed
from coenergy.util_assessement import load_simulation_data
from coenergy.dataset import split_data
from coenergy.baseline_constant import predict_constant

if __name__ == "__main__":
    config = load_config("config.yaml")
    set_seed(config.seed)

    print("Loading data...")
    data = load_simulation_data("data/")

    print("Splitting data...")
    train_ds, val_ds, test_ds = split_data(
        data, config.val_rate, config.test_rate, seed=config.seed
    )

    print("\n--- Baseline: Constant Predictor (Mean) ---")
    metrics_val = predict_constant(train_ds, val_ds)
    metrics_test = predict_constant(train_ds, test_ds)

    print(f"\nValidation: {metrics_val}")
    print(f"Test:       {metrics_test}")
