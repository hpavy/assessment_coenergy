"""Baseline 2: Linear Regression on piecewise time-series features.

Best configuration found via auto-research (25 experiments):
- 100 parts, stats = [mean, std, slope], plain LinearRegression
- 600 features per sample, ~22s training on 500k samples

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

    print("\n--- Baseline 2: Linear Regression (100 parts × mean+std+slope) ---")
    metrics_val = train_linear(
        train_ds, val_ds,
        n_parts=100,
        stats=["mean", "std", "slope"],
        add_global=False,
        model_type="linear",
    )
    metrics_test = train_linear(
        train_ds, test_ds,
        n_parts=100,
        stats=["mean", "std", "slope"],
        add_global=False,
        model_type="linear",
    )

    print(f"\nValidation: {metrics_val}")
    print(f"Test:       {metrics_test}")
