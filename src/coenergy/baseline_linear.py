"""Baseline 2: Linear Regression on piecewise time-series features.

Decomposes each time series into N equal parts, computes mean/std/max/min
for each part, and trains a plain LinearRegression model.
"""

import torch
import numpy as np
from sklearn.linear_model import LinearRegression

from coenergy.dataset import QuentinDataset
from coenergy.evaluate import evaluate, EvalMetrics


def extract_features(x: torch.Tensor, n_parts: int = 100) -> np.ndarray:
    """Extract piecewise statistical features from a time series.

    Args:
        x: [N, 2, T] — 2 channels (temperature, solicitation), T timesteps
        n_parts: Number of equal parts to split the time series into

    Returns:
        features: [N, 2 * n_parts * 4] — mean/std/max/min per part per channel
    """
    N, C, T = x.shape
    part_size = T // n_parts

    # Trim to exact multiple of n_parts, reshape to [N, C, n_parts, part_size]
    x_trimmed = x[:, :, : part_size * n_parts]
    x_parts = x_trimmed.reshape(N, C, n_parts, part_size)

    # Compute stats along the time dimension (dim=3)
    means = x_parts.mean(dim=3)          # [N, C, n_parts]
    stds = x_parts.std(dim=3)            # [N, C, n_parts]
    maxs = x_parts.amax(dim=3)           # [N, C, n_parts]
    mins = x_parts.amin(dim=3)           # [N, C, n_parts]

    # Stack stats: [N, C, n_parts, 4] then flatten to [N, C * n_parts * 4]
    features = torch.stack([means, stds, maxs, mins], dim=-1)
    features = features.reshape(N, -1)

    return features.numpy()


def _dataset_to_xy(ds: QuentinDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert dataset to (x, y) tensors in normalized space."""
    # x: [N, 2, T]
    x = torch.stack([ds.indoor_temperature, ds.indoor_U], dim=1)
    # y: [N, n_hb + 3]
    y = torch.cat([ds.Hb, ds.CDT], dim=1)
    return x, y


def train_linear(
    train_ds: QuentinDataset,
    eval_ds: QuentinDataset,
    n_parts: int = 100,
) -> EvalMetrics:
    """Train a linear regression on piecewise features and evaluate.

    Args:
        train_ds: Training dataset
        eval_ds: Evaluation dataset (val or test)
        n_parts: Number of equal parts to split each time series into

    Returns:
        EvalMetrics with MAE in real physical units
    """
    # Extract features and targets
    x_train, y_train = _dataset_to_xy(train_ds)
    x_eval, y_eval = _dataset_to_xy(eval_ds)

    X_train = extract_features(x_train, n_parts=n_parts)
    X_eval = extract_features(x_eval, n_parts=n_parts)

    # Train linear regression (targets are already normalized)
    reg = LinearRegression()
    reg.fit(X_train, y_train.numpy())

    # Predict (in normalized space)
    y_pred_norm = torch.from_numpy(reg.predict(X_eval))

    # Evaluate in real physical units
    n_hb = train_ds.Hb.shape[1]
    return evaluate(y_pred_norm, y_eval, train_ds.stats, n_hb)
