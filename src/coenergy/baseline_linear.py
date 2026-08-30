"""Baseline 2: Linear Regression on piecewise time-series features.

Decomposes each time series into N equal parts, computes configurable stats
for each part, and trains a linear model.
"""

import torch
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from typing import List, Optional

from coenergy.dataset import QuentinDataset
from coenergy.evaluate import evaluate, EvalMetrics


def extract_features(
    x: torch.Tensor,
    n_parts: int = 10,
    stats: List[str] = None,
    add_global: bool = False,
) -> np.ndarray:
    """Extract piecewise statistical features from a time series.

    Args:
        x: [N, 2, T] — 2 channels (temperature, solicitation), T timesteps
        n_parts: Number of equal parts to split the time series into
        stats: List of stats to compute per part.
               Options: "mean", "std", "max", "min", "median", "range", "slope"
        add_global: If True, append global stats (mean, std, max, min) per channel

    Returns:
        features: [N, feature_dim] numpy array
    """
    if stats is None:
        stats = ["mean", "std", "max", "min"]

    N, C, T = x.shape
    part_size = T // n_parts

    # Trim to exact multiple of n_parts, reshape to [N, C, n_parts, part_size]
    x_trimmed = x[:, :, : part_size * n_parts]
    x_parts = x_trimmed.reshape(N, C, n_parts, part_size)

    # Time axis for slope computation (normalized 0..1 within each part)
    t_axis = torch.linspace(0, 1, part_size, device=x.device)

    feature_list = []

    for stat in stats:
        if stat == "mean":
            feature_list.append(x_parts.mean(dim=3))          # [N, C, n_parts]
        elif stat == "std":
            feature_list.append(x_parts.std(dim=3))
        elif stat == "max":
            feature_list.append(x_parts.amax(dim=3))
        elif stat == "min":
            feature_list.append(x_parts.amin(dim=3))
        elif stat == "median":
            feature_list.append(x_parts.median(dim=3).values)
        elif stat == "range":
            feature_list.append(x_parts.amax(dim=3) - x_parts.amin(dim=3))
        elif stat == "slope":
            # Linear slope within each part: cov(t, x) / var(t)
            t_mean = t_axis.mean()
            t_centered = t_axis - t_mean
            t_var = (t_centered ** 2).sum()

            x_mean = x_parts.mean(dim=3, keepdim=True)
            x_centered = x_parts - x_mean

            # cov(t, x) = sum(t_c * x_c) / (part_size - 1)
            # slope = cov / var(t)
            cov = (t_centered.unsqueeze(0).unsqueeze(0).unsqueeze(0) * x_centered).sum(dim=3)
            slope = cov / t_var
            feature_list.append(slope)
        else:
            raise ValueError(f"Unknown stat: {stat}")

    # Stack all stats: [N, C, n_parts, n_stats] then flatten
    features = torch.stack(feature_list, dim=-1)  # [N, C, n_parts, n_stats]
    features = features.reshape(N, -1)             # [N, C * n_parts * n_stats]

    # Optional global features
    if add_global:
        global_means = x.mean(dim=2)   # [N, C]
        global_stds = x.std(dim=2)
        global_maxs = x.amax(dim=2)
        global_mins = x.amin(dim=2)
        global_features = torch.stack(
            [global_means, global_stds, global_maxs, global_mins], dim=-1
        ).reshape(N, -1)  # [N, C * 4]
        features = torch.cat([features, global_features], dim=1)

    return features.numpy()


def _dataset_to_xy(ds: QuentinDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert dataset to (x, y) tensors in normalized space."""
    N = len(ds)
    x = torch.stack([ds.indoor_temperature, ds.indoor_U], dim=1)  # [N, 2, T]
    y = torch.cat([ds.Hb, ds.CDT], dim=1)                          # [N, n_hb + 3]
    return x, y


def train_linear(
    train_ds: QuentinDataset,
    eval_ds: QuentinDataset,
    n_parts: int = 10,
    stats: List[str] = None,
    add_global: bool = False,
    model_type: str = "linear",
    alpha: float = 1.0,
) -> EvalMetrics:
    """Train a linear model on piecewise features and evaluate.

    Args:
        train_ds: Training dataset
        eval_ds: Evaluation dataset (val or test)
        n_parts: Number of equal parts to split each time series into
        stats: List of stats to compute per part
        add_global: If True, append global stats per channel
        model_type: "linear", "ridge", or "lasso"
        alpha: Regularization strength (for ridge/lasso)

    Returns:
        EvalMetrics with MAE in real physical units
    """
    if stats is None:
        stats = ["mean", "std", "max", "min"]

    # Extract features and targets
    x_train, y_train = _dataset_to_xy(train_ds)
    x_eval, y_eval = _dataset_to_xy(eval_ds)

    X_train = extract_features(x_train, n_parts=n_parts, stats=stats, add_global=add_global)
    X_eval = extract_features(x_eval, n_parts=n_parts, stats=stats, add_global=add_global)

    # Select model
    if model_type == "linear":
        reg = LinearRegression()
    elif model_type == "ridge":
        reg = Ridge(alpha=alpha)
    elif model_type == "lasso":
        reg = Lasso(alpha=alpha, max_iter=5000)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Train (targets are already normalized)
    reg.fit(X_train, y_train.numpy())

    # Predict (in normalized space)
    y_pred_norm = torch.from_numpy(reg.predict(X_eval))

    # Evaluate in real physical units
    n_hb = train_ds.Hb.shape[1]
    return evaluate(y_pred_norm, y_eval, train_ds.stats, n_hb)
