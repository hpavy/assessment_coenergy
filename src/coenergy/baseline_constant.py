"""Baseline 1: Constant predictor (mean of training targets).

Predicts the mean of the training targets for all samples in val/test.
This establishes the floor — if a model can't beat this, it's useless.
"""

import torch

from coenergy.dataset import QuentinDataset
from coenergy.evaluate import evaluate, EvalMetrics


def predict_constant(train_ds: QuentinDataset, eval_ds: QuentinDataset) -> EvalMetrics:
    """Predict the training mean for all samples in eval_ds and return metrics."""
    # The mean of the normalized training targets
    y_mean_norm = torch.cat([train_ds.Hb, train_ds.CDT], dim=1).mean(dim=0)

    # Repeat for all samples in the eval set
    y_pred_norm = y_mean_norm.unsqueeze(0).repeat(len(eval_ds), 1)

    # Ground truth (normalized)
    y_true_norm = torch.cat([eval_ds.Hb, eval_ds.CDT], dim=1)

    # Number of Hb components
    n_hb = train_ds.Hb.shape[1]

    # Evaluate in real physical units
    return evaluate(y_pred_norm, y_true_norm, train_ds.stats, n_hb)
