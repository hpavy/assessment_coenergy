"""Evaluation metrics for the coEnergy assessment.

All metrics are computed in real physical units (denormalized).
- Hb and HLC are reported in W/K.
- CDT (τ_b, τ_n, τ_inf) is reported in hours.
- R² is computed per target (1.0 = perfect, 0.0 = mean predictor, <0 = worse than mean).
"""

from dataclasses import dataclass
import torch

from coenergy.dataset import NormStats


@dataclass
class EvalMetrics:
    """Holds MAE and R² metrics in real physical units."""
    mae_hlc: float          # W/K  (derived: HLC = Σ Hb)
    mae_hb_avg: float       # W/K  (average across components)
    mae_tau_b: float         # hours
    mae_tau_n: float         # hours
    mae_tau_inf: float       # hours
    r2_hlc: float            # R² for HLC
    r2_hb_avg: float         # R² averaged across Hb components
    r2_tau_b: float          # R² for τ_b
    r2_tau_n: float          # R² for τ_n
    r2_tau_inf: float        # R² for τ_inf

    def as_dict(self) -> dict:
        return {
            "MAE HLC (W/K)": self.mae_hlc,
            "MAE Hb avg (W/K)": self.mae_hb_avg,
            "MAE τ_b (h)": self.mae_tau_b,
            "MAE τ_n (h)": self.mae_tau_n,
            "MAE τ_inf (h)": self.mae_tau_inf,
            "R² HLC": self.r2_hlc,
            "R² Hb avg": self.r2_hb_avg,
            "R² τ_b": self.r2_tau_b,
            "R² τ_n": self.r2_tau_n,
            "R² τ_inf": self.r2_tau_inf,
        }

    def __str__(self) -> str:
        return (
            f"MAE — HLC: {self.mae_hlc:.3f} W/K | Hb: {self.mae_hb_avg:.3f} W/K | "
            f"τ_b: {self.mae_tau_b:.3f}h | τ_n: {self.mae_tau_n:.3f}h | τ_inf: {self.mae_tau_inf:.3f}h\n"
            f"R²  — HLC: {self.r2_hlc:.3f} | Hb: {self.r2_hb_avg:.3f} | "
            f"τ_b: {self.r2_tau_b:.3f} | τ_n: {self.r2_tau_n:.3f} | τ_inf: {self.r2_tau_inf:.3f}"
        )


def _mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error, returns a scalar tensor."""
    return torch.abs(pred - target).mean()


def _r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """R² score (coefficient of determination).

    R² = 1 - SS_res / SS_tot
    - 1.0 = perfect prediction
    - 0.0 = predicting the mean
    - <0  = worse than the mean
    """
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def evaluate(
    y_pred_norm: torch.Tensor,
    y_true_norm: torch.Tensor,
    stats: NormStats,
    n_hb_components: int,
) -> EvalMetrics:
    """Evaluate predictions in real physical units.

    Args:
        y_pred_norm: Normalized predictions [N, n_hb + 3]
        y_true_norm: Normalized ground truth  [N, n_hb + 3]
        stats:       Normalization statistics from the training split
        n_hb_components: Number of Hb components (to split the vector)

    Returns:
        EvalMetrics with MAE and R² in real units (W/K and hours)
    """
    # Split normalized predictions and targets into Hb and CDT
    Hb_pred_norm = y_pred_norm[:, :n_hb_components]
    CDT_pred_norm = y_pred_norm[:, n_hb_components:]

    Hb_true_norm = y_true_norm[:, :n_hb_components]
    CDT_true_norm = y_true_norm[:, n_hb_components:]

    # Denormalize back to real physical units
    Hb_pred = Hb_pred_norm * stats.Hb_std + stats.Hb_mean
    Hb_true = Hb_true_norm * stats.Hb_std + stats.Hb_mean

    CDT_pred = CDT_pred_norm * stats.CDT_std + stats.CDT_mean
    CDT_true = CDT_true_norm * stats.CDT_std + stats.CDT_mean

    # Derive HLC = Σ Hb (the physical constraint)
    HLC_pred = Hb_pred.sum(dim=1)
    HLC_true = Hb_true.sum(dim=1)

    # Compute MAE in real units
    mae_hlc = _mae(HLC_pred, HLC_true).item()
    mae_hb_avg = _mae(Hb_pred, Hb_true).item()
    mae_tau_b = _mae(CDT_pred[:, 0], CDT_true[:, 0]).item()
    mae_tau_n = _mae(CDT_pred[:, 1], CDT_true[:, 1]).item()
    mae_tau_inf = _mae(CDT_pred[:, 2], CDT_true[:, 2]).item()

    # Compute R² in real units
    r2_hlc = _r2(HLC_pred, HLC_true).item()
    r2_hb_avg = _r2(Hb_pred, Hb_true).mean().item()  # average R² across components
    r2_tau_b = _r2(CDT_pred[:, 0], CDT_true[:, 0]).item()
    r2_tau_n = _r2(CDT_pred[:, 1], CDT_true[:, 1]).item()
    r2_tau_inf = _r2(CDT_pred[:, 2], CDT_true[:, 2]).item()

    return EvalMetrics(
        mae_hlc=mae_hlc,
        mae_hb_avg=mae_hb_avg,
        mae_tau_b=mae_tau_b,
        mae_tau_n=mae_tau_n,
        mae_tau_inf=mae_tau_inf,
        r2_hlc=r2_hlc,
        r2_hb_avg=r2_hb_avg,
        r2_tau_b=r2_tau_b,
        r2_tau_n=r2_tau_n,
        r2_tau_inf=r2_tau_inf,
    )
