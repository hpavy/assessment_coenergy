from dataclasses import dataclass
from torch.utils.data import Dataset
import torch
import numpy as np

from coenergy.util_assessement import ToSaveSimul


@dataclass
class NormStats:
    """Z-normalization statistics computed from the training split."""
    temp_mean: torch.Tensor       # scalar (global mean for temperature)
    temp_std: torch.Tensor        # scalar
    U_mean: torch.Tensor          # scalar (global mean for solicitation)
    U_std: torch.Tensor           # scalar
    Hb_mean: torch.Tensor         # [n_components]
    Hb_std: torch.Tensor          # [n_components]
    CDT_mean: torch.Tensor        # [3]
    CDT_std: torch.Tensor         # [3]


class QuentinDataset(Dataset):
    """Dataset for Quentin House simulations.

    Inputs  : indoor_temperature [T], indoor_U [T]
    Targets : Hb [n_components], CDT [3]
    Derived : HLC = Σ Hb  (computed at inference, not stored)
    """

    def __init__(
        self,
        indoor_temperature: torch.Tensor,
        indoor_U: torch.Tensor,
        Hb: torch.Tensor,
        CDT: torch.Tensor,
        stats: NormStats,
    ):
        self.indoor_temperature = (indoor_temperature - stats.temp_mean) / stats.temp_std
        self.indoor_U = (indoor_U - stats.U_mean) / stats.U_std
        self.Hb = (Hb - stats.Hb_mean) / stats.Hb_std
        self.CDT = (CDT - stats.CDT_mean) / stats.CDT_std
        self.stats = stats

    def __len__(self) -> int:
        return self.Hb.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack([self.indoor_temperature[i], self.indoor_U[i]])
        y = torch.cat([self.Hb[i], self.CDT[i]])
        return x, y


def _compute_stats(
    indoor_temperature: torch.Tensor,
    indoor_U: torch.Tensor,
    Hb: torch.Tensor,
    CDT: torch.Tensor,
) -> NormStats:
    """Compute Z-normalization statistics from a training split."""
    return NormStats(
        temp_mean=indoor_temperature.mean(),
        temp_std=indoor_temperature.std(),
        U_mean=indoor_U.mean(),
        U_std=indoor_U.std(),
        Hb_mean=Hb.mean(dim=0),
        Hb_std=Hb.std(dim=0),
        CDT_mean=CDT.mean(dim=0),
        CDT_std=CDT.std(dim=0),
    )


def split_data(
    data: ToSaveSimul,
    val_rate: float,
    test_rate: float,
    seed: int = 42,
) -> tuple[QuentinDataset, QuentinDataset, QuentinDataset]:
    """Split data into train/val/test and normalize using train statistics only."""
    nb_elem = data.HLC.shape[0]
    idxs = torch.randperm(nb_elem, generator=torch.Generator().manual_seed(seed))

    n_val = int(nb_elem * val_rate)
    n_test = int(nb_elem * test_rate)
    n_train = nb_elem - n_val - n_test

    idxs_train = idxs[:n_train]
    idxs_val = idxs[n_train : n_train + n_val]
    idxs_test = idxs[n_train + n_val :]

    # Compute normalization stats from training split ONLY
    stats = _compute_stats(
        indoor_temperature=data.indoor_temperature[idxs_train],
        indoor_U=data.indoor_U[idxs_train],
        Hb=data.Hb[idxs_train],
        CDT=data.CDT[idxs_train],
    )

    train_ds = QuentinDataset(
        indoor_temperature=data.indoor_temperature[idxs_train],
        indoor_U=data.indoor_U[idxs_train],
        Hb=data.Hb[idxs_train],
        CDT=data.CDT[idxs_train],
        stats=stats,
    )
    val_ds = QuentinDataset(
        indoor_temperature=data.indoor_temperature[idxs_val],
        indoor_U=data.indoor_U[idxs_val],
        Hb=data.Hb[idxs_val],
        CDT=data.CDT[idxs_val],
        stats=stats,
    )
    test_ds = QuentinDataset(
        indoor_temperature=data.indoor_temperature[idxs_test],
        indoor_U=data.indoor_U[idxs_test],
        Hb=data.Hb[idxs_test],
        CDT=data.CDT[idxs_test],
        stats=stats,
    )

    return train_ds, val_ds, test_ds
