from torch.utils.data import Dataset
import torch
import numpy as np

from coenergy.util_assessement import ToSaveSimul

class QuentinDataset(Dataset):
    def __init__(
            self,
            HLC: torch.Tensor,
            CDT: torch.Tensor,
            indoor_temperature: torch.Tensor,
            indoor_U: torch.Tensor,
            Hb: torch.Tensor,
            ):
       self.HLC = HLC
       self.CDT = CDT
       self.indoor_temperature = indoor_temperature
       self.indoor_U = indoor_U
       self.Hb = Hb 

    def __len__(self) -> int:
        return self.HLC.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor]:
        return (
            self.HLC[i],
            self.CDT[i, :],
            self.indoor_temperature[i, :],
            self.indoor_U[i, :],
            self.Hb[i, :]
        )


def split_data(data: ToSaveSimul ,val_rate: float, test_rate: float) -> tuple[QuentinDataset]:
    nb_elem = data.HLC.shape[0]
    idxs = np.arange(nb_elem)
    np.random.shuffle(idxs)
    idxs_val = idxs[: int(nb_elem * val_rate)]
    idxs_test = idxs[int(nb_elem * val_rate): int(nb_elem*val_rate) + int(nb_elem*test_rate)]
    idxs_train = idxs[int(nb_elem*val_rate) + int(nb_elem*test_rate):]
    return (
        QuentinDataset(
            data.HLC[idxs_train],
            data.CDT[idxs_train],
            data.indoor_temperature[idxs_train],
            data.indoor_U[idxs_train],
            data.Hb[idxs_train]
            ),
        QuentinDataset(
            data.HLC[idxs_val],
            data.CDT[idxs_val],
            data.indoor_temperature[idxs_val],
            data.indoor_U[idxs_val],
            data.Hb[idxs_val]
            ),
        QuentinDataset(
            data.HLC[idxs_test],
            data.CDT[idxs_test],
            data.indoor_temperature[idxs_test],
            data.indoor_U[idxs_test],
            data.Hb[idxs_test]
            )
    )