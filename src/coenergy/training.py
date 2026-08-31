import copy

from box import Box
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from coenergy.evaluate import evaluate


def train_loop(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    optimizer: torch.optim.Optimizer,
    config: Box,
) -> dict:
    """Run the training loop and return history + final metrics.

    Args:
        model: The neural network model
        train_ds: Training dataset
        val_ds: Validation dataset
        optimizer: Initialized optimizer
        config: Configuration object with training params

    Returns:
        best_weights: the best weights of the model
        dict: Contains 'history' (train/val loss per epoch) and 'metrics' (final EvalMetrics)
    """
    device = config.device
    model = model.to(device)
    criterion = nn.MSELoss()


    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(config.epochs):

        # --- Training phase ---
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_ds)

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)

        val_loss /= len(val_ds)

        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            print(f"  -> New best model saved (val_loss: {val_loss:.4f})")


        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch + 1}/{config.epochs} — train: {train_loss:.4f} | val: {val_loss:.4f}")

    return best_weights, history
