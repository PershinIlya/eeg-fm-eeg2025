from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import copy
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """Configuration of core hyperparameters for supervised regression training.

    Includes learning rate, weight decay, number of epochs, early stopping
    patience and minimum delta, and whether to use a learning rate scheduler.
    """
    lr: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 5
    patience: int = 5
    min_delta: float = 1e-4
    use_scheduler: bool = True


@dataclass
class EpochResult:
    """Stores training and validation loss and RMSE for a single epoch.

    Used for logging and analysis of training progress.
    """
    epoch: int
    train_loss: float
    train_rmse: float
    val_loss: float
    val_rmse: float


class SupervisedRegressorTrainer:
    """Trainer for supervised regression models with MSE loss and RMSE metric.

    This class supports:
    - Training a regression model using mean squared error loss.
    - Computing RMSE as the primary evaluation metric.
    - Optional cosine annealing learning rate scheduler.
    - Simple early stopping based on validation RMSE improvements.
    - Tracking and restoring the best model weights found during training.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        """Initialize the trainer with model, device, and training configuration.

        Args:
            model: The PyTorch model to train.
            device: Optional device string ('cuda' or 'cpu'). Defaults to CUDA if available.
            config: Optional TrainingConfig instance specifying hyperparameters.

        Internally creates:
            - MSE loss function.
            - AdamW optimizer with specified learning rate and weight decay.
            - Optional cosine annealing LR scheduler.
            - History list to store per-epoch results.
            - Variables to track best model state and best validation RMSE.
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.config = config or TrainingConfig()
        self.loss_fn = nn.MSELoss()

        self.optimizer: Optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler: Optional[LRScheduler] = None
        if self.config.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(self.config.n_epochs - 1, 1)
            )

        self.history: List[EpochResult] = []
        self.best_state: Optional[Dict[str, Any]] = None
        self.best_rmse: float = float("inf")
        self.best_epoch: Optional[int] = None

    # ---- Public API -----------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        print_batch_stats: bool = True,
    ) -> List[EpochResult]:
        """Run the full training loop over multiple epochs with validation.

        At each epoch:
            - Train on the training set.
            - Evaluate on the validation set.
            - Apply early stopping based on validation RMSE improvements.
            - Optionally print batch-level stats.

        Args:
            train_loader: DataLoader for training data.
            valid_loader: DataLoader for validation data.
            print_batch_stats: Whether to display batch-level progress bars.

        Returns:
            List of EpochResult instances summarizing training and validation metrics.
        """
        epochs_no_improve = 0

        for epoch in range(1, self.config.n_epochs + 1):
            print(f"Epoch {epoch}/{self.config.n_epochs}: ", end="")

            train_loss, train_rmse = self._train_one_epoch(
                train_loader, epoch, print_batch_stats=print_batch_stats
            )
            val_loss, val_rmse = self._evaluate(
                valid_loader, print_batch_stats=print_batch_stats
            )

            print(
                f"Train RMSE: {train_rmse:.6f}, "
                f"Average Train Loss: {train_loss:.6f}, "
                f"Val RMSE: {val_rmse:.6f}, "
                f"Average Val Loss: {val_loss:.6f}"
            )

            self.history.append(
                EpochResult(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_rmse=train_rmse,
                    val_loss=val_loss,
                    val_rmse=val_rmse,
                )
            )

            if val_rmse < self.best_rmse - self.config.min_delta:
                self.best_rmse = val_rmse
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config.patience:
                    print(
                        f"Early stopping at epoch {epoch}. "
                        f"Best Val RMSE: {self.best_rmse:.6f} (epoch {self.best_epoch})"
                    )
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return self.history

    @torch.no_grad()
    def evaluate(
        self, dataloader: DataLoader, print_batch_stats: bool = True
    ) -> Tuple[float, float]:
        """Run a validation loop over the given dataloader and compute loss and RMSE.

        Args:
            dataloader: DataLoader for validation or test data.
            print_batch_stats: Whether to display batch-level progress bars.

        Returns:
            Tuple of (average loss, RMSE) over the entire dataset.
        """
        return self._evaluate(dataloader, print_batch_stats=print_batch_stats)

    def save_best_weights(self, path: str) -> None:
        """Save the best model weights found during training to disk.

        Args:
            path: File path where the model weights will be saved.

        Raises:
            RuntimeError: If no best state is available (e.g., fit() was not called).
        """
        if self.best_state is None:
            raise RuntimeError("No best state saved. Call fit() first.")
        torch.save(self.best_state, path)

    # ---- Internal helpers -----------------------------------------------

    def _train_one_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        print_batch_stats: bool = True,
    ) -> Tuple[float, float]:
        """Perform one full training epoch over the provided dataloader.

        Args:
            dataloader: DataLoader for training data.
            epoch: Current epoch number (for logging).
            print_batch_stats: Whether to display batch-level progress bars.

        Returns:
            Tuple of (average training loss, training RMSE) over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        sum_sq_err = 0.0
        n_samples = 0

        # Use tqdm for optional batch-level progress reporting
        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=not print_batch_stats,
        )

        for batch_idx, batch in progress_bar:
            X, y = batch[0], batch[1]
            X = X.to(self.device).float()
            y = y.to(self.device).float()

            self.optimizer.zero_grad(set_to_none=True)
            preds = self.model(X)
            loss = self.loss_fn(preds, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Accumulate RMSE in streaming fashion over all samples in the epoch
            preds_flat = preds.detach().view(-1)
            y_flat = y.detach().view(-1)
            sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
            n_samples += y_flat.numel()

            if print_batch_stats:
                running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
                progress_bar.set_description(
                    f"Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, "
                    f"Loss: {loss.item():.6f}, RMSE: {running_rmse:.6f}"
                )

        # Update learning rate schedule once per epoch if scheduler is enabled
        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / len(dataloader)
        rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
        return avg_loss, rmse

    @torch.no_grad()
    def _evaluate(
        self,
        dataloader: DataLoader,
        print_batch_stats: bool = True,
    ) -> Tuple[float, float]:
        """Evaluate the model on the given dataloader without gradient updates.

        Mirrors the training loop but disables gradient computations.
        Aggregates loss and RMSE over the entire validation set.
        Optionally logs batch-wise progress with tqdm.

        Args:
            dataloader: DataLoader for validation or test data.
            print_batch_stats: Whether to display batch-level progress bars.

        Returns:
            Tuple of (average loss, RMSE) over the entire dataset.
        """
        self.model.eval()

        total_loss = 0.0
        sum_sq_err = 0.0
        n_batches = len(dataloader)
        n_samples = 0

        # Optional progress bar for validation batch-wise reporting
        iterator = tqdm(
            enumerate(dataloader),
            total=n_batches,
            disable=not print_batch_stats,
        )

        for batch_idx, batch in iterator:
            X, y = batch[0], batch[1]
            X = X.to(self.device).float()
            y = y.to(self.device).float()

            preds = self.model(X)
            batch_loss = self.loss_fn(preds, y).item()
            total_loss += batch_loss

            # Compute RMSE across the whole validation set by accumulating squared errors
            preds_flat = preds.detach().view(-1)
            y_flat = y.detach().view(-1)
            sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
            n_samples += y_flat.numel()

            if print_batch_stats:
                running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
                iterator.set_description(
                    f"Val Batch {batch_idx + 1}/{n_batches}, "
                    f"Loss: {batch_loss:.6f}, RMSE: {running_rmse:.6f}"
                )

        avg_loss = total_loss / n_batches if n_batches else float("nan")
        rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5

        print(f"Val RMSE: {rmse:.6f}, Val Loss: {avg_loss:.6f}\n")
        return avg_loss, rmse