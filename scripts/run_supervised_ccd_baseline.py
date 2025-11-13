from pathlib import Path
import sys

# Make project root importable when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_module import EEGCCDDataModule
from src.trainer import SupervisedRegressorTrainer, TrainingConfig

import torch
from braindecode.models import EEGNeX


def main() -> None:
    # 1) Data
    data_module = EEGCCDDataModule(
        cache_root=Path.home() / "eegdash_cache" / "eeg_challenge_cache",
        mini=True,
        batch_size=128,
        num_workers=4,
    )
    data_module.setup()
    train_loader, valid_loader, test_loader = data_module.get_dataloaders()

    # 2) Model (same as in original tutorial)
    model = EEGNeX(
        n_chans=129,
        n_outputs=1,
        n_times=200,
        sfreq=100,
    )

    # 3) Trainer
    config = TrainingConfig(
        lr=1e-3,
        weight_decay=1e-5,
        n_epochs=5,
        patience=5,
        min_delta=1e-4,
        use_scheduler=True,
    )
    trainer = SupervisedRegressorTrainer(model=model, config=config)
    trainer.fit(train_loader, valid_loader, print_batch_stats=True)

    # 4) Final evaluation and saving
    test_loss, test_rmse = trainer.evaluate(test_loader)
    print(f"Final test RMSE: {test_rmse:.6f}, test loss: {test_loss:.6f}")

    weights_fname = "outputs/weights_ccd_supervised_baseline.pt"
    trainer.save_best_weights("outputs/weights_ccd_supervised_baseline.pt")
    print(f"Model saved as {weights_fname}")


if __name__ == "__main__":
    main()