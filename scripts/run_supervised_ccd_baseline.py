from pathlib import Path

import torch
from braindecode.models import EEGNeX

from data_module import EEGCCDDataModule
from trainer import SupervisedRegressorTrainer, TrainingConfig


def main() -> None:
    # 1) Data
    data_module = EEGCCDDataModule(
        cache_root=Path.home() / "eegdash_cache" / "eeg_challenge_cache",
        mini=True,
        batch_size=128,
        num_workers=0,
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

    trainer.save_best_weights("weights_challenge_1.pt")
    print("Model saved as 'weights_challenge_1.pt'")


if __name__ == "__main__":
    main()