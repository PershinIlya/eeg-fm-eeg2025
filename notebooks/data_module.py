from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events,
)
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    keep_only_recordings_with,
    add_extras_columns,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state


class EEGCCDDataModule:
    """Data loading, preprocessing and split for EEG Challenge 1 (CCD RT regression)."""

    def __init__(
        self,
        cache_root: Path | str | None = None,
        release: str = "R5",
        mini: bool = True,
        batch_size: int = 128,
        num_workers: int = 0,
        valid_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 2025,
        sfreq: int = 100,
        epoch_len_s: float = 2.0,
        shift_after_stim: float = 0.5,
        window_len_s: float = 2.0,
    ) -> None:
        # Dataset / windowing config
        if cache_root is None:
            cache_root = Path.home() / "eegdash_cache" / "eeg_challenge_cache"
        self.cache_root = Path(cache_root).resolve()
        self.release = release
        self.mini = mini

        self.sfreq = sfreq
        self.epoch_len_s = epoch_len_s
        self.shift_after_stim = shift_after_stim
        self.window_len_s = window_len_s

        # Split / loader config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_frac = valid_frac
        self.test_frac = test_frac
        self.seed = seed

        # Internal attributes populated by setup()
        self.dataset_ccd: BaseConcatDataset | None = None
        self.single_windows: BaseConcatDataset | None = None
        self.train_set: BaseConcatDataset | None = None
        self.valid_set: BaseConcatDataset | None = None
        self.test_set: BaseConcatDataset | None = None

        self.train_loader: DataLoader | None = None
        self.valid_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

    # ---- Public API -----------------------------------------------------

    def setup(self) -> None:
        """Download/load dataset, create windows, split, and build loaders."""
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._load_raw_dataset()
        self._create_windows()
        self._split_by_subject()
        self._create_dataloaders()

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        if self.train_loader is None or self.valid_loader is None or self.test_loader is None:
            raise RuntimeError("Call .setup() before requesting dataloaders.")
        return self.train_loader, self.valid_loader, self.test_loader

    # ---- Internal helpers -----------------------------------------------

    def _load_raw_dataset(self) -> None:
        self.dataset_ccd = EEGChallengeDataset(
            task="contrastChangeDetection",
            release=self.release,
            cache_dir=self.cache_root,
            mini=self.mini,
        )

    def _create_windows(self) -> None:
        assert self.dataset_ccd is not None

        epoch_len_s = self.epoch_len_s
        sfreq = self.sfreq

        transformations = [
            Preprocessor(
                annotate_trials_with_target,
                target_field="rt_from_stimulus",
                epoch_length=epoch_len_s,
                require_stimulus=True,
                require_response=True,
                apply_on_array=False,
            ),
            Preprocessor(add_aux_anchors, apply_on_array=False),
        ]
        preprocess(self.dataset_ccd, transformations, n_jobs=1)

        anchor = "stimulus_anchor"
        shift_after_stim = self.shift_after_stim
        window_len = self.window_len_s

        dataset = keep_only_recordings_with(anchor, self.dataset_ccd)

        single_windows = create_windows_from_events(
            dataset,
            mapping={anchor: 0},
            trial_start_offset_samples=int(shift_after_stim * sfreq),
            trial_stop_offset_samples=int((shift_after_stim + window_len) * sfreq),
            window_size_samples=int(epoch_len_s * sfreq),
            window_stride_samples=sfreq,
            preload=True,
        )

        single_windows = add_extras_columns(
            single_windows,
            dataset,
            desc=anchor,
            keys=(
                "target",
                "rt_from_stimulus",
                "rt_from_trialstart",
                "stimulus_onset",
                "response_onset",
                "correct",
                "response_type",
            ),
        )

        self.single_windows = single_windows

    def _split_by_subject(self) -> None:
        assert self.single_windows is not None

        meta_information = self.single_windows.get_metadata()

        valid_frac = self.valid_frac
        test_frac = self.test_frac
        seed = self.seed

        subjects = meta_information["subject"].unique()

        train_subj, valid_test_subject = train_test_split(
            subjects,
            test_size=(valid_frac + test_frac),
            random_state=check_random_state(seed),
            shuffle=True,
        )
        valid_subj, test_subj = train_test_split(
            valid_test_subject,
            test_size=test_frac,
            random_state=check_random_state(seed + 1),
            shuffle=True,
        )

        subject_split = self.single_windows.split("subject")
        train_set_list = []
        valid_set_list = []
        test_set_list = []

        for subj_id, ds in subject_split.items():
            if subj_id in train_subj:
                train_set_list.append(ds)
            elif subj_id in valid_subj:
                valid_set_list.append(ds)
            elif subj_id in test_subj:
                test_set_list.append(ds)

        self.train_set = BaseConcatDataset(train_set_list)
        self.valid_set = BaseConcatDataset(valid_set_list)
        self.test_set = BaseConcatDataset(test_set_list)

    def _create_dataloaders(self) -> None:
        assert self.train_set is not None
        assert self.valid_set is not None
        assert self.test_set is not None

        bs = self.batch_size
        num_workers = self.num_workers

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
        )
        self.valid_loader = DataLoader(
            self.valid_set,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
        )
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
        )