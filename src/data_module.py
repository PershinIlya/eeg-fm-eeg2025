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
    """
    Data loading, preprocessing, window creation, and subject-wise splitting for EEG Challenge 1.

    This class wraps the entire data preparation pipeline tailored specifically for the EEG Challenge 1 task,
    which involves contrast change detection (CCD) and reaction time regression. It handles downloading
    or loading the raw dataset, applying preprocessing and windowing steps to create EEG windows aligned
    to stimulus events, and performs subject-wise splitting to avoid data leakage.

    The class exposes a simple public API consisting of `setup()` to prepare all data components, and
    `get_dataloaders()` to retrieve train, validation, and test dataloaders ready for model training and evaluation.
    """

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
        """
        Initialize the EEGCCDDataModule with configuration parameters.

        The constructor sets up dataset and release options, windowing parameters, and split/loader configurations.
        - Dataset/release options: control where data is cached, which release version to use, and whether to use a mini subset.
        - Windowing parameters: determine EEG sampling frequency, epoch length, temporal shifts, and window lengths for segment extraction.
        - Split and loader parameters: configure batch size, number of workers, validation and test fractions, and random seed for reproducibility.

        These parameters govern how EEG windows are created and how data splits are performed for training, validation, and testing.
        """
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
        """
        Prepare the dataset for training and evaluation.

        This is the single entry point to fully prepare the data. It orchestrates loading or downloading the raw dataset,
        applying preprocessing and windowing steps, performing subject-wise splitting into train/validation/test sets,
        and constructing PyTorch dataloaders. This method should be called once before starting model training or evaluation.
        """
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._load_raw_dataset()
        self._create_windows()
        self._split_by_subject()
        self._create_dataloaders()

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Return the prepared dataloaders for training, validation, and testing.

        The returned tuple contains DataLoader objects in the order: train_loader, valid_loader, test_loader.
        Raises a RuntimeError if called before `setup()` has been executed to ensure data is ready.
        """
        if self.train_loader is None or self.valid_loader is None or self.test_loader is None:
            raise RuntimeError("Call .setup() before requesting dataloaders.")
        return self.train_loader, self.valid_loader, self.test_loader

    # ---- Internal helpers -----------------------------------------------

    def _load_raw_dataset(self) -> None:
        """Load the raw CCD dataset from the EEGChallengeDataset helper (no preprocessing)."""
        self.dataset_ccd = EEGChallengeDataset(
            task="contrastChangeDetection",
            release=self.release,
            cache_dir=self.cache_root,
            mini=self.mini,
        )

    def _create_windows(self) -> None:
        """
        Apply EEGDash preprocessing pipeline to annotate trials, add auxiliary anchors,
        and create sliding windows aligned to the CCD stimulus anchor with configured temporal parameters.

        The resulting `single_windows` dataset contains windowed EEG segments with associated reaction time labels
        and metadata columns required for the regression baseline.
        """
        assert self.dataset_ccd is not None

        epoch_len_s = self.epoch_len_s
        sfreq = self.sfreq

        # Prepare target reaction times and auxiliary anchors
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

        # Extract windows relative to stimulus anchor with shift and length in seconds
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
        """
        Split the windowed dataset into train/validation/test sets on the subject level
        to avoid subject leakage between splits.

        This subject-wise splitting mirrors the official starter kit logic and is critical for ensuring
        a fair subject-independent evaluation of model performance.
        """
        assert self.single_windows is not None

        meta_information = self.single_windows.get_metadata()

        valid_frac = self.valid_frac
        test_frac = self.test_frac
        seed = self.seed

        subjects = meta_information["subject"].unique()

        # Split by unique subject IDs to ensure no subject leakage between sets
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

        # Reassemble subject-specific datasets into concatenated train, valid, and test sets
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
        """Build PyTorch dataloaders for train/validation/test with configured batch size and num_workers."""
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