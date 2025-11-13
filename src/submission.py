"""Baseline Codabench submission for EEG Foundation Challenge 2025.

Defines the `Submission` class used by the evaluation code.
"""
from pathlib import Path
from typing import Union

import torch
from braindecode.models import EEGNeX


def _resolve_path(name: str) -> Path:
    """Resolve a file name in a few common locations used by Codabench."""
    candidates = [
        Path(name),
        Path(__file__).resolve().parent / name,
        Path("/app/input/res") / name,
        Path("/app/input") / name,
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Could not find file {name!r} in any known location.")


class Submission:
    """Submission interface expected by the EEG Challenge evaluation code."""

    def __init__(self, SFREQ: float, DEVICE: Union[str, torch.device]):
        # SFREQ and DEVICE are provided by the evaluation environment
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self) -> torch.nn.Module:
        """Return a model for Challenge 1 with loaded weights."""
        model = EEGNeX(
            n_chans=129,
            n_outputs=1,
            sfreq=self.sfreq,
            n_times=int(2 * self.sfreq),
        ).to(self.device)

        state_dict = torch.load(
            _resolve_path("weights_challenge_1.pt"),
            map_location=self.device,
        )
        model.load_state_dict(state_dict)
        return model

    def get_model_challenge_2(self) -> torch.nn.Module:
        """Return a model for Challenge 2.

        If no weights_challenge_2.pt is provided, the randomly initialised
        model is returned so that the submission still runs.
        """
        model = EEGNeX(
            n_chans=129,
            n_outputs=1,
            sfreq=self.sfreq,
            n_times=int(2 * self.sfreq),
        ).to(self.device)

        try:
            state_dict = torch.load(
                _resolve_path("weights_challenge_2.pt"),
                map_location=self.device,
            )
        except FileNotFoundError:
            return model

        model.load_state_dict(state_dict)
        return model