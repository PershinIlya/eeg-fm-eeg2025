#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import shutil
import tempfile
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Codabench submission zip for EEG Challenge 2025."
    )
    parser.add_argument(
        "--submission-py",
        type=str,
        default="submission.py",
        help=(
            "Path to an existing submission.py file to include in the zip "
            "(default: ./submission.py). The file must exist and define the "
            "Submission class expected by the EEG Challenge evaluation code."
        ),
    )
    parser.add_argument(
        "--weights-ch1",
        type=str,
        required=True,
        help="Path to trained weights for Challenge 1 (will be saved as weights_challenge_1.pt inside the zip).",
    )
    parser.add_argument(
        "--weights-ch2",
        type=str,
        default=None,
        help="Optional path to weights for Challenge 2 (will be saved as weights_challenge_2.pt inside the zip).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="my_submission.zip",
        help="Output zip file name (default: my_submission.zip).",
    )
    parser.add_argument(
        "--extra",
        type=str,
        nargs="*",
        default=None,
        help="Optional extra files to include at top level of the zip (e.g. README, config).",
    )
    return parser.parse_args()


def ensure_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def main() -> None:
    args = parse_args()

    submission_py = Path(args.submission_py).resolve()
    weights_ch1_src = Path(args.weights_ch1).resolve()
    weights_ch2_src = Path(args.weights_ch2).resolve() if args.weights_ch2 else None
    output_zip = Path(args.output).resolve()

    ensure_file(submission_py, "submission.py")
    ensure_file(weights_ch1_src, "Challenge 1 weights")

    if weights_ch2_src is not None:
        ensure_file(weights_ch2_src, "Challenge 2 weights")

    extra_files = []
    if args.extra:
        for f in args.extra:
            p = Path(f).resolve()
            ensure_file(p, "Extra file")
            extra_files.append(p)

    # Use a temporary directory to stage files before zipping.
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        # Copy submission.py as-is into the temporary directory
        submission_target = tmpdir / "submission.py"
        shutil.copy2(submission_py, submission_target)

        # Copy weights for challenge 1 with the required name
        shutil.copy2(weights_ch1_src, tmpdir / "weights_challenge_1.pt")

        # Optional: weights for challenge 2
        if weights_ch2_src is not None:
            shutil.copy2(weights_ch2_src, tmpdir / "weights_challenge_2.pt")

        # Optional extra files (keep only base names in the archive)
        for extra_path in extra_files:
            shutil.copy2(extra_path, tmpdir / extra_path.name)

        # Create flat zip archive (no folders inside)
        if output_zip.exists():
            print(f"Overwriting existing zip: {output_zip}")
            output_zip.unlink()

        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in tmpdir.iterdir():
                # arcname ensures single-level depth, only file names
                zf.write(item, arcname=item.name)

        print(f"Created submission zip: {output_zip}")


if __name__ == "__main__":
    main()