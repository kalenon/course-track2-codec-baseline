#!/usr/bin/env python3
"""Build speaker-disjoint manifests from VCTK 0.92 and LibriTTS.

The script does not copy audio.  The training dataset resamples the 48-kHz
VCTK and 24-kHz LibriTTS files to the baseline's fixed 16-kHz rate on read.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


LIBRITTS_TRAIN_SPLITS = ("train-clean-100", "train-clean-360", "train-other-500")
LIBRITTS_VALID_SPLITS = ("dev-clean", "dev-other")


def audio_files(directory: Path, pattern: str) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"missing directory: {directory}")
    return sorted(path.resolve() for path in directory.rglob(pattern) if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vctk-root", required=True)
    parser.add_argument("--libritts-root", required=True)
    parser.add_argument("--output-dir", default="data/manifests/vctk_libritts")
    parser.add_argument("--vctk-valid-speaker-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 0 < args.vctk_valid_speaker_ratio < 1:
        raise ValueError("--vctk-valid-speaker-ratio must be in (0, 1)")

    vctk_root = Path(args.vctk_root)
    vctk_audio_root = vctk_root / "wav48_silence_trimmed"
    # One microphone per utterance prevents duplicate recordings in train/valid.
    vctk_files = audio_files(vctk_audio_root, "*_mic1.flac")
    if not vctk_files:
        raise ValueError(f"no *_mic1.flac files found in {vctk_audio_root}")
    speakers = sorted({path.parent.name for path in vctk_files})
    random.Random(args.seed).shuffle(speakers)
    valid_count = max(1, round(len(speakers) * args.vctk_valid_speaker_ratio))
    valid_speakers = set(speakers[:valid_count])
    vctk_train = [path for path in vctk_files if path.parent.name not in valid_speakers]
    vctk_valid = [path for path in vctk_files if path.parent.name in valid_speakers]

    libritts_root = Path(args.libritts_root)
    libritts_train = [
        path
        for split in LIBRITTS_TRAIN_SPLITS
        for path in audio_files(libritts_root / split, "*.wav")
    ]
    libritts_valid = [
        path
        for split in LIBRITTS_VALID_SPLITS
        for path in audio_files(libritts_root / split, "*.wav")
    ]
    if not libritts_train or not libritts_valid:
        raise ValueError("LibriTTS train and dev splits must both contain WAV files")

    # Keep corpus ordering independent of the absolute mount points. Each input
    # list is already sorted within its corpus.
    train_files = libritts_train + vctk_train
    valid_files = libritts_valid + vctk_valid
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.scp").write_text(
        "\n".join(str(path) for path in train_files) + "\n", encoding="utf-8"
    )
    (output_dir / "valid.scp").write_text(
        "\n".join(str(path) for path in valid_files) + "\n", encoding="utf-8"
    )
    summary = {
        "target_sample_rate_hz": 16000,
        "resampling": "scipy.signal.resample_poly at load time",
        "vctk_source": str(vctk_audio_root.resolve()),
        "vctk_microphone": "mic1 only",
        "vctk_train_speakers": len(speakers) - valid_count,
        "vctk_valid_speakers": sorted(valid_speakers),
        "vctk_train_files": len(vctk_train),
        "vctk_valid_files": len(vctk_valid),
        "libritts_train_splits": list(LIBRITTS_TRAIN_SPLITS),
        "libritts_valid_splits": list(LIBRITTS_VALID_SPLITS),
        "libritts_train_files": len(libritts_train),
        "libritts_valid_files": len(libritts_valid),
        "total_train_files": len(train_files),
        "total_valid_files": len(valid_files),
        "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
