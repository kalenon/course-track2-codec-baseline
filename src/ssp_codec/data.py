"""Audio loading and fixed-duration crop utilities for codec training."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset


SUPPORTED_EXTENSIONS = {".wav", ".flac"}


class WaveformDataset(Dataset):
    def __init__(
        self,
        directory: str | Path | None,
        sample_rate: int,
        segment_samples: int,
        deterministic: bool = False,
        manifest: str | Path | None = None,
    ) -> None:
        if (directory is None) == (manifest is None):
            raise ValueError("provide exactly one of directory or manifest")
        self.directory = Path(directory) if directory is not None else None
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.deterministic = deterministic
        if manifest is not None:
            manifest_path = Path(manifest)
            self.files = [
                Path(line.strip())
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            ]
        else:
            self.files = sorted(
                path
                for path in self.directory.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        if not self.files:
            location = manifest if manifest is not None else self.directory
            raise ValueError(f"no WAV or FLAC files found in {location}")
        missing = [path for path in self.files if not path.is_file()]
        if missing:
            raise ValueError(f"manifest contains missing audio file: {missing[0]}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.files[index]
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = np.mean(audio, axis=1)
        if sample_rate != self.sample_rate:
            divisor = np.gcd(sample_rate, self.sample_rate)
            waveform = resample_poly(
                waveform,
                up=self.sample_rate // divisor,
                down=sample_rate // divisor,
            ).astype(np.float32, copy=False)
        if waveform.size < self.segment_samples:
            waveform = np.pad(waveform, (0, self.segment_samples - waveform.size))
        elif waveform.size > self.segment_samples:
            if self.deterministic:
                start = (waveform.size - self.segment_samples) // 2
            else:
                start = random.randint(0, waveform.size - self.segment_samples)
            waveform = waveform[start : start + self.segment_samples]
        return torch.from_numpy(waveform.copy()).unsqueeze(0)
