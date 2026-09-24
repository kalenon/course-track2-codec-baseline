#!/usr/bin/env python3
"""Encode 16-kHz WAV/FLAC files into the course 3-byte-per-frame format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssp_codec.io import load_codec, write_bitstream  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, _ = load_codec(args.checkpoint, device)
    input_dir = Path(args.input_dir)
    files = sorted(
        path for path in input_dir.rglob("*") if path.suffix.lower() in {".wav", ".flac"}
    )
    for path in tqdm(files, desc="encode", unit="file"):
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if sample_rate != 16_000:
            raise ValueError(f"{path} is {sample_rate} Hz, expected 16 kHz")
        waveform = torch.from_numpy(audio.mean(axis=1).copy()).to(device).view(1, 1, -1)
        with torch.inference_mode():
            indices = model.encode(waveform)[0]
        relative = path.relative_to(input_dir).with_suffix(".bin")
        write_bitstream(Path(args.output_dir) / relative, indices, waveform.size(-1))


if __name__ == "__main__":
    main()

