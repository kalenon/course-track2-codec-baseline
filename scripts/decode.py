#!/usr/bin/env python3
"""Decode course-format bitstreams without access to source audio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssp_codec.io import load_codec, read_bitstream  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bitstream-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, _ = load_codec(args.checkpoint, device)
    bitstream_dir = Path(args.bitstream_dir)
    files = sorted(bitstream_dir.rglob("*.bin"))
    for path in tqdm(files, desc="decode", unit="file"):
        indices, metadata = read_bitstream(path)
        with torch.inference_mode():
            audio = model.decode(indices.unsqueeze(0).to(device), int(metadata["num_samples"]))
        relative = path.relative_to(bitstream_dir).with_suffix(".wav")
        output_path = Path(args.output_dir) / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio[0, 0].cpu().numpy(), 16_000, subtype="PCM_16")


if __name__ == "__main__":
    main()

