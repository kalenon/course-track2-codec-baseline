#!/usr/bin/env python3
"""Validate course bitstream size, metadata, and decoded WAV format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--bitstream-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    failures = []
    input_dir = Path(args.input_dir)
    for input_path in sorted(input_dir.rglob("*.wav")):
        relative = input_path.relative_to(input_dir)
        audio, sample_rate = sf.read(input_path, always_2d=True)
        expected_frames = (len(audio) + 159) // 160
        bitstream_path = Path(args.bitstream_dir) / relative.with_suffix(".bin")
        metadata_path = bitstream_path.with_suffix(".json")
        output_path = Path(args.output_dir) / relative
        if sample_rate != 16_000 or audio.shape[1] != 1:
            failures.append(f"{relative}: input must be 16-kHz mono")
        if not bitstream_path.is_file() or bitstream_path.stat().st_size != 3 * expected_frames:
            failures.append(f"{relative}: .bin must contain exactly {3 * expected_frames} bytes")
        if not metadata_path.is_file():
            failures.append(f"{relative}: missing .json metadata")
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if int(metadata.get("num_samples", -1)) != len(audio):
                failures.append(f"{relative}: metadata num_samples does not match input")
        if not output_path.is_file():
            failures.append(f"{relative}: missing decoded WAV")
        else:
            decoded, decoded_rate = sf.read(output_path, always_2d=True)
            if decoded_rate != 16_000 or decoded.shape != audio.shape or not np.isfinite(decoded).all():
                failures.append(f"{relative}: decoded WAV format, length, or values are invalid")
    if failures:
        raise SystemExit("submission validation failed:\n- " + "\n- ".join(failures))
    print("submission is valid: 16-kHz mono, exact 3 bytes/frame, matched decoded lengths")


if __name__ == "__main__":
    main()

