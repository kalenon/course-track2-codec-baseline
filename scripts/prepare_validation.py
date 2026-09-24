#!/usr/bin/env python3
"""Create a deterministic 16-kHz codec validation subset from an audio-path manifest."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="One source WAV/FLAC path per line")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sources = [Path(line.strip()) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not sources or args.limit < 1:
        raise ValueError("Manifest must be nonempty and limit positive")
    rng = np.random.default_rng(args.seed)
    indices = sorted(rng.choice(len(sources), size=min(args.limit, len(sources)), replace=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping = []
    for rank, index in enumerate(indices):
        source = sources[index]
        audio, sr = sf.read(source, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if sr != 16000:
            factor = np.gcd(sr, 16000)
            mono = resample_poly(mono, 16000 // factor, sr // factor).astype(np.float32)
        target = args.output_dir / f"item_{rank:04d}.wav"
        sf.write(target, mono, 16000, subtype="PCM_16")
        mapping.append({"id": target.stem, "source": str(source), "source_sr": sr})
    summary = {"manifest": str(args.manifest.resolve()),
               "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               "seed": args.seed, "files": len(mapping), "items": mapping}
    (args.output_dir / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Prepared {len(mapping)} mono 16-kHz WAV files in {args.output_dir}")


if __name__ == "__main__":
    main()
