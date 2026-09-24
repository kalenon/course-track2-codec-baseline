#!/usr/bin/env python3
"""Evaluate codec output against its 16-kHz input audio."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi


def load_audio(path):
    info = sf.info(path)
    if info.format != "WAV" or info.samplerate != 16000 or info.channels != 1:
        raise ValueError(f"Expected mono 16-kHz WAV: {path}")
    audio, _ = sf.read(path, dtype="float32")
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite audio: {path}")
    return audio


def si_snr(reference, estimate):
    reference = reference.astype(np.float64) - np.mean(reference)
    estimate = estimate.astype(np.float64) - np.mean(estimate)
    projection = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-8) * reference
    error = estimate - projection
    return float(10 * np.log10((np.dot(projection, projection) + 1e-8)
                               / (np.dot(error, error) + 1e-8)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--reconstructed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-utmos", action="store_true", help="Smoke tests only")
    args = parser.parse_args()
    references = sorted(args.reference_dir.rglob("*.wav"))
    if not references:
        raise ValueError("No reference WAV files found")
    device = torch.device(args.device)
    if not args.skip_utmos and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu")
    model = None
    if not args.skip_utmos:
        model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
        model.device = device
    rows = []
    for ref_path in references:
        relative = ref_path.relative_to(args.reference_dir)
        estimate_path = args.reconstructed_dir / relative
        reference = load_audio(ref_path)
        estimate = load_audio(estimate_path)
        if len(reference) != len(estimate):
            raise ValueError(f"Length mismatch: {relative}")
        row = {"file": str(relative), "PESQ": float(pesq(16000, reference, estimate, "wb")),
               "ESTOI": float(stoi(reference, estimate, 16000, extended=True)),
               "SI_SNR": si_snr(reference, estimate)}
        if model is not None:
            with torch.inference_mode():
                row["UTMOS"] = float(model(torch.from_numpy(estimate).unsqueeze(0).to(device),
                                           16000).cpu().item())
        rows.append(row)
    means = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "file"}
    score = None
    if "UTMOS" in means:
        # Normalize each file first, then average as specified by the course.
        per_file_scores = [
            100 * (
                0.25 * np.clip((row["PESQ"] - 1) / 3.5, 0, 1)
                + 0.25 * np.clip(row["ESTOI"], 0, 1)
                + 0.25 * np.clip((row["SI_SNR"] + 10) / 40, 0, 1)
                + 0.25 * np.clip((row["UTMOS"] - 1) / 4, 0, 1)
            )
            for row in rows
        ]
        score = float(np.mean(per_file_scores))
    summary = {"files": len(rows), "means": means, "objective_quality_score": score,
               "utmos_skipped": bool(args.skip_utmos)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_file.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
