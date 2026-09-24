#!/usr/bin/env python3
"""Report analytical Conv/Linear MAC counts for the course baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssp_codec.model import CourseCodec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="optional checkpoint; model configuration is read from it")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.duration <= 0:
        raise ValueError("duration must be positive")
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = CourseCodec(**state["model_config"])
    else:
        model = CourseCodec()
    macs: dict[str, int] = {}
    handles = []

    def hook(name: str):
        def count(module: nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
            if isinstance(module, nn.Conv1d):
                kernel = module.kernel_size[0]
                per_output = (module.in_channels // module.groups) * kernel
            elif isinstance(module, nn.ConvTranspose1d):
                kernel = module.kernel_size[0]
                per_output = (module.in_channels // module.groups) * kernel
            elif isinstance(module, nn.Linear):
                per_output = module.in_features
            else:
                return
            macs[name] = macs.get(name, 0) + output.numel() * per_output
        return count

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            handles.append(module.register_forward_hook(hook(name)))
    samples = round(args.duration * 16_000)
    with torch.inference_mode():
        model(torch.zeros(1, 1, samples))
    for handle in handles:
        handle.remove()
    encoder = sum(value for name, value in macs.items() if name.startswith("encoder."))
    decoder = sum(value for name, value in macs.items() if name.startswith("decoder."))
    quantizer_projections = sum(value for name, value in macs.items() if name.startswith("quantizer."))
    frames = (samples + 159) // 160
    # Squared-distance VQ lookup: approximately three operations per codebook dimension.
    vq_search = frames * 3 * 256 * model.quantizer.codebook_dim * 3
    report = {
        "sample_rate_hz": 16000,
        "duration_seconds": args.duration,
        "frame_samples": 160,
        "bitrate_bps": 2400,
        "encoder_mmac_per_s": encoder / args.duration / 1e6,
        "decoder_mmac_per_s": decoder / args.duration / 1e6,
        "quantizer_mmac_per_s": (quantizer_projections + vq_search) / args.duration / 1e6,
        "total_mflop_per_s": 2 * (encoder + decoder + quantizer_projections + vq_search) / args.duration / 1e6,
        "decoder_mflop_per_s": 2 * decoder / args.duration / 1e6,
        # EMA codebooks are inference state stored as buffers rather than Adam
        # parameters, but must still count toward the deployed model size.
        "parameters": sum(parameter.numel() for parameter in model.parameters()) + model.quantizer.codebooks.numel(),
        "notes": "Counts Conv1d, ConvTranspose1d, Linear and approximate RVQ nearest-neighbour search. Excludes ELU, tanh, padding and file I/O.",
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
