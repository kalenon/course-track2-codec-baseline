"""Checkpoint and fixed-format bitstream helpers."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .model import BITS_PER_FRAME, FRAME_SAMPLES, SAMPLE_RATE, CourseCodec


def load_codec(checkpoint: str | Path, device: torch.device) -> tuple[CourseCodec, dict]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CourseCodec(**state["model_config"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def write_bitstream(path: str | Path, indices: torch.Tensor, num_samples: int) -> None:
    """Write exactly three bytes per 10-ms frame plus non-signal JSON metadata."""
    if indices.ndim != 2 or indices.size(-1) != 3:
        raise ValueError("indices must have shape [frames, 3]")
    if indices.numel() and (indices.min() < 0 or indices.max() > 255):
        raise ValueError("baseline RVQ indices must fit in one unsigned byte")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(indices.detach().cpu().to(torch.uint8).contiguous().numpy().tobytes())
    metadata = {
        "sample_rate": SAMPLE_RATE,
        "num_samples": int(num_samples),
        "frame_samples": FRAME_SAMPLES,
        "bits_per_frame": BITS_PER_FRAME,
        "bitrate_bps": 2400,
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_bitstream(path: str | Path) -> tuple[torch.Tensor, dict]:
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("sample_rate") != SAMPLE_RATE:
        raise ValueError("bitstream metadata has an unsupported sample rate")
    raw = path.read_bytes()
    frames = (int(metadata["num_samples"]) + FRAME_SAMPLES - 1) // FRAME_SAMPLES
    if len(raw) != 3 * frames:
        raise ValueError(f"{path} must contain {3 * frames} bytes, found {len(raw)}")
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.long).reshape(frames, 3), metadata

