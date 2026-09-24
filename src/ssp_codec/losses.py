"""Reconstruction losses for the course codec baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multiscale_spectral_loss(
    estimate: torch.Tensor, reference: torch.Tensor, fft_sizes: list[int]
) -> torch.Tensor:
    """Log-magnitude L1 loss at several STFT resolutions."""
    if estimate.shape != reference.shape:
        raise ValueError("estimate and reference must have the same shape")
    estimate = estimate.squeeze(1)
    reference = reference.squeeze(1)
    loss = estimate.new_zeros(())
    for n_fft in fft_sizes:
        window = torch.hann_window(n_fft, device=estimate.device, dtype=estimate.dtype)
        estimated_stft = torch.stft(
            estimate, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True
        )
        reference_stft = torch.stft(
            reference, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True
        )
        loss = loss + F.l1_loss(
            torch.log1p(estimated_stft.abs()), torch.log1p(reference_stft.abs())
        )
    return loss / len(fft_sizes)

