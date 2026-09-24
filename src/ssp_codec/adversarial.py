"""Multi-resolution STFT discriminators and adversarial losses."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn


class STFTDiscriminator(nn.Module):
    """Judge a waveform from the real and imaginary parts of one STFT."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        channels = (2, 16, 32, 32, 64)
        strides = ((1, 2), (2, 2), (2, 2), (2, 2))
        self.convolutions = nn.ModuleList(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 5),
                stride=stride,
                padding=(1, 2),
            )
            for in_channels, out_channels, stride in zip(
                channels[:-1], channels[1:], strides
            )
        )
        self.output = nn.Conv2d(channels[-1], 1, kernel_size=3, padding=1)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if waveform.ndim != 3 or waveform.size(1) != 1:
            raise ValueError("waveform must have shape [batch, 1, samples]")
        spectrum = torch.stft(
            waveform[:, 0],
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=waveform.dtype),
            return_complex=True,
        )
        hidden = torch.view_as_real(spectrum).permute(0, 3, 1, 2)
        features = []
        for convolution in self.convolutions:
            hidden = F.leaky_relu(convolution(hidden), negative_slope=0.2)
            features.append(hidden)
        return self.output(hidden), features


class MultiResolutionSTFTDiscriminator(nn.Module):
    """Independent discriminators operating at several STFT resolutions."""

    def __init__(self, resolutions: Iterable[Iterable[int]]) -> None:
        super().__init__()
        parsed = [tuple(int(value) for value in resolution) for resolution in resolutions]
        if not parsed or any(len(resolution) != 3 for resolution in parsed):
            raise ValueError("each discriminator resolution must be [n_fft, hop, win]")
        self.discriminators = nn.ModuleList(
            STFTDiscriminator(n_fft, hop_length, win_length)
            for n_fft, hop_length, win_length in parsed
        )

    def forward(
        self, waveform: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        logits = []
        features = []
        for discriminator in self.discriminators:
            resolution_logits, resolution_features = discriminator(waveform)
            logits.append(resolution_logits)
            features.append(resolution_features)
        return logits, features


def discriminator_hinge_loss(
    real_logits: list[torch.Tensor], fake_logits: list[torch.Tensor]
) -> torch.Tensor:
    losses = [
        F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
        for real, fake in zip(real_logits, fake_logits)
    ]
    return torch.stack(losses).mean()


def generator_hinge_loss(fake_logits: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([-logits.mean() for logits in fake_logits]).mean()


def feature_matching_loss(
    real_features: list[list[torch.Tensor]],
    fake_features: list[list[torch.Tensor]],
) -> torch.Tensor:
    losses = [
        F.l1_loss(fake, real.detach())
        for real_resolution, fake_resolution in zip(real_features, fake_features)
        for real, fake in zip(real_resolution, fake_resolution)
    ]
    return torch.stack(losses).mean()
