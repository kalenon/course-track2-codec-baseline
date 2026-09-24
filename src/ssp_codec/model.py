"""Time-domain convolutional neural codec used by the course baseline.

The structure uses a compact encoder, RVQ, and decoder at a fixed course
operating point: waveform convolutional encoder, residual vector quantizer,
and waveform transposed-convolution decoder.  It is deliberately fixed to
16 kHz and a 160-sample (10 ms) embedding rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


SAMPLE_RATE = 16_000
FRAME_SAMPLES = 160
BITS_PER_FRAME = 24


class CausalConv1d(nn.Module):
    """A length-preserving causal 1-D convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class ResidualUnit(nn.Module):
    """Two causal convolutions with ELU activations and a skip connection."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size)
        self.conv2 = CausalConv1d(channels, channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.elu(self.conv1(x))
        residual = self.conv2(residual)
        return F.elu(x + residual)


class ResidualStack(nn.Module):
    def __init__(self, channels: int, num_blocks: int, kernel_size: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *(ResidualUnit(channels, kernel_size) for _ in range(num_blocks))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class WaveformEncoder(nn.Module):
    """Four-stage residual convolutional encoder with 160-sample stride."""

    def __init__(
        self,
        channels: Iterable[int],
        strides: Iterable[int],
        residual_blocks: int,
        residual_kernel_size: int,
    ) -> None:
        super().__init__()
        channels = list(channels)
        strides = list(strides)
        if len(channels) != len(strides) + 1:
            raise ValueError("encoder channels must contain one more value than strides")
        self.input_conv = CausalConv1d(1, channels[0], kernel_size=7)
        self.blocks = nn.ModuleList()
        for in_channels, out_channels, stride in zip(channels[:-1], channels[1:], strides):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "residual": ResidualStack(
                            in_channels, residual_blocks, residual_kernel_size
                        ),
                        "downsample": nn.Conv1d(
                            in_channels,
                            out_channels,
                            kernel_size=stride,
                            stride=stride,
                        ),
                    }
                )
            )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.input_conv(waveform))
        for block in self.blocks:
            x = block["downsample"](block["residual"](x))
        return x.transpose(1, 2)


class WaveformDecoder(nn.Module):
    """Four-stage transposed-convolution decoder with no implicit overlap-add."""

    def __init__(
        self,
        channels: Iterable[int],
        strides: Iterable[int],
        residual_blocks: int,
        residual_kernel_size: int,
    ) -> None:
        super().__init__()
        channels = list(channels)
        strides = list(strides)
        if len(channels) != len(strides) + 1:
            raise ValueError("decoder channels must contain one more value than strides")
        self.blocks = nn.ModuleList()
        for in_channels, out_channels, stride in zip(channels[:-1], channels[1:], strides):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "upsample": nn.ConvTranspose1d(
                            in_channels,
                            out_channels,
                            kernel_size=stride,
                            stride=stride,
                        ),
                        "residual": ResidualStack(
                            out_channels, residual_blocks, residual_kernel_size
                        ),
                    }
                )
            )
        self.output_conv = CausalConv1d(channels[-1], 1, kernel_size=7)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = embeddings.transpose(1, 2)
        for block in self.blocks:
            x = F.elu(block["upsample"](x))
            x = block["residual"](x)
        return torch.tanh(self.output_conv(x))


class ResidualVectorQuantizer(nn.Module):
    """EMA residual VQ with k-means initialization and dead-code replacement."""

    def __init__(
        self,
        embedding_dim: int,
        num_codebooks: int = 3,
        codebook_size: int = 256,
        codebook_dim: int = 160,
        commitment_weight: float = 0.25,
        usage_loss_weight: float = 0.0,
        usage_temperature: float = 0.1,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 0.5,
        dead_code_after_updates: int = 100,
        kmeans_iterations: int = 10,
    ) -> None:
        super().__init__()
        if num_codebooks != 3 or codebook_size != 256:
            raise ValueError("the course baseline is fixed to three 256-entry codebooks")
        if codebook_dim != embedding_dim:
            raise ValueError("EMA RVQ codebook_dim must equal embedding_dim")
        if not 0.0 < ema_decay < 1.0:
            raise ValueError("ema_decay must be between zero and one")
        self.embedding_dim = embedding_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_weight = commitment_weight
        # Retained only so older checkpoint configurations can still be loaded.
        # The differentiable usage loss was removed because its low-temperature
        # gradient dominated reconstruction and drove the latent scale upward.
        self.usage_loss_weight = 0.0
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold
        self.dead_code_after_updates = dead_code_after_updates
        self.kmeans_iterations = kmeans_iterations
        # EMA VQ codebooks are buffers: they are updated from assignments rather
        # than by Adam gradients. This avoids the early one-codeword collapse seen
        # with the original small, randomly initialized projected codebooks.
        self.register_buffer("codebooks", torch.zeros(num_codebooks, codebook_size, codebook_dim))
        self.register_buffer("ema_cluster_size", torch.zeros(num_codebooks, codebook_size))
        self.register_buffer(
            "ema_embed_sum", torch.zeros(num_codebooks, codebook_size, codebook_dim)
        )
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("update_count", torch.zeros((), dtype=torch.long))

    @property
    def bits_per_frame(self) -> int:
        return self.num_codebooks * 8

    def _distances(self, values: torch.Tensor, stage: int) -> torch.Tensor:
        table = self.codebooks[stage]
        return (
            values.square().sum(dim=-1, keepdim=True)
            - 2 * values @ table.t()
            + table.square().sum(dim=-1)
        )

    def _lookup(self, values: torch.Tensor, stage: int) -> tuple[torch.Tensor, torch.Tensor]:
        distance = self._distances(values, stage)
        indices = distance.argmin(dim=-1)
        return F.embedding(indices, self.codebooks[stage]), indices

    @staticmethod
    def _distributed() -> bool:
        return dist.is_available() and dist.is_initialized()

    @torch.no_grad()
    def _kmeans(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``codebook_size`` centroids from one local first-batch sample."""
        values = values.reshape(-1, self.codebook_dim)
        if values.size(0) == 0:
            raise ValueError("cannot initialize a codebook from an empty batch")
        if values.size(0) < self.codebook_size:
            repeat = (self.codebook_size + values.size(0) - 1) // values.size(0)
            values = values.repeat(repeat, 1)
        values = values[: min(values.size(0), 4096)]
        centroids = values[torch.randperm(values.size(0), device=values.device)[: self.codebook_size]].clone()
        for _ in range(self.kmeans_iterations):
            distance = (
                values.square().sum(dim=-1, keepdim=True)
                - 2 * values @ centroids.t()
                + centroids.square().sum(dim=-1)
            )
            assignments = distance.argmin(dim=-1)
            counts = torch.bincount(assignments, minlength=self.codebook_size)
            sums = torch.zeros_like(centroids)
            sums.index_add_(0, assignments, values)
            nonempty = counts > 0
            centroids[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
            if (~nonempty).any():
                replacements = torch.randint(
                    values.size(0), (int((~nonempty).sum()),), device=values.device
                )
                centroids[~nonempty] = values[replacements]
        return centroids

    @torch.no_grad()
    def _initialize_codebooks(self, embeddings: torch.Tensor) -> None:
        """Initialize all residual stages before their first assignment."""
        residual = embeddings.detach()
        for stage in range(self.num_codebooks):
            codebook = self._kmeans(residual)
            self.codebooks[stage].copy_(codebook)
            self.ema_cluster_size[stage].fill_(1.0)
            self.ema_embed_sum[stage].copy_(codebook)
            codeword, _ = self._lookup(residual, stage)
            residual = residual - codeword
        self.initialized.fill_(True)
        if self._distributed():
            # Each rank has a different first batch. Rank zero supplies one
            # deterministic initialization to every worker before EMA updates.
            for value in (
                self.codebooks,
                self.ema_cluster_size,
                self.ema_embed_sum,
                self.initialized,
            ):
                dist.broadcast(value, src=0)

    @torch.no_grad()
    def _ema_update(self, stage: int, values: torch.Tensor, indices: torch.Tensor) -> None:
        flat_values = values.reshape(-1, self.codebook_dim)
        flat_indices = indices.reshape(-1)
        counts = torch.bincount(flat_indices, minlength=self.codebook_size).to(flat_values.dtype)
        sums = torch.zeros_like(self.ema_embed_sum[stage])
        sums.index_add_(0, flat_indices, flat_values)
        if self._distributed():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        self.ema_cluster_size[stage].mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)
        self.ema_embed_sum[stage].mul_(self.ema_decay).add_(sums, alpha=1 - self.ema_decay)
        total_count = self.ema_cluster_size[stage].sum()
        smoothed_count = (
            (self.ema_cluster_size[stage] + 1e-5)
            / (total_count + self.codebook_size * 1e-5)
            * total_count
        )
        self.codebooks[stage].copy_(self.ema_embed_sum[stage] / smoothed_count.unsqueeze(1))

        if self.update_count.item() >= self.dead_code_after_updates:
            dead = self.ema_cluster_size[stage] < self.dead_code_threshold
            if dead.any():
                # Use live encoder residuals for replacement, then synchronize
                # the selected values and their EMA state under DDP.
                if not self._distributed() or dist.get_rank() == 0:
                    replacement_indices = torch.arange(
                        int(dead.sum()), device=flat_values.device
                    ) % flat_values.size(0)
                    replacements = flat_values[replacement_indices]
                    self.codebooks[stage, dead] = replacements
                    self.ema_embed_sum[stage, dead] = replacements * self.dead_code_threshold
                    self.ema_cluster_size[stage, dead] = self.dead_code_threshold
                if self._distributed():
                    for value in (
                        self.codebooks[stage],
                        self.ema_cluster_size[stage],
                        self.ema_embed_sum[stage],
                    ):
                        dist.broadcast(value, src=0)

    @torch.no_grad()
    def usage_metrics(self) -> dict[str, list[float] | list[int]]:
        """EMA-smoothed RVQ occupancy, suitable for checkpoint logging."""
        active = (self.ema_cluster_size >= self.dead_code_threshold).sum(dim=1)
        probabilities = self.ema_cluster_size / self.ema_cluster_size.sum(dim=1, keepdim=True).clamp_min(1e-12)
        perplexity = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1))
        return {
            "active_codes": [int(value) for value in active.cpu()],
            "perplexity": [float(value) for value in perplexity.cpu()],
        }

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, frames, channels]")
        if self.training and not bool(self.initialized):
            self._initialize_codebooks(embeddings)
        residual = embeddings
        quantized = torch.zeros_like(embeddings)
        all_indices = []
        commitment = embeddings.new_zeros(())
        for stage in range(self.num_codebooks):
            codeword, indices = self._lookup(residual, stage)
            commitment = commitment + F.mse_loss(residual, codeword.detach())
            if self.training:
                self._ema_update(stage, residual.detach(), indices)
            quantized = quantized + codeword
            residual = embeddings - quantized
            all_indices.append(indices)
        if self.training:
            self.update_count.add_(1)
        indices = torch.stack(all_indices, dim=-1)
        loss = self.commitment_weight * commitment
        straight_through = embeddings + (quantized - embeddings).detach()
        return straight_through, indices, loss

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 3 or indices.size(-1) != self.num_codebooks:
            raise ValueError("indices must have shape [batch, frames, 3]")
        quantized = torch.zeros(
            *indices.shape[:-1], self.embedding_dim, device=indices.device, dtype=self.codebooks.dtype
        )
        for stage in range(self.num_codebooks):
            quantized = quantized + F.embedding(indices[..., stage], self.codebooks[stage])
        return quantized


@dataclass(frozen=True)
class CodecSpecification:
    sample_rate: int = SAMPLE_RATE
    frame_samples: int = FRAME_SAMPLES
    bits_per_frame: int = BITS_PER_FRAME


class CourseCodec(nn.Module):
    """The complete fixed-16-kHz / fixed-2.4-kbps neural codec baseline."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_samples: int = FRAME_SAMPLES,
        encoder_channels: Iterable[int] = (8, 16, 32, 64, 160),
        encoder_strides: Iterable[int] = (2, 4, 4, 5),
        decoder_channels: Iterable[int] = (160, 48, 24, 12, 8),
        decoder_strides: Iterable[int] = (5, 4, 4, 2),
        residual_blocks: int = 3,
        residual_kernel_size: int = 3,
        rvq_num_codebooks: int = 3,
        rvq_codebook_size: int = 256,
        rvq_codebook_dim: int = 160,
        commitment_weight: float = 0.25,
        rvq_usage_loss_weight: float = 0.0,
        rvq_usage_temperature: float = 0.1,
        rvq_ema_decay: float = 0.99,
        rvq_dead_code_threshold: float = 0.5,
        rvq_dead_code_after_updates: int = 100,
        rvq_kmeans_iterations: int = 10,
    ) -> None:
        super().__init__()
        encoder_channels = tuple(encoder_channels)
        encoder_strides = tuple(encoder_strides)
        decoder_channels = tuple(decoder_channels)
        decoder_strides = tuple(decoder_strides)
        if sample_rate != SAMPLE_RATE or frame_samples != FRAME_SAMPLES:
            raise ValueError("this baseline is fixed to 16 kHz and 160 samples per frame")
        if torch.tensor(encoder_strides).prod().item() != frame_samples:
            raise ValueError("encoder strides must multiply to 160 samples")
        if torch.tensor(decoder_strides).prod().item() != frame_samples:
            raise ValueError("decoder strides must multiply to 160 samples")
        if encoder_channels[-1] != decoder_channels[0]:
            raise ValueError("encoder output and decoder input dimensions must agree")
        self.specification = CodecSpecification()
        self.encoder = WaveformEncoder(
            encoder_channels, encoder_strides, residual_blocks, residual_kernel_size
        )
        # Fixed per-frame normalization prevents the encoder and EMA codebooks
        # from reducing loss by inflating their absolute scale. With no affine
        # parameters, each latent frame remains centered with unit variance.
        self.latent_norm = nn.LayerNorm(
            encoder_channels[-1], elementwise_affine=False
        )
        self.quantizer = ResidualVectorQuantizer(
            embedding_dim=encoder_channels[-1],
            num_codebooks=rvq_num_codebooks,
            codebook_size=rvq_codebook_size,
            codebook_dim=rvq_codebook_dim,
            commitment_weight=commitment_weight,
            usage_loss_weight=rvq_usage_loss_weight,
            usage_temperature=rvq_usage_temperature,
            ema_decay=rvq_ema_decay,
            dead_code_threshold=rvq_dead_code_threshold,
            dead_code_after_updates=rvq_dead_code_after_updates,
            kmeans_iterations=rvq_kmeans_iterations,
        )
        self.decoder = WaveformDecoder(
            decoder_channels, decoder_strides, residual_blocks, residual_kernel_size
        )

    @staticmethod
    def pad_to_frame(waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
        samples = waveform.size(-1)
        padding = (-samples) % FRAME_SAMPLES
        return F.pad(waveform, (0, padding)), padding

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.size(1) != 1:
            raise ValueError("waveform must have shape [batch, 1, samples]")
        padded, _ = self.pad_to_frame(waveform)
        embeddings = self.latent_norm(self.encoder(padded))
        _, indices, _ = self.quantizer(embeddings)
        return indices

    def decode(self, indices: torch.Tensor, length: int | None = None) -> torch.Tensor:
        waveform = self.decoder(self.quantizer.decode(indices))
        return waveform[..., :length] if length is not None else waveform

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.size(1) != 1:
            raise ValueError("waveform must have shape [batch, 1, samples]")
        original_length = waveform.size(-1)
        padded, _ = self.pad_to_frame(waveform)
        embeddings = self.latent_norm(self.encoder(padded))
        quantized, indices, quantizer_loss = self.quantizer(embeddings)
        reconstructed = self.decoder(quantized)[..., :original_length]
        return reconstructed, indices, quantizer_loss
