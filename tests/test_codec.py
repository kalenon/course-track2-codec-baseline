import json

import torch

from ssp_codec.adversarial import (
    MultiResolutionSTFTDiscriminator,
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_hinge_loss,
)
from ssp_codec.io import read_bitstream, write_bitstream
from ssp_codec.model import CourseCodec


def test_fixed_2p4k_shape_and_indices() -> None:
    model = CourseCodec()
    waveform = torch.randn(2, 1, 16_080)
    reconstructed, indices, _ = model(waveform)
    assert reconstructed.shape == waveform.shape
    assert indices.shape == (2, 101, 3)
    assert indices.dtype == torch.long
    assert int(indices.min()) >= 0
    assert int(indices.max()) < 256


def test_ema_rvq_initializes_and_decode_matches_forward() -> None:
    model = CourseCodec()
    waveform = torch.randn(8, 1, 16_000)
    model.train()
    model(waveform)
    assert bool(model.quantizer.initialized)
    metrics = model.quantizer.usage_metrics()
    assert metrics["active_codes"] == [256, 256, 256]
    model.eval()
    direct, _, _ = model(waveform)
    indices = model.encode(waveform)
    decoded = model.decode(indices, waveform.size(-1))
    assert torch.allclose(direct, decoded, atol=1e-6)


def test_latent_scale_is_fixed_before_quantization() -> None:
    model = CourseCodec()
    waveform = torch.randn(2, 1, 16_000)
    with torch.no_grad():
        latent = model.latent_norm(model.encoder(waveform))
    assert torch.allclose(latent.mean(dim=-1), torch.zeros_like(latent[..., 0]), atol=1e-5)
    assert torch.allclose(
        latent.var(dim=-1, unbiased=False),
        torch.ones_like(latent[..., 0]),
        atol=2e-3,
    )


def test_bitstream_uses_exactly_three_bytes_per_frame(tmp_path) -> None:
    indices = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    path = tmp_path / "sample.bin"
    write_bitstream(path, indices, num_samples=161)
    assert path.stat().st_size == 6
    assert json.loads(path.with_suffix(".json").read_text())["bitrate_bps"] == 2400
    decoded_indices, metadata = read_bitstream(path)
    assert torch.equal(decoded_indices, indices)
    assert metadata["num_samples"] == 161


def test_multiresolution_discriminator_losses_are_finite() -> None:
    discriminator = MultiResolutionSTFTDiscriminator(
        [[256, 64, 256], [512, 128, 512], [1024, 256, 1024]]
    )
    real = torch.randn(2, 1, 4096)
    fake = torch.randn_like(real)
    real_logits, real_features = discriminator(real)
    fake_logits, fake_features = discriminator(fake)
    losses = (
        discriminator_hinge_loss(real_logits, fake_logits),
        generator_hinge_loss(fake_logits),
        feature_matching_loss(real_features, fake_features),
    )
    assert all(torch.isfinite(loss) for loss in losses)
