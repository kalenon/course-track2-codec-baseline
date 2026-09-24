#!/usr/bin/env python3
"""Train the fixed-16-kHz / fixed-2.4-kbps neural codec baseline.

Run directly for one device, or launch with ``torchrun`` for distributed data
parallel training. In the latter case ``WORLD_SIZE`` is set by torchrun and
the script automatically enables DDP.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssp_codec.adversarial import (  # noqa: E402
    MultiResolutionSTFTDiscriminator,
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_hinge_loss,
)
from ssp_codec.data import WaveformDataset  # noqa: E402
from ssp_codec.losses import multiscale_spectral_loss  # noqa: E402
from ssp_codec.model import CourseCodec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline_16k_2p4k.yaml")
    parser.add_argument("--train-dir", help="recursively searched WAV/FLAC directory")
    parser.add_argument("--valid-dir", help="recursively searched WAV/FLAC directory")
    parser.add_argument("--train-manifest", help="one absolute audio path per line")
    parser.add_argument("--valid-manifest", help="one absolute audio path per line")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override config")
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="optional per-epoch batch cap for a fast interface check",
    )
    parser.add_argument(
        "--max-valid-steps",
        type=int,
        default=None,
        help="optional validation batch cap for a fast interface check",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="override DataLoader worker count; use 0 in restricted environments",
    )
    args = parser.parse_args()
    for split in ("train", "valid"):
        directory = getattr(args, f"{split}_dir")
        manifest = getattr(args, f"{split}_manifest")
        if (directory is None) == (manifest is None):
            parser.error(f"provide exactly one of --{split}-dir and --{split}-manifest")
    return args


def distributed_context(requested: str) -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    use_cuda = torch.cuda.is_available() if requested == "auto" else requested == "cuda"
    if distributed:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
        if use_cuda:
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank), rank, world_size, True
        return torch.device("cpu"), rank, world_size, True
    return torch.device("cuda" if use_cuda else "cpu"), rank, world_size, False


def make_dataset(args: argparse.Namespace, split: str, config: dict) -> WaveformDataset:
    return WaveformDataset(
        directory=getattr(args, f"{split}_dir"),
        manifest=getattr(args, f"{split}_manifest"),
        sample_rate=config["audio"]["sample_rate"],
        segment_samples=config["audio"]["segment_samples"],
        deterministic=split == "valid",
    )


def adversarial_factor(global_step: int, config: dict) -> float:
    """Linearly introduce GAN losses after a reconstruction-only warm-up."""
    start = int(config["adversarial"]["start_step"])
    warmup = int(config["adversarial"]["warmup_steps"])
    if global_step < start:
        return 0.0
    if warmup <= 0:
        return 1.0
    return min(1.0, (global_step - start + 1) / warmup)


def evaluate(
    model: torch.nn.Module,
    discriminator: MultiResolutionSTFTDiscriminator,
    loader: DataLoader,
    device: torch.device,
    config: dict,
    limit: int,
    distributed: bool,
    gan_factor: float,
) -> tuple[dict[str, float], dict[str, list[float] | list[int]]]:
    model.eval()
    discriminator.eval()
    totals = torch.zeros(6, device=device)
    count = torch.zeros((), device=device)
    code_histogram = torch.zeros((3, 256), device=device)
    with torch.inference_mode():
        for step, reference in enumerate(loader):
            if step >= limit:
                break
            reference = reference.to(device, non_blocking=True)
            reconstructed, indices, quantizer_loss = model(reference)
            waveform_loss = F.l1_loss(reconstructed, reference)
            spectral_loss = multiscale_spectral_loss(
                reconstructed, reference, config["loss"]["fft_sizes"]
            )
            if gan_factor > 0.0:
                real_logits, real_features = discriminator(reference)
                fake_logits, fake_features = discriminator(reconstructed)
                adversarial_loss = generator_hinge_loss(fake_logits)
                matching_loss = feature_matching_loss(real_features, fake_features)
                discriminator_loss = discriminator_hinge_loss(real_logits, fake_logits)
            else:
                adversarial_loss = waveform_loss.new_zeros(())
                matching_loss = waveform_loss.new_zeros(())
                discriminator_loss = waveform_loss.new_zeros(())
            totals += torch.stack(
                (
                    waveform_loss,
                    spectral_loss,
                    quantizer_loss,
                    adversarial_loss,
                    matching_loss,
                    discriminator_loss,
                )
            )
            count += 1
            for stage in range(3):
                code_histogram[stage] += torch.bincount(
                    indices[..., stage].reshape(-1), minlength=256
                ).to(code_histogram.dtype)
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        dist.all_reduce(code_histogram, op=dist.ReduceOp.SUM)
    if count.item() == 0:
        raise RuntimeError("validation loader yielded no batches")
    probabilities = code_histogram / code_histogram.sum(dim=1, keepdim=True).clamp_min(1)
    perplexity = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    )
    waveform, spectral, commitment, adversarial, matching, discriminator_loss = (
        totals / count
    ).tolist()
    reconstruction = (
        config["loss"]["waveform_l1_weight"] * waveform
        + config["loss"]["spectral_weight"] * spectral
    )
    metrics = {
        "waveform_l1": waveform,
        "spectral": spectral,
        "commitment": commitment,
        "adversarial": adversarial,
        "feature_matching": matching,
        "discriminator": discriminator_loss,
        "adversarial_factor": gan_factor,
        "reconstruction": reconstruction,
        "total": (
            reconstruction
            + commitment
            + gan_factor
            * (
                config["loss"]["adversarial_weight"] * adversarial
                + config["loss"]["feature_matching_weight"] * matching
            )
        ),
    }
    return metrics, {
        "active_codes": [int(value) for value in (code_histogram > 0).sum(dim=1).cpu()],
        "perplexity": [float(value) for value in perplexity.cpu()],
    }


def write_reconstruction_samples(
    model: CourseCodec,
    dataset: WaveformDataset,
    device: torch.device,
    samples_dir: Path,
    count: int,
) -> None:
    """Write fixed validation segments and their reconstructions for listening."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        for index in range(min(count, len(dataset))):
            reference = dataset[index].unsqueeze(0).to(device)
            indices = model.encode(reference)
            reconstructed = model.decode(indices, reference.size(-1))
            sf.write(
                samples_dir / f"{index:02d}_reference.wav",
                reference[0, 0].cpu().numpy(),
                16_000,
                subtype="PCM_16",
            )
            sf.write(
                samples_dir / f"{index:02d}_reconstructed.wav",
                reconstructed[0, 0].cpu().numpy(),
                16_000,
                subtype="PCM_16",
            )


def validate_and_checkpoint(
    *,
    model: torch.nn.Module,
    base_model: CourseCodec,
    base_discriminator: MultiResolutionSTFTDiscriminator,
    valid_loader: DataLoader,
    valid_set: WaveformDataset,
    device: torch.device,
    config: dict,
    validation_limit: int,
    distributed: bool,
    is_main_process: bool,
    checkpoint_dir: Path,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    optimizer: AdamW,
    discriminator_optimizer: AdamW,
    best_validation: float,
    train_loss: float,
    write_snapshot: bool,
) -> float:
    gan_factor = adversarial_factor(global_step, config)
    validation_metrics, validation_usage = evaluate(
        model,
        base_discriminator,
        valid_loader,
        device,
        config,
        validation_limit,
        distributed,
        gan_factor,
    )
    selection_loss = validation_metrics["reconstruction"]
    if is_main_process:
        next_best = min(best_validation, selection_loss)
        state = {
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "global_step": global_step,
            "model": base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "discriminator": base_discriminator.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "model_config": config["model"],
            "best_validation": next_best,
            "validation_metrics": validation_metrics,
            "quantizer_usage": validation_usage,
            "quantizer_ema_usage": base_model.quantizer.usage_metrics(),
        }
        torch.save(state, checkpoint_dir / "last.pt")
        if write_snapshot:
            torch.save(state, checkpoint_dir / f"step_{global_step:08d}.pt")
        if selection_loss < best_validation:
            torch.save(state, checkpoint_dir / "best.pt")
        write_reconstruction_samples(
            base_model,
            valid_set,
            device,
            checkpoint_dir.parent / "samples" / f"step_{global_step:08d}",
            config["training"]["sample_inference_count"],
        )
        print(
            f"epoch={epoch} step={global_step} train_loss={train_loss:.5f} "
            f"val_recon={validation_metrics['reconstruction']:.5f} "
            f"val_wave={validation_metrics['waveform_l1']:.5f} "
            f"val_spec={validation_metrics['spectral']:.5f} "
            f"val_commit={validation_metrics['commitment']:.5f} "
            f"val_adv={validation_metrics['adversarial']:.5f} "
            f"val_fm={validation_metrics['feature_matching']:.5f} "
            f"val_d={validation_metrics['discriminator']:.5f} "
            f"gan_factor={gan_factor:.3f} "
            f"active_codes={validation_usage['active_codes']} "
            f"perplexity={[round(value, 1) for value in validation_usage['perplexity']]}"
        )
    return selection_loss


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if config["audio"]["sample_rate"] != 16_000 or config["audio"]["frame_samples"] != 160:
        raise ValueError("the baseline configuration must remain fixed at 16 kHz / 10 ms")
    device, rank, world_size, distributed = distributed_context(args.device)
    is_main_process = rank == 0
    seed = int(config["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    train_set = make_dataset(args, "train", config)
    valid_set = make_dataset(args, "valid", config)
    train_sampler = (
        DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    valid_sampler = (
        DistributedSampler(valid_set, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else None
    )
    num_workers = (
        config["training"]["num_workers"]
        if args.num_workers is None
        else args.num_workers
    )
    loader_options = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        **loader_options,
    )
    valid_loader = DataLoader(
        valid_set,
        sampler=valid_sampler,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    base_model = CourseCodec(**config["model"]).to(device)
    base_discriminator = MultiResolutionSTFTDiscriminator(
        config["adversarial"]["resolutions"]
    ).to(device)
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location=device, weights_only=False)
        base_model.load_state_dict(resume_state["model"])
        if "discriminator" not in resume_state:
            raise ValueError("resume checkpoint does not contain the GAN discriminator")
        base_discriminator.load_state_dict(resume_state["discriminator"])
    model: torch.nn.Module = base_model
    discriminator: torch.nn.Module = base_discriminator
    if distributed:
        model = DDP(
            base_model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
        discriminator = DDP(
            base_discriminator,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
        )
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    discriminator_optimizer = AdamW(
        discriminator.parameters(),
        lr=config["adversarial"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    start_epoch = 1
    resume_batch_in_epoch = 0
    global_step = 0
    best_validation = float("inf")
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        discriminator_optimizer.load_state_dict(
            resume_state["discriminator_optimizer"]
        )
        start_epoch = int(resume_state["epoch"])
        resume_batch_in_epoch = int(resume_state.get("batch_in_epoch", 0))
        global_step = int(resume_state.get("global_step", 0))
        best_validation = float(resume_state.get("best_validation", best_validation))

    checkpoint_dir = Path(config["training"]["output_dir"]) / "checkpoints"
    if is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    epochs = args.epochs or config["training"]["epochs"]
    validation_limit = (
        config["training"]["validation_limit"]
        if args.max_valid_steps is None
        else args.max_valid_steps
    )
    interval_steps = config["training"]["validation_checkpoint_interval_steps"]
    if interval_steps <= 0:
        raise ValueError("validation_checkpoint_interval_steps must be positive")
    last_validation_step = -1
    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running: list[float] = []
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{epochs}",
            unit="batch",
            disable=not is_main_process,
        )
        last_batch_in_epoch = 0
        for step, reference in enumerate(progress):
            if epoch == start_epoch and step < resume_batch_in_epoch:
                continue
            if args.max_train_steps is not None and step >= args.max_train_steps:
                break
            reference = reference.to(device, non_blocking=True)
            reconstructed, _, quantizer_loss = model(reference)
            waveform_loss = F.l1_loss(reconstructed, reference)
            spectral_loss = multiscale_spectral_loss(
                reconstructed, reference, config["loss"]["fft_sizes"]
            )
            reconstruction_loss = (
                config["loss"]["waveform_l1_weight"] * waveform_loss
                + config["loss"]["spectral_weight"] * spectral_loss
            )
            gan_factor = adversarial_factor(global_step + 1, config)
            discriminator_loss = reconstructed.new_zeros(())
            adversarial_loss = reconstructed.new_zeros(())
            matching_loss = reconstructed.new_zeros(())
            if gan_factor > 0.0:
                discriminator.train()
                discriminator_optimizer.zero_grad(set_to_none=True)
                discriminator_input = torch.cat(
                    (reference, reconstructed.detach()), dim=0
                )
                combined_logits, _ = discriminator(discriminator_input)
                batch_size = reference.size(0)
                real_logits = [logits[:batch_size] for logits in combined_logits]
                fake_logits = [logits[batch_size:] for logits in combined_logits]
                discriminator_loss = discriminator_hinge_loss(real_logits, fake_logits)
                discriminator_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(),
                    config["adversarial"]["gradient_clip_norm"],
                )
                discriminator_optimizer.step()

                for parameter in base_discriminator.parameters():
                    parameter.requires_grad_(False)
                with torch.no_grad():
                    _, real_features = base_discriminator(reference)
                fake_logits, fake_features = base_discriminator(reconstructed)
                adversarial_loss = generator_hinge_loss(fake_logits)
                matching_loss = feature_matching_loss(real_features, fake_features)
                for parameter in base_discriminator.parameters():
                    parameter.requires_grad_(True)

            loss = (
                reconstruction_loss
                + quantizer_loss
                + gan_factor
                * (
                    config["loss"]["adversarial_weight"] * adversarial_loss
                    + config["loss"]["feature_matching_weight"] * matching_loss
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["training"]["gradient_clip_norm"]
            )
            optimizer.step()
            global_step += 1
            last_batch_in_epoch = step + 1
            running.append(loss.item())
            if is_main_process:
                progress.set_postfix(loss=f"{np.mean(running[-20:]):.4f}")
            if global_step % interval_steps == 0:
                validation = validate_and_checkpoint(
                    model=model,
                    base_model=base_model,
                    base_discriminator=base_discriminator,
                    valid_loader=valid_loader,
                    valid_set=valid_set,
                    device=device,
                    config=config,
                    validation_limit=validation_limit,
                    distributed=distributed,
                    is_main_process=is_main_process,
                    checkpoint_dir=checkpoint_dir,
                    epoch=epoch,
                    batch_in_epoch=last_batch_in_epoch,
                    global_step=global_step,
                    optimizer=optimizer,
                    discriminator_optimizer=discriminator_optimizer,
                    best_validation=best_validation,
                    train_loss=float(np.mean(running)),
                    write_snapshot=True,
                )
                best_validation = min(best_validation, validation)
                last_validation_step = global_step
                model.train()
                if distributed:
                    dist.barrier()
        resume_batch_in_epoch = 0
        # Always validate and preserve the final state, even if the final step
        # does not coincide with the 10k-step interval.
        if epoch == epochs and global_step != last_validation_step:
            validation = validate_and_checkpoint(
                model=model,
                base_model=base_model,
                base_discriminator=base_discriminator,
                valid_loader=valid_loader,
                valid_set=valid_set,
                device=device,
                config=config,
                validation_limit=validation_limit,
                distributed=distributed,
                is_main_process=is_main_process,
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                batch_in_epoch=last_batch_in_epoch,
                global_step=global_step,
                optimizer=optimizer,
                discriminator_optimizer=discriminator_optimizer,
                best_validation=best_validation,
                train_loss=float(np.mean(running)),
                write_snapshot=False,
            )
            best_validation = min(best_validation, validation)
            if distributed:
                dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
