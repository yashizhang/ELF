"""Fixed held-out diagnostics; preserve training RNG and projection counters."""
from pathlib import Path

import torch
import torch.distributed as dist

from .common import rank, restore_rng, rng_state, world_size
from .corruption import SpanCollator
from .data import PackedC4
from .objectives import dropout_disabled


def covariance_spectrum(z):
    centered = z - z.mean(0)
    eigen = torch.linalg.eigvalsh(centered.T @ centered / len(z)).clamp_min(0)
    total = eigen.sum()
    if total.item() == 0:
        # A constant representation is a valid negative diagnostic result.
        # Avoid 0/0 probabilities that would abort a finite training run.
        return eigen, 0.0
    probabilities = eigen / total
    effective_rank = (-(probabilities * probabilities.clamp_min(1e-30).log()).sum()).exp().item()
    return eigen, effective_rank


@torch.no_grad()
def diagnose(objective, shared, tokenizer, cfg, device):
    rng = rng_state()
    projection_state = objective.regularizer.global_step.clone()
    results = {}
    try:
        for source_length in (568, 1023):
            data = PackedC4(Path(shared) / "c4/validation", source_length=source_length,
                            count=cfg["global_batch_size"])
            start = rank() * cfg["local_batch_size"]
            examples = [data[i] for i in range(start, start + cfg["local_batch_size"])]
            collator = SpanCollator(tokenizer, cfg["corruption_seed"], cfg["sampling_seed"],
                                    cfg["sigreg_samples_per_rank"])
            batch = {k: v.to(device) for k, v in collator(examples).items()}
            objective.regularizer.global_step.fill_(cfg["projection_seed"])
            with dropout_disabled(objective), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                              enabled=device.type == "cuda"):
                output = objective(batch)
            raw = output["clean_sample_raw"]
            pieces = [torch.empty_like(raw) for _ in range(world_size())]
            if dist.is_initialized():
                dist.all_gather(pieces, raw)
            else:
                pieces[0] = raw
            # The prescribed statistic is evaluated with exactly 2048 samples,
            # even for D: do not report its clean+corrupt mean as clean SIGReg.
            objective.regularizer.global_step.fill_(cfg["projection_seed"])
            with torch.autocast(device_type=device.type, enabled=False):
                clean_sigreg = objective.regularizer(output["clean_sample_normalized"].float()).item()
            if rank() == 0:
                raw = torch.cat(pieces).double().cpu()
                z = (raw - objective.mu0.item()) / objective.s0.item()
                eigen, effective_rank = covariance_spectrum(z)
                results[str(source_length + 1)] = {
                    "task_name": "span_ce" if objective.arm == "C" else "aligned_consistency",
                    "task_loss": output["task_global"].item(), "clean_sigreg": clean_sigreg,
                    "sample_count": len(raw), "raw_mean": raw.mean().item(), "raw_variance": raw.var(unbiased=False).item(),
                    "normalized_mean": z.mean().item(), "normalized_variance": z.var(unbiased=False).item(),
                    "effective_rank": effective_rank, "covariance_eigenvalues": eigen.tolist(),
                    "finite_fraction": torch.isfinite(raw).double().mean().item(),
                    "statistics_scope": "four ordinary token samples per held-out source sequence"}
    finally:
        objective.regularizer.global_step.copy_(projection_state)
        restore_rng(rng)
    return results
