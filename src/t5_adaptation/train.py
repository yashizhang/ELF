"""Resumable fixed-budget T5 adaptation. Launch with torchrun on eight H100s."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import signal
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5EncoderModel, T5ForConditionalGeneration

from . import checkpoint
from .common import (ROOT, append_json, atomic_json, file_hash, init_distributed, load_config,
                     object_hash, rank, read_json, runtime_info, seed_all, sources, state_hash, world_size)
from .corruption import SpanCollator
from .data import ContiguousRankBatches, PackedC4
from .diagnostics import diagnose
from .objectives import AdaptationObjective


def optimizer_for(model, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "norm" in name.lower() or "bias" in name.lower() else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg["learning_rate"], betas=tuple(cfg["betas"]), eps=cfg["epsilon"])


def schedule_for(optimizer, cfg):
    def multiplier(completed):
        # LambdaLR sets the LR for the *next* optimizer update, numbered from 1.
        update = completed + 1
        if update <= cfg["warmup_steps"]:
            return update / cfg["warmup_steps"]
        progress = min(1.0, (update - cfg["warmup_steps"]) / (cfg["updates"] - cfg["warmup_steps"]))
        floor = cfg["min_learning_rate"] / cfg["learning_rate"]
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def validate_production(cfg, device):
    if world_size() != 8 or cfg["world_size"] != 8:
        raise ValueError("The registered experiment requires eight ranks per arm")
    if device.type != "cuda" or torch.cuda.device_count() != 8:
        raise ValueError("Expected exactly eight allocated CUDA devices")
    if any("H100" not in torch.cuda.get_device_name(i) for i in range(8)):
        raise ValueError("The registered allocation must contain eight H100s")
    required = {"global_batch_size": 512, "local_batch_size": 64, "grad_accum_steps": 1,
                "sigreg_lambda": 0.05, "sigreg_samples_per_rank": 256, "sigreg_slices": 256,
                "sigreg_t_max": 3.0, "sigreg_n_points": 17, "sigreg_integration": "trapezoid",
                "source_length": 568, "nominal_length": 512, "nominal_budget": 10_000_000_000,
                "activation_checkpointing": False}
    for key, expected in required.items():
        if cfg[key] != expected:
            raise ValueError(f"Registered setting changed: {key}")


def component_grad_norms(wrapped, batch):
    result = {}
    # Both diagnostic passes also go through DDP. no_sync allows the two
    # component backwards; explicitly average their gradients for reporting.
    with wrapped.no_sync(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = wrapped(batch)
        for name, scale in (("task", 1.0), ("sigreg", 0.05)):
            wrapped.zero_grad(set_to_none=True)
            (output[name] * scale).backward(retain_graph=name == "task")
            squared = torch.zeros((), device=output["loss"].device)
            for p in wrapped.module.encoder.parameters():
                if p.grad is None:
                    continue
                grad = p.grad.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(grad)
                    grad /= world_size()
                squared += grad.float().square().sum()
            result[name + "_encoder_grad_norm"] = squared.sqrt().item()
    wrapped.zero_grad(set_to_none=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke-updates", type=int, default=0,
                        help="Separate disposable output; does not satisfy the adaptation budget")
    parser.add_argument("--stop-after", type=int, default=0, help="Checkpoint and exit after this many updates in this invocation")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = init_distributed()
    validate_production(cfg, device)
    torch.set_num_threads(4)
    seed_all(cfg["seed"])
    # FP32 parameters and optimizer state. Forward autocast only.
    torch.set_float32_matmul_precision("highest")
    shared, out = Path(args.shared), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dataset = PackedC4(shared / "c4/train")
    if len(dataset) != cfg["source_examples"]:
        raise ValueError("Prepared C4 prefix does not match the fixed budget")
    calibration = read_json(shared / "c4_normalization.json")
    if calibration["identity"]["data_identity"] != dataset.manifest["identity"]:
        raise ValueError("Calibration used a different C4 preparation")
    prefix_hash = hashlib.sha256(dataset.tokens[:4096 * cfg["source_length"]].tobytes()).hexdigest()
    if prefix_hash != calibration["identity"]["data_prefix_sha256"]:
        raise ValueError("Calibration prefix differs from the actual training data")
    original_manifest = read_json(shared / "t5/manifest.json")
    if calibration["identity"]["encoder_hash"] != original_manifest["content_hash"]:
        raise ValueError("C4 normalization was measured with a different original encoder")
    identity = {"arm": cfg["arm"], "config": cfg, "sources": sources(),
                "data_hash": dataset.manifest["content_hash"],
                "normalization_hash": calibration["content_hash"],
                "original_model_hash": original_manifest["content_hash"],
                "implementation": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
                "smoke": bool(args.smoke_updates)}
    if (out / "identity.json").exists() and read_json(out / "identity.json") != identity:
        raise ValueError("Output root belongs to a different experiment identity")
    if rank() == 0:
        atomic_json(out / "identity.json", identity)
        atomic_json(out / "runtime.json", runtime_info())
    cls = T5ForConditionalGeneration if cfg["arm"] == "C" else T5EncoderModel
    model = cls.from_pretrained(shared / "t5").to(device)
    if model.config.d_model != 512 or model.config.num_layers != 6 or model.config.vocab_size != 32128:
        raise ValueError("Expected the original T5-small architecture")
    initial_state_hash = state_hash({name: value for name, value in model.state_dict().items()
                                    if name.startswith("encoder.") or name == "shared.weight"})
    if initial_state_hash != calibration["model_state_hash"]:
        raise ValueError("Adaptation did not start from the original calibrated T5 encoder weights")
    tokenizer = AutoTokenizer.from_pretrained(shared / "t5")
    objective = AdaptationObjective(model, cfg["arm"], calibration["mean"], calibration["std"],
                                    cfg["projection_seed"], cfg["sigreg_slices"]).to(device).train()
    optimizer = optimizer_for(objective, cfg)
    scheduler = schedule_for(optimizer, cfg)
    counters = {"updates": 0, "source_examples": 0, "source_tokens": 0, "nominal_positions": 0,
                "decoder_target_tokens": 0, "corruption_target_tokens": 0, "noise_tokens": 0,
                "eligible_tokens": 0, "noise_spans": 0, "encoder_sequence_passes": 0,
                "diagnostic_encoder_sequence_passes": 0,
                "encoder_nonpadding_positions": 0, "wall_seconds": 0.0, "gpu_hours": 0.0}
    seed_all(cfg["seed"] + rank())
    latest = out / "latest.pt"
    if latest.exists():
        counters = checkpoint.load(latest, objective, optimizer, scheduler, identity)
    elif rank() == 0:
        atomic_json(out / "initial_encoder.json", {"state_hash": initial_state_hash,
                                                   "source": sources()["t5"]})
    wrapped = DDP(objective, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=True)
    start_update = counters["updates"]
    end_update = min(cfg["updates"], args.smoke_updates) if args.smoke_updates else cfg["updates"]
    if args.stop_after:
        end_update = min(end_update, start_update + args.stop_after)
    if start_update == cfg["updates"]:
        return
    collator = SpanCollator(tokenizer, cfg["corruption_seed"], cfg["sampling_seed"],
                           cfg["sigreg_samples_per_rank"], cfg["noise_density"], cfg["mean_noise_span_length"])
    batches = ContiguousRankBatches(end_update, cfg["local_batch_size"], rank(), world_size(), start_update)
    loader = DataLoader(dataset, batch_sampler=batches, collate_fn=collator,
                        num_workers=cfg["num_workers"], pin_memory=True,
                        generator=torch.Generator().manual_seed(cfg["seed"]),
                        persistent_workers=cfg["num_workers"] > 0)
    stopped = [False]
    def stop(signum, frame):
        stopped[0] = True
    for signum in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(signum, stop)
    invocation_start = time.monotonic()
    initial_wall = counters["wall_seconds"]
    initial_gpu = counters["gpu_hours"]
    if not start_update:
        summary = diagnose(objective, shared, tokenizer, cfg, device)
        counters["diagnostic_encoder_sequence_passes"] += cfg["global_batch_size"] * 4
        if rank() == 0:
            atomic_json(out / "diagnostics-000000.json", summary)
    milestone_updates = {math.ceil(n / (512 * 512)) for n in cfg["diagnostic_positions"]}
    last_time, last_step = time.monotonic(), start_update
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        diagnostic = counters["updates"] % cfg["gradient_diagnostic_every"] == 0
        # Preserve training RNGs/counters around the occasional diagnostic.
        grad_metrics = {}
        if diagnostic:
            from .common import rng_state, restore_rng
            saved_rng = rng_state()
            projection = objective.regularizer.global_step.clone()
            grad_metrics = component_grad_norms(wrapped, batch)
            restore_rng(saved_rng)
            objective.regularizer.global_step.copy_(projection)
            counters["diagnostic_encoder_sequence_passes"] += cfg["global_batch_size"] * 2
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = wrapped(batch)
        finite = torch.isfinite(output["loss"]).to(torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            if rank() == 0:
                atomic_json(out / "failure.json", {"reason": "nonfinite_loss", "update": counters["updates"],
                                                    "example_ids_rank0": batch["example_ids"].tolist()})
            raise FloatingPointError("Non-finite adaptation loss; previous checkpoint preserved")
        output["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(objective.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        counts = batch["counts"].clone()
        dist.all_reduce(counts)
        examples, raw_tokens, targets, noise, eligible, spans, encoder_positions = counts.tolist()
        counters["updates"] += 1
        counters["source_examples"] += examples
        counters["source_tokens"] += raw_tokens
        counters["nominal_positions"] += examples * cfg["nominal_length"]
        counters["corruption_target_tokens"] += targets
        counters["decoder_target_tokens"] += targets if cfg["arm"] == "C" else 0
        counters["noise_tokens"] += noise
        counters["eligible_tokens"] += eligible
        counters["noise_spans"] += spans
        counters["encoder_sequence_passes"] += examples * 2
        counters["encoder_nonpadding_positions"] += encoder_positions
        step = counters["updates"]
        counters["wall_seconds"] = initial_wall + time.monotonic() - invocation_start
        counters["gpu_hours"] = initial_gpu + (time.monotonic() - invocation_start) * world_size() / 3600
        if step % cfg["log_every"] == 0 or step == start_update + 1 or diagnostic:
            now = time.monotonic()
            seconds_per_update = (now - last_time) / (step - last_step)
            entry = {"time": time.time(), **counters, "task_loss": output["task_global"].item(),
                     "sigreg": output["sigreg"].item(), "weighted_sigreg": 0.05 * output["sigreg"].item(),
                     "loss": output["task_global"].item() + 0.05 * output["sigreg"].item(),
                     "grad_norm": grad_norm.item(), "lr_next_update": optimizer.param_groups[0]["lr"],
                     "seconds_per_update": seconds_per_update,
                     "eta_seconds": (cfg["updates"] - step) * seconds_per_update,
                     "max_memory_bytes": torch.cuda.max_memory_allocated(), **grad_metrics}
            if rank() == 0:
                append_json(out / "train.jsonl", entry)
                atomic_json(out / "status.json", {"state": "running", **entry})
                print(entry, flush=True)
            last_time, last_step = now, step
        signal_flag = torch.tensor(int(stopped[0]), device=device)
        dist.all_reduce(signal_flag, op=dist.ReduceOp.MAX)
        at_end = step == end_update
        milestone = step in milestone_updates
        if milestone:
            summary = diagnose(objective, shared, tokenizer, cfg, device)
            counters["diagnostic_encoder_sequence_passes"] += cfg["global_batch_size"] * 4
            if rank() == 0:
                atomic_json(out / f"diagnostics-{step:06d}.json", summary)
        if step % cfg["save_every"] == 0 or milestone or at_end or signal_flag.item():
            counters["wall_seconds"] = initial_wall + time.monotonic() - invocation_start
            counters["gpu_hours"] = initial_gpu + (time.monotonic() - invocation_start) * world_size() / 3600
            checkpoint.save(latest, objective, optimizer, scheduler, identity, counters,
                            final=step == cfg["updates"] and not args.smoke_updates)
            if rank() == 0 and milestone:
                for extension in (".pt", ".json"):
                    target = out / f"milestone-{step:06d}{extension}"
                    if not target.exists():
                        os.link(out / ("latest" + extension), target)
            if signal_flag.item():
                if rank() == 0:
                    atomic_json(out / "status.json", {"state": "preempted", **counters})
                raise SystemExit(75)
        if at_end:
            if rank() == 0:
                atomic_json(out / "status.json", {"state": "smoke_complete" if args.smoke_updates else
                            "completed" if step == cfg["updates"] else "checkpointed", **counters})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
