"""Short eight-H100 checks using committed real C4 and pinned pretrained T5.

This is a disposable correctness/memory pilot, never an adaptation checkpoint
eligible for export. It executes the registered local batch and SIGReg counts.
"""
import argparse
import gc
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, T5EncoderModel, T5ForConditionalGeneration

from . import checkpoint
from .common import (ROOT, atomic_json, init_distributed, load_config, rank, read_json,
                     runtime_info, seed_all, state_hash)
from .corruption import SpanCollator
from .export import export_encoder, verify_reload
from .objectives import AdaptationObjective
from .train import component_grad_norms, optimizer_for, schedule_for, validate_production


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = init_distributed()
    torch.set_num_threads(4)
    shared, out = Path(args.shared), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "src/configs/t5_adaptation/arm_c.yml")
    validate_production(cfg, device)
    data_state = read_json(shared / "c4/train/preparation.json")
    if data_state["tokens"] < 4096 * 568:
        raise ValueError("Need a committed real C4 prefix for the pilot")
    prefix = np.memmap(shared / "c4/train/tokens.u16", dtype="<u2", mode="r", shape=(4096, 568))
    normalization = read_json(shared / "c4_normalization.json")
    if hashlib.sha256(prefix.tobytes()).hexdigest() != normalization["identity"]["data_prefix_sha256"]:
        raise ValueError("Calibration and pilot C4 prefixes differ")
    tokenizer = AutoTokenizer.from_pretrained(shared / "t5")
    report = {"status": "running", "purpose": "disposable correctness pilot, not experimental training",
              "runtime": runtime_info() if rank() == 0 else {}, "arms": {}}
    for arm in ("C", "D"):
        cfg = load_config(ROOT / f"src/configs/t5_adaptation/arm_{arm.lower()}.yml")
        seed_all(42)
        cls = T5ForConditionalGeneration if arm == "C" else T5EncoderModel
        model = cls.from_pretrained(shared / "t5").to(device)
        objective = AdaptationObjective(model, arm, normalization["mean"], normalization["std"]).to(device).train()
        wrapped = DDP(objective, device_ids=[device.index], broadcast_buffers=False)
        optimizer, scheduler = optimizer_for(objective, cfg), None
        scheduler = schedule_for(optimizer, cfg)
        seed_all(42 + rank())
        collator = SpanCollator(tokenizer)
        torch.cuda.reset_peak_memory_stats()
        steps = []
        for step in range(3):
            start = step * 512 + rank() * 64
            examples = [{"id": i, "input_ids": prefix[i].astype(np.int64)} for i in range(start, start + 64)]
            batch = {k: v.to(device) for k, v in collator(examples).items()}
            begin = time.monotonic()
            if step == 0:
                norms = component_grad_norms(wrapped, batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = wrapped(batch)
            if not torch.isfinite(result["loss"]):
                raise FloatingPointError("Pilot objective non-finite")
            result["loss"].backward()
            gradient = torch.nn.utils.clip_grad_norm_(objective.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            steps.append({"step": step + 1, "seconds": time.monotonic() - begin,
                          "task": result["task_global"].item(), "sigreg": result["sigreg"].item(),
                          "gradient_norm": gradient.item(), "samples_per_rank": 256})
        identity = {"arm": arm, "purpose": "pilot", "normalization": normalization["content_hash"]}
        checkpoint.save(out / f"{arm}-resume.pt", objective, optimizer, scheduler, identity, {"updates": 3})
        before = state_hash(objective.state_dict())
        counters = checkpoint.load(out / f"{arm}-resume.pt", objective, optimizer, scheduler, identity)
        assert state_hash(objective.state_dict()) == before and counters["updates"] == 3
        if rank() == 0:
            export_path = out / f"{arm}-test-export"
            export_encoder(objective, tokenizer, export_path)
            verification = verify_reload(objective, export_path, batch["clean"][:2], batch["clean_mask"][:2], device)
            report["arms"][arm] = {"steps": steps, "gradient_components": norms, "reload": verification,
                                    "max_allocated_memory_bytes": torch.cuda.max_memory_allocated(),
                                    "resume_verified": True}
            atomic_json(out / "report.json", report)
        dist.barrier()
        del result, wrapped, objective, model, optimizer, scheduler, batch
        gc.collect()
        torch.cuda.empty_cache()
    report["status"] = "passed"
    if rank() == 0:
        atomic_json(out / "report.json", report)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
