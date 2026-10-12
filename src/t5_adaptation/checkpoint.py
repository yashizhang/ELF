"""Atomic, identity-checked adaptation checkpoints with per-rank RNG state."""
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .common import atomic_json, file_hash, rank, restore_rng, rng_state, world_size


def save(path, model, optimizer, scheduler, identity, counters, *, final=False):
    path = Path(path)
    states = [None] * world_size()
    local_rng = rng_state()
    if dist.is_initialized():
        dist.all_gather_object(states, local_rng)
    else:
        states[0] = local_rng
    if rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"identity": identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "counters": counters,
                   "rank_rngs": states, "world_size": world_size(), "final": final}
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        atomic_json(path.with_suffix(".json"), {"identity": identity, "counters": counters,
                                               "final": final, "sha256": file_hash(path)})
    if dist.is_initialized():
        dist.barrier()


def load(path, model, optimizer, scheduler, identity):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["identity"] != identity:
        raise ValueError("Refusing adaptation resume: arm/objective/data/calibration/source identity differs")
    if payload["world_size"] != world_size():
        raise ValueError("Cannot silently change SIGReg sample/world semantics on resume")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng(payload["rank_rngs"][rank()])
    return payload["counters"]
