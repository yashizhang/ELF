import logging
import os
import re
import uuid
from typing import Any, Optional, Tuple

import torch

from utils.logging_utils import log_for_0, _process_index
from utils.train_utils import unwrap_model


def _local_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def upload_output_dir_to_hf(output_dir: str, hf_repo_id: Optional[str], reason: str = "artifacts"):
    if not hf_repo_id or _process_index() != 0:
        return
    folder_path = _local_path(output_dir)
    if not os.path.isdir(folder_path):
        log_for_0(f"HF upload skipped; output directory does not exist: {folder_path}",
                  level=logging.WARNING)
        return
    try:
        from huggingface_hub import HfApi
        repo_id = hf_repo_id.strip("/")
        api = HfApi()
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        log_for_0(f"Uploading {reason} to HF: {repo_id}")
        api.upload_folder(repo_id=repo_id, folder_path=folder_path, repo_type="model")
        log_for_0(f"Uploaded {reason} to HF: {repo_id}")
    except Exception as e:
        log_for_0(f"Failed to upload {reason} to HF: {e}", level=logging.WARNING)


def _split_hf_path(path: str, min_parts: int) -> Optional[Tuple[str, str]]:
    if "://" in path:
        return None
    if path.startswith(("/", ".", "~")):
        return None
    if os.path.exists(_local_path(path)):
        return None
    parts = path.split("/")
    if len(parts) < min_parts:
        return None
    return "/".join(parts[:2]), "/".join(parts[2:])


def save_checkpoint(state, output_dir: str, step: int, hf_repo_id: str = None):
    """Save model checkpoint locally as a single `checkpoint_<step>` file."""
    rank_rngs = None
    rank_optimizer_files = None
    if state.experiment_identity is not None:
        import torch.distributed as dist
        from t5_adaptation.common import rng_state, file_hash
        # Upstream distributed Muon shards optimizer momentum across ranks.
        # Rank zero's optimizer state alone cannot resume the other shards.
        os.makedirs(output_dir, exist_ok=True)
        # Repeated saves at an epoch/final boundary must leave the optimizer
        # shards referenced by the previous atomic main checkpoint untouched.
        optimizer_name = f"optimizer_{step}_rank{_process_index()}_{uuid.uuid4().hex}.pt"
        optimizer_path = os.path.join(output_dir, optimizer_name)
        with open(optimizer_path + ".tmp", "wb") as f:
            torch.save(state.optimizer.state_dict(), f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(optimizer_path + ".tmp", optimizer_path)
        local_rng = rng_state()
        local_rng["dropout_generator"] = state.dropout_generator.get_state()
        local_rng["optimizer_file"] = optimizer_name
        local_rng["optimizer_sha256"] = file_hash(optimizer_path)
        rank_rngs = [None] * (dist.get_world_size() if dist.is_initialized() else 1)
        if dist.is_initialized():
            dist.all_gather_object(rank_rngs, local_rng)
        else:
            rank_rngs[0] = local_rng
        rank_optimizer_files = [r["optimizer_file"] for r in rank_rngs]
    if _process_index() != 0:
        return
    ckpt_dir = _local_path(output_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    inner_model = unwrap_model(state.model)
    grad_accum_buffers = {}
    if getattr(state, "grad_accum_buffers", None):
        for name, param in inner_model.named_parameters():
            buf = state.grad_accum_buffers.get(id(param))
            if buf is not None:
                grad_accum_buffers[name] = buf.detach().cpu()
    payload = {
        "params": inner_model.state_dict(),
        "ema_params1": state.ema_params1,
        "opt_state": state.optimizer.state_dict(),
        "lr_scheduler": state.lr_scheduler.state_dict() if state.lr_scheduler is not None else None,
        "step": int(state.step),
        "epoch": float(state.epoch),
        "dropout_rng": (state.dropout_generator.get_state()
                        if state.dropout_generator is not None else None),
        "grad_accum_buffers": grad_accum_buffers,
        "experiment_identity": state.experiment_identity,
        "rank_rngs": rank_rngs,
        "rank_optimizer_files": rank_optimizer_files,
    }
    out_path = os.path.join(ckpt_dir, f"checkpoint_{step}")
    log_for_0(f"Saving checkpoint to {out_path}")
    temporary = out_path + ".tmp"
    with open(temporary, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, out_path)
    log_for_0(f"Checkpoint written to {out_path}")
    upload_output_dir_to_hf(output_dir, hf_repo_id, reason="checkpoint")


def _checkpoint_step(checkpoint_name: str) -> int:
    """Extract the trailing checkpoint step from a name; -1 if absent."""
    match = re.search(r"(\d+)$", checkpoint_name)
    return int(match.group(1)) if match else -1


def find_all_checkpoints(ckpt_dir: str, prefix: str = "checkpoint_"):
    """Find local checkpoint paths in a directory, sorted by step ascending."""
    ckpt_dir = _local_path(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        return []
    names = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith(prefix) and _checkpoint_step(f) >= 0],
        key=_checkpoint_step,
    )
    return [os.path.join(ckpt_dir, name) for name in names]


def find_latest_checkpoint(ckpt_dir: str, prefix: str = "checkpoint_"):
    """Return the latest local checkpoint path, or None."""
    all_ckpts = find_all_checkpoints(ckpt_dir, prefix)
    return all_ckpts[-1] if all_ckpts else None


def _download_hf_checkpoint(checkpoint_path: str) -> Optional[str]:
    hf_path = _split_hf_path(checkpoint_path, min_parts=2)
    if hf_path is None:
        return None
    repo_id, sub_path = hf_path
    from huggingface_hub import snapshot_download
    log_for_0(f"Downloading checkpoint from HF: {repo_id}" + (f"/{sub_path}" if sub_path else ""))
    local_dir = snapshot_download(
        repo_id=repo_id, repo_type="model",
        allow_patterns=[f"{sub_path}/**"] if sub_path else None,
    )
    return os.path.join(local_dir, sub_path) if sub_path else local_dir


def _restore_checkpoint(checkpoint_path: str) -> Any:
    """Restore a checkpoint from a file or directory (latest inside dir)."""
    local = _local_path(checkpoint_path)
    resolved = local
    if os.path.isdir(local):
        latest = find_latest_checkpoint(local)
        if latest is not None and os.path.isfile(latest):
            resolved = latest
    if os.path.isfile(resolved):
        return torch.load(resolved, map_location="cpu", weights_only=False)
    return None


def _validate_checkpoint(ckpt: Any):
    if ckpt is None:
        raise ValueError("checkpoint restore returned None")
    required_keys = ("params", "opt_state", "step", "epoch")
    missing_keys = [key for key in required_keys if key not in ckpt]
    if missing_keys:
        raise ValueError(f"checkpoint restore missing keys: {missing_keys}")


def load_checkpoint(checkpoint_path: str, state) -> Tuple[Any, int]:
    """Load an ELF checkpoint.

    Uses an existing local path first; otherwise tries HF and then local fallback.
    """
    log_for_0(f"Loading ELF checkpoint from {checkpoint_path}...")
    ckpt, loaded_from = None, None
    errors = []

    local_path = _local_path(checkpoint_path)
    if os.path.exists(local_path):
        try:
            log_for_0(f"Loading local checkpoint from {local_path}...")
            ckpt = _restore_checkpoint(local_path)
            _validate_checkpoint(ckpt)
            loaded_from = "local"
        except Exception as e:
            errors.append(f"local: {e}")

    if ckpt is None:
        try:
            hf_path = _download_hf_checkpoint(checkpoint_path)
            if hf_path:
                log_for_0(f"Loading HF checkpoint from {hf_path}...")
                ckpt = _restore_checkpoint(hf_path)
                _validate_checkpoint(ckpt)
                loaded_from = "HF"
        except Exception as e:
            errors.append(f"HF: {e}")
            log_for_0(f"HF checkpoint restore failed ({e}); falling back to local path.")

    if ckpt is None:
        raise ValueError(
            f"Failed to load checkpoint from {checkpoint_path}. Tried: {'; '.join(errors)}"
        )

    log_for_0(f"Loaded checkpoint keys: {list(ckpt.keys())}")

    inner_model = unwrap_model(state.model)
    inner_model.load_state_dict(ckpt["params"])
    ema_src = ckpt.get("ema_params1", ckpt["params"])
    device_map = {n: p.device for n, p in inner_model.named_parameters()}
    for n, b in inner_model.named_buffers():
        device_map.setdefault(n, b.device)
    fallback_device = next(iter(device_map.values()), torch.device("cpu"))
    state.ema_params1 = {
        n: t.to(device_map.get(n, fallback_device)) for n, t in ema_src.items()
    }
    if state.experiment_identity is not None:
        if ckpt.get("experiment_identity") != state.experiment_identity:
            raise ValueError("ELF checkpoint provenance differs from the requested arm/encoder/data")
        from t5_adaptation.common import file_hash
        local_saved = ckpt["rank_rngs"][_process_index()]
        shard_path = os.path.join(os.path.dirname(local_path), local_saved["optimizer_file"])
        if os.path.isdir(local_path):
            shard_path = os.path.join(local_path, local_saved["optimizer_file"])
        if file_hash(shard_path) != local_saved["optimizer_sha256"]:
            raise ValueError("ELF optimizer shard is corrupt")
        state.optimizer.load_state_dict(torch.load(shard_path, map_location="cpu", weights_only=False))
    else:
        state.optimizer.load_state_dict(ckpt["opt_state"])
    if state.lr_scheduler is not None and ckpt.get("lr_scheduler") is not None:
        state.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    state.step = int(ckpt["step"])
    state.epoch = float(ckpt["epoch"])
    if state.experiment_identity is not None:
        if ckpt.get("experiment_identity") != state.experiment_identity:
            raise ValueError("ELF checkpoint provenance differs from the requested arm/encoder/data")
        import torch.distributed as dist
        from t5_adaptation.common import restore_rng
        states = ckpt["rank_rngs"]
        world = dist.get_world_size() if dist.is_initialized() else 1
        if len(states) != world:
            raise ValueError("ELF checkpoint world size mismatch")
        local_rng = states[_process_index()]
        restore_rng(local_rng)
        state.dropout_generator.set_state(local_rng["dropout_generator"])
    if state.experiment_identity is None and ckpt.get("dropout_rng") is not None and state.dropout_generator is not None:
        try:
            state.dropout_generator.set_state(ckpt["dropout_rng"])
        except Exception:
            pass
    if ckpt.get("grad_accum_buffers"):
        buffers = ckpt["grad_accum_buffers"]
        state.grad_accum_buffers = {}
        param_ids = []
        for name, param in inner_model.named_parameters():
            if not param.requires_grad:
                continue
            param_ids.append(id(param))
            saved = buffers.get(name)
            state.grad_accum_buffers[id(param)] = (
                saved.to(device=param.device, dtype=param.dtype)
                if saved is not None else torch.zeros_like(param)
            )
        state.grad_accum_param_ids = tuple(param_ids)

    step = int(ckpt["step"])
    log_for_0(f"Loaded {loaded_from} checkpoint from step {step} (epoch {state.epoch})")
    return state, step
