"""Small provenance, atomic I/O and reproducibility helpers."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_json(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        f.flush()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        value = value.detach().cpu().contiguous()
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def directory_manifest(path):
    path = Path(path)
    files = {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob("*"))
             if p.is_file() and p.name != "manifest.json" and ".cache" not in p.parts}
    return {"files": files, "content_hash": object_hash(files)}


def sources():
    return read_json(ROOT / "src/configs/t5_adaptation/sources.json")


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if "base_config" in cfg:
        base = yaml.safe_load((ROOT / cfg.pop("base_config")).read_text())
        cfg = base | cfg
    cfg["updates"] = (cfg["nominal_budget"] + cfg["global_batch_size"] * cfg["nominal_length"] - 1) // (
        cfg["global_batch_size"] * cfg["nominal_length"])
    cfg["source_examples"] = cfg["updates"] * cfg["global_batch_size"]
    return cfg


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def init_distributed():
    local = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextlib.contextmanager
def preparation_lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def runtime_info():
    import importlib.metadata
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    if (ROOT / "release.json").exists():
        commit = read_json(ROOT / "release.json")["commit"]
    elif shutil.which("git"):
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    else:
        head = (ROOT / ".git/HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            ref_file = ROOT / ".git" / ref
            if ref_file.exists():
                commit = ref_file.read_text().strip()
            else:
                commit = next(line.split()[0] for line in (ROOT / ".git/packed-refs").read_text().splitlines()
                              if line.endswith(" " + ref))
        else:
            commit = head
    info = {"time": time.time(), "packages": packages, "torch": torch.__version__,
            "cuda": torch.version.cuda, "git_commit": commit,
            "project_python_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT / "src").rglob("*.py")}}
    info["execution_environment"] = {key: os.environ[key] for key in
                                     ("CC", "CXX", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
                                     if key in os.environ}
    if os.environ.get("CXX"):
        info["compiler_version"] = subprocess.check_output([os.environ["CXX"], "--version"], text=True).splitlines()[0]
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["gpu_memory"] = [torch.cuda.get_device_properties(i).total_memory
                              for i in range(torch.cuda.device_count())]
        info["topology"] = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    return info
