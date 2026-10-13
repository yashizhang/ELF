"""One independent eight-H100 arm: adapt, export, normalize, ELF, evaluate."""
import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import torch
import yaml

from .common import (ROOT, append_json, atomic_json, file_hash, object_hash,
                     preparation_lock, read_json, runtime_info, sources)


def make_elf_config(arm, shared, arm_root):
    norm = read_json(arm_root / "owt_normalization.json")
    export = read_json(arm_root / "encoder/manifest.json")
    data = read_json(shared / "owt_manifest.json")
    if norm["identity"]["encoder_hash"] != export["content_hash"]:
        raise ValueError("OWT calibration does not belong to this encoder")
    identity = {"arm": arm, "encoder_hash": export["content_hash"], "data_hash": data["content_hash"],
                "data_examples": data["examples"], "latent_mean": norm["mean"], "latent_std": norm["std"],
                "normalization_hash": norm["content_hash"], "source_adaptation": export["provenance"],
                "epochs": 5, "seed": 42, "global_batch_size": 512, "grad_accum_steps": 1,
                "updates_per_epoch": math.ceil(data["examples"] / 512),
                "full_epoch_policy": "deterministic shuffled full dataset, pad each final batch with prefix repeats",
                "padding_repeats_per_epoch": math.ceil(data["examples"] / 512) * 512 - data["examples"]}
    atomic_json(arm_root / "elf_identity.json", identity)
    cfg = yaml.safe_load((ROOT / "src/configs/training_configs/train_owt_ELF-B.yml").read_text())
    cfg.update(encoder_model_name=str(arm_root / "encoder"), tokenizer_name=str(arm_root / "encoder"),
               latent_mean=norm["mean"], latent_std=norm["std"], data_path=str(shared / "owt"),
               output_dir=str(arm_root / "elf"), eval_freq=0, hf_repo_id=None, use_wandb=False,
               wandb_run_name=f"t5-sigreg-{arm}-elf", resume=None,
               experiment_manifest=str(arm_root / "elf_identity.json"), full_epochs=True,
               save_every_steps=500, training_smoke_sample=True,
               sampling_configs_path=str(ROOT / "src/configs/sampling_configs/uncond_sampling_configs.yml"))
    path = arm_root / "elf.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path, cfg


def run_stage(name, command, arm_root, verify, allocated_gpus=8):
    status_path = arm_root / f"stage-{name}.json"
    if status_path.exists() and read_json(status_path)["state"] == "completed":
        verify()
        return
    started = time.time()
    attempt_path = arm_root / "stage_attempts" / f"{name}-{uuid.uuid4().hex}.json"
    record = {"stage": name, "state": "running", "started": started, "command": command,
              "allocated_gpus": allocated_gpus, "source_commit": runtime_info()["git_commit"]}
    def persist():
        # Keep the start record even if the process is killed before it can
        # append an end event, or a scheduler restart replaces stage status.
        atomic_json(attempt_path, record)
        atomic_json(status_path, record)
    persist()
    child = None
    interrupted = [False]
    def terminate(signum, frame):
        interrupted[0] = True
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    with (arm_root / f"{name}.log").open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        code = child.wait()
    ended = time.time()
    record.update(ended=ended, wall_seconds=ended - started, gpu_hours=(ended - started) * allocated_gpus / 3600,
                  exit_code=code, interrupted=interrupted[0],
                  state="completed" if code == 0 else "preempted" if interrupted[0] or code == 75 else "failed")
    if code == 0:
        try:
            verify()
        except Exception as exc:
            record.update(state="failed", verification_error=str(exc))
            persist()
            append_json(arm_root / "compute.jsonl", record)
            raise
    persist()
    append_json(arm_root / "compute.jsonl", record)
    if interrupted[0] or code == 75:
        raise SystemExit(75)
    if code:
        raise SystemExit(1)


def require_json(path, field=None, value=None):
    record = read_json(path)
    if field is not None and record[field] != value:
        raise ValueError(f"Stage output failed verification: {path}, {field}")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["C", "D"], required=True)
    parser.add_argument("--shared", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    shared, out = Path(args.shared).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if torch.cuda.device_count() != 8 or any("H100" not in torch.cuda.get_device_name(i) for i in range(8)):
        raise ValueError("Pipeline requires its own scheduler allocation of eight H100s")
    require_json(shared / "c4/train/manifest.json", "status", "complete")
    require_json(shared / "c4/validation/manifest.json", "status", "complete")
    require_json(shared / "owt_manifest.json", "status", "complete")
    require_json(shared / "c4_normalization.json")
    python = sys.executable
    distributed = [python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=8"]
    with preparation_lock(out / "pipeline.lock"):
        atomic_json(out / "runtime.json", runtime_info())
        adapt = out / "adaptation"
        run_stage("adaptation", distributed + ["-m", "t5_adaptation.train", "--config",
                  str(ROOT / f"src/configs/t5_adaptation/arm_{args.arm.lower()}.yml"),
                  "--shared", str(shared), "--output", str(adapt)], out,
                  lambda: require_json(adapt / "latest.json", "final", True))
        run_stage("export", [python, "-m", "t5_adaptation.export", "--checkpoint", str(adapt / "latest.pt"),
                  "--shared", str(shared), "--output", str(out / "encoder")], out,
                  lambda: require_json(out / "encoder/manifest.json"))
        run_stage("owt_normalization", [python, "-m", "t5_adaptation.calibrate", "--shared", str(shared),
                  "--kind", "owt", "--encoder", str(out / "encoder"), "--output", str(out / "owt_normalization.json")], out,
                  lambda: require_json(out / "owt_normalization.json"))
        # The first arm to reach this stage runs the shared original probe.
        # A busy lock never makes one arm wait for the other.
        try:
            with preparation_lock(shared / "original_probe.lock"):
                if not (shared / "original_probe.json").exists():
                    run_stage("original_readout", [python, "-m", "t5_adaptation.probe", "--shared", str(shared),
                              "--encoder", str(shared / "t5"), "--output", str(shared / "original_probe.json")], out,
                              lambda: require_json(shared / "original_probe.json", "status", "completed"))
        except BlockingIOError:
            pass
        run_stage("readout", [python, "-m", "t5_adaptation.probe", "--shared", str(shared),
                  "--encoder", str(out / "encoder"), "--output", str(out / "readout.json")], out,
                  lambda: require_json(out / "readout.json", "status", "completed"))
        cfg_path, cfg = make_elf_config(args.arm, shared, out)
        run_stage("elf", distributed + [str(ROOT / "src/train.py"), "--config", str(cfg_path)], out,
                  lambda: require_json(out / "elf/completed.json", "epochs", 5))
        # Both jobs read the evaluator prepared before submission; neither arm
        # needs to wait for the other arm to download shared artifacts.
        require_json(shared / "evaluator/manifest.json", "status", "complete")
        completed = read_json(out / "elf/completed.json")
        final = out / "elf" / completed["final_checkpoint"]
        eval_cfg = cfg | {"output_dir": str(out / "samples"), "online_eval": False,
                          "eval_ppl_model": str(shared / "evaluator"), "num_samples": 1000,
                          "use_compile": True}
        eval_path = out / "evaluation.yml"
        eval_path.write_text(yaml.safe_dump(eval_cfg, sort_keys=False))
        def verify_samples():
            for seed in (0, 1, 2):
                paths = list((out / "samples" / f"seed_{seed}").glob("*/all_generated_*.jsonl"))
                if len(paths) != 2 or any(len(p.read_text().splitlines()) != 1000 for p in paths):
                    raise ValueError("Missing required generation samples")
        run_stage("generation", distributed + [str(ROOT / "src/eval.py"), "--config", str(eval_path),
                  "--checkpoint_path", str(final), "--seeds", "0,1,2"], out, verify_samples)
        run_stage("metrics", [python, "-m", "t5_adaptation.evaluate", "--samples-root", str(out / "samples"),
                  "--shared", str(shared), "--arm", args.arm, "--output", str(out / "results.json")], out,
                  lambda: require_json(out / "results.json", "status", "completed"))
        atomic_json(out / "completed.json", {"arm": args.arm, "state": "completed", "results": str(out / "results.json"),
                                              "final_checkpoint_sha256": file_hash(final)})


if __name__ == "__main__":
    main()
