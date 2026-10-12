"""Submit exactly two independent eight-H100 pipelines under snow.research.er."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

import yaml

from .common import ROOT, atomic_json, file_hash, preparation_lock, read_json

ACCOUNT = "snow.research.er"
LIVE = {"QUEUING", "QUEUED", "RUNNING", "CANCELLING"}


def eai_json(arguments):
    if "--account" not in arguments or arguments[arguments.index("--account") + 1] != ACCOUNT:
        raise ValueError("Every EAI job command must explicitly use snow.research.er")
    output = subprocess.check_output(["eai", "job", *arguments, "--format", "json"], text=True)
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def snapshot_release(artifacts):
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit and verify project/submodule changes before taking a training release")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    path = artifacts / "releases" / commit
    if (path / "release.json").exists():
        return path
    path.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(["git", "archive", commit], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(path, filter="data")
    for name in ("transformers", "lejepa"):
        mount = path / "third_party" / name
        if mount.exists():
            mount.rmdir()
        mount.symlink_to(ROOT / "third_party" / name, target_is_directory=True)
    atomic_json(path / "release.json", {"commit": commit, "created": time.time(),
                                        "goal_sha256": file_hash(ROOT / "GOAL.md")})
    return path


def ready(shared, pilot):
    for relative in ("c4/train/manifest.json", "c4/validation/manifest.json", "owt_manifest.json"):
        if read_json(shared / relative)["status"] != "complete":
            raise ValueError(f"Shared preparation incomplete: {relative}")
    read_json(shared / "c4_normalization.json")
    read_json(shared / "evaluator/manifest.json")
    report = read_json(pilot)
    if report["status"] != "passed" or set(report["arms"]) != {"C", "D"}:
        raise ValueError("Both eight-H100 objective pilots must pass before full submission")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", default=str(ROOT / "cluster/eai.yml"))
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--resume", action="store_true", help="Replace terminal unsuccessful jobs using verified checkpoints")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cluster = yaml.safe_load(Path(args.cluster).read_text())
    if cluster["account"] != ACCOUNT or cluster["gpus_per_arm"] != 8 or cluster["gpu_model"] != "H100":
        raise ValueError("Cluster config violates required ER/eight-H100 allocation")
    artifacts = Path(cluster["storage"])
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts / "submission.json"
    with preparation_lock(artifacts / "submission.lock"):
        manifest = read_json(manifest_path) if manifest_path.exists() else {"created": time.time(), "arms": {}}
        if args.status:
            for arm, record in manifest["arms"].items():
                if record.get("jobs"):
                    job = eai_json(["get", record["jobs"][-1]["id"], "--account", ACCOUNT])[0]
                    record["scheduler_state"] = job["state"]
                    print(json.dumps({"arm": arm, "id": job["id"], "state": job["state"], "info": job["stateInfo"]}))
            atomic_json(manifest_path, manifest)
            return
        ready(artifacts / "shared", Path(args.pilot))
        release = snapshot_release(artifacts)
        for arm in ("C", "D"):
            record = manifest["arms"].setdefault(arm, {"jobs": [], "output": str(artifacts / arm)})
            name = f"elf_cd_{arm.lower()}_10b_20261012"
            # Recover a job created before a client crashed while writing the
            # manifest. Query authoritative scheduler state, not lock age.
            candidates = eai_json(["ls", "--account", ACCOUNT, "--name", name, "--state", "all"])
            candidates = [job for job in candidates if job["name"] == name]
            live = [job for job in candidates if job["state"] in LIVE]
            if len(live) > 1:
                raise ValueError(f"Duplicate active allocations already exist for arm {arm}")
            if live:
                job = live[0]
                if not any(j["id"] == job["id"] for j in record["jobs"]):
                    record["jobs"].append({"id": job["id"], "recovered": True})
                record["scheduler_state"] = job["state"]
                atomic_json(manifest_path, manifest)
                print(f"{arm}: existing {job['id']} {job['state']}")
                continue
            if Path(record["output"], "completed.json").exists():
                print(f"{arm}: pipeline already completed")
                continue
            if candidates and not args.resume:
                raise ValueError(f"Arm {arm} has a terminal prior job; inspect it, then use --resume")
            python = str(ROOT / ".venv-gpu/bin/python")
            command = ["eai", "job", "new", "--account", ACCOUNT, "--restartable", "--name", name,
                       "--image", cluster["image"], "--cpu", str(cluster["cpu_per_arm"]),
                       "--mem", str(cluster["mem_gb_per_arm"]), "--gpu", "8", "--gpu-model-filter", "H100",
                       "--max-run-time", "0", "--data", cluster["data"], "--workdir", str(release),
                       "--env", f"PYTHONPATH={release / 'src'}", "--env", "PYTHONDONTWRITEBYTECODE=1",
                       "--env", "OMP_NUM_THREADS=4", "--env", "MKL_NUM_THREADS=4",
                       "--env", "TOKENIZERS_PARALLELISM=false"]
            entry = [python, "-u", "-m", "t5_adaptation.pipeline", "--arm", arm,
                     "--shared", str(artifacts / "shared"), "--output", record["output"]]
            # Confirm the explicit account immediately before both dry-run and
            # real submission. There is no default-account code path.
            assert command[command.index("--account") + 1] == ACCOUNT
            spec = subprocess.check_output(command + ["--dry-run", "--", *entry], text=True)
            record["job_spec_yaml"] = spec
            record["release"] = str(release)
            record["pilot_sha256"] = file_hash(args.pilot)
            if args.dry_run:
                print(spec)
                continue
            output = subprocess.check_output(command + ["--format", "json", "--", *entry], text=True)
            job = json.loads(output)
            if job["account"] != ACCOUNT or job["resources"]["gpu"] != 8:
                raise RuntimeError("Scheduler accepted unexpected account or GPU count")
            record["jobs"].append({"id": job["id"], "created": job["created"], "release": str(release)})
            record["scheduler_state"] = job["state"]
            atomic_json(manifest_path, manifest)
            print(f"{arm}: submitted {job['id']} {job['state']}")


if __name__ == "__main__":
    main()
