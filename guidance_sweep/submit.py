"""Submit two independent inference allocations per phase, explicitly in ER."""
import argparse
from datetime import datetime, timezone
import io
from pathlib import Path
import subprocess
import tarfile

import yaml
from t5_adaptation.common import atomic_json, file_hash, preparation_lock, read_json

ACCOUNT = "snow.research.er"
LIVE = {"QUEUING", "QUEUED", "RUNNING", "CANCELLING"}


def query(arguments):
    assert "--account" in arguments and arguments[arguments.index("--account") + 1] == ACCOUNT
    raw = subprocess.check_output(["eai", "job", *arguments, "--format", "json"], text=True)
    import json
    return [json.loads(line) for line in raw.splitlines() if line.strip().startswith("{")]


def snapshot(repo, root, commit=None):
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip(), "Commit study changes before submission"
    commit = subprocess.check_output(["git", "rev-parse", commit or "HEAD"], cwd=repo, text=True).strip()
    release = root / "releases" / commit
    if not (release / "release.json").exists():
        release.mkdir(parents=True, exist_ok=True)
        archive = subprocess.check_output(["git", "archive", commit], cwd=repo)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(release, filter="data")
        for name in ("transformers", "lejepa"):
            path = release / "third_party" / name
            if path.exists():
                path.rmdir()
            path.symlink_to(repo / "third_party" / name, target_is_directory=True)
        atomic_json(release / "release.json", {"commit": commit, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    return commit, release


def statuses(root, manifest):
    jobs = []
    total = 0
    for record in manifest["jobs"]:
        job = query(["get", record["id"], "--account", ACCOUNT])[0]
        assert job["account"] == ACCOUNT
        safe = {"arm": record["arm"], "phase": record["phase"], **{k: job[k] for k in ["id", "name", "account", "state", "resources"]}, "runs": []}
        for run in job["runs"]:
            row = {k: run[k] for k in ["id", "state", "nodeName", "startedOn", "endedOn", "exitCode", "stateInfo"]}
            if row["startedOn"]:
                begin = datetime.fromisoformat(row["startedOn"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(row["endedOn"].replace("Z", "+00:00")) if row["endedOn"] else datetime.now(timezone.utc)
                row["allocated_gpu_hours"] = (end - begin).total_seconds() * job["resources"]["gpu"] / 3600
            else:
                row["allocated_gpu_hours"] = 0
            row["provisional"] = row["endedOn"] is None
            total += row["allocated_gpu_hours"]
            safe["runs"].append(row)
        jobs.append(safe)
    result = {"observed_at_utc": datetime.now(timezone.utc).isoformat(), "jobs": jobs, "allocated_gpu_hours": total,
              "accounting": "Includes all run attempts; stage timing is nested within these allocations and is not added again. Queued time excluded."}
    atomic_json(root / "scheduler.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["screening", "confirmation"], default="screening")
    parser.add_argument("--root", default="artifacts/guidance_sweep_20261015")
    parser.add_argument("--name-suffix", default="20261015")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo, root = Path.cwd(), Path(args.root).resolve()
    cluster = yaml.safe_load((repo / "cluster/eai.yml").read_text())
    assert cluster["account"] == ACCOUNT and cluster["gpus_per_arm"] == 8
    assert cluster["gpu_model"] == "H100" and cluster["cpu_per_arm"] == 64 and cluster["mem_gb_per_arm"] == 512
    manifest_path = root / "submission.json"
    with preparation_lock(root / "submission.lock"):
        manifest = read_json(manifest_path) if manifest_path.exists() else {"jobs": []}
        if args.status:
            result = statuses(root, manifest)
            print({"gpu_hours": result["allocated_gpu_hours"], "jobs": [{k: j[k] for k in ["arm", "phase", "id", "state"]} for j in result["jobs"]]})
            return
        if args.phase == "confirmation":
            result = statuses(root, manifest)
            for arm in ("C", "D"):
                assert read_json(root / arm / "screening_job_completed.json")["state"] == "completed"
                assert any(j["arm"] == arm and j["phase"] == "screening" and j["state"] == "SUCCEEDED" for j in result["jobs"])
            selection = read_json(repo / "guidance_sweep/selection.json")
            assert selection["inputs_sha256"] == file_hash(repo / "guidance_sweep/inputs.json")
            assert set(selection["arms"]) == {"C", "D"}
        for arm in ("C", "D"):
            name = f"elf_guidance_{arm.lower()}_{args.phase}_{args.name_suffix}"
            existing = [j for j in query(["ls", "--account", ACCOUNT, "--name", name, "--state", "all"]) if j["name"] == name]
            live = [j for j in existing if j["state"] in LIVE]
            assert len(live) <= 1, "Duplicate live jobs"
            for job in existing:
                if not any(r["id"] == job["id"] for r in manifest["jobs"]):
                    manifest["jobs"].append({"arm": arm, "phase": args.phase, "id": job["id"], "recovered": True})
            if live or (root / arm / f"{args.phase}_job_completed.json").exists():
                atomic_json(manifest_path, manifest)
                print(arm, args.phase, "existing live/completed job; no duplicate submission")
                continue
            if existing and not args.resume:
                raise ValueError("Terminal unsuccessful attempt exists; inspect it and use --resume")
            # A resumed job must retain the original worker identity, even after report commits.
            previous = [r for r in manifest["jobs"] if r["arm"] == arm and r["phase"] == args.phase and "commit" in r]
            assert not existing or previous, "Cannot resolve the original release for this recovered job"
            commit, release = snapshot(repo, root, previous[-1]["commit"] if existing else None)
            command = ["eai", "job", "new", "--account", ACCOUNT, "--restartable", "--name", name,
                "--image", cluster["image"], "--cpu", "64", "--mem", "512", "--gpu", "8", "--gpu-model-filter", "H100",
                "--max-run-time", "0", "--data", cluster["data"], "--workdir", str(release)]
            env = {"PYTHONPATH": f"{release}/src:{release}", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                "CC": cluster["cc"], "CXX": cluster["cxx"], "TOKENIZERS_PARALLELISM": "false", "GUIDANCE_RELEASE": commit, "GUIDANCE_ACCOUNT": ACCOUNT,
                "TORCHINDUCTOR_CACHE_DIR": str(root / arm / "inductor"), "TRITON_CACHE_DIR": str(root / arm / "triton")}
            for key, value in env.items():
                command.extend(["--env", f"{key}={value}"])
            entry = ["--", "bash", "guidance_sweep/launch.sh", arm, args.phase, str(root)]
            assert command[command.index("--account") + 1] == ACCOUNT
            spec = subprocess.check_output(command + ["--dry-run"] + entry, text=True)
            (root / f"{arm}-{args.phase}-job.yml").write_text(spec)
            if args.dry_run:
                print({"arm": arm, "phase": args.phase, "account": ACCOUNT, "gpus": 8, "cpu": 64, "mem": 512, "spec": str(root / f"{arm}-{args.phase}-job.yml")})
                continue
            import json
            raw = subprocess.check_output(command + ["--format", "json"] + entry, text=True)
            job = json.loads(raw)
            assert job["account"] == ACCOUNT and job["resources"]["gpu"] == 8
            manifest["jobs"].append({"arm": arm, "phase": args.phase, "id": job["id"], "release": str(release),
                                    "commit": commit, "created": job["created"], "spec_sha256": file_hash(root / f"{arm}-{args.phase}-job.yml")})
            atomic_json(manifest_path, manifest)
            print({"arm": arm, "phase": args.phase, "id": job["id"], "account": job["account"], "state": job["state"]})


if __name__ == "__main__":
    main()
