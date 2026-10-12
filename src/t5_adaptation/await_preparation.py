"""Monitor one authoritative prep job, then submit C/D once dependencies finish."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .common import ROOT, atomic_json, read_json
from .submit import eai_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation-job", required=True)
    parser.add_argument("--pilot", required=True)
    args = parser.parse_args()
    marker = ROOT / "artifacts/preparation-controller.json"
    while True:
        # Transient observation errors do not imply a dead worker or trigger
        # a replacement. Keep polling the same scheduler handle.
        try:
            job = eai_json(["get", args.preparation_job, "--account", "snow.research.er"])[0]
        except subprocess.CalledProcessError as exc:
            atomic_json(marker, {"state": "observation_error", "job_id": args.preparation_job,
                                  "exit_code": exc.returncode, "time": time.time()})
            time.sleep(30)
            continue
        record = {"state": job["state"], "job_id": job["id"], "time": time.time(), "info": job["stateInfo"]}
        progress = ROOT / "artifacts/shared/c4/train/preparation.json"
        if progress.exists():
            state = read_json(progress)
            record["committed_c4_positions"] = state["tokens"]
            record["required_c4_positions"] = state["identity"]["examples"] * state["identity"]["source_length"]
        atomic_json(marker, record)
        print(json.dumps(record), flush=True)
        if job["state"] == "SUCCEEDED":
            break
        if job["state"] in {"FAILED", "CANCELLED", "INTERRUPTED"}:
            raise SystemExit("Preparation job is terminal; inspect preserved logs before any replacement")
        time.sleep(30)
    command = [sys.executable, "-m", "t5_adaptation.submit", "--pilot", args.pilot]
    subprocess.run(command, cwd=ROOT, check=True)
    atomic_json(marker, {"state": "submitted", "preparation_job": args.preparation_job, "time": time.time()})


if __name__ == "__main__":
    main()
