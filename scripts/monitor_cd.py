#!/usr/bin/env python3
"""Record authoritative C/D scheduler state and saved pipeline progress."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time


ACCOUNT = "snow.research.er"
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"}
ROOT = Path(__file__).resolve().parents[1]


def read_optional(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def observe_arm(arm, submission):
    job_id = submission["jobs"][-1]["id"]
    result = {"arm": arm, "job_id": job_id}
    try:
        response = subprocess.run(
            ["eai", "job", "get", job_id, "--account", ACCOUNT, "--format", "json"],
            check=True, capture_output=True, text=True, timeout=25,
        )
        job = json.loads(response.stdout)
        if job["id"] != job_id or job["account"] != ACCOUNT:
            raise ValueError("Scheduler identity/account does not match the requested job")
        result.update(
            scheduler_state=job["state"], scheduler_info=job.get("stateInfo"),
            account=job["account"], resources=job["resources"],
            scheduler_observed_at=time.time(),
            scheduler_runs=[{key: run.get(key) for key in (
                "id", "state", "stateInfo", "nodeName", "startedOn", "endedOn", "exitCode"
            )} for run in job.get("runs", [])],
        )
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        # A failed observation is not a terminal job state. The next cycle
        # queries the same job handle; this script never submits or cancels.
        result["scheduler_observation_error"] = {
            "type": type(exc).__name__, "message": str(exc)[:500]
        }
    try:
        output = Path(submission["output"])
        progress = read_optional(output / "adaptation/status.json")
        checkpoint = read_optional(output / "adaptation/latest.json")
        elf = read_optional(output / "elf/status.json")
        elf_done = read_optional(output / "elf/completed.json")
        result.update(
            stages={path.stem.removeprefix("stage-"): json.loads(path.read_text())["state"]
                    for path in sorted(output.glob("stage-*.json"))},
            adaptation_observed_updates=progress.get("updates"),
            adaptation_observed_positions=progress.get("nominal_positions"),
            adaptation_progress_age_seconds=(max(0, time.time() - progress["time"])
                                             if "time" in progress else None),
            adaptation_saved_updates=checkpoint.get("counters", {}).get("updates"),
            adaptation_saved_positions=checkpoint.get("counters", {}).get("nominal_positions"),
            adaptation_final=checkpoint.get("final", False),
            elf_observed_updates=elf.get("step"), elf_observed_epoch=elf.get("epoch"),
            elf_completed_updates=elf_done.get("updates"),
            elf_completed_epochs=elf_done.get("epochs"),
            pipeline_completed_marker=read_optional(output / "completed.json").get("state") == "completed",
        )
    except (OSError, ValueError, KeyError) as exc:
        result["artifact_observation_error"] = {
            "type": type(exc).__name__, "message": str(exc)[:500]
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--interval", type=float, default=45)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error("--interval must be between 1 and 60 seconds")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    with (args.artifacts / "monitor.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(2, "A C/D status recorder already owns this artifacts directory.\n")
        with ThreadPoolExecutor(max_workers=2) as executor:
            while True:
                snapshot = {
                    "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "monitor_pid": os.getpid(), "monitor_host": socket.gethostname(),
                    "scope": "Scheduler and progress observations; final GOAL.md acceptance requires a separate audit.",
                }
                try:
                    # Reload so a verified replacement recorded by the
                    # submitter is followed without inventing a new job.
                    manifest = json.loads((args.artifacts / "submission.json").read_text())
                    futures = {arm: executor.submit(observe_arm, arm, manifest["arms"][arm])
                               for arm in ("C", "D")}
                    snapshot["arms"] = {arm: future.result() for arm, future in futures.items()}
                    records = list(snapshot["arms"].values())
                    errors = any("scheduler_observation_error" in row or "artifact_observation_error" in row
                                 for row in records)
                    states = [row.get("scheduler_state") for row in records]
                    all_terminal = all(state in TERMINAL for state in states)
                    failed = any(state in TERMINAL - {"SUCCEEDED"} for state in states)
                    snapshot["state"] = ("observation_error" if errors else "attention_required" if failed else
                                         "scheduler_finished" if all_terminal else "observing")
                    snapshot["all_jobs_terminal"] = all_terminal
                    snapshot["final_acceptance_required"] = True
                except (OSError, ValueError, KeyError, IndexError) as exc:
                    snapshot.update(state="observation_error", all_jobs_terminal=False,
                                    observation_error={"type": type(exc).__name__, "message": str(exc)[:500]})
                encoded = json.dumps(snapshot, sort_keys=True, allow_nan=False)
                temporary = args.artifacts / "monitor.json.tmp"
                with temporary.open("w") as stream:
                    stream.write(encoded + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, args.artifacts / "monitor.json")
                with (args.artifacts / "monitor-history.jsonl").open("a") as stream:
                    stream.write(encoded + "\n")
                summary = {"time": snapshot["observed_at_utc"], "state": snapshot["state"],
                           "arms": {arm: {key: row.get(key) for key in (
                               "job_id", "scheduler_state", "adaptation_observed_updates",
                               "adaptation_saved_updates", "adaptation_observed_positions",
                               "elf_observed_updates", "elf_completed_epochs", "stages",
                               "scheduler_observation_error", "artifact_observation_error"
                           )} for arm, row in snapshot.get("arms", {}).items()}}
                print(json.dumps(summary, sort_keys=True), flush=True)
                if args.once or snapshot["all_jobs_terminal"]:
                    return 0 if snapshot["state"] in {"observing", "scheduler_finished"} else 1
                time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
